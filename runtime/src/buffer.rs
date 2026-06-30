use std::ops::{Deref, DerefMut};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

thread_local! {
    static LOCAL_POOL: RefCell<HashMap<usize, Vec<Vec<f64>>>> = RefCell::new(HashMap::new());
}

// ── VRAM Caching Allocator ──
// Pools freed VRAM device pointers by element-count to avoid repeated cuMemAlloc/cuMemFree syscalls.
fn vram_pool() -> &'static Mutex<HashMap<usize, Vec<u64>>> {
    static VRAM_POOL: OnceLock<Mutex<HashMap<usize, Vec<u64>>>> = OnceLock::new();
    VRAM_POOL.get_or_init(|| Mutex::new(HashMap::new()))
}

const VRAM_POOL_MAX_PER_BUCKET: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DirtyFlag {
    /// VRAM holds the current data; host_mirror is stale or absent.
    DeviceCurrent,
    /// host_mirror holds the current data; VRAM is stale.
    HostCurrent,
    /// Both copies are in sync.
    Clean,
}

#[derive(Debug)]
pub enum BufferStorage {
    Host(Vec<f64>),
    Uvm {
        ptr: *mut f64,
        size: usize,
        device_ptr: u64,
    },
    Vram {
        device_ptr: u64,
        size: usize,
        host_mirror: RefCell<Option<Vec<f64>>>,
        dirty: RefCell<DirtyFlag>,
    },
}

/// A wrapper around Vec<f64>, CUDA Managed pointer, or dedicated VRAM pointer
/// that recycles host storage or frees GPU memory on drop.
#[derive(Debug)]
pub struct Buffer {
    pub storage: BufferStorage,
}

impl Default for Buffer {
    fn default() -> Self {
        Self {
            storage: BufferStorage::Host(Vec::new()),
        }
    }
}

impl Buffer {
    /// Retrieve a buffer of the given size from the pool, or allocate a new one.
    /// Prefers VRAM > UVM > Host based on availability.
    pub fn new(size: usize) -> Self {
        if !crate::device::is_force_cpu() && crate::device::get_cuda_context().is_some() {
            // Try dedicated VRAM first
            if let Some(buf) = Self::try_new_vram(size) {
                return buf;
            }
            // Fall back to UVM
            Self::new_uvm(size)
        } else {
            Self::new_host(size)
        }
    }

    /// Allocate a CPU host buffer from the thread-local pool.
    pub fn new_host(size: usize) -> Self {
        let mut inner = None;
        LOCAL_POOL.with(|pool| {
            if let Ok(mut map) = pool.try_borrow_mut() {
                let mut found_cap = None;
                for &cap in map.keys() {
                    if cap >= size && cap <= size * 4 {
                        found_cap = Some(cap);
                        break;
                    }
                }
                if found_cap.is_none() {
                    for &cap in map.keys() {
                        if cap >= size {
                            found_cap = Some(cap);
                            break;
                        }
                    }
                }
                if let Some(cap) = found_cap {
                    if let Some(mut buf) = map.get_mut(&cap).and_then(|v| v.pop()) {
                        if map.get(&cap).map(|v| v.is_empty()).unwrap_or(false) {
                            map.remove(&cap);
                        }
                        buf.clear();
                        buf.resize(size, 0.0);
                        inner = Some(buf);
                    }
                }
            }
        });

        if let Some(buf) = inner {
            Self { storage: BufferStorage::Host(buf) }
        } else {
            Self { storage: BufferStorage::Host(vec![0.0; size]) }
        }
    }

    /// Explicitly allocate Unified Memory.
    pub fn new_uvm(size: usize) -> Self {
        // cuMemAllocManaged with 0 bytes is invalid (CUDA_ERROR_INVALID_VALUE).
        // Skip CUDA for empty buffers.
        if size == 0 {
            return Self { storage: BufferStorage::Host(Vec::new()) };
        }
        if let Some(ctx) = crate::device::get_cuda_context() {
            let mut device_ptr: u64 = 0;
            let byte_size = size * std::mem::size_of::<f64>();
            let res = unsafe {
                (ctx.cuda.cuMemAllocManaged)(&mut device_ptr, byte_size, 0x01)
            };
            if res == 0 {
                let ptr = device_ptr as *mut f64;
                unsafe {
                    std::ptr::write_bytes(ptr, 0, size);
                }
                return Self {
                    storage: BufferStorage::Uvm {
                        ptr,
                        size,
                        device_ptr,
                    }
                };
            } else {
                eprintln!("cuMemAllocManaged failed (code {}). Size: {} elems, {} bytes. Falling back to CPU.", res, size, byte_size);
                crate::device::set_force_cpu(true);
            }
        }
        
        Self { storage: BufferStorage::Host(vec![0.0; size]) }
    }

    /// Allocate a dedicated VRAM buffer via cuMemAlloc_v2.
    /// Returns None if allocation fails (VRAM budget exceeded, driver error, etc.)
    pub fn try_new_vram(size: usize) -> Option<Self> {
        if size == 0 {
            return Some(Self { storage: BufferStorage::Host(Vec::new()) });
        }
        let byte_size = size * std::mem::size_of::<f64>();

        // Check VRAM budget
        if crate::device::vram_available() < byte_size {
            return None;
        }

        // Try the VRAM caching pool first
        if let Ok(mut pool) = vram_pool().lock() {
            if let Some(ptrs) = pool.get_mut(&size) {
                if let Some(device_ptr) = ptrs.pop() {
                    if ptrs.is_empty() {
                        pool.remove(&size);
                    }
                    // Zero the recycled buffer on device
                    if let Some(ctx) = crate::device::get_cuda_context() {
                        unsafe {
                            (ctx.cuda.cuMemsetD8)(device_ptr, 0, byte_size);
                        }
                    }
                    return Some(Self {
                        storage: BufferStorage::Vram {
                            device_ptr,
                            size,
                            host_mirror: RefCell::new(None),
                            dirty: RefCell::new(DirtyFlag::DeviceCurrent),
                        },
                    });
                }
            }
        }

        // Allocate fresh VRAM
        let ctx = crate::device::get_cuda_context()?;
        let mut device_ptr: u64 = 0;
        let res = unsafe {
            (ctx.cuda.cuMemAlloc_v2)(&mut device_ptr, byte_size)
        };
        if res == 0 {
            crate::device::vram_alloc_track(byte_size);
            // Zero-initialize on device
            unsafe {
                (ctx.cuda.cuMemsetD8)(device_ptr, 0, byte_size);
            }
            Some(Self {
                storage: BufferStorage::Vram {
                    device_ptr,
                    size,
                    host_mirror: RefCell::new(None),
                    dirty: RefCell::new(DirtyFlag::DeviceCurrent),
                },
            })
        } else {
            None
        }
    }

    /// Allocate a dedicated VRAM buffer, falling back to UVM then Host.
    pub fn new_vram(size: usize) -> Self {
        if !crate::device::is_force_cpu() {
            if let Some(buf) = Self::try_new_vram(size) {
                return buf;
            }
        }
        // Fall back
        Self::new(size)
    }

    /// Wrap an existing vector into a Buffer.
    pub fn from_vec(vec: Vec<f64>) -> Self {
        if !crate::device::is_force_cpu() && crate::device::get_cuda_context().is_some() {
            let size = vec.len();
            // Try VRAM first: allocate + upload
            if let Some(buf) = Self::try_new_vram(size) {
                buf.upload_host_data(&vec);
                // Keep the host mirror for potential CPU reads
                if let BufferStorage::Vram { ref host_mirror, ref dirty, .. } = buf.storage {
                    *host_mirror.borrow_mut() = Some(vec);
                    *dirty.borrow_mut() = DirtyFlag::Clean;
                }
                return buf;
            }
            // Fall back to UVM
            let mut buf = Self::new_uvm(size);
            buf.copy_from_slice(&vec);
            buf
        } else {
            Self { storage: BufferStorage::Host(vec) }
        }
    }

    /// Retrieve a buffer from the pool and copy elements from a slice.
    pub fn from_slice(src: &[f64]) -> Self {
        let mut buf = Self::new(src.len());
        buf.copy_from_slice(src);
        buf
    }

    // ── VRAM Data Transfer Methods ──

    /// Upload host data to the VRAM device pointer.
    fn upload_host_data(&self, data: &[f64]) {
        if let BufferStorage::Vram { device_ptr, size, .. } = &self.storage {
            debug_assert!(data.len() <= *size);
            if let Some(ctx) = crate::device::get_cuda_context() {
                let byte_size = data.len() * std::mem::size_of::<f64>();
                unsafe {
                    (ctx.cuda.cuMemcpyHtoD_v2)(
                        *device_ptr,
                        data.as_ptr() as *const std::ffi::c_void,
                        byte_size,
                    );
                }
            }
        }
    }

    /// Ensure the host_mirror is populated and current. Downloads from VRAM if needed.
    pub fn ensure_host(&self) {
        if let BufferStorage::Vram { device_ptr, size, host_mirror, dirty } = &self.storage {
            let current_dirty = *dirty.borrow();
            match current_dirty {
                DirtyFlag::DeviceCurrent => {
                    // Download from VRAM to host
                    let mut mirror = host_mirror.borrow_mut();
                    let buf = mirror.get_or_insert_with(|| vec![0.0; *size]);
                    if buf.len() != *size {
                        buf.resize(*size, 0.0);
                    }
                    if let Some(ctx) = crate::device::get_cuda_context() {
                        let byte_size = *size * std::mem::size_of::<f64>();
                        unsafe {
                            (ctx.cuda.cuMemcpyDtoH_v2)(
                                buf.as_mut_ptr() as *mut std::ffi::c_void,
                                *device_ptr,
                                byte_size,
                            );
                        }
                    }
                    *dirty.borrow_mut() = DirtyFlag::Clean;
                }
                DirtyFlag::HostCurrent | DirtyFlag::Clean => {
                    // Host is already current, or both are in sync
                    let mut mirror = host_mirror.borrow_mut();
                    if mirror.is_none() {
                        *mirror = Some(vec![0.0; *size]);
                    }
                }
            }
        }
    }

    /// Ensure the VRAM device_ptr is current. Uploads from host_mirror if needed.
    pub fn ensure_device(&self) {
        if let BufferStorage::Vram { device_ptr, size, host_mirror, dirty } = &self.storage {
            let current_dirty = *dirty.borrow();
            if current_dirty == DirtyFlag::HostCurrent {
                // Upload from host to VRAM
                let mirror = host_mirror.borrow();
                if let Some(data) = mirror.as_ref() {
                    if let Some(ctx) = crate::device::get_cuda_context() {
                        let byte_size = data.len().min(*size) * std::mem::size_of::<f64>();
                        unsafe {
                            (ctx.cuda.cuMemcpyHtoD_v2)(
                                *device_ptr,
                                data.as_ptr() as *const std::ffi::c_void,
                                byte_size,
                            );
                        }
                    }
                }
                *dirty.borrow_mut() = DirtyFlag::Clean;
            }
        } else if let BufferStorage::Uvm { .. } = &self.storage {
            // Legacy UVM path: prefetch to device
            self.prefetch_to_device();
        }
    }

    /// Mark the VRAM copy as the current version (call after a kernel writes to it).
    pub fn mark_device_dirty(&self) {
        if let BufferStorage::Vram { dirty, .. } = &self.storage {
            *dirty.borrow_mut() = DirtyFlag::DeviceCurrent;
            // Invalidate host mirror — it's now stale
            // We keep the allocation for reuse but mark it as needing re-download
        }
    }

    /// Get the device pointer for kernel launch arguments.
    /// For Vram: returns the dedicated VRAM pointer (uploads if needed).
    /// For Uvm: returns the UVM pointer.
    /// For Host: returns 0 (triggers CPU fallback in the VM).
    pub fn device_ptr(&self) -> u64 {
        match &self.storage {
            BufferStorage::Vram { device_ptr, .. } => {
                self.ensure_device();
                *device_ptr
            }
            BufferStorage::Uvm { device_ptr, .. } => *device_ptr,
            BufferStorage::Host(_) => 0,
        }
    }

    /// Prefetch Unified Memory buffer to the GPU device (legacy UVM path).
    pub fn prefetch_to_device(&self) {
        if let BufferStorage::Uvm { device_ptr, size, .. } = &self.storage {
            if let Some(ctx) = crate::device::get_cuda_context() {
                unsafe {
                    (ctx.cuda.cuMemPrefetchAsync)(*device_ptr, size * 8, ctx.device, std::ptr::null_mut());
                }
            }
        }
    }

    /// Prefetch Unified Memory buffer back to CPU (legacy UVM path).
    pub fn prefetch_to_host(&self) {
        if let BufferStorage::Uvm { device_ptr, size, .. } = &self.storage {
            if let Some(ctx) = crate::device::get_cuda_context() {
                unsafe {
                    (ctx.cuda.cuMemPrefetchAsync)(*device_ptr, size * 8, -1, std::ptr::null_mut());
                }
            }
        }
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        match &self.storage {
            BufferStorage::Host(v) => v.len(),
            BufferStorage::Uvm { size, .. } => *size,
            BufferStorage::Vram { size, .. } => *size,
        }
    }

    /// Returns true if the buffer has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if this buffer is backed by dedicated VRAM.
    pub fn is_vram(&self) -> bool {
        matches!(&self.storage, BufferStorage::Vram { .. })
    }
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl From<Vec<f64>> for Buffer {
    fn from(vec: Vec<f64>) -> Self {
        Self::from_vec(vec)
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        match &self.storage {
            BufferStorage::Host(vec) => {
                Self::from_slice(vec)
            }
            BufferStorage::Uvm { size, .. } => {
                let mut new_buf = Self::new_uvm(*size);
                new_buf.copy_from_slice(self);
                new_buf
            }
            BufferStorage::Vram { size, .. } => {
                // Clone by allocating new VRAM and downloading+re-uploading
                self.ensure_host();
                let data: Vec<f64> = self.iter().copied().collect();
                let mut new_buf = if let Some(vram) = Self::try_new_vram(*size) {
                    vram
                } else {
                    Self::new(*size)
                };
                new_buf.copy_from_slice(&data);
                new_buf
            }
        }
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self[..] == other[..]
    }
}

impl PartialEq<Vec<f64>> for Buffer {
    fn eq(&self, other: &Vec<f64>) -> bool {
        self[..] == other[..]
    }
}

impl PartialEq<[f64]> for Buffer {
    fn eq(&self, other: &[f64]) -> bool {
        self[..] == other[..]
    }
}

impl Deref for Buffer {
    type Target = [f64];
    fn deref(&self) -> &Self::Target {
        match &self.storage {
            BufferStorage::Host(vec) => vec,
            BufferStorage::Uvm { ptr, size, .. } => {
                unsafe { std::slice::from_raw_parts(*ptr, *size) }
            }
            BufferStorage::Vram { size, host_mirror, .. } => {
                // Transparently download from VRAM on first CPU read
                self.ensure_host();
                let mirror = host_mirror.borrow();
                let slice = mirror.as_ref().unwrap();
                // SAFETY: The RefCell borrow is active for the lifetime of the
                // returned reference. We use a raw pointer to extend the lifetime.
                // This is safe because:
                // 1. The host_mirror Vec won't be reallocated while we hold &self
                // 2. The data pointer remains valid as long as the Buffer exists
                unsafe { std::slice::from_raw_parts(slice.as_ptr(), *size) }
            }
        }
    }
}

impl DerefMut for Buffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.storage {
            BufferStorage::Host(vec) => vec,
            BufferStorage::Uvm { ptr, size, .. } => {
                unsafe { std::slice::from_raw_parts_mut(*ptr, *size) }
            }
            BufferStorage::Vram { device_ptr, size, host_mirror, dirty } => {
                // Inline ensure_host: download from VRAM if device copy is current
                {
                    let current_dirty = *dirty.borrow();
                    if current_dirty == DirtyFlag::DeviceCurrent {
                        let mut mirror = host_mirror.borrow_mut();
                        let buf = mirror.get_or_insert_with(|| vec![0.0; *size]);
                        if buf.len() != *size {
                            buf.resize(*size, 0.0);
                        }
                        if let Some(ctx) = crate::device::get_cuda_context() {
                            let byte_size = *size * std::mem::size_of::<f64>();
                            unsafe {
                                (ctx.cuda.cuMemcpyDtoH_v2)(
                                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                                    *device_ptr,
                                    byte_size,
                                );
                            }
                        }
                    } else if host_mirror.borrow().is_none() {
                        *host_mirror.borrow_mut() = Some(vec![0.0; *size]);
                    }
                }
                *dirty.borrow_mut() = DirtyFlag::HostCurrent;
                let mut mirror = host_mirror.borrow_mut();
                let slice = mirror.as_mut().unwrap();
                unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr(), *size) }
            }
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        match &mut self.storage {
            BufferStorage::Host(ref mut inner) => {
                let buf = std::mem::take(inner);
                let cap = buf.capacity();
                if cap > 0 && cap <= 10_000_000 {
                    LOCAL_POOL.with(|pool| {
                        if let Ok(mut map) = pool.try_borrow_mut() {
                            let entry = map.entry(cap).or_default();
                            if entry.len() < 32 {
                                entry.push(buf);
                            }
                        }
                    });
                }
            }
            BufferStorage::Uvm { device_ptr, .. } => {
                if let Some(ctx) = crate::device::get_cuda_context() {
                    unsafe {
                        (ctx.cuda.cuMemFree_v2)(*device_ptr);
                    }
                }
            }
            BufferStorage::Vram { device_ptr, size, .. } => {
                let byte_size = *size * std::mem::size_of::<f64>();
                // Return to VRAM caching pool instead of freeing
                let pooled = if let Ok(mut pool) = vram_pool().lock() {
                    let entry = pool.entry(*size).or_default();
                    if entry.len() < VRAM_POOL_MAX_PER_BUCKET {
                        entry.push(*device_ptr);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                if !pooled {
                    // Pool is full, actually free the VRAM
                    if let Some(ctx) = crate::device::get_cuda_context() {
                        unsafe {
                            (ctx.cuda.cuMemFree_v2)(*device_ptr);
                        }
                    }
                    crate::device::vram_free_track(byte_size);
                }
            }
        }
    }
}

/// Retrieve statistics about the thread-local allocation pool.
/// Returns (number_of_capacity_buckets, total_cached_vectors).
pub fn get_pool_stats() -> (usize, usize) {
    LOCAL_POOL.with(|pool| {
        if let Ok(map) = pool.try_borrow() {
            let buckets = map.len();
            let total: usize = map.values().map(|v| v.len()).sum();
            (buckets, total)
        } else {
            (0, 0)
        }
    })
}
