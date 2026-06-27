use std::ops::{Deref, DerefMut};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static LOCAL_POOL: RefCell<HashMap<usize, Vec<Vec<f64>>>> = RefCell::new(HashMap::new());
}

#[derive(Debug)]
pub enum BufferStorage {
    Host(Vec<f64>),
    Uvm {
        ptr: *mut f64,
        size: usize,
        device_ptr: u64,
    },
}

/// A wrapper around Vec<f64> or CUDA Managed pointer that recycles host storage
/// or frees managed memory on drop.
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
    pub fn new(size: usize) -> Self {
        if !crate::device::is_force_cpu() && crate::device::get_cuda_context().is_some() {
            Self::new_uvm(size)
        } else {
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

    /// Wrap an existing vector into a Buffer.
    pub fn from_vec(vec: Vec<f64>) -> Self {
        if !crate::device::is_force_cpu() && crate::device::get_cuda_context().is_some() {
            let mut buf = Self::new_uvm(vec.len());
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

    /// Prefetch Unified Memory buffer to the GPU device.
    pub fn prefetch_to_device(&self) {
        if let BufferStorage::Uvm { device_ptr, size, .. } = &self.storage {
            if let Some(ctx) = crate::device::get_cuda_context() {
                unsafe {
                    (ctx.cuda.cuMemPrefetchAsync)(*device_ptr, size * 8, ctx.device, std::ptr::null_mut());
                }
            }
        }
    }

    /// Prefetch Unified Memory buffer back to CPU.
    pub fn prefetch_to_host(&self) {
        if let BufferStorage::Uvm { device_ptr, size, .. } = &self.storage {
            if let Some(ctx) = crate::device::get_cuda_context() {
                unsafe {
                    (ctx.cuda.cuMemPrefetchAsync)(*device_ptr, size * 8, -1, std::ptr::null_mut());
                }
            }
        }
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

