#![allow(non_snake_case)]

/// NEURON Device abstraction — CPU and GPU device management.
///
/// Dynamically loads the CUDA driver and NVRTC compiler libraries using libloading.
/// Provides unified memory allocation and compilation hooks.

use std::sync::OnceLock;


use libloading::Library;

thread_local! {
    pub static SIMULATE_CUDA: std::cell::Cell<Option<bool>> = std::cell::Cell::new(None);
}

pub fn set_simulate_cuda(val: bool) {
    SIMULATE_CUDA.with(|s| s.set(Some(val)));
}

pub fn is_simulate_cuda() -> bool {
    SIMULATE_CUDA.with(|s| s.get()).unwrap_or_else(|| std::env::var("NEURON_SIMULATE_CUDA").is_ok())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CPU,
    CUDA(usize),
}

impl Device {
    /// Detect the best available device.
    pub fn auto() -> Self {
        if Self::cuda_available() {
            Device::CUDA(0)
        } else {
            Device::CPU
        }
    }

    /// Check if CUDA (actual or simulated) is available.
    pub fn cuda_available() -> bool {
        if is_simulate_cuda() {
            true
        } else {
            get_cuda_context().is_some()
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Device::CPU => "cpu",
            Device::CUDA(_id) => "cuda",
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CPU => write!(f, "cpu"),
            Device::CUDA(id) => write!(f, "cuda:{}", id),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Dynamic CUDA & NVRTC Bindings via libloading
// ══════════════════════════════════════════════════════════════════════════════

pub struct CudaApi {
    _lib: Library,
    pub cuInit: unsafe extern "C" fn(flags: u32) -> u32,
    pub cuDeviceGet: unsafe extern "C" fn(device: *mut i32, ordinal: i32) -> u32,
    pub cuCtxCreate_v2: unsafe extern "C" fn(pctx: *mut *mut std::ffi::c_void, flags: u32, dev: i32) -> u32,
    pub cuCtxDestroy_v2: unsafe extern "C" fn(ctx: *mut std::ffi::c_void) -> u32,
    pub cuModuleLoadData: unsafe extern "C" fn(module: *mut *mut std::ffi::c_void, image: *const std::ffi::c_void) -> u32,
    pub cuModuleUnload: unsafe extern "C" fn(module: *mut std::ffi::c_void) -> u32,
    pub cuModuleGetFunction: unsafe extern "C" fn(hfunc: *mut *mut std::ffi::c_void, hmod: *mut std::ffi::c_void, name: *const std::os::raw::c_char) -> u32,
    pub cuMemAllocManaged: unsafe extern "C" fn(dptr: *mut u64, bytesize: usize, flags: u32) -> u32,
    pub cuMemFree_v2: unsafe extern "C" fn(dptr: u64) -> u32,
    pub cuMemPrefetchAsync: unsafe extern "C" fn(dev_ptr: u64, count: usize, dstDevice: i32, hStream: *mut std::ffi::c_void) -> u32,
    pub cuLaunchKernel: unsafe extern "C" fn(
        f: *mut std::ffi::c_void,
        gridDimX: u32, gridDimY: u32, gridDimZ: u32,
        blockDimX: u32, blockDimY: u32, blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: *mut std::ffi::c_void,
        kernelParams: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void
    ) -> u32,
    pub cuStreamCreate: unsafe extern "C" fn(phStream: *mut *mut std::ffi::c_void, flags: u32) -> u32,
    pub cuStreamDestroy_v2: unsafe extern "C" fn(hStream: *mut std::ffi::c_void) -> u32,
    pub cuStreamSynchronize: unsafe extern "C" fn(hStream: *mut std::ffi::c_void) -> u32,
    pub cuCtxSynchronize: unsafe extern "C" fn() -> u32,
    pub cuGetErrorString: unsafe extern "C" fn(error: u32, pStr: *mut *const std::os::raw::c_char) -> u32,
}

pub struct NvrtcApi {
    _lib: Library,
    pub nvrtcCreateProgram: unsafe extern "C" fn(
        prog: *mut *mut std::ffi::c_void,
        src: *const std::os::raw::c_char,
        name: *const std::os::raw::c_char,
        numHeaders: i32,
        headers: *const *const std::os::raw::c_char,
        includeNames: *const *const std::os::raw::c_char
    ) -> u32,
    pub nvrtcCompileProgram: unsafe extern "C" fn(
        prog: *mut std::ffi::c_void,
        numOptions: i32,
        options: *const *const std::os::raw::c_char
    ) -> u32,
    pub nvrtcGetPTXSize: unsafe extern "C" fn(prog: *mut std::ffi::c_void, ptxSizeRet: *mut usize) -> u32,
    pub nvrtcGetPTX: unsafe extern "C" fn(prog: *mut std::ffi::c_void, ptx: *mut std::os::raw::c_char) -> u32,
    pub nvrtcDestroyProgram: unsafe extern "C" fn(prog: *mut *mut std::ffi::c_void) -> u32,
    pub nvrtcGetProgramLogSize: unsafe extern "C" fn(prog: *mut std::ffi::c_void, logSizeRet: *mut usize) -> u32,
    pub nvrtcGetProgramLog: unsafe extern "C" fn(prog: *mut std::ffi::c_void, log: *mut std::os::raw::c_char) -> u32,
    pub nvrtcGetErrorString: unsafe extern "C" fn(result: u32) -> *const std::os::raw::c_char,
}

pub struct CudaContext {
    pub cuda: CudaApi,
    pub nvrtc: NvrtcApi,
    pub ctx: *mut std::ffi::c_void,
    pub device: i32,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaApi {
    pub fn load() -> Result<Self, String> {
        let filenames = if cfg!(target_os = "windows") {
            vec!["nvcuda.dll"]
        } else if cfg!(target_os = "macos") {
            vec!["libcuda.dylib"]
        } else {
            vec!["libcuda.so.1", "libcuda.so"]
        };
        
        let mut loaded_lib = None;
        for name in filenames {
            if let Ok(lib) = unsafe { Library::new(name) } {
                loaded_lib = Some(lib);
                break;
            }
        }
        
        let lib = loaded_lib.ok_or_else(|| "Could not load CUDA driver library".to_string())?;
        
        unsafe {
            let cuInit = *lib.get(b"cuInit").map_err(|e| e.to_string())?;
            let cuDeviceGet = *lib.get(b"cuDeviceGet").map_err(|e| e.to_string())?;
            let cuCtxCreate_v2 = *lib.get(b"cuCtxCreate_v2").map_err(|e| e.to_string())?;
            let cuCtxDestroy_v2 = *lib.get(b"cuCtxDestroy_v2").map_err(|e| e.to_string())?;
            let cuModuleLoadData = *lib.get(b"cuModuleLoadData").map_err(|e| e.to_string())?;
            let cuModuleUnload = *lib.get(b"cuModuleUnload").map_err(|e| e.to_string())?;
            let cuModuleGetFunction = *lib.get(b"cuModuleGetFunction").map_err(|e| e.to_string())?;
            let cuMemAllocManaged = *lib.get(b"cuMemAllocManaged").map_err(|e| e.to_string())?;
            let cuMemFree_v2 = *lib.get(b"cuMemFree_v2").map_err(|e| e.to_string())?;
            let cuMemPrefetchAsync = *lib.get(b"cuMemPrefetchAsync").map_err(|e| e.to_string())?;
            let cuLaunchKernel = *lib.get(b"cuLaunchKernel").map_err(|e| e.to_string())?;
            let cuStreamCreate = *lib.get(b"cuStreamCreate").map_err(|e| e.to_string())?;
            let cuStreamDestroy_v2 = *lib.get(b"cuStreamDestroy_v2").map_err(|e| e.to_string())?;
            let cuStreamSynchronize = *lib.get(b"cuStreamSynchronize").map_err(|e| e.to_string())?;
            let cuCtxSynchronize = *lib.get(b"cuCtxSynchronize").map_err(|e| e.to_string())?;
            let cuGetErrorString = *lib.get(b"cuGetErrorString").map_err(|e| e.to_string())?;
            
            Ok(CudaApi {
                _lib: lib,
                cuInit,
                cuDeviceGet,
                cuCtxCreate_v2,
                cuCtxDestroy_v2,
                cuModuleLoadData,
                cuModuleUnload,
                cuModuleGetFunction,
                cuMemAllocManaged,
                cuMemFree_v2,
                cuMemPrefetchAsync,
                cuLaunchKernel,
                cuStreamCreate,
                cuStreamDestroy_v2,
                cuStreamSynchronize,
                cuCtxSynchronize,
                cuGetErrorString,
            })
        }
    }
}

impl NvrtcApi {
    pub fn load() -> Result<Self, String> {
        let filenames = if cfg!(target_os = "windows") {
            vec![
                "nvrtc64_120_0.dll",
                "nvrtc64_112_0.dll",
                "nvrtc64_111_0.dll",
                "nvrtc64_101_0.dll",
                "nvrtc.dll"
            ]
        } else if cfg!(target_os = "macos") {
            vec!["libnvrtc.dylib"]
        } else {
            vec![
                "libnvrtc.so.12",
                "libnvrtc.so.11.2",
                "libnvrtc.so"
            ]
        };
        
        let mut loaded_lib = None;
        for name in filenames {
            if let Ok(lib) = unsafe { Library::new(name) } {
                loaded_lib = Some(lib);
                break;
            }
        }
        
        let lib = loaded_lib.ok_or_else(|| "Could not load NVRTC library".to_string())?;
        
        unsafe {
            let nvrtcCreateProgram = *lib.get(b"nvrtcCreateProgram").map_err(|e| e.to_string())?;
            let nvrtcCompileProgram = *lib.get(b"nvrtcCompileProgram").map_err(|e| e.to_string())?;
            let nvrtcGetPTXSize = *lib.get(b"nvrtcGetPTXSize").map_err(|e| e.to_string())?;
            let nvrtcGetPTX = *lib.get(b"nvrtcGetPTX").map_err(|e| e.to_string())?;
            let nvrtcDestroyProgram = *lib.get(b"nvrtcDestroyProgram").map_err(|e| e.to_string())?;
            let nvrtcGetProgramLogSize = *lib.get(b"nvrtcGetProgramLogSize").map_err(|e| e.to_string())?;
            let nvrtcGetProgramLog = *lib.get(b"nvrtcGetProgramLog").map_err(|e| e.to_string())?;
            let nvrtcGetErrorString = *lib.get(b"nvrtcGetErrorString").map_err(|e| e.to_string())?;
            
            Ok(NvrtcApi {
                _lib: lib,
                nvrtcCreateProgram,
                nvrtcCompileProgram,
                nvrtcGetPTXSize,
                nvrtcGetPTX,
                nvrtcDestroyProgram,
                nvrtcGetProgramLogSize,
                nvrtcGetProgramLog,
                nvrtcGetErrorString,
            })
        }
    }
}

impl CudaContext {
    pub fn init() -> Result<Self, String> {
        let cuda = CudaApi::load()?;
        let nvrtc = NvrtcApi::load()?;
        
        unsafe {
            let res = (cuda.cuInit)(0);
            if res != 0 {
                return Err(format!("cuInit failed: {}", res));
            }
            
            let mut device = 0;
            let res = (cuda.cuDeviceGet)(&mut device, 0);
            if res != 0 {
                return Err(format!("cuDeviceGet failed: {}", res));
            }
            
            let mut ctx = std::ptr::null_mut();
            let res = (cuda.cuCtxCreate_v2)(&mut ctx, 0, device);
            if res != 0 {
                return Err(format!("cuCtxCreate_v2 failed: {}", res));
            }
            
            Ok(CudaContext {
                cuda,
                nvrtc,
                ctx,
                device,
            })
        }
    }

    pub fn compile_to_ptx(&self, name: &str, src: &str) -> Result<Vec<u8>, String> {
        use std::ffi::CString;
        
        let c_src = CString::new(src).map_err(|e| e.to_string())?;
        let c_name = CString::new(name).map_err(|e| e.to_string())?;
        
        unsafe {
            let mut prog = std::ptr::null_mut();
            let res = (self.nvrtc.nvrtcCreateProgram)(
                &mut prog,
                c_src.as_ptr(),
                c_name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null()
            );
            if res != 0 {
                return Err(format!("nvrtcCreateProgram failed: {}", res));
            }
            
            let opt = CString::new("--std=c++11").unwrap();
            let options = [opt.as_ptr()];
            
            let res = (self.nvrtc.nvrtcCompileProgram)(prog, 1, options.as_ptr());
            if res != 0 {
                let mut log_size = 0;
                (self.nvrtc.nvrtcGetProgramLogSize)(prog, &mut log_size);
                let mut log_bytes = vec![0u8; log_size];
                (self.nvrtc.nvrtcGetProgramLog)(prog, log_bytes.as_mut_ptr() as *mut std::os::raw::c_char);
                let log = String::from_utf8_lossy(&log_bytes).to_string();
                (self.nvrtc.nvrtcDestroyProgram)(&mut prog);
                return Err(format!("NVRTC Compilation failed:\n{}", log));
            }
            
            let mut ptx_size = 0;
            let res = (self.nvrtc.nvrtcGetPTXSize)(prog, &mut ptx_size);
            if res != 0 {
                (self.nvrtc.nvrtcDestroyProgram)(&mut prog);
                return Err(format!("nvrtcGetPTXSize failed: {}", res));
            }
            
            let mut ptx = vec![0u8; ptx_size];
            let res = (self.nvrtc.nvrtcGetPTX)(prog, ptx.as_mut_ptr() as *mut std::os::raw::c_char);
            if res != 0 {
                (self.nvrtc.nvrtcDestroyProgram)(&mut prog);
                return Err(format!("nvrtcGetPTX failed: {}", res));
            }
            
            (self.nvrtc.nvrtcDestroyProgram)(&mut prog);
            Ok(ptx)
        }
    }

    pub fn load_module_and_get_function(&self, ptx: &[u8], entry_point: &str) -> Result<(*mut std::ffi::c_void, *mut std::ffi::c_void), String> {
        use std::ffi::CString;
        
        let c_entry = CString::new(entry_point).map_err(|e| e.to_string())?;
        
        unsafe {
            let mut module = std::ptr::null_mut();
            let res = (self.cuda.cuModuleLoadData)(&mut module, ptx.as_ptr() as *const std::ffi::c_void);
            if res != 0 {
                return Err(format!("cuModuleLoadData failed: {}", res));
            }
            
            let mut function = std::ptr::null_mut();
            let res = (self.cuda.cuModuleGetFunction)(&mut function, module, c_entry.as_ptr());
            if res != 0 {
                (self.cuda.cuModuleUnload)(module);
                return Err(format!("cuModuleGetFunction failed: {}", res));
            }
            
            Ok((module, function))
        }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            (self.cuda.cuCtxDestroy_v2)(self.ctx);
        }
    }
}


static CUDA_CONTEXT: OnceLock<Option<CudaContext>> = OnceLock::new();

/// Retrieve reference to initialized dynamic CUDA context.
pub fn get_cuda_context() -> Option<&'static CudaContext> {
    if is_simulate_cuda() {
        return None;
    }
    CUDA_CONTEXT.get_or_init(|| {
        match CudaContext::init() {
            Ok(ctx) => Some(ctx),
            Err(_) => None,
        }
    }).as_ref()
}
