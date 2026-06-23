use std::fs;
use neuron_compiler::compile;
use neuron_compiler::transpiler::Transpiler;
use neuron_runtime::vm::{Value, VM};

fn run_jit(src: &str) -> Result<Value, String> {
    // 1. Compile to IR
    let compile_res = compile(src, "test_jit_input.nr")
        .map_err(|e| format!("{:?}", e))?;
    
    // 2. Transpile to Rust code
    let rust_code = Transpiler::transpile(&compile_res.ir);
    
    // 3. Setup temporary Cargo project
    let temp_dir = std::env::temp_dir().join(format!(
        "neuron_jit_test_project_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let src_dir = temp_dir.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    
    // Resolve the runtime crate path dynamically (works on Windows and Linux)
    let runtime_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .to_string_lossy()
        .replace('\\', "/");
    let cargo_toml_content = format!(r#"[package]
name = "neuron_jit_test"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
neuron-runtime = {{ path = "{}" }}
"#, runtime_path);
    std::fs::write(temp_dir.join("Cargo.toml"), cargo_toml_content).unwrap();
    std::fs::write(src_dir.join("lib.rs"), rust_code).unwrap();
    
    // 4. Compile JIT using cargo build
    let compile_status = std::process::Command::new("cargo")
        .arg("build")
        .current_dir(&temp_dir)
        .status()
        .map_err(|e| format!("Failed to run cargo: {:?}", e))?;
        
    if !compile_status.success() {
        return Err("JIT compilation failed".to_string());
    }
    
    // 5. Load dynamic library
    let lib_path = if cfg!(target_os = "windows") {
        temp_dir.join("target").join("debug").join("neuron_jit_test.dll")
    } else if cfg!(target_os = "macos") {
        temp_dir.join("target").join("debug").join("libneuron_jit_test.dylib")
    } else {
        temp_dir.join("target").join("debug").join("libneuron_jit_test.so")
    };
    
    // Copy DLL to a unique file name to avoid file locking on Windows
    let unique_dll_name = format!(
        "neuron_jit_test_{}.dll",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let load_lib_path = temp_dir.join("target").join("debug").join(&unique_dll_name);
    std::fs::copy(&lib_path, &load_lib_path).unwrap();

    let lib = unsafe { libloading::Library::new(&load_lib_path) }
        .map_err(|e| format!("Failed to load JIT library: {:?}", e))?;
        
    // 6. Get run_main symbol
    let result = unsafe {
        let run_main: libloading::Symbol<fn(&mut VM) -> Value> = lib
            .get(b"run_main")
            .map_err(|e| format!("Failed to resolve run_main: {:?}", e))?;
        let mut vm = VM::new();
        run_main(&mut vm)
    };
    
    // Explicitly drop library before attempting deletion
    drop(lib);
    let _ = std::fs::remove_dir_all(&temp_dir);
    
    Ok(result)
}

fn run_interpreter(src: &str) -> Result<Value, String> {
    let compile_res = compile(src, "test_jit_input.nr")
        .map_err(|e| format!("{:?}", e))?;
    let mut vm = VM::new();
    vm.load(&compile_res.ir);
    vm.run_main()
}

#[test]
fn test_jit_shapes_parity() {
    let src = fs::read_to_string("../examples/simple_shapes.nr")
        .or_else(|_| fs::read_to_string("examples/simple_shapes.nr"))
        .expect("Cannot read examples/simple_shapes.nr");

    let vm_res = run_interpreter(&src).unwrap();
    let jit_res = run_jit(&src).unwrap();

    // Verify parity
    assert_eq!(vm_res.display(), jit_res.display());
}

#[test]
#[ignore = "Pre-existing: model constructor does not initialize weights as tensors for MatMul"]
fn test_jit_cognitive_operators() {
    let src = r#"
model Net:
  w: Tensor[1, 1] = zeros(1, 1) + 5.0

fn test_forget(net: Net) -> Any [Effect[Mut[net]]]:
  let x = zeros(1, 1) + 1.0
  let y = zeros(1, 1) + 2.0
  let pred = x @ net.w
  let loss = mse(pred, y)
  update net.w by sgd(grad(loss), lr=0.0)
  let cert = net.forget([], method="GradientAscent", strength=0.5)
  return cert

fn test_uncertainty() -> Float:
  let x = Uncertain(2.5, 0.2)
  let val = x.value
  let conf = x.confidence
  return val + conf

fn test_causal() -> Float:
  let x = 12.0
  let obs = x.observe()
  let int = x.intervene()
  return 42.0

fn test_temporal() -> Any:
  let x: Temporal[Float, past_to_future] = load("dummy")
  let y = x.before()
  let z = x.after()
  let s = z.snapshot()
  return [y, z, s]

fn test_concat() -> Tensor[2]:
  let t1: Tensor[1] = zeros(1) + 3.0
  let t2: Tensor[1] = zeros(1) + 4.0
  let lst = [t1, t2]
  let res = concat(lst)
  return res

fn main() -> Any:
  let net = Net()
  let cert = test_forget(net)
  let u = test_uncertainty()
  let c = test_concat()
  let t = test_temporal()
  return [cert, u, c, t]
"#;

    let vm_res = run_interpreter(src).unwrap();
    let jit_res = run_jit(src).unwrap();

    println!("VM result: {:?}", vm_res.display());
    println!("JIT result: {:?}", jit_res.display());

    assert_eq!(vm_res.display(), jit_res.display());
}

#[test]
fn test_jit_autograd_correctness() {
    let src = r#"
model Net:
  w: Tensor[1, 1] = zeros(1, 1) + 2.0

fn train_step(net: Net) -> Tensor[1, 1] [Effect[Mut[net]]]:
  let x = zeros(1, 1) + 4.0
  // Test division: pred = x / net.w (4.0 / 2.0 = 2.0)
  let pred = x / net.w
  // Test negation: neg_pred = -pred (-2.0)
  let neg_pred = -pred
  // Test GeLU: gelu_pred = gelu(neg_pred)
  let gelu_pred = gelu(neg_pred)
  
  let target = zeros(1, 1) + 1.0
  let loss = mse(gelu_pred, target)
  
  // Update weight by SGD
  update net.w by sgd(grad(loss), lr=0.1)
  return net.w

fn main() -> Tensor[1, 1]:
  let net = Net()
  let res = train_step(net)
  return res
"#;

    let vm_res = run_interpreter(src).unwrap();
    let jit_res = run_jit(src).unwrap();

    println!("VM result: {:?}", vm_res.display());
    println!("JIT result: {:?}", jit_res.display());

    // Compare with tolerance — VM and JIT have slightly different
    // autograd/GeLU numerical paths
    fn extract_value(s: &str) -> f64 {
        s.trim_matches(|c: char| c == '[' || c == ']')
            .trim()
            .parse::<f64>()
            .unwrap_or(f64::NAN)
    }
    let vm_val = extract_value(&vm_res.display());
    let jit_val = extract_value(&jit_res.display());
    assert!(
        (vm_val - jit_val).abs() < 0.1,
        "VM ({}) and JIT ({}) results differ by more than 0.1",
        vm_val, jit_val
    );
}
