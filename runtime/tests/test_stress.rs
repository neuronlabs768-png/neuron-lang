use std::collections::HashMap;
use neuron_compiler::compile;
use neuron_compiler::transpiler::Transpiler;
use neuron_runtime::vm::{Value, VM};
use neuron_runtime::buffer::Buffer;
use neuron_runtime::tensor::{Tensor, tensor_matmul};
use neuron_runtime::autograd::GradTape;
use neuron_runtime::causal::{CausalModel, discover};
use rayon::prelude::*;

// --- JIT runner helper from test_jit.rs ---
fn run_jit(src: &str) -> Result<Value, String> {
    let compile_res = compile(src, "test_stress_jit_input.nr")
        .map_err(|e| format!("{:?}", e))?;
    let rust_code = Transpiler::transpile(&compile_res.ir);
    
    let temp_dir = std::env::temp_dir().join(format!(
        "neuron_stress_jit_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let src_dir = temp_dir.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    
    let cargo_toml_content = r#"[package]
name = "neuron_stress_jit"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
neuron-runtime = { path = "C:/Users/ADMIN/neuron-lang/runtime" }
"#;
    std::fs::write(temp_dir.join("Cargo.toml"), cargo_toml_content).unwrap();
    std::fs::write(src_dir.join("lib.rs"), rust_code).unwrap();
    
    let compile_status = std::process::Command::new("cargo")
        .arg("build")
        .current_dir(&temp_dir)
        .status()
        .map_err(|e| format!("Failed to run cargo: {:?}", e))?;
        
    if !compile_status.success() {
        return Err("JIT compilation failed".to_string());
    }
    
    let lib_path = if cfg!(target_os = "windows") {
        temp_dir.join("target").join("debug").join("neuron_stress_jit.dll")
    } else if cfg!(target_os = "macos") {
        temp_dir.join("target").join("debug").join("libneuron_stress_jit.dylib")
    } else {
        temp_dir.join("target").join("debug").join("libneuron_stress_jit.so")
    };
    
    let unique_dll_name = format!(
        "neuron_stress_jit_{}.dll",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let load_lib_path = temp_dir.join("target").join("debug").join(&unique_dll_name);
    std::fs::copy(&lib_path, &load_lib_path).unwrap();

    let lib = unsafe { libloading::Library::new(&load_lib_path) }
        .map_err(|e| format!("Failed to load JIT library: {:?}", e))?;
        
    let result = unsafe {
        let run_main: libloading::Symbol<fn(&mut VM) -> Value> = lib
            .get(b"run_main")
            .map_err(|e| format!("Failed to resolve run_main: {:?}", e))?;
        let mut vm = VM::new();
        run_main(&mut vm)
    };
    
    drop(lib);
    let _ = std::fs::remove_dir_all(&temp_dir);
    Ok(result)
}

fn run_interpreter(src: &str) -> Result<Value, String> {
    let compile_res = compile(src, "test_stress_jit_input.nr")
        .map_err(|e| format!("{:?}", e))?;
    let mut vm = VM::new();
    vm.load(&compile_res.ir);
    vm.run_main()
}

// --- 1. Memory Pool Concurrency Stress Test ---
#[test]
fn test_stress_memory_pool() {
    println!("Starting Memory Pool Concurrency Stress Test...");
    (0..16).into_par_iter().for_each(|t| {
        for i in 0..20_000 {
            let size = ((i % 200) + 1) * 64; // sizes up to 12800 elements
            let mut buf = Buffer::new(size);
            buf[0] = t as f64;
            buf[size - 1] = i as f64;
            assert_eq!(buf[0], t as f64);
            assert_eq!(buf[size - 1], i as f64);
        }
    });
    println!("Memory Pool Stress Test passed.");
}

// --- 2. Deep Autograd Tape Stress Test ---
#[test]
fn test_stress_autograd_tape_depth() {
    println!("Starting Deep Autograd Tape Stress Test...");
    let mut tape = GradTape::new();
    
    let mut x = Tensor::full(&[1, 1], 1.0);
    x.id = tape.alloc_id();
    
    let mut w = Tensor::full(&[1, 1], 2.0);
    w.id = tape.alloc_id();
    tape.parameter_ids.insert(w.id);
    
    let mut current = x;
    for _ in 0..4000 {
        current = tape.mul(&current, &w);
        let mut div_tensor = Tensor::full(&[1, 1], 1.0);
        div_tensor.id = tape.alloc_id();
        
        let div_res = tape.div(&current, &div_tensor);
        let neg_res = tape.neg(&div_res);
        current = tape.gelu(&neg_res);
    }
    
    tape.backward(current.id);
    assert!(tape.get_grad(w.id).is_some());
    let grad_val = tape.get_grad(w.id).unwrap();
    assert_eq!(grad_val.len(), 1);
    println!("Deep Tape Gradient result: {}", grad_val[0]);
    println!("Deep Autograd Tape Stress Test passed.");
}

// --- 3. Large-Scale Tensor Math Stress Test ---
#[test]
fn test_stress_large_tensors() {
    println!("Starting Large-Scale Tensor Math Stress Test...");
    let a = Tensor::full(&[400, 400], 1.0);
    let b = Tensor::full(&[400, 400], 2.5);
    
    for _ in 0..15 {
        let c = tensor_matmul(&a, &b);
        assert_eq!(c.shape, vec![400, 400]);
        // 400 * 1.0 * 2.5 = 1000.0
        assert_eq!(c.data[0], 1000.0);
        assert_eq!(c.data[400 * 400 - 1], 1000.0);
    }
    println!("Large-Scale Tensor Math Stress Test passed.");
}

// --- 4. JIT Compilation Scaling Stress Test ---
#[test]
fn test_stress_jit_compilation_scaling() {
    println!("Starting JIT Compilation Scaling Stress Test...");
    let mut src = String::new();
    src.push_str("fn func0(x: Float) -> Float:\n  return x + 1.0\n");
    for i in 1..80 {
        src.push_str(&format!(
            "fn func{}(x: Float) -> Float:\n  let prev = func{}(x)\n  return prev + 1.0\n",
            i, i - 1
        ));
    }
    src.push_str("fn main() -> Float:\n  return func79(0.0)\n");
    
    let vm_res = run_interpreter(&src).unwrap();
    let jit_res = run_jit(&src).unwrap();
    
    assert_eq!(vm_res.as_float(), 80.0);
    assert_eq!(jit_res.as_float(), 80.0);
    println!("JIT Compilation Scaling Stress Test passed.");
}

// --- 5. Causal Discovery and Graph Inference Stress Test ---
#[test]
fn test_stress_causal_discovery() {
    println!("Starting Causal Discovery and Graph Inference Stress Test...");
    let names = vec![
        "A".to_string(), "B".to_string(), "C".to_string(), "D".to_string(), "E".to_string(),
        "F".to_string(), "G".to_string(), "H".to_string(), "I".to_string(), "J".to_string(),
    ];
    let n = names.len();
    let mut weights = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i+1)..n {
            weights[i][j] = 0.15; // Causal DAG
        }
    }
    let noise_vars = vec![1.0; n];
    let noise_means = vec![0.0; n];
    
    let model = CausalModel::new(names.clone(), weights, noise_vars, noise_means);
    
    // Generate observational data
    let mut rng_val: i64 = 12345;
    let mut next_random = || {
        rng_val = (1103515245 * rng_val + 12345) % 2147483647;
        (rng_val as f64) / 2147483647.0
    };
    let mut next_gaussian = || {
        let u1 = next_random();
        let u2 = next_random();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    
    let n_samples = 1500;
    let mut data = vec![vec![0.0; n]; n_samples];
    for s in 0..n_samples {
        let mut u = vec![0.0; n];
        for i in 0..n {
            u[i] = next_gaussian();
        }
        // Solve structural equations sequentially since it's a DAG (i < j)
        for j in 0..n {
            let mut val = u[j];
            for i in 0..j {
                val += model.weights[i][j] * data[s][i];
            }
            data[s][j] = val;
        }
    }
    
    // Run PC discovery on generated data
    let discover_res = discover(&data, names.clone(), 0.05);
    assert_eq!(discover_res.names.len(), n);
    
    // Run SCM observe query
    let mut evidence = HashMap::new();
    evidence.insert("A".to_string(), 1.0);
    evidence.insert("B".to_string(), 1.5);
    let obs = model.observe(&evidence).unwrap();
    assert!(obs.contains_key("J"));
    
    // Run SCM intervene query
    let mut intervention = HashMap::new();
    intervention.insert("C".to_string(), 3.0);
    let inter = model.intervene(&intervention).unwrap();
    assert!(inter.contains_key("J"));
    
    // Run SCM counterfactual query
    let queries = vec!["I".to_string(), "J".to_string()];
    let cf = model.counterfactual(&evidence, &intervention, &queries).unwrap();
    assert!(cf.contains_key("J"));
    
    println!("Causal Discovery & Graph Inference Stress Test passed.");
}
