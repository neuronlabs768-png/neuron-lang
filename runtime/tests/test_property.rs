use neuron_compiler::compile;
use neuron_compiler::transpiler::Transpiler;
use neuron_runtime::vm::{Value, VM};
use std::fs;

/// Simple LCG random number generator for deterministic property test generation.
struct SimpleRng {
    state: usize,
}

impl SimpleRng {
    fn new(seed: usize) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> usize {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.state / 65536) % 32768
    }

    fn next_float(&mut self) -> f64 {
        let val = (self.next() % 1000) as f64 / 1000.0;
        (val * 100.0).round() / 100.0
    }
}

/// Generate a randomized, semantically valid NEURON program.
fn generate_random_program(id: usize) -> String {
    let mut rng = SimpleRng::new(id * 773 + 13);
    let num_ops = rng.next() % 8 + 6; // between 6 and 13 operations
    
    let mut code = String::new();
    code.push_str("fn test_fused_math(x: Tensor[2, 2]) -> Tensor[2, 2]:\n");
    
    let mut vars = vec!["x".to_string()];
    
    for j in 1..=num_ops {
        let op_type = rng.next() % 10;
        let v_k = &vars[rng.next() % vars.len()];
        
        let expr = match op_type {
            0 => format!("-{}", v_k),
            1 => format!("gelu({})", v_k),
            2 => format!("relu({})", v_k),
            3 => format!("sigmoid({})", v_k),
            4 => format!("tanh({})", v_k),
            5 => format!("{} + {:.4}", v_k, rng.next_float() + 0.1),
            6 => format!("{} - {:.4}", v_k, rng.next_float() + 0.1),
            7 => format!("{} * {:.4}", v_k, rng.next_float() + 0.1),
            8 => {
                let v_l = &vars[rng.next() % vars.len()];
                format!("{} + {}", v_k, v_l)
            }
            _ => {
                let v_l = &vars[rng.next() % vars.len()];
                format!("{} - {}", v_k, v_l)
            }
        };
        
        code.push_str(&format!("  let v{} = {}\n", j, expr));
        vars.push(format!("v{}", j));
    }
    
    // Add random control flow
    let final_var_idx = vars.len() - 1;
    let mut current_var = vars[final_var_idx].clone();
    
    if rng.next() % 2 == 0 {
        // Add an if condition block
        let next_var_id = final_var_idx + 1;
        code.push_str(&format!("  let v{} = {}\n", next_var_id, current_var));
        code.push_str("  if 1.0 > 0.5:\n");
        code.push_str(&format!("    let v{} = v{} + 0.5\n", next_var_id, next_var_id));
        code.push_str("  else:\n");
        code.push_str(&format!("    let v{} = v{} - 0.5\n", next_var_id, next_var_id));
        current_var = format!("v{}", next_var_id);
        vars.push(current_var.clone());
    }
    
    if rng.next() % 2 == 0 {
        // Add a loop block
        let final_var_idx = vars.len() - 1;
        let next_var_id = final_var_idx + 1;
        code.push_str(&format!("  let v{} = {}\n", next_var_id, current_var));
        code.push_str("  for i in range(3):\n");
        code.push_str(&format!("    let v{} = v{} * 1.1\n", next_var_id, next_var_id));
        current_var = format!("v{}", next_var_id);
    }
    
    code.push_str(&format!("  return {}\n\n", current_var));
    
    // Main entry point
    code.push_str(
        r#"fn main() -> Tensor[2, 2]:
  let x = zeros(2, 2) + 1.0
  let res = test_fused_math(x)
  return res
"#
    );
    
    code
}

#[test]
fn test_jit_vs_vm_property() {
    let num_cases = 100;
    println!("Generating and transpiling {} property test cases...", num_cases);
    
    let mut combined_rust = String::new();
    
    // Header — import helpers from the runtime crate (pre-compiled, not re-compiled)
    combined_rust.push_str(
r#"// ═══════════════════════════════════════════════════════════════════
//  NEURON JIT Property Testing Dynamic Library
// ═══════════════════════════════════════════════════════════════════
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]

use std::collections::HashMap;
use neuron_runtime::tensor::{Tensor, tensor_neg, tensor_gelu};
use neuron_runtime::vm::{Value, VM};
use neuron_runtime::jit_helpers::*;

fn jit_obj_call(vm: &mut VM, fn_name: &str, args: Vec<Value>) -> Value {
    vm.execute(fn_name, args).unwrap_or_else(|e| panic!("JIT call '{}' failed: {}", fn_name, e))
}

"#);

    let mut ir_programs = Vec::new();

    // Compile and transpile each case
    for i in 0..num_cases {
        let src = generate_random_program(i);
        let compile_res = match compile(&src, &format!("prop_{}.nr", i)) {
            Ok(out) => out,
            Err(err) => {
                panic!("Failed to compile generated source code for iteration {}:\nSource:\n{}\nError:\n{:?}", i, src, err);
            }
        };
        
        ir_programs.push(compile_res.ir.clone());
        
        // Transpile to Rust code
        let raw_rust = Transpiler::transpile(&compile_res.ir);
        
        // Post-process the JIT code to rename functions uniquely for index `i`
        // Extract only initialize_globals and user functions, skip the jit_obj_call dispatcher
        let globals_start = raw_rust.find("pub fn initialize_globals").unwrap();
        let entry_start = raw_rust.find("// --- Entry Point ---").unwrap();
        
        let mut body = raw_rust[globals_start..entry_start].to_string();
        
        // Strip the jit_obj_call dispatcher (it's never called in property tests)
        if let Some(disp_start) = body.find("// --- Dynamic Method Dispatcher ---") {
            if let Some(disp_fn_end) = body[disp_start..].find("\n}\n") {
                let end = disp_start + disp_fn_end + 3;
                body = format!("{}{}", &body[..disp_start], &body[end..]);
            }
        }
        
        // Rename functions uniquely for this test case
        body = body.replace("pub fn initialize_globals", &format!("pub fn initialize_globals_{}", i));
        body = body.replace("fn main(", &format!("fn main_{}(", i));
        body = body.replace("main(vm,", &format!("main_{}(vm,", i));
        body = body.replace("fn test_fused_math(", &format!("fn test_fused_math_{}(", i));
        body = body.replace("test_fused_math(vm,", &format!("test_fused_math_{}(vm,", i));
        
        // Construct unique exported run_main_i
        let run_main_i = format!(
r#"
#[no_mangle]
pub extern "Rust" fn run_main_{}(vm: &mut VM) -> Value {{
    initialize_globals_{}(vm);
    main_{}(vm, vec![])
}}
"#,
            i, i, i
        );
        
        combined_rust.push_str(&body);
        combined_rust.push_str(&run_main_i);
    }
    
    // ── No more embedded helpers! They come from neuron_runtime::jit_helpers ──
    
    // Setup cargo project in a FIXED directory for incremental compilation
    let temp_dir = std::env::temp_dir().join("neuron_jit_property_test_cache");
    let src_dir = temp_dir.join("src");
    fs::create_dir_all(&src_dir).unwrap();
    
    let runtime_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .to_string_lossy()
        .replace('\\', "/");
    let cargo_toml_content = format!(r#"[package]
name = "neuron_jit_prop_test"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
neuron-runtime = {{ path = "{}" }}

[profile.dev]
opt-level = 0
debug = false
"#, runtime_path);
    fs::write(temp_dir.join("Cargo.toml"), cargo_toml_content).unwrap();
    fs::write(src_dir.join("lib.rs"), combined_rust).unwrap();
    
    // Compile JIT
    println!("Compiling dynamic library (JIT cargo build)...");
    let compile_status = std::process::Command::new("cargo")
        .arg("build")
        .current_dir(&temp_dir)
        .status()
        .unwrap();
        
    assert!(compile_status.success(), "Combined property JIT build failed");
    
    let lib_path = if cfg!(target_os = "windows") {
        temp_dir.join("target").join("debug").join("neuron_jit_prop_test.dll")
    } else if cfg!(target_os = "macos") {
        temp_dir.join("target").join("debug").join("libneuron_jit_prop_test.dylib")
    } else {
        temp_dir.join("target").join("debug").join("libneuron_jit_prop_test.so")
    };
    
    // Copy DLL to avoid lock
    let unique_dll_name = format!(
        "neuron_jit_prop_{}.dll",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let load_lib_path = temp_dir.join("target").join("debug").join(&unique_dll_name);
    fs::copy(&lib_path, &load_lib_path).unwrap();
    
    println!("Loading JIT dynamic library...");
    let lib = unsafe { libloading::Library::new(&load_lib_path) }.unwrap();
    
    // Run comparison for all cases
    println!("Running property tests comparison...");
    for i in 0..num_cases {
        let ir = &ir_programs[i];
        
        // 1. Run Interpreter
        let mut vm_interpreter = VM::new();
        vm_interpreter.load(ir);
        let res_interpreter = vm_interpreter.run_main().unwrap();
        
        // 2. Run JIT
        let symbol_name = format!("run_main_{}", i);
        let res_jit = unsafe {
            let run_main: libloading::Symbol<fn(&mut VM) -> Value> = lib
                .get(symbol_name.as_bytes())
                .unwrap();
            let mut vm_jit = VM::new();
            run_main(&mut vm_jit)
        };
        
        // 3. Assert parity
        match (&res_interpreter, &res_jit) {
            (Value::Tensor(t_int), Value::Tensor(t_jit)) => {
                assert_eq!(t_int.shape, t_jit.shape, "Shape mismatch on case {}", i);
                for idx in 0..t_int.data.len() {
                    let diff = (t_int.data[idx] - t_jit.data[idx]).abs();
                    assert!(diff < 1e-5, "Value mismatch on case {}, index {}: interpreter={:.6}, JIT={:.6}", i, idx, t_int.data[idx], t_jit.data[idx]);
                }
            }
            _ => {
                panic!("Expected Tensor values from main, got interpreter={:?}, JIT={:?}", res_interpreter, res_jit);
            }
        }
    }
    
    println!("Successfully validated JIT == VM parity for {} cases!", num_cases);
    drop(lib);
}
