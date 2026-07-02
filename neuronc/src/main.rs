/// neuronc — the NEURON Language Compiler CLI.
///
/// Usage:
///   neuronc check  <file.nr>   — type-check, print errors/warnings
///   neuronc build  <file.nr>   — compile to NEURON IR
///   neuronc run    <file.nr>   — compile and execute
///
/// Exit codes: 0 = success, 1 = errors, 2 = warnings only

use std::env;
use std::fs;
use std::process;

mod repl;
mod pkg;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let command = &args[1];

    match command.as_str() {
        "check" => {
            if args.len() < 3 {
                eprintln!("error: neuronc check requires a file argument");
                process::exit(1);
            }
            cmd_check(&args[2]);
        }
        "repl" => {
            repl::run_repl();
        }
        "add" => {
            if args.len() < 3 {
                eprintln!("error: neuronc add requires a dependency name");
                process::exit(1);
            }
            let dep_name = &args[2];
            let mut path = None;
            let mut git = None;
            let mut i = 3;
            while i < args.len() {
                if args[i] == "--path" && i + 1 < args.len() {
                    path = Some(args[i+1].as_str());
                    i += 2;
                } else if args[i] == "--git" && i + 1 < args.len() {
                    git = Some(args[i+1].as_str());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if path.is_none() && git.is_none() {
                path = Some("../");
            }
            if let Err(e) = pkg::add_dependency(".", dep_name, path, git) {
                eprintln!("error: {}", e);
                process::exit(1);
            }
        }
        "build" => {
            let path = if args.len() >= 3 { &args[2] } else { "." };
            let path_obj = std::path::Path::new(path);
            if path_obj.is_dir() && path_obj.join("neuron.toml").exists() {
                match pkg::build_package(path) {
                    Ok(source) => {
                        match neuron_compiler::compile(&source, "package_build") {
                            Ok(output) => {
                                let nir_path = path_obj.join("target").join("package.nir");
                                std::fs::create_dir_all(path_obj.join("target")).unwrap();
                                std::fs::write(&nir_path, format!("{:?}", output.ir)).unwrap();
                                println!("✓ Package built successfully to {:?}", nir_path);
                            }
                            Err(result) => {
                                eprintln!("error: package compilation failed");
                                for err in result.errors {
                                    eprintln!("  {}", err);
                                }
                                process::exit(1);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("error: {}", e);
                        process::exit(1);
                    }
                }
            } else {
                if args.len() < 3 {
                    eprintln!("error: neuronc build requires a file argument or a package directory with neuron.toml");
                    process::exit(1);
                }
                cmd_build(&args[2]);
            }
        }
        "run" => {
            if args.len() < 3 {
                eprintln!("error: neuronc run requires a file argument");
                process::exit(1);
            }
            cmd_run(&args[2]);
        }
        "jit" => {
            if args.len() < 3 {
                eprintln!("error: neuronc jit requires a file argument");
                process::exit(1);
            }
            cmd_jit(&args[2]);
        }
        "transpile" => {
            if args.len() < 3 {
                eprintln!("error: neuronc transpile requires a file argument");
                process::exit(1);
            }
            let file_path = &args[2];
            let mut target = "python";
            let mut output_path = None;
            let mut i = 3;
            while i < args.len() {
                if args[i] == "--target" && i + 1 < args.len() {
                    target = &args[i + 1];
                    i += 2;
                } else if (args[i] == "-o" || args[i] == "--output") && i + 1 < args.len() {
                    output_path = Some(&args[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if target != "python" {
                eprintln!("error: unsupported target '{}' (only 'python' is supported)", target);
                process::exit(1);
            }
            cmd_transpile(file_path, output_path);
        }
        "version" | "--version" | "-v" => {
            println!("neuronc {} — the NEURON Language Compiler", env!("CARGO_PKG_VERSION"));
            println!("Built for AGI model creation");
        }
        "help" | "--help" | "-h" => {
            print_usage();
        }
        _ => {
            eprintln!("error: unknown command '{}'. Use 'neuronc help' for usage.", command);
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!(
r#"neuronc — the NEURON Language Compiler

USAGE:
    neuronc <command> [options] <file.nr>

COMMANDS:
    check    Type-check a NEURON source file
    build    Compile to NEURON IR (produces .nir file, or builds package)
    run      Compile and execute a NEURON program
    jit      Compile and execute using native Rust JIT compilation
    transpile Transpile NEURON code to PyTorch Python script
    repl     Start interactive NEURON REPL
    add      Add a local or git dependency to neuron.toml
    version  Print version information

FLAGS:
    -h, --help       Print help
    -v, --version    Print version

EXAMPLES:
    neuronc check  examples/transformer.nr
    neuronc run    examples/simple_shapes.nr
    neuronc build  examples/bayesian_nn.nr

NEURON is a statically typed, natively differentiable language
designed for AGI model creation. Every function is differentiable
unless marked @opaque. Tensor shapes are verified at compile time.
Uncertainty, temporality, and causality are first-class type constructs."#
    );
}

fn read_source(path: &str) -> String {
    match fs::read_to_string(path) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("error: cannot read '{}': {}", path, e);
            process::exit(1);
        }
    }
}

fn cmd_check(path: &str) {
    let source = read_source(path);
    let result = neuron_compiler::check_with_imports(&source, path);

    let mut exit_code = 0;

    if result.has_errors() {
        eprintln!("\n{} — {} error(s) found:\n", path, result.errors.len());
        for err in &result.errors {
            eprintln!("  {}", err);
            eprintln!();
        }
        exit_code = 1;
    }

    if result.has_warnings() {
        eprintln!("\n{} — {} warning(s):\n", path, result.warnings.len());
        for warn in &result.warnings {
            eprintln!("  {}", warn);
            eprintln!();
        }
        if exit_code == 0 { exit_code = 0; } // Warnings don't fail
    }

    if exit_code == 0 {
        eprintln!("✓ {} — no errors", path);
    }

    process::exit(exit_code);
}

fn cmd_build(path: &str) {
    let source = read_source(path);

    match neuron_compiler::compile_with_imports(&source, path) {
        Ok(output) => {
            // Print warnings if any
            if output.result.has_warnings() {
                for warn in &output.result.warnings {
                    eprintln!("  {}", warn);
                }
            }

            let n_funcs = output.ir.functions.len();
            let n_globals = output.ir.globals.len();
            let total_ops: usize = output.ir.functions.iter().map(|f| f.blocks.iter().map(|b| b.instructions.len()).sum::<usize>()).sum();

            eprintln!("✓ {} — compiled to NEURON IR", path);
            eprintln!("  {} function(s), {} global(s), {} IR node(s)", n_funcs, n_globals, total_ops);

            // Print IR summary
            for func in &output.ir.functions {
                let func_ops: usize = func.blocks.iter().map(|b| b.instructions.len()).sum();
                eprintln!("  fn {}({} params) → {} nodes",
                    func.name, func.params.len(), func_ops);
            }
        }
        Err(result) => {
            eprintln!("\n{} — {} error(s) found:\n", path, result.errors.len());
            for err in &result.errors {
                eprintln!("  {}", err);
                eprintln!();
            }
            process::exit(1);
        }
    }
}

fn cmd_run(path: &str) {
    let source = read_source(path);

    match neuron_compiler::compile_with_imports(&source, path) {
        Ok(output) => {
            // Print warnings
            for warn in &output.result.warnings {
                eprintln!("  {}", warn);
            }

            // Execute via VM
            let mut vm = neuron_runtime::vm::VM::new();
            vm.load(&output.ir);

            match vm.run_main() {
                Ok(result) => {
                    match result {
                        neuron_runtime::vm::Value::Void => {}
                        _ => println!("{}", result.display()),
                    }
                }
                Err(e) => {
                    eprintln!("\nRUNTIME ERROR: {}", e);
                    process::exit(1);
                }
            }
        }
        Err(result) => {
            eprintln!("\n{} — {} error(s) found:\n", path, result.errors.len());
            for err in &result.errors {
                eprintln!("  {}", err);
                eprintln!();
            }
            process::exit(1);
        }
    }
}

fn cmd_jit(path: &str) {
    let source = read_source(path);

    match neuron_compiler::compile(&source, path) {
        Ok(output) => {
            // Print warnings
            for warn in &output.result.warnings {
                eprintln!("  {}", warn);
            }

            // 1. Transpile IR to optimized Rust source
            let rust_code = neuron_compiler::transpiler::Transpiler::transpile(&output.ir);

            // 2. Setup temporary Cargo project
            let temp_dir = std::env::temp_dir().join("neuron_jit_project");
            let src_dir = temp_dir.join("src");
            std::fs::create_dir_all(&src_dir).unwrap();

            let cargo_toml_content = r#"[package]
name = "neuron_jit"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
neuron-runtime = { path = "C:/Users/ADMIN/neuron-lang/runtime" }
"#;
            std::fs::write(temp_dir.join("Cargo.toml"), cargo_toml_content).unwrap();
            std::fs::write(src_dir.join("lib.rs"), rust_code).unwrap();

            // 3. Compile the JIT library using cargo build --release
            eprintln!("Compiling JIT library with cargo build --release...");
            let compile_start = std::time::Instant::now();
            let compile_status = std::process::Command::new("cargo")
                .arg("build")
                .arg("--release")
                .env("RUSTFLAGS", "-C target-cpu=native")
                .current_dir(&temp_dir)
                .status()
                .expect("Failed to run cargo build");

            if !compile_status.success() {
                eprintln!("error: JIT compilation failed");
                std::process::exit(1);
            }
            let compile_dur = compile_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("✓ JIT compilation completed in {:.2} ms", compile_dur);

            // 4. Load the compiled library
            eprintln!("Loading JIT dynamic library...");
            let lib_path = if cfg!(target_os = "windows") {
                temp_dir.join("target").join("release").join("neuron_jit.dll")
            } else if cfg!(target_os = "macos") {
                temp_dir.join("target").join("release").join("libneuron_jit.dylib")
            } else {
                temp_dir.join("target").join("release").join("libneuron_jit.so")
            };

            let lib = unsafe { libloading::Library::new(lib_path) }
                .expect("Failed to load compiled JIT library");

            // 5. Resolve and execute run_main
            let run_main: libloading::Symbol<fn(&mut neuron_runtime::vm::VM) -> neuron_runtime::vm::Value> = unsafe {
                lib.get(b"run_main")
            }.expect("Failed to resolve JIT run_main symbol");

            let mut vm = neuron_runtime::vm::VM::new();
            let run_start = std::time::Instant::now();
            let result = run_main(&mut vm);
            let run_dur = run_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("✓ JIT execution completed in {:.2} ms", run_dur);

            match result {
                neuron_runtime::vm::Value::Void => {}
                _ => println!("{}", result.display()),
            }
        }
        Err(result) => {
            eprintln!("\n{} — {} error(s) found:\n", path, result.errors.len());
            for err in &result.errors {
                eprintln!("  {}", err);
                eprintln!();
            }
            process::exit(1);
        }
    }
}

fn cmd_transpile(path: &str, output_path: Option<&String>) {
    let source = read_source(path);

    match neuron_compiler::compile_with_imports(&source, path) {
        Ok(output) => {
            // Print warnings if any
            if output.result.has_warnings() {
                for warn in &output.result.warnings {
                    eprintln!("  {}", warn);
                }
            }

            // Transpile to PyTorch Python code
            let py_code = neuron_compiler::py_transpiler::PyTranspiler::transpile(&output.ir);

            match output_path {
                Some(out) => {
                    if let Err(e) = std::fs::write(out, &py_code) {
                        eprintln!("error: failed to write output file: {}", e);
                        process::exit(1);
                    }
                    eprintln!("✓ Transpiled successfully to {}", out);
                }
                None => {
                    println!("{}", py_code);
                }
            }
        }
        Err(result) => {
            eprintln!("\n{} — {} error(s) found:\n", path, result.errors.len());
            for err in &result.errors {
                eprintln!("  {}", err);
                eprintln!();
            }
            process::exit(1);
        }
    }
}
