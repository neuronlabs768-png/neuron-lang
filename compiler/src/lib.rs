/// NEURON Compiler — the complete frontend for the NEURON language.
///
/// Provides: Lexer → Parser → TypeChecker → IR Lowering
/// Now with module import resolution.

pub mod token;
pub mod lexer;
pub mod ast;
pub mod parser;
pub mod errors;
pub mod types;
pub mod ir;
pub mod lower;
pub mod transpiler;
pub mod py_transpiler;
pub mod cuda_codegen;

use errors::CompileResult;
use ir::IRProgram;
use ast::{Program, TopLevel};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Compile result: contains the IR program and any diagnostics.
pub struct CompileOutput {
    pub ir: IRProgram,
    pub result: CompileResult,
}

/// Search paths for module resolution, in priority order:
/// 1. Same directory as the importing file
/// 2. stdlib/ relative to the source file's directory
/// 3. stdlib/ relative to the compiler binary (for installed usage)
/// 4. NEURON_PATH environment variable (colon/semicolon separated)
fn resolve_module_path(module_name: &str, source_path: &str) -> Option<PathBuf> {
    // Convert module.submodule to module/submodule
    let relative = module_name.replace('.', std::path::MAIN_SEPARATOR_STR);

    let source = Path::new(source_path);
    let source_dir = source.parent().unwrap_or(Path::new("."));

    // Search order:
    let search_dirs: Vec<PathBuf> = vec![
        // 1. Same directory as the source file
        source_dir.to_path_buf(),
        // 2. stdlib/ relative to the source file
        source_dir.join("stdlib"),
        // 3. ../stdlib/ (if source is in examples/ or src/)
        source_dir.join("..").join("stdlib"),
        // 4. stdlib/ in current working directory
        PathBuf::from("stdlib"),
    ];

    let extensions = vec!["neuron", "nr"];

    for dir in &search_dirs {
        for ext in &extensions {
            let candidate = dir.join(format!("{}.{}", relative, ext));
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    // 5. Check NEURON_PATH environment variable
    if let Ok(neuron_path) = std::env::var("NEURON_PATH") {
        let sep = if cfg!(windows) { ';' } else { ':' };
        for dir in neuron_path.split(sep) {
            for ext in &extensions {
                let candidate = Path::new(dir).join(format!("{}.{}", relative, ext));
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

/// Parse a single source file into a Program AST.
fn parse_source(source: &str, filename: &str) -> Result<Program, CompileResult> {
    let tokens = match lexer::Lexer::new(source).tokenize() {
        Ok(t) => t,
        Err(e) => {
            let mut result = CompileResult::new(filename);
            result.add_error(e);
            return Err(result);
        }
    };

    match parser::Parser::new(tokens, filename).parse() {
        Ok(p) => Ok(p),
        Err(e) => {
            let mut result = CompileResult::new(filename);
            result.add_error(e);
            Err(result)
        }
    }
}

/// Resolve all imports in a program by loading and parsing imported modules.
/// Returns a new program with imported declarations prepended.
/// Tracks already-imported modules to prevent cycles and duplicates.
fn resolve_imports(
    program: Program,
    source_path: &str,
    imported: &mut HashSet<String>,
    result: &mut CompileResult,
) -> Program {
    let mut merged_top_levels: Vec<TopLevel> = Vec::new();

    // First pass: process imports and collect imported declarations
    for tl in &program.top_levels {
        if let TopLevel::Import(imp) = tl {
            if imp.is_python {
                // Python interop imports — skip file resolution, keep the Import node
                continue;
            }

            let module_name = &imp.module;

            // Skip if already imported (cycle prevention)
            if imported.contains(module_name) {
                continue;
            }

            // Resolve the module file
            match resolve_module_path(module_name, source_path) {
                Some(module_path) => {
                    imported.insert(module_name.clone());

                    let module_source = match std::fs::read_to_string(&module_path) {
                        Ok(s) => s,
                        Err(e) => {
                            result.add_warning(errors::NeuronWarning::new(
                                errors::WarningCode::ImportWarning,
                                format!("Cannot read module '{}': {}", module_name, e),
                                imp.span.clone(),
                            ));
                            continue;
                        }
                    };

                    let module_path_str = module_path.to_string_lossy().to_string();
                    match parse_source(&module_source, &module_path_str) {
                        Ok(module_program) => {
                            // Recursively resolve imports in the imported module
                            let resolved = resolve_imports(
                                module_program,
                                &module_path_str,
                                imported,
                                result,
                            );

                            // If specific names were requested (from X import A, B),
                            // filter to only those declarations. Otherwise import everything.
                            let names = &imp.names;

                            for mod_tl in resolved.top_levels {
                                if names.is_empty() {
                                    // `import nn` — import everything except main()
                                    match &mod_tl {
                                        TopLevel::Fn(f) if f.name == "main" => continue,
                                        _ => merged_top_levels.push(mod_tl),
                                    }
                                } else {
                                    // `from nn import Linear, Transformer` — selective
                                    let decl_name = match &mod_tl {
                                        TopLevel::Fn(f) => Some(&f.name),
                                        TopLevel::Model(m) => Some(&m.name),
                                        TopLevel::Layer(l) => Some(&l.name),
                                        TopLevel::Agent(a) => Some(&a.name),
                                        TopLevel::Causal(c) => Some(&c.name),
                                        _ => None,
                                    };

                                    if let Some(name) = decl_name {
                                        if names.contains(name) {
                                            merged_top_levels.push(mod_tl);
                                        }
                                    }
                                }
                            }
                        }
                        Err(parse_err) => {
                            for err in parse_err.errors {
                                result.add_warning(errors::NeuronWarning::new(
                                    errors::WarningCode::ImportWarning,
                                    format!("Error in imported module '{}': {}", module_name, err),
                                    imp.span.clone(),
                                ));
                            }
                        }
                    }
                }
                None => {
                    // Module not found — add a warning but don't fail
                    // (the type checker already defines imported names as Any)
                    result.add_warning(errors::NeuronWarning::new(
                        errors::WarningCode::ImportWarning,
                        format!(
                            "Module '{}' not found in search paths (stdlib/, ./, NEURON_PATH)",
                            module_name
                        ),
                        imp.span.clone(),
                    ));
                }
            }
        }
    }

    // Second pass: add all original top-level declarations (including Import nodes)
    for tl in program.top_levels {
        merged_top_levels.push(tl);
    }

    Program {
        top_levels: merged_top_levels,
    }
}

/// Compile source code through all frontend phases (no import resolution).
pub fn compile(source: &str, filename: &str) -> Result<CompileOutput, CompileResult> {
    let program = parse_source(source, filename)?;

    // Phase 3: Type check
    let mut checker = types::TypeChecker::new(filename);
    checker.check(&program);

    let type_result = checker.result;

    // If type errors, return them
    if type_result.has_errors() {
        return Err(type_result);
    }

    // Phase 4: Lower to IR
    let lowerer = lower::Lowerer::new();
    let ir = lowerer.lower(&program);

    Ok(CompileOutput {
        ir,
        result: type_result,
    })
}

/// Compile source code with full import resolution.
/// Resolves `import X` and `from X import Y` by loading .nr files
/// from stdlib/, the source directory, or NEURON_PATH.
pub fn compile_with_imports(source: &str, filename: &str) -> Result<CompileOutput, CompileResult> {
    let program = parse_source(source, filename)?;

    // Resolve imports
    let mut imported = HashSet::new();
    let mut import_result = CompileResult::new(filename);
    let resolved_program = resolve_imports(program, filename, &mut imported, &mut import_result);

    // Phase 3: Type check
    let mut checker = types::TypeChecker::new(filename);
    checker.check(&resolved_program);

    // Merge import warnings into type checker result
    for warn in import_result.warnings {
        checker.result.add_warning(warn);
    }

    let type_result = checker.result;

    if type_result.has_errors() {
        return Err(type_result);
    }

    // Phase 4: Lower to IR
    let lowerer = lower::Lowerer::new();
    let ir = lowerer.lower(&resolved_program);

    Ok(CompileOutput {
        ir,
        result: type_result,
    })
}

/// Type-check only with import resolution. Used for `neuronc check`.
pub fn check_with_imports(source: &str, filename: &str) -> CompileResult {
    let program = match parse_source(source, filename) {
        Ok(p) => p,
        Err(result) => return result,
    };

    let mut imported = HashSet::new();
    let mut import_result = CompileResult::new(filename);
    let resolved_program = resolve_imports(program, filename, &mut imported, &mut import_result);

    let mut checker = types::TypeChecker::new(filename);
    checker.check(&resolved_program);

    for warn in import_result.warnings {
        checker.result.add_warning(warn);
    }

    checker.result
}

/// Type-check only (no IR generation, no import resolution). Used for basic checks.
pub fn check_only(source: &str, filename: &str) -> CompileResult {
    let program = match parse_source(source, filename) {
        Ok(p) => p,
        Err(result) => return result,
    };

    let mut checker = types::TypeChecker::new(filename);
    checker.check(&program);
    checker.result
}

