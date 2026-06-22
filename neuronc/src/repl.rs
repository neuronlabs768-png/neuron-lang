/// NEURON REPL — Interactive Read-Eval-Print Loop.

use std::io::{self, Write};
use neuron_compiler::{compile, types::TypeChecker, parser::Parser, lexer::Lexer};
use neuron_runtime::vm::VM;

pub fn run_repl() {
    println!("NEURON REPL — Version {}", env!("CARGO_PKG_VERSION"));
    println!("Type :q or :quit to exit, :type <expr> for type info, :explain <expr> for explainability.");
    println!("Every expression is differentiable. Try: do(treatment = 1.0)");
    println!();

    let mut accumulated_code = String::new();
    let mut vm = VM::new();

    loop {
        print!("neuron> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                println!("Error reading input: {}", e);
                break;
            }
        }

        let line = input.trim();
        if line.is_empty() {
            continue;
        }

        if line == ":q" || line == ":quit" {
            break;
        }

        if line.starts_with(":type ") {
            let expr = &line[6..];
            handle_type_command(&accumulated_code, expr);
            continue;
        }

        if line.starts_with(":explain ") {
            let expr = &line[9..];
            handle_explain_command(&accumulated_code, expr);
            continue;
        }

        // Try to evaluate as an expression first: fn main() { print(<expr>); }
        let expr_code = format!("{}\nfn main() {{ print({}); }}", accumulated_code, line);
        if let Ok(output) = compile(&expr_code, "repl_expr") {
            // Run the compiled program
            vm.load(&output.ir);
            if let Err(e) = vm.run_main() {
                println!("Runtime Error: {}", e);
            }
            continue;
        }

        // Try to evaluate as a statement / declaration at top-level
        let stmt_code = format!("{}\n{}\nfn main() {{}}", accumulated_code, line);
        match compile(&stmt_code, "repl_stmt") {
            Ok(output) => {
                // If it compiled, update accumulated_code
                accumulated_code.push_str("\n");
                accumulated_code.push_str(line);

                // Run it to initialize any new globals/models/functions in the VM
                vm.load(&output.ir);
                if let Err(e) = vm.run_main() {
                    println!("Runtime Error: {}", e);
                }
            }
            Err(result) => {
                // Print the compile errors
                for err in result.errors {
                    println!("Compile Error: {}", err);
                }
            }
        }
    }
}

fn handle_type_command(accumulated_code: &str, expr: &str) {
    // Wrap the expr in a dummy function so we can type check it:
    // fn __repl_temp_expr__() { expr }
    let test_code = format!("{}\nfn __repl_temp_expr__() {{ {} }}", accumulated_code, expr);
    
    // Parse
    let tokens = match Lexer::new(&test_code).tokenize() {
        Ok(t) => t,
        Err(e) => {
            println!("Lex Error: {}", e);
            return;
        }
    };
    let program = match Parser::new(tokens, "repl_type").parse() {
        Ok(p) => p,
        Err(e) => {
            println!("Parse Error: {}", e);
            return;
        }
    };

    let mut checker = TypeChecker::new("repl_type");
    checker.check(&program);

    if checker.result.has_errors() {
        for err in checker.result.errors {
            println!("Type Error: {}", err);
        }
    } else if let Some(ty) = checker.lookup("__repl_temp_expr__") {
        // Since it's a function type, the return type is what we want
        if let neuron_compiler::types::NType::Fn_(_, ret, _) = ty {
            println!("Type: {:?}", ret);
        } else {
            println!("Type: {:?}", ty);
        }
    } else {
        println!("Could not determine type of expression.");
    }
}

fn handle_explain_command(accumulated_code: &str, expr: &str) {
    // Wrap in main with explain(expr)
    let explain_code = format!("{}\nfn main() {{ explain({}); }}", accumulated_code, expr);
    match compile(&explain_code, "repl_explain") {
        Ok(output) => {
            let mut vm = VM::new();
            vm.load(&output.ir);
            match vm.run_main() {
                Ok(result) => {
                    println!("{}", result.display());
                }
                Err(e) => {
                    println!("Runtime Error: {}", e);
                }
            }
        }
        Err(result) => {
            for err in result.errors {
                println!("Compile Error: {}", err);
            }
        }
    }
}
