use neuron_compiler::compile;
use std::panic;

/// Generate different types of malformed programs.
fn generate_fuzz_input(i: usize) -> String {
    let mut rng = i;
    // Simple LCG random generator for test determinism
    let mut next_rand = move || {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        (rng / 65536) % 32768
    };

    match next_rand() % 6 {
        0 => {
            // Completely random binary/garbage bytes
            let len = next_rand() % 100 + 1;
            let mut s = String::new();
            for _ in 0..len {
                let c = (next_rand() % 256) as u8 as char;
                s.push(c);
            }
            s
        }
        1 => {
            // Unbalanced delimiters / missing parser structure
            let parts = vec![
                "fn ", "test(", "x: Tensor[2, 2]", ") -> ", "Tensor[2, 2]", ":",
                "\n  let y = ", " - ", "x", "\n  return ", "y", " { ", " } ", "[", "]", "(", ")"
            ];
            let num_parts = next_rand() % 15 + 1;
            let mut s = String::new();
            for _ in 0..num_parts {
                s.push_str(parts[next_rand() % parts.len()]);
            }
            s
        }
        2 => {
            // Mismatched shape check definitions (typecheck error targets)
            format!(
                r#"fn test_shapes(x: Tensor[{}, {}]) -> Tensor[{}, {}]:
  let y = x @ x
  return y
"#,
                next_rand() % 5, next_rand() % 5, next_rand() % 5, next_rand() % 5
            )
        }
        3 => {
            // Misplaced variables or invalid keywords
            let keywords = vec![
                "let", "fn", "return", "model", "update", "by", "grad", "zeros", "ones", "stop_grad",
                "before", "after", "snapshot", "observed", "intervened", "uncertain", "random"
            ];
            let mut s = String::new();
            for _ in 0..(next_rand() % 20 + 2) {
                s.push_str(keywords[next_rand() % keywords.len()]);
                s.push_str(" ");
            }
            s
        }
        4 => {
            // Partial temporal/causal statements (checking leak rules)
            format!(
                r#"fn test_temporal(x: Tensor[2, 2]) -> Tensor[2, 2]:
  let y = x.{}
  let z = {}
  return z
"#,
                if next_rand() % 2 == 0 { "before(1.0)" } else { "after(1.0)" },
                if next_rand() % 2 == 0 { "y" } else { "y.snapshot(1.0)" }
            )
        }
        _ => {
            // Randomly mutated correct program
            let correct = r#"fn test_fused(x: Tensor[2, 2]) -> Tensor[2, 2]:
  let y = -x
  let z = gelu(y)
  let w = z + 1.5
  return w
"#;
            let mut chars: Vec<char> = correct.chars().collect();
            let num_mutations = next_rand() % 5 + 1;
            for _ in 0..num_mutations {
                let idx = next_rand() % chars.len();
                chars[idx] = (next_rand() % 95 + 32) as u8 as char; // Printable ASCII
            }
            chars.into_iter().collect()
        }
    }
}

#[test]
fn test_compiler_fuzzing() {
    let mut panic_count = 0;
    let iterations = 10000;

    for i in 0..iterations {
        let input = generate_fuzz_input(i);
        
        let result = panic::catch_unwind(|| {
            let _ = compile(&input, "fuzz_input.nr");
        });
        
        if result.is_err() {
            println!("Panic occurred on iteration {} with input:\n{:?}", i, input);
            panic_count += 1;
        }
    }

    assert_eq!(panic_count, 0, "Compiler fuzzer encountered {} panics!", panic_count);
}
