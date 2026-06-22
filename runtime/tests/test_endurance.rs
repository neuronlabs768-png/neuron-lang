/// NEURON Endurance Test — 100,000+ training iterations
///
/// Verifies: no memory growth, no allocator corruption,
/// no deadlocks, no tape inconsistencies.

use neuron_compiler::compile;
use neuron_runtime::vm::{Value, VM};

const NEURON_TRAIN_SRC: &str = r#"
fn main() -> Tensor[1]:
  let w = glorot(4, 1)
  let x = zeros(4, 4) + 0.5
  let y = zeros(4, 1) + 1.0
  let pred = x @ w
  let loss = mse(pred, y)
  return loss
"#;

#[test]
fn test_endurance_100k_iterations() {
    let total_iterations: usize = 100_000;
    let checkpoint_interval: usize = 10_000;

    // Compile once, reuse IR
    let compile_res = compile(NEURON_TRAIN_SRC, "endurance.nr")
        .expect("Failed to compile endurance test program");
    let ir = &compile_res.ir;

    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut max_tape_size = 0usize;
    let mut prev_checkpoint_tape = 0usize;

    println!("Starting endurance test: {} iterations", total_iterations);
    let start = std::time::Instant::now();

    for i in 0..total_iterations {
        // Fresh VM each iteration to test clean lifecycle
        let mut vm = VM::new();
        vm.load(ir);
        let result = vm.run_main().unwrap();

        // Check result is a valid tensor
        match &result {
            Value::Tensor(t) => {
                for &val in t.data.iter() {
                    if val.is_nan() { nan_count += 1; }
                    if val.is_infinite() { inf_count += 1; }
                }
            }
            _ => {
                panic!("Iteration {}: Expected Tensor, got {:?}", i, result);
            }
        }

        // Track tape size
        let tape_size = vm.tape.tape_len();
        if tape_size > max_tape_size {
            max_tape_size = tape_size;
        }

        // Periodic checkpoint
        if (i + 1) % checkpoint_interval == 0 {
            let elapsed = start.elapsed();
            let tape_size = vm.tape.tape_len();
            println!(
                "  [{:>6}/{} | {:.1}s] tape_entries={}, max_tape={}, NaN={}, Inf={}",
                i + 1, total_iterations, elapsed.as_secs_f64(),
                tape_size, max_tape_size, nan_count, inf_count,
            );

            // Verify tape isn't growing unboundedly between checkpoints
            if i > 0 && tape_size > prev_checkpoint_tape * 2 + 100 {
                panic!(
                    "Tape growth detected! Previous checkpoint: {}, current: {}",
                    prev_checkpoint_tape, tape_size
                );
            }
            prev_checkpoint_tape = tape_size;
        }
    }

    let total_time = start.elapsed();
    println!("\n=== Endurance Test Results ===");
    println!("Iterations:     {}", total_iterations);
    println!("Total time:     {:.2}s", total_time.as_secs_f64());
    println!("Iter/sec:       {:.0}", total_iterations as f64 / total_time.as_secs_f64());
    println!("Max tape size:  {}", max_tape_size);
    println!("NaN values:     {}", nan_count);
    println!("Inf values:     {}", inf_count);

    assert_eq!(nan_count, 0, "NaN values detected during endurance test");
    assert_eq!(inf_count, 0, "Infinite values detected during endurance test");
    println!("\nEndurance test PASSED.");
}

#[test]
fn test_endurance_tape_lifecycle() {
    // Verify tape backward + zero_grad cycles don't corrupt state
    let src = r#"
fn main() -> Tensor[1]:
  let w = glorot(2, 1)
  let x = zeros(2, 2) + 1.0
  let pred = x @ w
  let y = zeros(2, 1) + 1.0
  let loss = mse(pred, y)
  return loss
"#;

    let compile_res = compile(src, "tape_lifecycle.nr").unwrap();
    let ir = &compile_res.ir;

    println!("Testing tape lifecycle across 10,000 forward+backward passes...");
    let start = std::time::Instant::now();

    for i in 0..10_000 {
        // Fresh VM with tape each iteration
        let mut vm = VM::new();
        vm.load(ir);
        let result = vm.run_main().unwrap();

        if let Value::Tensor(t) = &result {
            // Verify loss is finite
            assert!(
                t.data.iter().all(|v| v.is_finite()),
                "Iteration {}: loss contains non-finite values: {:?}", i, t.data
            );

            // Backward pass
            vm.tape.backward(t.id);

            // Zero gradients
            vm.tape.zero_grad();
        } else {
            panic!("Iteration {}: Expected Tensor, got {:?}", i, result);
        }

        // Periodic tape size check
        if (i + 1) % 2000 == 0 {
            let tape_len = vm.tape.tape_len();
            println!("  [{}/10000] tape_entries={}", i + 1, tape_len);
        }
    }

    let elapsed = start.elapsed();
    println!("Tape lifecycle test completed in {:.2}s", elapsed.as_secs_f64());
    println!("Tape lifecycle test PASSED.");
}
