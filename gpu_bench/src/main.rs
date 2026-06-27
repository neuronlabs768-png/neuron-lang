/// NEURON GPU Benchmark Suite
///
/// Measures execution time of fused element-wise kernels on CPU vs GPU (CUDA).
/// Designed to run on Google Colab (T4/A100) or any CUDA-capable machine.
///
/// Usage:
///   cargo run -p neuron-gpu-bench --release
///
/// Set NEURON_FORCE_CPU=1 to force CPU-only mode for comparison.

use std::time::Instant;

use neuron_compiler::compile;
use neuron_compiler::ir::DeviceTarget;
use neuron_runtime::vm::VM;
use neuron_runtime::device;

// ─────────────────────────────────────────────────────────────────────────────
//  Benchmark programs (NEURON source)
// ─────────────────────────────────────────────────────────────────────────────

/// Generates a chain of fused element-wise operations on tensors of size `n x n`.
/// This is the ideal workload for the GPU fusion backend.
fn gen_elementwise_chain(n: i64, chain_len: usize) -> String {
    let mut lines = vec![
        format!("fn main() -> Tensor[{}, {}]:", n, n),
        format!("  let x = zeros({}, {}) + 1.0", n, n),
    ];
    for i in 0..chain_len {
        let prev = if i == 0 { "x".to_string() } else { format!("v{}", i - 1) };
        // Alternate: negate -> gelu -> add const -> sigmoid -> relu
        let expr = match i % 5 {
            0 => format!("  let v{} = -{}", i, prev),
            1 => format!("  let v{} = gelu({})", i, prev),
            2 => format!("  let v{} = {} + 2.5", i, prev),
            3 => format!("  let v{} = sigmoid({})", i, prev),
            4 => format!("  let v{} = relu({})", i, prev),
            _ => unreachable!(),
        };
        lines.push(expr);
    }
    lines.push(format!("  return v{}", chain_len - 1));
    lines.join("\n") + "\n"
}

/// Generates a MatMul-heavy program (NOT fused on GPU — serves as a control).
fn gen_matmul_chain(n: i64, reps: usize) -> String {
    let mut lines = vec![
        format!("fn main() -> Tensor[{}, {}]:", n, n),
        format!("  let w = glorot({}, {})", n, n),
        format!("  let x = glorot({}, {})", n, n),
    ];
    for _ in 0..reps {
        lines.push("  let x = x @ w".to_string());
    }
    lines.push("  return x".to_string());
    lines.join("\n") + "\n"
}

/// Generates a mixed workload: matmul followed by element-wise activations.
fn gen_mixed_workload(n: i64) -> String {
    format!(
r#"fn main() -> Tensor[{n}, {n}]:
  let w1 = glorot({n}, {n})
  let w2 = glorot({n}, {n})
  let x = glorot({n}, {n})
  let h = x @ w1
  let h = relu(h)
  let h = h + 0.5
  let h = gelu(h)
  let h = h @ w2
  let h = sigmoid(h)
  let h = -h
  let h = h + 1.0
  return h
"#, n = n)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Benchmark runner
// ─────────────────────────────────────────────────────────────────────────────

struct BenchResult {
    name: String,
    cpu_ms: f64,
    gpu_ms: f64,
    speedup: f64,
    gpu_available: bool,
}

fn run_benchmark(name: &str, source: &str, device_tag: &str) -> Result<f64, String> {
    let out = compile(source, &format!("{}.nr", name))
        .map_err(|e| format!("Compile error: {:?}", e))?;

    // If device_tag is "cuda", set all IR nodes to CUDA(0)
    let mut prog = out.ir.clone();
    if device_tag == "cuda" {
        for func in &mut prog.functions {
            for block in &mut func.blocks {
                for node in &mut block.instructions {
                    node.device = DeviceTarget::CUDA(0);
                }
            }
        }
    } else {
        // Force CPU
        for func in &mut prog.functions {
            for block in &mut func.blocks {
                for node in &mut block.instructions {
                    node.device = DeviceTarget::CPU;
                }
            }
        }
    }

    // Warm up
    {
        let mut vm = VM::new();
        vm.load(&prog);
        let _ = vm.run_main();
    }

    // Timed run (average of 3 iterations)
    let mut total = 0.0;
    let iters = 3;
    for _ in 0..iters {
        let mut vm = VM::new();
        vm.load(&prog);
        let start = Instant::now();
        let _result = vm.run_main().map_err(|e| format!("Runtime error: {}", e))?;
        total += start.elapsed().as_secs_f64() * 1000.0;
    }

    Ok(total / iters as f64)
}

fn run_bench_pair(name: &str, source: &str) -> BenchResult {
    let gpu_available = device::Device::cuda_available();

    // CPU run (force CPU device)
    device::set_force_cpu(true);
    device::set_simulate_cuda(false);
    let cpu_ms = match run_benchmark(name, source, "cpu") {
        Ok(ms) => ms,
        Err(e) => {
            eprintln!("  CPU error: {}", e);
            f64::NAN
        }
    };
    device::set_force_cpu(false);

    // GPU run
    let gpu_ms = if gpu_available {
        match run_benchmark(name, source, "cuda") {
            Ok(ms) => ms,
            Err(e) => {
                eprintln!("  GPU error: {}", e);
                f64::NAN
            }
        }
    } else {
        f64::NAN
    };

    let speedup = if gpu_ms.is_finite() && cpu_ms.is_finite() && gpu_ms > 0.0 {
        cpu_ms / gpu_ms
    } else {
        f64::NAN
    };

    BenchResult {
        name: name.to_string(),
        cpu_ms,
        gpu_ms,
        speedup,
        gpu_available,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           NEURON GPU BACKEND BENCHMARK SUITE                    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Detect GPU
    let has_cuda = device::Device::cuda_available();
    if has_cuda {
        println!("  ✓ CUDA GPU detected — benchmarking CPU vs GPU");
    } else {
        println!("  ✗ No CUDA GPU detected — running CPU-only baselines");
        println!("    (GPU columns will show N/A)");
    }
    println!();

    // ── Benchmark Suite ──────────────────────────────────────────────────

    let benchmarks: Vec<(&str, String)> = vec![
        // Element-wise fusion benchmarks (these benefit from GPU fusion)
        ("elemwise_16x16_chain10",   gen_elementwise_chain(16, 10)),
        ("elemwise_64x64_chain10",   gen_elementwise_chain(64, 10)),
        ("elemwise_128x128_chain10", gen_elementwise_chain(128, 10)),
        ("elemwise_256x256_chain10", gen_elementwise_chain(256, 10)),
        ("elemwise_512x512_chain10", gen_elementwise_chain(512, 10)),
        ("elemwise_256x256_chain50", gen_elementwise_chain(256, 50)),

        // MatMul benchmarks (control — NOT fused on GPU)
        ("matmul_64x64_x20",  gen_matmul_chain(64, 20)),
        ("matmul_128x128_x20", gen_matmul_chain(128, 20)),
        ("matmul_256x256_x20", gen_matmul_chain(256, 20)),

        // Mixed workloads (matmul + activations)
        ("mixed_64",  gen_mixed_workload(64)),
        ("mixed_128", gen_mixed_workload(128)),
        ("mixed_256", gen_mixed_workload(256)),
    ];

    let mut results = Vec::new();

    for (name, source) in &benchmarks {
        print!("  Running {:.<45}", format!("{} ", name));
        let r = run_bench_pair(name, source);
        if r.gpu_available {
            if r.speedup.is_finite() {
                println!("CPU {:>8.2}ms  GPU {:>8.2}ms  ({:.1}x)", r.cpu_ms, r.gpu_ms, r.speedup);
            } else {
                println!("CPU {:>8.2}ms  GPU {:>8}  (N/A)", r.cpu_ms, "error");
            }
        } else {
            println!("CPU {:>8.2}ms  GPU {:>8}  (no GPU)", r.cpu_ms, "N/A");
        }
        results.push(r);
    }

    // ── Summary Table ────────────────────────────────────────────────────

    println!();
    println!("┌──────────────────────────────────┬────────────┬────────────┬──────────┐");
    println!("│ Benchmark                        │  CPU (ms)  │  GPU (ms)  │ Speedup  │");
    println!("├──────────────────────────────────┼────────────┼────────────┼──────────┤");

    for r in &results {
        let gpu_str = if r.gpu_ms.is_finite() {
            format!("{:>8.2}", r.gpu_ms)
        } else if r.gpu_available {
            "   error".to_string()
        } else {
            "     N/A".to_string()
        };
        let speedup_str = if r.speedup.is_finite() {
            format!("{:>5.1}x", r.speedup)
        } else {
            "   N/A".to_string()
        };
        println!("│ {:<32} │ {:>8.2}   │ {}   │ {:>6}   │",
                 r.name, r.cpu_ms, gpu_str, speedup_str);
    }

    println!("└──────────────────────────────────┴────────────┴────────────┴──────────┘");

    // ── Interpretation ───────────────────────────────────────────────────

    println!();
    if has_cuda {
        let fused_results: Vec<&BenchResult> = results.iter()
            .filter(|r| r.name.starts_with("elemwise"))
            .collect();

        if !fused_results.is_empty() {
            let avg_speedup: f64 = fused_results.iter()
                .filter(|r| r.speedup.is_finite())
                .map(|r| r.speedup)
                .sum::<f64>()
                / fused_results.iter().filter(|r| r.speedup.is_finite()).count().max(1) as f64;

            println!("  Average GPU speedup on fused element-wise ops: {:.1}x", avg_speedup);
        }

        let matmul_results: Vec<&BenchResult> = results.iter()
            .filter(|r| r.name.starts_with("matmul"))
            .collect();

        if !matmul_results.is_empty() {
            let avg_speedup: f64 = matmul_results.iter()
                .filter(|r| r.speedup.is_finite())
                .map(|r| r.speedup)
                .sum::<f64>()
                / matmul_results.iter().filter(|r| r.speedup.is_finite()).count().max(1) as f64;

            println!("  Average GPU speedup on matmul (not fused):     {:.1}x", avg_speedup);
        }
    } else {
        println!("  Re-run this benchmark on a CUDA-capable machine (e.g. Google Colab)");
        println!("  to see GPU vs CPU speedups.");
    }
    println!();
}
