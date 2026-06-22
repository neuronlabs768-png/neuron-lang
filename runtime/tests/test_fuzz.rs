/// NEURON Compiler Fuzzer — 1,000+ malformed inputs
///
/// Feeds random, malformed, and edge-case source code into the compiler
/// and verifies it produces errors (not panics/crashes).

use neuron_compiler::compile;

/// Simple deterministic LCG random number generator
struct FuzzRng {
    state: u64,
}

impl FuzzRng {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state >> 33
    }

    fn next_range(&mut self, max: usize) -> usize {
        (self.next() as usize) % max.max(1)
    }

    fn next_char(&mut self) -> char {
        let chars = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-*/=()[]{}:., \n\t";
        chars[self.next_range(chars.len())] as char
    }

    fn random_string(&mut self, max_len: usize) -> String {
        let len = self.next_range(max_len) + 1;
        (0..len).map(|_| self.next_char()).collect()
    }

    fn random_ident(&mut self) -> String {
        let len = self.next_range(12) + 1;
        let first = b"abcdefghijklmnopqrstuvwxyz"[self.next_range(26)] as char;
        let rest: String = (0..len).map(|_| {
            let chars = b"abcdefghijklmnopqrstuvwxyz0123456789_";
            chars[self.next_range(chars.len())] as char
        }).collect();
        format!("{}{}", first, rest)
    }
}

fn try_compile(source: &str) -> Result<&'static str, &'static str> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compile(source, "fuzz.nr")
    }));
    match result {
        Ok(Ok(_)) => Ok("compiled"),
        Ok(Err(_)) => Err("error"),
        Err(_) => Err("PANIC"),
    }
}

#[test]
fn test_fuzz_compiler() {
    let mut rng = FuzzRng::new(42);
    let mut total = 0usize;
    let mut compiled = 0usize;
    let mut errored = 0usize;
    let mut panicked = 0usize;

    println!("=== NEURON Compiler Fuzzer ===\n");

    // ── Category 1: Random garbage (200 cases) ─────────────────────
    print!("Category 1: Random garbage...        ");
    let (mut c1_ok, mut c1_err, mut c1_panic) = (0, 0, 0);
    for _ in 0..200 {
        let src = rng.random_string(200);
        match try_compile(&src) {
            Ok(_) => { c1_ok += 1; compiled += 1; }
            Err("PANIC") => { c1_panic += 1; panicked += 1; }
            Err(_) => { c1_err += 1; errored += 1; }
        }
        total += 1;
    }
    println!("ok={}, err={}, panic={}", c1_ok, c1_err, c1_panic);

    // ── Category 2: Partial/truncated programs (200 cases) ──────────
    print!("Category 2: Partial programs...      ");
    let templates = [
        "fn", "fn f", "fn f(", "fn f():", "fn f() ->",
        "fn f() -> Tensor", "fn f() -> Tensor[",
        "fn f():\n  let", "fn f():\n  let x", "fn f():\n  let x =",
        "fn f():\n  return", "fn f():\n  if", "fn f():\n  for",
        "model M:\n  fn", "model M:\n  fn new(self):",
        "fn f():\n  let x = zeros(",
        "fn main():\n  let x = 1\n  let y =",
        "fn f():\n  let x = relu(",
        "fn main() -> Tensor[2]:\n  let x = zeros(2) +",
        "import",
    ];
    let (mut c2_ok, mut c2_err, mut c2_panic) = (0, 0, 0);
    for i in 0..200 {
        let template = templates[i % templates.len()];
        let extra = rng.random_string(30);
        let src = if rng.next() % 2 == 0 {
            format!("{} {}", template, extra)
        } else {
            template.to_string()
        };
        match try_compile(&src) {
            Ok(_) => { c2_ok += 1; compiled += 1; }
            Err("PANIC") => { c2_panic += 1; panicked += 1; }
            Err(_) => { c2_err += 1; errored += 1; }
        }
        total += 1;
    }
    println!("ok={}, err={}, panic={}", c2_ok, c2_err, c2_panic);

    // ── Category 3: Malformed expressions (200 cases) ───────────────
    print!("Category 3: Malformed expressions... ");
    let (mut c3_ok, mut c3_err, mut c3_panic) = (0, 0, 0);
    for i in 0..200 {
        let src = match i % 10 {
            0 => format!("fn f():\n  return {}", "((".repeat(rng.next_range(20) + 1)),
            1 => format!("fn f():\n  let x = 1 2 3 {} 5", rng.random_ident()),
            2 => format!("fn f():\n  let x = {}", "9".repeat(rng.next_range(50) + 20)),
            3 => format!("fn f():\n  let {} = {}", rng.random_ident(), ")".repeat(rng.next_range(10) + 1)),
            4 => format!("fn f():\n  {} + {} * {}", rng.random_ident(), rng.random_ident(), rng.random_ident()),
            5 => format!("fn f():\n  let x = [[[{}]]]", rng.random_string(20)),
            6 => format!("fn f() -> Tensor[{}]:\n  return zeros(1)", "9".repeat(rng.next_range(10) + 1)),
            7 => format!("fn f():\n  let x = \"unterminated string"),
            8 => format!("fn f():\n  if if if:\n    return 1"),
            _ => format!("fn f():\n  let x = {} @ {} @ {}", rng.random_ident(), rng.random_ident(), rng.random_ident()),
        };
        match try_compile(&src) {
            Ok(_) => { c3_ok += 1; compiled += 1; }
            Err("PANIC") => { c3_panic += 1; panicked += 1; }
            Err(_) => { c3_err += 1; errored += 1; }
        }
        total += 1;
    }
    println!("ok={}, err={}, panic={}", c3_ok, c3_err, c3_panic);

    // ── Category 4: Type abuse (200 cases) ──────────────────────────
    print!("Category 4: Type abuse...            ");
    let (mut c4_ok, mut c4_err, mut c4_panic) = (0, 0, 0);
    for i in 0..200 {
        let src = match i % 8 {
            0 => format!("fn f() -> Tensor[{}, {}]:\n  return zeros(1)", rng.next_range(1000), rng.next_range(1000)),
            1 => format!("fn f():\n  let x = {}()\n  return x", rng.random_ident()),
            2 => "fn f():\n  let x: Tensor[2] = 42\n  return x".to_string(),
            3 => format!("fn f({}: Tensor[1]):\n  return {} + 1", rng.random_ident(), rng.random_ident()),
            4 => format!("fn f():\n  let x = zeros({})\n  let y = ones({})\n  return x + y", rng.next_range(5) + 1, rng.next_range(5) + 1),
            5 => format!("fn f():\n  let x = zeros(2, 2)\n  let y = zeros(3, 3)\n  return x @ y"),
            6 => format!("fn f():\n  return mse_loss(1.0, 2.0)"),
            _ => format!("fn f(x: Tensor[2], y: Tensor[3]):\n  return x + y"),
        };
        match try_compile(&src) {
            Ok(_) => { c4_ok += 1; compiled += 1; }
            Err("PANIC") => { c4_panic += 1; panicked += 1; }
            Err(_) => { c4_err += 1; errored += 1; }
        }
        total += 1;
    }
    println!("ok={}, err={}, panic={}", c4_ok, c4_err, c4_panic);

    // ── Category 5: Stress patterns (200 cases) ─────────────────────
    print!("Category 5: Stress patterns...       ");
    let (mut c5_ok, mut c5_err, mut c5_panic) = (0, 0, 0);
    for i in 0..200 {
        let src = match i % 10 {
            0 => "".to_string(),                           // empty source
            1 => "   \n\n\t\t\n   ".to_string(),          // whitespace only
            2 => "# just a comment\n# another\n".to_string(),
            3 => format!("fn {}():\n  return 1", "a".repeat(rng.next_range(200) + 50)), // long ident
            4 => {
                // Deeply nested expression
                let depth = rng.next_range(50) + 10;
                let mut expr = "x".to_string();
                for _ in 0..depth {
                    expr = format!("relu({})", expr);
                }
                format!("fn f(x: Tensor[1]):\n  return {}", expr)
            }
            5 => {
                // Many parameters
                let n = rng.next_range(30) + 5;
                let params: Vec<String> = (0..n).map(|j| format!("p{}: Tensor[1]", j)).collect();
                format!("fn f({}):\n  return p0", params.join(", "))
            }
            6 => {
                // Many let bindings
                let n = rng.next_range(50) + 10;
                let mut body = String::new();
                for j in 0..n {
                    body.push_str(&format!("  let v{} = {}.0\n", j, j));
                }
                body.push_str(&format!("  return {}.0", n - 1));
                format!("fn f():\n{}", body)
            }
            7 => {
                // Nested if blocks
                let depth = rng.next_range(8) + 2;
                let mut code = "fn f():\n".to_string();
                let indent_base = 2;
                for d in 0..depth {
                    let indent = " ".repeat(indent_base + d * 2);
                    code.push_str(&format!("{}if 1 > 0:\n", indent));
                }
                let final_indent = " ".repeat(indent_base + depth * 2);
                code.push_str(&format!("{}return 1\n", final_indent));
                code
            }
            8 => "fn f():\n  return\n".to_string(),      // return without value
            _ => {
                // Multiple functions with same name
                format!("fn f():\n  return 1\n\nfn f():\n  return 2\n")
            }
        };
        match try_compile(&src) {
            Ok(_) => { c5_ok += 1; compiled += 1; }
            Err("PANIC") => { c5_panic += 1; panicked += 1; }
            Err(_) => { c5_err += 1; errored += 1; }
        }
        total += 1;
    }
    println!("ok={}, err={}, panic={}", c5_ok, c5_err, c5_panic);

    // ── Summary ─────────────────────────────────────────────────────
    println!("\n=== Fuzzer Results ===");
    println!("Total cases:  {}", total);
    println!("Compiled OK:  {}", compiled);
    println!("Errored:      {}", errored);
    println!("PANICKED:     {}", panicked);

    assert_eq!(panicked, 0, "{} fuzz cases caused compiler panics!", panicked);
    println!("\nFuzzer test PASSED — all errors were handled gracefully.");
}
