<p align="center">
  <h1 align="center">NEURON</h1>
  <p align="center"><strong>The language that makes AI think differently.</strong></p>
  <p align="center">
    A statically typed, natively differentiable programming language<br>
    designed from the ground up for AGI model creation.
  </p>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#why-neuron">Why NEURON?</a> •
  <a href="#language-tour">Language Tour</a> •
  <a href="#installation">Installation</a> •
  <a href="#examples">Examples</a>
</p>

---

## Why NEURON?

Every existing ML framework bolts gradients onto a general-purpose language as an afterthought. **NEURON makes differentiation a first-class citizen of the type system.** Every function is differentiable unless you explicitly say otherwise. Tensor shapes are verified at compile time, not at 3 AM when your training run crashes.

```python
# NEURON — this is the entire language you need to build a transformer
model Transformer(d_model: Int, n_heads: Int):
  wq: Tensor[d_model, d_model] = glorot(d_model, d_model)
  wk: Tensor[d_model, d_model] = glorot(d_model, d_model)
  wv: Tensor[d_model, d_model] = glorot(d_model, d_model)

  fn forward(self, x: Tensor[B, S, d_model]) -> Tensor[B, S, d_model]:
    let q = x @ self.wq
    let k = x @ self.wk
    let v = x @ self.wv
    let scores = softmax(q @ transpose(k))
    return scores @ v

  fn train(self, data: Dataset) [Effect[Mut[self]]]:
    for batch in data:
      let loss = cross_entropy(self.forward(batch.x), batch.y)
      update self by adam(grad(loss), lr=3e-4)
```

**No `torch.nn.Module`. No `@tf.function`. No `.backward()` calls.** Just math, expressed directly.

### What makes NEURON different

| Feature | PyTorch/JAX | NEURON |
|---|---|---|
| Gradients | Library call (`.backward()`) | **Built into the language** — every `fn` is differentiable |
| Tensor shapes | Runtime crash | **Compile-time verification** |
| Temporal data | You manage it | **Type-tracked** — compiler catches lookahead bias |
| Uncertainty | Manual | **First-class type** — `Uncertain[Tensor]` propagates |
| Causality | Separate library | **Built-in** — `observe`, `intervene`, `counterfactual` |
| Training loop | Boilerplate | **One line** — `update self by adam(grad(loss))` |

---

## Quickstart

```bash
# Build from source (requires Rust 1.70+)
git clone https://github.com/neuron-lang/neuron
cd neuron
cargo build --release

# Run your first program
./target/release/neuronc run examples/simple_shapes.nr

# Start the interactive REPL
./target/release/neuronc repl
```

### Hello, Gradient World

Create `hello.nr`:

```python
fn main() -> Tensor[1, 1]:
  let w = glorot(2, 1)
  let x = zeros(2, 2) + 1.0
  let y = zeros(2, 1) + 3.0
  let pred = x @ w
  let loss = mse(pred, y)
  return loss
```

```bash
neuronc run hello.nr
# => Tensor([1, 1]) data=[2.847...]
```

That's it. Tensor created, matmul computed, loss calculated — all with automatic differentiation tracking, zero boilerplate.

---

## Language Tour

### Tensors and Shapes

Tensor shapes are part of the type system. The compiler catches mismatches before your code ever runs:

```python
fn matmul_example(a: Tensor[3, 4], b: Tensor[4, 5]) -> Tensor[3, 5]:
  return a @ b  # shapes match: [3,4] @ [4,5] = [3,5]

fn broken(a: Tensor[3, 4], b: Tensor[6, 5]) -> Tensor[3, 5]:
  return a @ b  # COMPILE ERROR: dimension 4 != 6
```

### Automatic Differentiation

Every function is differentiable by default. Use `grad()` to compute gradients and `update` to apply optimizers:

```python
model LinearRegression:
  w: Tensor[4, 1] = glorot(4, 1)

  fn predict(self, x: Tensor[B, 4]) -> Tensor[B, 1]:
    return x @ self.w

  fn train_step(self, x: Tensor[B, 4], y: Tensor[B, 1]) [Effect[Mut[self]]]:
    let loss = mse(self.predict(x), y)
    update self.w by adam(grad(loss), lr=0.001)
```

Want to exclude something from gradients? Use `stop_grad()` or mark with `@opaque`:

```python
@opaque
fn preprocessing(x: Tensor[B, D]) -> Tensor[B, D]:
  return relu(x)  # no gradient tracked through this
```

### Temporal Types

NEURON tracks data flow through time at the type level:

```python
fn safe_strategy(prices: Temporal[Tensor, past_to_future]) -> Tensor:
  let ma = prices.before(20)   # uses only past data
  return ma

fn buggy_strategy(prices: Temporal[Tensor, past_to_future]) -> Tensor:
  let future = prices.after(5)  # COMPILE ERROR: temporal leak detected
  return future
```

### Causal Inference

```python
fn treatment_effect(model, patient_data):
  let observed = observe(model, treatment=1, data=patient_data)
  let intervened = intervene(model, treatment=1)  # do-calculus
  let cf = counterfactual(model, treatment=0, evidence=patient_data)
  return intervened - cf  # individual treatment effect
```

### Uncertainty

```python
fn bayesian_predict(w: Uncertain[Tensor], x: Tensor) -> Uncertain[Tensor]:
  let pred = x @ w            # uncertainty propagates through matmul
  if pred.confidence < 0.8:   # compiler warns if you forget this check
    return fallback(x)
  return pred
```

### Effect System

Side effects (mutation, I/O) must be declared in the type signature:

```python
fn pure_fn(x: Tensor[2]) -> Tensor[2]:
  return relu(x)  # no effects needed

fn train(self, data) [Effect[Mut[self], IO]]:
  # must declare mutations and I/O
  update self.w by adam(grad(loss))
```

---

## CLI Reference

```
neuronc - the NEURON Language Compiler

COMMANDS:
    check    Type-check a source file (no execution)
    build    Compile to NEURON IR
    run      Compile and execute via interpreter
    jit      Compile and execute via native Rust JIT
    repl     Interactive REPL with :type and :explain
    add      Add dependency to neuron.toml
    version  Print version
```

---

## Examples

| File | Description |
|---|---|
| `simple_shapes.nr` | Basic tensor operations and shape verification |
| `training_demo.nr` | Linear regression with SGD optimizer |
| `transformer.nr` | Multi-head attention transformer model |
| `agi_model.nr` | Full AGI agent architecture |
| `stress_test.nr` | Compiler stress test with complex programs |

---

## Standard Library

| Module | Contents |
|---|---|
| `nn` | Linear, LayerNorm, MultiHeadAttention, FeedForward, Embedding, Dropout, Transformer |
| `optim` | Adam, SGD, AdamW, learning rate schedulers |
| `distributions` | Normal, Bernoulli, Categorical, sampling, conditioning |
| `data` | DataLoader, Dataset, batching utilities |
| `causal` | Causal graph discovery, intervention, counterfactuals |
| `rl` | PPO, DQN, environment abstractions |
| `finance` | OHLCV data, rolling statistics, temporal analysis |
| `agi` | AGI agent primitives, memory, planning |

---

## Architecture

```
neuron-lang/
  compiler/           # Frontend
    src/
      lexer.rs            Tokenization
      parser.rs           Recursive descent parser
      types.rs            Type checker (shapes, temporal, causal, uncertainty)
      lower.rs            AST to IR with basic block CFG
      transpiler.rs       IR to Rust (JIT compilation)
      errors.rs           Rust-style error display
  runtime/             # Backend
    src/
      vm.rs               Block-based interpreter with call stack
      autograd.rs         Gradient tape, backward pass, Adam/SGD
      tensor.rs           N-dimensional tensor operations
      causal.rs           Causal inference (observe/intervene/counterfactual)
      jit_helpers.rs      Shared runtime for JIT-compiled code
  neuronc/             # CLI tool
  stdlib/              # Standard library (.nr files)
  examples/            # Example programs
```

---

## Testing

NEURON has been battle-tested:

```bash
cargo test                    # Run all tests

# Individual suites:
cargo test --test test_property    # 100 random programs: VM == JIT parity
cargo test --test test_endurance   # 100,000 training iterations, zero memory leaks
cargo test --test test_fuzz        # 1,000 malformed inputs, zero compiler panics
cargo test --test test_training    # Gradient descent convergence
cargo test --test test_causal      # Causal inference operations
```

| Suite | Cases | Result |
|---|---|---|
| Property tests (VM == JIT) | 100 | All pass |
| Endurance (100k iterations) | 100,000 | 0 NaN, 0 Inf, constant memory |
| Compiler fuzzing | 1,000 | 0 panics |
| Integration tests | 20+ | All pass |

---

## Current Status: Alpha

NEURON is in **technical preview**. The core language is complete and tested:

- Full compiler pipeline (lex, parse, typecheck, lower, JIT)
- Working interpreter and JIT compiler with verified parity
- Automatic differentiation with tape-based autograd
- Type-safe tensor shapes, temporal types, causal types, uncertainty types
- Effect system for mutation tracking
- Multi-file module imports with `import` and `from ... import` syntax
- Standard library (nn, optim, distributions, data, causal, rl, finance)
- CLI with REPL, check, build, run, jit commands
- Comprehensive test suite (100k+ test iterations)

**Coming soon:**
- GPU acceleration (CUDA backend)
- Model serialization (save/load)
- Package registry

---

## Building from Source

Requirements: Rust 1.70+ and Cargo.

```bash
git clone https://github.com/neuron-lang/neuron
cd neuron
cargo build --release
```

The `neuronc` binary will be at `target/release/neuronc`.

---

## License

This project is licensed under the Business Source License 1.1 (BSL 1.1) - see the [LICENSE](LICENSE) file for details.

---

**NEURON** — Because the language you think in shapes the intelligence you create.
