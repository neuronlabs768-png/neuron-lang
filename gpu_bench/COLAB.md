# NEURON GPU Benchmark — Colab Instructions

## Quick Start (Colab)

Paste these into **separate Colab cells** in a notebook with **GPU runtime enabled**
(`Runtime → Change runtime type → T4 GPU`).

---

### Cell 1: Verify GPU
```python
!nvidia-smi
```

### Cell 2: Install Rust
```python
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
import os
os.environ['PATH'] = f"{os.environ['HOME']}/.cargo/bin:" + os.environ['PATH']
!rustc --version
```

### Cell 3: Clone & Build
```python
!git clone --depth 1 https://github.com/neuronlabs-ai/neuron-lang.git ~/neuron-lang 2>/dev/null || (cd ~/neuron-lang && git pull)
%cd ~/neuron-lang
!RUSTFLAGS="-C target-cpu=native" cargo build -p neuron-gpu-bench --release 2>&1 | tail -10
```

### Cell 4: Run GPU Benchmark
```python
!cd ~/neuron-lang && RUSTFLAGS="-C target-cpu=native" cargo run -p neuron-gpu-bench --release
```

---

## What the benchmark measures

| Benchmark Category | What it tests | GPU benefit? |
|---|---|---|
| `elemwise_*` | Fused element-wise chains (neg→gelu→add→sigmoid→relu) on varying tensor sizes | **Yes** — these are fused into single CUDA kernels |
| `matmul_*` | Chained matrix multiplications | **No** — MatMul is not in the GPU fusion set yet |
| `mixed_*` | MatMul + activations combined | **Partial** — only the activation chains run on GPU |

## Expected results on T4

On a Colab T4 GPU, you should see:
- **Element-wise fusion**: 5-50x speedup on large tensors (256x256+)
- **MatMul**: ~1x (no GPU acceleration yet — runs on CPU)
- **Mixed**: 1-3x depending on activation vs matmul ratio
