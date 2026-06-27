#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  NEURON GPU Benchmark — Google Colab Setup Script
# ═══════════════════════════════════════════════════════════════════════
#
#  Run this in a Colab cell with GPU runtime enabled:
#
#    !curl -sSf https://raw.githubusercontent.com/neuronlabs-ai/neuron-lang/main/gpu_bench/colab_setup.sh | bash
#
#  Or paste cell-by-cell (see below).
#
# ═══════════════════════════════════════════════════════════════════════

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  NEURON GPU Benchmark — Colab Environment Setup                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Verify GPU ──────────────────────────────────────────────────
echo "Step 1/4: Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "  ✗ nvidia-smi not found. Please enable GPU runtime:"
    echo "    Runtime → Change runtime type → T4 GPU"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "  ✓ GPU detected"
echo ""

# ── Step 2: Install Rust ────────────────────────────────────────────────
echo "Step 2/4: Installing Rust toolchain..."
if command -v rustc &> /dev/null; then
    echo "  ✓ Rust already installed: $(rustc --version)"
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable 2>&1 | tail -1
    source "$HOME/.cargo/env"
    echo "  ✓ Rust installed: $(rustc --version)"
fi
echo ""

# ── Step 3: Clone & Build ──────────────────────────────────────────────
echo "Step 3/4: Cloning and building NEURON (release mode)..."
NEURON_DIR="$HOME/neuron-lang"
if [ -d "$NEURON_DIR" ]; then
    echo "  → Updating existing repo..."
    cd "$NEURON_DIR"
    git pull --quiet
else
    git clone --depth 1 https://github.com/neuronlabs-ai/neuron-lang.git "$NEURON_DIR" 2>&1 | tail -1
    cd "$NEURON_DIR"
fi

# Build in release mode with native CPU instructions
RUSTFLAGS="-C target-cpu=native" cargo build -p neuron-gpu-bench --release 2>&1 | tail -5
echo "  ✓ Build complete"
echo ""

# ── Step 4: Run Benchmark ──────────────────────────────────────────────
echo "Step 4/4: Running GPU benchmark..."
echo ""
RUSTFLAGS="-C target-cpu=native" cargo run -p neuron-gpu-bench --release
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Benchmark complete! Share the table above for performance review."
echo "═══════════════════════════════════════════════════════════════════"
