# Running NEURON GPU Backend on Google Colab

Google Colab provides free access to NVIDIA GPUs (like Tesla T4). Since the NEURON runtime dynamically loads the CUDA driver and NVRTC libraries using `libloading`, you can easily compile and execute real CUDA JIT compilation loops in a free Colab notebook!

Follow these steps to set up and run the NEURON GPU tests:

---

## Step 1: Open a GPU-enabled Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **New Notebook**.
3. In the top menu, go to **Runtime > Change runtime type**.
4. Under **Hardware accelerator**, select **T4 GPU** (or any available GPU) and click **Save**.

---

## Step 2: Install Rust and Cargo
In the first code cell of your Colab notebook, copy-paste and run this command to install the Rust compiler:

```bash
# Install Rust
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Configure environment path
import os
os.environ["PATH"] += os.pathsep + "/root/.cargo/bin"
```

*Confirm the installation was successful by running:*
```bash
!rustc --version
```

---

## Step 3: Clone the Repository and Compile NEURON
In the next code cell, clone the repository and navigate into it:

```bash
# Clone NEURON repo
!git clone https://github.com/neuronlabs-ai/neuron-lang.git

# Move to repo directory
%cd neuron-lang
```

---

## Step 4: Run the Real GPU Compilation Tests
Since Google Colab pre-installs the CUDA Toolkit, drivers, and `libnvrtc.so` in `/usr/lib64-nvidia/`, we must expose this path to the dynamic linker.

Run the cargo tests using the real GPU back-end:

```bash
# Expose Colab CUDA path and run the GPU tests
!LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH cargo test --test test_gpu
```

### What happens under the hood?
1. The type checker compiles the `.nr` source code down to NEURON Control Flow Graph (CFG) basic blocks.
2. The compiler's `cuda_codegen.rs` module automatically groups and fuses contiguous element-wise math nodes into a custom CUDA C++ kernel string.
3. The VM runtime dynamically loads `/usr/lib64-nvidia/libnvrtc.so.12` and compiles the fused kernel string into Parallel Thread Execution (PTX) assembly in real-time.
4. The VM launches the compiled PTX directly on the GPU using the CUDA driver API (`cuLaunchKernel`).
5. Outputs reside directly in VRAM for zero-copy chaining, synchronized back to host memory on CPU read, and verified against expected values.
