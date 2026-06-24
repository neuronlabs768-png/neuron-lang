/// NEURON Tensor Engine — owns all tensor storage and operations.
///
/// Contiguous f32/f64 buffer with shape, strides. Supports gradient tracking.
/// No PyTorch, no NumPy — pure Rust.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use crate::buffer::Buffer;

/// Global seed counter — ensures every tensor initialization gets a unique seed.
/// Starts from system time hash for cross-run variation, then increments atomically.
static SEED_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Get a unique seed for tensor initialization.
/// Combines a global atomic counter with a time-based base to ensure
/// uniqueness both within a run and across runs.
fn next_seed() -> u64 {
    let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    // Mix counter with a time-based value on first call
    if counter == 0 {
        // Seed the base from system time on first use
        let time_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        SEED_COUNTER.store(time_seed, Ordering::Relaxed);
        return time_seed;
    }
    counter
}

/// Unique tensor identifier for the autograd tape.
pub type TensorId = usize;

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    F32,
    F64,
    I64,
    Bool,
}

/// A NEURON tensor — contiguous memory with shape and optional gradient.
#[derive(Clone)]
pub struct Tensor {
    pub id: TensorId,
    pub data: Buffer,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub requires_grad: bool,
    pub grad: Option<Box<Tensor>>,
    /// Index into the autograd tape (if this tensor was produced by a tracked op)
    pub tape_entry: Option<usize>,
}

impl Tensor {
    /// Create a new tensor from data and shape.
    pub fn new<B: Into<Buffer>>(data: B, shape: Vec<usize>) -> Self {
        let data = data.into();
        let strides = compute_strides(&shape);
        debug_assert_eq!(data.len(), shape.iter().product::<usize>(),
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(), shape, shape.iter().product::<usize>());
        Self {
            id: 0,
            data,
            shape,
            strides,
            requires_grad: false,
            grad: None,
            tape_entry: None,
        }
    }

    /// Create a zeros tensor.
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self::new(Buffer::new(n), shape.to_vec())
    }

    /// Create a ones tensor.
    pub fn ones(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        let mut buf = Buffer::new(n);
        for x in buf.iter_mut() { *x = 1.0; }
        Self::new(buf, shape.to_vec())
    }

    /// Create a tensor filled with a value.
    pub fn full(shape: &[usize], val: f64) -> Self {
        let n: usize = shape.iter().product();
        let mut buf = Buffer::new(n);
        for x in buf.iter_mut() { *x = val; }
        Self::new(buf, shape.to_vec())
    }

    /// Create a scalar tensor.
    pub fn scalar(val: f64) -> Self {
        let mut buf = Buffer::new(1);
        buf[0] = val;
        Self::new(buf, vec![1])
    }

    /// Glorot (Xavier) uniform initialization.
    /// Uses a globally unique seed per call to ensure different tensors
    /// get different initial values.
    pub fn glorot(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        let fan_in = if shape.len() >= 2 { shape[shape.len() - 2] } else { shape[0] };
        let fan_out = if shape.len() >= 2 { shape[shape.len() - 1] } else { shape[0] };
        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
        let mut data = Buffer::new(n);
        // PCG-style LCG with unique seed per tensor allocation
        let mut state: u64 = next_seed();
        for i in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 33) as f64 / (1u64 << 31) as f64; // 0..1
            data[i] = u * 2.0 * limit - limit;
        }
        Self::new(data, shape.to_vec())
    }

    /// Random normal initialization (Box-Muller transform).
    /// Uses a globally unique seed per call.
    pub fn randn(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        let mut data = Buffer::new(n);
        let mut state: u64 = next_seed();
        for i in 0..n {
            // Box-Muller transform: converts uniform samples to normal distribution
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (state >> 33) as f64 / (1u64 << 31) as f64;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (state >> 33) as f64 / (1u64 << 31) as f64;
            let u1 = u1.max(1e-10);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            data[i] = z;
        }
        Self::new(data, shape.to_vec())
    }

    /// Number of elements.
    pub fn numel(&self) -> usize { self.data.len() }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize { self.shape.len() }

    /// Get element at flat index.
    pub fn get_flat(&self, idx: usize) -> f64 { self.data[idx] }

    /// Get element at multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> f64 {
        let flat = indices.iter().zip(self.strides.iter()).map(|(i, s)| i * s).sum::<usize>();
        self.data[flat]
    }

    /// Set element at multi-dimensional index.
    pub fn set(&mut self, indices: &[usize], val: f64) {
        let flat = indices.iter().zip(self.strides.iter()).map(|(i, s)| i * s).sum::<usize>();
        self.data[flat] = val;
    }

    /// Enable gradient tracking.
    pub fn with_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Retrieve raw Unified Memory pointer address.
    pub fn uvm_device_ptr(&self) -> u64 {
        match &self.data.storage {
            crate::buffer::BufferStorage::Uvm { device_ptr, .. } => *device_ptr,
            _ => 0,
        }
    }


    /// Reshape the tensor (view — no data copy if contiguous).
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let new_n: usize = new_shape.iter().product();
        assert_eq!(self.numel(), new_n, "Cannot reshape {:?} to {:?}", self.shape, new_shape);
        Tensor::new(self.data.clone(), new_shape.to_vec())
    }

    /// Transpose (swap two axes).
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        let ndim = self.ndim();
        assert!(dim0 < ndim && dim1 < ndim);
        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);
        
        let n = self.numel();
        let mut new_data = Buffer::new(n);
        
        if ndim == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let src = &self.data;
            let dest = &mut new_data;
            if (dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0) {
                for r in 0..rows {
                    let r_offset = r * cols;
                    for c in 0..cols {
                        dest[c * rows + r] = src[r_offset + c];
                    }
                }
                return Tensor::new(new_data, new_shape);
            }
        }
        
        let mut new_strides = self.strides.clone();
        new_strides.swap(dim0, dim1);
        let mut old_indices = vec![0usize; ndim];
        for flat in 0..n {
            let mut rem = flat;
            for d in 0..ndim {
                old_indices[d] = rem / self.strides[d];
                rem %= self.strides[d];
            }
            let mut new_indices = old_indices.clone();
            new_indices.swap(dim0, dim1);
            let new_flat = new_indices.iter().zip(compute_strides(&new_shape).iter())
                .map(|(i, s)| i * s).sum::<usize>();
            new_data[new_flat] = self.data[flat];
        }
        Tensor::new(new_data, new_shape)
    }

    /// Sum all elements.
    pub fn sum_all(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Mean of all elements.
    pub fn mean_all(&self) -> f64 {
        self.sum_all() / self.numel() as f64
    }

    /// Sum elements globally or along a dimension.
    pub fn sum(&self, dim: Option<usize>) -> Tensor {
        match dim {
            None => {
                let s = self.data.iter().sum::<f64>();
                Tensor::new(vec![s], vec![1])
            }
            Some(d) => {
                assert!(d < self.ndim(), "Dimension out of bounds: {} for ndim {}", d, self.ndim());
                let mut new_shape = self.shape.clone();
                new_shape[d] = 1;
                let mut out_data = vec![0.0; new_shape.iter().product()];
                let n = self.numel();
                for i in 0..n {
                    let mut coords = vec![0; self.ndim()];
                    let mut temp = i;
                    for j in 0..self.ndim() {
                        coords[j] = temp / self.strides[j];
                        temp %= self.strides[j];
                    }
                    let mut out_coords = coords.clone();
                    out_coords[d] = 0;
                    
                    let mut out_strides = vec![1; new_shape.len()];
                    for j in (0..new_shape.len().saturating_sub(1)).rev() {
                        out_strides[j] = out_strides[j + 1] * new_shape[j + 1];
                    }
                    let out_idx: usize = out_coords.iter().zip(out_strides.iter()).map(|(c, s)| c * s).sum();
                    out_data[out_idx] += self.data[i];
                }
                Tensor::new(out_data, new_shape)
            }
        }
    }

    /// Mean of elements globally or along a dimension.
    pub fn mean(&self, dim: Option<usize>) -> Tensor {
        let sum_tensor = self.sum(dim);
        match dim {
            None => {
                let n = self.numel() as f64;
                sum_tensor.map(|x| if n > 0.0 { x / n } else { 0.0 })
            }
            Some(d) => {
                let count = self.shape[d] as f64;
                sum_tensor.map(|x| if count > 0.0 { x / count } else { 0.0 })
            }
        }
    }

    /// Broadcast tensor to a target shape.
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Tensor {
        if self.shape == target_shape {
            return self.clone();
        }
        
        let mut out_data = vec![0.0; target_shape.iter().product()];
        let ndim = target_shape.len();
        let self_ndim = self.ndim();
        
        let mut target_strides = vec![1; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            target_strides[i] = target_strides[i + 1] * target_shape[i + 1];
        }
        
        let mut padded_shape = vec![1; ndim];
        let offset = ndim.saturating_sub(self_ndim);
        for i in 0..self_ndim {
            padded_shape[i + offset] = self.shape[i];
        }
        
        let mut padded_strides = vec![1; ndim];
        let mut acc = 1;
        for j in (0..self_ndim).rev() {
            padded_strides[j + offset] = acc;
            acc *= self.shape[j];
        }
        for j in 0..offset {
            padded_strides[j] = 0;
        }
        
        for i in 0..out_data.len() {
            let mut coords = vec![0; ndim];
            let mut temp = i;
            for j in 0..ndim {
                coords[j] = temp / target_strides[j];
                temp %= target_strides[j];
            }
            
            let mut src_idx = 0;
            for j in 0..ndim {
                let c = if padded_shape[j] == 1 { 0 } else { coords[j] };
                src_idx += c * padded_strides[j];
            }
            
            out_data[i] = self.data[src_idx];
        }
        
        Tensor::new(out_data, target_shape.to_vec())
    }

    /// Element-wise apply.
    pub fn map<F: Fn(f64) -> f64 + Send + Sync>(&self, f: F) -> Tensor {
        let mut data = Buffer::new(self.data.len());
        if self.data.len() > 65536 {
            use rayon::prelude::*;
            data.par_iter_mut().zip(self.data.par_iter()).for_each(|(d, &x)| {
                *d = f(x);
            });
        } else {
            for (d, &x) in data.iter_mut().zip(self.data.iter()) {
                *d = f(x);
            }
        }
        Tensor::new(data, self.shape.clone())
    }

    /// Accumulate gradient.
    pub fn accumulate_grad(&mut self, grad: &Tensor) {
        if let Some(ref mut existing) = self.grad {
            if existing.data.len() > 65536 {
                use rayon::prelude::*;
                existing.data.par_iter_mut().zip(grad.data.par_iter()).for_each(|(a, b)| {
                    *a += b;
                });
            } else {
                for (a, b) in existing.data.iter_mut().zip(grad.data.iter()) {
                    *a += b;
                }
            }
        } else {
            self.grad = Some(Box::new(grad.clone()));
        }
    }

    /// Zero the gradient.
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, data=[", self.shape)?;
        let max_show = 6;
        for (i, &v) in self.data.iter().take(max_show).enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{:.4}", v)?;
        }
        if self.data.len() > max_show { write!(f, ", ...")?; }
        write!(f, "])")
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ndim() == 0 || (self.ndim() == 1 && self.shape[0] == 1) {
            write!(f, "{:.6}", self.data[0])
        } else if self.ndim() == 1 {
            write!(f, "[")?;
            for (i, &v) in self.data.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{:.4}", v)?;
            }
            write!(f, "]")
        } else if self.ndim() == 2 && self.shape[0] == 1 {
            write!(f, "[")?;
            for (i, &v) in self.data.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{:.4}", v)?;
            }
            write!(f, "]")
        } else {
            write!(f, "Tensor{:?}", self.shape)
        }
    }
}

/// Compute C-contiguous strides from shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ═══════════════════════════════════════════
//  Tensor operations (forward only — backward in autograd.rs)
// ═══════════════════════════════════════════

/// Helper to find the broadcasted shape of two inputs.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let ndim = a.len().max(b.len());
    let mut result = vec![0; ndim];
    let offset_a = ndim - a.len();
    let offset_b = ndim - b.len();
    for i in 0..ndim {
        let size_a = if i >= offset_a { a[i - offset_a] } else { 1 };
        let size_b = if i >= offset_b { b[i - offset_b] } else { 1 };
        if size_a == size_b {
            result[i] = size_a;
        } else if size_a == 1 {
            result[i] = size_b;
        } else if size_b == 1 {
            result[i] = size_a;
        } else {
            return Err(format!("Shapes are not compatible for broadcasting: {:?} vs {:?}", a, b));
        }
    }
    Ok(result)
}

/// Element-wise add.
pub fn tensor_add(a: &Tensor, b: &Tensor) -> Tensor {
    let common_shape = broadcast_shapes(&a.shape, &b.shape).expect("Broadcasting failed in add");
    let a_bc = a.broadcast_to(&common_shape);
    let b_bc = b.broadcast_to(&common_shape);
    let mut data = Buffer::new(a_bc.data.len());
    if a_bc.data.len() > 65536 {
        use rayon::prelude::*;
        data.par_iter_mut().zip(a_bc.data.par_iter().zip(b_bc.data.par_iter())).for_each(|(d, (x, y))| {
            *d = x + y;
        });
    } else {
        for (d, (x, y)) in data.iter_mut().zip(a_bc.data.iter().zip(b_bc.data.iter())) {
            *d = x + y;
        }
    }
    Tensor::new(data, common_shape)
}

/// Element-wise subtract.
pub fn tensor_sub(a: &Tensor, b: &Tensor) -> Tensor {
    let common_shape = broadcast_shapes(&a.shape, &b.shape).expect("Broadcasting failed in sub");
    let a_bc = a.broadcast_to(&common_shape);
    let b_bc = b.broadcast_to(&common_shape);
    let mut data = Buffer::new(a_bc.data.len());
    if a_bc.data.len() > 65536 {
        use rayon::prelude::*;
        data.par_iter_mut().zip(a_bc.data.par_iter().zip(b_bc.data.par_iter())).for_each(|(d, (x, y))| {
            *d = x - y;
        });
    } else {
        for (d, (x, y)) in data.iter_mut().zip(a_bc.data.iter().zip(b_bc.data.iter())) {
            *d = x - y;
        }
    }
    Tensor::new(data, common_shape)
}

/// Element-wise multiply.
pub fn tensor_mul(a: &Tensor, b: &Tensor) -> Tensor {
    let common_shape = broadcast_shapes(&a.shape, &b.shape).expect("Broadcasting failed in mul");
    let a_bc = a.broadcast_to(&common_shape);
    let b_bc = b.broadcast_to(&common_shape);
    let mut data = Buffer::new(a_bc.data.len());
    if a_bc.data.len() > 65536 {
        use rayon::prelude::*;
        data.par_iter_mut().zip(a_bc.data.par_iter().zip(b_bc.data.par_iter())).for_each(|(d, (x, y))| {
            *d = x * y;
        });
    } else {
        for (d, (x, y)) in data.iter_mut().zip(a_bc.data.iter().zip(b_bc.data.iter())) {
            *d = x * y;
        }
    }
    Tensor::new(data, common_shape)
}

/// Element-wise divide.
pub fn tensor_div(a: &Tensor, b: &Tensor) -> Tensor {
    let common_shape = broadcast_shapes(&a.shape, &b.shape).expect("Broadcasting failed in div");
    let a_bc = a.broadcast_to(&common_shape);
    let b_bc = b.broadcast_to(&common_shape);
    let mut data = Buffer::new(a_bc.data.len());
    if a_bc.data.len() > 65536 {
        use rayon::prelude::*;
        data.par_iter_mut().zip(a_bc.data.par_iter().zip(b_bc.data.par_iter())).for_each(|(d, (x, y))| {
            *d = x / y;
        });
    } else {
        for (d, (x, y)) in data.iter_mut().zip(a_bc.data.iter().zip(b_bc.data.iter())) {
            *d = x / y;
        }
    }
    Tensor::new(data, common_shape)
}

/// Negate.
pub fn tensor_neg(a: &Tensor) -> Tensor {
    a.map(|x| -x)
}

/// Matrix multiply. Supports batched matmul.
/// A: [..., M, K], B: [..., K, N] → [..., M, N]
pub fn tensor_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.ndim() >= 2 && b.ndim() >= 2, "matmul requires at least 2D tensors");
    let m = a.shape[a.ndim() - 2];
    let k_a = a.shape[a.ndim() - 1];
    let k_b = b.shape[b.ndim() - 2];
    let n = b.shape[b.ndim() - 1];
    assert_eq!(k_a, k_b, "matmul inner dimensions mismatch: {} vs {}", k_a, k_b);

    // Compute batch dimensions
    let batch_dims_a = &a.shape[..a.ndim() - 2];
    let _batch_dims_b = &b.shape[..b.ndim() - 2];
    let batch_size: usize = batch_dims_a.iter().product::<usize>().max(1);

    let mut result_shape = batch_dims_a.to_vec();
    result_shape.push(m);
    result_shape.push(n);
    let mut result = Buffer::new(batch_size * m * n);

    let a_stride = m * k_a;
    let b_stride = k_b * n;
    let c_stride = m * n;

    if batch_size > 1 {
        use rayon::prelude::*;
        result.par_chunks_mut(c_stride).enumerate().for_each(|(batch, c_slice)| {
            let a_ptr = unsafe { a.data.as_ptr().add(batch * a_stride) };
            let b_ptr = unsafe { b.data.as_ptr().add(batch * b_stride) };
            let c_ptr = c_slice.as_mut_ptr();
            unsafe {
                matrixmultiply::dgemm(
                    m, k_a, n,
                    1.0,
                    a_ptr, k_a as isize, 1,
                    b_ptr, n as isize, 1,
                    0.0,
                    c_ptr, n as isize, 1,
                );
            }
        });
    } else {
        let a_ptr = a.data.as_ptr();
        let b_ptr = b.data.as_ptr();
        let c_ptr = result.as_mut_ptr();
        unsafe {
            matrixmultiply::dgemm(
                m, k_a, n,
                1.0,
                a_ptr, k_a as isize, 1,
                b_ptr, n as isize, 1,
                0.0,
                c_ptr, n as isize, 1,
            );
        }
    }

    Tensor::new(result, result_shape)
}

/// ReLU activation: max(0, x)
pub fn tensor_relu(a: &Tensor) -> Tensor {
    a.map(|x| if x > 0.0 { x } else { 0.0 })
}

/// GeLU activation: x * Φ(x)
pub fn tensor_gelu(a: &Tensor) -> Tensor {
    a.map(|x| {
        let cdf = 0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2));
        x * cdf
    })
}

/// Sigmoid: 1 / (1 + exp(-x))
pub fn tensor_sigmoid(a: &Tensor) -> Tensor {
    a.map(|x| 1.0 / (1.0 + (-x).exp()))
}

/// Tanh
pub fn tensor_tanh(a: &Tensor) -> Tensor {
    a.map(|x| x.tanh())
}

/// Softmax along last dimension.
pub fn tensor_softmax(a: &Tensor) -> Tensor {
    let ndim = a.ndim();
    if ndim == 0 { return Tensor::scalar(1.0); }
    let last_dim = a.shape[ndim - 1];
    let outer: usize = a.numel() / last_dim;
    let mut result = a.data.clone();

    for i in 0..outer {
        let start = i * last_dim;
        let end = start + last_dim;
        let slice = &result[start..end];
        let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for j in start..end {
            result[j] = (result[j] - max_val).exp();
            sum += result[j];
        }
        for j in start..end {
            result[j] /= sum;
        }
    }

    Tensor::new(result, a.shape.clone())
}

/// Cross-entropy loss.
pub fn tensor_cross_entropy(pred: &Tensor, target: &Tensor) -> Tensor {
    let sm = tensor_softmax(pred);
    let n = sm.numel();
    let mut loss = 0.0;
    for i in 0..n {
        let p = sm.data[i].max(1e-12);
        loss -= target.data[i] * p.ln();
    }
    loss /= (pred.shape[0]) as f64;
    Tensor::scalar(loss)
}

/// MSE loss.
pub fn tensor_mse(pred: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(pred.shape, target.shape);
    let n = pred.numel() as f64;
    let loss: f64 = pred.data.iter().zip(target.data.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>() / n;
    Tensor::scalar(loss)
}

/// Approximate error function (for GeLU).
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(&[3, 4]);
        assert_eq!(t.numel(), 12);
        assert!(t.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_add() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = tensor_add(&a, &b);
        assert_eq!(c.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matmul() {
        // [2,3] @ [3,2] = [2,2]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let c = tensor_matmul(&a, &b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data[0], 1.0*7.0 + 2.0*9.0 + 3.0*11.0);   // 58
        assert_eq!(c.data[1], 1.0*8.0 + 2.0*10.0 + 3.0*12.0);  // 64
        assert_eq!(c.data[2], 4.0*7.0 + 5.0*9.0 + 6.0*11.0);   // 139
        assert_eq!(c.data[3], 4.0*8.0 + 5.0*10.0 + 6.0*12.0);  // 154
    }

    #[test]
    fn test_relu() {
        let a = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        let b = tensor_relu(&a);
        assert_eq!(b.data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_softmax() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let s = tensor_softmax(&a);
        let sum: f64 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1, got {}", sum);
        assert!(s.data[0] < s.data[1] && s.data[1] < s.data[2]);
    }

    #[test]
    fn test_glorot() {
        let t = Tensor::glorot(&[128, 64]);
        assert_eq!(t.shape, vec![128, 64]);
        let limit = (6.0 / 192.0_f64).sqrt();
        assert!(t.data.iter().all(|&x| x >= -limit && x <= limit));
    }

    #[test]
    fn test_reshape() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = a.reshape(&[3, 2]);
        assert_eq!(b.shape, vec![3, 2]);
        assert_eq!(b.data, a.data);
    }

    #[test]
    fn test_mse() {
        let pred = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let target = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let loss = tensor_mse(&pred, &target);
        assert!((loss.data[0]).abs() < 1e-10, "MSE of identical tensors should be 0");
    }
}
