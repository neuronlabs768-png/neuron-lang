/// NEURON Autograd — gradient tape and backward pass.
///
/// Every operation registers its backward pass on the tape. `backward()` walks
/// the tape in reverse, computing gradients via chain rule.
/// `grad(loss, wrt=param)` returns the gradient tensor.

use crate::tensor::*;
use crate::buffer::Buffer;

/// An entry in the gradient tape.
#[derive(Clone)]
pub struct TapeEntry {
    /// The operation that produced this value.
    pub op: TapeOp,
    /// IDs of input tensors.
    pub inputs: Vec<usize>,
    /// ID of the output tensor.
    pub output: usize,
    /// Shape of the output.
    pub output_shape: Vec<usize>,
}

/// Tape operations with the information needed for backward pass.
#[derive(Clone)]
pub enum TapeOp {
    Add { a_shape: Vec<usize>, b_shape: Vec<usize> },
    Sub { a_shape: Vec<usize>, b_shape: Vec<usize> },
    Mul { a_data: Buffer, a_shape: Vec<usize>, b_data: Buffer, b_shape: Vec<usize> },
    Div { a_data: Buffer, a_shape: Vec<usize>, b_data: Buffer, b_shape: Vec<usize> },
    MatMul { a_data: Buffer, a_shape: Vec<usize>, b_data: Buffer, b_shape: Vec<usize> },
    ReLU { input_data: Buffer },
    Sigmoid { output_data: Buffer },
    Tanh { output_data: Buffer },
    GeLU { input_data: Buffer },
    Softmax { output_data: Buffer, dim: usize },
    Neg,
    Sum { input_shape: Vec<usize>, dim: Option<usize> },
    Mean { input_shape: Vec<usize>, dim: Option<usize> },
    Sqrt { output_data: Buffer },
    Reshape { original_shape: Vec<usize> },
    CrossEntropy { pred_softmax: Buffer, target: Buffer, batch_size: usize },
    MSE { pred: Buffer, target: Buffer, n: usize },
}

/// The gradient tape — records operations for backward pass.
pub struct GradTape {
    entries: Vec<TapeEntry>,
    /// Gradient storage: maps tensor_id → gradient data
    grads: Vec<Option<Buffer>>,
    /// Next tensor id
    next_id: usize,
    /// Parameter IDs that should have their gradients preserved after backward
    pub parameter_ids: std::collections::HashSet<usize>,
}

impl GradTape {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            grads: Vec::new(),
            next_id: 0,
            parameter_ids: std::collections::HashSet::new(),
        }
    }

    /// Allocate a new tensor id.
    pub fn alloc_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        // Ensure grads vec is large enough
        while self.grads.len() <= id {
            self.grads.push(None);
        }
        id
    }

    /// Record an operation on the tape.
    pub fn record(&mut self, op: TapeOp, inputs: Vec<usize>, output: usize, output_shape: Vec<usize>) {
        self.entries.push(TapeEntry { op, inputs, output, output_shape });
    }

    /// Run tracked add.
    pub fn add(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let result = tensor_add(a, b);
        let out_id = self.alloc_id();
        self.record(TapeOp::Add { a_shape: a.shape.clone(), b_shape: b.shape.clone() },
                    vec![a.id, b.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked sub.
    pub fn sub(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let result = tensor_sub(a, b);
        let out_id = self.alloc_id();
        self.record(TapeOp::Sub { a_shape: a.shape.clone(), b_shape: b.shape.clone() },
                    vec![a.id, b.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked mul.
    pub fn mul(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let result = tensor_mul(a, b);
        let out_id = self.alloc_id();
        self.record(TapeOp::Mul {
            a_data: a.data.clone(), a_shape: a.shape.clone(),
            b_data: b.data.clone(), b_shape: b.shape.clone(),
        }, vec![a.id, b.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked div.
    pub fn div(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let result = tensor_div(a, b);
        let out_id = self.alloc_id();
        self.record(TapeOp::Div {
            a_data: a.data.clone(), a_shape: a.shape.clone(),
            b_data: b.data.clone(), b_shape: b.shape.clone(),
        }, vec![a.id, b.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked matmul.
    pub fn matmul(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let result = tensor_matmul(a, b);
        let out_id = self.alloc_id();
        self.record(TapeOp::MatMul {
            a_data: a.data.clone(), a_shape: a.shape.clone(),
            b_data: b.data.clone(), b_shape: b.shape.clone(),
        }, vec![a.id, b.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked relu.
    pub fn relu(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_relu(a);
        let out_id = self.alloc_id();
        self.record(TapeOp::ReLU { input_data: a.data.clone() }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked sigmoid.
    pub fn sigmoid(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_sigmoid(a);
        let out_id = self.alloc_id();
        self.record(TapeOp::Sigmoid { output_data: result.data.clone() }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked tanh.
    pub fn tanh(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_tanh(a);
        let out_id = self.alloc_id();
        self.record(TapeOp::Tanh { output_data: result.data.clone() }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked softmax.
    pub fn softmax(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_softmax(a);
        let out_id = self.alloc_id();
        let dim = a.ndim().saturating_sub(1);
        self.record(TapeOp::Softmax { output_data: result.data.clone(), dim }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked cross-entropy.
    pub fn cross_entropy(&mut self, pred: &Tensor, target: &Tensor) -> Tensor {
        let sm = tensor_softmax(pred);
        let result = tensor_cross_entropy(pred, target);
        let out_id = self.alloc_id();
        let batch = pred.shape[0];
        self.record(TapeOp::CrossEntropy { pred_softmax: sm.data, target: target.data.clone(), batch_size: batch },
                    vec![pred.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked MSE.
    pub fn mse(&mut self, pred: &Tensor, target: &Tensor) -> Tensor {
        let result = tensor_mse(pred, target);
        let out_id = self.alloc_id();
        self.record(TapeOp::MSE { pred: pred.data.clone(), target: target.data.clone(), n: pred.numel() },
                    vec![pred.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked negation.
    pub fn neg(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_neg(a);
        let out_id = self.alloc_id();
        self.record(TapeOp::Neg, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked gelu.
    pub fn gelu(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_gelu(a);
        let out_id = self.alloc_id();
        self.record(TapeOp::GeLU { input_data: a.data.clone() }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked sum.
    pub fn sum(&mut self, a: &Tensor, dim: Option<usize>) -> Tensor {
        let result = a.sum(dim);
        let out_id = self.alloc_id();
        self.record(TapeOp::Sum { input_shape: a.shape.clone(), dim }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked mean.
    pub fn mean(&mut self, a: &Tensor, dim: Option<usize>) -> Tensor {
        let result = a.mean(dim);
        let out_id = self.alloc_id();
        self.record(TapeOp::Mean { input_shape: a.shape.clone(), dim }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked sqrt.
    pub fn sqrt(&mut self, a: &Tensor) -> Tensor {
        let result = a.map(|x| x.sqrt());
        let out_id = self.alloc_id();
        self.record(TapeOp::Sqrt { output_data: result.data.clone() }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Backward pass — compute gradients for all tensors on the tape.
    /// Starts from `loss_id` with gradient 1.0.
    pub fn backward(&mut self, loss_id: usize) {
        // Clear previous gradients
        for g in self.grads.iter_mut() {
            *g = None;
        }
        // Initialize loss gradient to 1.0
        while self.grads.len() <= loss_id {
            self.grads.push(None);
        }

        // Find the tape entry for the loss
        let loss_shape = self.entries.iter().rev()
            .find(|e| e.output == loss_id)
            .map(|e| e.output_shape.clone())
            .unwrap_or(vec![1]);
        let loss_n: usize = loss_shape.iter().product::<usize>().max(1);
        let mut loss_grad = Buffer::new(loss_n);
        for x in loss_grad.iter_mut() { *x = 1.0; }
        self.grads[loss_id] = Some(loss_grad);

        // Walk tape in reverse
        while let Some(entry) = self.entries.pop() {
            let out_grad = match self.grads.get_mut(entry.output) {
                Some(g_opt) => {
                    if self.parameter_ids.contains(&entry.output) {
                        match g_opt.as_ref() {
                            Some(g) => g.clone(),
                            None => continue,
                        }
                    } else {
                        match g_opt.take() {
                            Some(g) => g,
                            None => continue,
                        }
                    }
                }
                None => continue, // No gradient flowing to this output
            };

            match &entry.op {
                TapeOp::Add { a_shape, b_shape } => {
                    let grad_a = reduce_grad(&out_grad, &entry.output_shape, a_shape);
                    let grad_b = reduce_grad(&out_grad, &entry.output_shape, b_shape);
                    self.accumulate_grad(entry.inputs[0], grad_a);
                    self.accumulate_grad(entry.inputs[1], grad_b);
                }
                TapeOp::Sub { a_shape, b_shape } => {
                    let grad_a = reduce_grad(&out_grad, &entry.output_shape, a_shape);
                    let mut neg = Buffer::new(out_grad.len());
                    for (n, &g) in neg.iter_mut().zip(out_grad.iter()) { *n = -g; }
                    let grad_b = reduce_grad(&neg, &entry.output_shape, b_shape);
                    self.accumulate_grad(entry.inputs[0], grad_a);
                    self.accumulate_grad(entry.inputs[1], grad_b);
                }
                TapeOp::Mul { a_data, a_shape, b_data, b_shape } => {
                    let a_tensor = Tensor::new(a_data.clone(), a_shape.clone());
                    let b_tensor = Tensor::new(b_data.clone(), b_shape.clone());
                    let a_bc = a_tensor.broadcast_to(&entry.output_shape);
                    let b_bc = b_tensor.broadcast_to(&entry.output_shape);
                    
                    let mut raw_grad_a = Buffer::new(out_grad.len());
                    for (g_a, (&g, &b)) in raw_grad_a.iter_mut().zip(out_grad.iter().zip(b_bc.data.iter())) {
                        *g_a = g * b;
                    }
                    let mut raw_grad_b = Buffer::new(out_grad.len());
                    for (g_b, (&g, &a)) in raw_grad_b.iter_mut().zip(out_grad.iter().zip(a_bc.data.iter())) {
                        *g_b = g * a;
                    }
                    
                    let grad_a = reduce_grad(&raw_grad_a, &entry.output_shape, a_shape);
                    let grad_b = reduce_grad(&raw_grad_b, &entry.output_shape, b_shape);
                    self.accumulate_grad(entry.inputs[0], grad_a);
                    self.accumulate_grad(entry.inputs[1], grad_b);
                }
                TapeOp::MatMul { a_data, a_shape, b_data, b_shape } => {
                    let a = Tensor::new(a_data.clone(), a_shape.clone());
                    let b = Tensor::new(b_data.clone(), b_shape.clone());
                    let grad_out = Tensor::new(out_grad, entry.output_shape.clone());

                    let b_t = b.transpose(b.ndim() - 2, b.ndim() - 1);
                    let grad_a = tensor_matmul(&grad_out, &b_t);
                    self.accumulate_grad(entry.inputs[0], grad_a.data);

                    let a_t = a.transpose(a.ndim() - 2, a.ndim() - 1);
                    let grad_b = tensor_matmul(&a_t, &grad_out);
                    self.accumulate_grad(entry.inputs[1], grad_b.data);
                }
                TapeOp::ReLU { input_data } => {
                    let mut grad = Buffer::new(out_grad.len());
                    for (g_in, (&g, &x)) in grad.iter_mut().zip(out_grad.iter().zip(input_data.iter())) {
                        *g_in = if x > 0.0 { g } else { 0.0 };
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Sigmoid { output_data } => {
                    let mut grad = Buffer::new(out_grad.len());
                    for (g_in, (&g, &s)) in grad.iter_mut().zip(out_grad.iter().zip(output_data.iter())) {
                        *g_in = g * s * (1.0 - s);
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Tanh { output_data } => {
                    let mut grad = Buffer::new(out_grad.len());
                    for (g_in, (&g, &t)) in grad.iter_mut().zip(out_grad.iter().zip(output_data.iter())) {
                        *g_in = g * (1.0 - t * t);
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::GeLU { input_data } => {
                    let mut grad = Buffer::new(out_grad.len());
                    for (g_in, (&g, &x)) in grad.iter_mut().zip(out_grad.iter().zip(input_data.iter())) {
                        let cdf = 0.5 * (1.0 + erf_approx(x / 2.0f64.sqrt()));
                        let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
                        *g_in = g * (cdf + x * pdf);
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Softmax { output_data, dim } => {
                    let last_dim = entry.output_shape[*dim];
                    let outer = output_data.len() / last_dim;
                    let mut grad_input = Buffer::new(output_data.len());
                    for i in 0..outer {
                        let start = i * last_dim;
                        for j in 0..last_dim {
                            for k in 0..last_dim {
                                let s_j = output_data[start + j];
                                let s_k = output_data[start + k];
                                let kronecker = if j == k { 1.0 } else { 0.0 };
                                grad_input[start + j] += out_grad[start + k] * s_k * (kronecker - s_j);
                            }
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad_input);
                }
                TapeOp::Neg => {
                    let mut grad = Buffer::new(out_grad.len());
                    for (g_in, &g) in grad.iter_mut().zip(out_grad.iter()) {
                        *g_in = -g;
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Sum { input_shape, dim } => {
                    let mut grad = Buffer::new(input_shape.iter().product::<usize>().max(1));
                    match dim {
                        None => {
                            let val = out_grad[0];
                            for g in grad.iter_mut() {
                                *g = val;
                            }
                        }
                        Some(d) => {
                            let mut out_shape = input_shape.clone();
                            out_shape[*d] = 1;
                            let mut out_strides = vec![1; out_shape.len()];
                            for j in (0..out_shape.len().saturating_sub(1)).rev() {
                                out_strides[j] = out_strides[j + 1] * out_shape[j + 1];
                            }
                            
                            let mut in_strides = vec![1; input_shape.len()];
                            for j in (0..input_shape.len().saturating_sub(1)).rev() {
                                in_strides[j] = in_strides[j + 1] * input_shape[j + 1];
                            }

                            for i in 0..grad.len() {
                                let mut coords = vec![0; input_shape.len()];
                                let mut temp = i;
                                for j in 0..input_shape.len() {
                                    coords[j] = temp / in_strides[j];
                                    temp %= in_strides[j];
                                }
                                let mut out_coords = coords.clone();
                                out_coords[*d] = 0;
                                let out_idx: usize = out_coords.iter().zip(out_strides.iter()).map(|(c, s)| c * s).sum();
                                grad[i] = out_grad[out_idx];
                            }
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Mean { input_shape, dim } => {
                    let mut grad = Buffer::new(input_shape.iter().product::<usize>().max(1));
                    match dim {
                        None => {
                            let n = input_shape.iter().product::<usize>() as f64;
                            let scale = if n > 0.0 { 1.0 / n } else { 0.0 };
                            let val = out_grad[0] * scale;
                            for g in grad.iter_mut() {
                                *g = val;
                            }
                        }
                        Some(d) => {
                            let count = input_shape[*d] as f64;
                            let scale = if count > 0.0 { 1.0 / count } else { 0.0 };
                            
                            let mut out_shape = input_shape.clone();
                            out_shape[*d] = 1;
                            let mut out_strides = vec![1; out_shape.len()];
                            for j in (0..out_shape.len().saturating_sub(1)).rev() {
                                out_strides[j] = out_strides[j + 1] * out_shape[j + 1];
                            }
                            
                            let mut in_strides = vec![1; input_shape.len()];
                            for j in (0..input_shape.len().saturating_sub(1)).rev() {
                                in_strides[j] = in_strides[j + 1] * input_shape[j + 1];
                            }

                            for i in 0..grad.len() {
                                let mut coords = vec![0; input_shape.len()];
                                let mut temp = i;
                                for j in 0..input_shape.len() {
                                    coords[j] = temp / in_strides[j];
                                    temp %= in_strides[j];
                                }
                                let mut out_coords = coords.clone();
                                out_coords[*d] = 0;
                                let out_idx: usize = out_coords.iter().zip(out_strides.iter()).map(|(c, s)| c * s).sum();
                                grad[i] = out_grad[out_idx] * scale;
                            }
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Sqrt { output_data } => {
                    let mut grad = Buffer::new(out_grad.len());
                    for (g_in, (&g, &y)) in grad.iter_mut().zip(out_grad.iter().zip(output_data.iter())) {
                        *g_in = if y > 1e-10 { g * 0.5 / y } else { 0.0 };
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Div { a_data, a_shape, b_data, b_shape } => {
                    let a_tensor = Tensor::new(a_data.clone(), a_shape.clone());
                    let b_tensor = Tensor::new(b_data.clone(), b_shape.clone());
                    let a_bc = a_tensor.broadcast_to(&entry.output_shape);
                    let b_bc = b_tensor.broadcast_to(&entry.output_shape);
                    
                    let mut raw_grad_a = Buffer::new(out_grad.len());
                    for (g_a, (&g, &b)) in raw_grad_a.iter_mut().zip(out_grad.iter().zip(b_bc.data.iter())) {
                        *g_a = if b.abs() > 1e-15 { g / b } else { 0.0 };
                    }
                    let mut raw_grad_b = Buffer::new(out_grad.len());
                    for (g_b, ((&g, &a), &b)) in raw_grad_b.iter_mut().zip(out_grad.iter().zip(a_bc.data.iter()).zip(b_bc.data.iter())) {
                        *g_b = if b.abs() > 1e-15 { -g * a / (b * b) } else { 0.0 };
                    }
                    
                    let grad_a = reduce_grad(&raw_grad_a, &entry.output_shape, a_shape);
                    let grad_b = reduce_grad(&raw_grad_b, &entry.output_shape, b_shape);
                    self.accumulate_grad(entry.inputs[0], grad_a);
                    self.accumulate_grad(entry.inputs[1], grad_b);
                }
                TapeOp::CrossEntropy { pred_softmax, target, batch_size } => {
                    let mut grad = Buffer::new(pred_softmax.len());
                    let scale = *batch_size as f64;
                    let out_g = out_grad[0];
                    for (g_in, (&s, &t)) in grad.iter_mut().zip(pred_softmax.iter().zip(target.iter())) {
                        *g_in = out_g * (s - t) / scale;
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::MSE { pred, target, n } => {
                    let mut grad = Buffer::new(pred.len());
                    let scale = *n as f64;
                    let out_g = out_grad[0];
                    for (g_in, (&p, &t)) in grad.iter_mut().zip(pred.iter().zip(target.iter())) {
                        *g_in = out_g * 2.0 * (p - t) / scale;
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Reshape { .. } => {
                    self.accumulate_grad(entry.inputs[0], out_grad);
                }
            }
        }
    }

    /// Get gradient for a tensor.
    pub fn get_grad(&self, tensor_id: usize) -> Option<&Buffer> {
        self.grads.get(tensor_id).and_then(|g| g.as_ref())
    }

    /// Detach a tensor from the tape to sever its gradients.
    pub fn detach(&mut self, tensor_id: usize) {
        if tensor_id < self.grads.len() {
            self.grads[tensor_id] = None;
        }
        for entry in self.entries.iter_mut() {
            if entry.output == tensor_id {
                entry.inputs.clear();
            }
        }
    }

    /// Get a checkpoint (the current number of tape entries).
    pub fn checkpoint(&self) -> usize {
        self.entries.len()
    }

    /// Truncate the tape to a previous checkpoint, releasing cached buffers.
    pub fn truncate(&mut self, checkpoint: usize) {
        if checkpoint < self.entries.len() {
            self.entries.truncate(checkpoint);
        }
    }

    /// Reset the tape.
    pub fn reset(&mut self) {
        self.entries.clear();
        for g in self.grads.iter_mut() { *g = None; }
        self.parameter_ids.clear();
    }

    /// Return the number of entries in the tape.
    pub fn tape_len(&self) -> usize {
        self.entries.len()
    }

    /// Return the ID of the last output tensor registered on the tape.
    pub fn last_output_id(&self) -> Option<usize> {
        self.entries.last().map(|e| e.output)
    }

    /// Zero all gradients without clearing the tape.
    pub fn zero_grad(&mut self) {
        for g in self.grads.iter_mut() { *g = None; }
    }

    fn accumulate_grad(&mut self, id: usize, grad: Buffer) {
        while self.grads.len() <= id {
            self.grads.push(None);
        }
        if let Some(ref mut existing) = self.grads[id] {
            if existing.len() == grad.len() {
                if existing.len() > 65536 {
                    use rayon::prelude::*;
                    existing.par_iter_mut().zip(grad.par_iter()).for_each(|(a, &b)| *a += b);
                } else {
                    for (a, &b) in existing.iter_mut().zip(grad.iter()) { *a += b; }
                }
            } else {
                *existing = grad;
            }
        } else {
            self.grads[id] = Some(grad);
        }
    }

    #[allow(dead_code)]
    fn add_grad(&mut self, id: usize, grad: &[f64]) {
        while self.grads.len() <= id {
            self.grads.push(None);
        }
        if let Some(ref mut existing) = self.grads[id] {
            // Accumulate: handle size mismatch by extending
            if existing.len() == grad.len() {
                if existing.len() > 65536 {
                    use rayon::prelude::*;
                    existing.par_iter_mut().zip(grad.par_iter()).for_each(|(a, b)| *a += b);
                } else {
                    for (a, b) in existing.iter_mut().zip(grad.iter()) { *a += b; }
                }
            } else {
                // Broadcast case — just set
                *existing = Buffer::from_slice(grad);
            }
        } else {
            self.grads[id] = Some(Buffer::from_slice(grad));
        }
    }
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    sign * y
}

// ═══════════════════════════════════════════
//  Optimizers
// ═══════════════════════════════════════════

pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    t: usize,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, m: Vec::new(), v: Vec::new(), t: 0 }
    }

    /// Apply one step of Adam to a list of (param, grad) pairs.
    pub fn step(&mut self, params: &mut [Tensor], grads: &[Vec<f64>]) {
        self.t += 1;
        while self.m.len() < params.len() {
            self.m.push(vec![]);
            self.v.push(vec![]);
        }
        let correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let n = param.numel();
            if self.m[i].is_empty() {
                self.m[i] = vec![0.0; n];
                self.v[i] = vec![0.0; n];
            }
            let m_slice = &mut self.m[i][..n];
            let v_slice = &mut self.v[i][..n];
            let grad_slice = &grad[..n];
            let param_slice = &mut param.data[..n];

            for j in 0..n {
                let g = grad_slice[j];
                m_slice[j] = self.beta1 * m_slice[j] + (1.0 - self.beta1) * g;
                v_slice[j] = self.beta2 * v_slice[j] + (1.0 - self.beta2) * g * g;
                let m_hat = m_slice[j] / correction1;
                let v_hat = v_slice[j] / correction2;
                param_slice[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}

pub struct SGD {
    pub lr: f64,
    pub momentum: f64,
    velocity: Vec<Vec<f64>>,
}

impl SGD {
    pub fn new(lr: f64, momentum: f64) -> Self {
        Self { lr, momentum, velocity: Vec::new() }
    }

    pub fn step(&mut self, params: &mut [Tensor], grads: &[Vec<f64>]) {
        while self.velocity.len() < params.len() {
            self.velocity.push(vec![]);
        }
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let n = param.numel();
            if self.velocity[i].is_empty() {
                self.velocity[i] = vec![0.0; n];
            }
            let vel_slice = &mut self.velocity[i][..n];
            let grad_slice = &grad[..n];
            let param_slice = &mut param.data[..n];

            for j in 0..n {
                vel_slice[j] = self.momentum * vel_slice[j] + grad_slice[j];
                param_slice[j] -= self.lr * vel_slice[j];
            }
        }
    }
}

fn reduce_grad(grad: &Buffer, current_shape: &[usize], target_shape: &[usize]) -> Buffer {
    if current_shape == target_shape {
        return grad.clone();
    }
    let mut current_t = Tensor::new(grad.clone(), current_shape.to_vec());
    let ndim = current_shape.len();
    let target_ndim = target_shape.len();
    let offset = ndim.saturating_sub(target_ndim);
    
    let mut padded_target_shape = vec![1; ndim];
    for i in 0..target_ndim {
        padded_target_shape[i + offset] = target_shape[i];
    }
    
    for i in (0..ndim).rev() {
        if padded_target_shape[i] == 1 && current_t.shape[i] > 1 {
            current_t = current_t.sum(Some(i));
        }
    }
    
    let mut out_data = current_t.data;
    let expected_len = target_shape.iter().product::<usize>();
    if out_data.len() != expected_len {
        let mut new_buf = Buffer::new(expected_len);
        let copy_len = out_data.len().min(expected_len);
        new_buf[..copy_len].copy_from_slice(&out_data[..copy_len]);
        out_data = new_buf;
    }
    out_data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_add() {
        let mut tape = GradTape::new();
        let mut a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        a.id = tape.alloc_id();
        let mut b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        b.id = tape.alloc_id();

        let c = tape.add(&a, &b);
        tape.backward(c.id);

        let grad_a = tape.get_grad(a.id).unwrap();
        let grad_b = tape.get_grad(b.id).unwrap();
        assert_eq!(grad_a, &vec![1.0, 1.0, 1.0]);
        assert_eq!(grad_b, &vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_grad_matmul() {
        let mut tape = GradTape::new();
        let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        a.id = tape.alloc_id();
        let mut b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        b.id = tape.alloc_id();

        let c = tape.matmul(&a, &b);
        // To create a scalar loss, just backward from the matmul result directly
        // (treating it as if dL/dC = ones)
        tape.backward(c.id);

        let grad_a = tape.get_grad(a.id).unwrap();
        assert_eq!(grad_a.len(), 4);
        // dL/dA = dL/dC @ B^T (with dL/dC = ones since output_shape matches)
    }

    #[test]
    fn test_grad_relu() {
        let mut tape = GradTape::new();
        let mut a = Tensor::new(vec![-1.0, 0.5, -0.3, 2.0], vec![4]);
        a.id = tape.alloc_id();

        let b = tape.relu(&a);
        tape.backward(b.id);

        let grad = tape.get_grad(a.id).unwrap();
        assert_eq!(grad[0], 0.0);  // -1.0 → dead
        assert_eq!(grad[1], 1.0);  // 0.5 → alive
        assert_eq!(grad[2], 0.0);  // -0.3 → dead
        assert_eq!(grad[3], 1.0);  // 2.0 → alive
    }

    #[test]
    fn test_adam() {
        let param = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let grad = vec![0.1, 0.2, 0.3];
        let mut adam = Adam::new(0.001);
        let _original = param.data.clone();
        adam.step(&mut [param.clone()], &[grad]);
        // Just verify it doesn't panic; values move
    }
}
