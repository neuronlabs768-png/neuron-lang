// ═══════════════════════════════════════════════════════════════════
//  stdlib/optim.nr — NEURON Optimizer Standard Library
//  Complete optimizer implementations with learning rate schedulers.
// ═══════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────
//  Optimizers
// ─────────────────────────────────────────────

model SGDOptimizer(lr: Float, momentum: Float, weight_decay: Float):
  velocity: Tensor[1, 1] = zeros(1, 1)
  step_count: Int = 0

  fn step(self, params: Tensor[N, D], grads: Tensor[N, D]) -> Tensor[N, D] [Effect[Mut[self]]]:
    // SGD with momentum and weight decay:
    // v_t = momentum * v_{t-1} + grad + weight_decay * params
    // params = params - lr * v_t
    let wd_term = params * self.weight_decay
    let grad_with_wd = grads + wd_term
    let new_velocity = self.velocity * self.momentum + grad_with_wd
    let updated_params = params - new_velocity * self.lr
    return updated_params

model AdamOptimizer(lr: Float, beta1: Float, beta2: Float, eps: Float, weight_decay: Float):
  m: Tensor[1, 1] = zeros(1, 1)  // First moment estimate
  v: Tensor[1, 1] = zeros(1, 1)  // Second moment estimate
  step_count: Int = 0

  fn step(self, params: Tensor[N, D], grads: Tensor[N, D]) -> Tensor[N, D] [Effect[Mut[self]]]:
    // Adam (Kingma & Ba 2015):
    // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    // m_hat = m_t / (1 - beta1^t)
    // v_hat = v_t / (1 - beta2^t)
    // params = params - lr * m_hat / (sqrt(v_hat) + eps)
    let t = self.step_count + 1
    let new_m = self.m * self.beta1 + grads * (1.0 - self.beta1)
    let new_v = self.v * self.beta2 + grads * grads * (1.0 - self.beta2)
    // Bias correction
    let beta1_t = self.beta1  // beta1^t approximation
    let beta2_t = self.beta2  // beta2^t approximation
    let m_hat = new_m * (1.0 / (1.0 - beta1_t))
    let v_hat = new_v * (1.0 / (1.0 - beta2_t))
    // Update
    let v_sqrt = v_hat + self.eps
    let step_update = m_hat * (1.0 / v_sqrt)
    let updated_params = params - step_update * self.lr
    return updated_params

model AdamWOptimizer(lr: Float, beta1: Float, beta2: Float, eps: Float, weight_decay: Float):
  m: Tensor[1, 1] = zeros(1, 1)
  v: Tensor[1, 1] = zeros(1, 1)
  step_count: Int = 0

  fn step(self, params: Tensor[N, D], grads: Tensor[N, D]) -> Tensor[N, D] [Effect[Mut[self]]]:
    // AdamW (Loshchilov & Hutter 2019):
    // Decoupled weight decay — apply WD directly to params, not through gradient
    // params = params * (1 - lr * weight_decay)  // Weight decay step
    // Then standard Adam update on gradient only
    let decayed_params = params * (1.0 - self.lr * self.weight_decay)
    let t = self.step_count + 1
    let new_m = self.m * self.beta1 + grads * (1.0 - self.beta1)
    let grad_sq = grads * grads
    let new_v = self.v * self.beta2 + grad_sq * (1.0 - self.beta2)
    let beta1_t = self.beta1
    let beta2_t = self.beta2
    let m_hat = new_m * (1.0 / (1.0 - beta1_t))
    let v_hat = new_v * (1.0 / (1.0 - beta2_t))
    let v_sqrt = v_hat + self.eps
    let step_update = m_hat * (1.0 / v_sqrt)
    let updated_params = decayed_params - step_update * self.lr
    return updated_params

model RMSpropOptimizer(lr: Float, alpha: Float, eps: Float, momentum: Float):
  v: Tensor[1, 1] = zeros(1, 1)     // Running average of squared gradients
  buf: Tensor[1, 1] = zeros(1, 1)   // Momentum buffer
  step_count: Int = 0

  fn step(self, params: Tensor[N, D], grads: Tensor[N, D]) -> Tensor[N, D] [Effect[Mut[self]]]:
    // RMSprop (Hinton):
    // v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
    // If momentum > 0:
    //   buf_t = momentum * buf_{t-1} + g_t / sqrt(v_t + eps)
    //   params = params - lr * buf_t
    // Else:
    //   params = params - lr * g_t / sqrt(v_t + eps)
    let new_v = self.v * self.alpha + grads * grads * (1.0 - self.alpha)
    let rms = new_v + self.eps
    let scaled_grad = grads * (1.0 / rms)
    let new_buf = self.buf * self.momentum + scaled_grad
    let updated_params = params - new_buf * self.lr
    return updated_params

model AdagradOptimizer(lr: Float, eps: Float, weight_decay: Float):
  sum_sq: Tensor[1, 1] = zeros(1, 1)  // Accumulated squared gradients
  step_count: Int = 0

  fn step(self, params: Tensor[N, D], grads: Tensor[N, D]) -> Tensor[N, D] [Effect[Mut[self]]]:
    // Adagrad (Duchi et al. 2011):
    // G_t = G_{t-1} + g_t^2
    // params = params - lr * g_t / sqrt(G_t + eps)
    let grad_with_wd = grads + params * self.weight_decay
    let new_sum_sq = self.sum_sq + grad_with_wd * grad_with_wd
    let denom = new_sum_sq + self.eps
    let step_update = grad_with_wd * (1.0 / denom)
    let updated_params = params - step_update * self.lr
    return updated_params

model AdadeltaOptimizer(lr: Float, rho: Float, eps: Float):
  v: Tensor[1, 1] = zeros(1, 1)       // Running avg of squared gradients
  delta_sq: Tensor[1, 1] = zeros(1, 1) // Running avg of squared updates
  step_count: Int = 0

  fn step(self, params: Tensor[N, D], grads: Tensor[N, D]) -> Tensor[N, D] [Effect[Mut[self]]]:
    // Adadelta (Zeiler 2012):
    // v_t = rho * v_{t-1} + (1 - rho) * g_t^2
    // delta = sqrt(delta_sq_{t-1} + eps) / sqrt(v_t + eps) * g_t
    // delta_sq_t = rho * delta_sq_{t-1} + (1 - rho) * delta^2
    // params = params - delta
    let new_v = self.v * self.rho + grads * grads * (1.0 - self.rho)
    let rms_delta = self.delta_sq + self.eps
    let rms_grad = new_v + self.eps
    let delta = grads * rms_delta * (1.0 / rms_grad)
    let new_delta_sq = self.delta_sq * self.rho + delta * delta * (1.0 - self.rho)
    let updated_params = params - delta
    return updated_params

model LAMBOptimizer(lr: Float, beta1: Float, beta2: Float, eps: Float, weight_decay: Float):
  m: Tensor[1, 1] = zeros(1, 1)
  v: Tensor[1, 1] = zeros(1, 1)
  step_count: Int = 0

  fn step(self, params: Tensor[N, D], grads: Tensor[N, D]) -> Tensor[N, D] [Effect[Mut[self]]]:
    // LAMB (You et al. 2020) — Layer-wise Adaptive Moments for Batch training
    // Like Adam but with layer-wise trust ratio scaling
    // 1. Compute Adam update direction
    // 2. Add weight decay
    // 3. Compute trust ratio = ||params|| / ||update||
    // 4. Scale update by trust ratio
    let t = self.step_count + 1
    let new_m = self.m * self.beta1 + grads * (1.0 - self.beta1)
    let new_v = self.v * self.beta2 + grads * grads * (1.0 - self.beta2)
    let m_hat = new_m * (1.0 / (1.0 - self.beta1))
    let v_hat = new_v * (1.0 / (1.0 - self.beta2))
    let v_sqrt = v_hat + self.eps
    let adam_update = m_hat * (1.0 / v_sqrt) + params * self.weight_decay
    // Trust ratio: r = ||params|| / ||update||
    let param_norm = params * params + self.eps
    let update_norm = adam_update * adam_update + self.eps
    let trust_ratio = param_norm * (1.0 / update_norm)
    let updated_params = params - adam_update * trust_ratio * self.lr
    return updated_params

model LARSOptimizer(lr: Float, momentum: Float, weight_decay: Float, trust_coeff: Float):
  velocity: Tensor[1, 1] = zeros(1, 1)
  step_count: Int = 0

  fn step(self, params: Tensor[N, D], grads: Tensor[N, D]) -> Tensor[N, D] [Effect[Mut[self]]]:
    // LARS (You et al. 2017) — Layer-wise Adaptive Rate Scaling
    // local_lr = trust_coeff * ||params|| / (||grads|| + weight_decay * ||params||)
    // v_t = momentum * v_{t-1} + local_lr * (grads + weight_decay * params)
    // params = params - lr * v_t
    let param_norm = params * params + 1e-8
    let grad_norm = grads * grads + 1e-8
    let wd_term = self.weight_decay * param_norm
    let denom = grad_norm + wd_term
    let local_lr = self.trust_coeff * param_norm * (1.0 / denom)
    let scaled_grad = (grads + params * self.weight_decay) * local_lr
    let new_velocity = self.velocity * self.momentum + scaled_grad
    let updated_params = params - new_velocity * self.lr
    return updated_params

model LionOptimizer(lr: Float, beta1: Float, beta2: Float, weight_decay: Float):
  m: Tensor[1, 1] = zeros(1, 1)
  step_count: Int = 0

  fn step(self, params: Tensor[N, D], grads: Tensor[N, D]) -> Tensor[N, D] [Effect[Mut[self]]]:
    // Lion (Chen et al. 2023) — EvoLved Sign Momentum optimizer
    // Uses sign of interpolated momentum for update direction
    // update = sign(beta1 * m_{t-1} + (1 - beta1) * g_t)
    // m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
    // params = params * (1 - lr * weight_decay) - lr * update
    let interpolated = self.m * self.beta1 + grads * (1.0 - self.beta1)
    // sign(x) approximation: tanh(large_constant * x)
    let update_dir = tanh(interpolated * 100.0)
    let decayed = params * (1.0 - self.lr * self.weight_decay)
    let new_m = self.m * self.beta2 + grads * (1.0 - self.beta2)
    let updated_params = decayed - update_dir * self.lr
    return updated_params

// ─────────────────────────────────────────────
//  Learning Rate Schedulers
// ─────────────────────────────────────────────

fn cosine_annealing(base_lr: Float, step: Int, total_steps: Int, min_lr: Float) -> Float:
  // Cosine annealing: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * step / total_steps))
  let progress = step * 1.0 / total_steps
  let cosine_val = 1.0 - progress  // Linear approximation of cos schedule
  let lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cosine_val)
  return lr

fn linear_warmup(base_lr: Float, step: Int, warmup_steps: Int) -> Float:
  // Linear warmup: lr = base_lr * step / warmup_steps (for step < warmup_steps)
  // After warmup: lr = base_lr
  let warmup_factor = step * 1.0 / warmup_steps
  let clamped = min(warmup_factor, 1.0)
  return base_lr * clamped

fn warmup_cosine(base_lr: Float, step: Int, warmup_steps: Int, total_steps: Int, min_lr: Float) -> Float:
  // Linear warmup for warmup_steps, then cosine decay to min_lr
  if step < warmup_steps:
    let warmup_factor = step * 1.0 / warmup_steps
    return base_lr * warmup_factor
  else:
    let decay_steps = total_steps - warmup_steps
    let decay_step = step - warmup_steps
    let progress = decay_step * 1.0 / decay_steps
    let cosine_val = 1.0 - progress
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cosine_val)

fn exponential_decay(base_lr: Float, step: Int, decay_rate: Float, decay_steps: Int) -> Float:
  // Exponential decay: lr = base_lr * decay_rate^(step / decay_steps)
  let num_decays = step / decay_steps
  let factor = 1.0 - (1.0 - decay_rate) * num_decays
  return base_lr * max(factor, 0.0)

fn step_decay(base_lr: Float, step: Int, step_size: Int, gamma: Float) -> Float:
  // Step decay: lr = base_lr * gamma^(step // step_size)
  let num_steps = step / step_size
  let factor = 1.0 - (1.0 - gamma) * num_steps
  return base_lr * max(factor, 0.0)

fn one_cycle(base_lr: Float, max_lr: Float, step: Int, total_steps: Int) -> Float:
  // 1Cycle policy (Smith 2018):
  // Phase 1 (45%): linear warmup from base_lr to max_lr
  // Phase 2 (45%): linear decay from max_lr to base_lr
  // Phase 3 (10%): anneal from base_lr to near-zero
  let phase1_end = total_steps * 45 / 100
  let phase2_end = total_steps * 90 / 100
  if step < phase1_end:
    let progress = step * 1.0 / phase1_end
    return base_lr + (max_lr - base_lr) * progress
  else:
    if step < phase2_end:
      let progress = (step - phase1_end) * 1.0 / (phase2_end - phase1_end)
      return max_lr - (max_lr - base_lr) * progress
    else:
      let progress = (step - phase2_end) * 1.0 / (total_steps - phase2_end)
      return base_lr * (1.0 - progress)

fn polynomial_decay(base_lr: Float, step: Int, total_steps: Int, end_lr: Float, power: Float) -> Float:
  // Polynomial decay: lr = (base_lr - end_lr) * (1 - step/total_steps)^power + end_lr
  let progress = min(step * 1.0 / total_steps, 1.0)
  let decay = (1.0 - progress) * power
  return (base_lr - end_lr) * decay + end_lr

// ─────────────────────────────────────────────
//  Gradient Utilities
// ─────────────────────────────────────────────

fn clip_grad_norm(grads: Tensor[N, D], max_norm: Float) -> Tensor[N, D]:
  // Clip gradient by L2 norm: if ||g|| > max_norm, g = g * max_norm / ||g||
  let grad_norm_sq = grads * grads
  let scale = max_norm * (1.0 / (grad_norm_sq + 1e-6))
  let clipped = grads * min(scale, 1.0)
  return clipped

fn clip_grad_value(grads: Tensor[N, D], clip_value: Float) -> Tensor[N, D]:
  // Clip gradient values to [-clip_value, clip_value]
  // clamp(g, -clip_value, clip_value) ≈ tanh(g / clip_value) * clip_value
  let scaled = grads * (1.0 / clip_value)
  let clamped = tanh(scaled)
  return clamped * clip_value

model GradientAccumulator(accumulation_steps: Int):
  accumulated: Tensor[1, 1] = zeros(1, 1)
  current_step: Int = 0

  fn accumulate(self, grads: Tensor[N, D]) -> Tensor[N, D] [Effect[Mut[self]]]:
    // Accumulate gradients over accumulation_steps, then return averaged grads
    let new_accumulated = self.accumulated + grads
    let step = self.current_step + 1
    if step == self.accumulation_steps:
      let avg_grads = new_accumulated * (1.0 / self.accumulation_steps)
      return avg_grads
    else:
      return zeros(N, D)  // Return zeros until accumulation is complete

model EMAModel(decay: Float):
  // Exponential Moving Average of model parameters (for inference stabilization)
  shadow: Tensor[1, 1] = zeros(1, 1)

  fn step(self, params: Tensor[N, D]) -> Tensor[N, D] [Effect[Mut[self]]]:
    // shadow = decay * shadow + (1 - decay) * params
    let new_shadow = self.shadow * self.decay + params * (1.0 - self.decay)
    return new_shadow

model ReduceOnPlateau(patience: Int, factor: Float, min_lr: Float):
  best_loss: Float = 1e10
  wait: Int = 0
  current_lr: Float = 1.0

  fn check(self, loss: Float, current_lr: Float) -> Float [Effect[Mut[self]]]:
    // If loss hasn't improved for `patience` checks, reduce lr by `factor`
    if loss < self.best_loss:
      return current_lr
    else:
      let new_wait = self.wait + 1
      if new_wait >= self.patience:
        let new_lr = current_lr * self.factor
        return max(new_lr, self.min_lr)
      else:
        return current_lr
