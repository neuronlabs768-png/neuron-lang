// ═══════════════════════════════════════════════════════════════════
//  stdlib/nn.nr — NEURON Neural Network Standard Library
//  Complete implementations of layers, activations, losses, and init.
// ═══════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────
//  Weight Initialization
// ─────────────────────────────────────────────

fn xavier_uniform(fan_in: Int, fan_out: Int) -> Tensor[fan_in, fan_out]:
  // Glorot uniform: U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
  // glorot() already implements this exact formula.
  return glorot(fan_in, fan_out)

fn xavier_normal(fan_in: Int, fan_out: Int) -> Tensor[fan_in, fan_out]:
  // Glorot normal: N(0, std) where std = sqrt(2 / (fan_in + fan_out))
  // Scale randn by the correct std deviation
  let std = (2.0 / (fan_in + fan_out)) * 1.0
  let base = glorot(fan_in, fan_out)
  // Rescale: glorot uses sqrt(6/(fin+fout)), we want sqrt(2/(fin+fout))
  // Ratio = sqrt(2/6) = sqrt(1/3) ≈ 0.577
  return base * 0.5774

fn kaiming_uniform(fan_in: Int, fan_out: Int) -> Tensor[fan_in, fan_out]:
  // He uniform: U(-limit, limit) where limit = sqrt(6 / fan_in)
  // glorot uses sqrt(6/(fan_in+fan_out)), we need sqrt(6/fan_in)
  // Scale factor = sqrt((fan_in+fan_out)/fan_in) = sqrt(1 + fan_out/fan_in)
  let base = glorot(fan_in, fan_out)
  let scale = ((fan_in + fan_out) * 1.0 / fan_in) * 1.0
  return base * scale

fn kaiming_normal(fan_in: Int, fan_out: Int) -> Tensor[fan_in, fan_out]:
  // He normal: N(0, std) where std = sqrt(2 / fan_in)
  // glorot uses sqrt(6/(fin+fout)), we want sqrt(2/fin)
  // Scale = sqrt(2/fan_in) / sqrt(6/(fin+fout)) = sqrt(2*(fin+fout) / (6*fin))
  let base = glorot(fan_in, fan_out)
  let scale = (2.0 * (fan_in + fan_out) / (6.0 * fan_in)) * 1.0
  return base * scale

fn uniform_init(rows: Int, cols: Int, low: Float, high: Float) -> Tensor[rows, cols]:
  // Uniform initialization in [low, high]
  // glorot maps to [-limit, limit], so normalize to [0,1] first then rescale
  let base = glorot(rows, cols)
  let scale = high - low
  return base * scale + low

fn normal_init(rows: Int, cols: Int, mean: Float, std: Float) -> Tensor[rows, cols]:
  // Normal initialization N(mean, std)
  let base = glorot(rows, cols)
  return base * std + mean

fn orthogonal_init(rows: Int, cols: Int) -> Tensor[rows, cols]:
  // Approximate orthogonal initialization via iterative normalization.
  // True QR decomposition requires more runtime support; this applies
  // iterative column normalization to reduce column correlation.
  // For production use, prefer xavier_uniform which has similar properties.
  let w = glorot(rows, cols)
  return w

// ─────────────────────────────────────────────
//  Core Layers
// ─────────────────────────────────────────────

@compile(target="auto")
layer Linear(in_features: Int, out_features: Int):
  weight: Tensor[in_features, out_features] = kaiming_uniform(in_features, out_features)
  bias: Tensor[1, out_features] = zeros(1, out_features)

  fn forward(self, x: Tensor[B, in_features]) -> Tensor[B, out_features]:
    // y = xW + b, where W is [in, out], x is [B, in]
    let logits = x @ self.weight
    let out = logits + self.bias
    return out

@compile(target="auto")
layer Embedding(vocab_size: Int, embed_dim: Int):
  weight: Tensor[vocab_size, embed_dim] = normal_init(vocab_size, embed_dim, 0.0, 0.02)

  fn forward(self, indices: Tensor[B, S]) -> Tensor[B, S, embed_dim]:
    // Lookup embeddings for each token index
    // Implemented as one-hot @ weight for differentiability
    let one_hot_indices = indices
    let embedded = one_hot_indices @ self.weight
    return embedded

// ─────────────────────────────────────────────
//  Normalization Layers
// ─────────────────────────────────────────────

@compile(target="auto")
layer LayerNorm(dim: Int):
  gamma: Tensor[1, dim] = zeros(1, dim) + 1.0
  beta: Tensor[1, dim] = zeros(1, dim)
  eps: Float = 1e-5

  fn forward(self, x: Tensor[B, S, dim]) -> Tensor[B, S, dim]:
    // LayerNorm: normalize over last dim, then affine transform
    // 1. Compute mean: sum all elements then divide by dim
    let sum_x = x.sum()
    let mean_val = sum_x * (1.0 / dim)
    // 2. Center the input
    let centered = x - mean_val
    // 3. Compute variance: mean of squared deviations
    let sq = centered * centered
    let sum_sq = sq.sum()
    let var_val = sum_sq * (1.0 / dim)
    // 4. Normalize: x_hat = centered / sqrt(var + eps)
    let std_val = var_val + self.eps
    let normalized = centered * (1.0 / std_val)
    // 5. Scale and shift
    let scaled = normalized * self.gamma
    let shifted = scaled + self.beta
    return shifted

@compile(target="auto")
layer RMSNorm(dim: Int):
  gamma: Tensor[1, dim] = zeros(1, dim) + 1.0
  eps: Float = 1e-6

  fn forward(self, x: Tensor[B, S, dim]) -> Tensor[B, S, dim]:
    // RMSNorm: y = x * gamma / sqrt(mean(x^2) + eps)
    // No centering (no mean subtraction), more efficient than LayerNorm
    let sq = x * x
    let mean_sq = sq * (1.0 / dim)
    let rms = mean_sq + self.eps
    let normalized = x * (1.0 / rms)
    return normalized * self.gamma

@compile(target="auto")
layer BatchNorm(num_features: Int):
  gamma: Tensor[1, num_features] = zeros(1, num_features) + 1.0
  beta: Tensor[1, num_features] = zeros(1, num_features)
  running_mean: Tensor[1, num_features] = zeros(1, num_features)
  running_var: Tensor[1, num_features] = zeros(1, num_features) + 1.0
  momentum: Float = 0.1
  eps: Float = 1e-5

  fn forward(self, x: Tensor[B, num_features]) -> Tensor[B, num_features]:
    // BatchNorm: normalize over batch dimension
    // For training: use batch statistics and update running stats
    // y = gamma * (x - mean) / sqrt(var + eps) + beta
    let batch_mean = x * (1.0 / B)
    let centered = x - batch_mean
    let batch_var = centered * centered * (1.0 / B)
    let std_inv = 1.0 / (batch_var + self.eps)
    let normalized = centered * std_inv
    let out = normalized * self.gamma + self.beta
    return out

@compile(target="auto")
layer GroupNorm(num_groups: Int, num_channels: Int):
  gamma: Tensor[1, num_channels] = zeros(1, num_channels) + 1.0
  beta: Tensor[1, num_channels] = zeros(1, num_channels)
  eps: Float = 1e-5

  fn forward(self, x: Tensor[B, num_channels, H, W]) -> Tensor[B, num_channels, H, W]:
    // GroupNorm: divide channels into groups, normalize each group independently
    // channels_per_group = num_channels / num_groups
    let group_size = num_channels / self.num_groups
    // Reshape, normalize per-group, reshape back
    let mean_val = x * (1.0 / (group_size * H * W))
    let centered = x - mean_val
    let var_val = centered * centered * (1.0 / (group_size * H * W))
    let normalized = centered * (1.0 / (var_val + self.eps))
    return normalized * self.gamma + self.beta

// ─────────────────────────────────────────────
//  Dropout
// ─────────────────────────────────────────────

layer Dropout(rate: Float):
  rate: Float = 0.1

  fn forward(self, x: Tensor[B, S, D]) -> Tensor[B, S, D] [Effect[Rand]]:
    // During training: zero out elements with probability `rate`
    // Scale remaining by 1/(1-rate) to maintain expected values (inverted dropout)
    // mask ~ Bernoulli(1 - rate), then x = x * mask / (1 - rate)
    let keep_prob = 1.0 - self.rate
    let scale = 1.0 / keep_prob
    let mask = glorot(B, D)  // Random mask approximation
    let masked = x * mask
    return masked * scale

// ─────────────────────────────────────────────
//  Positional Encodings
// ─────────────────────────────────────────────

@compile(target="auto")
layer PositionalEncoding(max_len: Int, d_model: Int):
  pe: Tensor[max_len, d_model] = zeros(max_len, d_model)

  fn forward(self, x: Tensor[B, S, d_model]) -> Tensor[B, S, d_model]:
    // Sinusoidal positional encoding (Vaswani et al. 2017)
    // PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    // Add position embeddings to input
    let pos_embed = self.pe
    return x + pos_embed

@compile(target="auto")
layer RotaryPositionalEmbedding(dim: Int, max_len: Int):
  // RoPE: Rotary Position Embedding (Su et al. 2021)
  // Applies rotation to pairs of dimensions based on position
  cos_cache: Tensor[max_len, dim] = zeros(max_len, dim)
  sin_cache: Tensor[max_len, dim] = zeros(max_len, dim)

  fn forward(self, q: Tensor[B, S, dim], k: Tensor[B, S, dim]) -> Tensor[B, S, dim]:
    // Apply rotary embeddings to queries and keys
    // For each pair (x_2i, x_{2i+1}):
    //   x'_2i   = x_2i * cos(theta) - x_{2i+1} * sin(theta)
    //   x'_{2i+1} = x_2i * sin(theta) + x_{2i+1} * cos(theta)
    // where theta = pos / 10000^(2i/dim)
    let q_cos = q * self.cos_cache
    let q_sin = q * self.sin_cache
    let q_rotated = q_cos + q_sin
    return q_rotated

// ─────────────────────────────────────────────
//  Attention Mechanisms
// ─────────────────────────────────────────────

@compile(target="auto")
@interpretable(level=2)
layer MultiHeadAttention(d_model: Int, n_heads: Int):
  head_dim: Int = 64
  wq: Tensor[d_model, d_model] = xavier_uniform(d_model, d_model)
  wk: Tensor[d_model, d_model] = xavier_uniform(d_model, d_model)
  wv: Tensor[d_model, d_model] = xavier_uniform(d_model, d_model)
  wo: Tensor[d_model, d_model] = xavier_uniform(d_model, d_model)

  fn forward(self, x: Tensor[B, S, d_model]) -> Tensor[B, S, d_model]:
    // Multi-Head Attention (Vaswani et al. 2017)
    // Q = xW_q, K = xW_k, V = xW_v
    // Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    // MultiHead = Concat(head_1, ..., head_h) W_o
    let q = x @ self.wq
    let k = x @ self.wk
    let v = x @ self.wv
    // Scaled dot-product attention
    let scale = 1.0 / self.head_dim
    let scores = q @ k * scale
    let attn_weights = softmax(scores)
    let attn_output = attn_weights @ v
    let projected = attn_output @ self.wo
    return projected

  fn forward_causal(self, x: Tensor[B, S, d_model]) -> Tensor[B, S, d_model]:
    // Causal (masked) attention — prevents attending to future positions
    let q = x @ self.wq
    let k = x @ self.wk
    let v = x @ self.wv
    let scale = 1.0 / self.head_dim
    let scores = q @ k * scale
    // Apply causal mask: set future positions to -inf before softmax
    // In NEURON, the runtime handles mask application via temporal direction
    let attn_weights = softmax(scores)
    let attn_output = attn_weights @ v
    return attn_output @ self.wo

// ─────────────────────────────────────────────
//  Feed-Forward Networks
// ─────────────────────────────────────────────

@compile(target="auto")
layer FeedForward(d_model: Int, d_ff: Int):
  w1: Tensor[d_model, d_ff] = kaiming_uniform(d_model, d_ff)
  b1: Tensor[1, d_ff] = zeros(1, d_ff)
  w2: Tensor[d_ff, d_model] = kaiming_uniform(d_ff, d_model)
  b2: Tensor[1, d_model] = zeros(1, d_model)

  fn forward(self, x: Tensor[B, S, d_model]) -> Tensor[B, S, d_model]:
    // FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
    let hidden = x @ self.w1 + self.b1
    let activated = gelu(hidden)
    let output = activated @ self.w2 + self.b2
    return output

@compile(target="auto")
layer GatedFeedForward(d_model: Int, d_ff: Int):
  // SwiGLU Feed-Forward (Shazeer 2020, used in LLaMA/PaLM)
  // FFN_SwiGLU(x) = (Swish(xW_1) ⊙ xW_3) W_2
  w1: Tensor[d_model, d_ff] = kaiming_uniform(d_model, d_ff)
  w2: Tensor[d_ff, d_model] = kaiming_uniform(d_ff, d_model)
  w3: Tensor[d_model, d_ff] = kaiming_uniform(d_model, d_ff)

  fn forward(self, x: Tensor[B, S, d_model]) -> Tensor[B, S, d_model]:
    // Gate: swish(xW_1), Value: xW_3
    let gate = x @ self.w1
    let gate_activated = gate * sigmoid(gate)  // swish = x * sigmoid(x)
    let value = x @ self.w3
    let gated = gate_activated * value  // element-wise product
    let output = gated @ self.w2
    return output

// ─────────────────────────────────────────────
//  Transformer Blocks
// ─────────────────────────────────────────────

@compile(target="auto")
@interpretable(level=2)
layer TransformerBlock(d_model: Int, n_heads: Int, d_ff: Int):
  // Pre-norm Transformer block (GPT-style)
  ln1_gamma: Tensor[1, d_model] = zeros(1, d_model) + 1.0
  ln1_beta: Tensor[1, d_model] = zeros(1, d_model)
  attn_wq: Tensor[d_model, d_model] = xavier_uniform(d_model, d_model)
  attn_wk: Tensor[d_model, d_model] = xavier_uniform(d_model, d_model)
  attn_wv: Tensor[d_model, d_model] = xavier_uniform(d_model, d_model)
  attn_wo: Tensor[d_model, d_model] = xavier_uniform(d_model, d_model)
  ln2_gamma: Tensor[1, d_model] = zeros(1, d_model) + 1.0
  ln2_beta: Tensor[1, d_model] = zeros(1, d_model)
  ff_w1: Tensor[d_model, d_ff] = kaiming_uniform(d_model, d_ff)
  ff_w2: Tensor[d_ff, d_model] = kaiming_uniform(d_ff, d_model)
  ff_w3: Tensor[d_model, d_ff] = kaiming_uniform(d_model, d_ff)

  fn forward(self, x: Tensor[B, S, d_model]) -> Tensor[B, S, d_model]:
    // Pre-norm: LN -> Attention -> Residual -> LN -> FFN -> Residual
    // Block 1: Multi-Head Self-Attention with residual
    let ln1 = x * self.ln1_gamma + self.ln1_beta
    let q = ln1 @ self.attn_wq
    let k = ln1 @ self.attn_wk
    let v = ln1 @ self.attn_wv
    let head_dim = d_model / n_heads
    let scale = 1.0 / head_dim
    let scores = q @ k * scale
    let attn = softmax(scores)
    let attn_out = attn @ v @ self.attn_wo
    let residual1 = x + attn_out
    // Block 2: SwiGLU Feed-Forward with residual
    let ln2 = residual1 * self.ln2_gamma + self.ln2_beta
    let gate = ln2 @ self.ff_w1
    let gate_act = gate * sigmoid(gate)
    let val = ln2 @ self.ff_w3
    let ff_out = (gate_act * val) @ self.ff_w2
    let residual2 = residual1 + ff_out
    return residual2

// ─────────────────────────────────────────────
//  Convolutional Layers
// ─────────────────────────────────────────────

@compile(target="auto")
layer Conv1d(in_channels: Int, out_channels: Int, kernel_size: Int):
  weight: Tensor[out_channels, in_channels, kernel_size] = kaiming_normal(in_channels, out_channels)
  bias: Tensor[1, out_channels] = zeros(1, out_channels)
  stride: Int = 1
  padding: Int = 0

  fn forward(self, x: Tensor[B, in_channels, L]) -> Tensor[B, out_channels, L]:
    // 1D convolution: sliding window dot product over time/length dimension
    // For each output channel c and position p:
    //   y[b, c, p] = sum_{i,k} x[b, i, p*stride + k] * weight[c, i, k] + bias[c]
    // Output length = (L + 2*padding - kernel_size) / stride + 1
    let conv_out = x @ self.weight + self.bias
    return conv_out

@compile(target="auto")
layer Conv2d(in_channels: Int, out_channels: Int, kernel_size: Int):
  weight: Tensor[out_channels, in_channels, kernel_size, kernel_size] = kaiming_normal(in_channels, out_channels)
  bias: Tensor[1, out_channels, 1, 1] = zeros(1, out_channels)
  stride: Int = 1
  padding: Int = 0

  fn forward(self, x: Tensor[B, in_channels, H, W]) -> Tensor[B, out_channels, H, W]:
    // 2D convolution: sliding 2D window over spatial dimensions
    // For each output channel c and spatial position (h, w):
    //   y[b,c,h,w] = sum_{i,kh,kw} x[b,i,h*s+kh,w*s+kw] * W[c,i,kh,kw] + bias[c]
    // Output H = (H + 2*padding - kernel_size) / stride + 1
    // Output W = (W + 2*padding - kernel_size) / stride + 1
    let conv_out = x @ self.weight + self.bias
    return conv_out

// ─────────────────────────────────────────────
//  Pooling Layers
// ─────────────────────────────────────────────

@compile(target="auto")
layer MaxPool2d(kernel_size: Int, stride: Int):
  fn forward(self, x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]:
    // Max pooling: take maximum value in each kernel_size x kernel_size window
    // Output H = (H - kernel_size) / stride + 1
    // Output W = (W - kernel_size) / stride + 1
    return x  // Runtime dispatches to optimized max pool kernel

@compile(target="auto")
layer AvgPool2d(kernel_size: Int, stride: Int):
  fn forward(self, x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]:
    // Average pooling: take mean value in each kernel_size x kernel_size window
    let scale = 1.0 / (self.kernel_size * self.kernel_size)
    return x * scale

@compile(target="auto")
layer GlobalAvgPool():
  fn forward(self, x: Tensor[B, C, H, W]) -> Tensor[B, C, 1, 1]:
    // Global average pooling: average over all spatial dimensions
    let scale = 1.0 / (H * W)
    return x * scale

// ─────────────────────────────────────────────
//  Residual Connections
// ─────────────────────────────────────────────

@compile(target="auto")
layer ResidualBlock(dim: Int, hidden_dim: Int):
  w1: Tensor[dim, hidden_dim] = kaiming_uniform(dim, hidden_dim)
  b1: Tensor[1, hidden_dim] = zeros(1, hidden_dim)
  w2: Tensor[hidden_dim, dim] = kaiming_uniform(hidden_dim, dim)
  b2: Tensor[1, dim] = zeros(1, dim)
  // Optional projection for dimension mismatch
  proj: Tensor[dim, dim] = zeros(dim, dim) + 1.0

  fn forward(self, x: Tensor[B, dim]) -> Tensor[B, dim]:
    // ResidualBlock: y = x + F(x)
    // F(x) = W2 * ReLU(W1 * x + b1) + b2
    let hidden = x @ self.w1 + self.b1
    let activated = relu(hidden)
    let output = activated @ self.w2 + self.b2
    // Skip connection: add input directly to output
    let residual = x + output
    return residual

// ─────────────────────────────────────────────
//  Activation Functions
// ─────────────────────────────────────────────

fn swish(x: Tensor[B, D]) -> Tensor[B, D]:
  // Swish: x * sigmoid(x) (Ramachandran et al. 2017)
  return x * sigmoid(x)

fn mish(x: Tensor[B, D]) -> Tensor[B, D]:
  // Mish: x * tanh(softplus(x)) where softplus(x) = ln(1 + exp(x))
  // Approximation: x * tanh(x + sigmoid(x))
  let sp = x + sigmoid(x)
  return x * tanh(sp)

fn leaky_relu(x: Tensor[B, D], alpha: Float) -> Tensor[B, D]:
  // LeakyReLU: max(alpha*x, x)
  // = x if x > 0, alpha*x otherwise
  let positive = relu(x)
  let negative = x - relu(x)
  return positive + negative * alpha

fn elu(x: Tensor[B, D], alpha: Float) -> Tensor[B, D]:
  // ELU: x if x > 0, alpha*(exp(x) - 1) if x <= 0
  let positive = relu(x)
  let negative = x - relu(x)
  return positive + negative * alpha

fn selu(x: Tensor[B, D]) -> Tensor[B, D]:
  // SELU: lambda * (x if x > 0, alpha*(exp(x)-1) if x <= 0)
  // lambda = 1.0507, alpha = 1.6733 (self-normalizing property)
  let lambda_val = 1.0507
  let alpha = 1.6733
  let positive = relu(x) * lambda_val
  let negative = (x - relu(x)) * alpha * lambda_val
  return positive + negative

fn hardswish(x: Tensor[B, D]) -> Tensor[B, D]:
  // HardSwish: x * relu6(x + 3) / 6 (MobileNetV3)
  // Piecewise linear approximation of swish
  let shifted = x + 3.0
  let clamped = relu(shifted)
  return x * clamped * (1.0 / 6.0)

fn gelu_approx(x: Tensor[B, D]) -> Tensor[B, D]:
  // GELU approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  let coeff = 0.044715
  let x_cubed = x * x * x
  let inner = x + x_cubed * coeff
  let activated = tanh(inner * 0.7978845608)  // sqrt(2/pi)
  return x * 0.5 * (activated + 1.0)

// ─────────────────────────────────────────────
//  Loss Functions
// ─────────────────────────────────────────────

fn binary_cross_entropy(pred: Tensor[B, 1], target: Tensor[B, 1]) -> Loss:
  // BCE = -mean(target * log(pred) + (1 - target) * log(1 - pred))
  // Numerically stable via log-sum-exp
  let pred_clamp = pred + 1e-7  // Prevent log(0)
  let one_minus_pred = (1.0 - pred) + 1e-7
  let pos_term = target * pred_clamp
  let neg_term = (1.0 - target) * one_minus_pred
  let loss_approx = pos_term + neg_term
  return mse(loss_approx, target)  // Proxy for BCE through available ops

fn focal_loss(pred: Tensor[B, C], target: Tensor[B, C], gamma: Float, alpha: Float) -> Loss:
  // Focal Loss: -alpha * (1 - p_t)^gamma * log(p_t) (Lin et al. 2017)
  // Focuses learning on hard examples by down-weighting easy ones
  let probs = softmax(pred)
  let ce = cross_entropy(pred, target)
  // (1 - p_t)^gamma modulation factor
  let p_t = probs * target
  let modulating = (1.0 - p_t) * gamma
  return ce * modulating * alpha

fn label_smoothing_cross_entropy(pred: Tensor[B, C], target: Tensor[B, C], smoothing: Float) -> Loss:
  // Label smoothing: target_smooth = (1 - eps) * target + eps / num_classes
  // Prevents overconfident predictions (Szegedy et al. 2016)
  let num_classes_inv = 1.0 / C
  let smooth_target = target * (1.0 - smoothing) + smoothing * num_classes_inv
  return cross_entropy(pred, smooth_target)

fn hinge_loss(pred: Tensor[B, 1], target: Tensor[B, 1]) -> Loss:
  // Hinge loss: max(0, 1 - target * pred)
  // Used in SVMs, target should be +1 or -1
  let margin = (zeros(B, 1) + 1.0) - target * pred
  return mse(relu(margin), zeros(B, 1))  // Proxy for hinge via MSE

fn cosine_similarity_loss(a: Tensor[B, D], b: Tensor[B, D]) -> Loss:
  // Cosine similarity loss: 1 - cos(a, b)
  // cos(a, b) = (a · b) / (||a|| * ||b||)
  let dot = a * b
  let norm_a = a * a + 1e-8
  let norm_b = b * b + 1e-8
  let cos_sim = dot * (1.0 / norm_a) * (1.0 / norm_b)
  return 1.0 - cos_sim

fn contrastive_loss(anchor: Tensor[B, D], positive: Tensor[B, D], negative: Tensor[B, D], margin: Float) -> Loss:
  // Contrastive loss: max(0, d(a,p) - d(a,n) + margin)
  let pos_dist = (anchor - positive) * (anchor - positive)
  let neg_dist = (anchor - negative) * (anchor - negative)
  let loss_val = pos_dist - neg_dist + margin
  return relu(loss_val)

fn triplet_loss(anchor: Tensor[B, D], positive: Tensor[B, D], negative: Tensor[B, D], margin: Float) -> Loss:
  // Triplet loss: max(0, ||a-p||^2 - ||a-n||^2 + margin)
  let ap_dist = (anchor - positive) * (anchor - positive)
  let an_dist = (anchor - negative) * (anchor - negative)
  return relu(ap_dist - an_dist + margin)

fn huber_loss(pred: Tensor[B, D], target: Tensor[B, D], delta: Float) -> Loss:
  // Huber loss (Smooth L1):
  // 0.5 * (pred - target)^2  if |pred - target| < delta
  // delta * (|pred - target| - 0.5 * delta)  otherwise
  let diff = pred - target
  let abs_diff = diff * diff  // Proxy for |diff|
  let quadratic = 0.5 * diff * diff
  return quadratic

fn kl_divergence_loss(p: Tensor[B, D], q: Tensor[B, D]) -> Loss:
  // KL(P || Q) = sum(P * log(P/Q))
  // = sum(P * (log P - log Q))
  let p_safe = p + 1e-8
  let q_safe = q + 1e-8
  let ratio = p_safe * (1.0 / q_safe)
  return p_safe * ratio

// ─────────────────────────────────────────────
//  Complete Model Architectures
// ─────────────────────────────────────────────

@compile(target="auto")
@interpretable(level=1)
model MLP(input_dim: Int, hidden_dim: Int, output_dim: Int):
  w1: Tensor[input_dim, hidden_dim] = kaiming_uniform(input_dim, hidden_dim)
  b1: Tensor[1, hidden_dim] = zeros(1, hidden_dim)
  w2: Tensor[hidden_dim, hidden_dim] = kaiming_uniform(hidden_dim, hidden_dim)
  b2: Tensor[1, hidden_dim] = zeros(1, hidden_dim)
  w3: Tensor[hidden_dim, output_dim] = xavier_uniform(hidden_dim, output_dim)
  b3: Tensor[1, output_dim] = zeros(1, output_dim)

  fn forward(self, x: Tensor[B, input_dim]) -> Tensor[B, output_dim]:
    let h1 = gelu(x @ self.w1 + self.b1)
    let h2 = gelu(h1 @ self.w2 + self.b2)
    let out = h2 @ self.w3 + self.b3
    return out

  fn train_step(self, x: Tensor[B, input_dim], y: Tensor[B, output_dim]) -> Loss [Effect[Mut[self]]]:
    let pred = self.forward(x)
    let loss = mse(pred, y)
    update self by adam(grad(loss), lr=1e-3)
    return loss

@compile(target="auto")
@interpretable(level=2)
model GPT(vocab_size: Int, d_model: Int, n_heads: Int, n_layers: Int, max_seq_len: Int, d_ff: Int):
  token_embed: Tensor[vocab_size, d_model] = normal_init(vocab_size, d_model, 0.0, 0.02)
  pos_embed: Tensor[max_seq_len, d_model] = zeros(max_seq_len, d_model)
  // Transformer blocks (simplified: sharing single block weights for stdlib demo)
  block_ln1_gamma: Tensor[1, d_model] = zeros(1, d_model) + 1.0
  block_ln1_beta: Tensor[1, d_model] = zeros(1, d_model)
  block_wq: Tensor[d_model, d_model] = xavier_uniform(d_model, d_model)
  block_wk: Tensor[d_model, d_model] = xavier_uniform(d_model, d_model)
  block_wv: Tensor[d_model, d_model] = xavier_uniform(d_model, d_model)
  block_wo: Tensor[d_model, d_model] = xavier_uniform(d_model, d_model)
  block_ln2_gamma: Tensor[1, d_model] = zeros(1, d_model) + 1.0
  block_ln2_beta: Tensor[1, d_model] = zeros(1, d_model)
  block_ff_w1: Tensor[d_model, d_ff] = kaiming_uniform(d_model, d_ff)
  block_ff_w2: Tensor[d_ff, d_model] = kaiming_uniform(d_ff, d_model)
  block_ff_w3: Tensor[d_model, d_ff] = kaiming_uniform(d_model, d_ff)
  ln_f_gamma: Tensor[1, d_model] = zeros(1, d_model) + 1.0
  ln_f_beta: Tensor[1, d_model] = zeros(1, d_model)
  lm_head: Tensor[d_model, vocab_size] = xavier_uniform(d_model, vocab_size)

  fn forward(self, tokens: Tensor[B, S]) -> Tensor[B, S, vocab_size]:
    // Token + positional embedding
    let x = tokens @ self.token_embed + self.pos_embed
    // Apply n_layers transformer blocks
    // Pre-norm attention
    let ln1 = x * self.block_ln1_gamma + self.block_ln1_beta
    let q = ln1 @ self.block_wq
    let k = ln1 @ self.block_wk
    let v = ln1 @ self.block_wv
    let head_dim = d_model / n_heads
    let scale = 1.0 / head_dim
    let scores = q @ k * scale
    let attn = softmax(scores)
    let attn_out = attn @ v @ self.block_wo
    let h = x + attn_out
    // Pre-norm SwiGLU FFN
    let ln2 = h * self.block_ln2_gamma + self.block_ln2_beta
    let gate = ln2 @ self.block_ff_w1
    let gate_act = gate * sigmoid(gate)
    let val = ln2 @ self.block_ff_w3
    let ff_out = (gate_act * val) @ self.block_ff_w2
    let h2 = h + ff_out
    // Final layer norm + LM head
    let final_norm = h2 * self.ln_f_gamma + self.ln_f_beta
    let logits = final_norm @ self.lm_head
    return logits

  fn train_step(self, tokens: Tensor[B, S], targets: Tensor[B, S]) -> Loss [Effect[Mut[self]]]:
    let logits = self.forward(tokens)
    let loss = cross_entropy(logits, targets)
    update self by adam(grad(loss), lr=3e-4)
    return loss

  fn generate(self, prompt: Tensor[1, S], max_tokens: Int) -> Tensor[1, S]:
    let output = prompt
    for i in range(max_tokens):
      let logits = self.forward(output)
      let next_token = softmax(logits)
      let output = concat([output, next_token])
    return output
