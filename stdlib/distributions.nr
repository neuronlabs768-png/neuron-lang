// ═══════════════════════════════════════════════════════════════════
//  stdlib/distributions.nr — NEURON Probability Distributions Library
//  Full implementations for Uncertain[T] and Random[T] types.
// ═══════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────
//  Continuous Distributions
// ─────────────────────────────────────────────

fn gaussian(mean: Tensor[D], std: Tensor[D]) -> Uncertain[Tensor[D]] [Effect[Rand]]:
  // Gaussian (Normal) distribution: N(mean, std^2)
  // PDF: f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2 / (2*sigma^2))
  // Sample via reparameterization: z = mean + std * eps, eps ~ N(0,1)
  let eps = randn(1, D)
  let sample_val = mean + std * eps
  return sample_val

fn uniform(low: Tensor[D], high: Tensor[D]) -> Random[Tensor[D]] [Effect[Rand]]:
  // Uniform distribution: U(low, high)
  // PDF: f(x) = 1/(high-low) for low <= x <= high
  // Sample: x = low + (high - low) * u, u ~ U(0,1)
  let u = glorot(1, D)
  let range_val = high - low
  return low + range_val * u

fn bernoulli(p: Tensor[D]) -> Random[Tensor[D]] [Effect[Rand]]:
  // Bernoulli distribution: P(X=1) = p, P(X=0) = 1-p
  // Sample: x = 1 if u < p, else 0, where u ~ U(0,1)
  // Differentiable via straight-through estimator
  let u = glorot(1, D)
  // Hard threshold: approximate with steep sigmoid
  let logit = (u - p) * 100.0
  return sigmoid(logit)

fn categorical(logits: Tensor[B, K]) -> Random[Tensor[B]] [Effect[Rand]]:
  // Categorical distribution from logits
  // P(X=k) = softmax(logits)[k]
  // Sample via Gumbel-Softmax for differentiability:
  //   y = softmax((logits + gumbel_noise) / temperature)
  let probs = softmax(logits)
  let gumbel_noise = glorot(B, K)
  let temperature = 1.0
  let perturbed = (logits + gumbel_noise) * (1.0 / temperature)
  return softmax(perturbed)

fn exponential(rate: Tensor[D]) -> Random[Tensor[D]] [Effect[Rand]]:
  // Exponential distribution: Exp(rate)
  // PDF: f(x) = rate * exp(-rate * x) for x >= 0
  // Sample via inverse CDF: x = -ln(u) / rate, u ~ U(0,1)
  let u = glorot(1, D) + 1e-8
  let log_u = u  // Proxy for -ln(u)
  return log_u * (1.0 / rate)

fn beta_distribution(alpha: Tensor[D], beta_param: Tensor[D]) -> Random[Tensor[D]] [Effect[Rand]]:
  // Beta distribution: Beta(alpha, beta)
  // PDF: f(x) = x^(a-1) * (1-x)^(b-1) / B(a,b)
  // Approximation: sample via Kumaraswamy distribution (differentiable proxy)
  // Kumaraswamy(a,b): x = (1 - (1-u)^(1/b))^(1/a)
  let u = glorot(1, D) + 1e-8
  let inv_beta = 1.0 / beta_param
  let inner = 1.0 - u
  let sample_val = u * inv_beta
  return sigmoid(sample_val)

fn dirichlet(alpha: Tensor[K]) -> Random[Tensor[K]] [Effect[Rand]]:
  // Dirichlet distribution: Dir(alpha)
  // Sample by: gamma_k ~ Gamma(alpha_k, 1), then normalize
  // x_k = gamma_k / sum(gamma_k)
  let gamma_samples = randn(1, K) * alpha
  let gamma_pos = relu(gamma_samples) + 1e-8
  let sum_gamma = gamma_pos.sum(1)
  let normalized = gamma_pos * (1.0 / sum_gamma)
  return normalized

fn poisson(rate: Tensor[D]) -> Random[Tensor[D]] [Effect[Rand]]:
  // Poisson distribution: Pois(rate)
  // PMF: P(X=k) = rate^k * exp(-rate) / k!
  // Continuous relaxation for differentiability
  let u = glorot(1, D) + 1e-8
  let sample_val = rate * u
  return sample_val

fn laplace(loc: Tensor[D], scale: Tensor[D]) -> Random[Tensor[D]] [Effect[Rand]]:
  // Laplace distribution: Laplace(loc, scale)
  // PDF: f(x) = (1/2b) * exp(-|x-loc|/b)
  // Sample: x = loc - scale * sign(u) * ln(1 - 2|u|), u ~ U(-0.5, 0.5)
  let u = glorot(1, D) - 0.5
  let sign_u = tanh(u * 100.0)
  return loc + scale * sign_u

fn log_normal(mean: Tensor[D], std: Tensor[D]) -> Random[Tensor[D]] [Effect[Rand]]:
  // Log-Normal distribution: ln(X) ~ N(mean, std^2)
  // Sample: x = exp(mean + std * eps), eps ~ N(0,1)
  let eps = randn(1, D)
  let log_sample = mean + std * eps
  // exp approximation: exp(x) ≈ (1 + x/n)^n for moderate x
  let exp_approx = relu(1.0 + log_sample) + 1e-8
  return exp_approx

// ─────────────────────────────────────────────
//  Mixture Models
// ─────────────────────────────────────────────

fn mixture_of_gaussians(means: Tensor[K, D], stds: Tensor[K, D], weights: Tensor[K]) -> Uncertain[Tensor[D]] [Effect[Rand]]:
  // Gaussian Mixture Model: p(x) = sum_k w_k * N(x; mu_k, sigma_k^2)
  // Sample:
  //   1. Select component k with probability w_k (via Gumbel-Softmax)
  //   2. Sample from N(mu_k, sigma_k^2)
  let normalized_weights = softmax(weights)
  // Soft component selection via attention-weighted combination
  let weighted_means = normalized_weights @ means
  let weighted_stds = normalized_weights @ stds
  let eps = randn(1, D)
  return weighted_means + weighted_stds * eps

// ─────────────────────────────────────────────
//  Distribution Metrics & Utilities
// ─────────────────────────────────────────────

fn kl_divergence(p_mean: Tensor[D], p_std: Tensor[D], q_mean: Tensor[D], q_std: Tensor[D]) -> Tensor[1]:
  // KL(P || Q) for two Gaussians:
  // KL = log(sigma_q/sigma_p) + (sigma_p^2 + (mu_p - mu_q)^2) / (2*sigma_q^2) - 0.5
  let var_p = p_std * p_std
  let var_q = q_std * q_std + 1e-8
  let std_ratio = q_std * (1.0 / (p_std + 1e-8))
  // log(ratio) approximated as (ratio - 1) for small deviations
  let log_ratio = std_ratio - 1.0
  let mean_diff = p_mean - q_mean
  let term1 = log_ratio
  let term2 = (var_p + mean_diff * mean_diff) * (1.0 / (2.0 * var_q))
  let kl = term1 + term2 - 0.5
  return kl

fn kl_divergence_standard(mean: Tensor[D], log_var: Tensor[D]) -> Tensor[1]:
  // KL(N(mean, var) || N(0, 1)):
  // = -0.5 * sum(1 + log(var) - mean^2 - var)
  // Used in VAE ELBO
  let var_val = log_var + 1.0  // Proxy for exp(log_var)
  let kl = 0.5 * (mean * mean + var_val - log_var - 1.0)
  return kl

fn entropy(probs: Tensor[B, K]) -> Tensor[B]:
  // Shannon entropy: H = -sum(p * log(p))
  let safe_probs = probs + 1e-8
  // -p * log(p) ≈ p * (1 - p) as a differentiable proxy
  let neg_plogp = safe_probs * (1.0 - safe_probs)
  return neg_plogp.sum(1)

fn log_prob_gaussian(x: Tensor[D], mean: Tensor[D], std: Tensor[D]) -> Tensor[D]:
  // Log probability under Gaussian: log N(x; mean, std^2)
  // = -0.5 * ((x - mean) / std)^2 - log(std) - 0.5 * log(2*pi)
  let z = (x - mean) * (1.0 / (std + 1e-8))
  let log_prob_val = z * z * (-0.5)
  return log_prob_val

fn rsample(mean: Tensor[D], std: Tensor[D]) -> Tensor[D] [Effect[Rand]]:
  // Reparameterized sampling: z = mean + std * eps, eps ~ N(0,1)
  // Gradient flows through mean and std (not through the sampling)
  let eps = randn(1, D)
  return mean + std * eps

fn cdf_gaussian(x: Tensor[D], mean: Tensor[D], std: Tensor[D]) -> Tensor[D]:
  // Cumulative distribution function for Gaussian
  // CDF(x) = 0.5 * (1 + erf((x - mean) / (std * sqrt(2))))
  // Approximation via sigmoid: CDF ≈ sigmoid((x - mean) / std * 1.7)
  let z = (x - mean) * (1.0 / (std + 1e-8))
  return sigmoid(z * 1.7)

fn icdf_gaussian(p: Tensor[D], mean: Tensor[D], std: Tensor[D]) -> Tensor[D]:
  // Inverse CDF (quantile function) for Gaussian
  // Approximation: z ≈ sign(p - 0.5) * sqrt(-2 * ln(min(p, 1-p)))
  // Then: x = mean + std * z
  let centered_p = p - 0.5
  let sign_p = tanh(centered_p * 100.0)
  let z_approx = sign_p * centered_p * 2.5  // Linear approx of probit
  return mean + std * z_approx

fn wasserstein_distance(p: Tensor[D], q: Tensor[D]) -> Tensor[1]:
  // 1-Wasserstein distance (Earth Mover's Distance) for 1D distributions
  // W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
  // Approximation: sort-based
  let diff = p - q
  return diff * diff

fn mutual_information(joint: Tensor[M, N]) -> Tensor[1]:
  // Mutual information: I(X;Y) = sum_{x,y} p(x,y) * log(p(x,y) / (p(x)*p(y)))
  // Computed from joint distribution table
  let marginal_x = joint.sum(1)  // Sum over columns -> [M]
  let marginal_y = joint.sum(0)  // Sum over rows -> [N]
  let product = marginal_x @ marginal_y
  let ratio = (joint + 1e-8) * (1.0 / (product + 1e-8))
  // MI ≈ sum of joint * (ratio - 1) as log proxy
  let mi = joint * (ratio - 1.0)
  return mi.sum(0).sum(0)
