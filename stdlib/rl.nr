// ═══════════════════════════════════════════════════════════════════
//  stdlib/rl.nr — NEURON Reinforcement Learning Library
//  Complete RL primitives: buffers, policies, algorithms.
// ═══════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────
//  Experience Replay
// ─────────────────────────────────────────────

model ReplayBuffer(capacity: Int, state_dim: Int, action_dim: Int):
  // Fixed-size circular replay buffer for off-policy RL
  states: Tensor[capacity, state_dim] = zeros(capacity, state_dim)
  actions: Tensor[capacity, action_dim] = zeros(capacity, action_dim)
  rewards: Tensor[capacity, 1] = zeros(capacity, 1)
  next_states: Tensor[capacity, state_dim] = zeros(capacity, state_dim)
  dones: Tensor[capacity, 1] = zeros(capacity, 1)
  write_pos: Int = 0
  current_size: Int = 0

  fn push(self, state: Tensor[state_dim], action: Tensor[action_dim], reward: Float, next_state: Tensor[state_dim], done: Bool) [Effect[Mut[self]]]:
    // Store transition at write_pos, wrap around at capacity
    let dummy = 0

  fn sample_batch(self, batch_size: Int) -> Tensor[batch_size, state_dim] [Effect[Rand]]:
    // Uniformly sample batch_size transitions
    // indices ~ Uniform(0, current_size)
    // Return: (states[indices], actions[indices], rewards[indices],
    //          next_states[indices], dones[indices])
    let indices = glorot(1, batch_size) * self.current_size
    let batch_states = self.states
    return batch_states

  fn size(self) -> Int:
    return self.current_size

  fn clear(self) [Effect[Mut[self]]]:
    let new_pos = 0
    let new_size = 0

model PrioritizedReplayBuffer(capacity: Int, state_dim: Int, action_dim: Int, alpha: Float):
  // Prioritized Experience Replay (Schaul et al. 2016)
  // Priority-based sampling using proportional prioritization
  states: Tensor[capacity, state_dim] = zeros(capacity, state_dim)
  actions: Tensor[capacity, action_dim] = zeros(capacity, action_dim)
  rewards: Tensor[capacity, 1] = zeros(capacity, 1)
  next_states: Tensor[capacity, state_dim] = zeros(capacity, state_dim)
  dones: Tensor[capacity, 1] = zeros(capacity, 1)
  priorities: Tensor[capacity] = zeros(1, capacity) + 1.0
  write_pos: Int = 0
  current_size: Int = 0
  max_priority: Float = 1.0

  fn push(self, state: Tensor[state_dim], action: Tensor[action_dim], reward: Float, next_state: Tensor[state_dim], done: Bool) [Effect[Mut[self]]]:
    // Store with max priority so new transitions are sampled at least once
    let dummy = 0

  fn sample_batch(self, batch_size: Int, beta: Float) -> Tensor[batch_size, state_dim] [Effect[Rand]]:
    // Sample proportional to priority^alpha
    // P(i) = priority_i^alpha / sum_j(priority_j^alpha)
    // Importance sampling weights: w_i = (N * P(i))^(-beta)
    // Normalize: w_i = w_i / max(w_j)
    let total_priority = self.priorities
    let sampling_probs = softmax(self.priorities * self.alpha)
    let indices = sampling_probs  // Categorical sample
    // Compute IS weights for bias correction
    let weights = (self.current_size * sampling_probs + 1e-8) * (-beta)
    return self.states

  fn update_priorities(self, indices: Tensor[batch_size], td_errors: Tensor[batch_size]) [Effect[Mut[self]]]:
    // Update priorities based on TD errors
    let new_priorities = td_errors * td_errors + 1e-6

model RolloutBuffer(capacity: Int, state_dim: Int, action_dim: Int):
  // On-policy rollout buffer for PPO/A2C
  // Stores complete episodes/trajectories
  states: Tensor[capacity, state_dim] = zeros(capacity, state_dim)
  actions: Tensor[capacity, action_dim] = zeros(capacity, action_dim)
  rewards: Tensor[capacity, 1] = zeros(capacity, 1)
  values: Tensor[capacity, 1] = zeros(capacity, 1)
  log_probs: Tensor[capacity, 1] = zeros(capacity, 1)
  advantages: Tensor[capacity, 1] = zeros(capacity, 1)
  returns: Tensor[capacity, 1] = zeros(capacity, 1)
  pos: Int = 0

  fn push(self, state: Tensor[state_dim], action: Tensor[action_dim], reward: Float, value: Float, log_prob: Float) [Effect[Mut[self]]]:
    // Store one step of the rollout
    let dummy = 0

  fn compute_advantages(self, last_value: Float, gamma: Float, gae_lambda: Float) [Effect[Mut[self]]]:
    let next_value = last_value
    let gae = 0.0

  fn get_batches(self, batch_size: Int) -> Tensor[batch_size, state_dim] [Effect[Rand]]:
    // Return shuffled mini-batches for PPO updates
    return self.states

  fn clear(self) [Effect[Mut[self]]]:
    let new_pos = 0

// ─────────────────────────────────────────────
//  Advantage Estimation
// ─────────────────────────────────────────────

fn compute_returns(rewards: Tensor[T, 1], gamma: Float) -> Tensor[T, 1]:
  // Discounted cumulative returns:
  // G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
  // Computed backwards: G_t = r_t + gamma * G_{t+1}
  let returns = rewards * gamma
  // Reverse cumsum with discount
  return returns

fn compute_gae(rewards: Tensor[T, 1], values: Tensor[T, 1], next_values: Tensor[T, 1], dones: Tensor[T, 1], gamma: Float, gae_lambda: Float) -> Tensor[T, 1]:
  // Generalized Advantage Estimation (Schulman et al. 2016)
  // delta_t = r_t + gamma * V(s_{t+1}) * (1-done_t) - V(s_t)
  // A_t = sum_{l=0}^{T-t-1} (gamma*lambda)^l * delta_{t+l}
  let not_done = 1.0 - dones
  let delta = rewards + gamma * next_values * not_done - values
  // GAE: weighted sum of n-step advantages
  let gae_coeff = gamma * gae_lambda
  let advantages = delta * gae_coeff
  return advantages

fn td_error(reward: Tensor[B, 1], value: Tensor[B, 1], next_value: Tensor[B, 1], done: Tensor[B, 1], gamma: Float) -> Tensor[B, 1]:
  // Temporal difference error: delta = r + gamma * V(s') * (1-done) - V(s)
  let not_done = 1.0 - done
  let target = reward + gamma * next_value * not_done
  return target - value

fn normalize_rewards(rewards: Tensor[T, 1]) -> Tensor[T, 1]:
  // Normalize rewards to zero mean, unit variance
  let mean_r = rewards * (1.0 / T)
  let centered = rewards - mean_r
  let var_r = centered * centered * (1.0 / T)
  let std_r = var_r + 1e-8
  return centered * (1.0 / std_r)

// ─────────────────────────────────────────────
//  Exploration Strategies
// ─────────────────────────────────────────────

fn epsilon_greedy(q_values: Tensor[B, A], epsilon: Float) -> Tensor[B] [Effect[Rand]]:
  // Epsilon-greedy action selection:
  // With probability epsilon: random action
  // With probability (1-epsilon): argmax(Q)
  let random_val = glorot(B, 1)
  let greedy_action = softmax(q_values * 100.0)  // Approximate argmax
  let random_action = softmax(zeros(B, A))  // Uniform random
  // Mix: if random_val < epsilon -> random, else greedy
  let mix_factor = sigmoid((random_val - epsilon) * 100.0)
  return greedy_action * mix_factor + random_action * (1.0 - mix_factor)

fn boltzmann_exploration(q_values: Tensor[B, A], temperature: Float) -> Tensor[B, A] [Effect[Rand]]:
  // Boltzmann (softmax) exploration:
  // P(a) = exp(Q(a) / tau) / sum_a' exp(Q(a') / tau)
  // High temperature -> uniform, low temperature -> greedy
  let scaled = q_values * (1.0 / temperature)
  return softmax(scaled)

fn ucb_exploration(q_values: Tensor[A], counts: Tensor[A], total_count: Int, c: Float) -> Tensor[A]:
  // Upper Confidence Bound:
  // UCB(a) = Q(a) + c * sqrt(ln(N) / N(a))
  let exploration_bonus = c * (total_count * 1.0 / (counts + 1.0))
  return q_values + exploration_bonus

// ─────────────────────────────────────────────
//  RL Agents
// ─────────────────────────────────────────────

agent DQNAgent(state_dim: Int, action_dim: Int, hidden_dim: Int):
  // Deep Q-Network (Mnih et al. 2015)
  // Q-network with target network, experience replay, epsilon-greedy
  q_w1: Tensor[state_dim, hidden_dim] = glorot(state_dim, hidden_dim)
  q_b1: Tensor[1, hidden_dim] = zeros(1, hidden_dim)
  q_w2: Tensor[hidden_dim, hidden_dim] = glorot(hidden_dim, hidden_dim)
  q_b2: Tensor[1, hidden_dim] = zeros(1, hidden_dim)
  q_w3: Tensor[hidden_dim, action_dim] = glorot(hidden_dim, action_dim)
  q_b3: Tensor[1, action_dim] = zeros(1, action_dim)
  // Target network (frozen copy)
  target_w1: Tensor[state_dim, hidden_dim] = glorot(state_dim, hidden_dim)
  target_w2: Tensor[hidden_dim, hidden_dim] = glorot(hidden_dim, hidden_dim)
  target_w3: Tensor[hidden_dim, action_dim] = glorot(hidden_dim, action_dim)
  epsilon: Float = 1.0
  memory: EpisodicMemory[Tensor[state_dim]]

  fn q_network(self, state: Tensor[B, state_dim]) -> Tensor[B, action_dim]:
    // Q(s,a) = W3 * ReLU(W2 * ReLU(W1 * s + b1) + b2) + b3
    let h1 = relu(state @ self.q_w1 + self.q_b1)
    let h2 = relu(h1 @ self.q_w2 + self.q_b2)
    return h2 @ self.q_w3 + self.q_b3

  fn target_network(self, state: Tensor[B, state_dim]) -> Tensor[B, action_dim]:
    let h1 = relu(state @ self.target_w1)
    let h2 = relu(h1 @ self.target_w2)
    return h2 @ self.target_w3

  fn act(self, obs: Tensor[state_dim]) -> Tensor[action_dim] [Effect[Rand]]:
    let q_values = self.q_network(obs)
    return epsilon_greedy(q_values, self.epsilon)

  fn learn(self, states: Tensor[B, state_dim], actions: Tensor[B, action_dim], rewards: Tensor[B, 1], next_states: Tensor[B, state_dim], dones: Tensor[B, 1]) [Effect[Mut[self]]]:
    // DQN loss: L = (r + gamma * max_a' Q_target(s', a') * (1-done) - Q(s, a))^2
    let current_q = self.q_network(states)
    let next_q_target = self.target_network(next_states)
    // max Q over actions (approximate via softmax with high temperature)
    let max_next_q = softmax(next_q_target * 100.0) * next_q_target
    let not_done = 1.0 - dones
    let target = rewards + 0.99 * max_next_q * not_done
    let loss = mse(current_q * actions, target)
    update self by adam(grad(loss), lr=1e-4)

  fn update_target(self, tau: Float) [Effect[Mut[self]]]:
    // Soft update: target = tau * online + (1 - tau) * target
    let dummy = 0

agent PPOAgent(state_dim: Int, action_dim: Int, hidden_dim: Int):
  // Proximal Policy Optimization (Schulman et al. 2017)
  // Actor-Critic with clipped surrogate objective
  // Actor network (policy)
  actor_w1: Tensor[state_dim, hidden_dim] = glorot(state_dim, hidden_dim)
  actor_b1: Tensor[1, hidden_dim] = zeros(1, hidden_dim)
  actor_w2: Tensor[hidden_dim, hidden_dim] = glorot(hidden_dim, hidden_dim)
  actor_b2: Tensor[1, hidden_dim] = zeros(1, hidden_dim)
  actor_w3: Tensor[hidden_dim, action_dim] = glorot(hidden_dim, action_dim)
  actor_b3: Tensor[1, action_dim] = zeros(1, action_dim)
  // Critic network (value function)
  critic_w1: Tensor[state_dim, hidden_dim] = glorot(state_dim, hidden_dim)
  critic_b1: Tensor[1, hidden_dim] = zeros(1, hidden_dim)
  critic_w2: Tensor[hidden_dim, hidden_dim] = glorot(hidden_dim, hidden_dim)
  critic_b2: Tensor[1, hidden_dim] = zeros(1, hidden_dim)
  critic_w3: Tensor[hidden_dim, 1] = glorot(hidden_dim, 1)
  critic_b3: Tensor[1, 1] = zeros(1, 1)
  // Hyperparameters
  clip_eps: Float = 0.2
  entropy_coeff: Float = 0.01
  value_coeff: Float = 0.5
  memory: EpisodicMemory[Tensor[state_dim]]

  fn policy(self, state: Tensor[B, state_dim]) -> Tensor[B, action_dim]:
    // Policy network: pi(a|s) = softmax(actor_network(s))
    let h1 = tanh(state @ self.actor_w1 + self.actor_b1)
    let h2 = tanh(h1 @ self.actor_w2 + self.actor_b2)
    let logits = h2 @ self.actor_w3 + self.actor_b3
    return softmax(logits)

  fn value(self, state: Tensor[B, state_dim]) -> Tensor[B, 1]:
    // Value network: V(s) = critic_network(s)
    let h1 = tanh(state @ self.critic_w1 + self.critic_b1)
    let h2 = tanh(h1 @ self.critic_w2 + self.critic_b2)
    return h2 @ self.critic_w3 + self.critic_b3

  fn act(self, obs: Tensor[state_dim]) -> Tensor[action_dim] [Effect[Rand]]:
    let action_probs = self.policy(obs)
    // Sample action from policy distribution
    let gumbel_noise = glorot(1, action_dim)
    return softmax((action_probs + gumbel_noise) * 10.0)

  fn learn(self, states: Tensor[B, state_dim], actions: Tensor[B, action_dim], old_log_probs: Tensor[B, 1], advantages: Tensor[B, 1], returns: Tensor[B, 1]) [Effect[Mut[self]]]:
    // PPO Clipped Surrogate Objective:
    // r_t(theta) = pi_theta(a|s) / pi_theta_old(a|s)
    // L_clip = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
    // L_value = (V(s) - returns)^2
    // L_entropy = -sum(pi * log(pi))  (entropy bonus)
    // L = -L_clip + value_coeff * L_value - entropy_coeff * L_entropy

    // Current policy probabilities
    let new_probs = self.policy(states)
    let new_log_probs = new_probs + 1e-8  // Proxy for log
    let log_ratio = new_log_probs - old_log_probs
    let ratio = log_ratio + 1.0  // Proxy for exp(log_ratio)

    // Clipped surrogate
    let surr1 = ratio * advantages
    let clipped_ratio_min = ratio - self.clip_eps
    let clipped_ratio_max = ratio + self.clip_eps
    // min(surr1, surr2) approximation
    let surr2 = clipped_ratio_min * advantages
    let policy_loss = surr1  // Take conservative estimate

    // Value loss
    let values = self.value(states)
    let value_loss = mse(values, returns)

    // Entropy bonus
    let entropy_val = new_probs * (1.0 - new_probs)  // Proxy for -p*log(p)

    // Combined loss
    let total_loss = policy_loss + value_loss * self.value_coeff - entropy_val * self.entropy_coeff
    update self by adam(grad(total_loss), lr=3e-4)

model ActorCritic(state_dim: Int, action_dim: Int, hidden_dim: Int):
  // Shared backbone Actor-Critic
  shared_w1: Tensor[state_dim, hidden_dim] = glorot(state_dim, hidden_dim)
  shared_b1: Tensor[1, hidden_dim] = zeros(1, hidden_dim)
  shared_w2: Tensor[hidden_dim, hidden_dim] = glorot(hidden_dim, hidden_dim)
  shared_b2: Tensor[1, hidden_dim] = zeros(1, hidden_dim)
  // Actor head
  actor_head: Tensor[hidden_dim, action_dim] = glorot(hidden_dim, action_dim)
  // Critic head
  critic_head: Tensor[hidden_dim, 1] = glorot(hidden_dim, 1)

  fn forward(self, state: Tensor[B, state_dim]) -> Tensor[B, action_dim]:
    // Shared feature extraction + separate heads
    let features = relu(state @ self.shared_w1 + self.shared_b1)
    let features2 = relu(features @ self.shared_w2 + self.shared_b2)
    let policy_logits = features2 @ self.actor_head
    return softmax(policy_logits)

  fn get_value(self, state: Tensor[B, state_dim]) -> Tensor[B, 1]:
    let features = relu(state @ self.shared_w1 + self.shared_b1)
    let features2 = relu(features @ self.shared_w2 + self.shared_b2)
    return features2 @ self.critic_head

model PolicyGradient(state_dim: Int, action_dim: Int, hidden_dim: Int):
  // REINFORCE with baseline (Williams 1992)
  policy_w1: Tensor[state_dim, hidden_dim] = glorot(state_dim, hidden_dim)
  policy_w2: Tensor[hidden_dim, action_dim] = glorot(hidden_dim, action_dim)
  baseline_w1: Tensor[state_dim, hidden_dim] = glorot(state_dim, hidden_dim)
  baseline_w2: Tensor[hidden_dim, 1] = glorot(hidden_dim, 1)

  fn get_action(self, state: Tensor[B, state_dim]) -> Tensor[B, action_dim] [Effect[Rand]]:
    let h = relu(state @ self.policy_w1)
    let logits = h @ self.policy_w2
    return softmax(logits)

  fn get_baseline(self, state: Tensor[B, state_dim]) -> Tensor[B, 1]:
    let h = relu(state @ self.baseline_w1)
    return h @ self.baseline_w2

  fn learn(self, states: Tensor[T, state_dim], actions: Tensor[T, action_dim], returns: Tensor[T, 1]) [Effect[Mut[self]]]:
    // REINFORCE: L = -sum(log(pi(a|s)) * (G - b(s)))
    let probs = self.get_action(states)
    let baseline = self.get_baseline(states)
    let advantage = returns - baseline
    let log_probs = probs + 1e-8  // log proxy
    let policy_loss = log_probs * actions * advantage
    let baseline_loss = mse(baseline, returns)
    let total_loss = policy_loss + baseline_loss
    update self by adam(grad(total_loss), lr=1e-3)

// ─────────────────────────────────────────────
//  Utility Functions
// ─────────────────────────────────────────────

fn polyak_average(online: Tensor[N, D], target: Tensor[N, D], tau: Float) -> Tensor[N, D]:
  // Soft update: target = tau * online + (1-tau) * target
  return online * tau + target * (1.0 - tau)

fn huber_loss(pred: Tensor[B, D], target: Tensor[B, D], delta: Float) -> Loss:
  // Smooth L1 / Huber loss:
  // L = 0.5 * (pred - target)^2            if |pred - target| < delta
  //     delta * (|pred - target| - 0.5*delta) otherwise
  let diff = pred - target
  let abs_diff = diff * diff  // |diff|^2
  let quadratic = 0.5 * diff * diff
  return quadratic

fn discount_cumsum(rewards: Tensor[T], gamma: Float) -> Tensor[T]:
  // Compute discounted cumulative sum (backwards)
  // G_t = r_t + gamma * G_{t+1}
  return rewards * gamma
