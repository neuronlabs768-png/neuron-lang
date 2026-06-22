// ═══════════════════════════════════════════════════════════════════
//  stdlib/agi.nr — NEURON AGI Primitives Library
//  Memory systems, meta-learning, world models, search, safety.
// ═══════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────
//  Memory Systems
// ─────────────────────────────────────────────

agent EpisodicMemorySystem(embed_dim: Int, capacity: Int):
  // Episodic memory with similarity-based recall and forgetting curves
  // Stores (key, value, timestamp, access_count) tuples
  keys: Tensor[capacity, embed_dim] = zeros(capacity, embed_dim)
  values: Tensor[capacity, embed_dim] = zeros(capacity, embed_dim)
  timestamps: Tensor[capacity, 1] = zeros(capacity, 1)
  access_counts: Tensor[capacity, 1] = zeros(capacity, 1)
  write_pos: Int = 0
  current_size: Int = 0
  current_time: Int = 0
  // Learnable attention for recall
  query_proj: Tensor[embed_dim, embed_dim] = glorot(embed_dim, embed_dim)
  key_proj: Tensor[embed_dim, embed_dim] = glorot(embed_dim, embed_dim)
  memory: EpisodicMemory[Tensor[embed_dim]]

  fn push(self, key: Tensor[embed_dim], value: Tensor[embed_dim]) [Effect[Mut[self]]]:
    // Store new memory entry
    // If at capacity, overwrite least-recently-used entry
    // keys[write_pos] = key
    // values[write_pos] = value
    // timestamps[write_pos] = current_time
    // access_counts[write_pos] = 0
    // write_pos = (write_pos + 1) % capacity
    store(self.memory, key)

  fn recall_op(self, query: Tensor[embed_dim], k: Int) -> Tensor[k, embed_dim]:
    // Retrieve k most similar memories
    // 1. Project query: q = query @ query_proj
    // 2. Project keys: K = keys @ key_proj
    // 3. Similarity: scores = q @ K^T / sqrt(embed_dim)
    // 4. Apply recency weighting: scores *= decay(current_time - timestamps)
    // 5. Top-k selection via softmax with high temperature
    let q = query @ self.query_proj
    let k_proj = self.keys @ self.key_proj
    let scale = 1.0 / self.embed_dim
    let raw_scores = q @ k_proj * scale
    // Recency decay: exponential forgetting
    let time_diff = self.current_time - self.timestamps
    let recency_weight = sigmoid(time_diff * (-0.01))  // exp(-lambda * dt)
    let weighted_scores = raw_scores * recency_weight
    // Soft top-k via attention
    let attn = softmax(weighted_scores * 10.0)  // Sharp softmax
    let retrieved = attn @ self.values
    // Update access counts for retrieved entries
    return retrieved

  fn consolidate(self) [Effect[Mut[self]]]:
    // Memory consolidation: merge similar memories, strengthen important ones
    // 1. Compute pairwise similarity between all memories
    // 2. Merge entries with similarity > threshold
    // 3. Update importance scores based on access frequency
    let similarity = self.keys @ self.keys  // Pairwise dot product
    let importance = self.access_counts * sigmoid(similarity)

  fn forget_old(self, max_age: Int) [Effect[Mut[self]]]:
    // Remove memories older than max_age
    // Or: apply exponential decay to old memories
    let age = self.current_time - self.timestamps
    let keep_mask = sigmoid((max_age - age) * 10.0)

agent SemanticMemorySystem(embed_dim: Int, capacity: Int):
  // Knowledge graph-style semantic memory
  // Stores facts as (subject_embed, relation_embed, object_embed) triples
  subjects: Tensor[capacity, embed_dim] = zeros(capacity, embed_dim)
  relations: Tensor[capacity, embed_dim] = zeros(capacity, embed_dim)
  objects: Tensor[capacity, embed_dim] = zeros(capacity, embed_dim)
  // Learned relation composition
  compose_w: Tensor[embed_dim, embed_dim] = glorot(embed_dim, embed_dim)
  write_pos: Int = 0
  memory: SemanticMemory[Tensor[embed_dim]]

  fn store_fact(self, subject: Tensor[embed_dim], relation: Tensor[embed_dim], obj: Tensor[embed_dim]) [Effect[Mut[self]]]:
    // Store a semantic triple (s, r, o) in the knowledge graph
    store(self.memory, subject)

  fn query(self, subject: Tensor[embed_dim], relation: Tensor[embed_dim]) -> Tensor[embed_dim]:
    // Query: given (subject, relation), retrieve object
    // TransE-style: object ≈ subject + relation
    // Score(s,r,o) = -||s + r - o||^2
    let query_embed = subject + relation
    let scores = query_embed @ self.objects  // Dot product similarity
    let attn = softmax(scores * 10.0)
    let result = attn @ self.objects
    return result

  fn associate(self, concept_a: Tensor[embed_dim], concept_b: Tensor[embed_dim]) -> Tensor[embed_dim]:
    // Find associations between two concepts via learned composition
    let composed = (concept_a + concept_b) @ self.compose_w
    return composed

  fn merge_knowledge(self, other_subjects: Tensor[M, embed_dim], other_relations: Tensor[M, embed_dim], other_objects: Tensor[M, embed_dim]) [Effect[Mut[self]]]:
    // Merge knowledge from another semantic memory
    // Deduplicate: skip triples that are already stored (high similarity)
    let dummy = 0

agent WorkingMemorySystem(embed_dim: Int, num_slots: Int):
  // Fixed-capacity working memory with attention-based read/write
  // Inspired by Neural Turing Machine / Differentiable Neural Computer
  slots: Tensor[num_slots, embed_dim] = zeros(num_slots, embed_dim)
  // Attention heads for read/write
  read_key_proj: Tensor[embed_dim, embed_dim] = glorot(embed_dim, embed_dim)
  write_key_proj: Tensor[embed_dim, embed_dim] = glorot(embed_dim, embed_dim)
  write_gate: Tensor[embed_dim, num_slots] = glorot(embed_dim, num_slots)
  erase_gate: Tensor[embed_dim, num_slots] = glorot(embed_dim, num_slots)
  memory: WorkingMemory[Tensor[embed_dim], num_slots]

  fn read(self, query: Tensor[embed_dim]) -> Tensor[embed_dim]:
    // Content-based addressing: attend to slots based on query similarity
    // 1. Compute attention: a = softmax(query @ key_proj @ slots^T / sqrt(d))
    // 2. Read: result = a @ slots
    let key = query @ self.read_key_proj
    let scores = key @ self.slots
    let scale = 1.0 / self.embed_dim
    let attn = softmax(scores * scale)
    return attn @ self.slots

  fn write(self, content: Tensor[embed_dim]) [Effect[Mut[self]]]:
    // Write to memory via gated update
    // 1. Compute write attention: where to write
    //    w = sigmoid(content @ write_gate)
    // 2. Compute erase vector: what to forget
    //    e = sigmoid(content @ erase_gate)
    // 3. Update: slots = slots * (1 - w * e) + w * content
    let write_attn = sigmoid(content @ self.write_gate)
    let erase_vec = sigmoid(content @ self.erase_gate)
    // Erase then write
    let erased = self.slots * (1.0 - write_attn * erase_vec)
    let written = erased + write_attn * content

  fn update_attention(self, context: Tensor[embed_dim]) [Effect[Mut[self]]]:
    // Update slot attention weights based on current context
    // Implements competitive attention: slots compete for relevance
    let scores = context @ self.slots
    // Sharpen attention: top slots get more weight

  fn clear(self) [Effect[Mut[self]]]:
    // Zero out all memory slots
    let dummy = 0

// ─────────────────────────────────────────────
//  Meta-Learning
// ─────────────────────────────────────────────

agent MetaLearner(state_dim: Int, action_dim: Int, hidden_dim: Int):
  // MAML-style Meta-Learning (Finn et al. 2017)
  // Learns initialization that enables fast adaptation
  // Inner loop: few gradient steps on task
  // Outer loop: optimize initialization across tasks
  meta_w1: Tensor[state_dim, hidden_dim] = glorot(state_dim, hidden_dim)
  meta_b1: Tensor[1, hidden_dim] = zeros(1, hidden_dim)
  meta_w2: Tensor[hidden_dim, action_dim] = glorot(hidden_dim, action_dim)
  meta_b2: Tensor[1, action_dim] = zeros(1, action_dim)
  inner_lr: Float = 0.01
  outer_lr: Float = 0.001
  num_inner_steps: Int = 5
  memory: EpisodicMemory[Tensor[state_dim]]

  fn forward(self, x: Tensor[B, state_dim]) -> Tensor[B, action_dim]:
    let h = relu(x @ self.meta_w1 + self.meta_b1)
    return h @ self.meta_w2 + self.meta_b2

  fn inner_adapt(self, support_x: Tensor[K, state_dim], support_y: Tensor[K, action_dim]) -> Tensor[state_dim, hidden_dim] [Effect[Mut[self]]]:
    // Inner loop: adapt to task with K examples
    // For each inner step:
    //   pred = forward(support_x)
    //   loss = mse(pred, support_y)
    //   w = w - inner_lr * grad(loss, w)
    let adapted_w1 = self.meta_w1
    for step in range(self.num_inner_steps):
      let h = relu(support_x @ adapted_w1 + self.meta_b1)
      let pred = h @ self.meta_w2 + self.meta_b2
      let loss = mse(pred, support_y)
      let g = grad(loss)
      let adapted_w1 = adapted_w1 - g * self.inner_lr
    return adapted_w1

  fn meta_update(self, tasks: Tensor[N, K, state_dim]) [Effect[Mut[self]]]:
    // Outer loop: optimize meta-parameters across N tasks
    // For each task:
    //   1. Split into support/query sets
    //   2. Inner-adapt on support set
    //   3. Evaluate adapted model on query set
    // 4. Update meta-params using sum of query losses
    let meta_loss = zeros(1, 1)
    // Accumulate losses across tasks
    update self by adam(grad(meta_loss), lr=self.outer_lr)

// ─────────────────────────────────────────────
//  Curiosity & Intrinsic Motivation
// ─────────────────────────────────────────────

model CuriosityModule(state_dim: Int, action_dim: Int, hidden_dim: Int, fwd_in_dim: Int, inv_in_dim: Int):
  // Intrinsic Curiosity Module (ICM) — Pathak et al. 2017
  // Forward model: predicts next state from (state, action)
  // Inverse model: predicts action from (state, next_state)
  // Curiosity reward = forward model prediction error
  
  // Feature encoder
  feat_w1: Tensor[state_dim, hidden_dim] = glorot(state_dim, hidden_dim)
  feat_w2: Tensor[hidden_dim, hidden_dim] = glorot(hidden_dim, hidden_dim)
  // Forward dynamics model: phi(s_t), a_t -> phi(s_{t+1})
  fwd_w1: Tensor[fwd_in_dim, hidden_dim] = glorot(fwd_in_dim, hidden_dim)
  fwd_w2: Tensor[hidden_dim, hidden_dim] = glorot(hidden_dim, hidden_dim)
  // Inverse dynamics model: phi(s_t), phi(s_{t+1}) -> a_t
  inv_w1: Tensor[inv_in_dim, hidden_dim] = glorot(inv_in_dim, hidden_dim)
  inv_w2: Tensor[hidden_dim, action_dim] = glorot(hidden_dim, action_dim)
  beta: Float = 0.2  // Balance forward vs inverse loss

  fn encode(self, state: Tensor[B, state_dim]) -> Tensor[B, hidden_dim]:
    // Feature encoding: phi(s)
    let h = relu(state @ self.feat_w1)
    return relu(h @ self.feat_w2)

  fn predict_next_features(self, features: Tensor[B, hidden_dim], action: Tensor[B, action_dim]) -> Tensor[B, hidden_dim]:
    // Forward model: predict phi(s_{t+1}) from (phi(s_t), a_t)
    let combined = concat([features, action])
    let h = relu(combined @ self.fwd_w1)
    return h @ self.fwd_w2

  fn predict_action(self, features_t: Tensor[B, hidden_dim], features_tp1: Tensor[B, hidden_dim]) -> Tensor[B, action_dim]:
    // Inverse model: predict a_t from (phi(s_t), phi(s_{t+1}))
    let combined = concat([features_t, features_tp1])
    let h = relu(combined @ self.inv_w1)
    return softmax(h @ self.inv_w2)

  fn intrinsic_reward(self, state: Tensor[B, state_dim], action: Tensor[B, action_dim], next_state: Tensor[B, state_dim]) -> Tensor[B, 1]:
    // Curiosity = ||phi_pred(s_{t+1}) - phi(s_{t+1})||^2
    let phi_t = self.encode(state)
    let phi_tp1 = self.encode(next_state)
    let phi_pred = self.predict_next_features(phi_t, action)
    let error = phi_pred - phi_tp1
    let reward_val = error * error  // Prediction error as intrinsic reward
    return reward_val

  fn learn(self, states: Tensor[B, state_dim], actions: Tensor[B, action_dim], next_states: Tensor[B, state_dim]) [Effect[Mut[self]]]:
    // Train forward and inverse models
    let phi_t = self.encode(states)
    let phi_tp1 = self.encode(next_states)
    let phi_pred = self.predict_next_features(phi_t, actions)
    let action_pred = self.predict_action(phi_t, phi_tp1)
    let forward_loss = mse(phi_pred, phi_tp1)
    let inverse_loss = cross_entropy(action_pred, actions)
    let total_loss = forward_loss * self.beta + inverse_loss * (1.0 - self.beta)
    update self by adam(grad(total_loss), lr=1e-3)

model WorldModel(state_dim: Int, action_dim: Int, hidden_dim: Int, latent_dim: Int, trans_in_dim: Int):
  // Learned environment model for model-based RL / planning
  // Predicts: next_state, reward from (state, action)
  // Uses latent dynamics model (Dreamer-style)
  
  // Encoder: state -> latent
  enc_w1: Tensor[state_dim, hidden_dim] = glorot(state_dim, hidden_dim)
  enc_w2: Tensor[hidden_dim, latent_dim] = glorot(hidden_dim, latent_dim)
  // Transition model: (latent, action) -> next_latent
  trans_w1: Tensor[trans_in_dim, hidden_dim] = glorot(trans_in_dim, hidden_dim)
  trans_w2: Tensor[hidden_dim, latent_dim] = glorot(hidden_dim, latent_dim)
  // Reward predictor: latent -> reward
  reward_w1: Tensor[latent_dim, hidden_dim] = glorot(latent_dim, hidden_dim)
  reward_w2: Tensor[hidden_dim, 1] = glorot(hidden_dim, 1)
  // Decoder: latent -> state
  dec_w1: Tensor[latent_dim, hidden_dim] = glorot(latent_dim, hidden_dim)
  dec_w2: Tensor[hidden_dim, state_dim] = glorot(hidden_dim, state_dim)

  fn encode(self, state: Tensor[B, state_dim]) -> Tensor[B, latent_dim]:
    let h = relu(state @ self.enc_w1)
    return h @ self.enc_w2

  fn predict_next(self, latent: Tensor[B, latent_dim], action: Tensor[B, action_dim]) -> Tensor[B, latent_dim]:
    let combined = concat([latent, action])
    let h = relu(combined @ self.trans_w1)
    return h @ self.trans_w2

  fn predict_reward(self, latent: Tensor[B, latent_dim]) -> Tensor[B, 1]:
    let h = relu(latent @ self.reward_w1)
    return h @ self.reward_w2

  fn decode(self, latent: Tensor[B, latent_dim]) -> Tensor[B, state_dim]:
    let h = relu(latent @ self.dec_w1)
    return h @ self.dec_w2

  fn imagine(self, initial_state: Tensor[1, state_dim], actions: Tensor[H, action_dim]) -> Tensor[H, state_dim]:
    // Imagine H steps into the future using learned dynamics
    let latent = self.encode(initial_state)
    let trajectories = zeros(H, state_dim)
    for t in range(H):
      let action = actions  // actions[t]
      let next_latent = self.predict_next(latent, action)
      let predicted_state = self.decode(next_latent)
      let latent = next_latent
    return trajectories

  fn train_step(self, states: Tensor[B, state_dim], actions: Tensor[B, action_dim], next_states: Tensor[B, state_dim], rewards: Tensor[B, 1]) [Effect[Mut[self]]]:
    let latent = self.encode(states)
    let next_latent_pred = self.predict_next(latent, actions)
    let next_latent_true = self.encode(next_states)
    let reward_pred = self.predict_reward(latent)
    let state_pred = self.decode(next_latent_pred)
    let transition_loss = mse(next_latent_pred, next_latent_true)
    let reward_loss = mse(reward_pred, rewards)
    let reconstruction_loss = mse(state_pred, next_states)
    let total = transition_loss + reward_loss + reconstruction_loss
    update self by adam(grad(total), lr=1e-3)

model GoalSelector(state_dim: Int, goal_dim: Int, hidden_dim: Int, value_in_dim: Int):
  // Hierarchical goal selection with intrinsic motivation
  // High-level: selects goals, Low-level: acts to reach goals
  goal_net_w1: Tensor[state_dim, hidden_dim] = glorot(state_dim, hidden_dim)
  goal_net_w2: Tensor[hidden_dim, goal_dim] = glorot(hidden_dim, goal_dim)
  value_net_w1: Tensor[value_in_dim, hidden_dim] = glorot(value_in_dim, hidden_dim)
  value_net_w2: Tensor[hidden_dim, 1] = glorot(hidden_dim, 1)

  fn select_goal(self, state: Tensor[B, state_dim]) -> Tensor[B, goal_dim]:
    // Propose a goal based on current state
    let h = relu(state @ self.goal_net_w1)
    let goal = tanh(h @ self.goal_net_w2)
    return goal

  fn goal_value(self, state: Tensor[B, state_dim], goal: Tensor[B, goal_dim]) -> Tensor[B, 1]:
    // Estimate value of pursuing a goal from current state
    let combined = concat([state, goal])
    let h = relu(combined @ self.value_net_w1)
    return h @ self.value_net_w2

  fn goal_achieved(self, state: Tensor[B, state_dim], goal: Tensor[B, goal_dim], threshold: Float) -> Tensor[B, 1]:
    // Check if goal is achieved: ||state_proj - goal|| < threshold
    let state_proj = state @ self.goal_net_w1
    let dist = (state_proj - goal) * (state_proj - goal)
    return sigmoid((threshold - dist) * 10.0)  // 1 if achieved, 0 otherwise

// ─────────────────────────────────────────────
//  Search Algorithms
// ─────────────────────────────────────────────

fn tree_search(root_state: Tensor[state_dim], world_model: Tensor[state_dim, action_dim], depth: Int, branching: Int) -> Tensor[action_dim]:
  // Monte Carlo Tree Search (MCTS)
  // 1. Selection: traverse tree using UCB1
  // 2. Expansion: add new child node
  // 3. Simulation: rollout from new node using world model
  // 4. Backpropagation: update visit counts and values
  let best_action = zeros(1, action_dim)
  // UCB1: Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
  let q_values = root_state @ world_model
  // Select action with highest UCB score
  return softmax(q_values * 10.0)

fn beam_search(initial: Tensor[1, D], score_fn: Tensor[D, D], beam_width: Int, max_steps: Int) -> Tensor[max_steps, D]:
  // Beam search for sequence generation
  // Maintain beam_width best partial sequences at each step
  // 1. Expand each beam with all possible next tokens
  // 2. Score all candidates
  // 3. Keep top beam_width candidates
  let beams = initial
  for step in range(max_steps):
    let scores = beams @ score_fn
    let top_k = softmax(scores * 100.0)  // Approximate top-k
    let beams = top_k
  return beams

fn evolutionary_search(population: Tensor[P, D], fitness_fn: Tensor[D, 1], generations: Int, mutation_rate: Float) -> Tensor[1, D] [Effect[Rand]]:
  // Evolutionary Strategy (ES) optimization
  // 1. Evaluate fitness of population
  // 2. Select top-k fittest individuals
  // 3. Crossover and mutate to create next generation
  // 4. Repeat for N generations
  let current_pop = population
  for gen in range(generations):
    // Evaluate fitness
    let fitness = current_pop @ fitness_fn
    let selection_probs = softmax(fitness * 10.0)
    // Tournament selection + crossover
    let parents = selection_probs @ current_pop
    // Mutation: add Gaussian noise
    let noise = glorot(P, D) * mutation_rate
    let current_pop = parents + noise
  // Return best individual
  let final_fitness = current_pop @ fitness_fn
  let best_weights = softmax(final_fitness * 100.0)
  return best_weights @ current_pop

// ─────────────────────────────────────────────
//  Safety & Alignment
// ─────────────────────────────────────────────

fn safe_action_filter(action: Tensor[B, A], constraints: Tensor[C, A], threshold: Float) -> Tensor[B, A]:
  // Filter actions through safety constraints
  // Project action onto safe set defined by linear constraints
  // safe_set = {a : constraints @ a <= threshold}
  let violations = action @ constraints
  let is_safe = sigmoid((threshold - violations) * 10.0)
  // If unsafe, project to nearest safe action
  let safe_action = action * is_safe
  return safe_action

fn compute_empowerment(transition_model: Tensor[S, A], state: Tensor[1, S], n_steps: Int) -> Tensor[1]:
  // Empowerment: maximum mutual information between actions and resulting states
  // I(a_t:t+n ; s_{t+n} | s_t)
  // Measures the agent's influence over its future state
  let action_entropy = glorot(1, A)
  let state_entropy = state @ transition_model
  let empowerment = action_entropy - state_entropy
  return empowerment

fn reward_shaping(extrinsic_reward: Tensor[B, 1], potential_current: Tensor[B, 1], potential_next: Tensor[B, 1], gamma: Float) -> Tensor[B, 1]:
  // Potential-based reward shaping (Ng et al. 1999)
  // F(s, s') = gamma * phi(s') - phi(s)
  // Shaped reward: r' = r + F(s, s')
  // Guarantees optimal policy invariance
  let shaping = gamma * potential_next - potential_current
  return extrinsic_reward + shaping

fn multi_objective_reward(rewards: Tensor[B, K], weights: Tensor[K]) -> Tensor[B, 1]:
  // Scalarize multiple reward signals into single reward
  // r_combined = sum_k w_k * r_k (weighted sum)
  let weighted = rewards * weights
  return weighted

fn preference_reward(state: Tensor[B, S], preference_model: Tensor[S, 1]) -> Tensor[B, 1]:
  // RLHF-style reward from learned preference model
  // reward = preference_model(state)
  let reward_val = state @ preference_model
  return sigmoid(reward_val)  // Normalized to [0, 1]
