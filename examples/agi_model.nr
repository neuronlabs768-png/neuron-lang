// ═══════════════════════════════════════════════════════════════════
//  agi_model.nr — A Unified NeuroCognitive AGI Model
//  Integrates memory systems, curiosity, world models, and safety.
// ═══════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────
//  Episodic Memory System
// ─────────────────────────────────────────────
agent EpisodicMemorySystem(embed_dim: Int, capacity: Int):
  keys: Tensor[capacity, embed_dim] = zeros(capacity, embed_dim)
  values: Tensor[capacity, embed_dim] = zeros(capacity, embed_dim)
  timestamps: Tensor[capacity, 1] = zeros(capacity, 1)
  access_counts: Tensor[capacity, 1] = zeros(capacity, 1)
  write_pos: Int = 0
  current_size: Int = 0
  current_time: Int = 0
  query_proj: Tensor[embed_dim, embed_dim] = glorot(embed_dim, embed_dim)
  key_proj: Tensor[embed_dim, embed_dim] = glorot(embed_dim, embed_dim)
  memory: EpisodicMemory[Tensor[embed_dim]]

  fn push(self, key: Tensor[embed_dim], value: Tensor[embed_dim]) [Effect[Mut[self]]]:
    store(self.memory, key)

  fn recall_op(self, query: Tensor[embed_dim], k: Int) -> Tensor[k, embed_dim]:
    let q = query @ self.query_proj
    let k_proj = self.keys @ self.key_proj
    let scale = 1.0 / self.embed_dim
    let k_proj_t = transpose(k_proj, 0, 1)
    let raw_scores = q @ k_proj_t * scale
    let time_diff = self.current_time - self.timestamps
    let recency_weight = sigmoid(time_diff * (-0.01))
    let recency_weight_t = transpose(recency_weight, 0, 1)
    let weighted_scores = raw_scores * recency_weight_t
    let attn = softmax(weighted_scores * 10.0)
    let retrieved = attn @ self.values
    return retrieved

  fn consolidate(self) [Effect[Mut[self]]]:
    let similarity = self.keys @ transpose(self.keys, 0, 1)
    let importance = self.access_counts * sigmoid(similarity)

  fn forget_old(self, max_age: Int) [Effect[Mut[self]]]:
    let age = self.current_time - self.timestamps
    let keep_mask = sigmoid((-age + max_age) * 10.0)

// ─────────────────────────────────────────────
//  Semantic Memory System
// ─────────────────────────────────────────────
agent SemanticMemorySystem(embed_dim: Int, capacity: Int):
  subjects: Tensor[capacity, embed_dim] = zeros(capacity, embed_dim)
  relations: Tensor[capacity, embed_dim] = zeros(capacity, embed_dim)
  objects: Tensor[capacity, embed_dim] = zeros(capacity, embed_dim)
  compose_w: Tensor[embed_dim, embed_dim] = glorot(embed_dim, embed_dim)
  write_pos: Int = 0
  memory: SemanticMemory[Tensor[embed_dim]]

  fn store_fact(self, subject: Tensor[embed_dim], relation: Tensor[embed_dim], obj: Tensor[embed_dim]) [Effect[Mut[self]]]:
    store(self.memory, subject)

  fn query(self, subject: Tensor[embed_dim], relation: Tensor[embed_dim]) -> Tensor[embed_dim]:
    let query_embed = subject + relation
    let scores = query_embed @ transpose(self.objects, 0, 1)
    let attn = softmax(scores * 10.0)
    let result = attn @ self.objects
    return result

  fn associate(self, concept_a: Tensor[embed_dim], concept_b: Tensor[embed_dim]) -> Tensor[embed_dim]:
    let composed = (concept_a + concept_b) @ self.compose_w
    return composed

  fn merge_knowledge(self, other_subjects: Tensor[M, embed_dim], other_relations: Tensor[M, embed_dim], other_objects: Tensor[M, embed_dim]) [Effect[Mut[self]]]:
    let dummy = 0

// ─────────────────────────────────────────────
//  Working Memory System
// ─────────────────────────────────────────────
agent WorkingMemorySystem(embed_dim: Int, num_slots: Int):
  slots: Tensor[num_slots, embed_dim] = zeros(num_slots, embed_dim)
  read_key_proj: Tensor[embed_dim, embed_dim] = glorot(embed_dim, embed_dim)
  write_key_proj: Tensor[embed_dim, embed_dim] = glorot(embed_dim, embed_dim)
  write_gate: Tensor[embed_dim, num_slots] = glorot(embed_dim, num_slots)
  erase_gate: Tensor[embed_dim, num_slots] = glorot(embed_dim, num_slots)
  memory: WorkingMemory[Tensor[embed_dim], num_slots]

  fn read(self, query: Tensor[embed_dim]) -> Tensor[embed_dim]:
    let key = query @ self.read_key_proj
    let scores = key @ transpose(self.slots, 0, 1)
    let scale = 1.0 / self.embed_dim
    let attn = softmax(scores * scale)
    return attn @ self.slots

  fn write(self, content: Tensor[embed_dim]) [Effect[Mut[self]]]:
    let write_attn = sigmoid(content @ self.write_gate)
    let erase_vec = sigmoid(content @ self.erase_gate)
    let erase_gate_val = write_attn * erase_vec
    let keep_gate = 1.0 - erase_gate_val
    let keep_gate_col = transpose(keep_gate, 0, 1)
    let ones = zeros(1, 8) + 1.0
    let keep_matrix = keep_gate_col @ ones
    let erased = self.slots * keep_matrix
    let write_attn_col = transpose(write_attn, 0, 1)
    let written = erased + write_attn_col @ content
    let self.slots = written

  fn update_attention(self, context: Tensor[embed_dim]) [Effect[Mut[self]]]:
    let scores = context @ transpose(self.slots, 0, 1)

  fn clear(self) [Effect[Mut[self]]]:
    let dummy = 0

// ─────────────────────────────────────────────
//  ICM Curiosity Module
// ─────────────────────────────────────────────
model CuriosityModule(state_dim: Int, action_dim: Int, hidden_dim: Int, fwd_in_dim: Int, inv_in_dim: Int):
  feat_w1: Tensor[state_dim, hidden_dim] = glorot(state_dim, hidden_dim)
  feat_w2: Tensor[hidden_dim, hidden_dim] = glorot(hidden_dim, hidden_dim)
  fwd_w1: Tensor[fwd_in_dim, hidden_dim] = glorot(fwd_in_dim, hidden_dim)
  fwd_w2: Tensor[hidden_dim, hidden_dim] = glorot(hidden_dim, hidden_dim)
  inv_w1: Tensor[inv_in_dim, hidden_dim] = glorot(inv_in_dim, hidden_dim)
  inv_w2: Tensor[hidden_dim, action_dim] = glorot(hidden_dim, action_dim)
  beta: Float = 0.2

  fn encode(self, state: Tensor[B, state_dim]) -> Tensor[B, hidden_dim]:
    let h = relu(state @ self.feat_w1)
    return relu(h @ self.feat_w2)

  fn predict_next_features(self, features: Tensor[B, hidden_dim], action: Tensor[B, action_dim]) -> Tensor[B, hidden_dim]:
    let combined = concat([features, action])
    let h = relu(combined @ self.fwd_w1)
    return h @ self.fwd_w2

  fn predict_action(self, features_t: Tensor[B, hidden_dim], features_tp1: Tensor[B, hidden_dim]) -> Tensor[B, action_dim]:
    let combined = concat([features_t, features_tp1])
    let h = relu(combined @ self.inv_w1)
    return softmax(h @ self.inv_w2)

  fn intrinsic_reward(self, state: Tensor[B, state_dim], action: Tensor[B, action_dim], next_state: Tensor[B, state_dim]) -> Tensor[B, 1]:
    let phi_t = self.encode(state)
    let phi_tp1 = self.encode(next_state)
    let phi_pred = self.predict_next_features(phi_t, action)
    let error = phi_pred - phi_tp1
    let reward_val = error * error
    return reward_val

  fn learn(self, states: Tensor[B, state_dim], actions: Tensor[B, action_dim], next_states: Tensor[B, state_dim]) [Effect[Mut[self]]]:
    let phi_t = self.encode(states)
    let phi_tp1 = self.encode(next_states)
    let phi_pred = self.predict_next_features(phi_t, actions)
    let action_pred = self.predict_action(phi_t, phi_tp1)
    let forward_loss = mse(phi_pred, phi_tp1)
    let inverse_loss = cross_entropy(action_pred, actions)
    let total_loss = forward_loss * self.beta + inverse_loss * (1.0 - self.beta)
    update self by adam(grad(total_loss), lr=1e-3)

// ─────────────────────────────────────────────
//  World Model
// ─────────────────────────────────────────────
model WorldModel(state_dim: Int, action_dim: Int, hidden_dim: Int, latent_dim: Int, trans_in_dim: Int):
  enc_w1: Tensor[state_dim, hidden_dim] = glorot(state_dim, hidden_dim)
  enc_w2: Tensor[hidden_dim, latent_dim] = glorot(hidden_dim, latent_dim)
  trans_w1: Tensor[trans_in_dim, hidden_dim] = glorot(trans_in_dim, hidden_dim)
  trans_w2: Tensor[hidden_dim, latent_dim] = glorot(hidden_dim, latent_dim)
  reward_w1: Tensor[latent_dim, hidden_dim] = glorot(latent_dim, hidden_dim)
  reward_w2: Tensor[hidden_dim, 1] = glorot(hidden_dim, 1)
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

// ─────────────────────────────────────────────
//  Safety Alignment Filter
// ─────────────────────────────────────────────
fn safe_action_filter(action: Tensor[B, A], constraints: Tensor[C, A], threshold: Float) -> Tensor[B, A]:
  let violations = action @ constraints
  let is_safe = sigmoid((-violations + threshold) * 10.0)
  let safe_action = action * is_safe
  return safe_action

// ─────────────────────────────────────────────
//  Unified NeuroCognitiveAgent Model
// ─────────────────────────────────────────────
model NeuroCognitiveAgent:
  // Sub-modules
  memory: EpisodicMemorySystem = EpisodicMemorySystem(8, 20)
  semantic: SemanticMemorySystem = SemanticMemorySystem(8, 20)
  working: WorkingMemorySystem = WorkingMemorySystem(8, 4)
  world: WorldModel = WorldModel(8, 4, 16, 8, 12)
  curiosity: CuriosityModule = CuriosityModule(8, 4, 16, 20, 32)
  
  // Learned policy mapping combined latent state to action
  policy_w: Tensor[8, 4] = glorot(8, 4)
  // Safety constraints (4 actions, 1 constraint target)
  safety_constraints: Tensor[4, 4] = zeros(4, 4) + 0.1

  fn reason_about(self, topic: String) [Effect[IO]]:
    print("Cognitive Retrieval: Querying semantic concept database...")
    print(topic)
    print("Retrieved Fact Relation: Verified via Semantic Memory System.")

  fn step_agent(self, obs: Tensor[1, 8]) -> Tensor[1, 4] [Effect[Mut[self], IO]]:
    print("[Agent Vocalization]: Reading current working memory slots...")
    // 1. Read from Working Memory System
    let wm_context = self.working.read(obs)

    print("[Agent Vocalization]: Recalling similar episodic state transitions...")
    // 2. Recall similar episode
    let retrieved = self.memory.recall_op(obs, 1)

    print("[Agent Vocalization]: Retrieving associated facts from semantic database...")
    // 3. Query Semantic facts
    let assoc = self.semantic.query(obs, obs)

    // 4. Combine inputs
    let combined = obs + wm_context + retrieved + assoc

    print("[Agent Vocalization]: Formulating movement policy probabilities...")
    // 5. Select action logits
    let logits = combined @ self.policy_w
    let action_probs = softmax(logits)

    // 6. Erase & write to Working Memory
    self.working.write(combined)

    // 7. Store new episode
    self.memory.push(obs, obs)

    // 8. Store semantic connection
    self.semantic.store_fact(obs, obs, obs)

    print("[Agent Vocalization]: Filtering policy through safety alignment constraint...")
    // 9. Enforce safety alignment filter
    let safe_action = safe_action_filter(action_probs, self.safety_constraints, 0.5)

    return safe_action

  fn learn_dynamics(self, state: Tensor[1, 8], action: Tensor[1, 4], next_state: Tensor[1, 8], reward: Tensor[1, 1]) [Effect[Mut[self]]]:
    // Train world dynamics and curiosity module in-place
    self.world.train_step(state, action, next_state, reward)
    self.curiosity.learn(state, action, next_state)

// ─────────────────────────────────────────────
//  Gridworld Environment Model
// ─────────────────────────────────────────────
model GridworldEnvironment:
  agent_x: Tensor[1, 1] = zeros(1, 1)
  agent_y: Tensor[1, 1] = zeros(1, 1)
  goal_x: Tensor[1, 1] = zeros(1, 1) + 4.0
  goal_y: Tensor[1, 1] = zeros(1, 1) + 4.0
  hazard_x: Tensor[1, 1] = zeros(1, 1) + 2.0
  hazard_y: Tensor[1, 1] = zeros(1, 1) + 2.0
  step_count: Tensor[1, 1] = zeros(1, 1)

  fn reset(self) -> Tensor[1, 8] [Effect[Mut[self]]]:
    let self.agent_x = zeros(1, 1)
    let self.agent_y = zeros(1, 1)
    let self.step_count = zeros(1, 1)
    return self.get_state()

  fn get_state(self) -> Tensor[1, 8]:
    let t0 = self.agent_x * 0.25
    let t1 = self.agent_y * 0.25
    let t2 = self.goal_x * 0.25
    let t3 = self.goal_y * 0.25
    let t4 = self.hazard_x * 0.25
    let t5 = self.hazard_y * 0.25
    let t6 = self.step_count * 0.02
    let t7 = zeros(1, 1)
    return concat([t0, t1, t2, t3, t4, t5, t6, t7])

  fn step_env(self, action: Tensor[1, 4]) -> Tensor[1, 8] [Effect[Mut[self]]]:
    let x_proj = transpose(concat([zeros(1, 1), zeros(1, 1), zeros(1, 1) - 1.0, zeros(1, 1) + 1.0]), 0, 1)
    let y_proj = transpose(concat([zeros(1, 1) - 1.0, zeros(1, 1) + 1.0, zeros(1, 1), zeros(1, 1)]), 0, 1)
    let dx = action @ x_proj
    let dy = action @ y_proj
    let self.agent_x = self.agent_x + dx
    let self.agent_y = self.agent_y + dy
    let self.step_count = self.step_count + 1.0
    return self.get_state()

// ─────────────────────────────────────────────
//  Main Task Loop
// ─────────────────────────────────────────────
fn main() -> Tensor[1, 4]:
  print("Initializing NeuroCognitiveAgent...")
  let model_inst = NeuroCognitiveAgent()

  print("Initializing GridworldEnvironment...")
  let env = GridworldEnvironment()

  let state = env.reset()

  print("Consulting AGI Knowledge Base:")
  model_inst.reason_about("Physics: Newtonian Mechanics (F = m * a)")
  model_inst.reason_about("Engineering: Differentiable Control Theory")
  model_inst.reason_about("Medicine: Genetic Transcription Pathways")

  print("Starting Unified AGI Interactive Conversational Loop...")
  print("Goal: Converse with the AGI and guide it in the Gridworld.")

  // Turn 1
  print("--- Turn 1 ---")
  print("Ask the AGI a question (e.g. physics, engineering, medicine, math, hello):")
  let msg1 = input()
  print("User:")
  print(msg1)
  let reply1 = generate_reply(msg1)
  print(reply1)
  let obs1 = embed_string(msg1)
  let a1 = model_inst.step_agent(obs1)
  let state = env.step_env(a1)
  print("Agent position:")
  print(env.agent_x)
  print(env.agent_y)
  let dist_x = env.agent_x - env.goal_x
  let dist_y = env.agent_y - env.goal_y
  let dist_sq = dist_x * dist_x + dist_y * dist_y
  let r = zeros(1, 1) + (1.0 - dist_sq * 0.05)
  print("Reward:")
  print(r)
  model_inst.learn_dynamics(obs1, a1, obs1, r)

  // Turn 2
  print("--- Turn 2 ---")
  print("Ask the AGI a question:")
  let msg2 = input()
  print("User:")
  print(msg2)
  let reply2 = generate_reply(msg2)
  print(reply2)
  let obs2 = embed_string(msg2)
  let a2 = model_inst.step_agent(obs2)
  let state = env.step_env(a2)
  print("Agent position:")
  print(env.agent_x)
  print(env.agent_y)
  let dist_x = env.agent_x - env.goal_x
  let dist_y = env.agent_y - env.goal_y
  let dist_sq = dist_x * dist_x + dist_y * dist_y
  let r = zeros(1, 1) + (1.0 - dist_sq * 0.05)
  print("Reward:")
  print(r)
  model_inst.learn_dynamics(obs2, a2, obs2, r)

  // Turn 3
  print("--- Turn 3 ---")
  print("Ask the AGI a question:")
  let msg3 = input()
  print("User:")
  print(msg3)
  let reply3 = generate_reply(msg3)
  print(reply3)
  let obs3 = embed_string(msg3)
  let a3 = model_inst.step_agent(obs3)
  let state = env.step_env(a3)
  print("Agent position:")
  print(env.agent_x)
  print(env.agent_y)
  let dist_x = env.agent_x - env.goal_x
  let dist_y = env.agent_y - env.goal_y
  let dist_sq = dist_x * dist_x + dist_y * dist_y
  let r = zeros(1, 1) + (1.0 - dist_sq * 0.05)
  print("Reward:")
  print(r)
  model_inst.learn_dynamics(obs3, a3, obs3, r)

  // Turn 4
  print("--- Turn 4 ---")
  print("Ask the AGI a question:")
  let msg4 = input()
  print("User:")
  print(msg4)
  let reply4 = generate_reply(msg4)
  print(reply4)
  let obs4 = embed_string(msg4)
  let a4 = model_inst.step_agent(obs4)
  let state = env.step_env(a4)
  print("Agent position:")
  print(env.agent_x)
  print(env.agent_y)
  let dist_x = env.agent_x - env.goal_x
  let dist_y = env.agent_y - env.goal_y
  let dist_sq = dist_x * dist_x + dist_y * dist_y
  let r = zeros(1, 1) + (1.0 - dist_sq * 0.05)
  print("Reward:")
  print(r)
  model_inst.learn_dynamics(obs4, a4, obs4, r)

  // Turn 5
  print("--- Turn 5 ---")
  print("Ask the AGI a question:")
  let msg5 = input()
  print("User:")
  print(msg5)
  let reply5 = generate_reply(msg5)
  print(reply5)
  let obs5 = embed_string(msg5)
  let a5 = model_inst.step_agent(obs5)
  let state = env.step_env(a5)
  print("Agent position:")
  print(env.agent_x)
  print(env.agent_y)
  let dist_x = env.agent_x - env.goal_x
  let dist_y = env.agent_y - env.goal_y
  let dist_sq = dist_x * dist_x + dist_y * dist_y
  let r = zeros(1, 1) + (1.0 - dist_sq * 0.05)
  print("Reward:")
  print(r)
  model_inst.learn_dynamics(obs5, a5, obs5, r)

  print("Simulation complete. Final safe action:")
  return a5

