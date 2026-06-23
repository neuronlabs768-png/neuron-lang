// ═══════════════════════════════════════════════════════════════
//  NEURON Advanced Demo: Causal Reinforcement Learning (CRL)
//  Distinguishes observation from action intervention in SCMs.
// ═══════════════════════════════════════════════════════════════

causal model MarketDynamics:
  variables: [state, sentiment, action, payoff]
  state -> sentiment
  sentiment -> action
  action -> payoff
  state -> payoff

fn evaluate_observations(model: MarketDynamics) -> Causal[Float, observed]:
  // Bayesian conditioning: what usually happens when the action is observed
  let obs: Causal[Float, observed] = model.observe(action = 1.0)
  return obs

fn evaluate_policy_intervention(model: MarketDynamics) -> Causal[Float, intervened]:
  // Do-calculus intervention: forcing the action active, breaking confounding
  let policy: Causal[Float, intervened] = model.intervene(action = 1.0)
  return policy

fn main() -> Tensor:
  // Initialize the structural causal SCM
  let env = MarketDynamics()
  
  // Calculate observational vs interventional distributions
  let correlation = evaluate_observations(env)
  let causal_effect = evaluate_policy_intervention(env)
  
  // The type checker blocks any direct mixture of correlation & causation:
  // let invalid_math = correlation + causal_effect // ERROR!
  
  print("Causal SCM loops evaluated safely (no causal confusion).")
  return zeros(1, 1)
