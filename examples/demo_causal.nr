// ═══════════════════════════════════════════════════════════════
//  DEMO 2: "Correlation Is Not Causation"
//  NEURON distinguishes observation from intervention at the
//  type level. You cannot accidentally confuse the two.
// ═══════════════════════════════════════════════════════════════
//
//  Question: "Will giving Treatment X improve patient outcomes?"
//
//  Traditional ML: correlations. Patients who got Treatment X
//  did better — but maybe they were healthier to begin with.
//
//  NEURON: real causal reasoning with do-calculus.
// ═══════════════════════════════════════════════════════════════

// Observational study — just watching what happened
// Returns Causal[Float, observed]
fn observational_effect(model, data):
  let obs: Causal[Float, observed] = model.observe(treatment = 1.0)
  return obs

// Interventional study — what happens if we FORCE the treatment
// Returns Causal[Float, intervened]  
fn interventional_effect(model):
  let forced: Causal[Float, intervened] = model.intervene(treatment = 1.0)
  return forced

// ✗ THIS IS THE BUG — mixing correlation with causation
fn naive_comparison(model, data):
  let correlation: Causal[Float, observed] = model.observe(treatment = 1.0)
  let causation: Causal[Float, intervened] = model.intervene(treatment = 1.0)
  return correlation + causation  // ERROR: Cannot combine observed and intervened
