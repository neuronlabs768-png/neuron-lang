// ═══════════════════════════════════════════════════════════════════
//  stdlib/causal.nr — NEURON Causal Inference & Discovery Library
//  Wraps native Structural Causal Model (SCM) operations and PC discovery.
// ═══════════════════════════════════════════════════════════════════

fn discover(data: List[List[Float]], names: List[String], alpha: Float) -> Any:
  // Discovers causal structure from observational data using the PC algorithm.
  // Returns PCResult model containing: adjacency, confidences, names.
  return causal_discover(data, names, alpha)

fn observe(names: List[String], weights: List[List[Float]], noise_vars: List[Float], noise_means: List[Float], evidence: Any) -> Any:
  // Performs Bayesian conditioning on structural equations without cutting edges.
  // evidence can be a model or list of (name, value) tuples.
  // Returns a model of Uncertain[Float] values representing conditional distributions.
  return causal_observe(names, weights, noise_vars, noise_means, evidence)

fn intervene(names: List[String], weights: List[List[Float]], noise_vars: List[Float], noise_means: List[Float], interventions: Any) -> Any:
  // Applies do-calculus interventions, cutting incoming edges and propagating values forward.
  // interventions can be a model or list of (name, value) tuples.
  // Returns a model of Uncertain[Float] values representing interventional distributions.
  return causal_intervene(names, weights, noise_vars, noise_means, interventions)

fn counterfactual(names: List[String], weights: List[List[Float]], noise_vars: List[Float], noise_means: List[Float], evidence: Any, interventions: Any, queries: List[String]) -> Any:
  // Computes Pearl's 3-step counterfactual logic: Abduction, Action, Prediction.
  // evidence: factual observations
  // interventions: counterfactual do-interventions
  // queries: variables to predict under counterfactual scenario
  // Returns a model of Float values containing the predicted counterfactual values.
  return causal_counterfactual(names, weights, noise_vars, noise_means, evidence, interventions, queries)
