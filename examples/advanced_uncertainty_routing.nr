// ═══════════════════════════════════════════════════════════════
//  NEURON Advanced Demo: Uncertainty-Aware Cascade Routing
//  Compiler enforces confidence threshold checks at compile time.
// ═══════════════════════════════════════════════════════════════

fn prediction_model() -> Uncertain[Float] [Effect[Rand]]:
  // Normal(mean, std) constructs an Uncertain[Float] type
  let mean = 0.75
  let std_dev = 0.12
  return Normal(mean, std_dev)

fn safe_routing_pipeline() -> Float [Effect[Rand]]:
  let estimate = prediction_model()
  
  // ✓ SAFE PATH: We query the confidence property, registering the safety check
  let conf = estimate.confidence
  
  if conf > 0.8:
    // Safe to access the raw predicted value
    return estimate.value
  else:
    // Fallback path when prediction is too noisy
    print("Warning: Prediction confidence low. Routing to fallback rule.")
    return 0.5

fn unsafe_ignored_pipeline() -> Float [Effect[Rand]]:
  let estimate = prediction_model()
  
  // ✗ UNSAFE PATH: Accessing value directly without querying .confidence
  // This causes the NEURON compiler to emit a warning:
  // warning[UncertaintyIgnored]: Uncertain value 'estimate' accessed without checking .confidence
  return estimate.value

fn main() -> Tensor:
  // Run the safe pipeline
  let result = safe_routing_pipeline()
  
  // We don't call the unsafe_ignored_pipeline here to keep check outputs clean,
  // but the compiler will still scan its function body and warn us!
  
  print("Uncertainty-aware routing evaluated safely.")
  return zeros(1, 1)
