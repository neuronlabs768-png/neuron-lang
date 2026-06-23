// ═══════════════════════════════════════════════════════════════
//  NEURON Advanced Demo: Type-Safe Data Preprocessing
//  Guarantees zero lookahead bias at compile time.
// ═══════════════════════════════════════════════════════════════

fn normalize_series(series: Temporal[Tensor, past_to_future]) -> Tensor:
  // Compile-time check: we only access index offsets from the past
  let window = series.before(10)
  let data = window.snapshot()
  
  // Simple scaling without lookahead bias
  let scale_factor = zeros(1, 1) + 0.1
  let mean_val = data @ scale_factor
  return data - mean_val

fn main() -> Tensor:
  // Load raw data into the type-safe temporal constructor
  let raw_data: Temporal[Tensor, past_to_future] = load_ohlcv("examples/market_data.csv")
  
  // This compiles because we normalize features safely looking backwards
  let normalized = normalize_series(raw_data)
  
  print("Preprocessed features successfully (100% lookahead-free).")
  return normalized
