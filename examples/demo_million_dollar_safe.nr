// ═══════════════════════════════════════════════════════════════
//  DEMO 1b: "The Safe Version" — How to do it RIGHT in NEURON
//  This version compiles and runs because it only uses past data.
// ═══════════════════════════════════════════════════════════════

// A prediction model that correctly uses only past data
fn predict_signal(prices: Temporal[Tensor, past_to_future]) -> Tensor[1]:
  let features = prices.before(20)   // ✓ Only past 20 bars
  let w = glorot(20, 1)
  return features @ w

// ✓ CORRECT — no future data leak
fn backtest_safe(prices: Temporal[Tensor, past_to_future]) -> Tensor[1]:
  return predict_signal(prices)  // Same temporal direction — compiles fine

fn main() -> Tensor[1]:
  let prices: Temporal[Tensor, past_to_future] = load("market_data.csv")
  return backtest_safe(prices)
