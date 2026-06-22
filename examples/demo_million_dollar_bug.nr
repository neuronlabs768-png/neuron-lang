// ═══════════════════════════════════════════════════════════════
//  DEMO 1: "The Million-Dollar Bug"
//  NEURON catches lookahead bias at compile time.
// ═══════════════════════════════════════════════════════════════
//
//  In Python/PyTorch, this bug silently passes:
//
//    signal = model(price[t+1])   # accidental future leak
//    # Trains fine. Backtests beautifully. Loses millions live.
//
//  In NEURON, it's a compile error:
// ═══════════════════════════════════════════════════════════════

// A prediction model that must only see past data
fn predict_signal(prices: Temporal[Tensor, past_to_future]) -> Tensor[1]:
  let features = prices.before(20)   // ✓ Only uses past 20 bars
  let w = glorot(20, 1)
  return features @ w

// ✗ THIS IS THE BUG — using future data in a past-only context
fn backtest_with_leak(prices: Temporal[Tensor, past_to_future]) -> Tensor[1]:
  let future_prices = prices.after(1)   // Peek 1 step into the future
  return predict_signal(future_prices)  // ERROR: future data into past-only function

fn main() -> Tensor[1]:
  let prices: Temporal[Tensor, past_to_future] = load("market_data.csv")
  return backtest_with_leak(prices)
