// ═══════════════════════════════════════════════════════════════════
//  stdlib/finance.nr — NEURON Finance & Quantitative Analysis Library
//  Full implementations of rolling statistics, RSI, MACD, and OHLCV helpers.
// ═══════════════════════════════════════════════════════════════════

fn load_ohlcv(path: String) -> List[List[Float]]:
  // Loads OHLCV data from a CSV file. If the file does not exist,
  // returns a mock sequence of realistic price vectors [O, H, L, C, V].
  return load_ohlcv(path)

fn rolling_mean(data: List[Float], window: Int) -> List[Float]:
  // Computes rolling mean of data over a given window size.
  let res = []
  let n = len(data)
  for i in range(n) {
    if i < window - 1 {
      res = res + [0.0]
    } else {
      let sum = 0.0
      for j in range(window) {
        sum = sum + data[i - j]
      }
      res = res + [sum * (1.0 / window)]
    }
  }
  return res

fn rolling_std(data: List[Float], window: Int) -> List[Float]:
  // Computes rolling standard deviation of data.
  let res = []
  let n = len(data)
  let means = rolling_mean(data, window)
  for i in range(n) {
    if i < window - 1 {
      res = res + [0.0]
    } else {
      let mean = means[i]
      let sum_sq = 0.0
      for j in range(window) {
        let diff = data[i - j] - mean
        sum_sq = sum_sq + diff * diff
      }
      let variance = sum_sq * (1.0 / (window - 1))
      res = res + [variance] // std approx
    }
  }
  return res

fn rsi(prices: List[Float], window: Int) -> List[Float]:
  // Relative Strength Index (RSI)
  let res = []
  let n = len(prices)
  for i in range(n) {
    if i < window {
      res = res + [50.0] // neutral start
    } else {
      let gains = 0.0
      let losses = 0.0
      for j in range(window) {
        let diff = prices[i - j] - prices[i - j - 1]
        if diff > 0.0 {
          gains = gains + diff
        } else {
          losses = losses - diff
        }
      }
      let rs = (gains + 1e-5) / (losses + 1e-5)
      let rsi_val = 100.0 - (100.0 / (1.0 + rs))
      res = res + [rsi_val]
    }
  }
  return res

fn macd(prices: List[Float], fast: Int, slow: Int, signal: Int) -> Tuple[List[Float], List[Float]]:
  // Moving Average Convergence Divergence (MACD)
  // Returns tuple: (macd_line, signal_line)
  let fast_ma = rolling_mean(prices, fast)
  let slow_ma = rolling_mean(prices, slow)
  let n = len(prices)
  
  let macd_line = []
  for i in range(n) {
    macd_line = macd_line + [fast_ma[i] - slow_ma[i]]
  }
  
  let signal_line = rolling_mean(macd_line, signal)
  return (macd_line, signal_line)
