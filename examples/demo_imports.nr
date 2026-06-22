// ═══════════════════════════════════════════════════════════════
//  DEMO 4: Multi-File Programs
//  Real-world NEURON code uses imports just like Python.
// ═══════════════════════════════════════════════════════════════

from nn import Linear, FeedForward
import optim

model TradingModel:
  w: Tensor[4, 1] = glorot(4, 1)

  fn predict(self, features: Tensor[B, 4]) -> Tensor[B, 1]:
    return sigmoid(features @ self.w)

fn main():
  let net = TradingModel()
  let features = zeros(10, 4) + 0.5
  let target = zeros(10, 1) + 1.0

  let pred = net.predict(features)
  let loss = mse(pred, target)

  print(loss)
  return loss
