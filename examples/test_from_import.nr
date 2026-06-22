// test_from_import.nr — Tests selective import
// This imports specific items from stdlib/nn.nr

from nn import Linear, FeedForward

fn main() -> Tensor[1, 1]:
  let w = glorot(4, 1)
  let x = zeros(4, 4) + 0.5
  let pred = x @ w
  let loss = mse(pred, zeros(4, 1) + 1.0)
  return loss
