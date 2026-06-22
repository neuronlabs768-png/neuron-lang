// test_import.nr — Tests the module import system
// This imports from stdlib/nn.nr

import nn

fn main() -> Tensor[1, 1]:
  let w = glorot(4, 1)
  let x = zeros(4, 4) + 0.5
  let pred = x @ w
  let loss = mse(pred, zeros(4, 1) + 1.0)
  return loss
