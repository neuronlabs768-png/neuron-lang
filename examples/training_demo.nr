// training_demo.nr — NEURON training demonstration
// Fits a single weight parameter w to map input x = 2.0 to target y = 6.0.
// Expected target weight: 3.0 (starting at 5.0).

model Net:
  w: Tensor[1, 1] = zeros(1, 1) + 5.0

  fn train_step(self, x: Tensor[1, 1], y: Tensor[1, 1]) -> Tensor[1, 1] [Effect[Mut[self], IO]]:
    let pred = x @ self.w
    let loss = mse(pred, y)
    print(loss)
    update self.w by sgd(grad(loss), lr=0.1)
    return self.w

fn main() -> Tensor[1, 1]:
  let net = Net()
  let x: Tensor[1, 1] = zeros(1, 1) + 2.0
  let y: Tensor[1, 1] = zeros(1, 1) + 6.0

  print("Initial loss and weight updates:")
  let w1 = net.train_step(x, y)
  let w2 = net.train_step(x, y)
  let w3 = net.train_step(x, y)
  let w4 = net.train_step(x, y)
  let w5 = net.train_step(x, y)
  
  print("Final weight:")
  return net.w
