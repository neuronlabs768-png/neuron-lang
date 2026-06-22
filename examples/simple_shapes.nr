// simple_shapes.nr — NEURON shape verification example
// This program demonstrates compile-time tensor shape checking.

let x: Tensor[2, 3] = zeros(2, 3)
let w: Tensor[3, 4] = glorot(3, 4)
let result: Tensor[2, 4] = x @ w
let activated: Tensor[2, 4] = relu(result)
print(activated)
