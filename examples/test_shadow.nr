fn main() -> Tensor[2, 2]:
  let x = glorot(2, 2)
  let x = x @ x
  return x
