use neuron_compiler::compile;
use neuron_runtime::vm::{VM, Value};

#[test]
fn test_integration_training_sgd() {
    let src = r#"model Net:
  w: Tensor[1, 1] = zeros(1, 1) + 2.0

  fn train_step(self, x: Tensor[1, 1], y: Tensor[1, 1]) -> Tensor[1, 1] [Effect[Mut[self]]]:
    let pred = x @ self.w
    let loss = mse(pred, y)
    update self.w by sgd(grad(loss), lr=0.5)
    return self.w

fn main() -> Tensor[1, 1]:
  let net = Net()
  let x: Tensor[1, 1] = zeros(1, 1) + 2.0
  let y: Tensor[1, 1] = zeros(1, 1) + 6.0
  let res = net.train_step(x, y)
  return res
"#;

    let out = compile(src, "training_sgd.nr").unwrap();
    let mut vm = VM::new();
    vm.load(&out.ir);

    let result = vm.run_main().unwrap();
    if let Value::Tensor(t) = &result {
        assert_eq!(t.shape, vec![1, 1]);
        assert!((t.data[0] - 6.0).abs() < 1e-5, "Expected weight to be updated to 6.0, got {:.4}", t.data[0]);
    } else {
        panic!("Expected tensor result from main, got {:?}", result);
    }
}

#[test]
fn test_integration_training_adam() {
    let src = r#"model Net:
  w: Tensor[1, 1] = zeros(1, 1) + 1.0

  fn train_step(self, x: Tensor[1, 1], y: Tensor[1, 1]) -> Tensor[1, 1] [Effect[Mut[self]]]:
    let pred = x @ self.w
    let loss = mse(pred, y)
    update self.w by adam(grad(loss), lr=0.1)
    return self.w

fn main() -> Tensor[1, 1]:
  let net = Net()
  let x: Tensor[1, 1] = zeros(1, 1) + 1.0
  let y: Tensor[1, 1] = zeros(1, 1) + 2.0
  // pred = 1.0 * w = 1.0. target = 2.0.
  // loss = (pred - target)^2 = (w - 2.0)^2.
  // grad = 2 * (w - 2.0) = -2.0.
  // Under Adam, the gradient direction is negative, so m gets negative, v gets positive.
  // The weight should move towards 2.0 (increase).
  let res = net.train_step(x, y)
  return res
"#;

    let out = compile(src, "training_adam.nr").unwrap();
    let mut vm = VM::new();
    vm.load(&out.ir);

    let result = vm.run_main().unwrap();
    if let Value::Tensor(t) = &result {
        assert_eq!(t.shape, vec![1, 1]);
        // Weight should have increased from 1.0 towards 2.0
        assert!(t.data[0] > 1.0, "Expected weight to increase, got {:.4}", t.data[0]);
    } else {
        panic!("Expected tensor result from main, got {:?}", result);
    }
}
