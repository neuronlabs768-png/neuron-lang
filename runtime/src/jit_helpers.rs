/// JIT Helper Functions
///
/// These are the runtime support functions used by transpiled (JIT-compiled) NEURON code.
/// By living in the runtime crate, they are compiled once and shared across all JIT invocations,
/// eliminating the need to recompile them for every property test or JIT build.

use crate::tensor::Tensor;
use crate::vm::{Value, VM};
use std::collections::HashMap;

// ── Arithmetic ──────────────────────────────────────────────────────────────

pub fn jit_add(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Tensor(ta), Value::Tensor(tb)) => Value::Tensor(vm.tape.add(ta, tb)),
        (Value::Tensor(ta), Value::Int(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y as f64);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.add(ta, &tb))
        }
        (Value::Tensor(ta), Value::Float(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.add(ta, &tb))
        }
        (Value::Int(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x as f64);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.add(&ta, tb))
        }
        (Value::Float(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.add(&ta, tb))
        }
        (Value::Int(x), Value::Int(y)) => Value::Int(x + y),
        (Value::Float(x), Value::Float(y)) => Value::Float(x + y),
        _ => Value::Float(a.as_float() + b.as_float()),
    }
}

pub fn jit_sub(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Tensor(ta), Value::Tensor(tb)) => Value::Tensor(vm.tape.sub(ta, tb)),
        (Value::Tensor(ta), Value::Int(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y as f64);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.sub(ta, &tb))
        }
        (Value::Tensor(ta), Value::Float(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.sub(ta, &tb))
        }
        (Value::Int(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x as f64);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.sub(&ta, tb))
        }
        (Value::Float(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.sub(&ta, tb))
        }
        (Value::Int(x), Value::Int(y)) => Value::Int(x - y),
        (Value::Float(x), Value::Float(y)) => Value::Float(x - y),
        _ => Value::Float(a.as_float() - b.as_float()),
    }
}

pub fn jit_mul(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Tensor(ta), Value::Tensor(tb)) => Value::Tensor(vm.tape.mul(ta, tb)),
        (Value::Tensor(ta), Value::Int(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y as f64);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.mul(ta, &tb))
        }
        (Value::Tensor(ta), Value::Float(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.mul(ta, &tb))
        }
        (Value::Int(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x as f64);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.mul(&ta, tb))
        }
        (Value::Float(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.mul(&ta, tb))
        }
        (Value::Int(x), Value::Int(y)) => Value::Int(x * y),
        (Value::Float(x), Value::Float(y)) => Value::Float(x * y),
        _ => Value::Float(a.as_float() * b.as_float()),
    }
}

pub fn jit_div(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Tensor(ta), Value::Tensor(tb)) => Value::Tensor(vm.tape.div(ta, tb)),
        (Value::Tensor(ta), Value::Int(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y as f64);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.div(ta, &tb))
        }
        (Value::Tensor(ta), Value::Float(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.div(ta, &tb))
        }
        (Value::Int(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x as f64);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.div(&ta, tb))
        }
        (Value::Float(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.div(&ta, tb))
        }
        (Value::Int(x), Value::Int(y)) => Value::Int(x / y),
        (Value::Float(x), Value::Float(y)) => Value::Float(x / y),
        _ => Value::Float(a.as_float() / b.as_float()),
    }
}

pub fn jit_neg(vm: &mut VM, a: &Value) -> Value {
    match a {
        Value::Tensor(ta) => Value::Tensor(vm.tape.neg(ta)),
        _ => Value::Float(-a.as_float()),
    }
}

// ── Activations ─────────────────────────────────────────────────────────────

pub fn jit_gelu(vm: &mut VM, a: &Value) -> Value {
    match a {
        Value::Tensor(ta) => Value::Tensor(vm.tape.gelu(ta)),
        _ => a.clone(),
    }
}

pub fn jit_relu(vm: &mut VM, a: &Value) -> Value {
    match a {
        Value::Tensor(ta) => Value::Tensor(vm.tape.relu(ta)),
        _ => Value::Float(a.as_float().max(0.0)),
    }
}

pub fn jit_sigmoid(vm: &mut VM, a: &Value) -> Value {
    match a {
        Value::Tensor(ta) => Value::Tensor(vm.tape.sigmoid(ta)),
        _ => Value::Float(1.0 / (1.0 + (-a.as_float()).exp())),
    }
}

pub fn jit_tanh(vm: &mut VM, a: &Value) -> Value {
    match a {
        Value::Tensor(ta) => Value::Tensor(vm.tape.tanh(ta)),
        _ => Value::Float(a.as_float().tanh()),
    }
}

// ── Tensor constructors ─────────────────────────────────────────────────────

pub fn jit_zeros(vm: &mut VM, shape: Vec<usize>) -> Value {
    let mut t = Tensor::zeros(&shape);
    t.id = vm.tape.alloc_id();
    Value::Tensor(t)
}

pub fn jit_ones(vm: &mut VM, shape: Vec<usize>) -> Value {
    let mut t = Tensor::ones(&shape);
    t.id = vm.tape.alloc_id();
    Value::Tensor(t)
}

pub fn jit_randn(vm: &mut VM, shape: Vec<usize>) -> Value {
    let mut t = Tensor::randn(&shape);
    t.id = vm.tape.alloc_id();
    Value::Tensor(t)
}

pub fn jit_glorot(vm: &mut VM, shape: Vec<usize>) -> Value {
    let mut t = Tensor::glorot(&shape);
    t.id = vm.tape.alloc_id();
    Value::Tensor(t)
}

// ── Linear algebra ──────────────────────────────────────────────────────────

pub fn jit_matmul(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Tensor(ta), Value::Tensor(tb)) => Value::Tensor(vm.tape.matmul(ta, tb)),
        _ => panic!("JIT matmul requires tensor operands"),
    }
}

// ── Loss functions ──────────────────────────────────────────────────────────

pub fn jit_mse_loss(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Tensor(ta), Value::Tensor(tb)) => Value::Tensor(vm.tape.mse(ta, tb)),
        _ => panic!("JIT mse_loss requires tensor operands"),
    }
}

pub fn jit_cross_entropy(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Tensor(ta), Value::Tensor(tb)) => Value::Tensor(vm.tape.cross_entropy(ta, tb)),
        _ => panic!("JIT cross_entropy requires tensor operands"),
    }
}

// ── Autograd ────────────────────────────────────────────────────────────────

pub fn jit_apply_optimizer(vm: &mut VM, method: &str, target_name: &str, _grad_val: &Value, args: HashMap<String, f64>) {
    let lr = args.get("lr").cloned().unwrap_or(0.01);
    let _ = vm.apply_optimizer(target_name, lr, method);
}

pub fn jit_grad(vm: &mut VM, loss_val: &Value) -> Value {
    let loss_id = match loss_val {
        Value::Tensor(t) => t.id,
        _ => return Value::Void,
    };
    vm.tape.backward(loss_id);
    Value::Void
}

pub fn jit_backward(vm: &mut VM, loss_val: &Value, _param_names: Vec<&str>) -> Value {
    let loss_id = match loss_val {
        Value::Tensor(t) => t.id,
        _ => return Value::Void,
    };
    vm.tape.backward(loss_id);
    Value::Void
}

pub fn jit_stop_grad(vm: &mut VM, a: &Value) -> Value {
    match a {
        Value::Tensor(ta) => {
            let mut t = ta.clone();
            t.id = vm.tape.alloc_id();
            vm.tape.detach(ta.id);
            Value::Tensor(t)
        }
        _ => a.clone(),
    }
}

// ── Comparisons ─────────────────────────────────────────────────────────────

pub fn jit_lt(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_float() < b.as_float())
}

pub fn jit_lte(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_float() <= b.as_float())
}

pub fn jit_gt(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_float() > b.as_float())
}

pub fn jit_gte(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_float() >= b.as_float())
}

pub fn jit_eq(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Value::Bool(x == y),
        (Value::Bool(x), Value::Bool(y)) => Value::Bool(x == y),
        (Value::Str(x), Value::Str(y)) => Value::Bool(x == y),
        _ => Value::Bool(a.as_float() == b.as_float()),
    }
}

pub fn jit_neq(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Value::Bool(x != y),
        (Value::Bool(x), Value::Bool(y)) => Value::Bool(x != y),
        (Value::Str(x), Value::Str(y)) => Value::Bool(x != y),
        _ => Value::Bool(a.as_float() != b.as_float()),
    }
}

// ── Collections ─────────────────────────────────────────────────────────────

pub fn jit_list_len(a: &Value) -> Value {
    if let Value::List(ref l) = a {
        Value::Int(l.len() as i64)
    } else { Value::Int(0) }
}

pub fn jit_index(a: &Value, idx: &Value) -> Value {
    match a {
        Value::List(items) => {
            let i = idx.as_int() as usize;
            items.get(i).cloned().unwrap_or(Value::Void)
        }
        Value::Tensor(t) => {
            let i = idx.as_int() as usize;
            if t.ndim() == 2 {
                let cols = t.shape[1];
                let start = i * cols;
                let end = start + cols;
                if end <= t.data.len() {
                    let row_data = t.data[start..end].to_vec();
                    return Value::Tensor(Tensor::new(row_data, vec![1, cols]));
                }
            } else if t.ndim() == 1 {
                if i < t.data.len() {
                    return Value::Float(t.data[i]);
                }
            }
            Value::Void
        }
        _ => Value::Void,
    }
}
