use neuron_compiler::compile;
use neuron_compiler::errors::ErrorCode;

#[test]
fn test_integration_effects_ok() {
    let src = r#"model Learner:
  w: Tensor[2, 2] = glorot(2, 2)
  
  fn train_step(self, loss: Tensor[1]) [Effect[Mut[self]]]:
    update self.w by adam(grad(loss), lr=1e-3)
"#;
    let out = compile(src, "effects_ok.nr");
    assert!(out.is_ok(), "Expected compilation to succeed, got errors: {:?}", out.err());
}

#[test]
fn test_integration_effect_undeclared() {
    let src = r#"model Learner:
  w: Tensor[2, 2] = glorot(2, 2)
  
  fn train_step_missing_effect(self, loss: Tensor[1]):
    update self.w by adam(grad(loss), lr=1e-3)
"#;
    let out = compile(src, "effects_error.nr");
    assert!(out.is_err(), "Expected undeclared effect compile error");
    let errs = out.err().unwrap();
    assert!(errs.has_errors());
    let has_effect_err = errs.errors.iter().any(|e| matches!(e.code, ErrorCode::EffectUndeclared));
    assert!(has_effect_err, "Expected ErrorCode::EffectUndeclared in: {:?}", errs.errors);
}
