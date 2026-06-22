use neuron_compiler::compile;
use neuron_compiler::errors::ErrorCode;

#[test]
fn test_integration_shape_compatible() {
    let src = r#"let x: Tensor[2, 3] = zeros(2, 3)
let w: Tensor[3, 4] = glorot(3, 4)
let result: Tensor[2, 4] = x @ w
"#;
    let out = compile(src, "compatible.nr");
    assert!(out.is_ok(), "Expected compilation to succeed, got errors: {:?}", out.err());
}

#[test]
fn test_integration_shape_mismatch() {
    let src = r#"let x: Tensor[2, 3] = zeros(2, 3)
let w: Tensor[4, 4] = glorot(4, 4)
let result: Tensor[2, 4] = x @ w
"#;
    let out = compile(src, "mismatch.nr");
    assert!(out.is_err(), "Expected shape mismatch compile error");
    let errs = out.err().unwrap();
    assert!(errs.has_errors());
    let has_shape_mismatch = errs.errors.iter().any(|e| matches!(e.code, ErrorCode::ShapeMismatch));
    assert!(has_shape_mismatch, "Expected ErrorCode::ShapeMismatch in: {:?}", errs.errors);
}
