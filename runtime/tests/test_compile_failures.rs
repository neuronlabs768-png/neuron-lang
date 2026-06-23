use neuron_compiler::compile;
use neuron_compiler::errors::ErrorCode;

#[test]
fn test_integration_temporal_leak() {
    // A function expects a past_to_future temporal, but gets a future_to_past temporal
    let src = r#"
fn process_series(series: Temporal[Float, past_to_future]) -> Float:
  return 0.0

let series: Temporal[Float, future_to_past] = load("series.nr")
let res = process_series(series)
"#;
    let out = compile(src, "temporal_leak.nr");
    assert!(out.is_err(), "Expected compilation to fail with temporal leak error");
    let errs = out.err().unwrap();
    assert!(errs.has_errors());
    let has_temp_leak = errs.errors.iter().any(|e| matches!(e.code, ErrorCode::TemporalLeak));
    assert!(has_temp_leak, "Expected ErrorCode::TemporalLeak in errors: {:?}", errs.errors);
}

#[test]
fn test_integration_uncertainty_mix() {
    // Addition between Uncertain (Normal) and Random (GaussianNoise)
    let src = r#"
let x = Normal(10.0, 1.0)
let y = GaussianNoise(0.5)
let z = x + y
"#;
    let out = compile(src, "uncertainty_mix.nr");
    assert!(out.is_err(), "Expected compilation to fail with uncertainty mismatch");
    let errs = out.err().unwrap();
    assert!(errs.has_errors());
    let has_unc_mix = errs.errors.iter().any(|e| matches!(e.code, ErrorCode::UncertaintyMismatch));
    assert!(has_unc_mix, "Expected ErrorCode::UncertaintyMismatch in errors: {:?}", errs.errors);
}

#[test]
fn test_integration_causal_mix() {
    // Binary operation between observed and intervened causal values
    let _src = r#"
let model = CausalModel_new()
let obs = model.observe(treatment = 1.0)
let int = model.intervene(treatment = 1.0)
let res = obs + int
"#;
    // Note: CausalModel_new returns any/model, .observe returns Causal[observed], .intervene returns Causal[intervened].
    // Then obs + int adds them, causing CausalTypeMismatch.
    let src = r#"
let x: Causal[Float, observed] = load("obs.nr")
let y: Causal[Float, intervened] = load("int.nr")
let z = x + y
"#;
    let out = compile(src, "causal_mix.nr");
    assert!(out.is_err(), "Expected compilation to fail with causal type mismatch");
    let errs = out.err().unwrap();
    assert!(errs.has_errors());
    let has_causal_mix = errs.errors.iter().any(|e| matches!(e.code, ErrorCode::CausalTypeMismatch));
    assert!(has_causal_mix, "Expected ErrorCode::CausalTypeMismatch in errors: {:?}", errs.errors);
}
