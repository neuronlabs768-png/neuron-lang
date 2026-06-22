/// NEURON Effect tracker — runtime enforcement of the effect system.
///
/// Pure-declared functions that attempt mutation trigger runtime panic.
/// Temporal direction violations panic with clear messages.
/// Causal type mismatches are caught.

use std::collections::HashSet;

/// Tracks effects observed during function execution.
#[derive(Debug, Default)]
pub struct EffectTracker {
    /// Stack of effect scopes (one per function call).
    scopes: Vec<EffectScope>,
}

#[derive(Debug)]
struct EffectScope {
    fn_name: String,
    declared_effects: HashSet<String>,
    observed_effects: Vec<ObservedEffect>,
    is_pure: bool,
}

#[derive(Debug)]
struct ObservedEffect {
    kind: String,
    target: Option<String>,
    location: String,
}

impl EffectTracker {
    pub fn new() -> Self {
        Self { scopes: Vec::new() }
    }

    /// Enter a function scope with its declared effects.
    pub fn enter_fn(&mut self, name: &str, declared: &[String], is_pure: bool) {
        self.scopes.push(EffectScope {
            fn_name: name.to_string(),
            declared_effects: declared.iter().cloned().collect(),
            observed_effects: Vec::new(),
            is_pure,
        });
    }

    /// Exit a function scope. Returns Err if undeclared effects were observed.
    pub fn exit_fn(&mut self) -> Result<(), String> {
        if let Some(scope) = self.scopes.pop() {
            let mut violations = Vec::new();
            for effect in &scope.observed_effects {
                if scope.is_pure {
                    violations.push(format!(
                        "EFFECT VIOLATION: pure function '{}' performed {} effect at {}",
                        scope.fn_name, effect.kind, effect.location
                    ));
                } else if !scope.declared_effects.contains(&effect.kind) {
                    violations.push(format!(
                        "EFFECT VIOLATION: function '{}' performed undeclared {} effect at {}",
                        scope.fn_name, effect.kind, effect.location
                    ));
                }
            }
            if !violations.is_empty() {
                return Err(violations.join("\n"));
            }
        }
        Ok(())
    }

    /// Record an observed IO effect.
    pub fn record_io(&mut self, location: &str) {
        self.record("IO", None, location);
    }

    /// Record an observed mutation effect.
    pub fn record_mutation(&mut self, target: &str, location: &str) {
        self.record("Mut", Some(target.to_string()), location);
    }

    /// Record an observed randomness effect.
    pub fn record_rand(&mut self, location: &str) {
        self.record("Rand", None, location);
    }

    fn record(&mut self, kind: &str, target: Option<String>, location: &str) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.observed_effects.push(ObservedEffect {
                kind: kind.to_string(),
                target,
                location: location.to_string(),
            });
        }
    }
}

/// Check temporal direction at runtime — panics on violation.
pub fn check_temporal_direction(actual: &str, expected: &str) -> Result<(), String> {
    if actual != expected {
        Err(format!(
            "RUNTIME PANIC: temporal direction violation\n\
             Expected:  {}\n\
             Got:       {}\n\
             This is a lookahead bias — future data is leaking into a past-only context.\n\
             Fix: use .before(t) to restrict data to the past, or .snapshot(at=t) to remove temporal ordering.",
            expected, actual
        ))
    } else {
        Ok(())
    }
}

/// Check causal mode compatibility at runtime.
pub fn check_causal_mode(mode_a: &str, mode_b: &str) -> Result<(), String> {
    if mode_a != mode_b {
        Err(format!(
            "RUNTIME PANIC: causal type mismatch\n\
             Cannot combine {} and {} causal values.\n\
             Observed data (P(Y|X=x)) and interventional data (P(Y|do(X=x))) are fundamentally different.\n\
             Fix: use only observed or only intervened data in the same expression.",
            mode_a, mode_b
        ))
    } else {
        Ok(())
    }
}

/// Check uncertainty type compatibility.
pub fn check_uncertainty(kind_a: &str, kind_b: &str) -> Result<(), String> {
    if kind_a != kind_b {
        Err(format!(
            "RUNTIME PANIC: uncertainty type mismatch\n\
             Cannot combine {} and {} — epistemic and aleatoric uncertainty are fundamentally distinct.\n\
             Epistemic: 'I lack data' (reducible with more data).\n\
             Aleatoric: 'It is inherently random' (irreducible).\n\
             Fix: convert to the same uncertainty type or process separately.",
            kind_a, kind_b
        ))
    } else {
        Ok(())
    }
}
