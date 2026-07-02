/// NEURON Compiler error types.
///
/// All diagnostics — errors and warnings — are collected into a `CompileResult`.

use crate::token::Span;
use std::fmt;

// ═══════════════════════════════════════════
//  Error / Warning codes
// ═══════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    // Lexer
    UnexpectedChar,
    UnterminatedString,
    TabIndent,
    InconsistentIndent,
    // Parser
    ParseError,
    // Type checker
    ShapeMismatch,
    TemporalLeak,
    CausalTypeMismatch,
    UncertaintyMismatch,
    EffectUndeclared,
    TypeMismatch,
    UndefinedVariable,
    DuplicateDefinition,
    InvalidOperation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningCode {
    DynamicDim,
    UncertaintyIgnored,
    UnusedVariable,
    ImportWarning,
}

// ═══════════════════════════════════════════
//  Error / Warning structs
// ═══════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct NeuronError {
    pub code: ErrorCode,
    pub message: String,
    pub span: Span,
    pub filename: Option<String>,
    pub expected: Option<String>,
    pub actual: Option<String>,
    pub notes: Vec<String>,
    pub fix: Option<String>,
}

impl NeuronError {
    pub fn new(code: ErrorCode, message: impl Into<String>, span: Span) -> Self {
        Self {
            code,
            message: message.into(),
            span,
            filename: None,
            expected: None,
            actual: None,
            notes: Vec::new(),
            fix: None,
        }
    }

    pub fn with_expected(mut self, expected: impl Into<String>) -> Self {
        self.expected = Some(expected.into());
        self
    }

    pub fn with_actual(mut self, actual: impl Into<String>) -> Self {
        self.actual = Some(actual.into());
        self
    }

    pub fn with_fix(mut self, fix: impl Into<String>) -> Self {
        self.fix = Some(fix.into());
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

impl fmt::Display for NeuronError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Red bold "error", followed by bold code and message
        write!(f, "\x1b[1;31merror\x1b[0m\x1b[1m[{:?}]: {}\x1b[0m\n", self.code, self.message)?;
        
        if let Some(ref filename) = self.filename {
            // Cyan bold arrow, blue filename/line/col
            write!(f, "\x1b[1;36m  --> \x1b[0m\x1b[34m{}:{}:{}\x1b[0m\n", filename, self.span.line, self.span.col)?;
            
            // Try to open the file and read the exact line
            if let Ok(content) = std::fs::read_to_string(filename) {
                let lines: Vec<&str> = content.lines().collect();
                if self.span.line > 0 && self.span.line <= lines.len() {
                    let line_text = lines[self.span.line - 1];
                    let line_num_str = format!(" {} | ", self.span.line);
                    let padding = " ".repeat(line_num_str.len());
                    
                    // Print the line code: blue line number separator, regular code
                    write!(f, "\x1b[1;34m{} | \x1b[0m{}\n", self.span.line, line_text)?;
                    
                    // Draw underline: red bold carets
                    let col = self.span.col.saturating_sub(1);
                    let underline_spaces = " ".repeat(col);
                    let underline_carets = "^".repeat(self.span.len.max(1));
                    write!(f, "{}{}\x1b[1;31m{}\x1b[0m\n", padding, underline_spaces, underline_carets)?;
                }
            }
        } else {
            write!(f, "  at {}:{}\n", self.span.line, self.span.col)?;
        }
        
        if let Some(ref exp) = self.expected {
            write!(f, "  \x1b[1;32mexpected:\x1b[0m {}\n", exp)?;
        }
        if let Some(ref act) = self.actual {
            write!(f, "  \x1b[1;31mgot:\x1b[0m      {}\n", act)?;
        }
        for note in &self.notes {
            write!(f, "  \x1b[1;36mnote:\x1b[0m {}\n", note)?;
        }
        if let Some(ref fix) = self.fix {
            write!(f, "  \x1b[1;32mhelp:\x1b[0m {}\n", fix)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct NeuronWarning {
    pub code: WarningCode,
    pub message: String,
    pub span: Span,
    pub filename: Option<String>,
    pub notes: Vec<String>,
    pub fix: Option<String>,
}

impl NeuronWarning {
    pub fn new(code: WarningCode, message: impl Into<String>, span: Span) -> Self {
        Self {
            code,
            message: message.into(),
            span,
            filename: None,
            notes: Vec::new(),
            fix: None,
        }
    }

    pub fn with_fix(mut self, fix: impl Into<String>) -> Self {
        self.fix = Some(fix.into());
        self
    }
}

impl fmt::Display for NeuronWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Yellow bold "warning", followed by bold code and message
        write!(f, "\x1b[1;33mwarning\x1b[0m\x1b[1m[{:?}]: {}\x1b[0m\n", self.code, self.message)?;
        
        if let Some(ref filename) = self.filename {
            write!(f, "\x1b[1;36m  --> \x1b[0m\x1b[34m{}:{}:{}\x1b[0m\n", filename, self.span.line, self.span.col)?;
            
            if let Ok(content) = std::fs::read_to_string(filename) {
                let lines: Vec<&str> = content.lines().collect();
                if self.span.line > 0 && self.span.line <= lines.len() {
                    let line_text = lines[self.span.line - 1];
                    let line_num_str = format!(" {} | ", self.span.line);
                    let padding = " ".repeat(line_num_str.len());
                    
                    write!(f, "\x1b[1;34m{} | \x1b[0m{}\n", self.span.line, line_text)?;
                    
                    // Draw underline: yellow bold carets
                    let col = self.span.col.saturating_sub(1);
                    let underline_spaces = " ".repeat(col);
                    let underline_carets = "^".repeat(self.span.len.max(1));
                    write!(f, "{}{}\x1b[1;33m{}\x1b[0m\n", padding, underline_spaces, underline_carets)?;
                }
            }
        } else {
            write!(f, "  at {}:{}\n", self.span.line, self.span.col)?;
        }
        
        for note in &self.notes {
            write!(f, "  \x1b[1;36mnote:\x1b[0m {}\n", note)?;
        }
        if let Some(ref fix) = self.fix {
            write!(f, "  \x1b[1;32mhelp:\x1b[0m {}\n", fix)?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════
//  CompileResult
// ═══════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct CompileResult {
    pub filename: String,
    pub errors: Vec<NeuronError>,
    pub warnings: Vec<NeuronWarning>,
}

impl CompileResult {
    pub fn new(filename: impl Into<String>) -> Self {
        Self {
            filename: filename.into(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    pub fn add_error(&mut self, mut err: NeuronError) {
        if err.filename.is_none() {
            err.filename = Some(self.filename.clone());
        }
        self.errors.push(err);
    }

    pub fn add_warning(&mut self, mut warn: NeuronWarning) {
        if warn.filename.is_none() {
            warn.filename = Some(self.filename.clone());
        }
        self.warnings.push(warn);
    }
}

// ═══════════════════════════════════════════
//  Convenience constructors
// ═══════════════════════════════════════════

pub fn shape_mismatch_error(span: Span, expected: &str, actual: &str, context: &str) -> NeuronError {
    NeuronError::new(ErrorCode::ShapeMismatch, format!("Tensor shape mismatch in {}", context), span)
        .with_expected(expected)
        .with_actual(actual)
        .with_fix("Ensure tensor dimensions are compatible for this operation")
}

pub fn temporal_leak_error(span: Span, found_dir: &str, expected_dir: &str) -> NeuronError {
    NeuronError::new(
        ErrorCode::TemporalLeak,
        format!(
            "Temporal direction violation: data flows {} but context expects {} — lookahead bias detected",
            found_dir, expected_dir
        ),
        span,
    )
    .with_expected(expected_dir)
    .with_actual(found_dir)
    .with_fix("Use .before(t) to restrict temporal data to the past, or .snapshot(at=t) to remove temporal ordering")
}

pub fn causal_type_mismatch_error(span: Span, mode_a: &str, mode_b: &str) -> NeuronError {
    NeuronError::new(
        ErrorCode::CausalTypeMismatch,
        format!("Cannot combine {} and {} causal values — causal type mismatch", mode_a, mode_b),
        span,
    )
    .with_fix("Use only observed or only intervened data in the same expression. To compare, use a causal estimator.")
}

pub fn uncertainty_mismatch_error(span: Span, kind_a: &str, kind_b: &str) -> NeuronError {
    NeuronError::new(
        ErrorCode::UncertaintyMismatch,
        format!("Cannot combine {}[T] with {}[T] — epistemic and aleatoric uncertainty are distinct", kind_a, kind_b),
        span,
    )
    .with_fix("Convert to the same uncertainty type, or use separate processing paths")
}

pub fn effect_undeclared_error(span: Span, fn_name: &str, missing: &[String]) -> NeuronError {
    NeuronError::new(
        ErrorCode::EffectUndeclared,
        format!("Function '{}' performs effects not declared in its signature: {}", fn_name, missing.join(", ")),
        span,
    )
    .with_fix(format!("Add [Effect[{}]] to the function signature", missing.join(", ")))
}

pub fn dynamic_dim_warning(span: Span) -> NeuronWarning {
    NeuronWarning::new(
        WarningCode::DynamicDim,
        "Dynamic dimension '?' bypasses compile-time shape checking",
        span,
    )
    .with_fix("Use a symbolic dimension name (e.g., B) for compile-time verification")
}

pub fn uncertainty_ignored_warning(span: Span, var_name: &str) -> NeuronWarning {
    NeuronWarning::new(
        WarningCode::UncertaintyIgnored,
        format!("Uncertain value '{}' accessed without checking .confidence — uncertainty may be silently ignored", var_name),
        span,
    )
    .with_fix(format!("Check {}.confidence before using {}.value", var_name, var_name))
}
