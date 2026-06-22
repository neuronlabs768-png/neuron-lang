/// NEURON internal type representations and type checker.
///
/// Types are resolved from AST TypeExpr nodes and used for all type checking.
/// The type checker walks the AST, builds scoped symbol tables, and enforces
/// all NEURON type rules: tensor shapes, uncertainty, temporal, causal, effects.

use crate::ast::*;
use crate::token::Span;
use crate::errors::*;
use std::collections::HashMap;

// ═══════════════════════════════════════════
//  Internal type representations
// ═══════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum NType {
    Base(String),                          // Int, Float, Bool, String, Timestamp, Loss, Dataset
    Tensor(Vec<Dim>),                      // Tensor[dims]
    Uncertain(Box<NType>),                 // Uncertain[T]
    Random(Box<NType>),                    // Random[T]
    Prob(Box<NType>),                      // Prob[T]
    Temporal(Box<NType>, String),          // Temporal[T, direction]
    Causal(Box<NType>, String),            // Causal[T, mode]
    Learnable(String, Option<Box<NType>>), // Learnable[FnType]
    Effect(Vec<EffectEntry>),              // Effect[IO, Rand, Mut[x]]
    List(Box<NType>),                      // List[T]
    Option_(Box<NType>),                   // Option[T]
    Tuple(Vec<NType>),                     // (T1, T2, ...)
    Fn_(Vec<NType>, Box<NType>, Option<Box<NType>>), // fn(params) -> ret [effects]
    Model(String, HashMap<String, NType>, HashMap<String, NType>), // Model { fields, methods }
    CausalModel(String, Vec<String>),      // CausalModel { variables }
    // AGI types
    Memory(Box<NType>),
    EpisodicMemory(Box<NType>),
    SemanticMemory(Box<NType>),
    WorkingMemory(Box<NType>, Option<i64>),
    Reward(Box<NType>),
    Agent(String),
    Void,
    Any,
    Explanation,
}

#[derive(Debug, Clone)]
pub enum Dim {
    Static(i64),
    Symbolic(String),
    Named(String, String),
    Dynamic,
}

#[derive(Debug, Clone)]
pub struct EffectEntry {
    pub kind: String,
    pub target: Option<String>,
}

impl NType {
    pub fn display(&self) -> String {
        match self {
            NType::Base(n) => n.clone(),
            NType::Tensor(dims) => {
                if dims.is_empty() { "Tensor".into() }
                else {
                    let ds: Vec<String> = dims.iter().map(|d| match d {
                        Dim::Static(v) => v.to_string(),
                        Dim::Symbolic(s) => s.clone(),
                        Dim::Named(a, n) => format!("{}:{}", a, n),
                        Dim::Dynamic => "?".into(),
                    }).collect();
                    format!("Tensor[{}]", ds.join(", "))
                }
            }
            NType::Uncertain(inner) => format!("Uncertain[{}]", inner.display()),
            NType::Random(inner) => format!("Random[{}]", inner.display()),
            NType::Temporal(inner, dir) => format!("Temporal[{}, {}]", inner.display(), dir),
            NType::Causal(inner, mode) => format!("Causal[{}, {}]", inner.display(), mode),
            NType::List(inner) => format!("List[{}]", inner.display()),
            NType::Model(name, _, _) => format!("Model[{}]", name),
            NType::Memory(inner) => format!("Memory[{}]", inner.display()),
            NType::EpisodicMemory(inner) => format!("EpisodicMemory[{}]", inner.display()),
            NType::Reward(inner) => format!("Reward[{}]", inner.display()),
            NType::Agent(name) => format!("Agent[{}]", name),
            NType::Void => "Void".into(),
            NType::Any => "Any".into(),
            _ => format!("{:?}", self),
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, NType::Base(n) if n == "Int" || n == "Float")
    }
    pub fn is_tensor(&self) -> bool { matches!(self, NType::Tensor(_)) }
}

fn types_compatible(a: &NType, b: &NType) -> bool {
    if matches!(a, NType::Any) || matches!(b, NType::Any) { return true; }
    match (a, b) {
        (NType::Base(x), NType::Base(y)) => x == y,
        (NType::Tensor(_), NType::Tensor(_)) => true, // Shape checked separately
        (NType::Uncertain(x), NType::Uncertain(y)) => types_compatible(x, y),
        (NType::Random(x), NType::Random(y)) => types_compatible(x, y),
        (NType::Temporal(x, d1), NType::Temporal(y, d2)) => d1 == d2 && types_compatible(x, y),
        (NType::Causal(x, m1), NType::Causal(y, m2)) => m1 == m2 && types_compatible(x, y),
        (NType::List(x), NType::List(y)) => types_compatible(x, y),
        (NType::Model(a, _, _), NType::Model(b, _, _)) => a == b,
        (NType::Model(a, _, _), NType::Base(b)) => a == b,
        (NType::Base(a), NType::Model(b, _, _)) => a == b,
        (NType::Void, NType::Void) => true,
        _ => false,
    }
}

fn type_from_ast(te: &TypeExpr) -> NType {
    match te {
        TypeExpr::Base(name, _) => NType::Base(name.clone()),
        TypeExpr::Tensor(dims, _) => NType::Tensor(dims.iter().map(|d| match d {
            DimExpr::Static(v) => Dim::Static(*v),
            DimExpr::Symbolic(s) => Dim::Symbolic(s.clone()),
            DimExpr::Named(a, n) => Dim::Named(a.clone(), n.clone()),
            DimExpr::Dynamic => Dim::Dynamic,
        }).collect()),
        TypeExpr::Uncertain(inner, _) => NType::Uncertain(Box::new(type_from_ast(inner))),
        TypeExpr::Random(inner, _) => NType::Random(Box::new(type_from_ast(inner))),
        TypeExpr::Prob(inner, _) => NType::Prob(Box::new(type_from_ast(inner))),
        TypeExpr::Temporal(inner, dir, _) => NType::Temporal(Box::new(type_from_ast(inner)), dir.clone()),
        TypeExpr::Causal(inner, mode, _) => NType::Causal(Box::new(type_from_ast(inner)), mode.clone()),
        TypeExpr::Learnable(fn_type, _, _) => NType::Learnable(fn_type.clone(), None),
        TypeExpr::ListType(inner, _) => NType::List(Box::new(type_from_ast(inner))),
        TypeExpr::OptionType(inner, _) => NType::Option_(Box::new(type_from_ast(inner))),
        TypeExpr::Memory(inner, _) => NType::Memory(Box::new(type_from_ast(inner))),
        TypeExpr::EpisodicMemory(inner, _) => NType::EpisodicMemory(Box::new(type_from_ast(inner))),
        TypeExpr::SemanticMemory(inner, _) => NType::SemanticMemory(Box::new(type_from_ast(inner))),
        TypeExpr::WorkingMemory(inner, cap, _) => {
            let c = cap.as_ref().and_then(|e| if let Expr::IntLit(v, _) = e.as_ref() { Some(*v) } else { None });
            NType::WorkingMemory(Box::new(type_from_ast(inner)), c)
        }
        TypeExpr::RewardType(inner, _) => NType::Reward(Box::new(type_from_ast(inner))),
        TypeExpr::Fn(params, ret, _) => {
            let ps: Vec<NType> = params.iter().map(|p| type_from_ast(p)).collect();
            NType::Fn_(ps, Box::new(type_from_ast(ret)), None)
        }
        TypeExpr::UserDefined(name, _) => NType::Base(name.clone()),
    }
}

// ═══════════════════════════════════════════
//  Unification environment for symbolic dims
// ═══════════════════════════════════════════

#[derive(Debug, Default)]
struct UnificationEnv {
    bindings: HashMap<String, Dim>,
}

impl UnificationEnv {
    fn resolve(&self, d: &Dim) -> Dim {
        match d {
            Dim::Symbolic(name) => {
                if let Some(bound) = self.bindings.get(name) {
                    self.resolve(bound)
                } else { d.clone() }
            }
            Dim::Named(alias, _) => {
                if let Some(bound) = self.bindings.get(alias) {
                    self.resolve(bound)
                } else { d.clone() }
            }
            _ => d.clone(),
        }
    }

    fn unify(&mut self, a: &Dim, b: &Dim) -> bool {
        let ra = self.resolve(a);
        let rb = self.resolve(b);
        match (&ra, &rb) {
            (Dim::Dynamic, _) | (_, Dim::Dynamic) => true,
            (Dim::Static(x), Dim::Static(y)) => x == y,
            (Dim::Symbolic(sa), Dim::Symbolic(sb)) if sa == sb => true,
            (Dim::Symbolic(s), other) | (other, Dim::Symbolic(s)) => {
                self.bindings.insert(s.clone(), other.clone());
                true
            }
            (Dim::Named(a, _), Dim::Named(b, _)) if a == b => true,
            (Dim::Named(a, _), other) | (other, Dim::Named(a, _)) => {
                self.bindings.insert(a.clone(), other.clone());
                true
            }
        }
    }
}

// ═══════════════════════════════════════════
//  Scope / Symbol Table
// ═══════════════════════════════════════════

#[derive(Debug)]
struct Scope {
    symbols: HashMap<String, NType>,
    mutations: Vec<String>,
    uncertain_accessed: Vec<(String, Span)>,
    uncertain_confidence_checked: Vec<String>,
}

impl Scope {
    fn new() -> Self {
        Self {
            symbols: HashMap::new(),
            mutations: Vec::new(),
            uncertain_accessed: Vec::new(),
            uncertain_confidence_checked: Vec::new(),
        }
    }
    
    fn record_uncertain_access(&mut self, name: &str, span: Span) {
        self.uncertain_accessed.push((name.to_string(), span));
    }

    fn record_uncertain_confidence_checked(&mut self, name: &str) {
        self.uncertain_confidence_checked.push(name.to_string());
    }

    fn define(&mut self, name: &str, ty: NType) {
        self.symbols.insert(name.to_string(), ty);
    }

    fn lookup(&self, name: &str) -> Option<&NType> {
        self.symbols.get(name)
    }
}

struct SymbolTable {
    scopes: Vec<Scope>,
}

impl SymbolTable {
    fn new() -> Self {
        let mut global = Scope::new();
        // Built-in functions
        let tensor = NType::Tensor(vec![]);
        let float = NType::Base("Float".into());
        let int = NType::Base("Int".into());
        let loss = NType::Base("Loss".into());
        let dataset = NType::Base("Dataset".into());
        let any = NType::Any;

        global.define("zeros", NType::Fn_(vec![any.clone()], Box::new(tensor.clone()), None));
        global.define("glorot", NType::Fn_(vec![any.clone()], Box::new(tensor.clone()), None));
        global.define("relu", NType::Fn_(vec![tensor.clone()], Box::new(tensor.clone()), None));
        global.define("gelu", NType::Fn_(vec![tensor.clone()], Box::new(tensor.clone()), None));
        global.define("softmax", NType::Fn_(vec![tensor.clone()], Box::new(tensor.clone()), None));
        global.define("sigmoid", NType::Fn_(vec![tensor.clone()], Box::new(tensor.clone()), None));
        global.define("tanh", NType::Fn_(vec![tensor.clone()], Box::new(tensor.clone()), None));
        global.define("cross_entropy", NType::Fn_(vec![tensor.clone(), tensor.clone()], Box::new(loss.clone()), None));
        global.define("mse", NType::Fn_(vec![tensor.clone(), tensor.clone()], Box::new(loss.clone()), None));
        global.define("negative_log_likelihood", NType::Fn_(vec![tensor.clone(), tensor.clone()], Box::new(loss.clone()), None));
        global.define("kl_divergence", NType::Fn_(vec![any.clone(), any.clone()], Box::new(loss.clone()), None));
        global.define("concat", NType::Fn_(vec![NType::List(Box::new(tensor.clone()))], Box::new(tensor.clone()), None));
        global.define("range", NType::Fn_(vec![int.clone()], Box::new(NType::List(Box::new(int.clone()))), None));
        global.define("min", NType::Fn_(vec![any.clone(), any.clone()], Box::new(any.clone()), None));
        global.define("max", NType::Fn_(vec![any.clone(), any.clone()], Box::new(any.clone()), None));
        global.define("abs", NType::Fn_(vec![any.clone()], Box::new(any.clone()), None));
        global.define("Normal", NType::Fn_(vec![float.clone(), float.clone()], Box::new(NType::Uncertain(Box::new(float.clone()))), None));
        global.define("Beta", NType::Fn_(vec![float.clone(), float.clone()], Box::new(NType::Uncertain(Box::new(float.clone()))), None));
        global.define("GaussianNoise", NType::Fn_(vec![float.clone()], Box::new(NType::Random(Box::new(float.clone()))), None));
        global.define("load", NType::Fn_(vec![NType::Base("String".into())], Box::new(any.clone()), None));
        global.define("load_dataset", NType::Fn_(vec![NType::Base("String".into())], Box::new(dataset.clone()), None));
        global.define("load_ohlcv", NType::Fn_(vec![NType::Base("String".into())], Box::new(any.clone()), None));
        global.define("aggregate", NType::Fn_(vec![NType::List(Box::new(any.clone()))], Box::new(any.clone()), None));
        global.define("estimate_epistemic_std", NType::Fn_(vec![tensor.clone()], Box::new(float.clone()), None));
        global.define("fractional_kelly", NType::Fn_(vec![NType::Uncertain(Box::new(float.clone())), float.clone()], Box::new(float.clone()), None));
        global.define("sample", NType::Fn_(vec![any.clone()], Box::new(any.clone()), None));
        global.define("condition", NType::Fn_(vec![any.clone(), any.clone()], Box::new(any.clone()), None));
        global.define("print", NType::Fn_(vec![any.clone()], Box::new(NType::Void), None));
        global.define("input", NType::Fn_(vec![], Box::new(NType::Base("String".into())), None));
        global.define("embed_string", NType::Fn_(vec![NType::Base("String".into())], Box::new(tensor.clone()), None));
        global.define("generate_reply", NType::Fn_(vec![NType::Base("String".into())], Box::new(NType::Base("String".into())), None));

        // Built-in type names
        global.define("Int", NType::Base("Int".into()));
        global.define("Float", NType::Base("Float".into()));
        global.define("Bool", NType::Base("Bool".into()));
        global.define("String", NType::Base("String".into()));
        global.define("Timestamp", NType::Base("Timestamp".into()));
        global.define("Loss", NType::Base("Loss".into()));
        global.define("Dataset", NType::Base("Dataset".into()));

        Self { scopes: vec![global] }
    }

    fn push(&mut self) { self.scopes.push(Scope::new()); }
    fn pop(&mut self) -> Scope { self.scopes.pop().unwrap_or_else(|| Scope::new()) }

    fn define(&mut self, name: &str, ty: NType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.define(name, ty);
        }
    }

    pub fn lookup(&self, name: &str) -> Option<NType> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.lookup(name) { return Some(ty.clone()); }
        }
        None
    }

    fn record_mutation(&mut self, target: &str) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.mutations.push(target.to_string());
        }
    }

    fn record_uncertain_access(&mut self, name: &str, span: Span) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.record_uncertain_access(name, span);
        }
    }

    fn record_uncertain_confidence_checked(&mut self, name: &str) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.record_uncertain_confidence_checked(name);
        }
    }
}

// ═══════════════════════════════════════════
//  Type Checker
// ═══════════════════════════════════════════

pub struct TypeChecker {
    pub result: CompileResult,
    symbols: SymbolTable,
    unifier: UnificationEnv,
    model_types: HashMap<String, NType>,
}

impl TypeChecker {
    pub fn new(filename: &str) -> Self {
        Self {
            result: CompileResult::new(filename),
            symbols: SymbolTable::new(),
            unifier: UnificationEnv::default(),
            model_types: HashMap::new(),
        }
    }

    pub fn lookup(&self, name: &str) -> Option<NType> {
        self.symbols.lookup(name)
    }

    pub fn check(&mut self, program: &Program) {
        // Phase 1: register all top-level declarations
        for tl in &program.top_levels {
            self.register_top_level(tl);
        }
        // Phase 2: type-check all bodies
        for tl in &program.top_levels {
            self.check_top_level(tl);
        }
    }

    // ── Phase 1: Registration ──

    fn register_top_level(&mut self, tl: &TopLevel) {
        match tl {
            TopLevel::Fn(f) => {
                let fn_ty = self.fn_to_type(f);
                self.symbols.define(&f.name, fn_ty);
            }
            TopLevel::Model(m) => {
                let mut fields = HashMap::new();
                let mut methods = HashMap::new();
                for f in &m.fields {
                    fields.insert(f.name.clone(), type_from_ast(&f.type_ann));
                }
                for p in &m.params {
                    if let Some(ref ta) = p.type_ann {
                        fields.insert(p.name.clone(), type_from_ast(ta));
                    }
                }
                for met in &m.methods {
                    methods.insert(met.name.clone(), self.fn_to_type(met));
                }
                let model_ty = NType::Model(m.name.clone(), fields, methods);
                self.model_types.insert(m.name.clone(), model_ty.clone());
                // Constructor
                let params: Vec<NType> = m.params.iter().map(|p| {
                    p.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any)
                }).collect();
                self.symbols.define(&m.name, NType::Fn_(params, Box::new(model_ty), None));
            }
            TopLevel::Layer(l) => {
                let mut fields = HashMap::new();
                let mut methods = HashMap::new();
                for f in &l.fields { fields.insert(f.name.clone(), type_from_ast(&f.type_ann)); }
                for p in &l.params {
                    if let Some(ref ta) = p.type_ann {
                        fields.insert(p.name.clone(), type_from_ast(ta));
                    }
                }
                for met in &l.methods { methods.insert(met.name.clone(), self.fn_to_type(met)); }
                let model_ty = NType::Model(l.name.clone(), fields, methods);
                self.model_types.insert(l.name.clone(), model_ty.clone());
                let params: Vec<NType> = l.params.iter().map(|p| {
                    p.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any)
                }).collect();
                self.symbols.define(&l.name, NType::Fn_(params, Box::new(model_ty), None));
            }
            TopLevel::Causal(c) => {
                let mut vars = Vec::new();
                for edge in &c.edges {
                    for s in &edge.sources { if !vars.contains(s) { vars.push(s.clone()); } }
                    if let Some(ref t) = edge.target { if !vars.contains(t) { vars.push(t.clone()); } }
                }
                let cm_ty = NType::CausalModel(c.name.clone(), vars);
                self.symbols.define(&c.name, cm_ty);
            }
            TopLevel::Agent(a) => {
                let mut fields = HashMap::new();
                let mut methods = HashMap::new();
                for f in &a.fields { fields.insert(f.name.clone(), type_from_ast(&f.type_ann)); }
                for p in &a.params {
                    if let Some(ref ta) = p.type_ann {
                        fields.insert(p.name.clone(), type_from_ast(ta));
                    }
                }
                for met in &a.methods { methods.insert(met.name.clone(), self.fn_to_type(met)); }
                let agent_ty = NType::Model(a.name.clone(), fields, methods);
                self.model_types.insert(a.name.clone(), agent_ty.clone());
                self.symbols.define(&a.name, agent_ty);
            }
            TopLevel::Let(l) => {
                let ty = l.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any);
                self.symbols.define(&l.name, ty);
            }
            TopLevel::Import(imp) => {
                for name in &imp.names { self.symbols.define(name, NType::Any); }
                if let Some(ref alias) = imp.alias { self.symbols.define(alias, NType::Any); }
            }
            _ => {}
        }
    }

    fn fn_to_type(&self, f: &FnDecl) -> NType {
        let params: Vec<NType> = f.params.iter().map(|p| {
            p.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any)
        }).collect();
        let ret = f.return_type.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Void);
        NType::Fn_(params, Box::new(ret), None)
    }

    // ── Phase 2: Checking ──

    fn check_top_level(&mut self, tl: &TopLevel) {
        match tl {
            TopLevel::Fn(f) => self.check_fn(f, None),
            TopLevel::Model(m) => {
                let self_ty = self.model_types.get(&m.name).cloned();
                self.symbols.push();
                if let Some(ref st) = self_ty { self.symbols.define("self", st.clone()); }
                for p in &m.params {
                    let ty = p.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any);
                    self.symbols.define(&p.name, ty);
                }
                for f in &m.fields {
                    self.symbols.define(&f.name, type_from_ast(&f.type_ann));
                }
                for method in &m.methods { self.check_fn(method, self_ty.as_ref()); }
                self.symbols.pop();
            }
            TopLevel::Layer(l) => {
                let self_ty = self.model_types.get(&l.name).cloned();
                self.symbols.push();
                if let Some(ref st) = self_ty { self.symbols.define("self", st.clone()); }
                for p in &l.params {
                    let ty = p.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any);
                    self.symbols.define(&p.name, ty);
                }
                for method in &l.methods { self.check_fn(method, self_ty.as_ref()); }
                self.symbols.pop();
            }
            TopLevel::Agent(a) => {
                let self_ty = self.model_types.get(&a.name).cloned();
                self.symbols.push();
                if let Some(ref st) = self_ty { self.symbols.define("self", st.clone()); }
                for method in &a.methods { self.check_fn(method, self_ty.as_ref()); }
                self.symbols.pop();
            }
            TopLevel::Let(l) => {
                let inferred = self.infer_expr(&l.value);
                if let Some(ref ta) = l.type_ann {
                    let declared = type_from_ast(ta);
                    if !types_compatible(&declared, &inferred) && !matches!(inferred, NType::Any) {
                        self.result.add_error(NeuronError::new(
                            ErrorCode::TypeMismatch,
                            format!("Variable '{}' declared as {} but initialized with {}", l.name, declared.display(), inferred.display()),
                            l.span.clone(),
                        ).with_expected(&declared.display()).with_actual(&inferred.display()));
                    }
                    self.symbols.define(&l.name, declared);
                } else {
                    self.symbols.define(&l.name, inferred);
                }
            }
            TopLevel::Meta(m) => self.check_fn(&m.func, None),
            TopLevel::Expr(e) => { self.infer_expr(&e.expr); }
            TopLevel::Update(u) => {
                self.symbols.record_mutation(&u.target);
                self.infer_expr(&u.expr);
            }
            _ => {}
        }
    }

    fn check_fn(&mut self, f: &FnDecl, self_ty: Option<&NType>) {
        self.symbols.push();
        if let Some(st) = self_ty { self.symbols.define("self", st.clone()); }
        for p in &f.params {
            let ty = p.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any);
            self.symbols.define(&p.name, ty);
        }
        for stmt in &f.body { self.check_stmt(stmt); }

        // Effect checking
        let scope = self.symbols.pop();
        
        // Uncertainty confidence check warnings
        for (name, span) in &scope.uncertain_accessed {
            if !scope.uncertain_confidence_checked.contains(name) {
                self.result.add_warning(uncertainty_ignored_warning(span.clone(), name));
            }
        }

        if !scope.mutations.is_empty() {
            if let Some(ref eff) = f.effect_clause {
                let has_mut = eff.effects.iter().any(|e| e.kind == "Mut");
                if !has_mut {
                    let missing: Vec<String> = scope.mutations.iter().map(|m| format!("Mut[{}]", m)).collect();
                    self.result.add_error(effect_undeclared_error(f.span.clone(), &f.name, &missing));
                }
            } else {
                let missing: Vec<String> = scope.mutations.iter().map(|m| format!("Mut[{}]", m)).collect();
                self.result.add_error(effect_undeclared_error(f.span.clone(), &f.name, &missing));
            }
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(l) => {
                let inferred = self.infer_expr(&l.value);
                if let Some(ref ta) = l.type_ann {
                    let declared = type_from_ast(ta);
                    if !types_compatible(&declared, &inferred) && !matches!(inferred, NType::Any) {
                        self.result.add_error(NeuronError::new(
                            ErrorCode::TypeMismatch,
                            format!("Variable '{}' declared as {} but initialized with {}", l.name, declared.display(), inferred.display()),
                            l.span.clone(),
                        ).with_expected(&declared.display()).with_actual(&inferred.display()));
                    }
                    self.symbols.define(&l.name, declared);
                } else {
                    self.symbols.define(&l.name, inferred);
                }
            }
            Stmt::For(f) => {
                let iter_ty = self.infer_expr(&f.iter_expr);
                let elem_ty = match &iter_ty {
                    NType::List(inner) => *inner.clone(),
                    _ => NType::Any,
                };
                self.symbols.push();
                self.symbols.define(&f.var, elem_ty);
                for s in &f.body { self.check_stmt(s); }
                self.symbols.pop();
            }
            Stmt::If(i) => {
                let cond_ty = self.infer_expr(&i.cond);
                if !matches!(cond_ty, NType::Base(ref n) if n == "Bool") && !matches!(cond_ty, NType::Any) {
                    self.result.add_error(NeuronError::new(
                        ErrorCode::TypeMismatch, "If condition must be Bool", i.span.clone(),
                    ).with_actual(&cond_ty.display()));
                }
                for s in &i.then_body { self.check_stmt(s); }
                for s in &i.else_body { self.check_stmt(s); }
            }
            Stmt::Return(r) => { self.infer_expr(&r.value); }
            Stmt::Update(u) => {
                self.symbols.record_mutation(&u.target);
                self.infer_expr(&u.expr);
            }
            Stmt::Expr(e) => { self.infer_expr(&e.expr); }
            Stmt::Constraint(c) => { self.infer_expr(&c.expr); }
        }
    }

    // ── Expression type inference ──

    fn infer_expr(&mut self, expr: &Expr) -> NType {
        match expr {
            Expr::IntLit(_, _) => NType::Base("Int".into()),
            Expr::FloatLit(_, _) => NType::Base("Float".into()),
            Expr::BoolLit(_, _) => NType::Base("Bool".into()),
            Expr::StringLit(_, _) => NType::Base("String".into()),
            Expr::Ident(name, span) => {
                let ty = self.symbols.lookup(name).unwrap_or(NType::Any);
                if let NType::Uncertain(_) = ty {
                    self.symbols.record_uncertain_access(name, span.clone());
                }
                ty
            }
            Expr::Self_(_) => self.symbols.lookup("self").unwrap_or(NType::Any),

            Expr::BinOp(b) => self.infer_binop(b),
            Expr::UnaryOp(u) => {
                let inner = self.infer_expr(&u.operand);
                match u.op {
                    UnaryOp::Neg => inner,
                    UnaryOp::Not => NType::Base("Bool".into()),
                }
            }
            Expr::FnCall(c) => self.infer_fn_call(c),
            Expr::Dot(d) => {
                if d.field == "confidence" {
                    if let Expr::Ident(ref name, _) = d.obj {
                        self.symbols.record_uncertain_confidence_checked(name);
                    }
                }
                self.infer_dot(d)
            }
            Expr::Index(idx) => {
                let obj_ty = self.infer_expr(&idx.obj);
                // Indexing a tensor returns a tensor (with reduced dims)
                if matches!(obj_ty, NType::Tensor(_)) { obj_ty.clone() } else { NType::Any }
            }
            Expr::Grad(g) => {
                let inner = self.infer_expr(&g.expr);
                // grad(loss) returns a tensor-like type
                if inner.is_tensor() { inner } else { NType::Tensor(vec![]) }
            }
            Expr::StopGrad(expr, _) => {
                self.infer_expr(expr)
            }
            Expr::Do(_d) => {
                NType::Causal(Box::new(NType::Any), "intervened".into())
            }
            Expr::Observe(_o) => {
                NType::Causal(Box::new(NType::Any), "observed".into())
            }
            Expr::Explain(e) => {
                let inner = self.infer_expr(&e.expr);
                NType::Tuple(vec![inner, NType::Explanation])
            }
            Expr::Merge(m) => {
                let left = self.infer_expr(&m.left);
                left // Merge returns same type as left operand
            }
            Expr::Forget(f) => {
                self.infer_expr(&f.obj) // Forget returns same type
            }
            Expr::List(elems, _) => {
                if elems.is_empty() {
                    NType::List(Box::new(NType::Any))
                } else {
                    let inner = self.infer_expr(&elems[0]);
                    NType::List(Box::new(inner))
                }
            }
            Expr::ListComp(lc) => {
                let inner = self.infer_expr(&lc.expr);
                NType::List(Box::new(inner))
            }
            Expr::Tuple(elems, _) => {
                let types: Vec<NType> = elems.iter().map(|e| self.infer_expr(e)).collect();
                NType::Tuple(types)
            }
            Expr::SearchExpr(s) => {
                self.infer_expr(&s.space);
                self.infer_expr(&s.evaluate);
                NType::Any // SearchResult
            }
            Expr::RecallExpr(r) => {
                let mem_ty = self.infer_expr(&r.memory);
                match mem_ty {
                    NType::EpisodicMemory(inner) | NType::SemanticMemory(inner) | NType::Memory(inner) => {
                        NType::List(inner)
                    }
                    _ => NType::List(Box::new(NType::Any)),
                }
            }
            Expr::StoreExpr(s) => {
                self.infer_expr(&s.memory);
                self.infer_expr(&s.item);
                NType::Void
            }
        }
    }

    fn infer_binop(&mut self, b: &BinOpExpr) -> NType {
        let left = self.infer_expr(&b.left);
        let right = self.infer_expr(&b.right);

        // ── Uncertainty mismatch ──
        if matches!((&left, &right), (NType::Uncertain(_), NType::Random(_)) | (NType::Random(_), NType::Uncertain(_))) {
            self.result.add_error(uncertainty_mismatch_error(
                b.span.clone(),
                if matches!(left, NType::Uncertain(_)) { "Uncertain" } else { "Random" },
                if matches!(right, NType::Random(_)) { "Random" } else { "Uncertain" },
            ));
            return NType::Any;
        }

        // ── Causal type mismatch ──
        if let (NType::Causal(_, ref m1), NType::Causal(_, ref m2)) = (&left, &right) {
            if m1 != m2 {
                self.result.add_error(causal_type_mismatch_error(b.span.clone(), m1, m2));
                return NType::Any;
            }
        }

        // ── Tensor operations ──
        if b.op == BinOp::MatMul {
            if let (NType::Tensor(ref da), NType::Tensor(ref db)) = (&left, &right) {
                return self.check_matmul(da, db, &b.span);
            }
        }

        if matches!(b.op, BinOp::Add | BinOp::Sub) {
            if let (NType::Tensor(ref da), NType::Tensor(ref db)) = (&left, &right) {
                self.check_elementwise(da, db, &b.span);
                return left;
            }
            // Uncertain propagation
            if let (NType::Uncertain(ref inner), _) = (&left, &right) {
                return NType::Uncertain(inner.clone());
            }
        }

        // Comparison operators
        if matches!(b.op, BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte) {
            return NType::Base("Bool".into());
        }
        if matches!(b.op, BinOp::And | BinOp::Or) {
            return NType::Base("Bool".into());
        }

        // Numeric promotion
        if left.is_numeric() && right.is_numeric() {
            if matches!(left, NType::Base(ref n) if n == "Float") || matches!(right, NType::Base(ref n) if n == "Float") {
                return NType::Base("Float".into());
            }
            return NType::Base("Int".into());
        }

        left
    }

    fn check_matmul(&mut self, a_dims: &[Dim], b_dims: &[Dim], span: &Span) -> NType {
        if a_dims.iter().any(|d| matches!(d, Dim::Dynamic)) || b_dims.iter().any(|d| matches!(d, Dim::Dynamic)) {
            self.result.add_warning(dynamic_dim_warning(span.clone()));
        }
        if a_dims.len() < 2 || b_dims.len() < 2 {
            // Allow for compatibility — return tensor
            return NType::Tensor(vec![]);
        }
        let a_inner = &a_dims[a_dims.len() - 1];
        let b_outer = &b_dims[b_dims.len() - 2];
        if !self.unifier.unify(a_inner, b_outer) {
            let a_str = match a_inner { Dim::Static(v) => v.to_string(), Dim::Symbolic(s) => s.clone(), _ => "?".into() };
            let b_str = match b_outer { Dim::Static(v) => v.to_string(), Dim::Symbolic(s) => s.clone(), _ => "?".into() };
            self.result.add_error(shape_mismatch_error(
                span.clone(),
                &format!("inner dim {}", b_str),
                &format!("inner dim {}", a_str),
                "matrix multiply (@)",
            ));
        }
        // Result dims: a[:-1] + b[-1:]
        let mut result_dims: Vec<Dim> = a_dims[..a_dims.len()-1].to_vec();
        result_dims.push(b_dims[b_dims.len()-1].clone());
        NType::Tensor(result_dims)
    }

    fn check_elementwise(&mut self, a: &[Dim], b: &[Dim], span: &Span) {
        if a.iter().any(|d| matches!(d, Dim::Dynamic)) || b.iter().any(|d| matches!(d, Dim::Dynamic)) {
            self.result.add_warning(dynamic_dim_warning(span.clone()));
        }
        if a.is_empty() || b.is_empty() {
            return;
        }
        if a.len() != b.len() {
            self.result.add_error(shape_mismatch_error(
                span.clone(),
                &format!("{} dims", a.len()),
                &format!("{} dims", b.len()),
                "element-wise operation",
            ));
            return;
        }
        for (da, db) in a.iter().zip(b.iter()) {
            if !self.unifier.unify(da, db) {
                let sa = match da { Dim::Static(v) => v.to_string(), Dim::Symbolic(s) => s.clone(), _ => "?".into() };
                let sb = match db { Dim::Static(v) => v.to_string(), Dim::Symbolic(s) => s.clone(), _ => "?".into() };
                self.result.add_error(shape_mismatch_error(span.clone(), &sa, &sb, "element-wise operation"));
            }
        }
    }

    fn infer_fn_call(&mut self, c: &FnCallExpr) -> NType {
        let callee_ty = self.infer_expr(&c.callee);
        if let Expr::Dot(ref d) = c.callee {
            if d.field == "before" || d.field == "after" || d.field == "snapshot" {
                return callee_ty;
            }
        }
        let callee_name = match &c.callee {
            Expr::Ident(ref name, _) => Some(name.as_str()),
            _ => None,
        };
        let is_variadic_shape_creator = match callee_name {
            Some("zeros") | Some("glorot") | Some("ones") | Some("randn") => true,
            _ => false,
        };

        let mut is_method_call = false;
        if let Expr::Dot(ref d) = c.callee {
            let obj_ty = self.infer_expr(&d.obj);
            if let NType::Model(_, _, ref methods) = obj_ty {
                if methods.contains_key(&d.field) {
                    is_method_call = true;
                }
            }
        }

        match callee_ty {
            NType::Fn_(ref params, ref ret, _) => {
                if is_variadic_shape_creator {
                    // Variadic shape creator: all arguments should be integers, lists, or tuples
                    for (i, arg) in c.args.iter().enumerate() {
                        let arg_ty = self.infer_expr(&arg.value);
                        if !types_compatible(&NType::Base("Int".into()), &arg_ty) && !matches!(arg_ty, NType::List(_) | NType::Tuple(_) | NType::Any) {
                            self.result.add_error(NeuronError::new(
                                ErrorCode::TypeMismatch,
                                format!("Argument {} of shape creator must be an integer, got {}", i + 1, arg_ty.display()),
                                c.span.clone(),
                            ));
                        }
                    }
                } else {
                    let expected_args_len = if is_method_call { params.len().saturating_sub(1) } else { params.len() };
                    if c.args.len() != expected_args_len {
                        self.result.add_error(NeuronError::new(
                            ErrorCode::TypeMismatch,
                            format!("Function call expected {} arguments but got {}", expected_args_len, c.args.len()),
                            c.span.clone(),
                        ));
                    } else {
                        let params_to_check = if is_method_call { &params[1..] } else { &params[..] };
                        for (i, (param_ty, arg)) in params_to_check.iter().zip(c.args.iter()).enumerate() {
                            let arg_ty = self.infer_expr(&arg.value);
                            if !types_compatible(param_ty, &arg_ty) {
                                self.result.add_error(NeuronError::new(
                                    ErrorCode::TypeMismatch,
                                    format!("Argument {} type mismatch: expected {} but got {}", i + 1, param_ty.display(), arg_ty.display()),
                                    c.span.clone(),
                                ).with_expected(&param_ty.display()).with_actual(&arg_ty.display()));
                            }
                            
                            // Rule 2: Temporal direction track check on calls
                            if let (NType::Temporal(_, ref expected_dir), NType::Temporal(_, ref found_dir)) = (param_ty, &arg_ty) {
                                if expected_dir == "past_to_future" && found_dir == "future_to_past" {
                                    self.result.add_error(temporal_leak_error(
                                        c.span.clone(),
                                        found_dir,
                                        expected_dir,
                                    ));
                                }
                            }
                        }
                    }
                }
                *ret.clone()
            }
            NType::Model(_, _, _) => callee_ty, // Constructor returns the model type
            _ => NType::Any,
        }
    }

    fn infer_dot(&mut self, d: &DotExpr) -> NType {
        let obj_ty = self.infer_expr(&d.obj);

        // Temporal direction checking
        if let NType::Temporal(ref inner, ref dir) = obj_ty {
            match d.field.as_str() {
                "before" => return NType::Temporal(inner.clone(), dir.clone()),
                "after" => {
                    let new_dir = if dir == "past_to_future" { "future_to_past" } else { "past_to_future" };
                    return NType::Temporal(inner.clone(), new_dir.into());
                }
                "snapshot" => return *inner.clone(),
                _ => {}
            }
        }

        // Uncertain field access
        if let NType::Uncertain(ref inner) = obj_ty {
            match d.field.as_str() {
                "value" => return *inner.clone(),
                "confidence" => return NType::Base("Float".into()),
                "std" => return NType::Base("Float".into()),
                "bounds" => return NType::Tuple(vec![NType::Base("Float".into()), NType::Base("Float".into())]),
                _ => {}
            }
        }

        // CausalModel methods
        if let NType::CausalModel(_, _) = obj_ty {
            match d.field.as_str() {
                "observe" => return NType::Causal(Box::new(NType::Any), "observed".into()),
                "intervene" => return NType::Causal(Box::new(NType::Any), "intervened".into()),
                _ => {}
            }
        }

        // Model field/method lookup
        if let NType::Model(_, ref fields, ref methods) = obj_ty {
            if let Some(ty) = fields.get(&d.field) { return ty.clone(); }
            if let Some(ty) = methods.get(&d.field) { return ty.clone(); }
        }

        NType::Any
    }
}
