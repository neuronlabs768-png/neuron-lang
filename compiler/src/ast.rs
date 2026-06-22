/// NEURON AST — all node types for the abstract syntax tree.
///
/// Every node carries a `Span` for error reporting. The AST is produced
/// by the parser and consumed by the type checker and IR lowering passes.

use crate::token::Span;

// ═══════════════════════════════════════════
//  Program root
// ═══════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct Program {
    pub top_levels: Vec<TopLevel>,
}

// ═══════════════════════════════════════════
//  Top-level declarations
// ═══════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum TopLevel {
    Model(ModelDecl),
    Layer(LayerDecl),
    Fn(FnDecl),
    Causal(CausalDecl),
    Agent(AgentDecl),
    Meta(MetaDecl),
    Import(ImportStmt),
    Let(LetStmt),
    Constraint(ConstraintDecl),
    Expr(ExprStmt),
    Update(UpdateStmt),
}

// ── Model declaration ──

#[derive(Debug, Clone)]
pub struct ModelDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub annotations: Vec<Annotation>,
    pub fields: Vec<FieldDecl>,
    pub methods: Vec<FnDecl>,
    pub forget_decls: Vec<ForgetDecl>,
    pub span: Span,
}

// ── Layer declaration ──

#[derive(Debug, Clone)]
pub struct LayerDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub annotations: Vec<Annotation>,
    pub fields: Vec<FieldDecl>,
    pub methods: Vec<FnDecl>,
    pub span: Span,
}

// ── Agent declaration (AGI) ──

#[derive(Debug, Clone)]
pub struct AgentDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub annotations: Vec<Annotation>,
    pub fields: Vec<FieldDecl>,
    pub methods: Vec<FnDecl>,
    pub span: Span,
}

// ── Meta declaration (AGI — safe self-modification) ──

#[derive(Debug, Clone)]
pub struct MetaDecl {
    pub func: FnDecl,
    pub span: Span,
}

// ── Function declaration ──

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub effect_clause: Option<EffectType>,
    pub annotations: Vec<Annotation>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

// ── Causal declaration ──

#[derive(Debug, Clone)]
pub struct CausalDecl {
    pub name: String,
    pub options: Vec<CausalOpt>,
    pub edges: Vec<CausalEdge>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct CausalOpt {
    pub name: String,
    pub value: String,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct CausalEdge {
    pub kind: CausalEdgeKind,
    pub sources: Vec<String>,
    pub target: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CausalEdgeKind {
    Simple,     // source -> target
    Fixed,      // fixed: source -> target
    Discover,   // discover: [sources] -> target
    Variables,  // variables: [names]
}

// ── Import statement ──

#[derive(Debug, Clone)]
pub struct ImportStmt {
    pub module: String,
    pub names: Vec<String>,
    pub alias: Option<String>,
    pub is_python: bool,
    pub span: Span,
}

// ── Helpers ──

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub type_ann: Option<TypeExpr>,
    pub default: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Annotation {
    pub name: String,
    pub args: Vec<AnnotationArg>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct AnnotationArg {
    pub key: Option<String>,
    pub value: AnnotationValue,
}

#[derive(Debug, Clone)]
pub enum AnnotationValue {
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

#[derive(Debug, Clone)]
pub struct FieldDecl {
    pub name: String,
    pub type_ann: TypeExpr,
    pub default: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ForgetDecl {
    pub type_ann: TypeExpr,
    pub description: String,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ConstraintDecl {
    pub expr: Expr,
    pub span: Span,
}

// ═══════════════════════════════════════════
//  Statements
// ═══════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum Stmt {
    Let(LetStmt),
    For(ForStmt),
    If(IfStmt),
    Return(ReturnStmt),
    Update(UpdateStmt),
    Expr(ExprStmt),
    Constraint(ConstraintDecl),
}

#[derive(Debug, Clone)]
pub struct LetStmt {
    pub name: String,
    pub type_ann: Option<TypeExpr>,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ForStmt {
    pub var: String,
    pub iter_expr: Expr,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct IfStmt {
    pub cond: Expr,
    pub then_body: Vec<Stmt>,
    pub else_body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ReturnStmt {
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct UpdateStmt {
    pub target: String,
    pub expr: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ExprStmt {
    pub expr: Expr,
    pub span: Span,
}

// ═══════════════════════════════════════════
//  Expressions
// ═══════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum Expr {
    IntLit(i64, Span),
    FloatLit(f64, Span),
    BoolLit(bool, Span),
    StringLit(String, Span),
    Ident(String, Span),
    Self_(Span),

    BinOp(Box<BinOpExpr>),
    UnaryOp(Box<UnaryOpExpr>),
    FnCall(Box<FnCallExpr>),
    Index(Box<IndexExpr>),
    Dot(Box<DotExpr>),

    Grad(Box<GradExpr>),
    StopGrad(Box<Expr>, Span),
    Do(Box<DoExpr>),
    Observe(Box<ObserveExpr>),
    Explain(Box<ExplainExpr>),
    Merge(Box<MergeExpr>),
    Forget(Box<ForgetExpr>),

    List(Vec<Expr>, Span),
    ListComp(Box<ListCompExpr>),
    Tuple(Vec<Expr>, Span),

    // AGI
    SearchExpr(Box<SearchExpr>),
    RecallExpr(Box<RecallExpr>),
    StoreExpr(Box<StoreExpr>),
}

impl Expr {
    pub fn span(&self) -> &Span {
        match self {
            Self::IntLit(_, s) | Self::FloatLit(_, s) | Self::BoolLit(_, s)
            | Self::StringLit(_, s) | Self::Ident(_, s) | Self::Self_(s)
            | Self::List(_, s) | Self::Tuple(_, s) => s,
            Self::BinOp(e) => &e.span,
            Self::UnaryOp(e) => &e.span,
            Self::FnCall(e) => &e.span,
            Self::Index(e) => &e.span,
            Self::Dot(e) => &e.span,
            Self::Grad(e) => &e.span,
            Self::StopGrad(_, s) => s,
            Self::Do(e) => &e.span,
            Self::Observe(e) => &e.span,
            Self::Explain(e) => &e.span,
            Self::Merge(e) => &e.span,
            Self::Forget(e) => &e.span,
            Self::ListComp(e) => &e.span,
            Self::SearchExpr(e) => &e.span,
            Self::RecallExpr(e) => &e.span,
            Self::StoreExpr(e) => &e.span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BinOpExpr {
    pub left: Expr,
    pub op: BinOp,
    pub right: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod, MatMul,
    Eq, Neq, Lt, Gt, Lte, Gte,
    And, Or,
}

impl BinOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Add => "+", Self::Sub => "-", Self::Mul => "*",
            Self::Div => "/", Self::Mod => "%", Self::MatMul => "@",
            Self::Eq => "==", Self::Neq => "!=", Self::Lt => "<",
            Self::Gt => ">", Self::Lte => "<=", Self::Gte => ">=",
            Self::And => "&&", Self::Or => "||",
        }
    }
}

#[derive(Debug, Clone)]
pub struct UnaryOpExpr {
    pub op: UnaryOp,
    pub operand: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
pub struct FnCallExpr {
    pub callee: Expr,
    pub args: Vec<CallArg>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct CallArg {
    pub name: Option<String>,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub struct IndexExpr {
    pub obj: Expr,
    pub indices: Vec<IndexItem>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum IndexItem {
    Expr(Expr),
    Slice { start: Option<Expr>, end: Option<Expr> },
    Full, // :
}

#[derive(Debug, Clone)]
pub struct DotExpr {
    pub obj: Expr,
    pub field: String,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct GradExpr {
    pub expr: Expr,
    pub wrt: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct DoExpr {
    pub assignments: Vec<(String, Expr)>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ObserveExpr {
    pub obj: Expr,
    pub assignments: Vec<(String, Expr)>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ExplainExpr {
    pub expr: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct MergeExpr {
    pub left: Expr,
    pub right: Expr,
    pub strategy: Option<Expr>,
    pub preserve: Vec<String>,
    pub forget_clauses: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ForgetExpr {
    pub obj: Expr,
    pub args: Vec<CallArg>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ListCompExpr {
    pub expr: Expr,
    pub var: String,
    pub iter: Expr,
    pub span: Span,
}

// AGI expression nodes

#[derive(Debug, Clone)]
pub struct SearchExpr {
    pub space: Expr,
    pub evaluate: Expr,
    pub strategy: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct RecallExpr {
    pub memory: Expr,
    pub query: Expr,
    pub k: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct StoreExpr {
    pub memory: Expr,
    pub item: Expr,
    pub span: Span,
}

// ═══════════════════════════════════════════
//  Type expressions
// ═══════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum TypeExpr {
    Base(String, Span),                          // Int, Float, Bool, etc.
    Tensor(Vec<DimExpr>, Span),                  // Tensor[B, 784]
    Uncertain(Box<TypeExpr>, Span),              // Uncertain[T]
    Random(Box<TypeExpr>, Span),                 // Random[T]
    Prob(Box<TypeExpr>, Span),                   // Prob[T]
    Temporal(Box<TypeExpr>, String, Span),        // Temporal[T, "past_to_future"]
    Causal(Box<TypeExpr>, String, Span),          // Causal[T, "observed"]
    Learnable(String, Option<Box<Expr>>, Span),   // Learnable[FnType, base=expr]
    ListType(Box<TypeExpr>, Span),               // List[T]
    OptionType(Box<TypeExpr>, Span),             // Option[T]
    Fn(Vec<TypeExpr>, Box<TypeExpr>, Span),       // Fn(A, B) -> C
    // AGI types
    Memory(Box<TypeExpr>, Span),                 // Memory[T]
    EpisodicMemory(Box<TypeExpr>, Span),         // EpisodicMemory[T]
    SemanticMemory(Box<TypeExpr>, Span),         // SemanticMemory[T]
    WorkingMemory(Box<TypeExpr>, Option<Box<Expr>>, Span), // WorkingMemory[T, N]
    RewardType(Box<TypeExpr>, Span),             // Reward[T]
    UserDefined(String, Span),                   // TransformerBlock, etc.
}

impl TypeExpr {
    pub fn span(&self) -> &Span {
        match self {
            Self::Base(_, s) | Self::Tensor(_, s) | Self::Uncertain(_, s)
            | Self::Random(_, s) | Self::Prob(_, s) | Self::Temporal(_, _, s)
            | Self::Causal(_, _, s) | Self::Learnable(_, _, s)
            | Self::ListType(_, s) | Self::OptionType(_, s) | Self::Fn(_, _, s)
            | Self::Memory(_, s) | Self::EpisodicMemory(_, s)
            | Self::SemanticMemory(_, s) | Self::WorkingMemory(_, _, s)
            | Self::RewardType(_, s) | Self::UserDefined(_, s) => s,
        }
    }
}

#[derive(Debug, Clone)]
pub enum DimExpr {
    Static(i64),         // 784
    Symbolic(String),    // B
    Named(String, String), // B:batch
    Dynamic,             // ?
}

// ── Effect type ──

#[derive(Debug, Clone)]
pub struct EffectType {
    pub effects: Vec<EffectKind>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct EffectKind {
    pub kind: String,        // "IO", "Rand", "Mut"
    pub target: Option<String>, // For Mut[x]
}
