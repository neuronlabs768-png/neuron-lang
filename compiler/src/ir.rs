
/// Unique identifier for an IR value (SSA-style).
pub type ValueId = usize;

/// The full IR program — a sequence of functions.
#[derive(Debug, Clone)]
pub struct IRProgram {
    pub functions: Vec<IRFunction>,
    pub globals: Vec<IRGlobal>,
}

#[derive(Debug, Clone)]
pub struct IRGlobal {
    pub name: String,
    pub value: IRConst,
    pub ty: IRType,
}

pub type BlockId = usize;

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instructions: Vec<IRNode>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone)]
pub enum Terminator {
    Jump(BlockId),
    Branch { cond: ValueId, true_block: BlockId, false_block: BlockId },
    Return(Option<ValueId>),
}

#[derive(Debug, Clone)]
pub struct IRFunction {
    pub name: String,
    pub params: Vec<IRParam>,
    pub return_type: IRType,
    pub blocks: Vec<BasicBlock>,
    pub entry: BlockId,
    pub is_differentiable: bool,
    pub effects: Vec<IREffect>,
}

#[derive(Debug, Clone)]
pub struct IRParam {
    pub name: String,
    pub ty: IRType,
    pub id: ValueId,
}

/// A single IR instruction.
#[derive(Debug, Clone)]
pub struct IRNode {
    pub id: ValueId,
    pub op: IROp,
    pub inputs: Vec<ValueId>,
    pub output_type: IRType,
    pub output_shape: Vec<i64>,
    /// The gradient function for this op (index into the function table, or built-in)
    pub grad_fn: Option<GradFn>,
    /// Device target for this operation
    pub device: DeviceTarget,
    /// Temporal direction tag (if applicable)
    pub temporal_dir: Option<String>,
    /// Effects this operation has
    pub effects: Vec<IREffect>,
}

/// IR operations — the instruction set of NEURON.
#[derive(Debug, Clone)]
pub enum IROp {
    // ── Constants ──
    Const(IRConst),
    
    // ── Tensor creation ──
    Zeros(Vec<i64>),
    Ones(Vec<i64>),
    Glorot(Vec<i64>),
    Randn(Vec<i64>),
    
    // ── Tensor arithmetic ──
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    MatMul,
    
    // ── Comparison and list helper ──
    Lt,
    Lte,
    Gt,
    Gte,
    Eq,
    Neq,
    ListLen,
    
    // ── Activations ──
    ReLU,
    GeLU,
    Sigmoid,
    Tanh,
    Softmax { dim: i64 },
    
    // ── Reductions ──
    Sum { dim: Option<i64> },
    Mean { dim: Option<i64> },
    Max { dim: Option<i64> },
    Min { dim: Option<i64> },
    
    // ── Shape operations ──
    Reshape(Vec<i64>),
    Transpose(usize, usize),
    Slice(Vec<SliceSpec>),
    Concat { dim: i64 },
    Index,
    
    // ── Neural network layers ──
    Linear { in_features: i64, out_features: i64 },
    Conv2D { in_channels: i64, out_channels: i64, kernel: i64, stride: i64 },
    LayerNorm { dim: i64 },
    Dropout { p: f64 },
    Embedding { vocab: i64, dim: i64 },
    
    // ── Loss functions ──
    CrossEntropy,
    MSELoss,
    NLLLoss,
    KLDivergence,
    
    // ── Autograd ──
    Grad { wrt: Option<String> },
    Backward,
    StopGrad,
    
    // ── Optimizer step ──
    Adam { target: String, lr: f64, beta1: f64, beta2: f64 },
    SGD { target: String, lr: f64, momentum: f64 },
    AdamW { target: String, lr: f64, weight_decay: f64 },
    
    // ── Control flow ──
    Call { function: String },
    Return,
    Branch { cond: ValueId, then_block: usize, else_block: usize },
    Loop { var: String, iter: ValueId, body_block: usize },
    
    // ── Memory operations ──
    Load { name: String },
    Store { name: String },
    Alloc(IRType),
    Free,
    
    // ── Uncertainty ──
    UncertainWrap,     // (value, std) → Uncertain[T]
    UncertainValue,    // Uncertain[T] → T (extract value)
    UncertainStd,      // Uncertain[T] → Float (extract std)
    UncertainConfidence, // Uncertain[T] → Float
    RandomSample,      // Random[T] → T (explicit sample)
    
    // ── Temporal ──
    TemporalBefore { t: ValueId },
    TemporalAfter { t: ValueId },
    TemporalSnapshot { at: ValueId },
    TemporalCheckDir { expected: String },
    
    // ── Causal ──
    Observe,
    Intervene,
    CausalCheckMode { expected: String },
    
    // ── Interpretability ──
    Explain,
    
    // ── Merge / Forget ──
    MergeModels { strategy: String },
    ForgetTask { method: String, strength: f64 },
    
    // ── AGI operations ──
    MemoryStore,
    MemoryRecall { k: i64 },
    Search { strategy: String, max_iter: i64 },
    
    // ── Python bridge (FFI) ──
    PythonCall { module: String, function: String },
    
    // ── Effect tracking ──
    EffectCheck { expected: Vec<String> },
    
    // ── Misc ──
    Print,
    Input,
    EmbedString,
    GenerateReply,
    Nop,
}

#[derive(Debug, Clone)]
pub enum IRConst {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Tensor(Vec<f64>, Vec<i64>), // data, shape
}

#[derive(Debug, Clone)]
pub struct SliceSpec {
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub step: Option<i64>,
}

/// Gradient function specification for an IR node.
#[derive(Debug, Clone)]
pub enum GradFn {
    /// Built-in gradient (the runtime knows how to differentiate this op)
    Builtin,
    /// Custom gradient function (index into the function table)
    Custom(String),
    /// No gradient (opaque operation)
    None,
}

/// Device target for an operation.
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceTarget {
    CPU,
    CUDA(usize),
    Auto,
}

/// Effect kind for IR-level effect tracking.
#[derive(Debug, Clone)]
pub enum IREffect {
    IO,
    Rand,
    Mut(String),
}

/// IR type — simplified type representation for codegen.
#[derive(Debug, Clone)]
pub enum IRType {
    F32,
    F64,
    I32,
    I64,
    Bool,
    String,
    Tensor(Vec<i64>),       // shape
    Uncertain(Box<IRType>),
    Random(Box<IRType>),
    Temporal(Box<IRType>, String),
    Causal(Box<IRType>, String),
    List(Box<IRType>),
    Tuple(Vec<IRType>),
    Void,
    Any,
}

impl IRProgram {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            globals: Vec::new(),
        }
    }
}

impl IRFunction {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            return_type: IRType::Void,
            blocks: Vec::new(),
            entry: 0,
            is_differentiable: true,
            effects: Vec::new(),
        }
    }
}
