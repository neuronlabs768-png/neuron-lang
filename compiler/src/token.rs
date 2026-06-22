/// NEURON Token types and Token struct.
///
/// Every lexeme produced by the lexer is tagged with one of these types.

#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub line: usize,
    pub col: usize,
    pub len: usize,
}

impl Span {
    pub fn new(line: usize, col: usize, len: usize) -> Self {
        Self { line, col, len }
    }
}

impl std::fmt::Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub ty: TokenType,
    pub span: Span,
}

impl Token {
    pub fn new(ty: TokenType, line: usize, col: usize, len: usize) -> Self {
        Self {
            ty,
            span: Span::new(line, col, len),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    // ── Literals ──
    IntLit(i64),
    FloatLit(f64),
    StringLit(String),
    True,
    False,

    // ── Identifiers ──
    Ident(String),

    // ── Keywords ──
    Model,
    Layer,
    Fn,
    Let,
    Return,
    If,
    Else,
    For,
    In,
    Import,
    From,
    As,
    Self_,
    Causal,
    Update,
    By,
    Constraint,
    Forget,
    Strategy,
    Preserve,
    Fixed,
    Discover,
    Variables,
    Do,
    Observe,
    Grad,
    StopGrad,
    Explain,
    // AGI keywords
    Agent,
    Meta,
    Search,
    Recall,
    Store,
    Stream,
    Reward,

    // ── Type keywords ──
    TensorKw,
    UncertainKw,
    RandomKw,
    ProbKw,
    TemporalKw,
    CausalKw,
    LearnableKw,
    EffectKw,
    ListKw,
    OptionKw,
    IntKw,
    FloatKw,
    BoolKw,
    StringKw,
    TimestampKw,
    LossKw,
    DatasetKw,
    // AGI type keywords
    MemoryKw,
    EpisodicMemoryKw,
    SemanticMemoryKw,
    WorkingMemoryKw,
    RewardKw,
    ExperienceKw,

    // ── Operators ──
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    At,
    Dot,
    Eq,
    EqEq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    And,
    Or,
    Bang,
    Arrow,       // ->
    UnicodeArrow, // →
    Question,

    // ── Delimiters ──
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Colon,
    Comma,

    // ── Annotation ──
    Annotation(String),

    // ── Structural ──
    Newline,
    Indent,
    Dedent,
    Eof,
}

impl TokenType {
    /// Return the keyword string for display.
    pub fn name(&self) -> &str {
        match self {
            Self::IntLit(_) => "INT_LIT",
            Self::FloatLit(_) => "FLOAT_LIT",
            Self::StringLit(_) => "STRING_LIT",
            Self::True => "TRUE",
            Self::False => "FALSE",
            Self::Ident(_) => "IDENT",
            Self::Model => "MODEL",
            Self::Layer => "LAYER",
            Self::Fn => "FN",
            Self::Let => "LET",
            Self::Return => "RETURN",
            Self::If => "IF",
            Self::Else => "ELSE",
            Self::For => "FOR",
            Self::In => "IN",
            Self::Import => "IMPORT",
            Self::From => "FROM",
            Self::As => "AS",
            Self::Self_ => "SELF",
            Self::Causal => "CAUSAL",
            Self::Update => "UPDATE",
            Self::By => "BY",
            Self::Constraint => "CONSTRAINT",
            Self::Forget => "FORGET",
            Self::Strategy => "STRATEGY",
            Self::Preserve => "PRESERVE",
            Self::Fixed => "FIXED",
            Self::Discover => "DISCOVER",
            Self::Variables => "VARIABLES",
            Self::Do => "DO",
            Self::Observe => "OBSERVE",
            Self::Grad => "GRAD",
            Self::StopGrad => "STOP_GRAD",
            Self::Explain => "EXPLAIN",
            Self::Agent => "AGENT",
            Self::Meta => "META",
            Self::Search => "SEARCH",
            Self::Recall => "RECALL",
            Self::Store => "STORE",
            Self::Stream => "STREAM",
            Self::Reward => "REWARD",
            Self::TensorKw => "TENSOR",
            Self::UncertainKw => "UNCERTAIN",
            Self::RandomKw => "RANDOM",
            Self::ProbKw => "PROB",
            Self::TemporalKw => "TEMPORAL",
            Self::CausalKw => "CAUSAL_TYPE",
            Self::LearnableKw => "LEARNABLE",
            Self::EffectKw => "EFFECT",
            Self::ListKw => "LIST",
            Self::OptionKw => "OPTION",
            Self::IntKw => "INT",
            Self::FloatKw => "FLOAT",
            Self::BoolKw => "BOOL",
            Self::StringKw => "STRING",
            Self::TimestampKw => "TIMESTAMP",
            Self::LossKw => "LOSS",
            Self::DatasetKw => "DATASET",
            Self::MemoryKw => "MEMORY",
            Self::EpisodicMemoryKw => "EPISODIC_MEMORY",
            Self::SemanticMemoryKw => "SEMANTIC_MEMORY",
            Self::WorkingMemoryKw => "WORKING_MEMORY",
            Self::RewardKw => "REWARD_TYPE",
            Self::ExperienceKw => "EXPERIENCE",
            Self::Plus => "PLUS",
            Self::Minus => "MINUS",
            Self::Star => "STAR",
            Self::Slash => "SLASH",
            Self::Percent => "PERCENT",
            Self::At => "AT",
            Self::Dot => "DOT",
            Self::Eq => "EQ",
            Self::EqEq => "EQEQ",
            Self::Neq => "NEQ",
            Self::Lt => "LT",
            Self::Gt => "GT",
            Self::Lte => "LTE",
            Self::Gte => "GTE",
            Self::And => "AND",
            Self::Or => "OR",
            Self::Bang => "BANG",
            Self::Arrow => "ARROW",
            Self::UnicodeArrow => "UNICODE_ARROW",
            Self::Question => "QUESTION",
            Self::LParen => "LPAREN",
            Self::RParen => "RPAREN",
            Self::LBracket => "LBRACKET",
            Self::RBracket => "RBRACKET",
            Self::LBrace => "LBRACE",
            Self::RBrace => "RBRACE",
            Self::Colon => "COLON",
            Self::Comma => "COMMA",
            Self::Annotation(_) => "ANNOTATION",
            Self::Newline => "NEWLINE",
            Self::Indent => "INDENT",
            Self::Dedent => "DEDENT",
            Self::Eof => "EOF",
        }
    }
}

/// Look up a word in the keyword table.
pub fn lookup_keyword(word: &str) -> Option<TokenType> {
    match word {
        "model" => Some(TokenType::Model),
        "layer" => Some(TokenType::Layer),
        "fn" => Some(TokenType::Fn),
        "let" => Some(TokenType::Let),
        "return" => Some(TokenType::Return),
        "if" => Some(TokenType::If),
        "else" => Some(TokenType::Else),
        "for" => Some(TokenType::For),
        "in" => Some(TokenType::In),
        "import" => Some(TokenType::Import),
        "from" => Some(TokenType::From),
        "as" => Some(TokenType::As),
        "self" => Some(TokenType::Self_),
        "causal" => Some(TokenType::Causal),
        "update" => Some(TokenType::Update),
        "by" => Some(TokenType::By),
        "constraint" => Some(TokenType::Constraint),
        "forget" => Some(TokenType::Forget),
        "strategy" => Some(TokenType::Strategy),
        "preserve" => Some(TokenType::Preserve),
        "fixed" => Some(TokenType::Fixed),
        "discover" => Some(TokenType::Discover),
        "variables" => Some(TokenType::Variables),
        "do" => Some(TokenType::Do),
        "observe" => Some(TokenType::Observe),
        "grad" => Some(TokenType::Grad),
        "stop_grad" => Some(TokenType::StopGrad),
        "explain" => Some(TokenType::Explain),
        "true" => Some(TokenType::True),
        "false" => Some(TokenType::False),
        "agent" => Some(TokenType::Agent),
        "meta" => Some(TokenType::Meta),
        "search" => Some(TokenType::Search),
        "recall" => Some(TokenType::Recall),
        "store" => Some(TokenType::Store),
        "stream" => Some(TokenType::Stream),
        "reward" => Some(TokenType::Reward),
        // Type keywords
        "Tensor" => Some(TokenType::TensorKw),
        "Uncertain" => Some(TokenType::UncertainKw),
        "Random" => Some(TokenType::RandomKw),
        "Prob" => Some(TokenType::ProbKw),
        "Temporal" => Some(TokenType::TemporalKw),
        "Causal" => Some(TokenType::CausalKw),
        "Learnable" => Some(TokenType::LearnableKw),
        "Effect" => Some(TokenType::EffectKw),
        "List" => Some(TokenType::ListKw),
        "Option" => Some(TokenType::OptionKw),
        "Int" => Some(TokenType::IntKw),
        "Float" => Some(TokenType::FloatKw),
        "Bool" => Some(TokenType::BoolKw),
        "String" => Some(TokenType::StringKw),
        "Timestamp" => Some(TokenType::TimestampKw),
        "Loss" => Some(TokenType::LossKw),
        "Dataset" => Some(TokenType::DatasetKw),
        "Memory" => Some(TokenType::MemoryKw),
        "EpisodicMemory" => Some(TokenType::EpisodicMemoryKw),
        "SemanticMemory" => Some(TokenType::SemanticMemoryKw),
        "WorkingMemory" => Some(TokenType::WorkingMemoryKw),
        "Reward" => Some(TokenType::RewardKw),
        "Experience" => Some(TokenType::ExperienceKw),
        _ => None,
    }
}
