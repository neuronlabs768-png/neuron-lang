/// NEURON Recursive-Descent Parser.
///
/// Parses a token stream into a Program AST using Pratt parsing for
/// expression precedence.

use crate::token::{Token, TokenType, Span};
use crate::ast::*;
use crate::errors::{ErrorCode, NeuronError};

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    #[allow(dead_code)]
    filename: String,
}

impl Parser {
    pub fn new(tokens: Vec<Token>, filename: impl Into<String>) -> Self {
        Self { tokens, pos: 0, filename: filename.into() }
    }

    // ─── token helpers ──────────────────

    fn peek(&self) -> &Token {
        if self.pos < self.tokens.len() { &self.tokens[self.pos] } else { self.tokens.last().unwrap() }
    }

    fn peek_type(&self) -> &TokenType { &self.peek().ty }

    fn peek_ahead(&self, offset: usize) -> &Token {
        let idx = self.pos + offset;
        if idx < self.tokens.len() { &self.tokens[idx] } else { self.tokens.last().unwrap() }
    }

    fn at_end(&self) -> bool { matches!(self.peek_type(), TokenType::Eof) }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        if !matches!(tok.ty, TokenType::Eof) { self.pos += 1; }
        tok
    }

    fn expect(&mut self, expected: &str) -> Result<Token, NeuronError> {
        let tok = self.peek().clone();
        let matches = match expected {
            "IDENT" => matches!(tok.ty, TokenType::Ident(_)),
            "COLON" => matches!(tok.ty, TokenType::Colon),
            "LPAREN" => matches!(tok.ty, TokenType::LParen),
            "RPAREN" => matches!(tok.ty, TokenType::RParen),
            "LBRACKET" => matches!(tok.ty, TokenType::LBracket),
            "RBRACKET" => matches!(tok.ty, TokenType::RBracket),
            "EQ" => matches!(tok.ty, TokenType::Eq),
            "INDENT" => matches!(tok.ty, TokenType::Indent),
            "DEDENT" => matches!(tok.ty, TokenType::Dedent),
            "ARROW" => matches!(tok.ty, TokenType::Arrow | TokenType::UnicodeArrow),
            "MODEL" => matches!(tok.ty, TokenType::Model),
            "LAYER" => matches!(tok.ty, TokenType::Layer),
            "FN" => matches!(tok.ty, TokenType::Fn),
            "LET" => matches!(tok.ty, TokenType::Let),
            "IN" => matches!(tok.ty, TokenType::In),
            "BY" => matches!(tok.ty, TokenType::By),
            "IMPORT" => matches!(tok.ty, TokenType::Import),
            "COMMA" => matches!(tok.ty, TokenType::Comma),
            "DOT" => matches!(tok.ty, TokenType::Dot),
            _ => false,
        };
        if matches {
            self.pos += 1;
            Ok(tok)
        } else {
            Err(NeuronError::new(
                ErrorCode::ParseError,
                format!("Expected {} but found {}", expected, tok.ty.name()),
                tok.span.clone(),
            ))
        }
    }

    fn match_token(&mut self, check: impl Fn(&TokenType) -> bool) -> Option<Token> {
        if check(self.peek_type()) {
            let tok = self.peek().clone();
            self.pos += 1;
            Some(tok)
        } else {
            None
        }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek_type(), TokenType::Newline) { self.pos += 1; }
    }

    fn span(&self) -> Span { self.peek().span.clone() }

    fn error(&self, msg: impl Into<String>) -> NeuronError {
        NeuronError::new(ErrorCode::ParseError, msg, self.span())
    }

    fn ident_str(tok: &Token) -> Result<String, NeuronError> {
        match &tok.ty {
            TokenType::Ident(s) => Ok(s.clone()),
            _ => Err(NeuronError::new(ErrorCode::ParseError, format!("Expected identifier, got {}", tok.ty.name()), tok.span.clone())),
        }
    }

    // ═══════════════════════════════════════
    //  Top-level parsing
    // ═══════════════════════════════════════

    pub fn parse(&mut self) -> Result<Program, NeuronError> {
        let mut top_levels = Vec::new();
        self.skip_newlines();
        while !self.at_end() {
            let tl = self.parse_top_level()?;
            top_levels.push(tl);
            self.skip_newlines();
        }
        Ok(Program { top_levels })
    }

    fn parse_top_level(&mut self) -> Result<TopLevel, NeuronError> {
        // Annotations
        if matches!(self.peek_type(), TokenType::Annotation(_)) {
            let anns = self.parse_annotations()?;
            return match self.peek_type() {
                TokenType::Model => Ok(TopLevel::Model(self.parse_model_decl(anns)?)),
                TokenType::Layer => Ok(TopLevel::Layer(self.parse_layer_decl(anns)?)),
                TokenType::Fn => Ok(TopLevel::Fn(self.parse_fn_decl(anns)?)),
                TokenType::Agent => Ok(TopLevel::Agent(self.parse_agent_decl(anns)?)),
                TokenType::Import => {
                    if anns.len() == 1 && anns[0].name == "python" {
                        Ok(TopLevel::Import(self.parse_python_import()?))
                    } else {
                        Err(self.error("Unexpected annotation before import"))
                    }
                }
                _ => Err(self.error("Annotations must precede model, layer, fn, or agent")),
            };
        }

        match self.peek_type() {
            TokenType::Model => Ok(TopLevel::Model(self.parse_model_decl(vec![])?)),
            TokenType::Layer => Ok(TopLevel::Layer(self.parse_layer_decl(vec![])?)),
            TokenType::Fn => Ok(TopLevel::Fn(self.parse_fn_decl(vec![])?)),
            TokenType::Causal => Ok(TopLevel::Causal(self.parse_causal_decl()?)),
            TokenType::Agent => Ok(TopLevel::Agent(self.parse_agent_decl(vec![])?)),
            TokenType::Meta => Ok(TopLevel::Meta(self.parse_meta_decl()?)),
            TokenType::Let => Ok(TopLevel::Let(self.parse_let_stmt()?)),
            TokenType::From => Ok(TopLevel::Import(self.parse_import_stmt()?)),
            TokenType::Import => Ok(TopLevel::Import(self.parse_bare_import()?)),
            TokenType::Constraint => Ok(TopLevel::Constraint(self.parse_constraint_decl()?)),
            TokenType::Update => Ok(TopLevel::Update(self.parse_update_stmt()?)),
            // Bare expression statements at top level (e.g., print(...))
            _ => Ok(TopLevel::Expr(self.parse_expr_stmt()?)),
        }
    }

    // ─── annotations ────────────────────

    fn parse_annotations(&mut self) -> Result<Vec<Annotation>, NeuronError> {
        let mut anns = Vec::new();
        while let TokenType::Annotation(_) = self.peek_type() {
            anns.push(self.parse_single_annotation()?);
            self.skip_newlines();
        }
        Ok(anns)
    }

    fn parse_single_annotation(&mut self) -> Result<Annotation, NeuronError> {
        let tok = self.advance().clone();
        let name = match &tok.ty {
            TokenType::Annotation(s) => s.clone(),
            _ => return Err(self.error("Expected annotation")),
        };
        let span = tok.span.clone();
        let mut args = Vec::new();
        if matches!(self.peek_type(), TokenType::LParen) {
            self.advance();
            if !matches!(self.peek_type(), TokenType::RParen) {
                args = self.parse_annotation_args()?;
            }
            self.expect("RPAREN")?;
        }
        Ok(Annotation { name, args, span })
    }

    fn parse_annotation_args(&mut self) -> Result<Vec<AnnotationArg>, NeuronError> {
        let mut args = Vec::new();
        loop {
            // key=value or positional
            if matches!(self.peek_type(), TokenType::Ident(_)) && matches!(self.peek_ahead(1).ty, TokenType::Eq) {
                let key_tok = self.advance().clone();
                let key = Self::ident_str(&key_tok)?;
                self.advance(); // =
                let value = self.parse_annotation_value()?;
                args.push(AnnotationArg { key: Some(key), value });
            } else {
                let value = self.parse_annotation_value()?;
                args.push(AnnotationArg { key: None, value });
            }
            if self.match_token(|t| matches!(t, TokenType::Comma)).is_none() { break; }
        }
        Ok(args)
    }

    fn parse_annotation_value(&mut self) -> Result<AnnotationValue, NeuronError> {
        let tok = self.advance().clone();
        match &tok.ty {
            TokenType::IntLit(v) => Ok(AnnotationValue::Int(*v)),
            TokenType::FloatLit(v) => Ok(AnnotationValue::Float(*v)),
            TokenType::StringLit(s) => Ok(AnnotationValue::Str(s.clone())),
            TokenType::True => Ok(AnnotationValue::Bool(true)),
            TokenType::False => Ok(AnnotationValue::Bool(false)),
            TokenType::Ident(s) => Ok(AnnotationValue::Ident(s.clone())),
            _ => Ok(AnnotationValue::Ident(tok.ty.name().to_string())),
        }
    }

    // ─── model declaration ──────────────

    fn parse_model_decl(&mut self, annotations: Vec<Annotation>) -> Result<ModelDecl, NeuronError> {
        let tok = self.expect("MODEL")?.clone();
        let span = tok.span.clone();
        let name_tok = self.expect("IDENT")?;
        let name = Self::ident_str(&name_tok)?;
        let params = if matches!(self.peek_type(), TokenType::LParen) { self.parse_param_list()? } else { vec![] };
        self.expect("COLON")?;
        self.skip_newlines();
        self.expect("INDENT")?;
        let (fields, methods, forget_decls) = self.parse_model_body()?;
        self.expect("DEDENT")?;
        Ok(ModelDecl { name, params, annotations, fields, methods, forget_decls, span })
    }

    fn parse_model_body(&mut self) -> Result<(Vec<FieldDecl>, Vec<FnDecl>, Vec<ForgetDecl>), NeuronError> {
        let mut fields = Vec::new();
        let mut methods = Vec::new();
        let mut forget_decls = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek_type(), TokenType::Dedent | TokenType::Eof) {
            self.skip_newlines();
            if matches!(self.peek_type(), TokenType::Dedent) { break; }
            match self.peek_type() {
                TokenType::Fn => methods.push(self.parse_fn_decl(vec![])?),
                TokenType::Annotation(_) => {
                    let anns = self.parse_annotations()?;
                    methods.push(self.parse_fn_decl(anns)?);
                }
                TokenType::Forget => forget_decls.push(self.parse_forget_decl()?),
                _ => fields.push(self.parse_field_decl()?),
            }
            self.skip_newlines();
        }
        Ok((fields, methods, forget_decls))
    }

    // ─── layer declaration ──────────────

    fn parse_layer_decl(&mut self, annotations: Vec<Annotation>) -> Result<LayerDecl, NeuronError> {
        let tok = self.expect("LAYER")?.clone();
        let span = tok.span.clone();
        let name_tok = self.expect("IDENT")?;
        let name = Self::ident_str(&name_tok)?;
        let params = if matches!(self.peek_type(), TokenType::LParen) { self.parse_param_list()? } else { vec![] };
        self.expect("COLON")?;
        self.skip_newlines();
        self.expect("INDENT")?;
        let mut fields = Vec::new();
        let mut methods = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek_type(), TokenType::Dedent | TokenType::Eof) {
            self.skip_newlines();
            if matches!(self.peek_type(), TokenType::Dedent) { break; }
            if matches!(self.peek_type(), TokenType::Fn) {
                methods.push(self.parse_fn_decl(vec![])?);
            } else if matches!(self.peek_type(), TokenType::Annotation(_)) {
                let anns = self.parse_annotations()?;
                methods.push(self.parse_fn_decl(anns)?);
            } else {
                fields.push(self.parse_field_decl()?);
            }
            self.skip_newlines();
        }
        self.expect("DEDENT")?;
        Ok(LayerDecl { name, params, annotations, fields, methods, span })
    }

    // ─── agent declaration (AGI) ────────

    fn parse_agent_decl(&mut self, annotations: Vec<Annotation>) -> Result<AgentDecl, NeuronError> {
        let tok = self.advance().clone(); // consume 'agent'
        let span = tok.span.clone();
        let name_tok = self.expect("IDENT")?;
        let name = Self::ident_str(&name_tok)?;
        let params = if matches!(self.peek_type(), TokenType::LParen) { self.parse_param_list()? } else { vec![] };
        self.expect("COLON")?;
        self.skip_newlines();
        self.expect("INDENT")?;
        let mut fields = Vec::new();
        let mut methods = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek_type(), TokenType::Dedent | TokenType::Eof) {
            self.skip_newlines();
            if matches!(self.peek_type(), TokenType::Dedent) { break; }
            if matches!(self.peek_type(), TokenType::Fn) {
                methods.push(self.parse_fn_decl(vec![])?);
            } else {
                fields.push(self.parse_field_decl()?);
            }
            self.skip_newlines();
        }
        self.expect("DEDENT")?;
        Ok(AgentDecl { name, params, annotations, fields, methods, span })
    }

    // ─── meta declaration (AGI) ─────────

    fn parse_meta_decl(&mut self) -> Result<MetaDecl, NeuronError> {
        let tok = self.advance().clone(); // consume 'meta'
        let span = tok.span.clone();
        let func = self.parse_fn_decl(vec![])?;
        Ok(MetaDecl { func, span })
    }

    // ─── function declaration ───────────

    fn parse_fn_decl(&mut self, annotations: Vec<Annotation>) -> Result<FnDecl, NeuronError> {
        let tok = self.expect("FN")?.clone();
        let span = tok.span.clone();
        let name_tok = self.expect("IDENT")?;
        let name = Self::ident_str(&name_tok)?;
        let params = self.parse_param_list()?;
        let return_type = if matches!(self.peek_type(), TokenType::Arrow | TokenType::UnicodeArrow) {
            self.advance();
            Some(self.parse_type()?)
        } else { None };
        let effect_clause = if matches!(self.peek_type(), TokenType::LBracket) {
            Some(self.parse_effect_clause()?)
        } else { None };
        self.expect("COLON")?;
        let body = self.parse_block()?;
        Ok(FnDecl { name, params, return_type, effect_clause, annotations, body, span })
    }

    fn parse_effect_clause(&mut self) -> Result<EffectType, NeuronError> {
        let span = self.span();
        self.expect("LBRACKET")?;
        // Expect Effect[...]
        let _effect_kw = self.advance(); // Effect keyword or ident
        self.expect("LBRACKET")?;
        let mut effects = Vec::new();
        loop {
            let tok = self.advance().clone();
            let kind = match &tok.ty {
                TokenType::Ident(s) => s.clone(),
                _ => tok.ty.name().to_string(),
            };
            let target = if kind == "Mut" && matches!(self.peek_type(), TokenType::LBracket) {
                self.advance();
                // Accept both ident and `self` keyword
                let name = if matches!(self.peek_type(), TokenType::Self_) {
                    self.advance();
                    "self".to_string()
                } else {
                    let t = self.expect("IDENT")?;
                    Self::ident_str(&t)?
                };
                self.expect("RBRACKET")?;
                Some(name)
            } else { None };
            effects.push(EffectKind { kind, target });
            if self.match_token(|t| matches!(t, TokenType::Comma)).is_none() { break; }
        }
        self.expect("RBRACKET")?;
        self.expect("RBRACKET")?;
        Ok(EffectType { effects, span })
    }

    // ─── causal declaration ─────────────

    fn parse_causal_decl(&mut self) -> Result<CausalDecl, NeuronError> {
        let tok = self.advance().clone(); // causal
        let span = tok.span.clone();
        self.expect("MODEL")?;
        let name_tok = self.expect("IDENT")?;
        let name = Self::ident_str(&name_tok)?;
        let mut options = Vec::new();
        if matches!(self.peek_type(), TokenType::LBracket) {
            self.advance();
            while !matches!(self.peek_type(), TokenType::RBracket | TokenType::Eof) {
                let key_tok = self.advance().clone();
                let key = match &key_tok.ty {
                    TokenType::Ident(s) => s.clone(),
                    _ => key_tok.ty.name().to_string(),
                };
                self.expect("EQ")?;
                let val_tok = self.advance().clone();
                let val = match &val_tok.ty {
                    TokenType::Ident(s) => s.clone(),
                    TokenType::StringLit(s) => s.clone(),
                    _ => val_tok.ty.name().to_string(),
                };
                options.push(CausalOpt { name: key, value: val, span: key_tok.span.clone() });
                self.match_token(|t| matches!(t, TokenType::Comma));
            }
            self.expect("RBRACKET")?;
        }
        self.expect("COLON")?;
        self.skip_newlines();
        self.expect("INDENT")?;
        let mut edges = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek_type(), TokenType::Dedent | TokenType::Eof) {
            self.skip_newlines();
            if matches!(self.peek_type(), TokenType::Dedent) { break; }
            edges.push(self.parse_causal_edge()?);
            self.skip_newlines();
        }
        self.expect("DEDENT")?;
        Ok(CausalDecl { name, options, edges, span })
    }

    fn parse_causal_edge(&mut self) -> Result<CausalEdge, NeuronError> {
        let span = self.span();
        match self.peek_type() {
            TokenType::Fixed => {
                self.advance();
                self.expect("COLON")?;
                let src_tok = self.expect("IDENT")?;
                let src = Self::ident_str(&src_tok)?;
                self.expect("ARROW")?;
                let tgt_tok = self.expect("IDENT")?;
                let tgt = Self::ident_str(&tgt_tok)?;
                Ok(CausalEdge { kind: CausalEdgeKind::Fixed, sources: vec![src], target: Some(tgt), span })
            }
            TokenType::Discover => {
                self.advance();
                self.expect("COLON")?;
                self.expect("LBRACKET")?;
                let mut srcs = Vec::new();
                while !matches!(self.peek_type(), TokenType::RBracket | TokenType::Eof) {
                    let t = self.expect("IDENT")?;
                    srcs.push(Self::ident_str(&t)?);
                    self.match_token(|t| matches!(t, TokenType::Comma));
                }
                self.expect("RBRACKET")?;
                self.expect("ARROW")?;
                let tgt_tok = self.expect("IDENT")?;
                let tgt = Self::ident_str(&tgt_tok)?;
                Ok(CausalEdge { kind: CausalEdgeKind::Discover, sources: srcs, target: Some(tgt), span })
            }
            TokenType::Variables => {
                self.advance();
                self.expect("COLON")?;
                self.expect("LBRACKET")?;
                let mut names = Vec::new();
                while !matches!(self.peek_type(), TokenType::RBracket | TokenType::Eof) {
                    let t = self.expect("IDENT")?;
                    names.push(Self::ident_str(&t)?);
                    self.match_token(|t| matches!(t, TokenType::Comma));
                }
                self.expect("RBRACKET")?;
                Ok(CausalEdge { kind: CausalEdgeKind::Variables, sources: names, target: None, span })
            }
            _ => {
                let src_tok = self.expect("IDENT")?;
                let src = Self::ident_str(&src_tok)?;
                self.expect("ARROW")?;
                let tgt_tok = self.expect("IDENT")?;
                let tgt = Self::ident_str(&tgt_tok)?;
                Ok(CausalEdge { kind: CausalEdgeKind::Simple, sources: vec![src], target: Some(tgt), span })
            }
        }
    }

    // ─── import ─────────────────────────

    fn parse_import_stmt(&mut self) -> Result<ImportStmt, NeuronError> {
        let tok = self.advance().clone(); // from
        let span = tok.span.clone();
        let mut parts = vec![Self::ident_str(&self.expect("IDENT")?)?];
        while matches!(self.peek_type(), TokenType::Dot) {
            self.advance();
            parts.push(Self::ident_str(&self.expect("IDENT")?)?);
        }
        let module = parts.join(".");
        self.expect("IMPORT")?;
        let mut names = vec![Self::ident_str(&self.expect("IDENT")?)?];
        while self.match_token(|t| matches!(t, TokenType::Comma)).is_some() {
            names.push(Self::ident_str(&self.expect("IDENT")?)?);
        }
        self.skip_newlines();
        Ok(ImportStmt { module, names, alias: None, is_python: false, span })
    }

    fn parse_python_import(&mut self) -> Result<ImportStmt, NeuronError> {
        self.parse_import_bare(true)
    }

    fn parse_bare_import(&mut self) -> Result<ImportStmt, NeuronError> {
        self.parse_import_bare(false)
    }

    fn parse_import_bare(&mut self, is_python: bool) -> Result<ImportStmt, NeuronError> {
        let span = self.span();
        self.expect("IMPORT")?;
        let mut parts = vec![Self::ident_str(&self.expect("IDENT")?)?];
        while matches!(self.peek_type(), TokenType::Dot) {
            self.advance();
            parts.push(Self::ident_str(&self.expect("IDENT")?)?);
        }
        let module = parts.join(".");
        let alias = if matches!(self.peek_type(), TokenType::As) {
            self.advance();
            Some(Self::ident_str(&self.expect("IDENT")?)?)
        } else { None };
        self.skip_newlines();
        Ok(ImportStmt { module, names: vec![], alias, is_python, span })
    }

    // ─── parameters ─────────────────────

    fn parse_param_list(&mut self) -> Result<Vec<Param>, NeuronError> {
        self.expect("LPAREN")?;
        let mut params = Vec::new();
        if !matches!(self.peek_type(), TokenType::RParen) {
            params.push(self.parse_param()?);
            while self.match_token(|t| matches!(t, TokenType::Comma)).is_some() {
                if matches!(self.peek_type(), TokenType::RParen) { break; }
                params.push(self.parse_param()?);
            }
        }
        self.expect("RPAREN")?;
        Ok(params)
    }

    fn parse_param(&mut self) -> Result<Param, NeuronError> {
        let span = self.span();
        if matches!(self.peek_type(), TokenType::Self_) {
            self.advance();
            return Ok(Param { name: "self".to_string(), type_ann: None, default: None, span });
        }
        // Accept contextual keywords as parameter names (reward, store, recall, etc.)
        let (name, span) = match self.peek_type() {
            TokenType::Ident(_) => {
                let tok = self.advance().clone();
                (Self::ident_str(&tok)?, tok.span.clone())
            }
            // Contextual keywords that can be used as parameter names
            _ => {
                let tok = self.advance().clone();
                let name = tok.ty.name().to_string().to_lowercase();
                if name.chars().all(|c| c.is_alphanumeric() || c == '_') && !name.is_empty() {
                    (name, tok.span.clone())
                } else {
                    return Err(NeuronError::new(
                        ErrorCode::ParseError,
                        format!("Expected parameter name, got {}", tok.ty.name()),
                        tok.span.clone(),
                    ));
                }
            }
        };
        let type_ann = if matches!(self.peek_type(), TokenType::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else { None };
        let default = if matches!(self.peek_type(), TokenType::Eq) {
            self.advance();
            Some(self.parse_expression(0)?)
        } else { None };
        Ok(Param { name, type_ann, default, span })
    }

    // ─── field declaration ──────────────

    fn parse_field_decl(&mut self) -> Result<FieldDecl, NeuronError> {
        let span = self.span();
        let name_tok = self.expect("IDENT")?;
        let name = Self::ident_str(&name_tok)?;
        self.expect("COLON")?;
        let type_ann = self.parse_type()?;
        let default = if matches!(self.peek_type(), TokenType::Eq) {
            self.advance();
            Some(self.parse_expression(0)?)
        } else if matches!(self.peek_type(), TokenType::LParen) {
            // Constructor call default: attn: MultiHeadAttention(d_model, n_heads)
            Some(self.parse_call_expr_from(Expr::Ident(name.clone(), span.clone()))?)
        } else { None };
        self.skip_newlines();
        Ok(FieldDecl { name, type_ann, default, span })
    }

    fn parse_forget_decl(&mut self) -> Result<ForgetDecl, NeuronError> {
        let span = self.span();
        self.advance(); // forget
        self.expect("COLON")?;
        let type_ann = self.parse_type()?;
        let description = if matches!(self.peek_type(), TokenType::LBracket) {
            self.advance();
            let s = match self.peek_type() {
                TokenType::StringLit(s) => { let v = s.clone(); self.advance(); v }
                _ => String::new(),
            };
            self.expect("RBRACKET")?;
            s
        } else { String::new() };
        self.skip_newlines();
        Ok(ForgetDecl { type_ann, description, span })
    }

    // ─── block ──────────────────────────

    fn parse_block(&mut self) -> Result<Vec<Stmt>, NeuronError> {
        self.skip_newlines();
        self.expect("INDENT")?;
        let mut stmts = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek_type(), TokenType::Dedent | TokenType::Eof) {
            self.skip_newlines();
            if matches!(self.peek_type(), TokenType::Dedent) { break; }
            stmts.push(self.parse_statement()?);
            self.skip_newlines();
        }
        self.expect("DEDENT")?;
        Ok(stmts)
    }

    // ═══════════════════════════════════════
    //  Statement parsing
    // ═══════════════════════════════════════

    fn parse_statement(&mut self) -> Result<Stmt, NeuronError> {
        match self.peek_type() {
            TokenType::Let => Ok(Stmt::Let(self.parse_let_stmt()?)),
            TokenType::For => Ok(Stmt::For(self.parse_for_stmt()?)),
            TokenType::If => Ok(Stmt::If(self.parse_if_stmt()?)),
            TokenType::Return => Ok(Stmt::Return(self.parse_return_stmt()?)),
            TokenType::Update => Ok(Stmt::Update(self.parse_update_stmt()?)),
            TokenType::Constraint => Ok(Stmt::Constraint(self.parse_constraint_decl()?)),
            _ => Ok(Stmt::Expr(self.parse_expr_stmt()?)),
        }
    }

    fn parse_let_stmt(&mut self) -> Result<LetStmt, NeuronError> {
        let span = self.span();
        self.advance(); // let
        let name = if matches!(self.peek_type(), TokenType::Self_) {
            self.advance();
            let mut t = "self".to_string();
            while matches!(self.peek_type(), TokenType::Dot) {
                self.advance();
                let field_tok = self.expect("IDENT")?;
                t.push('.');
                t.push_str(&Self::ident_str(&field_tok)?);
            }
            t
        } else {
            let name_tok = self.expect("IDENT")?;
            Self::ident_str(&name_tok)?
        };
        let type_ann = if matches!(self.peek_type(), TokenType::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else { None };
        self.expect("EQ")?;
        let value = self.parse_expression(0)?;
        self.skip_newlines();
        Ok(LetStmt { name, type_ann, value, span })
    }

    fn parse_for_stmt(&mut self) -> Result<ForStmt, NeuronError> {
        let span = self.span();
        self.advance(); // for
        let var_tok = self.expect("IDENT")?;
        let var = Self::ident_str(&var_tok)?;
        self.expect("IN")?;
        let iter_expr = self.parse_expression(0)?;
        self.expect("COLON")?;
        let body = self.parse_block()?;
        Ok(ForStmt { var, iter_expr, body, span })
    }

    fn parse_if_stmt(&mut self) -> Result<IfStmt, NeuronError> {
        let span = self.span();
        self.advance(); // if
        let cond = self.parse_expression(0)?;
        self.expect("COLON")?;
        let then_body = self.parse_block()?;
        let else_body = if matches!(self.peek_type(), TokenType::Else) {
            self.advance();
            self.expect("COLON")?;
            self.parse_block()?
        } else { vec![] };
        Ok(IfStmt { cond, then_body, else_body, span })
    }

    fn parse_return_stmt(&mut self) -> Result<ReturnStmt, NeuronError> {
        let span = self.span();
        self.advance(); // return
        let value = self.parse_expression(0)?;
        self.skip_newlines();
        Ok(ReturnStmt { value, span })
    }

    fn parse_update_stmt(&mut self) -> Result<UpdateStmt, NeuronError> {
        let span = self.span();
        self.advance(); // update
        // Accept self, self.field, or ident as target
        let target = if matches!(self.peek_type(), TokenType::Self_) {
            self.advance();
            let mut t = "self".to_string();
            // Handle dotted path: self.policy, self.weights, etc.
            while matches!(self.peek_type(), TokenType::Dot) {
                self.advance();
                let field_tok = self.expect("IDENT")?;
                t.push('.');
                t.push_str(&Self::ident_str(&field_tok)?);
            }
            t
        } else {
            let target_tok = self.expect("IDENT")?;
            let mut t = Self::ident_str(&target_tok)?;
            while matches!(self.peek_type(), TokenType::Dot) {
                self.advance();
                let field_tok = self.expect("IDENT")?;
                t.push('.');
                t.push_str(&Self::ident_str(&field_tok)?);
            }
            t
        };
        self.expect("BY")?;
        let expr = self.parse_expression(0)?;
        self.skip_newlines();
        Ok(UpdateStmt { target, expr, span })
    }

    fn parse_constraint_decl(&mut self) -> Result<ConstraintDecl, NeuronError> {
        let span = self.span();
        self.advance(); // constraint
        let expr = self.parse_expression(0)?;
        self.skip_newlines();
        Ok(ConstraintDecl { expr, span })
    }

    fn parse_expr_stmt(&mut self) -> Result<ExprStmt, NeuronError> {
        let span = self.span();
        let expr = self.parse_expression(0)?;
        self.skip_newlines();
        Ok(ExprStmt { expr, span })
    }

    // ═══════════════════════════════════════
    //  Expression parsing (Pratt)
    // ═══════════════════════════════════════

    fn parse_expression(&mut self, min_prec: u8) -> Result<Expr, NeuronError> {
        let mut left = self.parse_unary()?;

        loop {
            let (op, prec) = match self.peek_type() {
                TokenType::Or => (BinOp::Or, 1),
                TokenType::And => (BinOp::And, 2),
                TokenType::EqEq => (BinOp::Eq, 3),
                TokenType::Neq => (BinOp::Neq, 3),
                TokenType::Lt => (BinOp::Lt, 3),
                TokenType::Gt => (BinOp::Gt, 3),
                TokenType::Lte => (BinOp::Lte, 3),
                TokenType::Gte => (BinOp::Gte, 3),
                TokenType::Plus => (BinOp::Add, 4),
                TokenType::Minus => (BinOp::Sub, 4),
                TokenType::Star => (BinOp::Mul, 5),
                TokenType::Slash => (BinOp::Div, 5),
                TokenType::Percent => (BinOp::Mod, 5),
                TokenType::At => (BinOp::MatMul, 5),
                _ => break,
            };

            if prec < min_prec { break; }
            self.advance();
            let span = left.span().clone();
            let right = self.parse_expression(prec + 1)?;

            // Check for merge expression: expr + expr strategy: ...
            if op == BinOp::Add && matches!(self.peek_type(), TokenType::Strategy) {
                left = self.parse_merge_rest(left, right, span)?;
                continue;
            }

            left = Expr::BinOp(Box::new(BinOpExpr { left, op, right, span }));
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, NeuronError> {
        match self.peek_type() {
            TokenType::Minus => {
                let span = self.span();
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expr::UnaryOp(Box::new(UnaryOpExpr { op: UnaryOp::Neg, operand, span })))
            }
            TokenType::Bang => {
                let span = self.span();
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expr::UnaryOp(Box::new(UnaryOpExpr { op: UnaryOp::Not, operand, span })))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, NeuronError> {
        let mut expr = self.parse_primary()?;

        loop {
            match self.peek_type() {
                TokenType::Dot => {
                    self.advance();
                    let span = expr.span().clone();
                    // Check for .observe(), .forget(), .before(), .after(), .snapshot()
                    let field = if matches!(self.peek_type(), TokenType::Forget | TokenType::Observe) {
                        let tok = self.advance();
                        if tok.ty == TokenType::Forget {
                            "forget".to_string()
                        } else {
                            "observe".to_string()
                        }
                    } else {
                        let field_tok = self.expect("IDENT")?;
                        Self::ident_str(&field_tok)?
                    };

                    if &field == "observe" && matches!(self.peek_type(), TokenType::LParen) {
                        let assignments = self.parse_observe_args()?;
                        expr = Expr::Observe(Box::new(ObserveExpr { obj: expr, assignments, span }));
                    } else if &field == "forget" && matches!(self.peek_type(), TokenType::LParen) {
                        let args = self.parse_call_args()?;
                        expr = Expr::Forget(Box::new(ForgetExpr { obj: expr, args, span }));
                    } else {
                        expr = Expr::Dot(Box::new(DotExpr { obj: expr, field, span }));
                    }
                }
                TokenType::LParen => {
                    expr = self.parse_call_expr_from(expr)?;
                }
                TokenType::LBracket => {
                    let span = expr.span().clone();
                    self.advance();
                    let indices = self.parse_index_items()?;
                    self.expect("RBRACKET")?;
                    expr = Expr::Index(Box::new(IndexExpr { obj: expr, indices, span }));
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, NeuronError> {
        let span = self.span();
        match self.peek_type().clone() {
            TokenType::IntLit(v) => { let v = v; self.advance(); Ok(Expr::IntLit(v, span)) }
            TokenType::FloatLit(v) => { let v = v; self.advance(); Ok(Expr::FloatLit(v, span)) }
            TokenType::True => { self.advance(); Ok(Expr::BoolLit(true, span)) }
            TokenType::False => { self.advance(); Ok(Expr::BoolLit(false, span)) }
            TokenType::StringLit(s) => { let s = s.clone(); self.advance(); Ok(Expr::StringLit(s, span)) }
            TokenType::Ident(s) => { let s = s.clone(); self.advance(); Ok(Expr::Ident(s, span)) }
            TokenType::Self_ => { self.advance(); Ok(Expr::Self_(span)) }
            TokenType::Grad => self.parse_grad_expr(),
            TokenType::StopGrad => self.parse_stop_grad_expr(),
            TokenType::Do => self.parse_do_expr(),
            TokenType::Explain => self.parse_explain_expr(),
            TokenType::Search => self.parse_search_expr(),
            TokenType::Recall => self.parse_recall_expr(),
            TokenType::Store => self.parse_store_expr(),
            // Type constructors used as expressions (e.g., Uncertain(...))
            TokenType::UncertainKw | TokenType::RandomKw | TokenType::RewardKw => {
                let name = self.peek_type().name().to_string();
                self.advance();
                Ok(Expr::Ident(name, span))
            }
            TokenType::LParen => {
                self.advance();
                let expr = self.parse_expression(0)?;
                // Check for tuple
                if matches!(self.peek_type(), TokenType::Comma) {
                    let mut elems = vec![expr];
                    while self.match_token(|t| matches!(t, TokenType::Comma)).is_some() {
                        if matches!(self.peek_type(), TokenType::RParen) { break; }
                        elems.push(self.parse_expression(0)?);
                    }
                    self.expect("RPAREN")?;
                    return Ok(Expr::Tuple(elems, span));
                }
                self.expect("RPAREN")?;
                Ok(expr)
            }
            TokenType::LBracket => {
                self.advance();
                if matches!(self.peek_type(), TokenType::RBracket) {
                    self.advance();
                    return Ok(Expr::List(vec![], span));
                }
                let first = self.parse_expression(0)?;
                // List comprehension: [expr for var in iter]
                if matches!(self.peek_type(), TokenType::For) {
                    self.advance();
                    let var_tok = self.expect("IDENT")?;
                    let var = Self::ident_str(&var_tok)?;
                    self.expect("IN")?;
                    let iter = self.parse_expression(0)?;
                    self.expect("RBRACKET")?;
                    return Ok(Expr::ListComp(Box::new(ListCompExpr { expr: first, var, iter, span })));
                }
                let mut elems = vec![first];
                while self.match_token(|t| matches!(t, TokenType::Comma)).is_some() {
                    if matches!(self.peek_type(), TokenType::RBracket) { break; }
                    elems.push(self.parse_expression(0)?);
                }
                self.expect("RBRACKET")?;
                Ok(Expr::List(elems, span))
            }
            // Contextual keywords used as identifiers in expression position
            TokenType::Reward | TokenType::Stream | TokenType::Fixed
            | TokenType::Discover | TokenType::Variables | TokenType::Strategy
            | TokenType::Forget | TokenType::Agent | TokenType::Meta
            | TokenType::Model | TokenType::Layer => {
                let name = self.peek_type().name().to_string().to_lowercase();
                self.advance();
                Ok(Expr::Ident(name, span))
            }
            _ => Err(self.error(format!("Unexpected token in expression: {}", self.peek_type().name()))),
        }
    }

    // ─── special expression parsers ─────

    fn parse_grad_expr(&mut self) -> Result<Expr, NeuronError> {
        let span = self.span();
        self.advance(); // grad
        self.expect("LPAREN")?;
        let expr = self.parse_expression(0)?;
        let wrt = if self.match_token(|t| matches!(t, TokenType::Comma)).is_some() {
            // wrt=param
            if matches!(self.peek_type(), TokenType::Ident(_)) && matches!(self.peek_ahead(1).ty, TokenType::Eq) {
                self.advance(); // wrt
                self.advance(); // =
            }
            let p = self.expect("IDENT")?;
            Some(Self::ident_str(&p)?)
        } else { None };
        self.expect("RPAREN")?;
        Ok(Expr::Grad(Box::new(GradExpr { expr, wrt, span })))
    }

    fn parse_stop_grad_expr(&mut self) -> Result<Expr, NeuronError> {
        let span = self.span();
        self.advance(); // stop_grad
        self.expect("LPAREN")?;
        let expr = self.parse_expression(0)?;
        self.expect("RPAREN")?;
        Ok(Expr::StopGrad(Box::new(expr), span))
    }

    fn parse_do_expr(&mut self) -> Result<Expr, NeuronError> {
        let span = self.span();
        self.advance(); // do
        self.expect("LPAREN")?;
        let mut assignments = Vec::new();
        while !matches!(self.peek_type(), TokenType::RParen | TokenType::Eof) {
            let name_tok = self.expect("IDENT")?;
            let name = Self::ident_str(&name_tok)?;
            self.expect("EQ")?;
            let val = self.parse_expression(0)?;
            assignments.push((name, val));
            self.match_token(|t| matches!(t, TokenType::Comma));
        }
        self.expect("RPAREN")?;
        Ok(Expr::Do(Box::new(DoExpr { assignments, span })))
    }

    fn parse_explain_expr(&mut self) -> Result<Expr, NeuronError> {
        let span = self.span();
        self.advance(); // explain
        self.expect("LPAREN")?;
        let expr = self.parse_expression(0)?;
        self.expect("RPAREN")?;
        Ok(Expr::Explain(Box::new(ExplainExpr { expr, span })))
    }

    fn parse_search_expr(&mut self) -> Result<Expr, NeuronError> {
        let span = self.span();
        self.advance(); // search
        self.expect("LPAREN")?;
        let space = self.parse_expression(0)?;
        self.expect("COMMA")?;
        let evaluate = self.parse_expression(0)?;
        let strategy = if self.match_token(|t| matches!(t, TokenType::Comma)).is_some() {
            Some(self.parse_expression(0)?)
        } else { None };
        self.expect("RPAREN")?;
        Ok(Expr::SearchExpr(Box::new(SearchExpr { space, evaluate, strategy, span })))
    }

    fn parse_recall_expr(&mut self) -> Result<Expr, NeuronError> {
        let span = self.span();
        self.advance(); // recall
        self.expect("LPAREN")?;
        let memory = self.parse_expression(0)?;
        self.expect("COMMA")?;
        let query = self.parse_expression(0)?;
        let k = if self.match_token(|t| matches!(t, TokenType::Comma)).is_some() {
            Some(self.parse_expression(0)?)
        } else { None };
        self.expect("RPAREN")?;
        Ok(Expr::RecallExpr(Box::new(RecallExpr { memory, query, k, span })))
    }

    fn parse_store_expr(&mut self) -> Result<Expr, NeuronError> {
        let span = self.span();
        self.advance(); // store
        self.expect("LPAREN")?;
        let memory = self.parse_expression(0)?;
        self.expect("COMMA")?;
        let item = self.parse_expression(0)?;
        self.expect("RPAREN")?;
        Ok(Expr::StoreExpr(Box::new(StoreExpr { memory, item, span })))
    }

    fn parse_observe_args(&mut self) -> Result<Vec<(String, Expr)>, NeuronError> {
        self.expect("LPAREN")?;
        let mut assignments = Vec::new();
        while !matches!(self.peek_type(), TokenType::RParen | TokenType::Eof) {
            let name_tok = self.expect("IDENT")?;
            let name = Self::ident_str(&name_tok)?;
            self.expect("EQ")?;
            let val = self.parse_expression(0)?;
            assignments.push((name, val));
            self.match_token(|t| matches!(t, TokenType::Comma));
        }
        self.expect("RPAREN")?;
        Ok(assignments)
    }

    fn parse_merge_rest(&mut self, left: Expr, right: Expr, span: Span) -> Result<Expr, NeuronError> {
        self.advance(); // strategy
        self.expect("COLON")?;
        let strategy = Some(self.parse_expression(0)?);
        let mut preserve = Vec::new();
        let forget_clauses = Vec::new();
        // preserve: ["a", "b"]
        if matches!(self.peek_type(), TokenType::Preserve) {
            self.advance();
            self.expect("COLON")?;
            self.expect("LBRACKET")?;
            while !matches!(self.peek_type(), TokenType::RBracket | TokenType::Eof) {
                if let TokenType::StringLit(s) = self.peek_type().clone() {
                    preserve.push(s);
                    self.advance();
                } else { self.advance(); }
                self.match_token(|t| matches!(t, TokenType::Comma));
            }
            self.expect("RBRACKET")?;
        }
        Ok(Expr::Merge(Box::new(MergeExpr { left, right, strategy, preserve, forget_clauses, span })))
    }

    fn parse_call_expr_from(&mut self, callee: Expr) -> Result<Expr, NeuronError> {
        let span = callee.span().clone();
        let args = self.parse_call_args()?;
        Ok(Expr::FnCall(Box::new(FnCallExpr { callee, args, span })))
    }

    fn parse_call_args(&mut self) -> Result<Vec<CallArg>, NeuronError> {
        self.expect("LPAREN")?;
        let mut args = Vec::new();
        if !matches!(self.peek_type(), TokenType::RParen) {
            loop {
                // Named arg: name=expr
                if matches!(self.peek_type(), TokenType::Ident(_)) && matches!(self.peek_ahead(1).ty, TokenType::Eq) {
                    let name_tok = self.advance().clone();
                    let name = Self::ident_str(&name_tok)?;
                    self.advance(); // =
                    let value = self.parse_expression(0)?;
                    args.push(CallArg { name: Some(name), value });
                } else {
                    let value = self.parse_expression(0)?;
                    args.push(CallArg { name: None, value });
                }
                if self.match_token(|t| matches!(t, TokenType::Comma)).is_none() { break; }
                if matches!(self.peek_type(), TokenType::RParen) { break; }
            }
        }
        self.expect("RPAREN")?;
        Ok(args)
    }

    fn parse_index_items(&mut self) -> Result<Vec<IndexItem>, NeuronError> {
        let mut items = Vec::new();
        loop {
            if matches!(self.peek_type(), TokenType::Colon) {
                self.advance();
                // Full slice or slice with end
                if matches!(self.peek_type(), TokenType::Comma | TokenType::RBracket) {
                    items.push(IndexItem::Full);
                } else {
                    let end = self.parse_expression(0)?;
                    items.push(IndexItem::Slice { start: None, end: Some(end) });
                }
            } else {
                let expr = self.parse_expression(0)?;
                if matches!(self.peek_type(), TokenType::Colon) {
                    self.advance();
                    if matches!(self.peek_type(), TokenType::Comma | TokenType::RBracket) {
                        items.push(IndexItem::Slice { start: Some(expr), end: None });
                    } else {
                        let end = self.parse_expression(0)?;
                        items.push(IndexItem::Slice { start: Some(expr), end: Some(end) });
                    }
                } else {
                    items.push(IndexItem::Expr(expr));
                }
            }
            if self.match_token(|t| matches!(t, TokenType::Comma)).is_none() { break; }
            if matches!(self.peek_type(), TokenType::RBracket) { break; }
        }
        Ok(items)
    }

    // ═══════════════════════════════════════
    //  Type parsing
    // ═══════════════════════════════════════

    fn parse_type(&mut self) -> Result<TypeExpr, NeuronError> {
        let span = self.span();
        match self.peek_type().clone() {
            TokenType::IntKw => { self.advance(); Ok(TypeExpr::Base("Int".into(), span)) }
            TokenType::FloatKw => { self.advance(); Ok(TypeExpr::Base("Float".into(), span)) }
            TokenType::BoolKw => { self.advance(); Ok(TypeExpr::Base("Bool".into(), span)) }
            TokenType::StringKw => { self.advance(); Ok(TypeExpr::Base("String".into(), span)) }
            TokenType::TimestampKw => { self.advance(); Ok(TypeExpr::Base("Timestamp".into(), span)) }
            TokenType::LossKw => { self.advance(); Ok(TypeExpr::Base("Loss".into(), span)) }
            TokenType::DatasetKw => { self.advance(); Ok(TypeExpr::Base("Dataset".into(), span)) }
            TokenType::ExperienceKw => { self.advance(); Ok(TypeExpr::Base("Experience".into(), span)) }

            TokenType::TensorKw => {
                self.advance();
                if matches!(self.peek_type(), TokenType::LBracket) {
                    self.advance();
                    let dims = self.parse_dims()?;
                    self.expect("RBRACKET")?;
                    Ok(TypeExpr::Tensor(dims, span))
                } else {
                    Ok(TypeExpr::Tensor(vec![], span))
                }
            }
            TokenType::UncertainKw => {
                self.advance();
                self.expect("LBRACKET")?;
                let inner = self.parse_type()?;
                self.expect("RBRACKET")?;
                Ok(TypeExpr::Uncertain(Box::new(inner), span))
            }
            TokenType::RandomKw => {
                self.advance();
                self.expect("LBRACKET")?;
                let inner = self.parse_type()?;
                self.expect("RBRACKET")?;
                Ok(TypeExpr::Random(Box::new(inner), span))
            }
            TokenType::ProbKw => {
                self.advance();
                self.expect("LBRACKET")?;
                let inner = self.parse_type()?;
                self.expect("RBRACKET")?;
                Ok(TypeExpr::Prob(Box::new(inner), span))
            }
            TokenType::TemporalKw => {
                self.advance();
                self.expect("LBRACKET")?;
                let inner = self.parse_type()?;
                self.expect("COMMA")?;
                let dir = self.parse_temporal_direction()?;
                self.expect("RBRACKET")?;
                Ok(TypeExpr::Temporal(Box::new(inner), dir, span))
            }
            TokenType::CausalKw => {
                self.advance();
                self.expect("LBRACKET")?;
                let inner = self.parse_type()?;
                self.expect("COMMA")?;
                let mode_tok = self.expect("IDENT")?;
                let mode = Self::ident_str(&mode_tok)?;
                self.expect("RBRACKET")?;
                Ok(TypeExpr::Causal(Box::new(inner), mode.to_lowercase(), span))
            }
            TokenType::LearnableKw => {
                self.advance();
                self.expect("LBRACKET")?;
                let fn_type = if let TokenType::Ident(s) = self.peek_type().clone() {
                    self.advance();
                    s
                } else {
                    let t = self.parse_type()?;
                    format!("{:?}", t)
                };
                let base = if self.match_token(|t| matches!(t, TokenType::Comma)).is_some() {
                    // base=expr
                    if matches!(self.peek_type(), TokenType::Ident(_)) && matches!(self.peek_ahead(1).ty, TokenType::Eq) {
                        self.advance(); // base
                        self.advance(); // =
                    }
                    Some(Box::new(self.parse_expression(0)?))
                } else { None };
                self.expect("RBRACKET")?;
                Ok(TypeExpr::Learnable(fn_type, base, span))
            }
            TokenType::ListKw => {
                self.advance();
                if matches!(self.peek_type(), TokenType::LBracket) {
                    self.advance();
                    let inner = self.parse_type()?;
                    self.expect("RBRACKET")?;
                    Ok(TypeExpr::ListType(Box::new(inner), span))
                } else {
                    Ok(TypeExpr::ListType(Box::new(TypeExpr::Base("Any".into(), span.clone())), span))
                }
            }
            TokenType::OptionKw => {
                self.advance();
                self.expect("LBRACKET")?;
                let inner = self.parse_type()?;
                self.expect("RBRACKET")?;
                Ok(TypeExpr::OptionType(Box::new(inner), span))
            }
            TokenType::EffectKw => {
                self.advance();
                // Effect type parsed but returned as Base for simplicity in type expressions
                Ok(TypeExpr::Base("Effect".into(), span))
            }
            // AGI types
            TokenType::MemoryKw => {
                self.advance();
                self.expect("LBRACKET")?;
                let inner = self.parse_type()?;
                self.expect("RBRACKET")?;
                Ok(TypeExpr::Memory(Box::new(inner), span))
            }
            TokenType::EpisodicMemoryKw => {
                self.advance();
                self.expect("LBRACKET")?;
                let inner = self.parse_type()?;
                self.expect("RBRACKET")?;
                Ok(TypeExpr::EpisodicMemory(Box::new(inner), span))
            }
            TokenType::SemanticMemoryKw => {
                self.advance();
                self.expect("LBRACKET")?;
                let inner = self.parse_type()?;
                self.expect("RBRACKET")?;
                Ok(TypeExpr::SemanticMemory(Box::new(inner), span))
            }
            TokenType::WorkingMemoryKw => {
                self.advance();
                self.expect("LBRACKET")?;
                let inner = self.parse_type()?;
                let capacity = if self.match_token(|t| matches!(t, TokenType::Comma)).is_some() {
                    Some(Box::new(self.parse_expression(0)?))
                } else { None };
                self.expect("RBRACKET")?;
                Ok(TypeExpr::WorkingMemory(Box::new(inner), capacity, span))
            }
            TokenType::RewardKw => {
                self.advance();
                if matches!(self.peek_type(), TokenType::LBracket) {
                    self.advance();
                    let inner = self.parse_type()?;
                    self.expect("RBRACKET")?;
                    Ok(TypeExpr::RewardType(Box::new(inner), span))
                } else {
                    Ok(TypeExpr::RewardType(Box::new(TypeExpr::Base("Float".into(), span.clone())), span))
                }
            }
            TokenType::Fn => {
                self.advance();
                self.expect("LPAREN")?;
                let mut param_types = Vec::new();
                if !matches!(self.peek_type(), TokenType::RParen) {
                    param_types.push(self.parse_type()?);
                    while self.match_token(|t| matches!(t, TokenType::Comma)).is_some() {
                        param_types.push(self.parse_type()?);
                    }
                }
                self.expect("RPAREN")?;
                self.expect("ARROW")?;
                let ret = self.parse_type()?;
                Ok(TypeExpr::Fn(param_types, Box::new(ret), span))
            }
            TokenType::Ident(name) => {
                let name = name.clone();
                self.advance();
                Ok(TypeExpr::UserDefined(name, span))
            }
            _ => Err(self.error(format!("Expected type, got {}", self.peek_type().name()))),
        }
    }

    fn parse_dims(&mut self) -> Result<Vec<DimExpr>, NeuronError> {
        let mut dims = Vec::new();
        loop {
            dims.push(self.parse_dim()?);
            if self.match_token(|t| matches!(t, TokenType::Comma)).is_none() { break; }
            if matches!(self.peek_type(), TokenType::RBracket) { break; }
        }
        Ok(dims)
    }

    fn parse_dim(&mut self) -> Result<DimExpr, NeuronError> {
        match self.peek_type().clone() {
            TokenType::IntLit(v) => { self.advance(); Ok(DimExpr::Static(v)) }
            TokenType::Question => { self.advance(); Ok(DimExpr::Dynamic) }
            TokenType::Ident(name) => {
                let name = name.clone();
                self.advance();
                // Check for named dim: B:batch
                if matches!(self.peek_type(), TokenType::Colon) {
                    self.advance();
                    let alias_tok = self.expect("IDENT")?;
                    let alias = Self::ident_str(&alias_tok)?;
                    Ok(DimExpr::Named(name, alias))
                } else {
                    Ok(DimExpr::Symbolic(name))
                }
            }
            _ => Err(self.error("Expected dimension: integer, identifier, or '?'")),
        }
    }

    fn parse_temporal_direction(&mut self) -> Result<String, NeuronError> {
        // past→future or future→past
        let first_tok = self.expect("IDENT")?;
        let first = Self::ident_str(&first_tok)?;
        if matches!(self.peek_type(), TokenType::Arrow | TokenType::UnicodeArrow) {
            self.advance();
            let second_tok = self.expect("IDENT")?;
            let second = Self::ident_str(&second_tok)?;
            Ok(format!("{}_to_{}", first, second))
        } else {
            Ok(first)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse_str(src: &str) -> Program {
        let tokens = Lexer::new(src).tokenize().unwrap();
        Parser::new(tokens, "<test>").parse().unwrap()
    }

    #[test]
    fn test_simple_let() {
        let prog = parse_str("let x: Int = 42");
        assert_eq!(prog.top_levels.len(), 1);
        if let TopLevel::Let(ref s) = prog.top_levels[0] {
            assert_eq!(s.name, "x");
        } else { panic!("Expected Let"); }
    }

    #[test]
    fn test_model_decl() {
        let src = "model Foo(d: Int):\n  w: Tensor[d, 128]\n\n  fn forward(x: Tensor[B, d]) -> Tensor[B, 128]:\n    return x @ self.w";
        let prog = parse_str(src);
        if let TopLevel::Model(ref m) = prog.top_levels[0] {
            assert_eq!(m.name, "Foo");
            assert_eq!(m.params.len(), 1);
            assert_eq!(m.fields.len(), 1);
            assert_eq!(m.methods.len(), 1);
        } else { panic!("Expected Model"); }
    }

    #[test]
    fn test_causal_decl() {
        let src = "causal model Trial [confidence=Certain]:\n  treatment -> outcome\n  age -> outcome";
        let prog = parse_str(src);
        if let TopLevel::Causal(ref c) = prog.top_levels[0] {
            assert_eq!(c.name, "Trial");
            assert_eq!(c.edges.len(), 2);
        } else { panic!("Expected Causal"); }
    }

    #[test]
    fn test_annotation() {
        let src = "@compile(target=\"auto\")\nmodel M:\n  x: Int = 0";
        let prog = parse_str(src);
        if let TopLevel::Model(ref m) = prog.top_levels[0] {
            assert_eq!(m.annotations.len(), 1);
            assert_eq!(m.annotations[0].name, "compile");
        } else { panic!("Expected Model"); }
    }

    #[test]
    fn test_temporal_type() {
        let src = "let x: Temporal[Tensor[B, 5], past→future] = zeros(1, 5)";
        let prog = parse_str(src);
        if let TopLevel::Let(ref l) = prog.top_levels[0] {
            assert!(matches!(l.type_ann, Some(TypeExpr::Temporal(_, _, _))));
        } else { panic!("Expected Let"); }
    }

    #[test]
    fn test_agent_decl() {
        let src = "agent Explorer(dim: Int):\n  policy: Tensor[dim, 4]\n\n  fn act(x: Tensor[dim]) -> Tensor[4]:\n    return x @ self.policy";
        let prog = parse_str(src);
        assert!(matches!(prog.top_levels[0], TopLevel::Agent(_)));
    }
}
