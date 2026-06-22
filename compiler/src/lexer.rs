/// NEURON Lexer — tokenizes source into a stream of tokens.
///
/// Handles indentation-based blocks (INDENT/DEDENT), unicode arrows,
/// `//` comments, implicit line continuation in brackets, annotations,
/// scientific notation, underscore separators in numbers.

use crate::token::{lookup_keyword, Span, Token, TokenType};
use crate::errors::{ErrorCode, NeuronError};

pub struct Lexer {
    source: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
    tokens: Vec<Token>,
    indent_stack: Vec<usize>,
    bracket_depth: usize,
    at_line_start: bool,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
            tokens: Vec::new(),
            indent_stack: vec![0],
            bracket_depth: 0,
            at_line_start: true,
        }
    }

    pub fn tokenize(mut self) -> Result<Vec<Token>, NeuronError> {
        while !self.at_end() {
            self.skip_blank_lines_and_comments();
            if self.at_end() {
                break;
            }

            // Process indentation at the start of a non-blank, non-comment line
            if self.at_line_start && self.bracket_depth == 0 {
                self.process_indentation()?;
            }

            self.lex_line()?;

            // Emit NEWLINE at end of line content (if not inside brackets)
            if self.bracket_depth == 0 && !self.at_line_start {
                self.emit(TokenType::Newline, self.line, self.col, 1);
                self.at_line_start = true;
            }
        }

        // Emit remaining DEDENTs at EOF
        let eof_line = self.line;
        while self.indent_stack.len() > 1 {
            self.indent_stack.pop();
            self.emit(TokenType::Dedent, eof_line, 1, 0);
        }
        self.emit(TokenType::Eof, eof_line, self.col, 0);
        Ok(self.tokens)
    }

    // ── helpers ──────────────────────────────

    fn at_end(&self) -> bool {
        self.pos >= self.source.len()
    }

    fn peek(&self) -> char {
        if self.at_end() { '\0' } else { self.source[self.pos] }
    }

    fn peek_ahead(&self, offset: usize) -> char {
        let idx = self.pos + offset;
        if idx >= self.source.len() { '\0' } else { self.source[idx] }
    }

    fn advance(&mut self) -> char {
        let ch = self.source[self.pos];
        self.pos += 1;
        self.col += 1;
        ch
    }

    fn emit(&mut self, ty: TokenType, line: usize, col: usize, len: usize) {
        self.tokens.push(Token::new(ty, line, col, len));
    }

    fn error(&self, msg: impl Into<String>) -> NeuronError {
        NeuronError::new(ErrorCode::UnexpectedChar, msg, Span::new(self.line, self.col, 1))
    }

    // ── blank lines / comments ──────────────

    fn skip_blank_lines_and_comments(&mut self) {
        loop {
            // Skip to first non-space character on this line
            let start = self.pos;
            while !self.at_end() && self.peek() == ' ' {
                self.advance();
            }
            if self.at_end() {
                return;
            }
            // Check for comment
            if self.peek() == '/' && self.peek_ahead(1) == '/' {
                self.skip_to_eol();
                self.consume_newline();
                continue;
            }
            // Check for blank line
            if self.peek() == '\n' {
                self.consume_newline();
                continue;
            }
            if self.peek() == '\r' {
                self.advance();
                if self.peek() == '\n' {
                    self.consume_newline();
                } else {
                    self.line += 1;
                    self.col = 1;
                    self.at_line_start = true;
                }
                continue;
            }
            // Non-blank content found — rewind to line start for indent processing
            self.pos = start;
            self.col = 1;
            return;
        }
    }

    fn skip_to_eol(&mut self) {
        while !self.at_end() && self.peek() != '\n' && self.peek() != '\r' {
            self.pos += 1;
        }
    }

    fn consume_newline(&mut self) {
        if !self.at_end() && self.peek() == '\r' {
            self.pos += 1;
        }
        if !self.at_end() && self.peek() == '\n' {
            self.pos += 1;
        }
        self.line += 1;
        self.col = 1;
        self.at_line_start = true;
    }

    // ── indentation ─────────────────────────

    fn process_indentation(&mut self) -> Result<(), NeuronError> {
        let mut indent = 0usize;
        while !self.at_end() && self.peek() == ' ' {
            indent += 1;
            self.advance();
        }
        if !self.at_end() && self.peek() == '\t' {
            return Err(NeuronError::new(
                ErrorCode::TabIndent,
                "Tabs are not allowed for indentation; use spaces",
                Span::new(self.line, self.col, 1),
            ));
        }
        let current = *self.indent_stack.last().unwrap();
        if indent > current {
            self.indent_stack.push(indent);
            self.emit(TokenType::Indent, self.line, 1, indent);
        } else if indent < current {
            while self.indent_stack.len() > 1 && *self.indent_stack.last().unwrap() > indent {
                self.indent_stack.pop();
                self.emit(TokenType::Dedent, self.line, 1, 0);
            }
            if *self.indent_stack.last().unwrap() != indent {
                return Err(NeuronError::new(
                    ErrorCode::InconsistentIndent,
                    format!(
                        "Inconsistent indentation: got {} spaces, but no matching outer indent level",
                        indent
                    ),
                    Span::new(self.line, 1, indent),
                ));
            }
        }
        Ok(())
    }

    // ── line lexing ─────────────────────────

    fn lex_line(&mut self) -> Result<(), NeuronError> {
        while !self.at_end() {
            let ch = self.peek();

            // End of line
            if ch == '\n' || ch == '\r' {
                if self.bracket_depth > 0 {
                    // Implicit line continuation inside brackets
                    self.consume_newline();
                    // Skip leading whitespace on continuation line
                    while !self.at_end() && self.peek() == ' ' {
                        self.advance();
                    }
                    continue;
                }
                self.consume_newline();
                return Ok(());
            }

            // Skip spaces
            if ch == ' ' || ch == '\t' {
                self.advance();
                continue;
            }

            // Comment
            if ch == '/' && self.peek_ahead(1) == '/' {
                self.skip_to_eol();
                continue;
            }

            let start_col = self.col;
            let start_line = self.line;

            // ── String literal ──
            if ch == '"' {
                self.lex_string(start_line, start_col)?;
                self.at_line_start = false;
                continue;
            }

            // ── Number literal ──
            if ch.is_ascii_digit() {
                self.lex_number(start_line, start_col)?;
                self.at_line_start = false;
                continue;
            }

            // ── Identifier / keyword ──
            if ch.is_alphabetic() || ch == '_' {
                self.lex_identifier(start_line, start_col);
                self.at_line_start = false;
                continue;
            }

            // ── Unicode arrow → ──
            if ch == '\u{2192}' {
                self.advance();
                self.emit(TokenType::UnicodeArrow, start_line, start_col, 1);
                self.at_line_start = false;
                continue;
            }

            // ── @ — annotation vs AT ──
            if ch == '@' {
                self.lex_at(start_line, start_col);
                continue;
            }

            // ── Two-character operators ──
            let next = self.peek_ahead(1);
            match (ch, next) {
                ('-', '>') => {
                    self.advance(); self.advance();
                    self.emit(TokenType::Arrow, start_line, start_col, 2);
                    self.at_line_start = false;
                    continue;
                }
                ('>', '=') => {
                    self.advance(); self.advance();
                    self.emit(TokenType::Gte, start_line, start_col, 2);
                    self.at_line_start = false;
                    continue;
                }
                ('<', '=') => {
                    self.advance(); self.advance();
                    self.emit(TokenType::Lte, start_line, start_col, 2);
                    self.at_line_start = false;
                    continue;
                }
                ('=', '=') => {
                    self.advance(); self.advance();
                    self.emit(TokenType::EqEq, start_line, start_col, 2);
                    self.at_line_start = false;
                    continue;
                }
                ('!', '=') => {
                    self.advance(); self.advance();
                    self.emit(TokenType::Neq, start_line, start_col, 2);
                    self.at_line_start = false;
                    continue;
                }
                ('&', '&') => {
                    self.advance(); self.advance();
                    self.emit(TokenType::And, start_line, start_col, 2);
                    self.at_line_start = false;
                    continue;
                }
                ('|', '|') => {
                    self.advance(); self.advance();
                    self.emit(TokenType::Or, start_line, start_col, 2);
                    self.at_line_start = false;
                    continue;
                }
                _ => {}
            }

            // ── Single-character operators / delimiters ──
            let tok = match ch {
                '+' => Some(TokenType::Plus),
                '-' => Some(TokenType::Minus),
                '*' => Some(TokenType::Star),
                '/' => Some(TokenType::Slash),
                '%' => Some(TokenType::Percent),
                '>' => Some(TokenType::Gt),
                '<' => Some(TokenType::Lt),
                '=' => Some(TokenType::Eq),
                '!' => Some(TokenType::Bang),
                '.' => Some(TokenType::Dot),
                ':' => Some(TokenType::Colon),
                ',' => Some(TokenType::Comma),
                '?' => Some(TokenType::Question),
                '(' => { self.bracket_depth += 1; Some(TokenType::LParen) }
                ')' => { if self.bracket_depth > 0 { self.bracket_depth -= 1; } Some(TokenType::RParen) }
                '[' => { self.bracket_depth += 1; Some(TokenType::LBracket) }
                ']' => { if self.bracket_depth > 0 { self.bracket_depth -= 1; } Some(TokenType::RBracket) }
                '{' => { self.bracket_depth += 1; Some(TokenType::LBrace) }
                '}' => { if self.bracket_depth > 0 { self.bracket_depth -= 1; } Some(TokenType::RBrace) }
                _ => None,
            };

            if let Some(tt) = tok {
                self.advance();
                self.emit(tt, start_line, start_col, 1);
                self.at_line_start = false;
                continue;
            }

            return Err(self.error(format!("Unexpected character: {:?} (U+{:04X})", ch, ch as u32)));
        }
        Ok(())
    }

    // ── individual token lexers ─────────────

    fn lex_string(&mut self, line: usize, col: usize) -> Result<(), NeuronError> {
        self.advance(); // skip opening "
        let mut value = String::new();
        while !self.at_end() && self.peek() != '"' {
            let ch = self.peek();
            if ch == '\n' || ch == '\r' {
                return Err(NeuronError::new(
                    ErrorCode::UnterminatedString,
                    "Unterminated string literal",
                    Span::new(line, col, 1),
                ));
            }
            value.push(self.advance());
        }
        if self.at_end() {
            return Err(NeuronError::new(
                ErrorCode::UnterminatedString,
                "Unterminated string literal",
                Span::new(line, col, 1),
            ));
        }
        self.advance(); // skip closing "
        let len = value.len() + 2;
        self.emit(TokenType::StringLit(value), line, col, len);
        Ok(())
    }

    fn lex_number(&mut self, line: usize, col: usize) -> Result<(), NeuronError> {
        let start = self.pos;
        let mut is_float = false;

        // Consume integer part (digits + underscores)
        self.consume_digits();

        // Decimal point
        if self.peek() == '.' && self.peek_ahead(1).is_ascii_digit() {
            is_float = true;
            self.advance(); // '.'
            self.consume_digits();
        }

        // Scientific notation
        if self.peek() == 'e' || self.peek() == 'E' {
            let next = self.peek_ahead(1);
            let after = if next == '+' || next == '-' { self.peek_ahead(2) } else { next };
            if after.is_ascii_digit() {
                is_float = true;
                self.advance(); // 'e'/'E'
                if self.peek() == '+' || self.peek() == '-' {
                    self.advance();
                }
                self.consume_digits();
            }
        }

        let raw: String = self.source[start..self.pos].iter().collect();
        let clean = raw.replace('_', "");
        let len = raw.len();

        if is_float {
            let val: f64 = clean.parse().map_err(|_| self.error(format!("Invalid float: {}", raw)))?;
            self.emit(TokenType::FloatLit(val), line, col, len);
        } else {
            let val: i64 = clean.parse().map_err(|_| self.error(format!("Invalid integer: {}", raw)))?;
            self.emit(TokenType::IntLit(val), line, col, len);
        }
        Ok(())
    }

    fn consume_digits(&mut self) {
        while !self.at_end() && (self.peek().is_ascii_digit() || self.peek() == '_') {
            self.advance();
        }
    }

    fn lex_identifier(&mut self, line: usize, col: usize) {
        let start = self.pos;
        while !self.at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
            self.advance();
        }
        let word: String = self.source[start..self.pos].iter().collect();
        let len = word.len();

        if let Some(kw) = lookup_keyword(&word) {
            self.emit(kw, line, col, len);
        } else {
            self.emit(TokenType::Ident(word), line, col, len);
        }
    }

    fn lex_at(&mut self, line: usize, col: usize) {
        self.advance(); // skip '@'

        // If followed by identifier at line start → annotation
        if !self.at_end() && (self.peek().is_alphabetic() || self.peek() == '_') && self.at_line_start {
            let start = self.pos;
            while !self.at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
                self.advance();
            }
            let name: String = self.source[start..self.pos].iter().collect();
            let len = name.len() + 1;
            self.emit(TokenType::Annotation(name), line, col, len);
        } else {
            self.emit(TokenType::At, line, col, 1);
        }
        self.at_line_start = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Vec<Token> {
        Lexer::new(src).tokenize().unwrap()
    }

    fn types(src: &str) -> Vec<String> {
        lex(src).iter().map(|t| t.ty.name().to_string()).collect()
    }

    #[test]
    fn test_empty() {
        let toks = lex("");
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].ty, TokenType::Eof);
    }

    #[test]
    fn test_simple_let() {
        let toks = lex("let x = 42");
        assert!(matches!(toks[0].ty, TokenType::Let));
        assert!(matches!(toks[1].ty, TokenType::Ident(ref s) if s == "x"));
        assert!(matches!(toks[2].ty, TokenType::Eq));
        assert!(matches!(toks[3].ty, TokenType::IntLit(42)));
    }

    #[test]
    fn test_float_scientific() {
        let toks = lex("3.14 1e-3 1.0e+5");
        assert!(matches!(toks[0].ty, TokenType::FloatLit(v) if (v - 3.14).abs() < 1e-10));
        assert!(matches!(toks[1].ty, TokenType::FloatLit(v) if (v - 0.001).abs() < 1e-10));
        assert!(matches!(toks[2].ty, TokenType::FloatLit(v) if (v - 100000.0).abs() < 1e-5));
    }

    #[test]
    fn test_unicode_arrow() {
        let toks = lex("past→future");
        assert!(matches!(toks[0].ty, TokenType::Ident(ref s) if s == "past"));
        assert!(matches!(toks[1].ty, TokenType::UnicodeArrow));
        assert!(matches!(toks[2].ty, TokenType::Ident(ref s) if s == "future"));
    }

    #[test]
    fn test_indent_dedent() {
        let src = "model Foo:\n  let x = 1\n  let y = 2\nlet z = 3";
        let toks = lex(src);
        let names: Vec<&str> = toks.iter().map(|t| t.ty.name()).collect();
        assert!(names.contains(&"INDENT"));
        assert!(names.contains(&"DEDENT"));
    }

    #[test]
    fn test_annotation() {
        let toks = lex("@compile(target=\"auto\")");
        assert!(matches!(toks[0].ty, TokenType::Annotation(ref s) if s == "compile"));
    }

    #[test]
    fn test_two_char_ops() {
        let toks = lex("-> >= <= == != && ||");
        assert!(matches!(toks[0].ty, TokenType::Arrow));
        assert!(matches!(toks[1].ty, TokenType::Gte));
        assert!(matches!(toks[2].ty, TokenType::Lte));
        assert!(matches!(toks[3].ty, TokenType::EqEq));
        assert!(matches!(toks[4].ty, TokenType::Neq));
        assert!(matches!(toks[5].ty, TokenType::And));
        assert!(matches!(toks[6].ty, TokenType::Or));
    }

    #[test]
    fn test_bracket_continuation() {
        let src = "foo(\n  1,\n  2\n)";
        let toks = lex(src);
        // Should NOT have NEWLINE between args inside parens
        let newline_count = toks.iter().filter(|t| matches!(t.ty, TokenType::Newline)).count();
        // Only one newline at end of the entire expression line (after closing paren)
        assert!(newline_count <= 1, "got {} newlines inside brackets", newline_count);
    }

    #[test]
    fn test_int_underscores() {
        let toks = lex("100_000");
        assert!(matches!(toks[0].ty, TokenType::IntLit(100_000)));
    }

    #[test]
    fn test_keywords() {
        let toks = lex("agent meta search recall store stream");
        assert!(matches!(toks[0].ty, TokenType::Agent));
        assert!(matches!(toks[1].ty, TokenType::Meta));
        assert!(matches!(toks[2].ty, TokenType::Search));
        assert!(matches!(toks[3].ty, TokenType::Recall));
        assert!(matches!(toks[4].ty, TokenType::Store));
        assert!(matches!(toks[5].ty, TokenType::Stream));
    }
}
