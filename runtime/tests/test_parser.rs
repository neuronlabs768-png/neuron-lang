use neuron_compiler::lexer::Lexer;
use neuron_compiler::parser::Parser;

#[test]
fn test_integration_parser_declarations() {
    let src = r#"agent A(x: Int):
  w: Tensor[x, 1] = zeros(x, 1)

model M(y: Int):
  w: Tensor[y, y] = glorot(y, y)

fn add_one(val: Int) -> Int:
  return val + 1
"#;
    let tokens = Lexer::new(src).tokenize().unwrap();
    let program = Parser::new(tokens, "test_parser_declarations.nr").parse();
    assert!(program.is_ok(), "Failed to parse: {:?}", program.err());
    let prog = program.unwrap();
    assert_eq!(prog.top_levels.len(), 3);
}
