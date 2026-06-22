use neuron_compiler::lexer::Lexer;
use neuron_compiler::token::TokenType;

#[test]
fn test_integration_lexer_basics() {
    let src = r#"agent Explorer(state_dim: Int, action_dim: Int):
  policy: Tensor[state_dim, action_dim] = glorot(state_dim, action_dim)
  
  fn act(self, obs: Tensor[state_dim]) -> Tensor[action_dim]:
    let logits = obs @ self.policy
    return softmax(logits)
"#;
    let tokens = Lexer::new(src).tokenize().unwrap();
    assert!(!tokens.is_empty());
    
    // Verify we have some expected keywords
    let has_agent = tokens.iter().any(|t| matches!(t.ty, TokenType::Agent));
    let has_fn = tokens.iter().any(|t| matches!(t.ty, TokenType::Fn));
    let has_let = tokens.iter().any(|t| matches!(t.ty, TokenType::Let));
    let has_return = tokens.iter().any(|t| matches!(t.ty, TokenType::Return));
    
    assert!(has_agent, "Missing agent keyword");
    assert!(has_fn, "Missing fn keyword");
    assert!(has_let, "Missing let keyword");
    assert!(has_return, "Missing return keyword");
}
