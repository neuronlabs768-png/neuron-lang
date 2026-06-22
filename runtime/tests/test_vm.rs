use std::fs;
use neuron_compiler::compile;
use neuron_runtime::vm::VM;

#[test]
fn test_integration_vm_simple_shapes() {
    let src = fs::read_to_string("../examples/simple_shapes.nr")
        .or_else(|_| fs::read_to_string("examples/simple_shapes.nr"))
        .expect("Cannot read examples/simple_shapes.nr");
        
    let out = compile(&src, "simple_shapes.nr").unwrap();
    let mut vm = VM::new();
    vm.load(&out.ir);
    
    let result = vm.run_main();
    assert!(result.is_ok(), "Running simple_shapes.nr failed: {:?}", result.err());
}

#[test]
fn test_integration_vm_transformer_compiles() {
    let src = fs::read_to_string("../examples/transformer.nr")
        .or_else(|_| fs::read_to_string("examples/transformer.nr"))
        .expect("Cannot read examples/transformer.nr");
        
    let out = compile(&src, "transformer.nr");
    assert!(out.is_ok(), "Compiling transformer.nr failed: {:?}", out.err());
}

#[test]
fn test_integration_vm_agi_agent_compiles() {
    let src = fs::read_to_string("../examples/agi_agent.nr")
        .or_else(|_| fs::read_to_string("examples/agi_agent.nr"))
        .expect("Cannot read examples/agi_agent.nr");
        
    let out = compile(&src, "agi_agent.nr");
    assert!(out.is_ok(), "Compiling agi_agent.nr failed: {:?}", out.err());
}
