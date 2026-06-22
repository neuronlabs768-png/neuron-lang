use neuron_compiler::compile;
use neuron_runtime::vm::VM;

#[test]
fn test_integration_uncertainty() {
    let src = r#"let x = Normal(10.0, 1.0)
let val = x.value
let conf = x.confidence
let std_dev = x.std
print(val)
"#;
    let out = compile(src, "uncertainty.nr");
    assert!(out.is_ok(), "Expected compilation to succeed, got: {:?}", out.err());
    
    let output = out.unwrap();
    let mut vm = VM::new();
    vm.load(&output.ir);
    let result = vm.run_main();
    assert!(result.is_ok(), "Runtime failed: {:?}", result.err());
}
