use neuron_compiler::ir::{IRProgram, IRFunction, IRNode, IROp, IRType, DeviceTarget, BasicBlock, Terminator};
use neuron_runtime::vm::VM;

#[test]
fn test_integration_causal_mismatch() {
    let mut prog = IRProgram::new();
    let mut func = IRFunction::new("main");
    
    let block = BasicBlock {
        id: 0,
        instructions: vec![
            // ValueId 0: a constant value
            IRNode {
                id: 0,
                op: IROp::Const(neuron_compiler::ir::IRConst::Float(2.5)),
                inputs: vec![],
                output_type: IRType::F64,
                output_shape: vec![],
                grad_fn: None,
                device: DeviceTarget::Auto,
                temporal_dir: None,
                effects: vec![],
            },
            // ValueId 1: wrap as observed causal
            IRNode {
                id: 1,
                op: IROp::Observe,
                inputs: vec![0],
                output_type: IRType::Causal(Box::new(IRType::F64), "observed".into()),
                output_shape: vec![],
                grad_fn: None,
                device: DeviceTarget::Auto,
                temporal_dir: None,
                effects: vec![],
            },
            // ValueId 2: causal check mode, expecting "intervened" on the observed value, causing mismatch
            IRNode {
                id: 2,
                op: IROp::CausalCheckMode { expected: "intervened".into() },
                inputs: vec![1],
                output_type: IRType::Void,
                output_shape: vec![],
                grad_fn: None,
                device: DeviceTarget::Auto,
                temporal_dir: None,
                effects: vec![],
            },
        ],
        terminator: Terminator::Return(Some(0)),
    };
    
    func.blocks.push(block);
    func.entry = 0;
    prog.functions.push(func);
    
    let mut vm = VM::new();
    vm.load(&prog);
    
    let result = vm.run_main();
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.contains("causal type mismatch"), "Expected causal mismatch error, got: {}", err);
}
