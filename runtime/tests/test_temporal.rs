use neuron_compiler::ir::{IRProgram, IRFunction, IRNode, IROp, IRType, DeviceTarget, BasicBlock, Terminator};
use neuron_runtime::vm::{VM, Value};

#[test]
fn test_integration_temporal_violation() {
    let mut prog = IRProgram::new();
    let mut func = IRFunction::new("main");
    
    let block = BasicBlock {
        id: 0,
        instructions: vec![
            // Load the temporal value from global scope
            // ValueId 0
            IRNode {
                id: 0,
                op: IROp::Load { name: "my_val".into() },
                inputs: vec![],
                output_type: IRType::Temporal(Box::new(IRType::F64), "past_to_future".into()),
                output_shape: vec![],
                grad_fn: None,
                device: DeviceTarget::Auto,
                temporal_dir: None,
                effects: vec![],
            },
            // Check direction: expect "future_to_past", which causes a violation
            IRNode {
                id: 1,
                op: IROp::TemporalCheckDir { expected: "future_to_past".into() },
                inputs: vec![0],
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
    
    // Inject the Temporal value into globals
    vm.set_global(
        "my_val".to_string(),
        Value::Temporal {
            data: Box::new(Value::Float(1.0)),
            direction: "past_to_future".to_string(),
        },
    );
    
    let result = vm.run_main();
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.contains("temporal direction violation"), "Expected temporal violation error, got: {}", err);
}
