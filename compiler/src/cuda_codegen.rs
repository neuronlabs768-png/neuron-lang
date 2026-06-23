use std::collections::HashSet;
use crate::ir::{IRFunction, IRNode, IROp, IRType, IRConst};

/// A generated CUDA kernel representation
#[derive(Debug, Clone)]
pub struct CudaKernel {
    pub name: String,
    pub code: String,
    pub inputs: Vec<usize>,      // Value IDs of the inputs to the fused block
    pub input_is_tensor: Vec<bool>, // True if corresponding input is a tensor, false if scalar
    pub output: usize,           // Value ID of the terminal output of the fused block
    pub output_shape: Vec<i64>,  // Shape of the output tensor
}

/// A group of fused IR instructions
#[derive(Debug, Clone)]
pub struct FusedGroup {
    pub instructions: Vec<IRNode>,
}

impl FusedGroup {
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

/// Checks if an instruction is fusable (element-wise or constant)
fn is_fusable(op: &IROp) -> bool {
    matches!(
        op,
        IROp::Add
            | IROp::Sub
            | IROp::Mul
            | IROp::Div
            | IROp::Neg
            | IROp::ReLU
            | IROp::GeLU
            | IROp::Sigmoid
            | IROp::Tanh
            | IROp::Const(_)
    )
}

fn is_tensor_type(ty: &IRType) -> bool {
    match ty {
        IRType::Tensor(_) => true,
        IRType::Uncertain(inner) => is_tensor_type(inner),
        IRType::Random(inner) => is_tensor_type(inner),
        IRType::Temporal(inner, _) => is_tensor_type(inner),
        IRType::Causal(inner, _) => is_tensor_type(inner),
        _ => false,
    }
}

pub fn get_tensor_ids(func: &IRFunction) -> HashSet<usize> {
    let mut set = HashSet::new();
    for param in &func.params {
        if is_tensor_type(&param.ty) {
            set.insert(param.id);
        }
    }
    for block in &func.blocks {
        for node in &block.instructions {
            if is_tensor_type(&node.output_type) || !node.output_shape.is_empty() {
                set.insert(node.id);
            }
        }
    }
    set
}

/// Identify contiguous groups of element-wise operators inside basic blocks
pub fn find_fused_groups(func: &IRFunction) -> Vec<FusedGroup> {
    let tensor_ids = get_tensor_ids(func);
    let mut groups = Vec::new();

    for block in &func.blocks {
        let mut current_group = Vec::new();
        let mut current_shape: Option<Vec<i64>> = None;

        for node in &block.instructions {
            if matches!(node.op, IROp::Store { .. }) {
                continue;
            }

            if is_fusable(&node.op) {
                let node_shape = node.output_shape.clone();
                
                // If it is a tensor operation, check shape matching
                let is_tensor = tensor_ids.contains(&node.id);
                let shape_compatible = if is_tensor {
                    if let Some(ref s) = current_shape {
                        s == &node_shape
                    } else {
                        current_shape = Some(node_shape.clone());
                        true
                    }
                } else {
                    true
                };

                if shape_compatible {
                    current_group.push(node.clone());
                } else {
                    // Finalize old group and start a new one
                    if !current_group.is_empty() {
                        groups.push(FusedGroup { instructions: current_group });
                    }
                    current_group = vec![node.clone()];
                    current_shape = if is_tensor { Some(node_shape) } else { None };
                }
            } else {
                // Non-fusable instruction: finalize current group
                if !current_group.is_empty() {
                    groups.push(FusedGroup { instructions: current_group });
                    current_group = Vec::new();
                    current_shape = None;
                }
            }
        }

        if !current_group.is_empty() {
            groups.push(FusedGroup { instructions: current_group });
        }
    }

    groups
}

/// Generates CUDA C++ kernels from fused groups
pub fn generate_cuda_kernels(func: &IRFunction) -> Vec<CudaKernel> {
    let tensor_ids = get_tensor_ids(func);
    let groups = find_fused_groups(func);
    let mut kernels = Vec::new();

    for (g_idx, group) in groups.iter().enumerate() {
        if group.is_empty() {
            continue;
        }

        // 1. Identify inputs & outputs of this group
        let mut produced_vals = HashSet::new();
        for node in &group.instructions {
            produced_vals.insert(node.id);
        }

        let mut input_set = HashSet::new();
        for node in &group.instructions {
            for &input in &node.inputs {
                if !produced_vals.contains(&input) {
                    input_set.insert(input);
                }
            }
        }

        let mut inputs: Vec<usize> = input_set.into_iter().collect();
        inputs.sort(); // Consistent order

        // The terminal instruction's output is our group output
        let terminal_node = group.instructions.last().unwrap();
        let output = terminal_node.id;
        let output_shape = terminal_node.output_shape.clone();

        // 2. Classify inputs as Tensor or Scalar
        let mut input_is_tensor = Vec::new();
        let mut arg_declarations = Vec::new();

        for &val_id in &inputs {
            let is_tensor = tensor_ids.contains(&val_id);
            input_is_tensor.push(is_tensor);

            if is_tensor {
                arg_declarations.push(format!("const double* v{}", val_id));
            } else {
                arg_declarations.push(format!("double v{}", val_id));
            }
        }

        // Add output parameter
        arg_declarations.insert(0, format!("double* v{}", output));
        // Add size parameter
        arg_declarations.push("int n".to_string());

        // 3. Generate kernel body
        let kernel_name = format!("fused_{}_{}", func.name, g_idx);
        let mut code = String::new();

        code.push_str("extern \"C\" __global__ void ");
        code.push_str(&kernel_name);
        code.push_str("(\n    ");
        code.push_str(&arg_declarations.join(",\n    "));
        code.push_str("\n) {\n");
        code.push_str("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
        code.push_str("    if (idx < n) {\n");

        // Generate element-wise assignments
        for node in &group.instructions {
            let expr = match &node.op {
                IROp::Const(c) => match c {
                    IRConst::Int(v) => format!("{}.0", v),
                    IRConst::Float(v) => format!("{:.6}", v),
                    IRConst::Bool(v) => format!("{}", if *v { "1.0" } else { "0.0" }),
                    _ => "0.0".to_string(),
                },
                IROp::Add => {
                    let in0 = format_input(node.inputs[0], &inputs, &tensor_ids);
                    let in1 = format_input(node.inputs[1], &inputs, &tensor_ids);
                    format!("{} + {}", in0, in1)
                }
                IROp::Sub => {
                    let in0 = format_input(node.inputs[0], &inputs, &tensor_ids);
                    let in1 = format_input(node.inputs[1], &inputs, &tensor_ids);
                    format!("{} - {}", in0, in1)
                }
                IROp::Mul => {
                    let in0 = format_input(node.inputs[0], &inputs, &tensor_ids);
                    let in1 = format_input(node.inputs[1], &inputs, &tensor_ids);
                    format!("{} * {}", in0, in1)
                }
                IROp::Div => {
                    let in0 = format_input(node.inputs[0], &inputs, &tensor_ids);
                    let in1 = format_input(node.inputs[1], &inputs, &tensor_ids);
                    format!("{} / {}", in0, in1)
                }
                IROp::Neg => {
                    let in0 = format_input(node.inputs[0], &inputs, &tensor_ids);
                    format!("-{}", in0)
                }
                IROp::ReLU => {
                    let in0 = format_input(node.inputs[0], &inputs, &tensor_ids);
                    format!("{} > 0.0 ? {} : 0.0", in0, in0)
                }
                IROp::GeLU => {
                    let in0 = format_input(node.inputs[0], &inputs, &tensor_ids);
                    format!("{} * 0.5 * (1.0 + erf({} / 1.41421356))", in0, in0)
                }
                IROp::Sigmoid => {
                    let in0 = format_input(node.inputs[0], &inputs, &tensor_ids);
                    format!("1.0 / (1.0 + exp(-{}))", in0)
                }
                IROp::Tanh => {
                    let in0 = format_input(node.inputs[0], &inputs, &tensor_ids);
                    format!("tanh({})", in0)
                }
                _ => "0.0".to_string(),
            };

            // Write output assignment
            if node.id == output {
                code.push_str(&format!("        v{}[idx] = {};\n", output, expr));
            } else {
                code.push_str(&format!("        double v{} = {};\n", node.id, expr));
            }
        }

        code.push_str("    }\n");
        code.push_str("}\n");

        kernels.push(CudaKernel {
            name: kernel_name,
            code,
            inputs,
            input_is_tensor,
            output,
            output_shape,
        });
    }

    kernels
}

/// Helper to format input operand: dereferences if array, uses raw variable name if scalar
fn format_input(id: usize, group_inputs: &[usize], tensor_ids: &HashSet<usize>) -> String {
    let is_tensor = tensor_ids.contains(&id);
    
    if group_inputs.contains(&id) {
        if is_tensor {
            format!("v{}[idx]", id)
        } else {
            format!("v{}", id)
        }
    } else {
        // Produced locally within the fused block
        format!("v{}", id)
    }
}
