/// NEURON AST → IR lowering pass.
///
/// Translates the typed AST into NEURON IR, resolving grad(), update,
/// function calls, and all expression nodes into IR operations.

use crate::ast::*;
use crate::ir::*;

pub struct Lowerer {
    program: IRProgram,
    next_id: ValueId,
    /// Maps variable names to their current ValueId
    env: Vec<std::collections::HashMap<String, ValueId>>,
    current_blocks: Vec<BasicBlock>,
    current_block_id: Option<BlockId>,
}

impl Lowerer {
    pub fn new() -> Self {
        Self {
            program: IRProgram::new(),
            next_id: 0,
            env: vec![std::collections::HashMap::new()],
            current_blocks: Vec::new(),
            current_block_id: None,
        }
    }

    fn fresh_id(&mut self) -> ValueId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn new_block(&mut self) -> BlockId {
        let id = self.current_blocks.len();
        self.current_blocks.push(BasicBlock {
            id,
            instructions: Vec::new(),
            terminator: Terminator::Return(None),
        });
        id
    }

    fn terminate(&mut self, term: Terminator) {
        if let Some(bid) = self.current_block_id {
            self.current_blocks[bid].terminator = term;
        }
        self.current_block_id = None;
    }

    fn push_scope(&mut self) {
        self.env.push(std::collections::HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.env.pop();
    }

    fn define(&mut self, name: &str, id: ValueId) {
        if let Some(scope) = self.env.last_mut() {
            scope.insert(name.to_string(), id);
        }
    }

    fn lookup(&self, name: &str) -> Option<ValueId> {
        for scope in self.env.iter().rev() {
            if let Some(&id) = scope.get(name) { return Some(id); }
        }
        None
    }

    fn emit(&mut self, _func: &mut IRFunction, op: IROp, inputs: Vec<ValueId>, output_type: IRType) -> ValueId {
        let id = self.fresh_id();
        let grad_fn = match &op {
            IROp::Add | IROp::Sub | IROp::Mul | IROp::Div | IROp::MatMul
            | IROp::ReLU | IROp::GeLU | IROp::Sigmoid | IROp::Tanh
            | IROp::Softmax { .. } | IROp::CrossEntropy | IROp::MSELoss => Some(GradFn::Builtin),
            _ => None,
        };
        let node = IRNode {
            id,
            op,
            inputs,
            output_type,
            output_shape: vec![],
            grad_fn,
            device: DeviceTarget::Auto,
            temporal_dir: None,
            effects: vec![],
        };
        if let Some(bid) = self.current_block_id {
            self.current_blocks[bid].instructions.push(node);
        }
        id
    }

    pub fn lower(mut self, program: &Program) -> IRProgram {
        // Collect all top-level statements into a __main__ function
        let mut main_fn = IRFunction::new("main");
        let mut has_main_stmts = false;

        self.current_blocks = Vec::new();
        let entry = self.new_block();
        self.current_block_id = Some(entry);

        for tl in &program.top_levels {
            match tl {
                TopLevel::Let(_) | TopLevel::Expr(_) | TopLevel::Update(_) => {
                    has_main_stmts = true;
                }
                _ => {}
            }
            self.lower_top_level_into(tl, &mut main_fn);
        }

        if has_main_stmts {
            if self.current_block_id.is_some() {
                self.terminate(Terminator::Return(None));
            }
            main_fn.blocks = std::mem::take(&mut self.current_blocks);
            main_fn.entry = entry;
            self.program.functions.push(main_fn);
        }

        self.program
    }

    fn lower_top_level_into(&mut self, tl: &TopLevel, main_fn: &mut IRFunction) {
        match tl {
            TopLevel::Fn(f) => {
                let ir_fn = self.lower_fn_decl(f);
                self.program.functions.push(ir_fn);
            }
            TopLevel::Model(m) => {
                // Lower each method as a function
                for method in &m.methods {
                    let mut ir_fn = self.lower_fn_decl(method);
                    ir_fn.name = format!("{}_{}", m.name, method.name);
                    self.program.functions.push(ir_fn);
                }
                
                let saved_blocks = std::mem::take(&mut self.current_blocks);
                let saved_block_id = self.current_block_id;
                let saved_env = std::mem::replace(&mut self.env, vec![std::collections::HashMap::new()]);

                self.current_blocks = Vec::new();
                let entry = self.new_block();
                self.current_block_id = Some(entry);

                // Create constructor function
                let mut ctor = IRFunction::new(format!("{}_new", m.name));
                for p in &m.params {
                    let id = self.fresh_id();
                    ctor.params.push(IRParam {
                        name: p.name.clone(),
                        ty: IRType::Any,
                        id,
                    });
                }
                // Save parameters as fields
                for p in &m.params {
                    let param_id = ctor.params.iter().find(|param| param.name == p.name).map(|param| param.id);
                    if let Some(pid) = param_id {
                        self.emit(&mut ctor, IROp::Store { name: format!("self.{}", p.name) }, vec![pid], IRType::Void);
                    }
                }
                // Init fields
                for field in &m.fields {
                    if let Some(ref default) = field.default {
                        let val_id = self.lower_expr(&mut ctor, default);
                        self.emit(&mut ctor, IROp::Store { name: format!("self.{}", field.name) }, vec![val_id], IRType::Void);
                    }
                }
                
                self.terminate(Terminator::Return(None));
                ctor.blocks = std::mem::take(&mut self.current_blocks);
                ctor.entry = entry;
                ctor.return_type = IRType::Any;
                self.program.functions.push(ctor);

                self.current_blocks = saved_blocks;
                self.current_block_id = saved_block_id;
                self.env = saved_env;
            }
            TopLevel::Layer(l) => {
                for method in &l.methods {
                    let mut ir_fn = self.lower_fn_decl(method);
                    ir_fn.name = format!("{}_{}", l.name, method.name);
                    self.program.functions.push(ir_fn);
                }
                
                let saved_blocks = std::mem::take(&mut self.current_blocks);
                let saved_block_id = self.current_block_id;
                let saved_env = std::mem::replace(&mut self.env, vec![std::collections::HashMap::new()]);

                self.current_blocks = Vec::new();
                let entry = self.new_block();
                self.current_block_id = Some(entry);

                // Create constructor function
                let mut ctor = IRFunction::new(format!("{}_new", l.name));
                for p in &l.params {
                    let id = self.fresh_id();
                    ctor.params.push(IRParam {
                        name: p.name.clone(),
                        ty: IRType::Any,
                        id,
                    });
                }
                // Save parameters as fields
                for p in &l.params {
                    let param_id = ctor.params.iter().find(|param| param.name == p.name).map(|param| param.id);
                    if let Some(pid) = param_id {
                        self.emit(&mut ctor, IROp::Store { name: format!("self.{}", p.name) }, vec![pid], IRType::Void);
                    }
                }
                // Init fields
                for field in &l.fields {
                    if let Some(ref default) = field.default {
                        let val_id = self.lower_expr(&mut ctor, default);
                        self.emit(&mut ctor, IROp::Store { name: format!("self.{}", field.name) }, vec![val_id], IRType::Void);
                    }
                }
                
                self.terminate(Terminator::Return(None));
                ctor.blocks = std::mem::take(&mut self.current_blocks);
                ctor.entry = entry;
                ctor.return_type = IRType::Any;
                self.program.functions.push(ctor);

                self.current_blocks = saved_blocks;
                self.current_block_id = saved_block_id;
                self.env = saved_env;
            }
            TopLevel::Agent(a) => {
                for method in &a.methods {
                    let mut ir_fn = self.lower_fn_decl(method);
                    ir_fn.name = format!("{}_{}", a.name, method.name);
                    self.program.functions.push(ir_fn);
                }
                
                let saved_blocks = std::mem::take(&mut self.current_blocks);
                let saved_block_id = self.current_block_id;
                let saved_env = std::mem::replace(&mut self.env, vec![std::collections::HashMap::new()]);

                self.current_blocks = Vec::new();
                let entry = self.new_block();
                self.current_block_id = Some(entry);

                // Create constructor function
                let mut ctor = IRFunction::new(format!("{}_new", a.name));
                for p in &a.params {
                    let id = self.fresh_id();
                    ctor.params.push(IRParam {
                        name: p.name.clone(),
                        ty: IRType::Any,
                        id,
                    });
                }
                // Save parameters as fields
                for p in &a.params {
                    let param_id = ctor.params.iter().find(|param| param.name == p.name).map(|param| param.id);
                    if let Some(pid) = param_id {
                        self.emit(&mut ctor, IROp::Store { name: format!("self.{}", p.name) }, vec![pid], IRType::Void);
                    }
                }
                // Init fields
                for field in &a.fields {
                    if let Some(ref default) = field.default {
                        let val_id = self.lower_expr(&mut ctor, default);
                        self.emit(&mut ctor, IROp::Store { name: format!("self.{}", field.name) }, vec![val_id], IRType::Void);
                    }
                }
                
                self.terminate(Terminator::Return(None));
                ctor.blocks = std::mem::take(&mut self.current_blocks);
                ctor.entry = entry;
                ctor.return_type = IRType::Any;
                self.program.functions.push(ctor);

                self.current_blocks = saved_blocks;
                self.current_block_id = saved_block_id;
                self.env = saved_env;
            }
            TopLevel::Let(l) => {
                // Lower into main function
                let val_id = self.lower_expr(main_fn, &l.value);
                self.define(&l.name, val_id);
                self.emit(main_fn, IROp::Store { name: l.name.clone() }, vec![val_id], IRType::Void);
            }
            TopLevel::Expr(e) => {
                // Lower expression into main function
                self.lower_expr(main_fn, &e.expr);
            }
            TopLevel::Update(u) => {
                let (opt_op, loss_id) = self.extract_optimizer(&u.expr, main_fn, u.target.clone());
                if let Some(lid) = loss_id {
                    self.emit(main_fn, IROp::Backward, vec![lid], IRType::Void);
                }
                self.emit(main_fn, opt_op, vec![], IRType::Void);
            }
            TopLevel::Causal(c) => {
                self.program.globals.push(IRGlobal {
                    name: c.name.clone(),
                    value: IRConst::String(format!("causal_model:{}", c.name)),
                    ty: IRType::Any,
                });
            }
            _ => {}
        }
    }

    fn lower_fn_decl(&mut self, f: &FnDecl) -> IRFunction {
        let mut ir_fn = IRFunction::new(&f.name);

        // Save the outer scope stack and start fresh for this function.
        // This prevents the function body from resolving global variable names
        // to SSA IDs that belong to a different function's IR. Instead,
        // unresolved names will emit Load nodes resolved at runtime from vm.globals.
        let saved_env = std::mem::replace(&mut self.env, vec![std::collections::HashMap::new()]);

        for p in &f.params {
            let id = self.fresh_id();
            ir_fn.params.push(IRParam {
                name: p.name.clone(),
                ty: p.type_ann.as_ref().map(|t| self.lower_type(t)).unwrap_or(IRType::Any),
                id,
            });
            self.define(&p.name, id);
        }

        ir_fn.return_type = f.return_type.as_ref().map(|t| self.lower_type(t)).unwrap_or(IRType::Void);

        // Check annotations for @opaque
        ir_fn.is_differentiable = !f.annotations.iter().any(|a| a.name == "opaque");

        let saved_blocks = std::mem::take(&mut self.current_blocks);
        let saved_block_id = self.current_block_id;

        self.current_blocks = Vec::new();
        let entry = self.new_block();
        self.current_block_id = Some(entry);

        // Lower body
        for stmt in &f.body {
            self.lower_stmt(&mut ir_fn, stmt);
        }

        if self.current_block_id.is_some() {
            self.terminate(Terminator::Return(None));
        }

        ir_fn.blocks = std::mem::take(&mut self.current_blocks);
        ir_fn.entry = entry;

        self.current_blocks = saved_blocks;
        self.current_block_id = saved_block_id;

        // Restore the outer scope stack
        self.env = saved_env;
        ir_fn
    }

    fn lower_stmt(&mut self, func: &mut IRFunction, stmt: &Stmt) {
        match stmt {
            Stmt::Let(l) => {
                let val_id = self.lower_expr(func, &l.value);
                self.define(&l.name, val_id);
                self.emit(func, IROp::Store { name: l.name.clone() }, vec![val_id], IRType::Void);
            }
            Stmt::For(f) => {
                let iter_id = self.lower_expr(func, &f.iter_expr);
                
                let cond_block = self.new_block();
                let body_block = self.new_block();
                let incr_block = self.new_block();
                let exit_block = self.new_block();
                
                let loop_id = self.fresh_id();
                let list_var = format!("$list_{}", loop_id);
                let idx_var = format!("$idx_{}", loop_id);
                
                // Store iter list
                self.emit(func, IROp::Store { name: list_var.clone() }, vec![iter_id], IRType::Void);
                // Store initial index (0)
                let zero_id = self.emit(func, IROp::Const(IRConst::Int(0)), vec![], IRType::I64);
                self.emit(func, IROp::Store { name: idx_var.clone() }, vec![zero_id], IRType::Void);
                
                self.terminate(Terminator::Jump(cond_block));
                
                // Condition check:
                self.current_block_id = Some(cond_block);
                let current_idx = self.emit(func, IROp::Load { name: idx_var.clone() }, vec![], IRType::I64);
                let current_list = self.emit(func, IROp::Load { name: list_var.clone() }, vec![], IRType::Any);
                let list_len = self.emit(func, IROp::ListLen, vec![current_list], IRType::I64);
                let is_less = self.emit(func, IROp::Lt, vec![current_idx, list_len], IRType::Bool);
                
                self.terminate(Terminator::Branch {
                    cond: is_less,
                    true_block: body_block,
                    false_block: exit_block,
                });
                
                // Snapshot env before body to detect modifications
                let env_snapshot: Vec<std::collections::HashMap<String, ValueId>> = self.env.iter().map(|s| s.clone()).collect();
                
                // Body:
                self.current_block_id = Some(body_block);
                let elem_id = self.emit(func, IROp::Index, vec![current_list, current_idx], IRType::Any);
                
                self.push_scope();
                self.define(&f.var, elem_id);
                self.emit(func, IROp::Store { name: f.var.clone() }, vec![elem_id], IRType::Void);
                
                for s in &f.body {
                    self.lower_stmt(func, s);
                }
                
                self.pop_scope();
                
                if self.current_block_id.is_some() {
                    self.terminate(Terminator::Jump(incr_block));
                }
                
                // Collect names modified inside the loop body
                let body_modified: std::collections::HashSet<String> = self.env.iter()
                    .zip(env_snapshot.iter())
                    .flat_map(|(current, snapshot)| {
                        current.keys()
                            .filter(|k| !snapshot.contains_key(*k) || snapshot.get(*k) != current.get(*k))
                            .cloned()
                    })
                    .collect();
                
                // Increment:
                self.current_block_id = Some(incr_block);
                let current_idx_for_incr = self.emit(func, IROp::Load { name: idx_var.clone() }, vec![], IRType::I64);
                let one_id = self.emit(func, IROp::Const(IRConst::Int(1)), vec![], IRType::I64);
                let next_idx = self.emit(func, IROp::Add, vec![current_idx_for_incr, one_id], IRType::I64);
                self.emit(func, IROp::Store { name: idx_var.clone() }, vec![next_idx], IRType::Void);
                
                self.terminate(Terminator::Jump(cond_block));
                
                // Exit: invalidate variables modified inside the loop
                // so subsequent references use runtime Load
                self.env = env_snapshot;
                for name in &body_modified {
                    for scope in self.env.iter_mut() {
                        scope.remove(name);
                    }
                }
                self.current_block_id = Some(exit_block);
            }
            Stmt::If(i) => {
                let cond_id = self.lower_expr(func, &i.cond);
                let then_block = self.new_block();
                let else_block = self.new_block();
                let merge_block = self.new_block();
                
                self.terminate(Terminator::Branch {
                    cond: cond_id,
                    true_block: then_block,
                    false_block: else_block,
                });
                
                // Snapshot env before branches so we can invalidate branch-local bindings
                let env_snapshot: Vec<std::collections::HashMap<String, ValueId>> = self.env.iter().map(|s| s.clone()).collect();
                
                // Lower then branch
                self.current_block_id = Some(then_block);
                for s in &i.then_body {
                    self.lower_stmt(func, s);
                }
                if self.current_block_id.is_some() {
                    self.terminate(Terminator::Jump(merge_block));
                }
                
                // Collect names defined in then branch
                let then_defined: std::collections::HashSet<String> = self.env.iter()
                    .zip(env_snapshot.iter())
                    .flat_map(|(current, snapshot)| {
                        current.keys()
                            .filter(|k| !snapshot.contains_key(*k) || snapshot.get(*k) != current.get(*k))
                            .cloned()
                    })
                    .collect();
                
                // Restore env before lowering else branch
                self.env = env_snapshot.clone();
                
                // Lower else branch
                self.current_block_id = Some(else_block);
                for s in &i.else_body {
                    self.lower_stmt(func, s);
                }
                if self.current_block_id.is_some() {
                    self.terminate(Terminator::Jump(merge_block));
                }
                
                // Collect names defined/modified in else branch
                let else_defined: std::collections::HashSet<String> = self.env.iter()
                    .zip(env_snapshot.iter())
                    .flat_map(|(current, snapshot)| {
                        current.keys()
                            .filter(|k| !snapshot.contains_key(*k) || snapshot.get(*k) != current.get(*k))
                            .cloned()
                    })
                    .collect();
                
                // Restore env to pre-branch state, then invalidate any names
                // that were modified in either branch so that subsequent references
                // use runtime Load { name } instead of stale SSA IDs
                self.env = env_snapshot;
                let modified_names: std::collections::HashSet<String> = then_defined.union(&else_defined).cloned().collect();
                for name in &modified_names {
                    for scope in self.env.iter_mut() {
                        scope.remove(name);
                    }
                }
                
                self.current_block_id = Some(merge_block);
            }
            Stmt::Return(r) => {
                let val_id = self.lower_expr(func, &r.value);
                self.terminate(Terminator::Return(Some(val_id)));
            }
            Stmt::Update(u) => {
                // Extract optimizer from expression
                let (opt_op, loss_id) = self.extract_optimizer(&u.expr, func, u.target.clone());
                if let Some(lid) = loss_id {
                    self.emit(func, IROp::Backward, vec![lid], IRType::Void);
                }
                self.emit(func, opt_op, vec![], IRType::Void);
            }
            Stmt::Expr(e) => { self.lower_expr(func, &e.expr); }
            Stmt::Constraint(c) => {
                let expr_id = self.lower_expr(func, &c.expr);
                self.emit(func, IROp::EffectCheck { expected: vec!["constraint".into()] }, vec![expr_id], IRType::Void);
            }
        }
    }

    fn lower_expr(&mut self, func: &mut IRFunction, expr: &Expr) -> ValueId {
        match expr {
            Expr::IntLit(v, _) => self.emit(func, IROp::Const(IRConst::Int(*v)), vec![], IRType::I64),
            Expr::FloatLit(v, _) => self.emit(func, IROp::Const(IRConst::Float(*v)), vec![], IRType::F64),
            Expr::BoolLit(v, _) => self.emit(func, IROp::Const(IRConst::Bool(*v)), vec![], IRType::Bool),
            Expr::StringLit(s, _) => self.emit(func, IROp::Const(IRConst::String(s.clone())), vec![], IRType::String),
            Expr::Ident(name, _) => {
                self.lookup(name).unwrap_or_else(|| {
                    self.emit(func, IROp::Load { name: name.clone() }, vec![], IRType::Any)
                })
            }
            Expr::Self_(_) => {
                self.lookup("self").unwrap_or_else(|| {
                    self.emit(func, IROp::Load { name: "self".into() }, vec![], IRType::Any)
                })
            }
            Expr::BinOp(b) => {
                let left = self.lower_expr(func, &b.left);
                let right = self.lower_expr(func, &b.right);
                let op = match b.op {
                    BinOp::Add => IROp::Add,
                    BinOp::Sub => IROp::Sub,
                    BinOp::Mul => IROp::Mul,
                    BinOp::Div => IROp::Div,
                    BinOp::MatMul => IROp::MatMul,
                    BinOp::Lt => IROp::Lt,
                    BinOp::Lte => IROp::Lte,
                    BinOp::Gt => IROp::Gt,
                    BinOp::Gte => IROp::Gte,
                    BinOp::Eq => IROp::Eq,
                    BinOp::Neq => IROp::Neq,
                    _ => IROp::Add,
                };
                self.emit(func, op, vec![left, right], IRType::Any)
            }
            Expr::UnaryOp(u) => {
                let operand = self.lower_expr(func, &u.operand);
                match u.op {
                    UnaryOp::Neg => self.emit(func, IROp::Neg, vec![operand], IRType::Any),
                    UnaryOp::Not => self.emit(func, IROp::Neg, vec![operand], IRType::Bool), // Simplified
                }
            }
            Expr::FnCall(c) => {
                if let Expr::Dot(d) = &c.callee {
                    if d.field == "before" || d.field == "after" || d.field == "snapshot" {
                        let receiver_id = self.lower_expr(func, &d.obj);
                        let t_id = if !c.args.is_empty() {
                            self.lower_expr(func, &c.args[0].value)
                        } else {
                            0
                        };
                        let op = match d.field.as_str() {
                            "before" => IROp::TemporalBefore { t: t_id },
                            "after" => IROp::TemporalAfter { t: t_id },
                            "snapshot" => IROp::TemporalSnapshot { at: t_id },
                            _ => unreachable!(),
                        };
                        let out_ty = match d.field.as_str() {
                            "snapshot" => IRType::Any,
                            _ => IRType::Temporal(Box::new(IRType::Any), "past_to_future".into()),
                        };
                        return self.emit(func, op, vec![receiver_id], out_ty);
                    }
                }

                let mut arg_ids: Vec<ValueId> = Vec::new();
                for arg in &c.args {
                    arg_ids.push(self.lower_expr(func, &arg.value));
                }
                // Check for built-in functions
                if let Expr::Ident(name, _) = &c.callee {
                    match name.as_str() {
                        "zeros" => {
                            let mut shape_ids = Vec::new();
                            for arg in &c.args {
                                shape_ids.push(self.lower_expr(func, &arg.value));
                            }
                            return self.emit(func, IROp::Zeros(vec![]), shape_ids, IRType::Tensor(vec![]));
                        }
                        "glorot" => {
                            let mut shape_ids = Vec::new();
                            for arg in &c.args {
                                shape_ids.push(self.lower_expr(func, &arg.value));
                            }
                            return self.emit(func, IROp::Glorot(vec![]), shape_ids, IRType::Tensor(vec![]));
                        }
                        "relu" => return self.emit(func, IROp::ReLU, arg_ids, IRType::Tensor(vec![])),
                        "gelu" => return self.emit(func, IROp::GeLU, arg_ids, IRType::Tensor(vec![])),
                        "sqrt" => return self.emit(func, IROp::Sqrt, arg_ids, IRType::Tensor(vec![])),
                        "softmax" => return self.emit(func, IROp::Softmax { dim: -1 }, arg_ids, IRType::Tensor(vec![])),
                        "sigmoid" => return self.emit(func, IROp::Sigmoid, arg_ids, IRType::Tensor(vec![])),
                        "tanh" => return self.emit(func, IROp::Tanh, arg_ids, IRType::Tensor(vec![])),
                        "cross_entropy" => return self.emit(func, IROp::CrossEntropy, arg_ids, IRType::F64),
                        "mse" => return self.emit(func, IROp::MSELoss, arg_ids, IRType::F64),
                        "concat" => return self.emit(func, IROp::Concat { dim: -1 }, arg_ids, IRType::Tensor(vec![])),
                        "transpose" => {
                            let dim0 = c.args.get(1).and_then(|a| {
                                if let Expr::IntLit(v, _) = &a.value { Some(*v as usize) } else { None }
                            }).unwrap_or(0);
                            let dim1 = c.args.get(2).and_then(|a| {
                                if let Expr::IntLit(v, _) = &a.value { Some(*v as usize) } else { None }
                            }).unwrap_or(1);
                            return self.emit(func, IROp::Transpose(dim0, dim1), vec![arg_ids[0]], IRType::Tensor(vec![]));
                        }
                        "update_row" => return self.emit(func, IROp::UpdateRow, arg_ids, IRType::Tensor(vec![])),
                        "print" => return self.emit(func, IROp::Print, arg_ids, IRType::Void),
                        "UNCERTAIN" | "Normal" | "Beta" | "GaussianNoise" => {
                            return self.emit(func, IROp::UncertainWrap, arg_ids, IRType::Uncertain(Box::new(IRType::F64)));
                        }
                        "recall" => {
                            let k = c.args.get(2).and_then(|a| {
                                if let Expr::IntLit(v, _) = &a.value { Some(*v) } else { None }
                            }).unwrap_or(5);
                            return self.emit(func, IROp::MemoryRecall { k }, arg_ids, IRType::List(Box::new(IRType::Any)));
                        }
                        "store" => return self.emit(func, IROp::MemoryStore, arg_ids, IRType::Void),
                        "input" => return self.emit(func, IROp::Input, arg_ids, IRType::String),
                        "embed_string" => return self.emit(func, IROp::EmbedString, arg_ids, IRType::Tensor(vec![1, 8])),
                        "generate_reply" => return self.emit(func, IROp::GenerateReply, arg_ids, IRType::String),
                        "search" => return self.emit(func, IROp::Search { strategy: "default".into(), max_iter: 100 }, arg_ids, IRType::Any),
                        _ => {}
                    }
                }
                let callee_name = match &c.callee {
                    Expr::Ident(n, _) => n.clone(),
                    Expr::Dot(d) => {
                        let receiver_id = self.lower_expr(func, &d.obj);
                        match d.field.as_str() {
                            "sum" => {
                                let mut dim = None;
                                if !c.args.is_empty() {
                                    if let Expr::IntLit(val, _) = &c.args[0].value {
                                        dim = Some(*val);
                                    }
                                }
                                return self.emit(func, IROp::Sum { dim }, vec![receiver_id], IRType::Tensor(vec![]));
                            }
                            "mean" => {
                                let mut dim = None;
                                if !c.args.is_empty() {
                                    if let Expr::IntLit(val, _) = &c.args[0].value {
                                        dim = Some(*val);
                                    }
                                }
                                return self.emit(func, IROp::Mean { dim }, vec![receiver_id], IRType::Tensor(vec![]));
                            }
                            _ => {
                                arg_ids.insert(0, receiver_id);
                                format!("{}_{}", "obj", d.field)
                            }
                        }
                    }
                    _ => "__call__".to_string(),
                };
                self.emit(func, IROp::Call { function: callee_name }, arg_ids, IRType::Any)
            }
            Expr::Dot(d) => {
                let obj = self.lower_expr(func, &d.obj);
                self.emit(func, IROp::Load { name: d.field.clone() }, vec![obj], IRType::Any)
            }
            Expr::Index(idx) => {
                let obj = self.lower_expr(func, &idx.obj);
                // Lower index items
                let mut index_ids = vec![obj];
                for item in &idx.indices {
                    match item {
                        IndexItem::Expr(e) => { index_ids.push(self.lower_expr(func, e)); }
                        _ => {} // Slices handled as special IR
                    }
                }
                self.emit(func, IROp::Index, index_ids, IRType::Any)
            }
            Expr::Grad(g) => {
                let expr_id = self.lower_expr(func, &g.expr);
                self.emit(func, IROp::Grad { wrt: g.wrt.clone() }, vec![expr_id], IRType::Tensor(vec![]))
            }
            Expr::StopGrad(expr, _) => {
                let expr_id = self.lower_expr(func, expr);
                self.emit(func, IROp::StopGrad, vec![expr_id], IRType::Any)
            }
            Expr::Do(d) => {
                let mut ids = Vec::new();
                for (_, val) in &d.assignments {
                    ids.push(self.lower_expr(func, val));
                }
                self.emit(func, IROp::Intervene, ids, IRType::Causal(Box::new(IRType::Any), "intervened".into()))
            }
            Expr::Observe(o) => {
                let obj = self.lower_expr(func, &o.obj);
                self.emit(func, IROp::Observe, vec![obj], IRType::Causal(Box::new(IRType::Any), "observed".into()))
            }
            Expr::Explain(e) => {
                let expr_id = self.lower_expr(func, &e.expr);
                self.emit(func, IROp::Explain, vec![expr_id], IRType::Tuple(vec![IRType::Any, IRType::Any]))
            }
            Expr::Merge(m) => {
                let left = self.lower_expr(func, &m.left);
                let right = self.lower_expr(func, &m.right);
                let strat = m.strategy.as_ref().map(|_| "TIES").unwrap_or("TaskVector");
                self.emit(func, IROp::MergeModels { strategy: strat.to_string() }, vec![left, right], IRType::Any)
            }
            Expr::Forget(f) => {
                let obj = self.lower_expr(func, &f.obj);
                let mut inputs = vec![obj];
                let mut method = "TaskNegation".to_string();
                let mut strength = 1.0;

                if !f.args.is_empty() {
                    // Positional first argument is task_data
                    let td_id = self.lower_expr(func, &f.args[0].value);
                    inputs.push(td_id);

                    // Look at remaining positional arguments
                    if f.args.len() > 1 {
                        if let Expr::StringLit(val, _) = &f.args[1].value {
                            method = val.clone();
                        }
                    }
                    if f.args.len() > 2 {
                        if let Expr::FloatLit(val, _) = &f.args[2].value {
                            strength = *val;
                        } else if let Expr::IntLit(val, _) = &f.args[2].value {
                            strength = *val as f64;
                        }
                    }

                    // Named arguments override positional
                    for arg in &f.args {
                        if let Some(ref name) = arg.name {
                            match name.as_str() {
                                "method" => {
                                    if let Expr::StringLit(val, _) = &arg.value {
                                        method = val.clone();
                                    }
                                }
                                "strength" => {
                                    if let Expr::FloatLit(val, _) = &arg.value {
                                        strength = *val;
                                    } else if let Expr::IntLit(val, _) = &arg.value {
                                        strength = *val as f64;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }

                self.emit(func, IROp::ForgetTask { method, strength }, inputs, IRType::Any)
            }
            Expr::List(elems, _) => {
                let ids: Vec<ValueId> = elems.iter().map(|e| self.lower_expr(func, e)).collect();
                self.emit(func, IROp::Nop, ids, IRType::List(Box::new(IRType::Any)))
            }
            Expr::ListComp(lc) => {
                let iter = self.lower_expr(func, &lc.iter);
                self.push_scope();
                let var_id = self.fresh_id();
                self.define(&lc.var, var_id);
                let body_id = self.lower_expr(func, &lc.expr);
                self.pop_scope();
                self.emit(func, IROp::Loop { var: lc.var.clone(), iter, body_block: 0 }, vec![body_id], IRType::List(Box::new(IRType::Any)))
            }
            Expr::Tuple(elems, _) => {
                let ids: Vec<ValueId> = elems.iter().map(|e| self.lower_expr(func, e)).collect();
                let types: Vec<IRType> = ids.iter().map(|_| IRType::Any).collect();
                self.emit(func, IROp::Nop, ids, IRType::Tuple(types))
            }
            Expr::SearchExpr(s) => {
                let space = self.lower_expr(func, &s.space);
                let eval = self.lower_expr(func, &s.evaluate);
                self.emit(func, IROp::Search { strategy: "MCTS".into(), max_iter: 1000 }, vec![space, eval], IRType::Any)
            }
            Expr::RecallExpr(r) => {
                let mem = self.lower_expr(func, &r.memory);
                let query = self.lower_expr(func, &r.query);
                let k = r.k.as_ref().map(|e| {
                    if let Expr::IntLit(v, _) = e { *v } else { 10 }
                }).unwrap_or(10);
                self.emit(func, IROp::MemoryRecall { k }, vec![mem, query], IRType::List(Box::new(IRType::Any)))
            }
            Expr::StoreExpr(s) => {
                let mem = self.lower_expr(func, &s.memory);
                let item = self.lower_expr(func, &s.item);
                self.emit(func, IROp::MemoryStore, vec![mem, item], IRType::Void)
            }
        }
    }

    #[allow(dead_code)]
    fn extract_shape_args(&self, args: &[CallArg]) -> Vec<i64> {
        args.iter().filter_map(|a| {
            if let Expr::IntLit(v, _) = &a.value { Some(*v) } else { None }
        }).collect()
    }

    fn extract_optimizer(&mut self, expr: &Expr, func: &mut IRFunction, target: String) -> (IROp, Option<ValueId>) {
        if let Expr::FnCall(c) = expr {
            let name = match &c.callee {
                Expr::Ident(n, _) => n.as_str(),
                _ => "adam",
            };
            let lr = c.args.iter().find(|a| a.name.as_deref() == Some("lr")).and_then(|a| {
                if let Expr::FloatLit(v, _) = &a.value { Some(*v) } else { None }
            }).unwrap_or(1e-3);
            // Find grad(loss) in args
            let loss_id = c.args.iter().find_map(|a| {
                if let Expr::Grad(g) = &a.value {
                    Some(self.lower_expr(func, &g.expr))
                } else { None }
            });
            let op = match name {
                "sgd" => IROp::SGD { target, lr, momentum: 0.0 },
                "adamw" => IROp::AdamW { target, lr, weight_decay: 0.01 },
                _ => IROp::Adam { target, lr, beta1: 0.9, beta2: 0.999 },
            };
            (op, loss_id)
        } else {
            (IROp::Adam { target, lr: 1e-3, beta1: 0.9, beta2: 0.999 }, None)
        }
    }

    fn lower_type(&self, te: &TypeExpr) -> IRType {
        match te {
            TypeExpr::Base(name, _) => match name.as_str() {
                "Int" => IRType::I64,
                "Float" => IRType::F64,
                "Bool" => IRType::Bool,
                "String" => IRType::String,
                _ => IRType::Any,
            },
            TypeExpr::Tensor(dims, _) => {
                let shape: Vec<i64> = dims.iter().map(|d| match d {
                    DimExpr::Static(v) => *v,
                    _ => -1,
                }).collect();
                IRType::Tensor(shape)
            }
            TypeExpr::Uncertain(inner, _) => IRType::Uncertain(Box::new(self.lower_type(inner))),
            TypeExpr::Random(inner, _) => IRType::Random(Box::new(self.lower_type(inner))),
            TypeExpr::Temporal(inner, dir, _) => IRType::Temporal(Box::new(self.lower_type(inner)), dir.clone()),
            TypeExpr::Causal(inner, mode, _) => IRType::Causal(Box::new(self.lower_type(inner)), mode.clone()),
            TypeExpr::ListType(inner, _) => IRType::List(Box::new(self.lower_type(inner))),
            _ => IRType::Any,
        }
    }
}
