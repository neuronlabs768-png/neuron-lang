use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use neuron_compiler::{compile_with_imports, check_with_imports};
use neuron_runtime::vm::{VM, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot};
use tower_http::cors::CorsLayer;

enum SaasCommand {
    Compile {
        source: String,
        filename: String,
        resp: oneshot::Sender<CompileResponse>,
    },
    Run {
        source: String,
        resp: oneshot::Sender<RunResponse>,
    },
    Deploy {
        source: String,
        agent_id: Option<String>,
        resp: oneshot::Sender<DeployResponse>,
    },
    Interact {
        agent_id: String,
        function_name: String,
        arguments: Vec<serde_json::Value>,
        resp: oneshot::Sender<InteractResponse>,
    },
    List {
        resp: oneshot::Sender<Vec<AgentInfo>>,
    },
    Memory {
        agent_id: String,
        resp: oneshot::Sender<Result<HashMap<String, serde_json::Value>, String>>,
    },
}

#[derive(Clone)]
struct AppState {
    tx: mpsc::Sender<SaasCommand>,
}

#[derive(Deserialize)]
struct CompileRequest {
    source: String,
    filename: Option<String>,
}

#[derive(Serialize)]
struct CompileResponse {
    success: bool,
    errors: Vec<String>,
    warnings: Vec<String>,
    functions: Vec<String>,
    globals: Vec<String>,
}

#[derive(Deserialize)]
struct RunRequest {
    source: String,
}

#[derive(Serialize)]
struct RunResponse {
    success: bool,
    error: Option<String>,
    stdout: Vec<String>,
    value: Option<serde_json::Value>,
    effects: Vec<String>,
    warnings: Vec<String>,
}

#[derive(Deserialize)]
struct DeployRequest {
    source: String,
    agent_id: Option<String>,
}

#[derive(Serialize)]
struct DeployResponse {
    success: bool,
    agent_id: Option<String>,
    errors: Vec<String>,
    functions: Vec<String>,
    globals: Vec<String>,
}

#[derive(Deserialize)]
struct InteractRequest {
    agent_id: String,
    function_name: String,
    arguments: Vec<serde_json::Value>,
}

#[derive(Serialize)]
struct InteractResponse {
    success: bool,
    error: Option<String>,
    result: Option<serde_json::Value>,
    stdout: Vec<String>,
    effects: Vec<String>,
}

#[derive(Serialize)]
struct AgentInfo {
    id: String,
    functions: Vec<String>,
    globals: Vec<String>,
}

#[derive(Serialize)]
struct MemoryResponse {
    success: bool,
    error: Option<String>,
    globals: HashMap<String, serde_json::Value>,
}

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel::<SaasCommand>(64);

    // Background worker thread: runs non-Send/non-Sync VM instances sequentially
    std::thread::spawn(move || {
        let mut agents: HashMap<String, VM> = HashMap::new();

        while let Some(cmd) = rx.blocking_recv() {
            match cmd {
                SaasCommand::Compile { source, filename, resp } => {
                    let result = check_with_imports(&source, &filename);
                    let success = !result.has_errors();
                    let errors = result.errors.iter().map(|e| e.to_string()).collect();
                    let warnings = result.warnings.iter().map(|w| w.to_string()).collect();

                    let mut functions = Vec::new();
                    let mut globals = Vec::new();

                    if success {
                        if let Ok(output) = compile_with_imports(&source, &filename) {
                            functions = output.ir.functions.iter().map(|f| f.name.clone()).collect();
                            globals = output.ir.globals.iter().map(|g| g.name.clone()).collect();
                        }
                    }

                    let _ = resp.send(CompileResponse {
                        success,
                        errors,
                        warnings,
                        functions,
                        globals,
                    });
                }

                SaasCommand::Run { source, resp } => {
                    let filename = "api_run.nr".to_string();
                    let compile_res = compile_with_imports(&source, &filename);

                    match compile_res {
                        Ok(output) => {
                            let warnings = output.result.warnings.iter().map(|w| w.to_string()).collect();
                            let mut vm = VM::new();
                            vm.load(&output.ir);

                            match vm.run_main() {
                                Ok(result_value) => {
                                    let _ = resp.send(RunResponse {
                                        success: true,
                                        error: None,
                                        stdout: vm.stdout_log.clone(),
                                        value: Some(value_to_json(&result_value)),
                                        effects: vm.effect_log.clone(),
                                        warnings,
                                    });
                                }
                                Err(vm_err) => {
                                    let _ = resp.send(RunResponse {
                                        success: false,
                                        error: Some(format!("Runtime error: {}", vm_err)),
                                        stdout: vm.stdout_log.clone(),
                                        value: None,
                                        effects: Vec::new(),
                                        warnings,
                                    });
                                }
                            }
                        }
                        Err(result) => {
                            let errors = result.errors.iter().map(|e| e.to_string()).collect();
                            let warnings = result.warnings.iter().map(|w| w.to_string()).collect();
                            let _ = resp.send(RunResponse {
                                success: false,
                                error: Some("Compilation failed".to_string()),
                                stdout: errors,
                                value: None,
                                effects: Vec::new(),
                                warnings,
                            });
                        }
                    }
                }

                SaasCommand::Deploy { source, agent_id, resp } => {
                    let id = agent_id.unwrap_or_else(|| format!("agent_{}", uuid_simple()));
                    let filename = format!("{}.nr", id);
                    let compile_res = compile_with_imports(&source, &filename);

                    match compile_res {
                        Ok(output) => {
                            let mut vm = VM::new();
                            vm.load(&output.ir);

                            // Run global initialization: the compiler names the top-level
                            // init function "main", not "__global_init__"
                            let init_fn = if vm.functions.contains_key("__global_init__") {
                                Some("__global_init__")
                            } else if vm.functions.contains_key("main") {
                                Some("main")
                            } else {
                                None
                            };
                            if let Some(init_name) = init_fn {
                                let _ = vm.execute(init_name, vec![]);
                            }

                            let functions = vm.functions.keys().cloned().collect();
                            let globals = vm.globals.keys().cloned().collect();

                            agents.insert(id.clone(), vm);

                            let _ = resp.send(DeployResponse {
                                success: true,
                                agent_id: Some(id),
                                errors: Vec::new(),
                                functions,
                                globals,
                            });
                        }
                        Err(result) => {
                            let errors = result.errors.iter().map(|e| e.to_string()).collect();
                            let _ = resp.send(DeployResponse {
                                success: false,
                                agent_id: None,
                                errors,
                                functions: Vec::new(),
                                globals: Vec::new(),
                            });
                        }
                    }
                }

                SaasCommand::Interact { agent_id, function_name, arguments, resp } => {
                    if let Some(vm) = agents.get_mut(&agent_id) {
                        vm.stdout_log.clear();
                        let args: Vec<Value> = arguments.iter().map(json_to_value).collect();

                        match vm.execute(&function_name, args) {
                            Ok(result) => {
                                let _ = resp.send(InteractResponse {
                                    success: true,
                                    error: None,
                                    result: Some(value_to_json(&result)),
                                    stdout: vm.stdout_log.clone(),
                                    effects: vm.effect_log.clone(),
                                });
                            }
                            Err(e) => {
                                let _ = resp.send(InteractResponse {
                                    success: false,
                                    error: Some(format!("Interaction failed: {}", e)),
                                    result: None,
                                    stdout: vm.stdout_log.clone(),
                                    effects: Vec::new(),
                                });
                            }
                        }
                    } else {
                        let _ = resp.send(InteractResponse {
                            success: false,
                            error: Some(format!("Agent '{}' not found", agent_id)),
                            result: None,
                            stdout: Vec::new(),
                            effects: Vec::new(),
                        });
                    }
                }

                SaasCommand::List { resp } => {
                    let list = agents.iter().map(|(id, vm)| {
                        AgentInfo {
                            id: id.clone(),
                            functions: vm.functions.keys().cloned().collect(),
                            globals: vm.globals.keys().cloned().collect(),
                        }
                    }).collect();
                    let _ = resp.send(list);
                }

                SaasCommand::Memory { agent_id, resp } => {
                    if let Some(vm) = agents.get(&agent_id) {
                        let mut map = HashMap::new();
                        for (k, v) in &vm.globals {
                            map.insert(k.clone(), value_to_json(v));
                        }
                        let _ = resp.send(Ok(map));
                    } else {
                        let _ = resp.send(Err(format!("Agent '{}' not found", agent_id)));
                    }
                }
            }
        }
    });

    let state = AppState { tx };

    let app = Router::new()
        .route("/api/compile", post(handle_compile))
        .route("/api/run", post(handle_run))
        .route("/api/agent/deploy", post(handle_deploy))
        .route("/api/agent/interact", post(handle_interact))
        .route("/api/agent/list", get(handle_list))
        .route("/api/agent/:id/memory", get(handle_memory))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("NEURON SaaS Backend listening on http://localhost:8080");
    axum::serve(listener, app).await.unwrap();
}

// ── JSON Helper Conversions ──

fn json_to_value(jv: &serde_json::Value) -> Value {
    match jv {
        serde_json::Value::Null => Value::None,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(num) => {
            if let Some(i) = num.as_i64() {
                Value::Int(i)
            } else if let Some(f) = num.as_f64() {
                Value::Float(f)
            } else {
                Value::Void
            }
        }
        serde_json::Value::String(s) => Value::Str(s.clone()),
        serde_json::Value::Array(arr) => {
            let vals: Vec<Value> = arr.iter().map(json_to_value).collect();
            Value::List(vals)
        }
        serde_json::Value::Object(obj) => {
            if let Some(t_val) = obj.get("type").and_then(|v| v.as_str()) {
                match t_val {
                    "Uncertain" => {
                        let value = obj.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let std = obj.get("std").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let confidence = obj.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        return Value::Uncertain { value, std, confidence };
                    }
                    "Random" => {
                        let mean = obj.get("mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let variance = obj.get("variance").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        return Value::Random { mean, variance };
                    }
                    "Tensor" => {
                        if let Some(data_arr) = obj.get("data").and_then(|v| v.as_array()) {
                            let data: Vec<f64> = data_arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
                            let shape: Vec<usize> = obj.get("shape")
                                .and_then(|v| v.as_array())
                                .map(|arr| arr.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
                                .unwrap_or_else(|| vec![data.len()]);
                            return Value::Tensor(neuron_runtime::tensor::Tensor::new(data, shape));
                        }
                    }
                    _ => {}
                }
            }
            let name = obj.get("name").and_then(|v| v.as_str()).unwrap_or("Model").to_string();
            Value::Model {
                name,
                fields: std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())),
            }
        }
    }
}

fn value_to_json(val: &Value) -> serde_json::Value {
    match val {
        Value::Void => serde_json::Value::Null,
        Value::None => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int(i) => serde_json::Value::Number((*i).into()),
        Value::Float(f) => {
            if let Some(num) = serde_json::Number::from_f64(*f) {
                serde_json::Value::Number(num)
            } else {
                serde_json::Value::Null
            }
        }
        Value::Str(s) => serde_json::Value::String(s.clone()),
        Value::List(arr) => {
            let j_arr: Vec<serde_json::Value> = arr.iter().map(value_to_json).collect();
            serde_json::Value::Array(j_arr)
        }
        Value::Tuple(arr) => {
            let j_arr: Vec<serde_json::Value> = arr.iter().map(value_to_json).collect();
            serde_json::Value::Array(j_arr)
        }
        Value::Tensor(t) => {
            t.data.prefetch_to_host();
            let mut map = serde_json::Map::new();
            map.insert("type".to_string(), serde_json::Value::String("Tensor".to_string()));
            map.insert("id".to_string(), serde_json::Value::Number(t.id.into()));
            
            let data_vals: Vec<serde_json::Value> = t.data.iter().map(|&v| {
                serde_json::Number::from_f64(v).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null)
            }).collect();
            map.insert("data".to_string(), serde_json::Value::Array(data_vals));
            
            let shape_vals: Vec<serde_json::Value> = t.shape.iter().map(|&v| {
                serde_json::Value::Number(v.into())
            }).collect();
            map.insert("shape".to_string(), serde_json::Value::Array(shape_vals));
            
            serde_json::Value::Object(map)
        }
        Value::Uncertain { value, std, confidence } => {
            let mut map = serde_json::Map::new();
            map.insert("type".to_string(), serde_json::Value::String("Uncertain".to_string()));
            map.insert("value".to_string(), serde_json::Number::from_f64(*value).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null));
            map.insert("std".to_string(), serde_json::Number::from_f64(*std).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null));
            map.insert("confidence".to_string(), serde_json::Number::from_f64(*confidence).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null));
            serde_json::Value::Object(map)
        }
        Value::Random { mean, variance } => {
            let mut map = serde_json::Map::new();
            map.insert("type".to_string(), serde_json::Value::String("Random".to_string()));
            map.insert("mean".to_string(), serde_json::Number::from_f64(*mean).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null));
            map.insert("variance".to_string(), serde_json::Number::from_f64(*variance).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null));
            serde_json::Value::Object(map)
        }
        Value::Temporal { data, direction } => {
            let mut map = serde_json::Map::new();
            map.insert("type".to_string(), serde_json::Value::String("Temporal".to_string()));
            map.insert("data".to_string(), value_to_json(data));
            map.insert("direction".to_string(), serde_json::Value::String(direction.clone()));
            serde_json::Value::Object(map)
        }
        Value::Causal { data, mode } => {
            let mut map = serde_json::Map::new();
            map.insert("type".to_string(), serde_json::Value::String("Causal".to_string()));
            map.insert("data".to_string(), value_to_json(data));
            map.insert("mode".to_string(), serde_json::Value::String(mode.clone()));
            serde_json::Value::Object(map)
        }
        Value::Model { name, fields } => {
            let mut map = serde_json::Map::new();
            map.insert("type".to_string(), serde_json::Value::String("Model".to_string()));
            map.insert("name".to_string(), serde_json::Value::String(name.clone()));
            
            let mut fields_map = serde_json::Map::new();
            for (k, v) in fields.borrow().iter() {
                fields_map.insert(k.clone(), value_to_json(v));
            }
            map.insert("fields".to_string(), serde_json::Value::Object(fields_map));
            serde_json::Value::Object(map)
        }
        Value::CausalModel { name, variables } => {
            let mut map = serde_json::Map::new();
            map.insert("type".to_string(), serde_json::Value::String("CausalModel".to_string()));
            map.insert("name".to_string(), serde_json::Value::String(name.clone()));
            map.insert("variables".to_string(), serde_json::Value::Array(variables.iter().map(|v| serde_json::Value::String(v.clone())).collect()));
            serde_json::Value::Object(map)
        }
    }
}

// ── Route Handlers ──

async fn handle_compile(
    State(state): State<AppState>,
    Json(payload): Json<CompileRequest>,
) -> impl IntoResponse {
    let filename = payload.filename.clone().unwrap_or_else(|| "api_compile.nr".to_string());
    let (resp_tx, resp_rx) = oneshot::channel();
    let _ = state.tx.send(SaasCommand::Compile {
        source: payload.source,
        filename,
        resp: resp_tx,
    }).await;

    match resp_rx.await {
        Ok(res) => Json(res),
        Err(_) => Json(CompileResponse {
            success: false,
            errors: vec!["Internal actor thread error".to_string()],
            warnings: Vec::new(),
            functions: Vec::new(),
            globals: Vec::new(),
        }),
    }
}

async fn handle_run(
    State(state): State<AppState>,
    Json(payload): Json<RunRequest>,
) -> impl IntoResponse {
    let (resp_tx, resp_rx) = oneshot::channel();
    let _ = state.tx.send(SaasCommand::Run {
        source: payload.source,
        resp: resp_tx,
    }).await;

    match resp_rx.await {
        Ok(res) => Json(res),
        Err(_) => Json(RunResponse {
            success: false,
            error: Some("Internal actor thread error".to_string()),
            stdout: Vec::new(),
            value: None,
            effects: Vec::new(),
            warnings: Vec::new(),
        }),
    }
}

async fn handle_deploy(
    State(state): State<AppState>,
    Json(payload): Json<DeployRequest>,
) -> impl IntoResponse {
    let (resp_tx, resp_rx) = oneshot::channel();
    let _ = state.tx.send(SaasCommand::Deploy {
        source: payload.source,
        agent_id: payload.agent_id,
        resp: resp_tx,
    }).await;

    match resp_rx.await {
        Ok(res) => Json(res),
        Err(_) => Json(DeployResponse {
            success: false,
            agent_id: None,
            errors: vec!["Internal actor thread error".to_string()],
            functions: Vec::new(),
            globals: Vec::new(),
        }),
    }
}

async fn handle_interact(
    State(state): State<AppState>,
    Json(payload): Json<InteractRequest>,
) -> impl IntoResponse {
    let (resp_tx, resp_rx) = oneshot::channel();
    let _ = state.tx.send(SaasCommand::Interact {
        agent_id: payload.agent_id,
        function_name: payload.function_name,
        arguments: payload.arguments,
        resp: resp_tx,
    }).await;

    match resp_rx.await {
        Ok(res) => Json(res),
        Err(_) => Json(InteractResponse {
            success: false,
            error: Some("Internal actor thread error".to_string()),
            result: None,
            stdout: Vec::new(),
            effects: Vec::new(),
        }),
    }
}

async fn handle_list(State(state): State<AppState>) -> impl IntoResponse {
    let (resp_tx, resp_rx) = oneshot::channel();
    let _ = state.tx.send(SaasCommand::List { resp: resp_tx }).await;

    match resp_rx.await {
        Ok(res) => Json(res),
        Err(_) => Json(Vec::new()),
    }
}

async fn handle_memory(
    State(state): State<AppState>,
    Path(agent_id): Path<String>,
) -> impl IntoResponse {
    let (resp_tx, resp_rx) = oneshot::channel();
    let _ = state.tx.send(SaasCommand::Memory {
        agent_id,
        resp: resp_tx,
    }).await;

    match resp_rx.await {
        Ok(Ok(globals)) => (
            StatusCode::OK,
            Json(MemoryResponse {
                success: true,
                error: None,
                globals,
            }),
        ),
        Ok(Err(e)) => (
            StatusCode::NOT_FOUND,
            Json(MemoryResponse {
                success: false,
                error: Some(e),
                globals: HashMap::new(),
            }),
        ),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(MemoryResponse {
                success: false,
                error: Some("Internal actor thread error".to_string()),
                globals: HashMap::new(),
            }),
        ),
    }
}

// ── Simple UUID Helper ──
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    format!("{:x}", start & 0xffffffffffff)
}
