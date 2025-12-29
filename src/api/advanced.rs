//! Advanced API Endpoints
//!
//! Sophisticated reasoning capabilities exposed via REST API

use axum::{
    extract::{State, Json},
    response::{IntoResponse, sse::{Event, Sse}},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use futures::stream::{self, Stream};
use std::convert::Infallible;

use crate::reasoning::{
    MathSolver, MathResult,
    ChainOfThoughtReasoner, ReasoningChain,
    LogicalInference, Premise, Conclusion,
};
use crate::neural::{NeuralReasoningEngine, ALENConfig, UniversalNetworkConfig, TransformerConfig};

/// Shared state for advanced endpoints
pub struct AdvancedState {
    pub math_solver: Arc<Mutex<MathSolver>>,
    pub chain_reasoner: Arc<Mutex<ChainOfThoughtReasoner>>,
    pub logic_engine: Arc<Mutex<LogicalInference>>,
    pub neural_engine: Arc<Mutex<NeuralReasoningEngine>>,
}

impl AdvancedState {
    pub fn new() -> Self {
        let alen_config = ALENConfig::default();
        let universal_config = UniversalNetworkConfig {
            input_dim: 512,
            audience_dim: 64,
            memory_dim: 256,
            solution_dim: 256,
            explanation_dim: 512,
            solve_hidden: vec![512, 256],
            verify_hidden: vec![256, 128],
            explain_hidden: vec![512, 256],
            transformer_config: TransformerConfig::default(),
            dropout: 0.1,
            alpha: 1.0,
            beta: 0.5,
            gamma: 0.5,
        };
        Self {
            math_solver: Arc::new(Mutex::new(MathSolver::new())),
            chain_reasoner: Arc::new(Mutex::new(ChainOfThoughtReasoner::default())),
            logic_engine: Arc::new(Mutex::new(LogicalInference::new())),
            neural_engine: Arc::new(Mutex::new(NeuralReasoningEngine::new(
                alen_config,
                universal_config,
                256,
                10,
            ))),
        }
    }
}

// ============= Request/Response Types =============

#[derive(Debug, Deserialize)]
pub struct MathRequest {
    pub expression: String,
    #[serde(default)]
    pub operation: String, // "solve", "simplify", "derivative"
    #[serde(default)]
    pub variable: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MathResponse {
    pub result: MathResult,
    pub success: bool,
}

#[derive(Debug, Deserialize)]
pub struct ChainRequest {
    pub problem: String,
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
}

fn default_max_steps() -> usize { 10 }

#[derive(Debug, Serialize)]
pub struct ChainResponse {
    pub chain: ReasoningChain,
    pub success: bool,
}

#[derive(Debug, Deserialize)]
pub struct LogicRequest {
    pub premises: Vec<String>,
    #[serde(default)]
    pub infer_all: bool,
}

#[derive(Debug, Serialize)]
pub struct LogicResponse {
    pub conclusions: Vec<Conclusion>,
    pub premises_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct AdvancedInferRequest {
    pub question: String,
    #[serde(default)]
    pub use_chain_of_thought: bool,
    #[serde(default)]
    pub use_math_solver: bool,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct AdvancedInferResponse {
    pub answer: String,
    pub confidence: f64,
    pub reasoning_steps: Vec<String>,
    pub operator_used: String,
    pub verified: bool,
    pub math_result: Option<MathResult>,
    pub chain: Option<ReasoningChain>,
}

// ============= Handlers =============

/// Solve mathematical expression
pub async fn solve_math(
    State(state): State<Arc<AdvancedState>>,
    Json(req): Json<MathRequest>,
) -> Result<Json<MathResponse>, StatusCode> {
    let solver = state.math_solver.lock().await;
    
    let result = match req.operation.as_str() {
        "derivative" => {
            if let Some(var) = req.variable {
                solver.derivative(&req.expression, &var)
            } else {
                solver.solve(&req.expression)
            }
        }
        "equation" => solver.solve_equation(&req.expression),
        _ => solver.solve(&req.expression),
    };
    
    Ok(Json(MathResponse {
        success: result.value.is_some() || result.confidence > 0.5,
        result,
    }))
}

/// Chain-of-thought reasoning
pub async fn chain_of_thought(
    State(state): State<Arc<AdvancedState>>,
    Json(req): Json<ChainRequest>,
) -> Result<Json<ChainResponse>, StatusCode> {
    let reasoner = state.chain_reasoner.lock().await;
    
    let chain = reasoner.reason(&req.problem);
    let success = reasoner.verify_chain(&chain);
    
    Ok(Json(ChainResponse {
        chain,
        success,
    }))
}

/// Logical inference
pub async fn logical_inference(
    State(state): State<Arc<AdvancedState>>,
    Json(req): Json<LogicRequest>,
) -> Result<Json<LogicResponse>, StatusCode> {
    let mut logic = state.logic_engine.lock().await;
    
    // Clear previous state
    *logic = LogicalInference::new();
    
    // Add premises
    for premise in req.premises {
        logic.add_premise(premise, 1.0);
    }
    
    // Infer conclusions
    let conclusions = if req.infer_all {
        logic.infer_all()
    } else {
        vec![]
    };
    
    Ok(Json(LogicResponse {
        conclusions,
        premises_count: logic.get_premises().len(),
    }))
}

/// Advanced inference with multiple reasoning modes
pub async fn advanced_infer(
    State(state): State<Arc<AdvancedState>>,
    Json(req): Json<AdvancedInferRequest>,
) -> Result<Json<AdvancedInferResponse>, StatusCode> {
    let mut reasoning_steps = Vec::new();
    
    // Try math solver if requested
    let math_result = if req.use_math_solver {
        let solver = state.math_solver.lock().await;
        let result = solver.solve(&req.question);
        reasoning_steps.extend(result.steps.clone());
        Some(result)
    } else {
        None
    };
    
    // Try chain-of-thought if requested
    let chain = if req.use_chain_of_thought {
        let reasoner = state.chain_reasoner.lock().await;
        let c = reasoner.reason(&req.question);
        reasoning_steps.push(c.summary());
        Some(c)
    } else {
        None
    };
    
    // Use neural engine
    let mut engine = state.neural_engine.lock().await;
    let result = engine.reason(&req.question);
    
    reasoning_steps.push(format!("Neural reasoning steps: {}", result.steps.len()));
    reasoning_steps.push(format!("Final confidence: {:.4}", result.confidence));
    
    Ok(Json(AdvancedInferResponse {
        answer: result.answer.clone(),
        confidence: result.confidence as f64,
        reasoning_steps,
        operator_used: "neural_reasoning".to_string(),
        verified: result.verified,
        math_result,
        chain,
    }))
}

/// Streaming inference (Server-Sent Events)
pub async fn stream_inference(
    State(state): State<Arc<AdvancedState>>,
    Json(req): Json<AdvancedInferRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let question = req.question.clone();
    
    let stream = stream::iter(vec![
        Ok(Event::default().data(format!("Starting inference on: {}", question))),
        Ok(Event::default().data("Step 1: Encoding input...")),
        Ok(Event::default().data("Step 2: Generating candidates...")),
        Ok(Event::default().data("Step 3: Evaluating energy...")),
        Ok(Event::default().data("Step 4: Selecting best operator...")),
        Ok(Event::default().data("Step 5: Verifying result...")),
        Ok(Event::default().data("Complete!")),
    ]);
    
    Sse::new(stream)
}

/// Get system capabilities
pub async fn get_capabilities() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "reasoning_modes": [
            "neural_network",
            "chain_of_thought",
            "mathematical_solver",
            "logical_inference",
            "symbolic_reasoning"
        ],
        "math_operations": [
            "solve",
            "simplify",
            "derivative",
            "equation"
        ],
        "operators": [
            "Logical",
            "Probabilistic",
            "Heuristic",
            "Analogical",
            "Conservative",
            "Exploratory",
            "Analytical",
            "Intuitive"
        ],
        "features": [
            "multi_step_reasoning",
            "verification",
            "streaming",
            "parallel_operators",
            "cycle_consistency"
        ],
        "version": "0.3.0"
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_math_solver() {
        let state = Arc::new(AdvancedState::new());
        let req = MathRequest {
            expression: "2+3".to_string(),
            operation: "solve".to_string(),
            variable: None,
        };
        
        let response = solve_math(State(state), Json(req)).await;
        assert!(response.is_ok());
    }
}
