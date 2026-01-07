//! Master Neural System Training API
//!
//! Endpoints for training the Master Neural System with:
//! - Document upload (txt, json, pdf)
//! - Batch training on uploaded documents
//! - Integration with database persistence
//! - Progress tracking and statistics

use super::AppState;
use crate::neural::{MasterNeuralSystem, MasterSystemConfig};
use axum::{
    extract::{State, Multipart},
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::path::PathBuf;
use tokio::sync::Mutex;
use once_cell::sync::Lazy;

/// Global master neural system (shared across requests)
static MASTER_SYSTEM: Lazy<Arc<Mutex<Option<MasterNeuralSystem>>>> =
    Lazy::new(|| Arc::new(Mutex::new(None)));

/// Global storage for uploaded training examples
static UPLOADED_EXAMPLES: Lazy<Arc<Mutex<Vec<TrainingExample>>>> =
    Lazy::new(|| Arc::new(Mutex::new(Vec::new())));

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentUploadResponse {
    pub success: bool,
    pub message: String,
    pub examples_parsed: usize,
    pub file_name: String,
    pub file_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainMasterRequest {
    pub examples: Vec<TrainingExample>,
    pub save_checkpoint: Option<bool>,
    pub checkpoint_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingExample {
    pub input: String,
    pub target: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainMasterResponse {
    pub success: bool,
    pub message: String,
    pub examples_trained: usize,
    pub average_loss: f64,
    pub average_confidence: f64,
    pub total_training_steps: u64,
    pub controller_updates: u64,
    pub core_model_updates: u64,
    pub memories_in_db: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MasterChatRequest {
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MasterChatResponse {
    pub response: String,
    pub confidence: f64,
    pub perplexity: f64,
    pub reasoning_depth: usize,
    pub creativity_level: f64,
    pub action: String,
    pub total_memories: usize,
    pub training_steps: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MasterStatsResponse {
    pub initialized: bool,
    pub total_training_steps: u64,
    pub controller_updates: u64,
    pub core_model_updates: u64,
    pub avg_confidence: f64,
    pub avg_perplexity: f64,
    pub controller_lr: f64,
    pub core_lr: f64,
    pub total_memories: usize,
    pub db_path: Option<String>,
}

/// Initialize master neural system
async fn ensure_master_system() {
    let mut system_lock = MASTER_SYSTEM.lock().await;

    if system_lock.is_none() {
        eprintln!("🔧 Initializing Master Neural System...");

        let config = MasterSystemConfig {
            thought_dim: 128,
            hidden_dim: 256,
            vocab_size: 5000,
            controller_lr: 0.001,  // SMALL - governance
            controller_patterns: 50,
            core_model_lr: 0.1,    // LARGE - learning
            transformer_layers: 4,
            attention_heads: 4,
            memory_capacity: 1000,
            retrieval_top_k: 3,
            use_meta_learning: true,
            use_creativity: true,
            use_self_discovery: true,
            batch_size: 16,
            max_epochs: 10,
            enable_persistence: true,
            db_path: Some(PathBuf::from("./data/alen_neural.db")),
            checkpoint_interval: 100,
        };

        let system = MasterNeuralSystem::new(config);
        *system_lock = Some(system);

        eprintln!("✅ Master Neural System initialized!");
    }
}

/// Upload document for training (supports txt, json)
pub async fn upload_document(
    State(_state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut examples = Vec::new();
    let mut file_name = String::new();
    let mut file_size = 0;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();

        if name == "file" {
            file_name = field.file_name().unwrap_or("unknown").to_string();
            let data = field.bytes().await.unwrap_or_default();
            file_size = data.len();

            let content = String::from_utf8_lossy(&data).to_string();

            // Parse based on file type
            if file_name.ends_with(".json") {
                // Try to parse as JSON training data
                if let Ok(json_data) = serde_json::from_str::<TrainMasterRequest>(&content) {
                    examples.extend(json_data.examples);
                } else if let Ok(json_examples) = serde_json::from_str::<Vec<TrainingExample>>(&content) {
                    examples.extend(json_examples);
                }
            } else if file_name.ends_with(".txt") {
                // Parse Q: A: format
                examples.extend(parse_qa_format(&content));
            }
        }
    }

    // Store uploaded examples for training
    if !examples.is_empty() {
        let mut uploaded = UPLOADED_EXAMPLES.lock().await;
        uploaded.clear(); // Clear previous uploads
        uploaded.extend(examples.clone());
        eprintln!("📁 Stored {} examples from upload", examples.len());
    }

    Json(DocumentUploadResponse {
        success: !examples.is_empty(),
        message: if examples.is_empty() {
            "No training examples found in file".to_string()
        } else {
            format!("Successfully parsed and stored {} examples", examples.len())
        },
        examples_parsed: examples.len(),
        file_name,
        file_size,
    })
}

/// Train master neural system with examples
pub async fn train_master(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<TrainMasterRequest>,
) -> impl IntoResponse {
    ensure_master_system().await;
    let mut system_lock = MASTER_SYSTEM.lock().await;

    if let Some(ref mut system) = system_lock.as_mut() {
        // Use uploaded examples if request is empty
        let examples = if req.examples.is_empty() {
            let uploaded = UPLOADED_EXAMPLES.lock().await;
            uploaded.clone()
        } else {
            req.examples.clone()
        };

        if examples.is_empty() {
            return Json(TrainMasterResponse {
                success: false,
                message: "No training examples provided. Please upload a document first.".to_string(),
                examples_trained: 0,
                average_loss: 0.0,
                average_confidence: 0.0,
                total_training_steps: 0,
                controller_updates: 0,
                core_model_updates: 0,
                memories_in_db: 0,
            });
        }

        let mut total_loss = 0.0;
        let mut total_confidence = 0.0;
        let examples_count = examples.len();

        for example in &examples {
            let metrics = system.train_step(&example.input, &example.target);
            total_loss += metrics.total_loss;
            total_confidence += metrics.confidence;
        }

        // Save checkpoint if requested
        if req.save_checkpoint.unwrap_or(false) {
            let checkpoint_name = req.checkpoint_name
                .unwrap_or_else(|| format!("web_upload_{}", chrono::Utc::now().timestamp()));
            let _ = system.save_checkpoint(&checkpoint_name);
        }

        let stats = system.get_stats();
        let avg_loss = if examples_count > 0 { total_loss / examples_count as f64 } else { 0.0 };
        let avg_conf = if examples_count > 0 { total_confidence / examples_count as f64 } else { 0.0 };

        Json(TrainMasterResponse {
            success: true,
            message: format!("Successfully trained on {} examples", examples_count),
            examples_trained: examples_count,
            average_loss: avg_loss,
            average_confidence: avg_conf,
            total_training_steps: stats.total_training_steps,
            controller_updates: stats.controller_updates,
            core_model_updates: stats.core_model_updates,
            memories_in_db: system.get_total_memories(),
        })
    } else {
        Json(TrainMasterResponse {
            success: false,
            message: "Failed to initialize master system".to_string(),
            examples_trained: 0,
            average_loss: 0.0,
            average_confidence: 0.0,
            total_training_steps: 0,
            controller_updates: 0,
            core_model_updates: 0,
            memories_in_db: 0,
        })
    }
}

/// Chat with master neural system
pub async fn chat_master(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<MasterChatRequest>,
) -> impl IntoResponse {
    ensure_master_system().await;
    let mut system_lock = MASTER_SYSTEM.lock().await;

    if let Some(ref mut system) = system_lock.as_mut() {
        let response_obj = system.forward(&req.message);
        let stats = system.get_stats();

        Json(MasterChatResponse {
            response: response_obj.response,
            confidence: response_obj.confidence,
            perplexity: response_obj.perplexity,
            reasoning_depth: response_obj.controls.reasoning_depth,
            creativity_level: response_obj.controls.style.creativity,
            action: format!("{:?}", response_obj.controls.action),
            total_memories: system.get_total_memories(),
            training_steps: stats.total_training_steps,
        })
    } else {
        Json(MasterChatResponse {
            response: "Master system not initialized".to_string(),
            confidence: 0.0,
            perplexity: 100.0,
            reasoning_depth: 0,
            creativity_level: 0.0,
            action: "Error".to_string(),
            total_memories: 0,
            training_steps: 0,
        })
    }
}

/// Get master system statistics
pub async fn get_master_stats(
    State(_state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ensure_master_system().await;
    let system_lock = MASTER_SYSTEM.lock().await;

    if let Some(ref system) = system_lock.as_ref() {
        let stats = system.get_stats();
        let db_path = system.get_db_path().map(|p| p.display().to_string());

        Json(MasterStatsResponse {
            initialized: true,
            total_training_steps: stats.total_training_steps,
            controller_updates: stats.controller_updates,
            core_model_updates: stats.core_model_updates,
            avg_confidence: stats.avg_confidence,
            avg_perplexity: stats.avg_perplexity,
            controller_lr: stats.controller_lr,
            core_lr: stats.core_lr,
            total_memories: system.get_total_memories(),
            db_path,
        })
    } else {
        Json(MasterStatsResponse {
            initialized: false,
            total_training_steps: 0,
            controller_updates: 0,
            core_model_updates: 0,
            avg_confidence: 0.0,
            avg_perplexity: 0.0,
            controller_lr: 0.0,
            core_lr: 0.0,
            total_memories: 0,
            db_path: None,
        })
    }
}

/// Parse Q: A: format from text
fn parse_qa_format(text: &str) -> Vec<TrainingExample> {
    let mut examples = Vec::new();
    let mut current_question = String::new();
    let mut current_answer = String::new();
    let mut in_answer = false;

    for line in text.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("Q:") || trimmed.starts_with("Question:") {
            // Save previous Q&A if exists
            if !current_question.is_empty() && !current_answer.is_empty() {
                examples.push(TrainingExample {
                    input: current_question.clone(),
                    target: current_answer.clone(),
                });
            }

            // Start new question
            current_question = trimmed.split_once(':')
                .map(|(_, q)| q.trim().to_string())
                .unwrap_or_default();
            current_answer.clear();
            in_answer = false;
        } else if trimmed.starts_with("A:") || trimmed.starts_with("Answer:") {
            current_answer = trimmed.split_once(':')
                .map(|(_, a)| a.trim().to_string())
                .unwrap_or_default();
            in_answer = true;
        } else if !trimmed.is_empty() {
            if in_answer {
                if !current_answer.is_empty() {
                    current_answer.push(' ');
                }
                current_answer.push_str(trimmed);
            } else if !current_question.is_empty() {
                current_question.push(' ');
                current_question.push_str(trimmed);
            }
        }
    }

    // Save last Q&A
    if !current_question.is_empty() && !current_answer.is_empty() {
        examples.push(TrainingExample {
            input: current_question,
            target: current_answer,
        });
    }

    examples
}
