//! Export and Import System
//!
//! Allows users to export conversations, memories, and training data
//! to files (like real LLMs) for inspection, backup, and transfer.

use super::{AppState};
use crate::memory::{Episode, SemanticFact};
use crate::api::conversation::Conversation;

use axum::{
    extract::{State, Json},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::fs;
use std::path::PathBuf;

/// Export conversations to JSON file
#[derive(Debug, Deserialize)]
pub struct ExportRequest {
    /// Output file path (optional, uses default if not provided)
    pub output_path: Option<String>,
    /// What to export: "conversations", "episodic", "semantic", "all"
    pub export_type: String,
}

#[derive(Debug, Serialize)]
pub struct ExportResponse {
    pub success: bool,
    pub file_path: String,
    pub items_exported: usize,
    pub file_size_bytes: u64,
}

/// Export conversations to JSON
pub async fn export_conversations(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExportRequest>,
) -> impl IntoResponse {
    let conv_store = state.conversation_manager.store.lock().await;

    // Collect all conversations
    let conversations: Vec<Conversation> = conv_store.conversations.values().cloned().collect();

    // Determine output path
    let output_path = req.output_path.unwrap_or_else(|| {
        format!("{}/exports/conversations_{}.json",
            state.storage.base_dir.display(),
            chrono::Utc::now().format("%Y%m%d_%H%M%S"))
    });

    // Create parent directory
    if let Some(parent) = PathBuf::from(&output_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    // Serialize and write
    match serde_json::to_string_pretty(&conversations) {
        Ok(json) => {
            match fs::write(&output_path, &json) {
                Ok(_) => {
                    let file_size = fs::metadata(&output_path)
                        .map(|m| m.len())
                        .unwrap_or(0);

                    Json(serde_json::json!({
                        "success": true,
                        "file_path": output_path,
                        "items_exported": conversations.len(),
                        "file_size_bytes": file_size,
                        "file_size_kb": file_size as f64 / 1024.0,
                        "format": "JSON (human-readable)"
                    }))
                },
                Err(e) => {
                    Json(serde_json::json!({
                        "success": false,
                        "error": format!("Failed to write file: {}", e)
                    }))
                }
            }
        },
        Err(e) => {
            Json(serde_json::json!({
                "success": false,
                "error": format!("Failed to serialize: {}", e)
            }))
        }
    }
}

/// Export episodic memory to JSON
pub async fn export_episodic_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExportRequest>,
) -> impl IntoResponse {
    let engine = state.engine.lock().await;

    // Get all episodes
    let episodes = match engine.episodic_memory.get_all_episodes(1000) {
        Ok(eps) => eps,
        Err(e) => {
            return Json(serde_json::json!({
                "success": false,
                "error": format!("Failed to retrieve episodes: {}", e)
            }));
        }
    };

    // Determine output path
    let output_path = req.output_path.unwrap_or_else(|| {
        format!("{}/exports/episodic_memory_{}.json",
            state.storage.base_dir.display(),
            chrono::Utc::now().format("%Y%m%d_%H%M%S"))
    });

    // Create parent directory
    if let Some(parent) = PathBuf::from(&output_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    // Serialize and write
    match serde_json::to_string_pretty(&episodes) {
        Ok(json) => {
            match fs::write(&output_path, &json) {
                Ok(_) => {
                    let file_size = fs::metadata(&output_path)
                        .map(|m| m.len())
                        .unwrap_or(0);

                    Json(serde_json::json!({
                        "success": true,
                        "file_path": output_path,
                        "items_exported": episodes.len(),
                        "file_size_bytes": file_size,
                        "file_size_kb": file_size as f64 / 1024.0,
                        "format": "JSON (human-readable)"
                    }))
                },
                Err(e) => {
                    Json(serde_json::json!({
                        "success": false,
                        "error": format!("Failed to write file: {}", e)
                    }))
                }
            }
        },
        Err(e) => {
            Json(serde_json::json!({
                "success": false,
                "error": format!("Failed to serialize: {}", e)
            }))
        }
    }
}

/// Export semantic memory to JSON
pub async fn export_semantic_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExportRequest>,
) -> impl IntoResponse {
    let engine = state.engine.lock().await;

    // Get all facts
    let facts = match engine.semantic_memory.get_all_facts(1000) {
        Ok(facts) => facts,
        Err(e) => {
            return Json(serde_json::json!({
                "success": false,
                "error": format!("Failed to retrieve facts: {}", e)
            }));
        }
    };

    // Determine output path
    let output_path = req.output_path.unwrap_or_else(|| {
        format!("{}/exports/semantic_memory_{}.json",
            state.storage.base_dir.display(),
            chrono::Utc::now().format("%Y%m%d_%H%M%S"))
    });

    // Create parent directory
    if let Some(parent) = PathBuf::from(&output_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    // Serialize and write
    match serde_json::to_string_pretty(&facts) {
        Ok(json) => {
            match fs::write(&output_path, &json) {
                Ok(_) => {
                    let file_size = fs::metadata(&output_path)
                        .map(|m| m.len())
                        .unwrap_or(0);

                    Json(serde_json::json!({
                        "success": true,
                        "file_path": output_path,
                        "items_exported": facts.len(),
                        "file_size_bytes": file_size,
                        "file_size_kb": file_size as f64 / 1024.0,
                        "format": "JSON (human-readable)"
                    }))
                },
                Err(e) => {
                    Json(serde_json::json!({
                        "success": false,
                        "error": format!("Failed to write file: {}", e)
                    }))
                }
            }
        },
        Err(e) => {
            Json(serde_json::json!({
                "success": false,
                "error": format!("Failed to serialize: {}", e)
            }))
        }
    }
}

/// List available exports
pub async fn list_exports(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let exports_dir = state.storage.base_dir.join("exports");

    if !exports_dir.exists() {
        return Json(serde_json::json!({
            "exports": [],
            "count": 0,
            "exports_dir": exports_dir
        }));
    }

    let mut exports = Vec::new();

    if let Ok(entries) = fs::read_dir(&exports_dir) {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                if metadata.is_file() {
                    exports.push(serde_json::json!({
                        "name": entry.file_name().to_string_lossy(),
                        "path": entry.path(),
                        "size_bytes": metadata.len(),
                        "size_kb": metadata.len() as f64 / 1024.0,
                        "modified": metadata.modified().ok()
                    }));
                }
            }
        }
    }

    Json(serde_json::json!({
        "exports": exports,
        "count": exports.len(),
        "exports_dir": exports_dir
    }))
}
