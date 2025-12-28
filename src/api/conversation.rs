//! Conversation API Module
//!
//! Provides natural conversational interface for human interaction with ALEN.
//! Maintains conversation history, context, and personality.

use super::{AppState, Problem};
use crate::core::ThoughtState;
use crate::generation::ContentGenerator;
use crate::memory::{SemanticFact, Episode};

use axum::{
    extract::{State, Json},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Conversation message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub role: MessageRole,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_vector: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// Conversation session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub messages: Vec<Message>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: ConversationMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMetadata {
    pub title: Option<String>,
    pub summary: Option<String>,
    pub tags: Vec<String>,
    pub system_prompt: String,
}

impl Conversation {
    pub fn new(system_prompt: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            messages: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: ConversationMetadata {
                title: None,
                summary: None,
                tags: vec![],
                system_prompt,
            },
        }
    }

    pub fn add_message(&mut self, role: MessageRole, content: String, thought: Option<Vec<f64>>, confidence: Option<f64>) {
        self.messages.push(Message {
            id: Uuid::new_v4().to_string(),
            role,
            content,
            timestamp: Utc::now(),
            thought_vector: thought,
            confidence,
        });
        self.updated_at = Utc::now();
    }

    /// Get recent context (last N messages)
    pub fn get_context(&self, last_n: usize) -> String {
        self.messages
            .iter()
            .rev()
            .take(last_n)
            .rev()
            .map(|m| format!("{:?}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ============================================================================
// API Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub include_context: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub conversation_id: String,
    pub message: String,
    pub confidence: f64,
    pub energy: f64,
    pub operator_used: String,
    pub thought_vector: Vec<f64>,
    pub context_used: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_media: Option<GeneratedMedia>,
}

#[derive(Debug, Serialize)]
pub struct GeneratedMedia {
    pub image_data: Option<String>,
    pub video_frames: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateSystemPromptRequest {
    pub conversation_id: String,
    pub system_prompt: String,
}

#[derive(Debug, Serialize)]
pub struct GetConversationResponse {
    pub conversation: Conversation,
    pub message_count: usize,
    pub average_confidence: f64,
}

// ============================================================================
// Global Conversation Store
// ============================================================================

use tokio::sync::Mutex as TokioMutex;
use std::collections::HashMap;

pub struct ConversationStore {
    conversations: HashMap<String, Conversation>,
    default_system_prompt: String,
}

impl ConversationStore {
    pub fn new() -> Self {
        Self {
            conversations: HashMap::new(),
            default_system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        }
    }

    pub fn get_or_create(&mut self, id: Option<String>, custom_prompt: Option<String>) -> String {
        let id = id.unwrap_or_else(|| Uuid::new_v4().to_string());

        if !self.conversations.contains_key(&id) {
            let prompt = custom_prompt.unwrap_or_else(|| self.default_system_prompt.clone());
            self.conversations.insert(id.clone(), Conversation::new(prompt));
        }

        id
    }

    pub fn get(&self, id: &str) -> Option<&Conversation> {
        self.conversations.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Conversation> {
        self.conversations.get_mut(id)
    }

    pub fn update_system_prompt(&mut self, id: &str, prompt: String) -> bool {
        if let Some(conv) = self.conversations.get_mut(id) {
            conv.metadata.system_prompt = prompt;
            conv.updated_at = Utc::now();
            true
        } else {
            false
        }
    }

    pub fn set_default_system_prompt(&mut self, prompt: String) {
        self.default_system_prompt = prompt;
    }

    pub fn list_conversations(&self) -> Vec<(String, DateTime<Utc>, usize)> {
        self.conversations
            .iter()
            .map(|(id, conv)| (id.clone(), conv.updated_at, conv.messages.len()))
            .collect()
    }
}

// Extend AppState
pub struct ConversationManager {
    pub store: TokioMutex<ConversationStore>,
}

impl ConversationManager {
    pub fn new() -> Self {
        Self {
            store: TokioMutex::new(ConversationStore::new()),
        }
    }
}

// Default system prompt
const DEFAULT_SYSTEM_PROMPT: &str = r#"You are ALEN (Advanced Learning Engine with Neural Understanding), a sophisticated AI system that learns by proving understanding through verified learning.

Your core capabilities:
- Multimodal understanding (text, images, video, audio)
- Verified learning through backward inference
- Energy-based reasoning with multiple operators
- Generation of text, images, and videos
- Episodic and semantic memory
- Mathematical reasoning and neural computation

You respond thoughtfully, explaining your reasoning process when helpful. You can discuss your internal workings, including thought vectors, energy functions, and verification mechanisms. You learn from every interaction and store verified knowledge in your memory systems.

When uncertain, you express appropriate confidence levels. You prioritize genuine understanding over pattern matching."#;

// ============================================================================
// API Handlers
// ============================================================================

/// Chat endpoint - main conversational interface
pub async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    let dim = state.config.dimension;

    // Get or create conversation
    let mut conv_store = state.conversation_manager.store.lock().await;
    let conv_id = conv_store.get_or_create(req.conversation_id.clone(), req.system_prompt.clone());
    let conversation = conv_store.get_mut(&conv_id).unwrap();

    // Add user message
    conversation.add_message(MessageRole::User, req.message.clone(), None, None);

    // Build context
    let context_size = req.include_context.unwrap_or(5);
    let context = conversation.get_context(context_size);

    // Create problem with context and system prompt
    let full_input = format!(
        "System: {}\n\nContext:\n{}\n\nUser: {}",
        conversation.metadata.system_prompt,
        context,
        req.message
    );

    let problem = Problem::new(&full_input, dim);

    // Run inference
    let result = engine.infer(&problem);

    // Generate response text
    let generator = ContentGenerator::new(dim);
    let max_tokens = req.max_tokens.unwrap_or(100);
    let response_content = generator.generate_text(&result.thought, max_tokens);

    // Add assistant message
    conversation.add_message(
        MessageRole::Assistant,
        response_content.text.clone().unwrap_or_default(),
        Some(result.thought.vector.clone()),
        Some(result.confidence),
    );

    // Store in episodic memory
    let episode = Episode::from_inference(
        &req.message,
        &response_content.text.clone().unwrap_or_default(),
        &result.thought,
        &result.energy,
        &result.operator_id,
    );
    let _ = engine.episodic_memory.store(&episode);

    Json(ChatResponse {
        conversation_id: conv_id,
        message: response_content.text.unwrap_or_default(),
        confidence: result.confidence,
        energy: result.energy.total,
        operator_used: result.operator_id,
        thought_vector: result.thought.vector,
        context_used: context_size,
        generated_media: None,
    })
}

/// Get conversation history
pub async fn get_conversation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let conv_id = req["conversation_id"].as_str().unwrap_or_default();
    let conv_store = state.conversation_manager.store.lock().await;

    if let Some(conversation) = conv_store.get(conv_id) {
        let avg_confidence = conversation.messages.iter()
            .filter_map(|m| m.confidence)
            .sum::<f64>() / conversation.messages.len().max(1) as f64;

        Json(serde_json::json!({
            "success": true,
            "conversation": conversation,
            "message_count": conversation.messages.len(),
            "average_confidence": avg_confidence,
        }))
    } else {
        Json(serde_json::json!({
            "success": false,
            "error": "Conversation not found"
        }))
    }
}

/// Update system prompt for a conversation
pub async fn update_system_prompt(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UpdateSystemPromptRequest>,
) -> impl IntoResponse {
    let mut conv_store = state.conversation_manager.store.lock().await;

    if conv_store.update_system_prompt(&req.conversation_id, req.system_prompt.clone()) {
        Json(serde_json::json!({
            "success": true,
            "message": "System prompt updated",
            "conversation_id": req.conversation_id
        }))
    } else {
        Json(serde_json::json!({
            "success": false,
            "error": "Conversation not found"
        }))
    }
}

/// Set global default system prompt
pub async fn set_default_system_prompt(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let prompt = req["system_prompt"].as_str().unwrap_or(DEFAULT_SYSTEM_PROMPT).to_string();
    let mut conv_store = state.conversation_manager.store.lock().await;
    conv_store.set_default_system_prompt(prompt.clone());

    Json(serde_json::json!({
        "success": true,
        "message": "Default system prompt updated",
        "prompt": prompt
    }))
}

/// Get current default system prompt
pub async fn get_default_system_prompt(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let conv_store = state.conversation_manager.store.lock().await;

    Json(serde_json::json!({
        "system_prompt": conv_store.default_system_prompt
    }))
}

/// List all conversations
pub async fn list_conversations(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let conv_store = state.conversation_manager.store.lock().await;
    let conversations = conv_store.list_conversations();

    Json(serde_json::json!({
        "conversations": conversations,
        "count": conversations.len()
    }))
}

/// Clear a conversation
pub async fn clear_conversation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let conv_id = req["conversation_id"].as_str().unwrap_or_default();
    let mut conv_store = state.conversation_manager.store.lock().await;

    if let Some(conv) = conv_store.get_mut(conv_id) {
        conv.messages.clear();
        conv.updated_at = Utc::now();
        Json(serde_json::json!({
            "success": true,
            "message": "Conversation cleared"
        }))
    } else {
        Json(serde_json::json!({
            "success": false,
            "error": "Conversation not found"
        }))
    }
}
