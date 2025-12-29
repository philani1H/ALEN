//! Conversation API Module
//!
//! Provides natural conversational interface for human interaction with ALEN.
//! Maintains conversation history, context, and personality.
//!
//! Uses Intent Extraction: I = (τ, θ, C) to understand prompts before responding.
//! - τ = task vector (what to do)
//! - θ = target variables (what it's about)
//! - C = constraint set (how to do it)

use super::{AppState, Problem};
use crate::core::{ThoughtState, IntentExtractor, IntentState, TaskType, ResponseEnergy};
use crate::learning::feedback_loop::InferenceResult;
use crate::memory::{SemanticFact, Episode, SemanticMemory};
use crate::control::{MoodEngine, EmotionSystem};
use crate::generation::ContentGenerator;

use axum::{
    extract::{State, Json},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::fs::OpenOptions;
use std::io::Write;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_steps: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mood: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion: Option<String>,
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
    pub conversations: HashMap<String, Conversation>,
    pub default_system_prompt: String,
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

    // Generate response using full generation pipeline
    let response_text = generate_intelligent_response(
        &req.message,
        &result,
        &engine.semantic_memory,
        &engine.mood_engine,
        &engine.emotion_system,
        dim,
    );

    // Add assistant message
    conversation.add_message(
        MessageRole::Assistant,
        response_text.clone(),
        Some(result.thought.vector.clone()),
        Some(result.confidence),
    );

    // Store in episodic memory
    let episode = Episode::from_inference(
        &req.message,
        &response_text,
        &result.thought,
        &result.energy,
        &result.operator_id,
    );
    let _ = engine.episodic_memory.store(&episode);

    // Get mood and emotion state
    let mood = engine.mood_engine.current_mood().as_str().to_string();
    let emotion = engine.emotion_system.current_emotion().as_str().to_string();

    // Generate reasoning steps
    let reasoning_steps = vec![
        format!("Analyzed input using {} operator", result.operator_id),
        format!("Processed with confidence: {:.1}%", result.confidence * 100.0),
        format!("Generated response in current mood: {}", mood),
    ];

    Json(ChatResponse {
        conversation_id: conv_id,
        message: response_text,
        confidence: result.confidence,
        energy: result.energy.total,
        operator_used: result.operator_id,
        thought_vector: result.thought.vector,
        context_used: context_size,
        generated_media: None,
        reasoning_steps: Some(reasoning_steps),
        mood: Some(mood),
        emotion: Some(emotion),
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

/// Generate intelligent response using full generation pipeline
/// Now uses Intent Extraction: I = (τ, θ, C) before generation
fn generate_intelligent_response(
    user_input: &str,
    inference_result: &InferenceResult,
    semantic_memory: &SemanticMemory,
    mood_engine: &MoodEngine,
    emotion_system: &EmotionSystem,
    dim: usize,
) -> String {
    // Step 1: Extract intent I = (τ, θ, C)
    let intent_extractor = IntentExtractor::new(dim);
    let intent = intent_extractor.extract(user_input);
    
    // Step 2: Route based on task type τ
    let response = match intent.task.primary_task {
        TaskType::Explain => generate_explanation_response(user_input, &intent, semantic_memory, mood_engine),
        TaskType::Summarize => generate_summary_response(user_input, &intent, semantic_memory),
        TaskType::Solve => generate_solve_response(user_input, &intent, inference_result),
        TaskType::Generate => generate_creative_response_with_intent(user_input, &intent, mood_engine.current_mood().as_str()),
        TaskType::List => generate_list_response(user_input, &intent, semantic_memory),
        TaskType::Compare => generate_comparison_response(user_input, &intent, semantic_memory),
        TaskType::Define => generate_definition_response(user_input, &intent, semantic_memory),
        TaskType::Analyze => generate_analysis_response(user_input, &intent, inference_result, semantic_memory),
        TaskType::Debug => generate_debug_response(user_input, &intent, inference_result),
        TaskType::Verify => generate_verification_response(user_input, &intent, inference_result),
        _ => generate_general_response(user_input, &intent, inference_result, semantic_memory, mood_engine),
    };
    
    // Step 3: Apply constraints C and verify response energy
    let response_energy = ResponseEnergy::default();
    let energy = response_energy.compute(&intent, &response);
    
    // If energy too high (response doesn't satisfy constraints), try to improve
    if energy > 0.5 {
        apply_constraints_to_response(&response, &intent)
    } else {
        response
    }
}

/// Generate explanation response (TaskType::Explain)
fn generate_explanation_response(
    user_input: &str,
    intent: &IntentState,
    semantic_memory: &SemanticMemory,
    mood_engine: &MoodEngine,
) -> String {
    let knowledge = retrieve_relevant_knowledge(user_input, semantic_memory);
    
    if !knowledge.is_empty() {
        let mut response = knowledge.join(" ");
        
        // Add explanation structure based on constraints
        if intent.constraints.format.include_math == Some(true) {
            response.push_str(" This can be expressed mathematically.");
        }
        
        // Adjust for audience (technical level)
        if let Some(level) = intent.constraints.style.technical_level {
            if level < 0.3 {
                response = simplify_response(&response);
            }
        }
        
        response
    } else {
        format!(
            "I'd like to explain '{}' for you. Could you provide more context about what specific aspect you'd like me to focus on?",
            extract_topic(user_input)
        )
    }
}

/// Generate summary response (TaskType::Summarize)
fn generate_summary_response(
    user_input: &str,
    intent: &IntentState,
    semantic_memory: &SemanticMemory,
) -> String {
    let knowledge = retrieve_relevant_knowledge(user_input, semantic_memory);
    
    if !knowledge.is_empty() {
        // Apply length constraints
        let max_words = intent.constraints.length.max_words.unwrap_or(50);
        let summary = truncate_to_words(&knowledge.join(" "), max_words);
        format!("Summary: {}", summary)
    } else {
        "I don't have enough information to summarize. Could you provide the content you'd like me to summarize?".to_string()
    }
}

/// Generate solve response (TaskType::Solve)
fn generate_solve_response(
    user_input: &str,
    intent: &IntentState,
    inference_result: &InferenceResult,
) -> String {
    let confidence = inference_result.confidence;
    
    if confidence > 0.7 {
        format!(
            "Solution (confidence: {:.1}%): Processing your problem about '{}'. The reasoning operators have analyzed this with {} candidates considered.",
            confidence * 100.0,
            extract_topic(user_input),
            inference_result.candidates_considered
        )
    } else {
        format!(
            "I'm working on solving '{}' but my confidence is {:.1}%. Could you provide more details or constraints?",
            extract_topic(user_input),
            confidence * 100.0
        )
    }
}

/// Generate creative response with intent awareness
fn generate_creative_response_with_intent(
    user_input: &str,
    intent: &IntentState,
    mood: &str,
) -> String {
    let input_lower = user_input.to_lowercase();
    
    // Check for specific creative types
    if input_lower.contains("poem") || input_lower.contains("poetry") {
        let theme = detect_theme(&input_lower);
        return generate_poem(&theme, mood);
    }
    
    if input_lower.contains("story") {
        return generate_story(mood);
    }
    
    // General creative request
    format!(
        "I'd love to create something for you! Based on your request about '{}', I can write poems, stories, or other creative content. My current mood is {} which will influence the tone. What would you like?",
        extract_topic(user_input),
        mood.to_lowercase()
    )
}

/// Generate list response (TaskType::List)
fn generate_list_response(
    user_input: &str,
    intent: &IntentState,
    semantic_memory: &SemanticMemory,
) -> String {
    let knowledge = retrieve_relevant_knowledge(user_input, semantic_memory);
    
    if !knowledge.is_empty() {
        let items: Vec<String> = knowledge.iter()
            .enumerate()
            .map(|(i, k)| format!("{}. {}", i + 1, k))
            .collect();
        items.join("\n")
    } else {
        format!("I don't have a list for '{}' yet. Could you specify what items you'd like me to list?", extract_topic(user_input))
    }
}

/// Generate comparison response (TaskType::Compare)
fn generate_comparison_response(
    user_input: &str,
    intent: &IntentState,
    semantic_memory: &SemanticMemory,
) -> String {
    let targets: Vec<String> = intent.targets.iter()
        .map(|t| t.name.clone())
        .collect();
    
    if targets.len() >= 2 {
        format!(
            "Comparing {} and {}: I'll analyze the similarities and differences. Let me search my knowledge base for relevant information.",
            targets[0], targets[1]
        )
    } else {
        "To compare, I need at least two items. What would you like me to compare?".to_string()
    }
}

/// Generate definition response (TaskType::Define)
fn generate_definition_response(
    user_input: &str,
    intent: &IntentState,
    semantic_memory: &SemanticMemory,
) -> String {
    let topic = extract_topic(user_input);
    let knowledge = retrieve_relevant_knowledge(&topic, semantic_memory);
    
    if !knowledge.is_empty() {
        format!("Definition of {}: {}", topic, knowledge[0])
    } else {
        format!("I don't have a definition for '{}' in my knowledge base yet. Would you like to teach me?", topic)
    }
}

/// Generate analysis response (TaskType::Analyze)
fn generate_analysis_response(
    user_input: &str,
    intent: &IntentState,
    inference_result: &InferenceResult,
    semantic_memory: &SemanticMemory,
) -> String {
    let knowledge = retrieve_relevant_knowledge(user_input, semantic_memory);
    let confidence = inference_result.confidence;
    
    let mut analysis = format!(
        "Analysis of '{}' (confidence: {:.1}%):\n",
        extract_topic(user_input),
        confidence * 100.0
    );
    
    if !knowledge.is_empty() {
        analysis.push_str(&format!("Based on my knowledge: {}\n", knowledge.join(" ")));
    }
    
    analysis.push_str(&format!(
        "Reasoning operator used: {}\nCandidates considered: {}",
        inference_result.operator_id,
        inference_result.candidates_considered
    ));
    
    analysis
}

/// Generate debug response (TaskType::Debug)
fn generate_debug_response(
    user_input: &str,
    intent: &IntentState,
    inference_result: &InferenceResult,
) -> String {
    format!(
        "Debug analysis for '{}': I'm examining this with {:.1}% confidence. The {} operator processed {} candidates. Could you describe the specific issue or error you're encountering?",
        extract_topic(user_input),
        inference_result.confidence * 100.0,
        inference_result.operator_id,
        inference_result.candidates_considered
    )
}

/// Generate verification response (TaskType::Verify)
fn generate_verification_response(
    user_input: &str,
    intent: &IntentState,
    inference_result: &InferenceResult,
) -> String {
    let verified = inference_result.energy.verified;
    let confidence = inference_result.confidence;
    
    if verified && confidence > 0.7 {
        format!("Verification: The statement about '{}' appears to be consistent with my knowledge (confidence: {:.1}%).", extract_topic(user_input), confidence * 100.0)
    } else {
        format!("Verification: I cannot fully verify '{}' (confidence: {:.1}%). More information may be needed.", extract_topic(user_input), confidence * 100.0)
    }
}

/// Generate general response for unknown task types
fn generate_general_response(
    user_input: &str,
    intent: &IntentState,
    inference_result: &InferenceResult,
    semantic_memory: &SemanticMemory,
    mood_engine: &MoodEngine,
) -> String {
    let input_lower = user_input.to_lowercase();
    
    // Check if this is a creative request
    let is_creative = input_lower.contains("poem") || 
                     input_lower.contains("poetry") ||
                     input_lower.contains("story") ||
                     input_lower.contains("creative") ||
                     input_lower.contains("imagine") ||
                     input_lower.contains("write");
    
    // First try semantic memory for factual knowledge
    let knowledge = retrieve_relevant_knowledge(user_input, semantic_memory);
    
    if !knowledge.is_empty() && !is_creative {
        return knowledge.join(" ");
    }
    
    // For creative requests
    if is_creative {
        return generate_creative_response(user_input, mood_engine.current_mood().as_str());
    }
    
    // Fallback
    generate_contextual_fallback(
        user_input,
        inference_result,
        mood_engine.current_mood().as_str(),
    )
}

/// Apply constraints to response
fn apply_constraints_to_response(response: &str, intent: &IntentState) -> String {
    let mut result = response.to_string();
    
    // Apply length constraint
    if let Some(max_words) = intent.constraints.length.max_words {
        result = truncate_to_words(&result, max_words);
    }
    
    // Remove excluded content
    for exclude in &intent.constraints.content.must_exclude {
        result = result.replace(exclude, "");
    }
    
    result
}

/// Helper: Extract main topic from input
fn extract_topic(input: &str) -> String {
    let skip_words = ["what", "is", "the", "a", "an", "how", "does", "do", "can", "you", "explain", "tell", "me", "about"];
    input.split_whitespace()
        .filter(|w| !skip_words.contains(&w.to_lowercase().as_str()))
        .take(3)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Helper: Detect theme from input
fn detect_theme(input: &str) -> String {
    if input.contains("love") { "love".to_string() }
    else if input.contains("nature") { "nature".to_string() }
    else if input.contains("time") { "time".to_string() }
    else if input.contains("hope") { "hope".to_string() }
    else { "life".to_string() }
}

/// Helper: Truncate to word count
fn truncate_to_words(text: &str, max_words: usize) -> String {
    text.split_whitespace()
        .take(max_words)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Helper: Simplify response for non-technical audience
fn simplify_response(text: &str) -> String {
    // Remove overly technical terms
    text.replace("neural network", "AI system")
        .replace("algorithm", "method")
        .replace("optimization", "improvement")
}

/// Check if text is gibberish (too many technical terms without structure)
fn is_gibberish(text: &str) -> bool {
    let technical_words = ["tensor", "parameter", "gradient", "embedding", "vector", 
                          "matrix", "optimization", "activation", "token", "dimension"];
    let word_count = text.split_whitespace().count();
    if word_count < 5 {
        return true;
    }
    
    let technical_count = technical_words.iter()
        .filter(|&word| text.to_lowercase().contains(word))
        .count();
    
    // If more than 40% technical words, likely gibberish
    technical_count as f64 / word_count as f64 > 0.4
}

/// Generate creative response (poems, stories, etc.)
fn generate_creative_response(user_input: &str, mood: &str) -> String {
    let input_lower = user_input.to_lowercase();
    
    // Detect what kind of creative content is requested
    if input_lower.contains("poem") || input_lower.contains("poetry") {
        // Detect theme
        let theme = if input_lower.contains("love") {
            "love"
        } else if input_lower.contains("nature") {
            "nature"
        } else if input_lower.contains("time") {
            "time"
        } else if input_lower.contains("hope") {
            "hope"
        } else {
            "life"
        };
        
        return generate_poem(theme, mood);
    }
    
    if input_lower.contains("story") {
        return generate_story(mood);
    }
    
    // General creative request
    format!(
        "I'd love to create something for you! I can write poems about love, nature, time, or hope. \
        I can also tell stories. What would you like me to create? My current mood is {} which will \
        influence the tone of my creation.",
        mood.to_lowercase()
    )
}

/// Generate a poem based on theme and mood
fn generate_poem(theme: &str, mood: &str) -> String {
    match theme {
        "love" => {
            if mood == "Optimistic" {
                "In the garden of my heart, you bloom,\nA flower bright that chases gloom.\nYour laughter dances on the breeze,\nAnd brings my restless soul to ease.\n\nWith every sunrise, love grows strong,\nIn your embrace, I belong.\nTwo hearts that beat as one, so true,\nMy world begins and ends with you."
            } else {
                "Love whispers soft in twilight's glow,\nA gentle touch, a warmth I know.\nThrough quiet moments, side by side,\nIn you, my heart will always hide.\n\nNo grand gestures, just your hand,\nTogether, here we gently stand.\nIn simple ways, our love takes root,\nA tender bond, forever absolute."
            }
        },
        "nature" => {
            "The forest breathes with ancient song,\nWhere moss and fern have grown so long.\nSunlight filters through the trees,\nAnd dances with the morning breeze.\n\nThe river flows with stories old,\nOf secrets that the stones have told.\nIn nature's arms, I find my peace,\nWhere all my worries gently cease."
        },
        "time" => {
            "Time flows like water through my hands,\nEach moment slips like shifting sands.\nThe past is but a fading dream,\nThe future's not quite what it seems.\n\nBut in this breath, this present now,\nI find the strength to make my vow:\nTo live each second, full and true,\nAnd cherish all that time brings through."
        },
        "hope" => {
            "When darkness falls and shadows creep,\nAnd worries rob me of my sleep,\nI look beyond the clouded sky,\nAnd see the dawn is drawing nigh.\n\nFor hope is like a candle's flame,\nThat burns despite the wind and rain.\nIt whispers soft, 'Hold on, be strong,\nThe night is dark, but not for long.'"
        },
        _ => {
            "Life unfolds in mystery,\nA tapestry of history.\nEach thread we weave with choice and chance,\nIn this eternal, cosmic dance.\n\nThrough joy and sorrow, loss and gain,\nThrough sunshine bright and falling rain,\nWe learn, we grow, we find our way,\nAnd greet the promise of each day."
        }
    }.to_string()
}

/// Generate a short story
fn generate_story(mood: &str) -> String {
    if mood == "Optimistic" {
        "Once upon a time, in a village nestled between rolling hills, there lived a young dreamer named Aria. \
        Every morning, she would climb to the highest hill and watch the sunrise, imagining all the adventures \
        that awaited her beyond the horizon.\n\nOne day, she found a mysterious map tucked inside an old book. \
        The map showed a path to a hidden garden where, legend said, wishes came true. Without hesitation, \
        Aria packed her bag and set off on her journey.\n\nThe path was long and winding, but Aria's spirit never \
        wavered. She helped travelers along the way, shared her food with the hungry, and sang songs to lift \
        the spirits of the weary. And when she finally reached the garden, she discovered something wonderful: \
        the garden was real, but the true magic wasn't in the place—it was in the journey itself, and in the \
        kindness she had shared along the way.\n\nAria returned home not with a granted wish, but with something \
        far more valuable: the knowledge that she could create her own magic through compassion and courage.".to_string()
    } else {
        "In a quiet town where time moved slowly, there lived an old clockmaker named Thomas. His shop was filled \
        with timepieces of every kind, each one ticking away the moments of life.\n\nOne evening, a young woman \
        entered his shop carrying a broken pocket watch. 'Can you fix it?' she asked. 'It belonged to my grandfather.'\n\n\
        Thomas examined the watch carefully. It was old, worn, and seemed beyond repair. But he saw the hope in \
        her eyes and said, 'I'll try.'\n\nFor weeks, Thomas worked on the watch, carefully replacing each tiny gear, \
        cleaning each delicate spring. And finally, one morning, the watch began to tick again.\n\nWhen the young \
        woman returned, tears filled her eyes as she heard the familiar sound. 'Thank you,' she whispered. 'You've \
        given me back a piece of my grandfather.'\n\nThomas smiled. He had learned long ago that fixing watches wasn't \
        just about gears and springs—it was about preserving memories and keeping love alive.".to_string()
    }
}

/// Generate contextual fallback response
fn generate_contextual_fallback(
    user_input: &str,
    inference_result: &InferenceResult,
    mood: &str,
) -> String {
    let confidence = inference_result.confidence;
    
    format!(
        "I understand you're asking about '{}'. I'm processing this with {:.1}% confidence in a {} mood. \
        Could you provide more details or rephrase your question? I'm here to help!",
        user_input,
        confidence * 100.0,
        mood.to_lowercase()
    )
}

/// Retrieve relevant knowledge from semantic memory
fn retrieve_relevant_knowledge(query: &str, semantic_memory: &SemanticMemory) -> Vec<String> {
    let mut knowledge = Vec::new();
    let query_lower = query.to_lowercase();
    
    // First try searching the full query
    if let Ok(facts) = semantic_memory.search_by_concept(&query_lower, 3) {
        for fact in facts {
            knowledge.push(fact.content.clone());
        }
    }
    
    // If no results, try individual words
    if knowledge.is_empty() {
        let concepts: Vec<&str> = query_lower.split_whitespace()
            .filter(|w| w.len() > 2)  // Changed from 3 to 2
            .collect();
        
        for concept in concepts.iter().take(3) {
            if let Ok(facts) = semantic_memory.search_by_concept(concept, 2) {
                for fact in facts {
                    if !knowledge.contains(&fact.content) {
                        knowledge.push(fact.content.clone());
                    }
                }
            }
        }
    }
    
    knowledge
}

/// Generate response from thought vector and knowledge base
fn generate_from_thought_and_knowledge(
    user_input: &str,
    thought: &ThoughtState,
    knowledge: &[String],
    confidence: f64,
    mood: &str,
) -> String {
    // Analyze thought vector to determine response characteristics
    let thought_magnitude = thought.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
    let thought_complexity = thought.vector.iter().filter(|&&x| x.abs() > 0.1).count();
    
    // Build response based on neural activation patterns
    let mut response = String::new();
    
    // Use knowledge if available
    if !knowledge.is_empty() {
        response.push_str(&knowledge.join(" "));
        response.push_str(" ");
    }
    
    // Add context based on thought activation
    if thought_complexity > 50 {
        response.push_str("This is a complex topic that involves multiple interconnected concepts. ");
    }
    
    // Add confidence-based qualifier
    if confidence > 0.8 {
        response.push_str("I'm quite confident about this. ");
    } else if confidence < 0.6 {
        response.push_str("I'm still learning about this area. ");
    }
    
    // Add mood-influenced perspective
    match mood {
        "Optimistic" => response.push_str("I'm excited to explore this with you! "),
        "Curious" => response.push_str("This is fascinating to think about. "),
        "Anxious" => response.push_str("Let me carefully consider this. "),
        _ => {}
    }
    
    // If no knowledge found, generate from thought patterns
    if knowledge.is_empty() {
        response = generate_from_thought_patterns(user_input, thought, confidence, mood);
    }
    
    response.trim().to_string()
}

/// Generate response purely from neural thought patterns
fn generate_from_thought_patterns(
    user_input: &str,
    thought: &ThoughtState,
    confidence: f64,
    mood: &str,
) -> String {
    let input_lower = user_input.to_lowercase();
    
    // Analyze thought vector dimensions for semantic understanding
    let dominant_dimensions: Vec<(usize, f64)> = thought.vector.iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .filter(|(_, v)| *v > 0.15)
        .collect();
    
    let mut response_parts = Vec::new();
    
    // Generate response based on neural activation patterns
    if dominant_dimensions.len() > 60 {
        response_parts.push("This involves multiple interconnected concepts that I'm processing through my neural network.".to_string());
    }
    
    // Add mood-influenced opening
    let opening = match mood {
        "Optimistic" => "I'm excited to help with this! ",
        "Curious" => "This is an interesting question. ",
        "Neutral" => "",
        _ => "Let me think about this carefully. ",
    };
    
    if !opening.is_empty() {
        response_parts.push(opening.to_string());
    }
    
    // Generate content based on confidence and thought complexity
    if confidence > 0.75 {
        response_parts.push(format!(
            "Based on my understanding (confidence: {:.1}%), I can address your question about '{}'.",
            confidence * 100.0,
            user_input
        ));
    } else {
        response_parts.push(format!(
            "I'm processing your question about '{}' with {:.1}% confidence. I'm still learning in this area.",
            user_input,
            confidence * 100.0
        ));
    }
    
    // Add thought-driven insights
    if thought.vector.iter().any(|&x| x > 0.2) {
        response_parts.push("My neural network shows strong activation in areas related to this topic.".to_string());
    }
    
    // Encourage interaction
    response_parts.push("Could you provide more context or specific aspects you'd like me to focus on?".to_string());
    
    response_parts.join(" ")
}

/// Feedback request structure
#[derive(Debug, Deserialize)]
pub struct FeedbackRequest {
    pub user_message: String,
    pub alen_response: String,
    pub feedback_type: String,  // "positive" or "negative"
    pub improvement_suggestion: Option<String>,
    pub timestamp: String,
}

/// Submit user feedback for learning
pub async fn submit_feedback(
    State(state): State<Arc<AppState>>,
    Json(req): Json<FeedbackRequest>,
) -> impl IntoResponse {
    // Log feedback to file for analysis
    let feedback_log = format!(
        "=== Feedback {} ===\nTimestamp: {}\nUser: {}\nALEN: {}\nType: {}\nSuggestion: {}\n\n",
        if req.feedback_type == "positive" { "✅" } else { "❌" },
        req.timestamp,
        req.user_message,
        req.alen_response,
        req.feedback_type,
        req.improvement_suggestion.as_deref().unwrap_or("None")
    );
    
    // Append to feedback log file
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("storage/feedback.log")
    {
        let _ = writeln!(file, "{}", feedback_log);
    }
    
    // If negative feedback with suggestion, add to semantic memory for learning
    if req.feedback_type == "negative" {
        if let Some(suggestion) = req.improvement_suggestion {
            if !suggestion.is_empty() {
                let mut engine = state.engine.lock().await;
                
                // Store the improved response in semantic memory
                let fact = SemanticFact {
                    id: Uuid::new_v4().to_string(),
                    concept: req.user_message.clone(),
                    content: suggestion.clone(),
                    embedding: vec![0.0; state.config.dimension],
                    confidence: 0.9,
                    reinforcement_count: 1,
                    last_accessed: Utc::now(),
                    source: Some("user_feedback".to_string()),
                    category: Some("improvement".to_string()),
                    related_concepts: vec![],
                };
                
                let _ = engine.semantic_memory.store(&fact);
            }
        }
    }
    
    Json(serde_json::json!({
        "success": true,
        "message": "Feedback received and will be used for learning",
        "feedback_type": req.feedback_type
    }))
}
