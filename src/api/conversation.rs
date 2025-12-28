//! Conversation API Module
//!
//! Provides natural conversational interface for human interaction with ALEN.
//! Maintains conversation history, context, and personality.

use super::{AppState, Problem};
use crate::core::ThoughtState;
use crate::learning::feedback_loop::InferenceResult;
use crate::memory::{SemanticFact, Episode, SemanticMemory};
use crate::control::MoodEngine;

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

    // Generate response text using semantic memory and patterns
    let response_text = generate_conversational_response(
        &req.message,
        &result,
        &engine.semantic_memory,
        &engine.mood_engine,
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

/// Generate a conversational response based on input and context
fn generate_conversational_response(
    user_input: &str,
    inference_result: &InferenceResult,
    semantic_memory: &SemanticMemory,
    mood_engine: &MoodEngine,
) -> String {
    let input_lower = user_input.to_lowercase();
    let mood = mood_engine.current_mood();
    let confidence = inference_result.confidence;
    
    // Greeting patterns
    if input_lower.contains("hello") || input_lower.contains("hi ") || input_lower == "hi" {
        return format!("Hello! I'm ALEN, an advanced learning engine. I'm currently feeling {} and ready to help you. How can I assist you today?", mood.as_str().to_lowercase());
    }
    
    if input_lower.contains("how are you") || input_lower.contains("how r u") {
        let mood_desc = match mood.as_str() {
            "Optimistic" => "great! I'm feeling optimistic and energized",
            "Neutral" => "good! I'm in a balanced, neutral state",
            "Stressed" => "a bit stressed, but I'm managing well",
            "Anxious" => "slightly anxious, but focused on helping you",
            _ => "fine, thank you for asking",
        };
        return format!("I'm doing {}! My confidence level is at {:.1}%. How can I help you today?", 
            mood_desc, confidence * 100.0);
    }
    
    if input_lower.contains("what is your name") || input_lower.contains("who are you") {
        return "I'm ALEN (Advanced Learning Engine with Neural understanding). I'm an AI system that learns from interactions and uses emotional intelligence to provide thoughtful responses. I can help with questions, generate images and videos, and have natural conversations. What would you like to know?".to_string();
    }
    
    // Math/formula questions
    if input_lower.contains("quadratic formula") {
        return "The quadratic formula is: $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$ where $a$, $b$, and $c$ are coefficients from the quadratic equation $ax^2 + bx + c = 0$. This formula gives you the roots (solutions) of any quadratic equation. Would you like me to explain how to use it?".to_string();
    }
    
    if input_lower.contains("pythagorean theorem") {
        return "The Pythagorean theorem states: $a^2 + b^2 = c^2$ where $c$ is the hypotenuse (longest side) of a right triangle, and $a$ and $b$ are the other two sides. This fundamental relationship helps us calculate distances and solve geometric problems. 📐".to_string();
    }
    
    if input_lower.contains("einstein") && (input_lower.contains("formula") || input_lower.contains("equation")) {
        return "Einstein's famous mass-energy equivalence formula is: $E = mc^2$ where $E$ is energy, $m$ is mass, and $c$ is the speed of light. This equation shows that mass and energy are interchangeable - a small amount of mass can be converted into a huge amount of energy! ⚡".to_string();
    }
    
    if input_lower.contains("derivative") {
        return "A derivative represents the rate of change of a function. Mathematically: $f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}$. It tells us how quickly something is changing at any given point. For example, if $f(x) = x^2$, then $f'(x) = 2x$. 📊".to_string();
    }
    
    // Neural network questions
    if input_lower.contains("neural network") || input_lower.contains("machine learning") {
        return "Neural networks are computing systems inspired by biological brains 🧠. They consist of interconnected nodes (neurons) organized in layers. Each connection has a weight that's adjusted during training. The network learns patterns by processing examples and adjusting these weights through backpropagation. I myself use neural networks for reasoning and learning! 💡".to_string();
    }
    
    if input_lower.contains("backpropagation") {
        return "Backpropagation is the algorithm neural networks use to learn! Here's how it works:\n\n1. **Forward pass**: Input flows through the network to produce output\n2. **Calculate error**: Compare output to expected result\n3. **Backward pass**: Error propagates back through layers\n4. **Update weights**: Adjust connections to reduce error\n\nIt's like learning from mistakes - the network figures out which connections need adjustment! 🎯".to_string();
    }
    
    // Emoji questions
    if input_lower.contains("emoji") {
        if input_lower.contains("happiness") || input_lower.contains("happy") {
            return "Happiness can be expressed with these emojis: 😊 😃 😄 🎉 ✨ 💖 🌟 representing joy, celebration, and positive emotions! Each one conveys a slightly different shade of happiness.".to_string();
        }
        if input_lower.contains("learning") {
            return "Learning emojis: 📚 🧠 💡 ✏️ 🎓 📖 🔬 representing books, brain power, ideas, writing, education, and discovery! These symbols capture the essence of knowledge and growth.".to_string();
        }
        return "Emojis are a universal language! 😊 They help convey emotions and concepts quickly. I can use emojis naturally in my responses to make communication more expressive and engaging. What kind of emojis would you like to know about?".to_string();
    }
    
    // Image/video generation
    if input_lower.contains("generate") && (input_lower.contains("image") || input_lower.contains("picture")) {
        return "I can generate images from text descriptions! 🖼️ Just tell me what you'd like to see, and I'll create a visual representation. For example, you could ask for 'a sunset over mountains' or 'a neural network visualization'. Try using the Generate Media tab to create images!".to_string();
    }
    
    if input_lower.contains("generate") && input_lower.contains("video") {
        return "I can create videos from your descriptions! 🎬 Videos can show motion like 'ocean waves', 'rotating shapes', or 'flowing water'. Use the Generate Media tab and choose a motion type (linear, circular, or random) for different effects. The video player will let you play, pause, and watch frame by frame!".to_string();
    }
    
    // Capabilities
    if input_lower.contains("what can you do") || input_lower.contains("capabilities") {
        return "I can do many things! 🌟\n\n• **Answer questions** with detailed explanations\n• **Solve math problems** with LaTeX formulas\n• **Generate images** from text descriptions 🖼️\n• **Create videos** with different motion types 🎬\n• **Use emojis** naturally in conversation 😊\n• **Remember context** from our conversation\n• **Show my reasoning** process step-by-step\n• **Express emotions** and moods that affect my responses\n\nWhat would you like to try?".to_string();
    }
    
    // Thank you
    if input_lower.contains("thank") {
        return format!("You're welcome! 😊 I'm happy to help. I'm currently feeling {} and ready for more questions if you have them!", mood.as_str().to_lowercase());
    }
    
    // Try to find relevant information in semantic memory
    if let Ok(facts) = semantic_memory.search_by_concept(&input_lower, 3) {
        if !facts.is_empty() {
            let fact = &facts[0];
            return format!("Based on what I know: {}. My confidence in this response is {:.1}%. Would you like me to elaborate?", 
                fact.content, confidence * 100.0);
        }
    }
    
    // Default intelligent response
    format!(
        "I understand you're asking about '{}'. While I'm processing this with {:.1}% confidence, I'm currently in a {} mood. \
        Could you provide more details or rephrase your question? I'm here to help with:\n\n\
        • Mathematical formulas and explanations 📐\n\
        • Science and physics concepts ⚡\n\
        • Neural networks and AI 🧠\n\
        • Image and video generation 🎨\n\
        • General knowledge questions 📚\n\n\
        What specific aspect interests you?",
        user_input,
        confidence * 100.0,
        mood.as_str().to_lowercase()
    )
}
