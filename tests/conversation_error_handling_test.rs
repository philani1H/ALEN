//! Tests for conversation error handling
//!
//! Ensures that the conversation API handles errors gracefully
//! without panicking.

use alen::api::{AppState, EngineConfig, ReasoningEngine, ConversationManager};
use alen::learning::LearningConfig;
use alen::core::EnergyWeights;
use alen::memory::EmbeddingConfig;
use alen::storage::StorageConfig;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::test]
async fn test_conversation_store_consistency() {
    let config = EngineConfig {
        dimension: 64,
        learning: LearningConfig::default(),
        energy_weights: EnergyWeights::default(),
        embedding: EmbeddingConfig {
            dimension: 64,
            normalize: true,
            vocab_size: 1000,
            use_bpe: false,
        },
        evaluator_confidence_threshold: 0.6,
        evaluator_energy_threshold: 0.5,
        backward_similarity_threshold: 0.7,
        backward_path_threshold: 0.3,
    };

    let engine = ReasoningEngine::new(config.clone()).expect("Failed to create engine");
    let conversation_manager = ConversationManager::new();
    
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        config: config.clone(),
        conversation_manager: conversation_manager.clone(),
        storage: StorageConfig::in_memory(),
    });

    // Test 1: Create a conversation and verify it exists
    let mut store = conversation_manager.store.lock().await;
    let conv_id = store.get_or_create(None, None);
    
    // This should not panic
    let conversation = store.get_mut(&conv_id);
    assert!(conversation.is_some(), "Conversation should exist after creation");
}

#[tokio::test]
async fn test_conversation_retrieval_after_creation() {
    let conversation_manager = ConversationManager::new();
    let mut store = conversation_manager.store.lock().await;
    
    // Create conversation with custom prompt
    let custom_prompt = "Test system prompt".to_string();
    let conv_id = store.get_or_create(None, Some(custom_prompt.clone()));
    
    // Retrieve and verify
    let conversation = store.get_mut(&conv_id);
    assert!(conversation.is_some(), "Should retrieve created conversation");
    
    if let Some(conv) = conversation {
        assert_eq!(conv.metadata.system_prompt, custom_prompt);
        assert_eq!(conv.id, conv_id);
    }
}

#[tokio::test]
async fn test_conversation_with_existing_id() {
    let conversation_manager = ConversationManager::new();
    let mut store = conversation_manager.store.lock().await;
    
    // Create first conversation
    let conv_id_1 = store.get_or_create(None, None);
    
    // Try to get same conversation by ID
    let conv_id_2 = store.get_or_create(Some(conv_id_1.clone()), None);
    
    // Should return same ID
    assert_eq!(conv_id_1, conv_id_2, "Should return existing conversation ID");
    
    // Should be retrievable
    let conversation = store.get_mut(&conv_id_2);
    assert!(conversation.is_some(), "Should retrieve existing conversation");
}

#[test]
fn test_semantic_memory_creation_error_handling() {
    use alen::memory::SemanticMemory;
    
    // Test that we can handle memory creation errors
    let result = SemanticMemory::in_memory(64);
    
    match result {
        Ok(_) => {
            // Success case - memory created
        }
        Err(e) => {
            // Error case - should not panic, just return error
            eprintln!("Memory creation failed (expected in some cases): {}", e);
        }
    }
}

#[tokio::test]
async fn test_conversation_message_addition() {
    use alen::api::conversation::{MessageRole, ConversationStore};
    
    let mut store = ConversationStore::new();
    let conv_id = store.get_or_create(None, None);
    
    let conversation = store.get_mut(&conv_id).expect("Should get conversation");
    
    // Add user message
    conversation.add_message(
        MessageRole::User,
        "Test message".to_string(),
        None,
        None,
    );
    
    assert_eq!(conversation.messages.len(), 1);
    assert_eq!(conversation.messages[0].content, "Test message");
    assert_eq!(conversation.messages[0].role, MessageRole::User);
}

#[tokio::test]
async fn test_conversation_context_building() {
    use alen::api::conversation::{MessageRole, ConversationStore};
    
    let mut store = ConversationStore::new();
    let conv_id = store.get_or_create(None, None);
    
    let conversation = store.get_mut(&conv_id).expect("Should get conversation");
    
    // Add multiple messages
    conversation.add_message(MessageRole::User, "Message 1".to_string(), None, None);
    conversation.add_message(MessageRole::Assistant, "Response 1".to_string(), None, None);
    conversation.add_message(MessageRole::User, "Message 2".to_string(), None, None);
    
    // Get context
    let context = conversation.get_context(2);
    
    // Should contain last 2 messages
    assert!(context.contains("Response 1"));
    assert!(context.contains("Message 2"));
    assert!(!context.contains("Message 1")); // Should not include first message
}

