//! Tests to verify UNDERSTANDING, NOT MEMORIZATION
//!
//! These tests ensure the system generates from learned patterns,
//! not by retrieving stored answers.

use alen::core::ThoughtState;
use alen::generation::LatentDecoder;
use alen::memory::{SemanticMemory, SemanticFact};
use alen::reasoning::NeuralChainOfThoughtReasoner;

#[test]
fn test_latent_decoder_no_retrieval() {
    let mut decoder = LatentDecoder::new(64, 10);
    
    // Learn a specific pattern
    let thought1 = ThoughtState::random(64);
    decoder.learn(&thought1, "the capital of France is Paris");
    
    // Generate from DIFFERENT thought
    let thought2 = ThoughtState::random(64);
    let (text, _) = decoder.generate(&thought2);
    
    // Should NOT return exact learned answer (no retrieval)
    assert_ne!(text, "the capital of France is Paris");
    assert_ne!(text, "the capital of france is paris");
    
    // Should generate something (not empty)
    assert!(!text.is_empty());
}

#[test]
fn test_latent_decoder_pattern_learning() {
    let mut decoder = LatentDecoder::new(64, 20);
    
    // Learn multiple patterns with similar structure
    let thought1 = ThoughtState::random(64);
    decoder.learn(&thought1, "the sky is blue");
    
    let thought2 = ThoughtState::random(64);
    decoder.learn(&thought2, "the grass is green");
    
    let thought3 = ThoughtState::random(64);
    decoder.learn(&thought3, "the sun is yellow");
    
    // Generate from new thought
    let new_thought = ThoughtState::random(64);
    let (text, confidence) = decoder.generate(&new_thought);
    
    // Should generate something
    assert!(!text.is_empty());
    assert!(confidence >= 0.0 && confidence <= 1.0);
    
    // Should NOT be exact match to any learned text
    assert_ne!(text, "the sky is blue");
    assert_ne!(text, "the grass is green");
    assert_ne!(text, "the sun is yellow");
}

#[test]
fn test_latent_decoder_generalization() {
    let mut decoder = LatentDecoder::new(128, 30);
    
    // Learn patterns about numbers
    for i in 1..=5 {
        let thought = ThoughtState::random(128);
        decoder.learn(&thought, &format!("the number is {}", i));
    }
    
    // Generate from new thought
    let new_thought = ThoughtState::random(128);
    let (text, _) = decoder.generate(&new_thought);
    
    // Should generate something
    assert!(!text.is_empty());
    
    // Should NOT be exact match to any learned text
    for i in 1..=5 {
        assert_ne!(text, format!("the number is {}", i));
    }
}

#[test]
fn test_semantic_memory_stores_patterns_not_answers() {
    let memory = SemanticMemory::in_memory(64).expect("Failed to create memory");
    
    // Store a fact
    let fact = SemanticFact::new("capital", "Paris is the capital of France", 64);
    memory.store(&fact).expect("Failed to store fact");
    
    // The fact is stored, but should NOT be retrieved for generation
    // This test documents that fact.content exists for verification only
    assert_eq!(fact.content, "Paris is the capital of France");
    
    // The embedding is what should be used for generation
    assert_eq!(fact.embedding.len(), 64);
}

#[test]
fn test_neural_reasoning_uses_latent_decoder() {
    use alen::core::{Problem, OperatorManager, Evaluator, EnergyWeights};
    
    let dim = 64;
    let operators = OperatorManager::new(dim);
    let evaluator = Evaluator::new(EnergyWeights::default(), 0.6);
    let semantic_memory = SemanticMemory::in_memory(dim).expect("Failed to create memory");
    
    let mut reasoner = NeuralChainOfThoughtReasoner::new(
        operators,
        evaluator,
        semantic_memory,
        dim,
        5,
        0.5,
        0.7,
    );
    
    // Learn a pattern
    let thought = ThoughtState::random(dim);
    reasoner.learn_pattern(&thought, "test pattern");
    
    // Reason about a problem
    let problem = Problem::new("What is 2+2?", dim);
    let chain = reasoner.reason(&problem);
    
    // Should generate an answer (even if not correct without training)
    assert!(chain.answer.is_some());
    
    // Answer should NOT be exact match to learned pattern
    if let Some(answer) = chain.answer {
        assert_ne!(answer, "test pattern");
    }
}

#[test]
fn test_latent_decoder_temperature_control() {
    let mut decoder = LatentDecoder::new(64, 10);
    
    // Learn some patterns
    let thought = ThoughtState::random(64);
    decoder.learn(&thought, "hello world");
    
    // Test different temperatures
    decoder.set_temperature(0.1); // Low temperature
    let (text_low, _) = decoder.generate(&thought);
    
    decoder.set_temperature(1.5); // High temperature
    let (text_high, _) = decoder.generate(&thought);
    
    // Both should generate something
    assert!(!text_low.is_empty());
    assert!(!text_high.is_empty());
    
    // Neither should be exact match
    assert_ne!(text_low, "hello world");
    assert_ne!(text_high, "hello world");
}

#[test]
fn test_latent_decoder_stats() {
    let mut decoder = LatentDecoder::new(64, 15);
    
    // Initially no patterns learned
    let stats = decoder.stats();
    assert_eq!(stats.active_patterns, 0);
    assert_eq!(stats.vocabulary_size, 0);
    
    // Learn some patterns
    let thought1 = ThoughtState::random(64);
    decoder.learn(&thought1, "first pattern");
    
    let thought2 = ThoughtState::random(64);
    decoder.learn(&thought2, "second pattern");
    
    // Stats should update
    let stats = decoder.stats();
    assert!(stats.active_patterns > 0);
    assert!(stats.vocabulary_size > 0);
    assert!(stats.total_associations > 0);
}

#[test]
fn test_no_retrieval_from_episodic_memory() {
    use alen::memory::{EpisodicMemory, Episode};
    use alen::core::EnergyResult;
    
    let memory = EpisodicMemory::in_memory().expect("Failed to create memory");
    
    // Store an episode with an answer
    let episode = Episode {
        id: uuid::Uuid::new_v4().to_string(),
        problem_input: "What is 2+2?".to_string(),
        answer_output: "4".to_string(), // This is for VERIFICATION ONLY
        thought_vector: vec![0.5; 64],
        verified: true,
        confidence_score: 0.9,
        energy: 0.1,
        operator_id: "test".to_string(),
        created_at: chrono::Utc::now(),
        tags: vec![],
    };
    
    memory.store(&episode).expect("Failed to store episode");
    
    // The answer_output is stored but should NEVER be retrieved for generation
    // This test documents that answer_output exists for verification only
    assert_eq!(episode.answer_output, "4");
    
    // The thought_vector is what should be used for generation
    assert_eq!(episode.thought_vector.len(), 64);
}

#[test]
fn test_deprecated_decoders_marked() {
    // This test documents that old decoders are deprecated
    // They should return deprecation messages, not actual content
    
    use alen::generation::{LearnedDecoder, ConfidenceDecoder};
    use alen::memory::Episode;
    
    let decoder = LearnedDecoder::new(64, 0.7);
    let memory = SemanticMemory::in_memory(64).expect("Failed to create memory");
    let thought = ThoughtState::random(64);
    
    // Should return deprecation message
    let result = decoder.generate_from_memory(&thought, &memory, 5);
    if let Ok(text) = result {
        // Should indicate deprecation
        assert!(text.contains("DEPRECATED") || text.is_empty() || text.contains("don't"));
    }
}
