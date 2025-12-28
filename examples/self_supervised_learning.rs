//! Self-Supervised Learning Demonstration
//!
//! Shows how ALEN learns through:
//! - Self-prediction
//! - Surprise minimization
//! - Curiosity-driven exploration
//! - Persistent memory storage

use alen::{
    neural::{NeuralReasoningEngine, ALENConfig},
    memory::{SemanticStore, SemanticEntry},
    control::{CuriosityEngine, SelfSupervisedLoop},
    core::{Problem, ThoughtState},
};
use std::path::Path;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         ALEN Self-Supervised Learning System                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let config = ALENConfig::small();
    let storage_dir = "storage/semantic";
    
    println!("Configuration:");
    println!("  Thought dimension: {}", config.thought_dim);
    println!("  Storage directory: {}", storage_dir);
    println!();

    // Initialize components
    println!("Initializing systems...");
    let mut neural_engine = NeuralReasoningEngine::new(config.clone(), 0.001);
    let mut semantic_store = SemanticStore::new(storage_dir, 1000, 0.7)
        .expect("Failed to create semantic store");
    let mut curiosity = CuriosityEngine::new(config.thought_dim);
    let mut self_supervised = SelfSupervisedLoop::new(config.thought_dim, 5);
    
    println!("✓ Neural engine initialized");
    println!("✓ Semantic store initialized ({} existing entries)", semantic_store.len());
    println!("✓ Curiosity engine initialized");
    println!("✓ Self-supervised loop initialized\n");

    println!("═══════════════════════════════════════════════════════════════");
    println!("                  PHASE 1: SUPERVISED LEARNING");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Train on some examples
    let training_examples = vec![
        ("What is 2+2?", "4"),
        ("What is the capital of France?", "Paris"),
        ("What is gravity?", "A force that attracts objects with mass"),
    ];

    for (i, (question, answer)) in training_examples.iter().enumerate() {
        println!("Training example {}: \"{}\"", i + 1, question);
        
        let problem = Problem::training(question, answer, config.thought_dim);
        let result = neural_engine.train_verified(&problem);
        
        println!("  Verified: {}", result.verified);
        println!("  Loss: {:.4}", result.loss);
        println!("  Operator: {}", result.selected_operator);
        
        // If verified, store in semantic memory
        if result.verified {
            let inference_result = neural_engine.infer(question);
            let entry = SemanticEntry::new(
                inference_result.thought_vector.iter().map(|&x| x as f32).collect(),
                question.to_string(),
                1.0 - inference_result.verification_error,
                inference_result.selected_operator,
                inference_result.operator_name.clone(),
                inference_result.verification_error,
            );
            
            match semantic_store.insert(entry) {
                Ok(true) => println!("  ✓ Stored in semantic memory"),
                Ok(false) => println!("  ⚠ Not stored (duplicate or low confidence)"),
                Err(e) => println!("  ✗ Storage error: {}", e),
            }
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("                PHASE 2: SELF-SUPERVISED LEARNING");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Running self-supervised learning cycles...\n");

    let current_state = ThoughtState::new(config.thought_dim);
    
    for cycle in 1..=3 {
        println!("Cycle {}:", cycle);
        println!("─────────────────────────────────────────────────────────────");
        
        // Generate self-directed questions
        let results = self_supervised.run_cycle(&current_state);
        
        for (i, (question, surprise)) in results.iter().enumerate() {
            println!("  Question {}: {}", i + 1, question);
            println!("    Surprise magnitude: {:.4}", surprise.magnitude);
            println!("    Expected: {}", surprise.expected);
            println!("    Significant: {}", surprise.is_significant());
        }
        
        let stats = self_supervised.statistics();
        println!("\n  Cycle statistics:");
        println!("    Total predictions: {}", stats.total_predictions);
        println!("    Average surprise: {:.4}", stats.avg_surprise);
        println!("    Curiosity score: {:.4}", stats.curiosity_score);
        println!("    Should continue: {}", self_supervised.should_continue());
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("                PHASE 3: MEMORY RETRIEVAL");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Test memory retrieval
    println!("Testing semantic memory retrieval...\n");

    let test_queries = vec![
        "What is 2+2?",
        "What is the capital of France?",
        "What is gravity?",
    ];

    for query in test_queries {
        println!("Query: \"{}\"", query);
        
        // Get thought vector for query
        let result = neural_engine.infer(query);
        let query_thought: Vec<f32> = result.thought_vector.iter().map(|&x| x as f32).collect();
        
        // Find similar entries
        let similar = semantic_store.find_similar(&query_thought, 3);
        
        println!("  Found {} similar entries:", similar.len());
        for (i, (similarity, entry)) in similar.iter().enumerate() {
            println!("    {}: {} (similarity: {:.4})", i + 1, entry.concept, similarity);
            println!("       Confidence: {:.4}", entry.confidence);
            println!("       Operator: {}", entry.operator_name);
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("                PHASE 4: CURIOSITY-DRIVEN EXPLORATION");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Testing curiosity-driven exploration...\n");

    for i in 1..=5 {
        let question = curiosity.generate_question();
        println!("Generated question {}: {}", i, question);
        
        // Make prediction
        let prediction = curiosity.predict(&current_state);
        println!("  Prediction confidence: {:.4}", prediction.confidence);
        
        // Simulate observation
        let observation = curiosity.observe(&current_state);
        
        // Compute surprise
        let surprise = curiosity.compute_surprise(&prediction, &observation);
        println!("  Surprise: {:.4}", surprise.magnitude);
        println!("  Expected: {}", surprise.expected);
        println!();
    }

    let curiosity_stats = curiosity.statistics();
    println!("Curiosity statistics:");
    println!("  Total predictions: {}", curiosity_stats.total_predictions);
    println!("  Average surprise: {:.4}", curiosity_stats.avg_surprise);
    println!("  Significant surprises: {}", curiosity_stats.significant_surprises);
    println!("  Curiosity score: {:.4}", curiosity_stats.curiosity_score);
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("                      FINAL STATISTICS");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Semantic memory stats
    let semantic_stats = semantic_store.statistics();
    println!("Semantic Memory:");
    println!("  Total entries: {}", semantic_stats.total_entries);
    println!("  Average confidence: {:.4}", semantic_stats.avg_confidence);
    println!("  Storage size: {} bytes", semantic_stats.storage_size_bytes);
    println!("\n  Operator usage:");
    for (operator, count) in semantic_stats.operator_usage {
        println!("    {}: {} times", operator, count);
    }
    println!();

    // Neural engine stats
    println!("Neural Engine:");
    println!("  {}", neural_engine.summary());
    println!();

    // Curiosity stats
    println!("Curiosity Engine:");
    println!("  Total predictions: {}", curiosity_stats.total_predictions);
    println!("  Total surprises: {}", curiosity_stats.total_surprises);
    println!("  Average surprise: {:.4}", curiosity_stats.avg_surprise);
    println!("  Significant surprises: {}", curiosity_stats.significant_surprises);
    println!("  Current curiosity: {:.4}", curiosity_stats.curiosity_score);
    println!();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              SELF-SUPERVISED LEARNING COMPLETE               ║");
    println!("║                                                              ║");
    println!("║  ALEN has demonstrated:                                      ║");
    println!("║  • Supervised learning with verification                     ║");
    println!("║  • Self-supervised prediction loops                          ║");
    println!("║  • Persistent semantic memory storage                        ║");
    println!("║  • Curiosity-driven exploration                              ║");
    println!("║  • Surprise-based learning signals                           ║");
    println!("║                                                              ║");
    println!("║  Memory persists across runs in: {}                  ║", storage_dir);
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}
