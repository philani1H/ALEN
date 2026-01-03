//! Train Master Neural System with Full Persistence
//!
//! Features:
//! - Loads all 554 training examples from JSON
//! - Trains with database persistence
//! - Saves checkpoints every 100 steps
//! - Can resume from previous training
//! - Shows comprehensive statistics

use alen::neural::{MasterNeuralSystem, MasterSystemConfig};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Serialize)]
struct TrainingData {
    version: String,
    description: String,
    total_examples: usize,
    examples: Vec<TrainingExample>,
}

#[derive(Debug, Deserialize, Serialize)]
struct TrainingExample {
    id: usize,
    input: String,
    target: String,
    difficulty: String,
}

fn main() {
    let sep = "=".repeat(80);
    println!("{}", sep);
    println!("  MASTER NEURAL SYSTEM - COMPREHENSIVE TRAINING WITH PERSISTENCE");
    println!("  All 554 Examples + Database Persistence");
    println!("{}", sep);

    // Load training data
    println!("\n📚 Loading training data...");
    let training_path = PathBuf::from("training_data/master_training.json");

    if !training_path.exists() {
        eprintln!("❌ Training data not found at {:?}", training_path);
        eprintln!("   Run: python3 train_master_system.py");
        return;
    }

    let file = File::open(&training_path).expect("Failed to open training data");
    let reader = BufReader::new(file);
    let training_data: TrainingData = serde_json::from_reader(reader)
        .expect("Failed to parse training data");

    println!("✅ Loaded {} examples", training_data.total_examples);
    println!("   Version: {}", training_data.version);
    println!("   Description: {}", training_data.description);

    // Count by difficulty
    let easy = training_data.examples.iter().filter(|e| e.difficulty == "easy").count();
    let medium = training_data.examples.iter().filter(|e| e.difficulty == "medium").count();
    let hard = training_data.examples.iter().filter(|e| e.difficulty == "hard").count();
    println!("   Easy: {}, Medium: {}, Hard: {}", easy, medium, hard);

    // Create system with persistence enabled
    println!("\n🔧 Configuring Master Neural System...");
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
        // PERSISTENCE ENABLED!
        enable_persistence: true,
        db_path: Some(PathBuf::from("./data/alen_neural.db")),
        checkpoint_interval: 100,  // Save every 100 steps
    };

    println!("   Thought dimension: {}", config.thought_dim);
    println!("   Controller LR: {} (governance)", config.controller_lr);
    println!("   Core Model LR: {} (learning)", config.core_model_lr);
    println!("   Database: {:?}", config.db_path);
    println!("   Checkpoint interval: {} steps", config.checkpoint_interval);

    let mut system = MasterNeuralSystem::new(config);

    // Show initial state
    let initial_stats = system.get_stats();
    println!("\n📊 Initial State:");
    println!("   Total training steps: {}", initial_stats.total_training_steps);
    println!("   Controller updates: {}", initial_stats.controller_updates);
    println!("   Core model updates: {}", initial_stats.core_model_updates);
    println!("   Average confidence: {:.2}%", initial_stats.avg_confidence * 100.0);
    println!("   Total memories in DB: {}", system.get_total_memories());

    // Training
    println!("\n{}", sep);
    println!("TRAINING ON ALL {} EXAMPLES", training_data.total_examples);
    println!("{}", sep);

    let start_time = std::time::Instant::now();
    let mut total_loss = 0.0;
    let mut difficulty_stats: std::collections::HashMap<String, (usize, f64)> =
        std::collections::HashMap::new();

    for (i, example) in training_data.examples.iter().enumerate() {
        let metrics = system.train_step(&example.input, &example.target);

        total_loss += metrics.total_loss;

        // Update difficulty stats
        let entry = difficulty_stats.entry(example.difficulty.clone())
            .or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += metrics.confidence;

        // Print progress every 50 examples
        if (i + 1) % 50 == 0 {
            let avg_loss = total_loss / (i + 1) as f64;
            println!("\n📈 Progress: {}/{} examples", i + 1, training_data.total_examples);
            println!("   Average loss: {:.4}", avg_loss);
            println!("   Recent confidence: {:.2}%", metrics.confidence * 100.0);
            println!("   Memories in DB: {}", system.get_total_memories());
        }

        // Show sample every 100 examples
        if (i + 1) % 100 == 0 {
            println!("\n   Sample [{}]:", example.difficulty);
            println!("   Q: {}...", &example.input[..example.input.len().min(60)]);
            println!("   A: {}...", &example.target[..example.target.len().min(60)]);
        }
    }

    let elapsed = start_time.elapsed();

    // Final statistics
    println!("\n{}", sep);
    println!("TRAINING COMPLETE!");
    println!("{}", sep);

    let final_stats = system.get_stats();
    println!("\n⏱️  Training Time: {:.2}s", elapsed.as_secs_f64());
    println!("   Examples/second: {:.2}",
             training_data.total_examples as f64 / elapsed.as_secs_f64());

    println!("\n📊 Final Statistics:");
    println!("   Total training steps: {}", final_stats.total_training_steps);
    println!("   Controller updates (φ): {}", final_stats.controller_updates);
    println!("   Core model updates (θ): {}", final_stats.core_model_updates);
    println!("   Average confidence: {:.2}%", final_stats.avg_confidence * 100.0);
    println!("   Average perplexity: {:.2}", final_stats.avg_perplexity);
    println!("   Average loss: {:.4}", total_loss / training_data.total_examples as f64);

    println!("\n📈 Learning Rates:");
    println!("   Controller LR (φ): {:.6} (governance)", final_stats.controller_lr);
    println!("   Core Model LR (θ): {:.6} (learning)", final_stats.core_lr);

    println!("\n🎯 Performance by Difficulty:");
    for (difficulty, (count, total_conf)) in difficulty_stats.iter() {
        let avg_conf = (total_conf / *count as f64) * 100.0;
        println!("   {}: {} examples, {:.2}% avg confidence",
                 difficulty, count, avg_conf);
    }

    println!("\n💾 Persistence Status:");
    if let Some(db_path) = system.get_db_path() {
        println!("   Database: {}", db_path.display());
        println!("   Total memories stored: {}", system.get_total_memories());

        if let Ok(metadata) = std::fs::metadata(db_path) {
            println!("   Database size: {:.2} MB", metadata.len() as f64 / 1_048_576.0);
        }
    }

    // Save final checkpoint
    println!("\n💾 Saving final checkpoint...");
    if let Err(e) = system.save_checkpoint("training_complete") {
        eprintln!("⚠️  Failed to save final checkpoint: {}", e);
    }

    // Test inference on some examples
    println!("\n{}", sep);
    println!("TESTING INFERENCE");
    println!("{}", sep);

    let test_prompts = vec![
        "What is 2 + 2?",
        "Explain neural networks",
        "What is the meaning of life?",
        "Tell me a short story",
        "What is creativity?",
    ];

    for prompt in test_prompts {
        println!("\n🔮 Prompt: {}", prompt);
        let response = system.forward(prompt);
        println!("   Response: {}", response.response);
        println!("   Confidence: {:.2}%", response.confidence * 100.0);
        println!("   Reasoning depth: {}", response.controls.reasoning_depth);
        println!("   Creativity: {:.2}", response.controls.style.creativity);
        println!("   Action: {:?}", response.controls.action);
    }

    println!("\n{}", sep);
    println!("✅ TRAINING SESSION COMPLETE!");
    println!("{}", sep);
    println!("\nAll patterns and memories have been saved to the database.");
    println!("You can now:");
    println!("  1. Deploy this model to production");
    println!("  2. Resume training later with the same database");
    println!("  3. Run inference on new inputs");
    println!("\nTo resume training:");
    println!("  cargo run --release --example train_with_persistence");
    println!("\nTo test inference:");
    println!("  cargo run --release --example test_master_system");
    println!("{}", sep);
}
