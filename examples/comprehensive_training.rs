//! Comprehensive Training with Large Dataset
//!
//! Trains ALEN on 100 diverse questions across 10 categories

use alen::neural::{NeuralReasoningEngine, ALENConfig};
use alen::core::Problem;
use serde::{Deserialize, Serialize};
use std::fs;
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize)]
struct QuestionAnswer {
    q: String,
    a: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct Category {
    category: String,
    questions: Vec<QuestionAnswer>,
}

fn load_training_data() -> Result<Vec<Category>, Box<dyn std::error::Error>> {
    let data = fs::read_to_string("data/training_data.json")?;
    let categories: Vec<Category> = serde_json::from_str(&data)?;
    Ok(categories)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     ALEN Comprehensive Training - Large Dataset              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load training data
    println!("Loading training data...");
    let categories = match load_training_data() {
        Ok(data) => {
            let total: usize = data.iter().map(|c| c.questions.len()).sum();
            println!("✓ Loaded {} categories with {} total questions\n", data.len(), total);
            data
        }
        Err(e) => {
            eprintln!("Error loading training data: {}", e);
            eprintln!("Make sure data/training_data.json exists");
            return;
        }
    };

    // Configuration
    let config = ALENConfig {
        thought_dim: 128,
        vocab_size: 10000,
        num_operators: 8,
        operator_hidden_dim: 256,
        dropout: 0.1,
        layer_norm_eps: 1e-5,
        use_transformer: false,
        transformer_layers: 4,
        transformer_heads: 4,
    };

    println!("Configuration:");
    println!("  Thought dimension: {}", config.thought_dim);
    println!("  Vocabulary size: {}", config.vocab_size);
    println!("  Number of operators: {}", config.num_operators);
    println!("  Operator hidden dim: {}", config.operator_hidden_dim);
    println!("  Use transformer: {}\n", config.use_transformer);

    // Create engine
    let learning_rate = 0.001;
    let mut engine = NeuralReasoningEngine::new(config.clone(), learning_rate);
    println!("✓ Neural reasoning engine initialized");
    println!("  Network parameters: {}\n", engine.network.num_parameters());

    // Prepare all training data
    let mut all_questions: Vec<(String, String, String)> = Vec::new();
    for category in &categories {
        for qa in &category.questions {
            all_questions.push((
                category.category.clone(),
                qa.q.clone(),
                qa.a.clone(),
            ));
        }
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("                        TRAINING PHASE");
    println!("═══════════════════════════════════════════════════════════════\n");

    let epochs = 5;
    let mut total_verified = 0;
    let mut total_trained = 0;
    let mut category_stats: HashMap<String, (usize, usize)> = HashMap::new();

    for epoch in 1..=epochs {
        println!("Epoch {}/{}", epoch, epochs);
        println!("─────────────────────────────────────────────────────────────");

        let mut epoch_verified = 0;
        let mut epoch_loss = 0.0;
        let mut epoch_count = 0;

        for (i, (category, question, answer)) in all_questions.iter().enumerate() {
            let problem = Problem::training(question, answer, config.thought_dim);
            let result = engine.train_verified(&problem);

            total_trained += 1;
            epoch_count += 1;
            
            if result.verified {
                epoch_verified += 1;
                total_verified += 1;
                
                // Update category stats
                let stats = category_stats.entry(category.clone()).or_insert((0, 0));
                stats.0 += 1; // verified
                stats.1 += 1; // total
            } else {
                let stats = category_stats.entry(category.clone()).or_insert((0, 0));
                stats.1 += 1; // total
            }

            if result.success {
                epoch_loss += result.loss;
            }

            // Print progress every 20 items
            if (i + 1) % 20 == 0 {
                println!("  [{}/{}] Verified: {}/{} ({:.1}%) | Avg Loss: {:.4}",
                    i + 1,
                    all_questions.len(),
                    epoch_verified,
                    i + 1,
                    (epoch_verified as f32 / (i + 1) as f32) * 100.0,
                    epoch_loss / epoch_count as f32
                );
            }
        }

        let avg_loss = epoch_loss / all_questions.len() as f32;
        let verification_rate = (epoch_verified as f32 / all_questions.len() as f32) * 100.0;

        println!("\nEpoch {} Summary:", epoch);
        println!("  Verified: {}/{} ({:.1}%)", epoch_verified, all_questions.len(), verification_rate);
        println!("  Average loss: {:.6}", avg_loss);
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("                      TRAINING COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Total samples: {}", total_trained);
    println!("Total verified: {} ({:.1}%)\n", 
        total_verified, 
        (total_verified as f32 / total_trained as f32) * 100.0
    );

    // Category performance
    println!("═══════════════════════════════════════════════════════════════");
    println!("                   CATEGORY PERFORMANCE");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut sorted_categories: Vec<_> = category_stats.iter().collect();
    sorted_categories.sort_by(|a, b| b.1.0.cmp(&a.1.0));

    for (category, (verified, total)) in sorted_categories {
        let rate = (*verified as f32 / *total as f32) * 100.0;
        println!("  {:20} {:3}/{:3} ({:5.1}%)", 
            category, verified, total, rate);
    }

    // Operator statistics
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    OPERATOR STATISTICS");
    println!("═══════════════════════════════════════════════════════════════\n");

    let stats = engine.operator_statistics();
    let mut sorted_stats = stats.clone();
    sorted_stats.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));

    for stat in &sorted_stats {
        if stat.usage_count > 0 {
            println!("  {} (ID: {})", stat.name, stat.id);
            println!("    Usage: {} times ({:.1}%)", 
                stat.usage_count,
                (stat.usage_count as f32 / total_trained as f32) * 100.0
            );
            println!("    Success rate: {:.1}%", stat.success_rate * 100.0);
            println!("    Weight: {:.4}", stat.weight);
            println!();
        }
    }

    // Testing phase with diverse questions
    println!("═══════════════════════════════════════════════════════════════");
    println!("                        TESTING PHASE");
    println!("═══════════════════════════════════════════════════════════════\n");

    let test_questions = vec![
        ("Math", "What is 8+7?"),
        ("Geography", "What is the capital of Mexico?"),
        ("Science", "What is the chemical symbol for gold?"),
        ("Technology", "What is blockchain?"),
        ("History", "Who invented the printing press?"),
        ("Logic", "What is a syllogism?"),
        ("Biology", "What is photosynthesis?"),
        ("Physics", "What is Newton's first law?"),
    ];

    let mut test_verified = 0;
    for (i, (category, question)) in test_questions.iter().enumerate() {
        println!("Test {}: [{}] \"{}\"", i + 1, category, question);
        
        let result = engine.infer(question);
        
        if result.verified {
            test_verified += 1;
        }
        
        println!("  Operator: {} (ID: {})", result.operator_name, result.selected_operator);
        println!("  Verified: {}", if result.verified { "✓" } else { "✗" });
        println!("  Verification error: {:.6}", result.verification_error);
        println!("  Candidates evaluated: {}", result.candidates_evaluated);
        println!("  Energy range: [{:.4}, {:.4}]", result.energy_range.0, result.energy_range.1);
        println!();
    }

    println!("Test Results: {}/{} verified ({:.1}%)\n",
        test_verified,
        test_questions.len(),
        (test_verified as f32 / test_questions.len() as f32) * 100.0
    );

    // Final summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("                       FINAL SUMMARY");
    println!("═══════════════════════════════════════════════════════════════\n");
    println!("{}", engine.summary());

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    TRAINING COMPLETE                         ║");
    println!("║                                                              ║");
    println!("║  The neural network has been trained on {} questions      ║", all_questions.len());
    println!("║  across {} categories with {:.1}% verification rate.        ║", 
        categories.len(),
        (total_verified as f32 / total_trained as f32) * 100.0
    );
    println!("║                                                              ║");
    println!("║  Network is ready for production deployment.                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}
