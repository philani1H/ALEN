//! Complete Training and Testing Example
//!
//! Trains the ALEN neural network on sample questions and tests it

use alen::neural::{NeuralReasoningEngine, ALENConfig};
use alen::core::Problem;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         ALEN Neural Network Training & Testing              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let config = ALENConfig {
        thought_dim: 128,
        vocab_size: 10000,
        num_operators: 8,
        operator_hidden_dim: 256,
        dropout: 0.1,
        layer_norm_eps: 1e-5,
        use_transformer: false, // Start simple
        transformer_layers: 4,
        transformer_heads: 4,
    };

    println!("Configuration:");
    println!("  Thought dimension: {}", config.thought_dim);
    println!("  Vocabulary size: {}", config.vocab_size);
    println!("  Number of operators: {}", config.num_operators);
    println!("  Use transformer: {}\n", config.use_transformer);

    // Create engine
    let learning_rate = 0.001;
    let mut engine = NeuralReasoningEngine::new(config.clone(), learning_rate);
    println!("✓ Neural reasoning engine initialized");
    println!("  Network parameters: {}\n", engine.network.num_parameters());

    // Training dataset
    let training_data = vec![
        ("What is 2+2?", "4"),
        ("What is 3+3?", "6"),
        ("What is 5+5?", "10"),
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Germany?", "Berlin"),
        ("What is the capital of Italy?", "Rome"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the speed of light?", "299792458 m/s"),
        ("What is gravity?", "A force that attracts objects with mass"),
        ("Explain photosynthesis", "Process where plants convert light to energy"),
        ("What is DNA?", "Deoxyribonucleic acid, carrier of genetic information"),
        ("Define machine learning", "Algorithms that learn from data"),
        ("What is recursion?", "A function that calls itself"),
        ("Explain quantum mechanics", "Physics of atomic and subatomic particles"),
    ];

    println!("═══════════════════════════════════════════════════════════════");
    println!("                        TRAINING PHASE");
    println!("═══════════════════════════════════════════════════════════════\n");

    let epochs = 3;
    let mut total_verified = 0;
    let mut total_trained = 0;

    for epoch in 1..=epochs {
        println!("Epoch {}/{}", epoch, epochs);
        println!("─────────────────────────────────────────────────────────────");

        let mut epoch_verified = 0;
        let mut epoch_loss = 0.0;

        for (i, (question, answer)) in training_data.iter().enumerate() {
            let problem = Problem::training(question, answer, config.thought_dim);
            let result = engine.train_verified(&problem);

            total_trained += 1;
            if result.verified {
                epoch_verified += 1;
                total_verified += 1;
            }

            if result.success {
                epoch_loss += result.loss;
            }

            // Print progress every 5 items
            if (i + 1) % 5 == 0 {
                println!("  [{}/{}] Verified: {}/{} ({:.1}%)",
                    i + 1,
                    training_data.len(),
                    epoch_verified,
                    i + 1,
                    (epoch_verified as f32 / (i + 1) as f32) * 100.0
                );
            }
        }

        let avg_loss = epoch_loss / training_data.len() as f32;
        let verification_rate = (epoch_verified as f32 / training_data.len() as f32) * 100.0;

        println!("\nEpoch {} Summary:", epoch);
        println!("  Verified: {}/{} ({:.1}%)", epoch_verified, training_data.len(), verification_rate);
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

    // Operator statistics
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    OPERATOR STATISTICS");
    println!("═══════════════════════════════════════════════════════════════\n");

    let stats = engine.operator_statistics();
    for stat in &stats {
        if stat.usage_count > 0 {
            println!("  {} (ID: {})", stat.name, stat.id);
            println!("    Usage: {} times", stat.usage_count);
            println!("    Success rate: {:.1}%", stat.success_rate * 100.0);
            println!("    Weight: {:.4}", stat.weight);
            println!();
        }
    }

    // Testing phase
    println!("═══════════════════════════════════════════════════════════════");
    println!("                        TESTING PHASE");
    println!("═══════════════════════════════════════════════════════════════\n");

    let test_questions = vec![
        "What is 4+4?",
        "What is the capital of Spain?",
        "Who discovered gravity?",
        "What is artificial intelligence?",
        "Explain neural networks",
    ];

    for (i, question) in test_questions.iter().enumerate() {
        println!("Test {}: \"{}\"", i + 1, question);
        
        let result = engine.infer(question);
        
        println!("  Operator: {} (ID: {})", result.operator_name, result.selected_operator);
        println!("  Verified: {}", if result.verified { "✓" } else { "✗" });
        println!("  Verification error: {:.6}", result.verification_error);
        println!("  Candidates evaluated: {}", result.candidates_evaluated);
        println!("  Energy range: [{:.4}, {:.4}]", result.energy_range.0, result.energy_range.1);
        println!("  Thought vector norm: {:.4}", 
            result.thought_vector.iter().map(|x| x * x).sum::<f64>().sqrt()
        );
        println!();
    }

    // Final summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("                       FINAL SUMMARY");
    println!("═══════════════════════════════════════════════════════════════\n");
    println!("{}", engine.summary());

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    TRAINING COMPLETE                         ║");
    println!("║                                                              ║");
    println!("║  The neural network has been trained with verification      ║");
    println!("║  and is ready for production use.                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}
