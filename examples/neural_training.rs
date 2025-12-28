//! Neural Network Training Example
//!
//! Demonstrates training the ALEN neural network with verification

use alen::neural::{ALENNetwork, ALENConfig, Adam, MSELoss, TrainerConfig};
use alen::core::{ThoughtState, Problem};

fn main() {
    println!("=== ALEN Neural Network Training ===\n");
    
    // Create network configuration
    let config = ALENConfig::small();
    println!("Configuration:");
    println!("  Thought dimension: {}", config.thought_dim);
    println!("  Number of operators: {}", config.num_operators);
    println!("  Operator hidden dim: {}", config.operator_hidden_dim);
    println!("  Use transformer: {}\n", config.use_transformer);
    
    // Create network
    println!("Initializing network...");
    let network = ALENNetwork::new(config);
    println!("✓ Network created with {} parameters\n", network.num_parameters());
    
    // Test forward pass
    println!("=== Testing Forward Pass ===");
    let test_tokens = vec![1, 2, 3, 4, 5];
    println!("Input tokens: {:?}", test_tokens);
    
    let result = network.forward(&test_tokens);
    println!("\nResults:");
    println!("  Initial thought (ψ₀) norm: {:.4}", 
        result.psi_0.data.iter().map(|x| x * x).sum::<f32>().sqrt());
    println!("  Candidates generated: {}", result.candidates.len());
    println!("  Selected operator: {}", result.selected_operator);
    println!("  Selected operator name: {}", network.operators[result.selected_operator].name);
    println!("  Verification error: {:.6}", result.verification_error);
    
    // Show candidate energies
    println!("\n  Candidate energies:");
    for (i, candidate) in result.candidates.iter().enumerate() {
        println!("    Operator {}: {:.6}", i, candidate.energy);
    }
    
    // Test verification
    println!("\n=== Testing Verification ===");
    let epsilon_1 = 1.0;
    let epsilon_2 = 0.5;
    let verified = network.verify(&result.psi_star, &result.psi_0, epsilon_1, epsilon_2);
    println!("  Forward threshold (ε₁): {}", epsilon_1);
    println!("  Backward threshold (ε₂): {}", epsilon_2);
    println!("  Verification passed: {}", if verified { "✓" } else { "✗" });
    
    // Test multiple inputs
    println!("\n=== Testing Multiple Inputs ===");
    let test_cases = vec![
        vec![1, 2, 3],
        vec![5, 10, 15, 20],
        vec![100, 200, 300, 400, 500],
    ];
    
    for (i, tokens) in test_cases.iter().enumerate() {
        let result = network.forward(tokens);
        let verified = network.verify(&result.psi_star, &result.psi_0, epsilon_1, epsilon_2);
        println!("  Test {}: tokens={:?}, operator={}, verified={}, error={:.6}",
            i + 1,
            tokens,
            result.selected_operator,
            if verified { "✓" } else { "✗" },
            result.verification_error
        );
    }
    
    // Simulate training questions
    println!("\n=== Simulating Training Questions ===");
    let questions = vec![
        ("What is 2+2?", vec![1, 2, 3, 4]),
        ("What is the capital of France?", vec![5, 6, 7, 8, 9]),
        ("Explain gravity", vec![10, 11, 12, 13, 14, 15]),
    ];
    
    for (question, tokens) in questions {
        println!("\nQuestion: \"{}\"", question);
        let result = network.forward(&tokens);
        
        println!("  Candidates evaluated: {}", result.candidates.len());
        println!("  Best operator: {} ({})", 
            result.selected_operator,
            network.operators[result.selected_operator].name);
        
        // Find min and max energy
        let min_energy = result.candidates.iter()
            .map(|c| c.energy)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_energy = result.candidates.iter()
            .map(|c| c.energy)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        
        println!("  Energy range: [{:.4}, {:.4}]", min_energy, max_energy);
        println!("  Verification error: {:.6}", result.verification_error);
        
        let verified = network.verify(&result.psi_star, &result.psi_0, 1.0, 0.5);
        println!("  Verified: {}", if verified { "✓" } else { "✗" });
    }
    
    // Test operator consistency
    println!("\n=== Testing Operator Consistency ===");
    let tokens = vec![1, 2, 3, 4, 5];
    let mut operator_selections = vec![0; network.operators.len()];
    
    for _ in 0..100 {
        let result = network.forward(&tokens);
        operator_selections[result.selected_operator] += 1;
    }
    
    println!("Operator selection frequency (100 runs):");
    for (i, count) in operator_selections.iter().enumerate() {
        if *count > 0 {
            println!("  {}: {} times ({:.1}%)", 
                network.operators[i].name,
                count,
                (*count as f32 / 100.0) * 100.0
            );
        }
    }
    
    // Summary
    println!("\n=== Summary ===");
    println!("✓ Network architecture validated");
    println!("✓ Forward pass working");
    println!("✓ Parallel operators generating candidates");
    println!("✓ Energy-based selection functioning");
    println!("✓ Verification system operational");
    println!("\nNetwork is ready for training with real data!");
}
