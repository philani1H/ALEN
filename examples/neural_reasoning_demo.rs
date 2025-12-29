use alen::neural::{
    NeuralReasoningEngine, ALENConfig, UniversalNetworkConfig,
};

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("🧠 ALEN Neural Reasoning Engine - Real-Time Visualization");
    println!("{}", "=".repeat(80));
    println!("\nAll reasoning steps use neural networks for real-time observation.\n");
    
    // Configuration
    println!("📋 Configuration:");
    let alen_config = ALENConfig {
        thought_dim: 128,
        vocab_size: 5000,
        num_operators: 6,
        operator_hidden_dim: 256,
        dropout: 0.1,
        layer_norm_eps: 1e-5,
        use_transformer: true,
        transformer_layers: 3,
        transformer_heads: 8,
    };
    
    let universal_config = UniversalNetworkConfig::default();
    
    println!("   - Thought dimension: {}", alen_config.thought_dim);
    println!("   - Neural operators: {}", alen_config.num_operators);
    println!("   - Transformer layers: {}", alen_config.transformer_layers);
    println!("   - Transformer heads: {}", alen_config.transformer_heads);
    println!();
    
    // Create engine
    println!("🔧 Initializing Neural Reasoning Engine...");
    let mut engine = NeuralReasoningEngine::new(
        alen_config,
        universal_config,
        128,
        5,  // max_steps
    );
    println!("   ✓ Engine initialized with neural-backed reasoning");
    println!();
    
    // Test problems
    let problems = vec![
        "What is 2 + 2?",
        "Explain the concept of gravity",
        "Write a function to sort a list",
    ];
    
    for (i, problem) in problems.iter().enumerate() {
        println!("\n{}", "━".repeat(80));
        println!("Problem {}/{}: {}", i + 1, problems.len(), problem);
        println!("{}", "━".repeat(80));
        
        // Run neural reasoning with visualization
        let trace = engine.reason(problem);
        
        // Display summary
        println!("\n📊 Reasoning Summary:");
        println!("   Problem: {}", trace.problem);
        println!("   Answer: {}", trace.answer);
        println!("   Confidence: {:.1}%", trace.confidence * 100.0);
        println!("   Total steps: {}", trace.steps.len());
        println!("   Total energy: {:.4}", trace.total_energy);
        println!("   Verified: {}", if trace.verified { "✅ Yes" } else { "❌ No" });
        
        if !trace.discoveries.is_empty() {
            println!("\n   🔬 Discoveries:");
            for discovery in &trace.discoveries {
                println!("      • {}", discovery);
            }
        }
        
        println!("\n   💡 Explanation:");
        println!("      {}", trace.explanation);
        
        // Display step-by-step breakdown
        println!("\n   📝 Step-by-Step Breakdown:");
        for step in &trace.steps {
            println!("      Step {}: {} (confidence: {:.1}%, energy: {:.4}, verified: {})",
                step.step_number,
                step.operator_name,
                step.confidence * 100.0,
                step.energy,
                if step.verified { "✅" } else { "❌" }
            );
        }
    }
    
    // Get engine statistics
    println!("\n{}", "=".repeat(80));
    println!("📈 Engine Statistics:");
    let stats = engine.get_stats();
    println!("   - Thought dimension: {}", stats.thought_dim);
    println!("   - Max reasoning steps: {}", stats.max_steps);
    println!("   - Energy threshold: {:.4}", stats.energy_threshold);
    println!("   - Discovery knowledge base: {} entries", stats.discovery_stats.knowledge_base_size);
    println!("   - Discovery operators: {}", stats.discovery_stats.num_operators);
    
    println!("\n{}", "=".repeat(80));
    println!("✅ Neural Reasoning Demo Complete!");
    println!("{}", "=".repeat(80));
    println!();
    println!("Key Features Demonstrated:");
    println!("  ✓ Neural encoding: Problem → Thought Vector");
    println!("  ✓ Multi-step neural reasoning with operators");
    println!("  ✓ Neural verification: Consistency checking");
    println!("  ✓ Neural decoding: Thought → Answer");
    println!("  ✓ Neural explanation generation");
    println!("  ✓ Self-discovery: Finding new knowledge");
    println!("  ✓ Real-time visualization of all steps");
    println!();
    println!("All reasoning steps use neural networks:");
    println!("  • Encoding uses neural embeddings");
    println!("  • Reasoning uses neural operators");
    println!("  • Verification uses neural consistency checks");
    println!("  • Decoding uses neural transformations");
    println!("  • Explanation uses neural generation");
    println!("  • Discovery uses neural exploration");
    println!();
}
