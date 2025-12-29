use alen::neural::{
    SelfDiscoveryLoop, ExplanationLevel, Tensor,
};

fn main() {
    println!("\n{}", "=".repeat(70));
    println!("🧠 ALEN Self-Discovery Loop Demo");
    println!("{}", "=".repeat(70));
    println!("\nDemonstrating autonomous knowledge discovery and inference.\n");
    
    // Configuration
    let input_dim = 128;
    let latent_dim = 64;
    let output_dim = 128;
    let consistency_threshold = 0.5;
    let max_iterations = 5;
    
    println!("📋 Configuration:");
    println!("   - Input dimension: {}", input_dim);
    println!("   - Latent dimension: {}", latent_dim);
    println!("   - Output dimension: {}", output_dim);
    println!("   - Consistency threshold: {}", consistency_threshold);
    println!("   - Max iterations: {}", max_iterations);
    println!();
    
    // Create self-discovery loop
    println!("🔧 Initializing Self-Discovery Loop...");
    let mut discovery_loop = SelfDiscoveryLoop::new(
        input_dim,
        latent_dim,
        output_dim,
        consistency_threshold,
        max_iterations,
    );
    println!("   ✓ Self-Discovery Loop initialized");
    println!();
    
    // Create initial knowledge
    println!("📚 Creating initial knowledge...");
    let initial_knowledge = Tensor::randn(&[1, input_dim]);
    println!("   ✓ Initial knowledge created (shape: {:?})", initial_knowledge.shape());
    println!();
    
    // Run single discovery step
    println!("🔍 Running single discovery step...");
    let result = discovery_loop.discover_step(
        &initial_knowledge,
        None,
        ExplanationLevel::Detailed,
    );
    
    println!("   ✓ Discovery step complete");
    println!("   - New knowledge shape: {:?}", result.z_new.shape());
    println!("   - Explanation shape: {:?}", result.explanation.shape());
    println!("   - Valid candidates: {}", result.num_valid_candidates);
    println!("   - Uncertainty: {:.4}", result.uncertainty);
    println!("   - Iteration: {}", result.iteration);
    println!();
    
    // Reset for full loop
    discovery_loop.reset();
    
    // Run full discovery loop
    println!("🔄 Running full discovery loop...");
    let results = discovery_loop.discover_loop(
        &initial_knowledge,
        None,
        ExplanationLevel::Detailed,
    );
    
    println!("   ✓ Discovery loop complete");
    println!("   - Total iterations: {}", results.len());
    println!();
    
    // Display results for each iteration
    println!("📊 Discovery Results:");
    println!("{}", "-".repeat(70));
    println!("{:<10} {:<15} {:<15} {:<15}", "Iteration", "Valid Cand.", "Uncertainty", "Status");
    println!("{}", "-".repeat(70));
    
    for result in &results {
        let status = if result.num_valid_candidates > 0 {
            "Discovering"
        } else {
            "Converged"
        };
        
        println!(
            "{:<10} {:<15} {:<15.4} {:<15}",
            result.iteration,
            result.num_valid_candidates,
            result.uncertainty,
            status
        );
    }
    println!("{}", "-".repeat(70));
    println!();
    
    // Get final statistics
    let stats = discovery_loop.get_stats();
    println!("📈 Final Statistics:");
    println!("   - Total iterations: {}", stats.iteration);
    println!("   - Knowledge base size: {}", stats.knowledge_base_size);
    println!("   - Number of operators: {}", stats.num_operators);
    println!("   - Consistency threshold: {:.2}", stats.consistency_threshold);
    println!();
    
    // Demonstrate different explanation levels
    println!("💡 Testing Different Explanation Levels:");
    println!();
    
    let test_knowledge = Tensor::randn(&[1, input_dim]);
    
    for level in &[ExplanationLevel::Simple, ExplanationLevel::Detailed, ExplanationLevel::Expert] {
        discovery_loop.reset();
        let result = discovery_loop.discover_step(&test_knowledge, None, *level);
        
        let level_name = match level {
            ExplanationLevel::Simple => "Simple",
            ExplanationLevel::Detailed => "Detailed",
            ExplanationLevel::Expert => "Expert",
        };
        
        println!("   {} Level:", level_name);
        println!("      - Explanation shape: {:?}", result.explanation.shape());
        println!("      - Valid candidates: {}", result.num_valid_candidates);
        println!();
    }
    
    println!("{}", "=".repeat(70));
    println!("✅ Self-Discovery Demo Complete!");
    println!("{}", "=".repeat(70));
    println!();
    println!("Key Features Demonstrated:");
    println!("  ✓ Knowledge encoding into latent space");
    println!("  ✓ Transformation operators generating candidates");
    println!("  ✓ Consistency verification filtering invalid inferences");
    println!("  ✓ Knowledge integration merging valid discoveries");
    println!("  ✓ Explanation generation at multiple levels");
    println!("  ✓ Uncertainty estimation for exploration");
    println!("  ✓ Iterative self-discovery loop");
    println!();
    println!("The system can now:");
    println!("  • Discover new knowledge autonomously");
    println!("  • Verify consistency to prevent hallucination");
    println!("  • Integrate discoveries into latent knowledge");
    println!("  • Generate explanations at appropriate levels");
    println!("  • Estimate uncertainty for guided exploration");
    println!();
}
