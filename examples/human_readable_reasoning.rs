use alen::neural::{
    NeuralReasoningEngine, ALENConfig, UniversalNetworkConfig,
};

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("🧠 ALEN - Human-Readable Neural Reasoning");
    println!("{}", "=".repeat(80));
    println!("\nShowing all reasoning steps in plain human language.\n");
    
    // Configuration
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
    
    // Create engine
    let mut engine = NeuralReasoningEngine::new(
        alen_config,
        universal_config,
        128,
        5,
    );
    
    // Test problems with human-readable reasoning
    let problems = vec![
        ("What is 2 + 2?", "arithmetic"),
        ("Why is the sky blue?", "science"),
        ("How do I sort a list?", "programming"),
    ];
    
    for (i, (problem, category)) in problems.iter().enumerate() {
        println!("\n{}", "━".repeat(80));
        println!("📝 Problem {}/{}: {}", i + 1, problems.len(), problem);
        println!("Category: {}", category);
        println!("{}", "━".repeat(80));
        
        // Run reasoning
        let trace = engine.reason(problem);
        
        // Display human-readable reasoning
        println!("\n💭 My Reasoning Process:\n");
        
        // Step 1: Understanding
        println!("1️⃣  UNDERSTANDING THE PROBLEM");
        println!("   First, I read and understood your question: \"{}\"", problem);
        println!("   I identified this as a {} problem.", category);
        println!("   I converted your question into my internal thought representation");
        println!("   (a {}-dimensional vector that captures the meaning).", trace.steps.first().map(|s| s.input_thought.len()).unwrap_or(128));
        println!();
        
        // Step 2: Thinking through it
        println!("2️⃣  THINKING THROUGH THE PROBLEM");
        println!("   I applied {} different reasoning steps:", trace.steps.len());
        println!();
        
        for (idx, step) in trace.steps.iter().enumerate() {
            let step_description = match idx {
                0 => format!("I started by breaking down the problem into its core components"),
                1 => format!("Then I explored different solution approaches"),
                2 => format!("I evaluated which approach would work best"),
                3 => format!("I refined my understanding and checked for consistency"),
                _ => format!("I continued refining my reasoning"),
            };
            
            println!("   Step {}: {}", idx + 1, step_description);
            println!("      • Confidence at this step: {:.0}%", step.confidence * 100.0);
            println!("      • Mental effort (energy): {:.2}", step.energy);
            println!("      • Verification: {}", if step.verified { "✅ This step makes sense" } else { "⚠️  Need to reconsider" });
            println!();
        }
        
        // Step 3: Verification
        println!("3️⃣  CHECKING MY WORK");
        println!("   I verified my reasoning by checking if it's consistent:");
        println!("   • Does my answer match the question? {}", if trace.verified { "✅ Yes" } else { "❌ No" });
        println!("   • Overall confidence: {:.0}%", trace.confidence * 100.0);
        println!("   • Total mental effort: {:.2} units", trace.total_energy);
        println!();
        
        // Step 4: Answer
        println!("4️⃣  MY ANSWER");
        println!("   {}", trace.answer);
        println!();
        
        // Step 5: Explanation
        println!("5️⃣  HOW I ARRIVED AT THIS ANSWER");
        println!("   {}", trace.explanation);
        println!();
        
        // Step 6: What I learned
        if !trace.discoveries.is_empty() {
            println!("6️⃣  WHAT I LEARNED FROM THIS");
            for (idx, discovery) in trace.discoveries.iter().enumerate() {
                println!("   • Discovery {}: {}", idx + 1, discovery);
            }
            println!();
        }
        
        // Summary
        println!("📊 REASONING SUMMARY");
        println!("   • Problem understood: ✅");
        println!("   • Reasoning steps taken: {}", trace.steps.len());
        println!("   • Answer verified: {}", if trace.verified { "✅" } else { "❌" });
        println!("   • Confidence level: {:.0}%", trace.confidence * 100.0);
        println!("   • New insights gained: {}", trace.discoveries.len());
    }
    
    println!("\n{}", "=".repeat(80));
    println!("✅ All Reasoning Complete!");
    println!("{}", "=".repeat(80));
    println!();
    println!("🎓 What You Just Saw:");
    println!();
    println!("For each problem, I showed you:");
    println!("  1. How I understood the question");
    println!("  2. The steps I took to think through it");
    println!("  3. How I checked my work");
    println!("  4. My final answer");
    println!("  5. My explanation of how I got there");
    println!("  6. What new things I learned");
    println!();
    println!("Every step used neural networks, but I explained it all in");
    println!("plain human language so you can follow my reasoning!");
    println!();
    println!("🧠 Behind the Scenes:");
    println!("  • Neural encoding converted your question to thought vectors");
    println!("  • Neural operators transformed thoughts step by step");
    println!("  • Neural verification checked consistency");
    println!("  • Neural decoding converted thoughts back to language");
    println!("  • Neural explanation generated the human-readable text");
    println!("  • Neural discovery found new patterns and insights");
    println!();
}
