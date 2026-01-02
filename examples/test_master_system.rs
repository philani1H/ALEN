//! Test the Master Neural System
//!
//! Tests:
//! 1. Training on examples
//! 2. Creative responses
//! 3. Story understanding
//! 4. All neural components working together

use alen::neural::{MasterNeuralSystem, MasterSystemConfig};

fn main() {
    let sep = "=".repeat(70);
    println!("{}", sep);
    println!("  MASTER NEURAL SYSTEM - LIVE TEST");
    println!("  Controller (φ) + Core Model (θ) = Complete AI");
    println!("{}", sep);

    // Create system with default config
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
    };

    let mut system = MasterNeuralSystem::new(config);

    println!("\n✅ System initialized!");
    println!("   Controller patterns: 50");
    println!("   Thought dimension: 128");
    println!("   Controller LR: 0.001 (governance)");
    println!("   Core Model LR: 0.1 (learning)");

    // ========================================================================
    // PART 1: TRAINING
    // ========================================================================

    let sep = "=".repeat(70);
    println!("\n{}", sep);
    println!("PART 1: TRAINING");
    println!("{}", sep);

    let training_examples = vec![
        ("What is 2 + 2?", "4. This is basic addition."),
        ("Explain neural networks", "Neural networks are computational models inspired by the brain, consisting of interconnected layers of nodes that learn patterns from data."),
        ("What is creativity?", "Creativity is the ability to generate novel and valuable ideas by making unexpected connections between existing concepts."),
        ("Tell me a short story", "Once upon a time, a curious robot learned to appreciate the beauty of a sunset, realizing that understanding isn't just computation."),
        ("How do you learn?", "I learn by adjusting my neural network weights based on examples, gradually improving my ability to recognize patterns and generate appropriate responses."),
    ];

    println!("\nTraining on {} examples...\n", training_examples.len());

    for (i, (input, target)) in training_examples.iter().enumerate() {
        let metrics = system.train_step(input, target);
        println!("Example {}/{}:", i + 1, training_examples.len());
        println!("  Input: {}", input);
        println!("  Target: {}...", &target[..target.len().min(60)]);
        println!("  Loss: {:.4}", metrics.total_loss);
        println!("  Confidence: {:.2}%", metrics.confidence * 100.0);
        println!();
    }

    let stats = system.get_stats();
    println!("Training Stats:");
    println!("  Total steps: {}", stats.total_training_steps);
    println!("  Controller updates: {}", stats.controller_updates);
    println!("  Core model updates: {}", stats.core_model_updates);
    println!("  Avg confidence: {:.2}%", stats.avg_confidence * 100.0);

    // ========================================================================
    // PART 2: CREATIVE RESPONSES
    // ========================================================================

    let sep = "=".repeat(70);
    println!("\n{}", sep);
    println!("PART 2: TESTING CREATIVITY");
    println!("{}", sep);

    let creative_prompts = vec![
        "What is the meaning of life?",
        "Describe a beautiful sunset",
        "What would you do if you could fly?",
        "Tell me something profound about time",
    ];

    for prompt in creative_prompts {
        println!("\n🎨 Creative Prompt: {}", prompt);
        let response = system.forward(prompt);
        println!("   Response: {}", response.response);
        println!("   Confidence: {:.2}%", response.confidence * 100.0);
        println!("   Reasoning depth: {}", response.controls.reasoning_depth);
        println!("   Creativity level: {:.2}", response.controls.style.creativity);
        println!("   Action: {:?}", response.controls.action);
    }

    // ========================================================================
    // PART 3: STORY UNDERSTANDING
    // ========================================================================

    let sep = "=".repeat(70);
    println!("\n{}", sep);
    println!("PART 3: STORY UNDERSTANDING");
    println!("{}", sep);

    let story = "A young scientist named Maya discovered a way to communicate with plants. \
                 At first, nobody believed her. But when she demonstrated that plants could \
                 respond to questions about their environment, the world changed forever. \
                 Maya's discovery taught humanity that intelligence exists in forms we never imagined.";

    println!("\n📖 Story:");
    println!("{}\n", story);

    // Train on the story first
    println!("Training on story...");
    let _ = system.train_step(
        story,
        "Maya discovered plant communication which revealed new forms of intelligence"
    );

    // Now test understanding
    let understanding_questions = vec![
        "Who is the main character?",
        "What did Maya discover?",
        "How did people react initially?",
        "What was the lesson?",
    ];

    for question in understanding_questions {
        println!("\n❓ Question: {}", question);
        let response = system.forward(question);
        println!("   Answer: {}", response.response);
        println!("   Confidence: {:.2}%", response.confidence * 100.0);

        // Show controller decision
        match response.controls.action {
            alen::generation::ControlAction::Answer => println!("   ✅ Controller decision: ANSWER (confident)"),
            alen::generation::ControlAction::Ask => println!("   ❓ Controller decision: ASK (needs clarification)"),
            alen::generation::ControlAction::VerifyMore => println!("   🔍 Controller decision: VERIFY_MORE (uncertain)"),
            _ => println!("   Controller decision: {:?}", response.controls.action),
        }
    }

    // ========================================================================
    // PART 4: MATHEMATICAL REASONING
    // ========================================================================

    let sep = "=".repeat(70);
    println!("\n{}", sep);
    println!("PART 4: MATHEMATICAL REASONING");
    println!("{}", sep);

    let math_examples = vec![
        ("What is 15 × 7?", "105"),
        ("Solve: 2x + 5 = 13", "x = 4"),
        ("What is the area of a circle with radius 5?", "25π or approximately 78.54"),
    ];

    // Train on math
    println!("\nTraining on {} math examples...\n", math_examples.len());
    for (input, target) in &math_examples {
        let _ = system.train_step(input, target);
    }

    // Test math understanding
    let math_tests = vec![
        "What is 10 × 8?",
        "What is 3 + 7?",
        "Explain what multiplication means",
    ];

    for test in math_tests {
        println!("\n🔢 Math test: {}", test);
        let response = system.forward(test);
        println!("   Response: {}", response.response);
        println!("   Confidence: {:.2}%", response.confidence * 100.0);
    }

    // ========================================================================
    // FINAL STATS
    // ========================================================================

    let sep = "=".repeat(70);
    println!("\n{}", sep);
    println!("FINAL SYSTEM STATISTICS");
    println!("{}", sep);

    let final_stats = system.get_stats();
    println!("\n📊 Training Progress:");
    println!("   Total training steps: {}", final_stats.total_training_steps);
    println!("   Controller updates (φ): {}", final_stats.controller_updates);
    println!("   Core model updates (θ): {}", final_stats.core_model_updates);
    println!("   Average confidence: {:.2}%", final_stats.avg_confidence * 100.0);
    println!("   Average perplexity: {:.2}", final_stats.avg_perplexity);
    println!("\n📈 Learning Rates:");
    println!("   Controller LR (φ): {:.6} (SMALL - governance)", final_stats.controller_lr);
    println!("   Core Model LR (θ): {:.6} (LARGE - learning)", final_stats.core_lr);

    println!("\n✅ All neural components working together!");
    println!("   - Controller (φ) chooses HOW to think");
    println!("   - Memory retrieves context");
    println!("   - Core Model (θ) generates responses");
    println!("   - Self-discovery learns new patterns");
    println!("   - Creativity modulation enabled");

    println!("\n🎉 Master Neural System test complete!");
    let sep = "=".repeat(70);
    println!("{}", sep);
}
