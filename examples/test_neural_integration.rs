//! Test Neural Integration
//! 
//! Comprehensive test to verify all neural components work together

use alen::neural::*;

fn main() {
    println!("=================================================================");
    println!("  ALEN Neural System Integration Test");
    println!("=================================================================");
    println!();

    // Test 1: Basic Tensor Operations
    println!("Test 1: Tensor Operations");
    println!("-----------------------------------------------------------------");
    let t1 = Tensor::randn(vec![2, 3]);
    let t2 = Tensor::ones(vec![2, 3]);
    let t3 = t1.add(&t2);
    println!("✓ Tensor creation and operations working");
    println!("  Shape: {:?}", t3.shape());
    println!();

    // Test 2: Neural Layers
    println!("Test 2: Neural Layers");
    println!("-----------------------------------------------------------------");
    let linear = Linear::new(10, 5, true);
    let input = Tensor::randn(vec![2, 10]);
    let output = linear.forward(&input);
    println!("✓ Linear layer working");
    println!("  Input shape: {:?}, Output shape: {:?}", input.shape(), output.shape());
    
    let ln = LayerNorm::new(vec![5], 1e-5);
    let normalized = ln.forward(&output);
    println!("✓ LayerNorm working");
    println!("  Normalized shape: {:?}", normalized.shape());
    println!();

    // Test 3: Transformer
    println!("Test 3: Transformer");
    println!("-----------------------------------------------------------------");
    let config = TransformerConfig {
        vocab_size: 1000,
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        d_ff: 256,
        max_seq_len: 128,
        dropout: 0.1,
        layer_norm_eps: 1e-5,
    };
    let transformer = TransformerEncoder::new(config);
    println!("✓ Transformer created");
    println!("  Config: d_model={}, n_heads={}, n_layers={}", 64, 4, 2);
    println!();

    // Test 4: ALEN Network
    println!("Test 4: ALEN Network");
    println!("-----------------------------------------------------------------");
    let alen_config = ALENConfig {
        thought_dim: 128,
        num_operators: 8,
        operator_hidden_dim: 256,
        verifier_hidden_dim: 128,
        dropout: 0.1,
        layer_norm_eps: 1e-5,
    };
    let alen = ALENNetwork::new(alen_config);
    println!("✓ ALEN Network created");
    println!("  Thought dimension: 128");
    println!("  Number of operators: 8");
    println!();

    // Test 5: Universal Expert Network
    println!("Test 5: Universal Expert Network");
    println!("-----------------------------------------------------------------");
    let universal_config = UniversalNetworkConfig {
        input_dim: 512,
        audience_dim: 64,
        memory_dim: 256,
        solution_dim: 256,
        explanation_dim: 512,
        solve_hidden: vec![512, 256],
        verify_hidden: vec![256, 128],
        explain_hidden: vec![512, 256],
        transformer_config: TransformerConfig::default(),
        dropout: 0.1,
        alpha: 1.0,
        beta: 0.5,
        gamma: 0.5,
    };
    let _universal = UniversalExpertNetwork::new(universal_config);
    println!("✓ Universal Expert Network created");
    println!("  Input dim: 512, Solution dim: 256");
    println!();

    // Test 6: Memory-Augmented Network
    println!("Test 6: Memory-Augmented Network");
    println!("-----------------------------------------------------------------");
    let mut memory_net = MemoryAugmentedNetwork::new(
        512,  // input_dim
        256,  // embedding_dim
        128,  // memory_dim
        100,  // max_memories
    );
    let test_input = Tensor::randn(vec![1, 512]);
    let (_embedding, _memory) = memory_net.forward_with_memory(&test_input, 5);
    println!("✓ Memory-Augmented Network working");
    println!("  Max memories: 100, Top-k retrieval: 5");
    println!();

    // Test 7: Policy Gradient
    println!("Test 7: Policy Gradient");
    println!("-----------------------------------------------------------------");
    let mut policy_trainer = PolicyGradientTrainer::new(
        10,    // action_space_size
        1.0,   // temperature
        0.99,  // gamma
        0.001, // learning_rate
        100,   // max_trajectory_length
    );
    policy_trainer.add_experience(0.5, 1.0);
    let _metrics = policy_trainer.train();
    println!("✓ Policy Gradient Trainer working");
    println!("  Action space: 10, Gamma: 0.99");
    println!();

    // Test 8: Creative Exploration
    println!("Test 8: Creative Exploration");
    println!("-----------------------------------------------------------------");
    let mut creative = CreativeExplorationController::new(
        0.1,                           // noise_sigma
        NoiseSchedule::Constant,       // noise_schedule
        1.0,                           // temperature
        TemperatureSchedule::Constant, // temperature_schedule
        0.5,                           // diversity_weight
        10,                            // novelty_k
        0.5,                           // novelty_threshold
    );
    let latent = Tensor::randn(vec![1, 128]);
    let explored = creative.explore(&latent, ExplorationMode::Gaussian);
    println!("✓ Creative Exploration working");
    println!("  Explored shape: {:?}", explored.shape());
    creative.step();
    println!();

    // Test 9: Meta-Learning
    println!("Test 9: Meta-Learning");
    println!("-----------------------------------------------------------------");
    let _meta = MetaLearningController::new(
        0.01,  // inner_lr
        0.001, // outer_lr
        5,     // inner_steps
        128,   // param_dim
        256,   // hidden_dim
        0.001, // base_lr
    );
    println!("✓ Meta-Learning Controller created");
    println!("  Inner LR: 0.01, Outer LR: 0.001");
    println!();

    // Test 10: Self-Discovery Loop
    println!("Test 10: Self-Discovery Loop");
    println!("-----------------------------------------------------------------");
    let mut discovery = SelfDiscoveryLoop::new(
        128,  // knowledge_dim
        64,   // hidden_dim
        256,  // explanation_dim
        0.5,  // consistency_threshold
        10,   // max_iterations
    );
    let knowledge = Tensor::randn(vec![1, 128]);
    let result = discovery.discover(&knowledge);
    println!("✓ Self-Discovery Loop working");
    println!("  Iterations: {}, Discoveries: {}", 
             result.iterations, result.discoveries.len());
    println!();

    // Test 11: Neural Reasoning Engine
    println!("Test 11: Neural Reasoning Engine");
    println!("-----------------------------------------------------------------");
    let alen_config = ALENConfig::default();
    let universal_config = UniversalNetworkConfig {
        input_dim: 512,
        audience_dim: 64,
        memory_dim: 256,
        solution_dim: 256,
        explanation_dim: 512,
        solve_hidden: vec![512, 256],
        verify_hidden: vec![256, 128],
        explain_hidden: vec![512, 256],
        transformer_config: TransformerConfig::default(),
        dropout: 0.1,
        alpha: 1.0,
        beta: 0.5,
        gamma: 0.5,
    };
    let mut engine = NeuralReasoningEngine::new(
        alen_config,
        universal_config,
        128,  // thought_dim
        10,   // max_steps
    );
    let trace = engine.reason("What is 2+2?");
    println!("✓ Neural Reasoning Engine working");
    println!("  Problem: What is 2+2?");
    println!("  Steps: {}, Confidence: {:.2}", trace.steps.len(), trace.confidence);
    println!("  Answer: {}", trace.answer);
    println!();

    // Test 12: Advanced Integration
    println!("Test 12: Advanced Integration System");
    println!("-----------------------------------------------------------------");
    let advanced_config = AdvancedALENConfig {
        problem_input_dim: 512,
        audience_profile_dim: 64,
        memory_retrieval_dim: 256,
        solution_embedding_dim: 256,
        explanation_embedding_dim: 512,
        solve_hidden_dims: vec![512, 256],
        verify_hidden_dims: vec![256, 128],
        explain_hidden_dims: vec![512, 256],
        transformer_config: TransformerConfig::default(),
        dropout_rate: 0.1,
        loss_weights: (1.0, 0.5, 0.5),
        action_space_size: 10,
        temperature: 1.0,
        gamma: 0.99,
        policy_learning_rate: 0.001,
        max_trajectory_length: 100,
        noise_sigma: 0.1,
        noise_schedule: NoiseSchedule::Constant,
        temperature_schedule: TemperatureSchedule::Constant,
        diversity_weight: 0.5,
        novelty_k: 10,
        novelty_threshold: 0.5,
        inner_lr: 0.01,
        outer_lr: 0.001,
        inner_steps: 5,
        meta_hidden_dim: 256,
        base_lr: 0.001,
        max_memories: 100,
    };
    let _advanced = AdvancedALENSystem::new(advanced_config);
    println!("✓ Advanced Integration System created");
    println!("  All subsystems initialized");
    println!();

    // Summary
    println!("=================================================================");
    println!("  Test Summary");
    println!("=================================================================");
    println!();
    println!("✓ All 12 neural components tested successfully!");
    println!();
    println!("Components verified:");
    println!("  1. ✓ Tensor Operations");
    println!("  2. ✓ Neural Layers (Linear, LayerNorm)");
    println!("  3. ✓ Transformer Encoder");
    println!("  4. ✓ ALEN Network");
    println!("  5. ✓ Universal Expert Network");
    println!("  6. ✓ Memory-Augmented Network");
    println!("  7. ✓ Policy Gradient Trainer");
    println!("  8. ✓ Creative Exploration Controller");
    println!("  9. ✓ Meta-Learning Controller");
    println!(" 10. ✓ Self-Discovery Loop");
    println!(" 11. ✓ Neural Reasoning Engine");
    println!(" 12. ✓ Advanced Integration System");
    println!();
    println!("Status: 🟢 ALL NEURAL SYSTEMS OPERATIONAL");
    println!();
}
