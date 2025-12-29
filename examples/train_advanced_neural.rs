use alen::neural::{
    AdvancedALENSystem, AdvancedALENConfig, Tensor, TransformerConfig,
    NoiseSchedule, TemperatureSchedule,
};

fn main() {
    println!("🚀 Training Advanced ALEN Neural Network\n");
    println!("=" .repeat(60));
    
    // Create configuration
    println!("\n📋 Configuration:");
    let config = AdvancedALENConfig {
        problem_input_dim: 128,
        audience_profile_dim: 32,
        memory_retrieval_dim: 64,
        solution_embedding_dim: 128,
        explanation_embedding_dim: 128,
        solve_hidden_dims: vec![256, 128],
        verify_hidden_dims: vec![128, 64],
        explain_hidden_dims: vec![256, 128],
        transformer_config: TransformerConfig {
            d_model: 128,
            n_heads: 4,
            n_layers: 2,
            d_ff: 512,
            dropout: 0.1,
            max_seq_len: 256,
            vocab_size: 1000,
            layer_norm_eps: 1e-5,
        },
        dropout_rate: 0.1,
        loss_weights: (0.5, 0.3, 0.2),
        max_memories: 100,
        action_space_size: 1000,
        temperature: 1.0,
        gamma: 0.99,
        policy_learning_rate: 0.001,
        max_trajectory_length: 50,
        noise_sigma: 0.1,
        noise_schedule: NoiseSchedule::CosineAnneal { total_steps: 1000 },
        temperature_schedule: TemperatureSchedule::ExponentialCooling { decay_rate: 0.001 },
        diversity_weight: 0.1,
        novelty_k: 10,
        novelty_threshold: 0.5,
        inner_lr: 0.01,
        outer_lr: 0.001,
        inner_steps: 3,
        meta_hidden_dim: 128,
        base_lr: 0.001,
    };
    
    println!("   - Problem input dim: {}", config.problem_input_dim);
    println!("   - Solution embedding dim: {}", config.solution_embedding_dim);
    println!("   - Transformer layers: {}", config.transformer_config.n_layers);
    println!("   - Max memories: {}", config.max_memories);
    
    // Create system
    println!("\n🔧 Initializing system...");
    let mut system = AdvancedALENSystem::new(config);
    println!("   ✓ System initialized");
    
    // Training parameters
    let num_epochs = 50;
    let batch_size = 1;
    
    println!("\n🎯 Training parameters:");
    println!("   - Epochs: {}", num_epochs);
    println!("   - Batch size: {}", batch_size);
    
    // Training loop
    println!("\n🏋️  Training...\n");
    println!("{:<8} {:<12} {:<12} {:<12} {:<12}", "Epoch", "Total Loss", "Sol Loss", "Ver Loss", "Exp Loss");
    println!("{}", "-".repeat(60));
    
    for epoch in 0..num_epochs {
        // Generate synthetic training data
        let problem_input = Tensor::randn(&[batch_size, 128]);
        let audience_profile = Tensor::randn(&[batch_size, 32]);
        let target_solution = Tensor::randn(&[batch_size, 128]);
        let target_explanation = Tensor::randn(&[batch_size, 128]);
        let verification_target = 0.8 + (epoch as f32 / num_epochs as f32) * 0.2; // Gradually increase
        
        // Training step
        let metrics = system.train_step(
            &problem_input,
            &audience_profile,
            &target_solution,
            &target_explanation,
            verification_target,
        );
        
        // Print progress
        if epoch % 5 == 0 || epoch == num_epochs - 1 {
            println!(
                "{:<8} {:<12.4} {:<12.4} {:<12.4} {:<12.4}",
                epoch,
                metrics.universal_loss.total_loss,
                metrics.universal_loss.solution_loss,
                metrics.universal_loss.verification_loss,
                metrics.universal_loss.explanation_loss,
            );
        }
    }
    
    // Final statistics
    println!("\n{}", "=".repeat(60));
    println!("\n📊 Final Statistics:");
    let stats = system.get_stats();
    println!("   - Total training steps: {}", stats.total_steps);
    println!("   - Memories stored: {}", stats.memory_stats.total_memories);
    println!("   - Memory capacity used: {:.1}%", stats.memory_stats.capacity_used * 100.0);
    println!("   - Average memory usage: {:.2}", stats.memory_stats.avg_usage);
    println!("   - Curriculum difficulty: {:.2}", stats.curriculum_difficulty);
    println!("   - Policy baseline: {:.4}", stats.policy_baseline);
    
    println!("\n✅ Training complete!");
}
