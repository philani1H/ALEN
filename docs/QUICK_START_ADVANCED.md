# Quick Start: Advanced Neural Features

## Installation

Add to your `Cargo.toml`:
```toml
[dependencies]
alen = { path = "." }
```

## Basic Usage

### 1. Math Problem Solving

```rust
use alen::neural::{MathProblemSolver, AdvancedALENConfig, AudienceLevel};

fn main() {
    // Create solver with default configuration
    let config = AdvancedALENConfig::default();
    let mut solver = MathProblemSolver::new(config);
    
    // Solve a problem
    let solution = solver.solve(
        "Solve the equation x^2 + 2x + 1 = 0",
        AudienceLevel::HighSchool
    );
    
    println!("Solution: {}", solution.solution);
    println!("Explanation: {}", solution.explanation);
    println!("Confidence: {:.2}%", solution.confidence * 100.0);
}
```

### 2. Code Generation

```rust
use alen::neural::{CodeGenerationSystem, AdvancedALENConfig, ProgrammingLanguage};

fn main() {
    let config = AdvancedALENConfig::default();
    let mut generator = CodeGenerationSystem::new(config);
    
    let code = generator.generate(
        "Write a function to compute the nth Fibonacci number",
        ProgrammingLanguage::Rust
    );
    
    println!("Generated Code:\n{}", code.code);
    println!("\nExplanation: {}", code.explanation);
}
```

### 3. Custom Training

```rust
use alen::neural::{AdvancedALENSystem, AdvancedALENConfig, ExplorationMode, Tensor};

fn main() {
    let config = AdvancedALENConfig::default();
    let mut system = AdvancedALENSystem::new(config);
    
    // Prepare training data
    let problem_input = Tensor::randn(&[1, 512]);
    let audience_profile = Tensor::randn(&[1, 64]);
    let target_solution = Tensor::randn(&[1, 512]);
    let target_explanation = Tensor::randn(&[1, 512]);
    let verification_target = 0.9;
    
    // Training loop
    for epoch in 0..100 {
        let metrics = system.train_step(
            &problem_input,
            &audience_profile,
            &target_solution,
            &target_explanation,
            verification_target,
        );
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.4}", 
                epoch, 
                metrics.universal_loss.total_loss
            );
            println!("  Curriculum difficulty: {:.2}", 
                metrics.curriculum_difficulty
            );
        }
    }
}
```

## Configuration

### Small Model (Fast)

```rust
use alen::neural::{AdvancedALENConfig, TransformerConfig, NoiseSchedule, TemperatureSchedule};

let config = AdvancedALENConfig {
    problem_input_dim: 256,
    solution_embedding_dim: 256,
    solve_hidden_dims: vec![512, 256],
    transformer_config: TransformerConfig {
        d_model: 256,
        n_heads: 4,
        n_layers: 3,
        d_ff: 1024,
        dropout: 0.1,
        max_seq_len: 256,
    },
    max_memories: 1000,
    ..Default::default()
};
```

### Large Model (Quality)

```rust
let config = AdvancedALENConfig {
    problem_input_dim: 1024,
    solution_embedding_dim: 1024,
    solve_hidden_dims: vec![2048, 2048, 1024],
    transformer_config: TransformerConfig {
        d_model: 1024,
        n_heads: 16,
        n_layers: 12,
        d_ff: 4096,
        dropout: 0.1,
        max_seq_len: 1024,
    },
    max_memories: 100000,
    ..Default::default()
};
```

## Advanced Features

### Creative Exploration

```rust
use alen::neural::{ExplorationMode, SamplingMode};

// Gaussian noise exploration
let result = system.forward(
    &problem_input,
    &audience_profile,
    ExplorationMode::Gaussian,
    true, // use memory
);

// Structured noise with correlation
let result = system.forward(
    &problem_input,
    &audience_profile,
    ExplorationMode::Structured { correlation: 0.5 },
    true,
);

// Creative sampling
let samples = system.sample_creative(
    &logits,
    SamplingMode::Nucleus { p: 0.9 }
);
```

### Memory Management

```rust
// Get memory statistics
let stats = system.get_stats();
println!("Total memories: {}", stats.memory_stats.total_memories);
println!("Average usage: {:.2}", stats.memory_stats.avg_usage);
println!("Capacity used: {:.1}%", stats.memory_stats.capacity_used * 100.0);
```

### Meta-Learning

```rust
use alen::neural::{Task, DataSet, Tensor};

// Create tasks
let tasks = vec![
    Task {
        initial_params: /* ... */,
        support_set: DataSet { samples: /* ... */ },
        query_set: DataSet { samples: /* ... */ },
    },
    // More tasks...
];

// Meta-train
let metrics = system.meta_train(&tasks);
println!("Meta-loss: {:.4}", metrics.meta_loss);
```

## Audience Levels

```rust
use alen::neural::AudienceLevel;

// Available levels
AudienceLevel::Elementary    // Simple explanations
AudienceLevel::HighSchool    // Moderate detail
AudienceLevel::Undergraduate // Technical detail
AudienceLevel::Graduate      // Advanced concepts
AudienceLevel::Expert        // Full technical depth
```

## Programming Languages

```rust
use alen::neural::ProgrammingLanguage;

// Supported languages
ProgrammingLanguage::Python
ProgrammingLanguage::Rust
ProgrammingLanguage::JavaScript
ProgrammingLanguage::Java
```

## Monitoring and Debugging

### System Statistics

```rust
let stats = system.get_stats();
println!("Training steps: {}", stats.total_steps);
println!("Memory capacity: {:.1}%", stats.memory_stats.capacity_used * 100.0);
println!("Curriculum difficulty: {:.2}", stats.curriculum_difficulty);
println!("Policy baseline: {:.4}", stats.policy_baseline);
```

### Training Metrics

```rust
let metrics = system.train_step(/* ... */);

// Universal network losses
println!("Solution loss: {:.4}", metrics.universal_loss.solution_loss);
println!("Verification loss: {:.4}", metrics.universal_loss.verification_loss);
println!("Explanation loss: {:.4}", metrics.universal_loss.explanation_loss);
println!("Total loss: {:.4}", metrics.universal_loss.total_loss);

// Policy gradient metrics (if available)
if let Some(policy) = metrics.policy_metrics {
    println!("Actor loss: {:.4}", policy.actor_loss);
    println!("Critic loss: {:.4}", policy.critic_loss);
}
```

## Common Patterns

### Batch Processing

```rust
// Process multiple problems
let problems = vec![
    "Solve x^2 = 4",
    "Find derivative of x^3",
    "Integrate sin(x)",
];

for problem in problems {
    let solution = solver.solve(problem, AudienceLevel::HighSchool);
    println!("{}: {}", problem, solution.solution);
}
```

### Adaptive Difficulty

```rust
// System automatically adjusts difficulty based on performance
for epoch in 0..1000 {
    let metrics = system.train_step(/* ... */);
    
    // Difficulty increases as performance improves
    if epoch % 100 == 0 {
        println!("Current difficulty: {:.2}", metrics.curriculum_difficulty);
    }
}
```

### Memory-Enhanced Learning

```rust
// First solve (no memory)
let result1 = system.forward(&problem1, &audience, ExplorationMode::None, false);

// Train and store in memory
system.train_step(/* ... */);

// Second solve (with memory from first)
let result2 = system.forward(&problem2, &audience, ExplorationMode::None, true);
// Should be faster and more accurate if problems are similar
```

## Performance Tips

1. **Use appropriate model size**: Start with small model for prototyping
2. **Enable memory**: Set `use_memory=true` for similar problems
3. **Batch processing**: Process multiple problems together when possible
4. **GPU acceleration**: Enable CUDA for large models (when available)
5. **Curriculum learning**: Let difficulty increase naturally
6. **Creative exploration**: Use for novel problems, disable for known patterns

## Troubleshooting

### Low Confidence Scores

```rust
// Increase training iterations
for _ in 0..1000 {
    system.train_step(/* ... */);
}

// Or use more memories
let config = AdvancedALENConfig {
    max_memories: 100000,
    ..Default::default()
};
```

### Poor Explanations

```rust
// Increase explanation loss weight
let config = AdvancedALENConfig {
    loss_weights: (0.4, 0.2, 0.4), // (solution, verify, explain)
    ..Default::default()
};
```

### Slow Training

```rust
// Use smaller model
let config = AdvancedALENConfig {
    problem_input_dim: 256,
    solve_hidden_dims: vec![512, 256],
    transformer_config: TransformerConfig {
        n_layers: 3,
        ..Default::default()
    },
    ..Default::default()
};
```

## Next Steps

- Read [ADVANCED_NEURAL_ARCHITECTURE.md](ADVANCED_NEURAL_ARCHITECTURE.md) for detailed architecture
- See [NEURAL_IMPROVEMENTS_SUMMARY.md](NEURAL_IMPROVEMENTS_SUMMARY.md) for implementation details
- Check examples in `examples/` directory
- Join community discussions for tips and best practices

## API Reference

Full API documentation available at: [docs.rs/alen](https://docs.rs/alen)

Key modules:
- `alen::neural::AdvancedALENSystem` - Main system
- `alen::neural::MathProblemSolver` - Math interface
- `alen::neural::CodeGenerationSystem` - Code generation
- `alen::neural::UniversalExpertNetwork` - Core network
- `alen::neural::MemoryAugmentedNetwork` - Memory system
- `alen::neural::PolicyGradientTrainer` - RL training
- `alen::neural::CreativeExplorationController` - Exploration
- `alen::neural::MetaLearningController` - Meta-learning
