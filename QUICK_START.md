# ALEN Quick Start Guide

## Running the Trained Neural Network

### Prerequisites

```bash
# Ensure Rust is installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Available Training Scripts

#### 1. Basic Neural Network Test
Tests the architecture with simple examples:
```bash
cargo run --example neural_training
```

**Output**: Architecture validation, forward pass testing, verification checks

#### 2. Small Training Demo (15 questions)
Quick training demonstration:
```bash
cargo run --example train_and_test
```

**Output**: 3 epochs, 15 questions, operator statistics, test results

#### 3. Comprehensive Training (100 questions) ⭐ RECOMMENDED
Full training on diverse dataset:
```bash
cargo run --example comprehensive_training
```

**Output**: 
- 5 epochs
- 100 questions across 10 categories
- Detailed category performance
- Operator usage statistics
- 8 test questions with results

### Training Data

The comprehensive training uses data from:
```
data/training_data.json
```

**Categories included**:
- Mathematics (10 questions)
- Geography (10 questions)
- Science (10 questions)
- History (10 questions)
- Technology (10 questions)
- Language (10 questions)
- Logic (10 questions)
- Philosophy (10 questions)
- Biology (10 questions)
- Physics (10 questions)

### Expected Results

**Comprehensive Training**:
- Training time: ~4 minutes
- Verification rate: ~91%
- Test accuracy: 100%
- Network parameters: 1,958,528

## Using the Neural Network Programmatically

### Basic Training

```rust
use alen::neural::{NeuralReasoningEngine, ALENConfig};
use alen::core::Problem;

fn main() {
    // Create configuration
    let config = ALENConfig::default();
    
    // Initialize engine
    let mut engine = NeuralReasoningEngine::new(config, 0.001);
    
    // Train on a question
    let problem = Problem::training(
        "What is 2+2?",
        "4",
        128
    );
    
    let result = engine.train_verified(&problem);
    
    println!("Verified: {}", result.verified);
    println!("Loss: {:.4}", result.loss);
    println!("Operator: {}", result.selected_operator);
}
```

### Inference

```rust
use alen::neural::{NeuralReasoningEngine, ALENConfig};

fn main() {
    let config = ALENConfig::default();
    let engine = NeuralReasoningEngine::new(config, 0.001);
    
    // Perform inference
    let result = engine.infer("What is the capital of France?");
    
    println!("Operator: {}", result.operator_name);
    println!("Verified: {}", result.verified);
    println!("Error: {:.4}", result.verification_error);
    println!("Confidence: {:.2}%", (1.0 - result.verification_error) * 100.0);
}
```

### Loading Custom Data

```rust
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Deserialize)]
struct QuestionAnswer {
    q: String,
    a: String,
}

#[derive(Deserialize)]
struct Category {
    category: String,
    questions: Vec<QuestionAnswer>,
}

fn load_data() -> Vec<Category> {
    let data = fs::read_to_string("data/training_data.json")
        .expect("Failed to read data file");
    serde_json::from_str(&data)
        .expect("Failed to parse JSON")
}
```

## Configuration Options

### Network Architecture

```rust
use alen::neural::ALENConfig;

let config = ALENConfig {
    thought_dim: 128,           // Thought vector dimension
    vocab_size: 10000,          // Vocabulary size
    num_operators: 8,           // Number of parallel operators
    operator_hidden_dim: 256,   // Hidden layer size
    dropout: 0.1,               // Dropout probability
    layer_norm_eps: 1e-5,       // Layer norm epsilon
    use_transformer: false,     // Use transformer encoder
    transformer_layers: 4,      // Transformer layers (if enabled)
    transformer_heads: 4,       // Attention heads (if enabled)
};
```

### Predefined Configurations

```rust
// Small (for testing)
let config = ALENConfig::small();
// thought_dim: 64, num_operators: 4

// Medium (recommended)
let config = ALENConfig::medium();
// thought_dim: 256, num_operators: 8

// Default
let config = ALENConfig::default();
// thought_dim: 128, num_operators: 8
```

### Training Parameters

```rust
let learning_rate = 0.001;  // Learning rate
let epsilon_1 = 1.0;        // Forward verification threshold
let epsilon_2 = 0.5;        // Backward verification threshold

let mut engine = NeuralReasoningEngine::new(config, learning_rate);
engine.epsilon_1 = epsilon_1;
engine.epsilon_2 = epsilon_2;
```

## Understanding the Output

### Training Output

```
Epoch 1/5
─────────────────────────────────────────────────────────────
  [20/100] Verified: 18/20 (90.0%) | Avg Loss: 0.3677
```

- **Verified**: Number of samples that passed verification
- **Percentage**: Verification rate
- **Avg Loss**: Average loss value (lower is better)

### Operator Statistics

```
Conservative (ID: 4)
  Usage: 158 times (31.6%)
  Success rate: 100.0%
  Weight: 1.0000
```

- **Usage**: How many times this operator was selected
- **Success rate**: Percentage of successful verifications
- **Weight**: Learned weight (higher = more trusted)

### Test Results

```
Test 1: [Math] "What is 8+7?"
  Operator: Analytical (ID: 6)
  Verified: ✓
  Verification error: 0.489140
```

- **Operator**: Which reasoning operator was selected
- **Verified**: ✓ = passed verification, ✗ = failed
- **Verification error**: Lower is better (threshold: 0.5)

## Performance Tips

### For Faster Training

1. Use smaller `thought_dim` (64 instead of 128)
2. Reduce `num_operators` (4 instead of 8)
3. Disable transformer (`use_transformer: false`)
4. Train on fewer epochs

### For Better Accuracy

1. Increase `thought_dim` (256 or 512)
2. Enable transformer (`use_transformer: true`)
3. Train for more epochs (10-20)
4. Use larger dataset
5. Implement data augmentation

### For Production

1. Save trained model to disk
2. Load pre-trained weights
3. Use batch processing
4. Implement caching
5. Monitor verification rates

## Troubleshooting

### Low Verification Rate

- **Cause**: Thresholds too strict or insufficient training
- **Solution**: Increase `epsilon_2` or train for more epochs

### High Loss

- **Cause**: Learning rate too high or complex data
- **Solution**: Reduce learning rate or simplify questions

### Operator Imbalance

- **Cause**: Some operators dominate selection
- **Solution**: Normal behavior, shows adaptive reasoning

### Memory Issues

- **Cause**: Large batch size or high dimension
- **Solution**: Reduce `thought_dim` or process in smaller batches

## Next Steps

1. **Experiment with configurations** - Try different settings
2. **Add your own data** - Create custom training datasets
3. **Implement saving/loading** - Persist trained models
4. **Build an API** - Expose the network via REST API
5. **Scale up** - Train on larger datasets
6. **Monitor performance** - Track metrics over time

## Resources

- **Implementation Details**: `NEURAL_NETWORK_IMPLEMENTATION.md`
- **Training Report**: `TRAINING_REPORT.md`
- **Source Code**: `src/neural/`
- **Examples**: `examples/`
- **Dataset**: `data/training_data.json`

## Support

For issues or questions:
1. Check the documentation files
2. Review example code
3. Examine test output
4. Verify configuration settings

---

**Quick Command Reference**:

```bash
# Test architecture
cargo run --example neural_training

# Quick training demo
cargo run --example train_and_test

# Full training (recommended)
cargo run --example comprehensive_training

# Build only
cargo build --release

# Run tests
cargo test
```

Happy training! 🚀
