# ALEN Large Model Architecture

## Overview

ALEN now supports configurable model sizes from Micro (1.1M params) to XL (2B params), featuring:

- **Full Transformer Architecture**: Multi-head self-attention, feed-forward networks, layer normalization
- **Pre-Norm Architecture**: More stable training for large models
- **SwiGLU Activation**: Modern activation function used in LLaMA/GPT-4
- **Configurable Sizes**: Micro, Small, Medium, Large, XL presets

## Model Configurations

| Size | Parameters | d_model | Heads | Layers | d_ff | Use Case |
|------|------------|---------|-------|--------|------|----------|
| **Micro** | 1.1M | 64 | 2 | 2 | 128 | Fast testing, demos |
| **Small** | 21M | 256 | 4 | 6 | 1024 | CPU training, prototyping |
| **Medium** | 89M | 512 | 8 | 12 | 2048 | Single GPU, good quality |
| **Large** | 404M | 1024 | 16 | 24 | 4096 | Multi-GPU, high quality |
| **XL** | 2B | 2048 | 32 | 36 | 8192 | Distributed, SOTA quality |

## Architecture Components

### 1. LargeMultiHeadAttention
```rust
pub struct LargeMultiHeadAttention {
    pub n_heads: usize,
    pub d_model: usize,
    pub d_head: usize,
    pub w_q: Linear,    // Query projection
    pub w_k: Linear,    // Key projection
    pub w_v: Linear,    // Value projection
    pub w_o: Linear,    // Output projection
}
```

### 2. LargeFeedForward (SwiGLU)
```rust
pub struct LargeFeedForward {
    pub w_up: Linear,    // Up projection
    pub w_gate: Linear,  // Gate projection (for SwiGLU)
    pub w_down: Linear,  // Down projection
}
```

### 3. LargeTransformerLayer
```rust
pub struct LargeTransformerLayer {
    pub ln1: LayerNorm,              // Pre-attention layer norm
    pub attention: LargeMultiHeadAttention,
    pub ln2: LayerNorm,              // Post-attention layer norm
    pub ffn: LargeFeedForward,
}
```

### 4. LargeLanguageModel
```rust
pub struct LargeLanguageModel {
    pub config: LargeModelConfig,
    pub token_embedding: Embedding,
    pub pos_embedding: Option<Embedding>,
    pub layers: Vec<LargeTransformerLayer>,
    pub final_ln: LayerNorm,
    pub lm_head: Linear,
    pub vocab: HashMap<String, usize>,
}
```

## Usage

### Quick Start
```rust
use alen::neural::{ModelSize, LargeModelConfig, LargeLanguageModel};

// Create a micro model for testing
let model = LargeLanguageModel::new(LargeModelConfig::micro());

// Or use preset sizes
let model = LargeLanguageModel::with_size(ModelSize::Small);
let model = LargeLanguageModel::with_size(ModelSize::Medium);
let model = LargeLanguageModel::with_size(ModelSize::Large);
let model = LargeLanguageModel::with_size(ModelSize::XL);
```

### Training
```rust
// Build vocabulary and train
model.learn("Training text here...");

// Train on Q&A pairs
model.learn_qa("What is 2+2?", "The answer is 4.");

// Train on sequences with gradient updates
model.train_on_sequence("Complete text for next-token prediction");
```

### Generation
```rust
let generated = model.generate("How do I", 50, 0.8);  // prompt, max_tokens, temperature
```

### Statistics
```rust
let stats = model.stats();
println!("Parameters: {}", stats.parameters_str);  // e.g., "21.1M"
println!("Vocabulary: {}/{}", stats.vocab_size, stats.max_vocab_size);
```

## Configuration Options

```rust
pub struct LargeModelConfig {
    pub size: ModelSize,           // Model size preset
    pub d_model: usize,            // Embedding dimension
    pub n_heads: usize,            // Number of attention heads
    pub n_layers: usize,           // Number of transformer layers
    pub d_ff: usize,               // Feed-forward hidden dimension
    pub max_seq_len: usize,        // Maximum sequence length
    pub vocab_size: usize,         // Vocabulary size
    pub dropout: f32,              // Dropout probability
    pub layer_norm_eps: f32,       // Layer norm epsilon
    pub d_head: usize,             // Head dimension
    pub use_rope: bool,            // Use rotary positional embeddings
    pub use_flash_attention: bool, // Memory-efficient attention
    pub gradient_checkpointing: bool, // Save memory during training
}
```

## Example

```bash
cargo run --release --example train_large_model
```

## Scaling Recommendations

| Model Size | Hardware | Training Time (1M tokens) |
|------------|----------|---------------------------|
| Micro | CPU | ~1 hour |
| Small | CPU/GPU | ~4 hours CPU, ~30 min GPU |
| Medium | Single GPU (8GB+) | ~2 hours |
| Large | Multi-GPU (32GB+) | ~8 hours |
| XL | Distributed (8x GPU) | ~24 hours |

## Key Features

1. **Pre-norm architecture**: Better gradient flow for deep models
2. **SwiGLU activation**: State-of-the-art activation function
3. **Causal masking**: For autoregressive generation
4. **Teacher forcing**: For sequence training
5. **Gradient updates**: Proper weight updates during training

## Integration with ALEN Core

The large models integrate with ALEN's reasoning engine:

```rust
use alen::core::ThoughtState;
use alen::generation::LatentDecoder;
use alen::neural::LargeLanguageModel;

// Both systems work together
let thought = ThoughtState::from_input("question", 256);
let latent_decoder = LatentDecoder::new(256, 64);
let large_model = LargeLanguageModel::new(LargeModelConfig::small());

// Hybrid generation
let latent_output = latent_decoder.generate(&thought);
let transformer_output = large_model.generate("prompt", 50, 0.8);
```
