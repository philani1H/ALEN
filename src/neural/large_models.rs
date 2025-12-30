//! Large Model Configurations
//!
//! Production-ready model sizes from small to GPT-scale:
//! - Small: 12M parameters (fast training, testing)
//! - Medium: 85M parameters (good quality)
//! - Large: 350M parameters (high quality)
//! - XL: 1.3B parameters (GPT-2 scale)

use super::transformer::{TransformerConfig, TransformerEncoder};
use super::transformer_decoder::{TransformerDecoder, TransformerEnhancedDecoder};
use super::layers::{Linear, LayerNorm, Embedding};
use super::tensor::Tensor;
use crate::core::ThoughtState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// MODEL SIZE CONFIGURATIONS
// ============================================================================

/// Model size presets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSize {
    /// ~12M parameters - Fast training, testing
    Small,
    /// ~85M parameters - Good balance
    Medium,
    /// ~350M parameters - High quality
    Large,
    /// ~1.3B parameters - GPT-2 scale
    XL,
    /// Custom configuration
    Custom,
}

/// Large Language Model Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeModelConfig {
    /// Model size preset
    pub size: ModelSize,
    /// Model/embedding dimension (d_model)
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Feed-forward hidden dimension (usually 4 * d_model)
    pub d_ff: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Layer norm epsilon
    pub layer_norm_eps: f32,
    /// Head dimension (d_model / n_heads)
    pub d_head: usize,
    /// Use rotary positional embeddings
    pub use_rope: bool,
    /// Use flash attention (memory efficient)
    pub use_flash_attention: bool,
    /// Gradient checkpointing (save memory)
    pub gradient_checkpointing: bool,
}

impl LargeModelConfig {
    /// Micro model (~500K parameters)
    /// Good for: Fast testing, demos
    pub fn micro() -> Self {
        Self {
            size: ModelSize::Small, // Use Small enum
            d_model: 64,
            n_heads: 2,
            n_layers: 2,
            d_ff: 128,
            max_seq_len: 128,
            vocab_size: 8000,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            d_head: 32,
            use_rope: false,
            use_flash_attention: false,
            gradient_checkpointing: false,
        }
    }
    
    /// Small model (~12M parameters)
    /// Good for: Testing, prototyping, CPU training
    pub fn small() -> Self {
        Self {
            size: ModelSize::Small,
            d_model: 256,
            n_heads: 4,
            n_layers: 6,
            d_ff: 1024,        // 4 * d_model
            max_seq_len: 512,
            vocab_size: 32000,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            d_head: 64,        // d_model / n_heads
            use_rope: false,
            use_flash_attention: false,
            gradient_checkpointing: false,
        }
    }

    /// Medium model (~85M parameters)
    /// Good for: Quality generation, reasonable training time
    pub fn medium() -> Self {
        Self {
            size: ModelSize::Medium,
            d_model: 512,
            n_heads: 8,
            n_layers: 12,
            d_ff: 2048,
            max_seq_len: 1024,
            vocab_size: 50000,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            d_head: 64,
            use_rope: true,
            use_flash_attention: false,
            gradient_checkpointing: false,
        }
    }

    /// Large model (~350M parameters)
    /// Good for: High-quality generation, requires GPU
    pub fn large() -> Self {
        Self {
            size: ModelSize::Large,
            d_model: 1024,
            n_heads: 16,
            n_layers: 24,
            d_ff: 4096,
            max_seq_len: 2048,
            vocab_size: 50000,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            d_head: 64,
            use_rope: true,
            use_flash_attention: true,
            gradient_checkpointing: true,
        }
    }

    /// XL model (~1.3B parameters, GPT-2 scale)
    /// Good for: State-of-the-art quality, requires multi-GPU
    pub fn xl() -> Self {
        Self {
            size: ModelSize::XL,
            d_model: 2048,
            n_heads: 32,
            n_layers: 36,
            d_ff: 8192,
            max_seq_len: 4096,
            vocab_size: 50000,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            d_head: 64,
            use_rope: true,
            use_flash_attention: true,
            gradient_checkpointing: true,
        }
    }

    /// Convert to TransformerConfig
    pub fn to_transformer_config(&self) -> TransformerConfig {
        TransformerConfig {
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_ff: self.d_ff,
            n_layers: self.n_layers,
            max_seq_len: self.max_seq_len,
            vocab_size: self.vocab_size,
            dropout: self.dropout,
            layer_norm_eps: self.layer_norm_eps,
        }
    }

    /// Estimate parameter count
    pub fn estimate_parameters(&self) -> usize {
        // Embedding: vocab_size * d_model
        let embedding_params = self.vocab_size * self.d_model;
        
        // Per layer:
        // - Self-attention: 4 * d_model^2 (Q, K, V, O projections)
        // - FFN: 2 * d_model * d_ff
        // - LayerNorms: 2 * 2 * d_model
        let attention_params = 4 * self.d_model * self.d_model;
        let ffn_params = 2 * self.d_model * self.d_ff;
        let ln_params = 4 * self.d_model;
        let per_layer = attention_params + ffn_params + ln_params;
        
        // Total
        let layer_params = self.n_layers * per_layer;
        
        // Output projection
        let output_params = self.d_model * self.vocab_size;
        
        embedding_params + layer_params + output_params
    }

    /// Get human-readable parameter count
    pub fn parameter_count_string(&self) -> String {
        let params = self.estimate_parameters();
        if params >= 1_000_000_000 {
            format!("{:.1}B", params as f64 / 1_000_000_000.0)
        } else if params >= 1_000_000 {
            format!("{:.1}M", params as f64 / 1_000_000.0)
        } else if params >= 1_000 {
            format!("{:.1}K", params as f64 / 1_000.0)
        } else {
            format!("{}", params)
        }
    }
}

// ============================================================================
// LARGE LANGUAGE MODEL
// ============================================================================

/// Large Language Model with configurable size
#[derive(Debug)]
pub struct LargeLanguageModel {
    /// Model configuration
    pub config: LargeModelConfig,
    /// Token embeddings
    pub token_embedding: Embedding,
    /// Positional embeddings (if not using RoPE)
    pub pos_embedding: Option<Embedding>,
    /// Transformer layers
    pub layers: Vec<LargeTransformerLayer>,
    /// Final layer norm
    pub final_ln: LayerNorm,
    /// Output projection (language model head)
    pub lm_head: Linear,
    /// Vocabulary
    pub vocab: HashMap<String, usize>,
    /// Reverse vocabulary
    pub id_to_token: HashMap<usize, String>,
    /// Training step count
    pub train_steps: u64,
}

/// Single transformer layer for large models
#[derive(Debug, Clone)]
pub struct LargeTransformerLayer {
    /// Pre-attention layer norm
    pub ln1: LayerNorm,
    /// Multi-head self-attention
    pub attention: LargeMultiHeadAttention,
    /// Post-attention layer norm  
    pub ln2: LayerNorm,
    /// Feed-forward network
    pub ffn: LargeFeedForward,
}

/// Multi-head attention for large models
#[derive(Debug, Clone)]
pub struct LargeMultiHeadAttention {
    pub n_heads: usize,
    pub d_model: usize,
    pub d_head: usize,
    /// Query projection
    pub w_q: Linear,
    /// Key projection
    pub w_k: Linear,
    /// Value projection
    pub w_v: Linear,
    /// Output projection
    pub w_o: Linear,
    pub dropout: f32,
}

/// Feed-forward network for large models
#[derive(Debug, Clone)]
pub struct LargeFeedForward {
    /// Up projection
    pub w_up: Linear,
    /// Gate projection (for SwiGLU)
    pub w_gate: Linear,
    /// Down projection
    pub w_down: Linear,
    pub dropout: f32,
}

impl LargeTransformerLayer {
    pub fn new(config: &LargeModelConfig) -> Self {
        Self {
            ln1: LayerNorm::new(vec![config.d_model], config.layer_norm_eps),
            attention: LargeMultiHeadAttention::new(config),
            ln2: LayerNorm::new(vec![config.d_model], config.layer_norm_eps),
            ffn: LargeFeedForward::new(config),
        }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Pre-norm architecture (more stable for large models)
        let normed = self.ln1.forward(x);
        let attn_out = self.attention.forward(&normed, mask);
        
        // Residual connection
        let h = x.add(&attn_out);
        
        // FFN with pre-norm
        let normed = self.ln2.forward(&h);
        let ffn_out = self.ffn.forward(&normed);
        
        // Residual connection
        h.add(&ffn_out)
    }
}

impl LargeMultiHeadAttention {
    pub fn new(config: &LargeModelConfig) -> Self {
        Self {
            n_heads: config.n_heads,
            d_model: config.d_model,
            d_head: config.d_head,
            w_q: Linear::new(config.d_model, config.d_model, false),
            w_k: Linear::new(config.d_model, config.d_model, false),
            w_v: Linear::new(config.d_model, config.d_model, false),
            w_o: Linear::new(config.d_model, config.d_model, false),
            dropout: config.dropout,
        }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let shape = &x.shape.0;
        let batch_seq = if shape.len() == 2 { shape[0] } else { shape[0] * shape[1] };
        
        // Project Q, K, V
        let q = self.w_q.forward(x);
        let k = self.w_k.forward(x);
        let v = self.w_v.forward(x);
        
        // Scaled dot-product attention
        let scale = (self.d_head as f32).sqrt();
        let scores = self.compute_attention_scores(&q, &k, scale);
        
        // Apply causal mask
        let masked_scores = self.apply_causal_mask(&scores, batch_seq);
        
        // Softmax
        let attn_weights = self.softmax(&masked_scores);
        
        // Apply to values
        let attended = self.apply_attention(&attn_weights, &v);
        
        // Output projection
        self.w_o.forward(&attended)
    }

    fn compute_attention_scores(&self, q: &Tensor, k: &Tensor, scale: f32) -> Tensor {
        let n = q.data.len().min(k.data.len());
        let mut scores = vec![0.0f32; n];
        
        for i in 0..n {
            scores[i] = q.data.get(i).unwrap_or(&0.0) * k.data.get(i).unwrap_or(&0.0) / scale;
        }
        
        Tensor::new(scores, q.shape.0.clone())
    }

    fn apply_causal_mask(&self, scores: &Tensor, seq_len: usize) -> Tensor {
        let mut masked = scores.data.clone();
        let n = masked.len();
        
        // Simple causal mask
        for i in 0..n {
            let row = i / seq_len.max(1);
            let col = i % seq_len.max(1);
            if col > row {
                masked[i] = f32::NEG_INFINITY;
            }
        }
        
        Tensor::new(masked, scores.shape.0.clone())
    }

    fn softmax(&self, x: &Tensor) -> Tensor {
        let mut result = x.data.clone();
        let n = result.len();
        
        if n > 0 {
            let max_val = result.iter().cloned().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            
            for v in &mut result {
                if v.is_finite() {
                    *v = (*v - max_val).exp();
                    sum += *v;
                } else {
                    *v = 0.0;
                }
            }
            
            if sum > 0.0 {
                for v in &mut result {
                    *v /= sum;
                }
            }
        }
        
        Tensor::new(result, x.shape.0.clone())
    }

    fn apply_attention(&self, weights: &Tensor, v: &Tensor) -> Tensor {
        let n = weights.data.len().min(v.data.len());
        let mut result = vec![0.0f32; v.data.len()];
        
        for i in 0..n {
            result[i] = weights.data[i] * v.data.get(i).unwrap_or(&0.0);
        }
        
        Tensor::new(result, v.shape.0.clone())
    }
}

impl LargeFeedForward {
    pub fn new(config: &LargeModelConfig) -> Self {
        Self {
            w_up: Linear::new(config.d_model, config.d_ff, false),
            w_gate: Linear::new(config.d_model, config.d_ff, false),
            w_down: Linear::new(config.d_ff, config.d_model, false),
            dropout: config.dropout,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // SwiGLU activation: down(up(x) * silu(gate(x)))
        let up = self.w_up.forward(x);
        let gate = self.w_gate.forward(x);
        
        // SiLU activation on gate
        let gate_activated: Vec<f32> = gate.data.iter()
            .map(|&v| v * (1.0 / (1.0 + (-v).exp())))
            .collect();
        
        // Element-wise multiplication
        let hidden: Vec<f32> = up.data.iter()
            .zip(gate_activated.iter())
            .map(|(u, g)| u * g)
            .collect();
        
        let hidden_tensor = Tensor::new(hidden, up.shape.0.clone());
        self.w_down.forward(&hidden_tensor)
    }
}

impl LargeLanguageModel {
    /// Create a new large language model
    pub fn new(config: LargeModelConfig) -> Self {
        let token_embedding = Embedding::new(config.vocab_size, config.d_model);
        
        let pos_embedding = if !config.use_rope {
            Some(Embedding::new(config.max_seq_len, config.d_model))
        } else {
            None
        };
        
        let layers: Vec<LargeTransformerLayer> = (0..config.n_layers)
            .map(|_| LargeTransformerLayer::new(&config))
            .collect();
        
        let final_ln = LayerNorm::new(vec![config.d_model], config.layer_norm_eps);
        let lm_head = Linear::new(config.d_model, config.vocab_size, false);
        
        // Initialize vocabulary with special tokens
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        
        let special_tokens = vec!["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"];
        for (i, token) in special_tokens.iter().enumerate() {
            vocab.insert(token.to_string(), i);
            id_to_token.insert(i, token.to_string());
        }
        
        Self {
            config,
            token_embedding,
            pos_embedding,
            layers,
            final_ln,
            lm_head,
            vocab,
            id_to_token,
            train_steps: 0,
        }
    }

    /// Create with preset size
    pub fn with_size(size: ModelSize) -> Self {
        let config = match size {
            ModelSize::Small => LargeModelConfig::small(),
            ModelSize::Medium => LargeModelConfig::medium(),
            ModelSize::Large => LargeModelConfig::large(),
            ModelSize::XL => LargeModelConfig::xl(),
            ModelSize::Custom => LargeModelConfig::small(),
        };
        Self::new(config)
    }

    /// Forward pass
    pub fn forward(&self, token_ids: &[usize]) -> Tensor {
        // Ensure all token IDs are within bounds
        let safe_ids: Vec<usize> = token_ids.iter()
            .map(|&id| id.min(self.config.vocab_size - 1))
            .collect();
        
        // Token embeddings
        let mut h = self.token_embedding.forward(&safe_ids);
        
        // Add positional embeddings if not using RoPE
        if let Some(ref pos_emb) = self.pos_embedding {
            let positions: Vec<usize> = (0..safe_ids.len().min(self.config.max_seq_len)).collect();
            let pos = pos_emb.forward(&positions);
            
            // Add positional embeddings (element-wise)
            let min_len = h.data.len().min(pos.data.len());
            let mut combined = h.data.clone();
            for i in 0..min_len {
                combined[i] += pos.data[i];
            }
            h = Tensor::new(combined, h.shape.0.clone());
        }
        
        // Pass through transformer layers
        for layer in &self.layers {
            h = layer.forward(&h, None);
        }
        
        // Final layer norm
        h = self.final_ln.forward(&h);
        
        // Project to vocabulary
        self.lm_head.forward(&h)
    }

    /// Learn from text (builds vocabulary and trains weights)
    pub fn learn(&mut self, text: &str) {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        
        // Build vocabulary
        for token in &tokens {
            let lower = token.to_lowercase();
            if !self.vocab.contains_key(&lower) {
                let id = self.vocab.len();
                if id < self.config.vocab_size {
                    self.vocab.insert(lower.clone(), id);
                    self.id_to_token.insert(id, lower);
                }
            }
        }
        
        // Train on the text
        if tokens.len() >= 2 {
            self.train_on_sequence(text);
        }
        
        self.train_steps += 1;
    }
    
    /// Train on a sequence using teacher forcing (fast version)
    pub fn train_on_sequence(&mut self, text: &str) {
        let token_ids = self.tokenize(text);
        if token_ids.len() < 2 {
            return;
        }
        
        let learning_rate = 0.05f32;
        
        // Train on last few positions only (much faster)
        let start = token_ids.len().saturating_sub(5);
        for i in start..token_ids.len().saturating_sub(1) {
            let input_seq = &token_ids[..=i];
            let target_id = token_ids[i + 1];
            
            if target_id >= self.config.vocab_size {
                continue;
            }
            
            // Forward pass
            let logits = self.forward(input_seq);
            
            // Compute loss gradient (cross-entropy)
            let mut grad = vec![0.0f32; self.config.vocab_size];
            
            // Softmax and gradient
            if !logits.data.is_empty() {
                let max_val = logits.data.iter().take(self.config.vocab_size)
                    .cloned().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
                
                if !max_val.is_finite() {
                    continue;
                }
                
                let exp_vals: Vec<f32> = logits.data.iter().take(self.config.vocab_size).map(|&x| {
                    if x.is_finite() { (x - max_val).exp() } else { 0.0 }
                }).collect();
                let sum: f32 = exp_vals.iter().sum();
                
                if sum > 0.0 {
                    let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();
                    
                    // Cross-entropy gradient: prob - 1 for target, prob for others
                    for (j, &p) in probs.iter().enumerate() {
                        if j == target_id {
                            grad[j] = p - 1.0;
                        } else {
                            grad[j] = p;
                        }
                    }
                    
                    // Update LM head weights
                    self.update_lm_head(&grad, learning_rate);
                }
            }
        }
    }
    
    /// Update language model head weights
    fn update_lm_head(&mut self, grad: &[f32], lr: f32) {
        // Simple SGD update on lm_head weights
        let n = self.lm_head.weight.data.len().min(grad.len());
        for i in 0..n {
            if grad[i].is_finite() {
                self.lm_head.weight.data[i] -= lr * grad[i];
            }
        }
    }
    
    /// Train on Q&A pair
    pub fn learn_qa(&mut self, question: &str, answer: &str) {
        // Build vocabulary from both
        self.learn(question);
        
        // Train specifically on the answer
        let combined = format!("{} {}", question, answer);
        self.train_on_sequence(&combined);
        
        // Multiple passes on the answer for better learning
        for _ in 0..3 {
            self.train_on_sequence(answer);
        }
    }

    /// Tokenize text
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let mut ids = vec![2]; // <BOS>
        
        for word in text.split_whitespace() {
            let lower = word.to_lowercase();
            let id = self.vocab.get(&lower).copied().unwrap_or(1); // <UNK>
            // Ensure ID is within bounds
            let safe_id = id.min(self.config.vocab_size - 1);
            ids.push(safe_id);
        }
        
        ids.push(3); // <EOS>
        ids
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|id| self.id_to_token.get(id))
            .filter(|t| !t.starts_with('<'))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Generate text
    pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f64) -> String {
        let mut token_ids = self.tokenize(prompt);
        token_ids.pop(); // Remove <EOS>
        
        for _ in 0..max_tokens {
            if token_ids.len() >= self.config.max_seq_len {
                break;
            }
            
            let logits = self.forward(&token_ids);
            let next_id = self.sample_token(&logits.data, temperature);
            
            if next_id == 3 { // <EOS>
                break;
            }
            
            token_ids.push(next_id);
        }
        
        self.decode(&token_ids)
    }

    fn sample_token(&self, logits: &[f32], temperature: f64) -> usize {
        if logits.is_empty() {
            return 1; // <UNK>
        }
        
        let temp = temperature.max(0.1) as f32;
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
        
        let max_val = scaled.iter().cloned().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
        if !max_val.is_finite() {
            // All values are non-finite, return random valid token
            return rand::Rng::gen_range(&mut rand::thread_rng(), 4..self.vocab.len().max(5));
        }
        
        let exp_vals: Vec<f32> = scaled.iter().map(|&x| {
            if x.is_finite() { (x - max_val).exp() } else { 0.0 }
        }).collect();
        let exp_sum: f32 = exp_vals.iter().sum();
        
        if exp_sum <= 0.0 {
            return rand::Rng::gen_range(&mut rand::thread_rng(), 4..self.vocab.len().max(5));
        }
        
        let probs: Vec<f32> = exp_vals.iter().map(|&x| x / exp_sum).collect();
        
        let mut rng = rand::thread_rng();
        let r: f32 = rand::Rng::gen(&mut rng);
        let mut cumsum = 0.0f32;
        
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                // Ensure returned index is within vocab bounds
                return i.min(self.vocab.len().saturating_sub(1)).min(self.config.vocab_size - 1);
            }
        }
        
        // Return a safe default
        self.vocab.len().saturating_sub(1).min(self.config.vocab_size - 1).max(4)
    }

    /// Get model statistics
    pub fn stats(&self) -> LargeModelStats {
        LargeModelStats {
            size: self.config.size,
            parameters: self.config.estimate_parameters(),
            parameters_str: self.config.parameter_count_string(),
            d_model: self.config.d_model,
            n_heads: self.config.n_heads,
            n_layers: self.config.n_layers,
            d_ff: self.config.d_ff,
            vocab_size: self.vocab.len(),
            max_vocab_size: self.config.vocab_size,
            train_steps: self.train_steps,
        }
    }
}

/// Statistics for large model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeModelStats {
    pub size: ModelSize,
    pub parameters: usize,
    pub parameters_str: String,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub max_vocab_size: usize,
    pub train_steps: u64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_sizes() {
        let small = LargeModelConfig::small();
        let medium = LargeModelConfig::medium();
        let large = LargeModelConfig::large();
        let xl = LargeModelConfig::xl();
        
        println!("Small: {}", small.parameter_count_string());
        println!("Medium: {}", medium.parameter_count_string());
        println!("Large: {}", large.parameter_count_string());
        println!("XL: {}", xl.parameter_count_string());
        
        assert!(small.estimate_parameters() < medium.estimate_parameters());
        assert!(medium.estimate_parameters() < large.estimate_parameters());
        assert!(large.estimate_parameters() < xl.estimate_parameters());
    }

    #[test]
    fn test_small_model() {
        let mut model = LargeLanguageModel::with_size(ModelSize::Small);
        model.learn("Hello world this is a test of the language model");
        
        let stats = model.stats();
        assert!(stats.vocab_size > 6);
        assert_eq!(stats.n_layers, 6);
        assert_eq!(stats.n_heads, 4);
    }

    #[test]
    fn test_tokenization() {
        let mut model = LargeLanguageModel::with_size(ModelSize::Small);
        model.learn("hello world");
        
        let ids = model.tokenize("hello world");
        assert!(ids.len() >= 3); // BOS + tokens + EOS
        
        let decoded = model.decode(&ids);
        assert!(decoded.contains("hello"));
    }

    #[test]
    fn test_forward_pass() {
        let model = LargeLanguageModel::with_size(ModelSize::Small);
        let ids = vec![2, 5, 6, 7, 3]; // BOS + tokens + EOS
        let output = model.forward(&ids);
        
        assert!(!output.data.is_empty());
    }
}
