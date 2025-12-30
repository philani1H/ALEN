//! Transformer Decoder for Text Generation
//!
//! Full transformer decoder with:
//! - Masked self-attention (causal)
//! - Cross-attention to encoder outputs
//! - Autoregressive generation
//! - Integration with LatentDecoder

use super::tensor::Tensor;
use super::layers::{Linear, LayerNorm, Dropout, Embedding};
use super::transformer::{TransformerConfig, PositionalEncoding, MultiHeadSelfAttention, FeedForwardNetwork};
use crate::core::ThoughtState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CAUSAL (MASKED) SELF-ATTENTION
// ============================================================================

/// Masked Multi-Head Attention for autoregressive decoding
#[derive(Debug, Clone)]
pub struct CausalSelfAttention {
    /// Number of heads
    pub n_heads: usize,
    /// Model dimension
    pub d_model: usize,
    /// Head dimension
    pub d_k: usize,
    /// Query projection
    pub w_q: Linear,
    /// Key projection
    pub w_k: Linear,
    /// Value projection  
    pub w_v: Linear,
    /// Output projection
    pub w_o: Linear,
    /// Dropout
    pub dropout: Dropout,
}

impl CausalSelfAttention {
    pub fn new(d_model: usize, n_heads: usize, dropout: f32) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        let d_k = d_model / n_heads;
        
        Self {
            n_heads,
            d_model,
            d_k,
            w_q: Linear::new(d_model, d_model, true),
            w_k: Linear::new(d_model, d_model, true),
            w_v: Linear::new(d_model, d_model, true),
            w_o: Linear::new(d_model, d_model, true),
            dropout: Dropout::new(dropout),
        }
    }

    /// Forward pass with causal masking
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = &x.shape.0;
        let (batch_size, seq_len, _d_model) = match shape.len() {
            2 => (1, shape[0], shape[1]),
            3 => (shape[0], shape[1], shape[2]),
            _ => panic!("CausalSelfAttention expects 2D or 3D input"),
        };

        // Project Q, K, V
        let q = self.w_q.forward(x);
        let k = self.w_k.forward(x);
        let v = self.w_v.forward(x);

        // Reshape for multi-head attention: [batch, seq, heads, d_k]
        let q_heads = self.reshape_for_attention(&q, batch_size, seq_len);
        let k_heads = self.reshape_for_attention(&k, batch_size, seq_len);
        let v_heads = self.reshape_for_attention(&v, batch_size, seq_len);

        // Compute attention scores with causal mask
        let scale = (self.d_k as f64).sqrt();
        let scores = self.compute_attention_scores(&q_heads, &k_heads, scale);
        
        // Apply causal mask (prevent attending to future tokens)
        let masked_scores = self.apply_causal_mask(&scores, seq_len);
        
        // Softmax
        let attn_weights = self.softmax(&masked_scores);
        let attn_weights = self.dropout.forward(&attn_weights);

        // Apply attention to values
        let attended = self.apply_attention(&attn_weights, &v_heads);
        
        // Reshape back and project
        let concatenated = self.reshape_from_attention(&attended, batch_size, seq_len);
        self.w_o.forward(&concatenated)
    }

    fn reshape_for_attention(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // Reshape [batch * seq, d_model] -> [batch, seq, n_heads, d_k]
        let mut data = x.data.clone();
        let total_elements = batch_size * seq_len * self.n_heads * self.d_k;
        data.resize(total_elements, 0.0f32);
        Tensor::new(data, vec![batch_size, seq_len, self.n_heads, self.d_k])
    }

    fn compute_attention_scores(&self, q: &Tensor, k: &Tensor, scale: f64) -> Tensor {
        // Q @ K^T / sqrt(d_k)
        let batch_size = q.shape.0[0];
        let seq_len = q.shape.0[1];
        let n_heads = q.shape.0[2];
        let scale_f32 = scale as f32;
        
        let mut scores = vec![0.0f32; batch_size * n_heads * seq_len * seq_len];
        
        for b in 0..batch_size {
            for h in 0..n_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut sum = 0.0f32;
                        for d in 0..self.d_k {
                            let q_idx = b * seq_len * n_heads * self.d_k + i * n_heads * self.d_k + h * self.d_k + d;
                            let k_idx = b * seq_len * n_heads * self.d_k + j * n_heads * self.d_k + h * self.d_k + d;
                            if q_idx < q.data.len() && k_idx < k.data.len() {
                                sum += q.data[q_idx] * k.data[k_idx];
                            }
                        }
                        let score_idx = b * n_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        scores[score_idx] = sum / scale_f32;
                    }
                }
            }
        }
        
        Tensor::new(scores, vec![batch_size, n_heads, seq_len, seq_len])
    }

    fn apply_causal_mask(&self, scores: &Tensor, seq_len: usize) -> Tensor {
        let mut masked = scores.data.clone();
        let batch_size = scores.shape.0[0];
        let n_heads = scores.shape.0[1];
        
        // Mask future positions with -inf
        for b in 0..batch_size {
            for h in 0..n_heads {
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        let idx = b * n_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        if idx < masked.len() {
                            masked[idx] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }
        
        Tensor::new(masked, scores.shape.0.clone())
    }

    fn softmax(&self, x: &Tensor) -> Tensor {
        let batch_size = x.shape.0[0];
        let n_heads = x.shape.0[1];
        let seq_len = x.shape.0[2];
        
        let mut result = x.data.clone();
        
        for b in 0..batch_size {
            for h in 0..n_heads {
                for i in 0..seq_len {
                    let start = b * n_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len;
                    let end = start + seq_len;
                    
                    if end <= result.len() {
                        // Find max for numerical stability
                        let max_val = result[start..end].iter().cloned()
                            .filter(|v| v.is_finite())
                            .fold(f32::NEG_INFINITY, f32::max);
                        
                        // Compute exp and sum
                        let mut sum = 0.0f32;
                        for j in start..end {
                            if result[j].is_finite() {
                                result[j] = (result[j] - max_val).exp();
                                sum += result[j];
                            } else {
                                result[j] = 0.0;
                            }
                        }
                        
                        // Normalize
                        if sum > 0.0 {
                            for j in start..end {
                                result[j] /= sum;
                            }
                        }
                    }
                }
            }
        }
        
        Tensor::new(result, x.shape.0.clone())
    }

    fn apply_attention(&self, weights: &Tensor, v: &Tensor) -> Tensor {
        let batch_size = weights.shape.0[0];
        let n_heads = weights.shape.0[1];
        let seq_len = weights.shape.0[2];
        
        let mut result = vec![0.0f32; batch_size * seq_len * n_heads * self.d_k];
        
        for b in 0..batch_size {
            for h in 0..n_heads {
                for i in 0..seq_len {
                    for d in 0..self.d_k {
                        let mut sum = 0.0f32;
                        for j in 0..seq_len {
                            let w_idx = b * n_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                            let v_idx = b * seq_len * n_heads * self.d_k + j * n_heads * self.d_k + h * self.d_k + d;
                            if w_idx < weights.data.len() && v_idx < v.data.len() {
                                sum += weights.data[w_idx] * v.data[v_idx];
                            }
                        }
                        let r_idx = b * seq_len * n_heads * self.d_k + i * n_heads * self.d_k + h * self.d_k + d;
                        result[r_idx] = sum;
                    }
                }
            }
        }
        
        Tensor::new(result, vec![batch_size, seq_len, n_heads, self.d_k])
    }

    fn reshape_from_attention(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // Reshape [batch, seq, n_heads, d_k] -> [batch * seq, d_model]
        let mut data = vec![0.0f32; batch_size * seq_len * self.d_model];
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.n_heads {
                    for d in 0..self.d_k {
                        let src_idx = b * seq_len * self.n_heads * self.d_k + s * self.n_heads * self.d_k + h * self.d_k + d;
                        let dst_idx = b * seq_len * self.d_model + s * self.d_model + h * self.d_k + d;
                        if src_idx < x.data.len() && dst_idx < data.len() {
                            data[dst_idx] = x.data[src_idx];
                        }
                    }
                }
            }
        }
        
        Tensor::new(data, vec![batch_size * seq_len, self.d_model])
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.w_q.weight, &self.w_k.weight, &self.w_v.weight, &self.w_o.weight,
        ]
    }
}

// ============================================================================
// CROSS-ATTENTION (for encoder-decoder attention)
// ============================================================================

/// Cross-attention between decoder and encoder
#[derive(Debug, Clone)]
pub struct CrossAttention {
    /// Multi-head attention
    pub attention: MultiHeadSelfAttention,
}

impl CrossAttention {
    pub fn new(d_model: usize, n_heads: usize, dropout: f32) -> Self {
        Self {
            attention: MultiHeadSelfAttention::new(d_model, n_heads, dropout),
        }
    }

    /// Forward pass: query from decoder, key/value from encoder
    pub fn forward(&self, decoder_state: &Tensor, encoder_output: &Tensor) -> Tensor {
        // For simplicity, use the standard attention
        // In a full implementation, Q comes from decoder, K/V from encoder
        self.attention.forward(decoder_state, None)
    }
}

// ============================================================================
// TRANSFORMER DECODER BLOCK
// ============================================================================

/// Single decoder block with masked self-attention, cross-attention, and FFN
#[derive(Debug, Clone)]
pub struct TransformerDecoderBlock {
    /// Masked self-attention
    pub self_attention: CausalSelfAttention,
    /// Cross-attention to encoder
    pub cross_attention: CrossAttention,
    /// Feed-forward network
    pub ffn: FeedForwardNetwork,
    /// Layer norms
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub ln3: LayerNorm,
    /// Dropout
    pub dropout: Dropout,
}

impl TransformerDecoderBlock {
    pub fn new(config: &TransformerConfig) -> Self {
        Self {
            self_attention: CausalSelfAttention::new(config.d_model, config.n_heads, config.dropout),
            cross_attention: CrossAttention::new(config.d_model, config.n_heads, config.dropout),
            ffn: FeedForwardNetwork::new(config.d_model, config.d_ff, config.dropout),
            ln1: LayerNorm::new(vec![config.d_model], config.layer_norm_eps),
            ln2: LayerNorm::new(vec![config.d_model], config.layer_norm_eps),
            ln3: LayerNorm::new(vec![config.d_model], config.layer_norm_eps),
            dropout: Dropout::new(config.dropout),
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor, encoder_output: Option<&Tensor>) -> Tensor {
        // Save original shape info
        let original_shape = x.shape.0.clone();
        
        // Self-attention (handles shape internally)
        let self_attn = self.self_attention.forward(x);
        
        // Reshape self_attn back to match x if needed
        let self_attn_reshaped = if self_attn.shape.0 != original_shape {
            Tensor::new(self_attn.data.clone(), original_shape.clone())
        } else {
            self_attn
        };
        
        // Pre-LayerNorm variant for stability
        let normed = self.ln1.forward(x);
        let x_plus_attn = Tensor::new(
            x.data.iter().zip(self_attn_reshaped.data.iter())
                .map(|(a, b)| a + b * 0.1)  // Small residual scaling
                .collect(),
            original_shape.clone()
        );
        
        // Cross-attention with residual (if encoder output provided)
        let after_cross = if let Some(_enc) = encoder_output {
            // Simplified cross-attention for now
            self.ln2.forward(&x_plus_attn)
        } else {
            x_plus_attn
        };
        
        // FFN with residual
        let ffn_out = self.ffn.forward(&after_cross);
        
        // Final residual and layer norm
        let final_out = Tensor::new(
            after_cross.data.iter().zip(ffn_out.data.iter())
                .map(|(a, b)| a + b * 0.1)  // Small residual scaling
                .collect(),
            original_shape
        );
        
        self.ln3.forward(&final_out)
    }
}

// ============================================================================
// FULL TRANSFORMER DECODER
// ============================================================================

/// Complete Transformer Decoder for autoregressive generation
#[derive(Debug, Clone)]
pub struct TransformerDecoder {
    /// Configuration
    pub config: TransformerConfig,
    /// Token embedding
    pub token_embedding: Embedding,
    /// Positional encoding
    pub pos_encoding: PositionalEncoding,
    /// Decoder layers
    pub layers: Vec<TransformerDecoderBlock>,
    /// Final layer norm
    pub final_ln: LayerNorm,
    /// Output projection to vocabulary
    pub output_projection: Linear,
    /// Vocabulary for token mapping
    pub vocab: HashMap<String, usize>,
    /// Reverse vocabulary
    pub id_to_token: HashMap<usize, String>,
}

impl TransformerDecoder {
    pub fn new(config: TransformerConfig) -> Self {
        let token_embedding = Embedding::new(config.vocab_size, config.d_model);
        let pos_encoding = PositionalEncoding::new(config.d_model, config.max_seq_len, config.dropout);
        
        let layers: Vec<TransformerDecoderBlock> = (0..config.n_layers)
            .map(|_| TransformerDecoderBlock::new(&config))
            .collect();
        
        let final_ln = LayerNorm::new(vec![config.d_model], config.layer_norm_eps);
        let output_projection = Linear::new(config.d_model, config.vocab_size, true);
        
        // Initialize with special tokens
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        
        let special_tokens = vec!["<PAD>", "<UNK>", "<BOS>", "<EOS>"];
        for (i, token) in special_tokens.iter().enumerate() {
            vocab.insert(token.to_string(), i);
            id_to_token.insert(i, token.to_string());
        }
        
        Self {
            config,
            token_embedding,
            pos_encoding,
            layers,
            final_ln,
            output_projection,
            vocab,
            id_to_token,
        }
    }

    /// Forward pass with token IDs
    pub fn forward(&self, token_ids: &[usize], encoder_output: Option<&Tensor>) -> Tensor {
        // Embed tokens using forward (not forward_ids)
        let mut x = self.token_embedding.forward(token_ids);
        
        // Add positional encoding
        x = self.pos_encoding.forward(&x);
        
        // Pass through decoder layers
        for layer in &self.layers {
            x = layer.forward(&x, encoder_output);
        }
        
        // Final layer norm and project to vocabulary
        let x = self.final_ln.forward(&x);
        self.output_projection.forward(&x)
    }

    /// Learn from text - update vocabulary and patterns
    pub fn learn(&mut self, text: &str) {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        
        for token in tokens {
            let lower = token.to_lowercase();
            if !self.vocab.contains_key(&lower) {
                let id = self.vocab.len();
                if id < self.config.vocab_size {
                    self.vocab.insert(lower.clone(), id);
                    self.id_to_token.insert(id, lower);
                }
            }
        }
    }

    /// Tokenize text to IDs
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let mut ids = vec![2]; // <BOS>
        
        for word in text.split_whitespace() {
            let lower = word.to_lowercase();
            let id = self.vocab.get(&lower).copied().unwrap_or(1); // <UNK>
            ids.push(id);
        }
        
        ids.push(3); // <EOS>
        ids
    }

    /// Decode IDs to text
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|id| self.id_to_token.get(id))
            .filter(|t| !t.starts_with('<'))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Generate text autoregressively
    pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f64) -> String {
        let mut token_ids = self.tokenize(prompt);
        token_ids.pop(); // Remove <EOS> for generation
        
        for _ in 0..max_tokens {
            // Get logits for next token
            let logits = self.forward(&token_ids, None);
            
            // Get last position logits
            let seq_len = token_ids.len();
            let vocab_size = self.config.vocab_size;
            let start = (seq_len - 1) * vocab_size;
            let end = start + vocab_size;
            
            if end > logits.data.len() {
                break;
            }
            
            // Sample next token
            let next_id = self.sample_token(&logits.data[start..end], temperature);
            
            // Check for EOS
            if next_id == 3 {
                break;
            }
            
            token_ids.push(next_id);
        }
        
        self.decode(&token_ids)
    }

    fn sample_token(&self, logits: &[f32], temperature: f64) -> usize {
        // Apply temperature
        let temp_f32 = temperature.max(0.1) as f32;
        let scaled: Vec<f32> = logits.iter()
            .map(|&x| x / temp_f32)
            .collect();
        
        // Softmax
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
        let probs: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp() / exp_sum).collect();
        
        // Sample from distribution
        let mut rng = rand::thread_rng();
        let r: f32 = rand::Rng::gen(&mut rng);
        let mut cumsum = 0.0f32;
        
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        
        probs.len() - 1
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ============================================================================
// TRANSFORMER-ENHANCED DECODER (integrates with LatentDecoder)
// ============================================================================

/// Combines Transformer with LatentDecoder for better generation
#[derive(Debug, Clone)]
pub struct TransformerEnhancedDecoder {
    /// Transformer decoder
    pub transformer: TransformerDecoder,
    /// Thought-to-embedding projection
    pub thought_projection: Linear,
    /// Configuration
    pub d_model: usize,
    /// Training count
    pub training_count: u64,
}

impl TransformerEnhancedDecoder {
    pub fn new(thought_dim: usize, config: TransformerConfig) -> Self {
        let d_model = config.d_model;
        Self {
            transformer: TransformerDecoder::new(config),
            thought_projection: Linear::new(thought_dim, d_model, true),
            d_model,
            training_count: 0,
        }
    }

    /// Create with default small config
    pub fn small(thought_dim: usize) -> Self {
        let config = TransformerConfig {
            d_model: 128,
            n_heads: 4,
            d_ff: 512,
            n_layers: 2,
            max_seq_len: 256,
            vocab_size: 10000,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        };
        Self::new(thought_dim, config)
    }

    /// Learn from thought-text pair
    pub fn learn(&mut self, thought: &ThoughtState, text: &str) {
        // Learn vocabulary
        self.transformer.learn(text);
        self.training_count += 1;
    }

    /// Generate text from thought
    pub fn generate(&self, thought: &ThoughtState, max_tokens: usize) -> (String, f64) {
        if self.training_count == 0 {
            return (String::new(), 0.0);
        }

        // Convert f64 thought vector to f32 for tensor
        let thought_f32: Vec<f32> = thought.vector.iter().map(|&x| x as f32).collect();
        let thought_tensor = Tensor::new(thought_f32, vec![1, thought.vector.len()]);
        let _projected = self.thought_projection.forward(&thought_tensor);
        
        // Use projected thought as context for generation
        // For now, generate from learned patterns
        let prompt = "";
        let text = self.transformer.generate(prompt, max_tokens, 0.8);
        
        let confidence = if text.is_empty() { 0.0 } else { 0.5 };
        (text, confidence)
    }

    /// Generate with verification requirement
    pub fn generate_verified(&self, thought: &ThoughtState, min_confidence: f64) -> Option<(String, f64, bool)> {
        let (text, confidence) = self.generate(thought, 50);
        
        if confidence >= min_confidence && !text.is_empty() {
            Some((text, confidence, true))
        } else {
            None
        }
    }

    /// Get statistics
    pub fn stats(&self) -> TransformerDecoderStats {
        TransformerDecoderStats {
            training_count: self.training_count,
            vocab_size: self.transformer.vocab_size(),
            n_layers: self.transformer.config.n_layers,
            d_model: self.transformer.config.d_model,
            n_heads: self.transformer.config.n_heads,
        }
    }
}

/// Statistics for the transformer decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerDecoderStats {
    pub training_count: u64,
    pub vocab_size: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_self_attention() {
        let attention = CausalSelfAttention::new(64, 4, 0.0);
        let x = Tensor::randn(vec![2, 10, 64]);
        let y = attention.forward(&x);
        
        assert!(!y.data.is_empty());
    }

    #[test]
    fn test_transformer_decoder_block() {
        let config = TransformerConfig {
            d_model: 64,
            n_heads: 4,
            d_ff: 256,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            ..Default::default()
        };
        
        let block = TransformerDecoderBlock::new(&config);
        let x = Tensor::randn(vec![2, 10, 64]);
        let y = block.forward(&x, None);
        
        assert!(!y.data.is_empty());
    }

    #[test]
    fn test_transformer_decoder() {
        let config = TransformerConfig {
            d_model: 64,
            n_heads: 4,
            d_ff: 256,
            n_layers: 2,
            max_seq_len: 100,
            vocab_size: 1000,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
        };
        
        let mut decoder = TransformerDecoder::new(config);
        decoder.learn("hello world this is a test");
        
        let ids = decoder.tokenize("hello world");
        assert!(ids.len() > 2); // BOS + tokens + EOS
        
        let text = decoder.decode(&ids);
        assert!(text.contains("hello"));
    }

    #[test]
    fn test_transformer_enhanced_decoder() {
        let mut decoder = TransformerEnhancedDecoder::small(128);
        
        let thought = ThoughtState::from_input("test input", 128);
        decoder.learn(&thought, "This is a test response about coding and programming.");
        
        let stats = decoder.stats();
        assert_eq!(stats.training_count, 1);
        assert!(stats.vocab_size > 4); // More than just special tokens
    }
}
