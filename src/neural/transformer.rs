//! Transformer Architecture
//!
//! Full transformer encoder implementation:
//! - Multi-head self-attention
//! - Positional encoding
//! - Feed-forward networks
//! - Residual connections + LayerNorm

use super::tensor::Tensor;
use super::layers::{Linear, LayerNorm, Dropout};
use serde::{Deserialize, Serialize};

/// Transformer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Model dimension (d_model)
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Feed-forward hidden dimension
    pub d_ff: usize,
    /// Number of encoder layers
    pub n_layers: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Layer norm epsilon
    pub layer_norm_eps: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            n_heads: 8,
            d_ff: 2048,
            n_layers: 6,
            max_seq_len: 512,
            vocab_size: 32000,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }
}

impl TransformerConfig {
    /// Small config for testing/prototyping
    pub fn small() -> Self {
        Self {
            d_model: 256,
            n_heads: 4,
            d_ff: 1024,
            n_layers: 4,
            max_seq_len: 256,
            vocab_size: 10000,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }

    /// Medium config
    pub fn medium() -> Self {
        Self {
            d_model: 512,
            n_heads: 8,
            d_ff: 2048,
            n_layers: 6,
            max_seq_len: 512,
            vocab_size: 32000,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }

    /// Large config
    pub fn large() -> Self {
        Self {
            d_model: 768,
            n_heads: 12,
            d_ff: 3072,
            n_layers: 12,
            max_seq_len: 1024,
            vocab_size: 50000,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }
}

/// Sinusoidal Positional Encoding
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    /// Precomputed positional encodings [max_seq_len, d_model]
    pub encoding: Tensor,
    /// Model dimension
    pub d_model: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Dropout
    pub dropout: Dropout,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_seq_len: usize, dropout: f32) -> Self {
        // Compute sinusoidal positional encodings
        let mut data = vec![0.0; max_seq_len * d_model];
        
        for pos in 0..max_seq_len {
            for i in 0..d_model {
                let angle = pos as f32 / (10000.0_f32).powf(2.0 * (i / 2) as f32 / d_model as f32);
                data[pos * d_model + i] = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
            }
        }
        
        Self {
            encoding: Tensor::new(data, vec![max_seq_len, d_model]),
            d_model,
            max_seq_len,
            dropout: Dropout::new(dropout),
        }
    }

    /// Add positional encoding to input
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let ndim = x.shape.ndim();

        // Handle both 2D [batch, d_model] and 3D [batch, seq_len, d_model] inputs
        let (batch, seq_len, d_model) = if ndim == 2 {
            // 2D input: [batch, d_model] - treat as single token (seq_len=1)
            (x.shape.dim(0), 1, x.shape.dim(1))
        } else if ndim == 3 {
            // 3D input: [batch, seq_len, d_model]
            (x.shape.dim(0), x.shape.dim(1), x.shape.dim(2))
        } else {
            panic!("PositionalEncoding expects 2D or 3D input, got {}D", ndim);
        };

        assert_eq!(d_model, self.d_model, "d_model mismatch");
        assert!(seq_len <= self.max_seq_len, "Sequence too long: {} > {}", seq_len, self.max_seq_len);

        let mut output = x.to_vec();

        for b in 0..batch {
            for s in 0..seq_len {
                for d in 0..d_model {
                    let idx = b * seq_len * d_model + s * d_model + d;
                    let pe_idx = s * self.d_model + d;
                    output[idx] += self.encoding.data[pe_idx];
                }
            }
        }

        let result = Tensor::new(output, x.shape.clone());
        self.dropout.forward(&result)
    }
}

/// Scaled Dot-Product Attention
pub fn scaled_dot_product_attention(
    query: &Tensor,  // [batch, heads, seq_q, d_k]
    key: &Tensor,    // [batch, heads, seq_k, d_k]
    value: &Tensor,  // [batch, heads, seq_k, d_v]
    mask: Option<&Tensor>,
) -> Tensor {
    let d_k = query.shape.dim(3) as f32;
    
    // Q @ K^T / sqrt(d_k)
    // For simplicity, we'll compute this per batch and head
    let batch = query.shape.dim(0);
    let heads = query.shape.dim(1);
    let seq_q = query.shape.dim(2);
    let seq_k = key.shape.dim(2);
    let d_v = value.shape.dim(3);
    
    let mut scores = vec![0.0; batch * heads * seq_q * seq_k];
    
    // Compute attention scores
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..seq_q {
                for j in 0..seq_k {
                    let mut dot = 0.0;
                    for k in 0..d_k as usize {
                        let q_idx = b * heads * seq_q * d_k as usize 
                            + h * seq_q * d_k as usize 
                            + i * d_k as usize + k;
                        let k_idx = b * heads * seq_k * d_k as usize 
                            + h * seq_k * d_k as usize 
                            + j * d_k as usize + k;
                        dot += query.data[q_idx] * key.data[k_idx];
                    }
                    scores[b * heads * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j] = 
                        dot / d_k.sqrt();
                }
            }
        }
    }
    
    // Apply mask if provided
    if let Some(m) = mask {
        for i in 0..scores.len() {
            if m.data.get(i % m.data.len()).copied().unwrap_or(1.0) == 0.0 {
                scores[i] = f32::NEG_INFINITY;
            }
        }
    }
    
    // Softmax over last dimension (seq_k)
    let scores_tensor = Tensor::new(scores, vec![batch, heads, seq_q, seq_k]);
    let attn_weights = softmax_last_dim(&scores_tensor);
    
    // Attention @ Value
    let mut output = vec![0.0; batch * heads * seq_q * d_v];
    
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..seq_q {
                for v in 0..d_v {
                    let mut sum = 0.0;
                    for j in 0..seq_k {
                        let attn_idx = b * heads * seq_q * seq_k 
                            + h * seq_q * seq_k + i * seq_k + j;
                        let v_idx = b * heads * seq_k * d_v 
                            + h * seq_k * d_v + j * d_v + v;
                        sum += attn_weights.data[attn_idx] * value.data[v_idx];
                    }
                    output[b * heads * seq_q * d_v + h * seq_q * d_v + i * d_v + v] = sum;
                }
            }
        }
    }
    
    Tensor::new(output, vec![batch, heads, seq_q, d_v])
}

/// Softmax over last dimension
fn softmax_last_dim(x: &Tensor) -> Tensor {
    let last_dim = x.shape.dim(x.shape.ndim() - 1);
    let outer_size = x.shape.numel() / last_dim;
    
    let mut data = vec![0.0; x.shape.numel()];
    
    for i in 0..outer_size {
        let start = i * last_dim;
        let end = start + last_dim;
        
        // Find max for numerical stability
        let max_val = x.data[start..end].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        // Compute exp and sum
        let mut sum = 0.0;
        for j in start..end {
            let exp_val = (x.data[j] - max_val).exp();
            data[j] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for j in start..end {
            data[j] /= sum + 1e-10;
        }
    }
    
    Tensor::new(data, x.shape.clone())
}

/// Multi-Head Self-Attention
#[derive(Debug, Clone)]
pub struct MultiHeadSelfAttention {
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

impl MultiHeadSelfAttention {
    pub fn new(d_model: usize, n_heads: usize, dropout: f32) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        let d_k = d_model / n_heads;
        
        Self {
            n_heads,
            d_model,
            d_k,
            w_q: Linear::new(d_model, d_model, false),
            w_k: Linear::new(d_model, d_model, false),
            w_v: Linear::new(d_model, d_model, false),
            w_o: Linear::new(d_model, d_model, false),
            dropout: Dropout::new(dropout),
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let ndim = x.shape.ndim();
        let is_2d = ndim == 2;

        // Handle both 2D [batch, d_model] and 3D [batch, seq_len, d_model] inputs
        let (batch, seq_len) = if is_2d {
            // 2D input: [batch, d_model] - treat as single token (seq_len=1)
            (x.shape.dim(0), 1)
        } else if ndim == 3 {
            // 3D input: [batch, seq_len, d_model]
            (x.shape.dim(0), x.shape.dim(1))
        } else {
            panic!("MultiHeadSelfAttention expects 2D or 3D input, got {}D", ndim);
        };

        // Project Q, K, V
        let q = self.w_q.forward(x);
        let k = self.w_k.forward(x);
        let v = self.w_v.forward(x);

        // Reshape to [batch, heads, seq, d_k]
        let q = self.split_heads(&q, batch, seq_len);
        let k = self.split_heads(&k, batch, seq_len);
        let v = self.split_heads(&v, batch, seq_len);

        // Scaled dot-product attention
        let attn_output = scaled_dot_product_attention(&q, &k, &v, mask);

        // Concatenate heads
        let concat = self.concat_heads(&attn_output, batch, seq_len);

        // Final projection
        let mut output = self.w_o.forward(&concat);

        // If input was 2D, squeeze the seq_len dimension back to 2D
        if is_2d && output.shape.ndim() == 3 {
            // [batch, 1, d_model] -> [batch, d_model]
            let new_shape = vec![output.shape.dim(0), output.shape.dim(2)];
            output = Tensor::new(output.data, new_shape);
        }

        self.dropout.forward(&output)
    }

    /// Split into multiple heads: [batch, seq, d_model] -> [batch, heads, seq, d_k]
    fn split_heads(&self, x: &Tensor, batch: usize, seq_len: usize) -> Tensor {
        let mut data = vec![0.0; batch * self.n_heads * seq_len * self.d_k];
        
        for b in 0..batch {
            for s in 0..seq_len {
                for h in 0..self.n_heads {
                    for k in 0..self.d_k {
                        let src_idx = b * seq_len * self.d_model + s * self.d_model + h * self.d_k + k;
                        let dst_idx = b * self.n_heads * seq_len * self.d_k 
                            + h * seq_len * self.d_k + s * self.d_k + k;
                        data[dst_idx] = x.data[src_idx];
                    }
                }
            }
        }
        
        Tensor::new(data, vec![batch, self.n_heads, seq_len, self.d_k])
    }

    /// Concatenate heads: [batch, heads, seq, d_k] -> [batch, seq, d_model]
    fn concat_heads(&self, x: &Tensor, batch: usize, seq_len: usize) -> Tensor {
        let mut data = vec![0.0; batch * seq_len * self.d_model];
        
        for b in 0..batch {
            for s in 0..seq_len {
                for h in 0..self.n_heads {
                    for k in 0..self.d_k {
                        let src_idx = b * self.n_heads * seq_len * self.d_k 
                            + h * seq_len * self.d_k + s * self.d_k + k;
                        let dst_idx = b * seq_len * self.d_model + s * self.d_model + h * self.d_k + k;
                        data[dst_idx] = x.data[src_idx];
                    }
                }
            }
        }
        
        Tensor::new(data, vec![batch, seq_len, self.d_model])
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.w_q.parameters());
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }
}

/// Feed-Forward Network (FFN)
#[derive(Debug, Clone)]
pub struct FeedForwardNetwork {
    /// First linear layer
    pub linear1: Linear,
    /// Second linear layer
    pub linear2: Linear,
    /// Dropout
    pub dropout: Dropout,
}

impl FeedForwardNetwork {
    pub fn new(d_model: usize, d_ff: usize, dropout: f32) -> Self {
        Self {
            linear1: Linear::new(d_model, d_ff, true),
            linear2: Linear::new(d_ff, d_model, true),
            dropout: Dropout::new(dropout),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.linear1.forward(x).gelu();
        let h = self.dropout.forward(&h);
        self.linear2.forward(&h)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params
    }
}

/// Single Attention Block (Pre-LN Transformer)
#[derive(Debug, Clone)]
pub struct AttentionBlock {
    /// Self-attention
    pub attention: MultiHeadSelfAttention,
    /// Feed-forward network
    pub ffn: FeedForwardNetwork,
    /// Layer norm 1
    pub ln1: LayerNorm,
    /// Layer norm 2
    pub ln2: LayerNorm,
    /// Dropout
    pub dropout: Dropout,
}

impl AttentionBlock {
    pub fn new(config: &TransformerConfig) -> Self {
        Self {
            attention: MultiHeadSelfAttention::new(config.d_model, config.n_heads, config.dropout),
            ffn: FeedForwardNetwork::new(config.d_model, config.d_ff, config.dropout),
            ln1: LayerNorm::new(vec![config.d_model], config.layer_norm_eps),
            ln2: LayerNorm::new(vec![config.d_model], config.layer_norm_eps),
            dropout: Dropout::new(config.dropout),
        }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Pre-LN: LN -> Attention -> Residual
        let normed = self.ln1.forward(x);
        let attn_out = self.attention.forward(&normed, mask);
        let x = x.add(&self.dropout.forward(&attn_out));
        
        // Pre-LN: LN -> FFN -> Residual
        let normed = self.ln2.forward(&x);
        let ffn_out = self.ffn.forward(&normed);
        x.add(&self.dropout.forward(&ffn_out))
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters());
        params.extend(self.ffn.parameters());
        params.extend(self.ln1.parameters());
        params.extend(self.ln2.parameters());
        params
    }
}

/// Full Transformer Encoder
#[derive(Debug, Clone)]
pub struct TransformerEncoder {
    /// Configuration
    pub config: TransformerConfig,
    /// Token embedding
    pub token_embedding: super::layers::Embedding,
    /// Positional encoding
    pub pos_encoding: PositionalEncoding,
    /// Encoder layers
    pub layers: Vec<AttentionBlock>,
    /// Final layer norm
    pub final_ln: LayerNorm,
}

impl TransformerEncoder {
    pub fn new(config: TransformerConfig) -> Self {
        let token_embedding = super::layers::Embedding::new(config.vocab_size, config.d_model);
        let pos_encoding = PositionalEncoding::new(config.d_model, config.max_seq_len, config.dropout);
        
        let layers: Vec<AttentionBlock> = (0..config.n_layers)
            .map(|_| AttentionBlock::new(&config))
            .collect();
        
        let final_ln = LayerNorm::new(vec![config.d_model], config.layer_norm_eps);
        
        Self {
            config,
            token_embedding,
            pos_encoding,
            layers,
            final_ln,
        }
    }

    /// Forward pass with token indices
    pub fn forward(&self, token_ids: &[Vec<usize>], mask: Option<&Tensor>) -> Tensor {
        // Embed tokens
        let mut x = self.token_embedding.forward_batch(token_ids);
        
        // Add positional encoding
        x = self.pos_encoding.forward(&x);
        
        // Pass through encoder layers
        for layer in &self.layers {
            x = layer.forward(&x, mask);
        }
        
        // Final layer norm
        self.final_ln.forward(&x)
    }

    /// Forward pass with pre-embedded input
    pub fn forward_embedded(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let mut h = self.pos_encoding.forward(x);
        
        for layer in &self.layers {
            h = layer.forward(&h, mask);
        }
        
        self.final_ln.forward(&h)
    }

    /// Get the output embedding dimension
    pub fn output_dim(&self) -> usize {
        self.config.d_model
    }

    /// Get all parameters
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.token_embedding.parameters());
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.final_ln.parameters());
        params
    }

    /// Count total parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.shape.numel()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positional_encoding() {
        let pe = PositionalEncoding::new(64, 100, 0.0);
        let x = Tensor::randn(vec![2, 10, 64]);
        let y = pe.forward(&x);
        
        assert_eq!(y.shape.0, vec![2, 10, 64]);
    }

    #[test]
    fn test_multi_head_attention() {
        let mha = MultiHeadSelfAttention::new(64, 4, 0.0);
        let x = Tensor::randn(vec![2, 10, 64]);
        let y = mha.forward(&x, None);
        
        assert_eq!(y.shape.0, vec![2, 10, 64]);
    }

    #[test]
    fn test_attention_block() {
        let config = TransformerConfig {
            d_model: 64,
            n_heads: 4,
            d_ff: 256,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            ..Default::default()
        };
        
        let block = AttentionBlock::new(&config);
        let x = Tensor::randn(vec![2, 10, 64]);
        let y = block.forward(&x, None);
        
        assert_eq!(y.shape.0, vec![2, 10, 64]);
    }

    #[test]
    fn test_transformer_encoder() {
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
        
        let encoder = TransformerEncoder::new(config);
        let tokens = vec![vec![1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10]];
        let output = encoder.forward(&tokens, None);
        
        assert_eq!(output.shape.0, vec![2, 5, 64]);
        
        println!("Transformer parameters: {}", encoder.num_parameters());
    }
}
