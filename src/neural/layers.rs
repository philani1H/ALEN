//! Neural Network Layers
//!
//! Core building blocks for neural networks:
//! - Linear (fully connected)
//! - LayerNorm
//! - Dropout
//! - Embedding
//! - Conv1D

use super::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Linear (fully connected) layer: y = xW^T + b
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix [out_features, in_features]
    pub weight: Tensor,
    /// Bias vector [out_features]
    pub bias: Option<Tensor>,
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
}

impl Linear {
    /// Create a new linear layer with Xavier initialization
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = Tensor::xavier(vec![out_features, in_features]).with_grad();
        let bias = if bias {
            Some(Tensor::zeros(vec![out_features]).with_grad())
        } else {
            None
        };
        
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: [batch, in_features] or [batch, seq, in_features]
        let ndim = x.shape.ndim();
        
        if ndim == 2 {
            // [batch, in] @ [in, out] -> [batch, out]
            let mut output = x.matmul(&self.weight.transpose());
            
            if let Some(ref bias) = self.bias {
                // Broadcast add bias
                let batch = x.shape.dim(0);
                let mut data = output.to_vec();
                for i in 0..batch {
                    for j in 0..self.out_features {
                        data[i * self.out_features + j] += bias.data[j];
                    }
                }
                output = Tensor::new(data, output.shape.clone());
            }
            
            output
        } else if ndim == 3 {
            // [batch, seq, in] -> [batch, seq, out]
            let batch = x.shape.dim(0);
            let seq = x.shape.dim(1);
            
            // Reshape to [batch * seq, in]
            let x_flat = x.reshape(vec![batch * seq, self.in_features]);
            let out_flat = self.forward(&x_flat);
            
            // Reshape back to [batch, seq, out]
            out_flat.reshape(vec![batch, seq, self.out_features])
        } else {
            panic!("Linear layer expects 2D or 3D input");
        }
    }

    /// Get parameters for optimization
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    /// Get mutable parameters
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }
}

/// Layer Normalization
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Normalized shape (last dimensions)
    pub normalized_shape: Vec<usize>,
    /// Learnable scale parameter
    pub gamma: Tensor,
    /// Learnable shift parameter
    pub beta: Tensor,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl LayerNorm {
    /// Create a new layer norm
    pub fn new(normalized_shape: Vec<usize>, eps: f32) -> Self {
        let size: usize = normalized_shape.iter().product();
        Self {
            normalized_shape: normalized_shape.clone(),
            gamma: Tensor::ones(vec![size]).with_grad(),
            beta: Tensor::zeros(vec![size]).with_grad(),
            eps,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Normalize
        let normalized = x.layer_norm(self.eps);
        
        // Apply learnable parameters (element-wise)
        let last_dim = x.shape.dim(x.shape.ndim() - 1);
        let outer_size = x.shape.numel() / last_dim;
        
        let mut data = vec![0.0; x.shape.numel()];
        for i in 0..outer_size {
            for j in 0..last_dim {
                let idx = i * last_dim + j;
                data[idx] = normalized.data[idx] * self.gamma.data[j] + self.beta.data[j];
            }
        }
        
        Tensor::new(data, x.shape.clone())
    }

    /// Get parameters
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }
}

/// Dropout layer
#[derive(Debug, Clone)]
pub struct Dropout {
    /// Dropout probability
    pub p: f32,
    /// Training mode
    pub training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Self { p, training: true }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.dropout(self.p, self.training)
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}

/// Token Embedding layer
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Embedding matrix [vocab_size, embed_dim]
    pub weight: Tensor,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension
    pub embed_dim: usize,
}

impl Embedding {
    /// Create a new embedding layer
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let weight = Tensor::randn(vec![vocab_size, embed_dim])
            .scale(0.02)
            .with_grad();
        
        Self {
            weight,
            vocab_size,
            embed_dim,
        }
    }

    /// Forward pass - lookup embeddings for token indices
    pub fn forward(&self, indices: &[usize]) -> Tensor {
        let seq_len = indices.len();
        let mut data = vec![0.0; seq_len * self.embed_dim];
        
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.vocab_size, "Token index out of bounds");
            let start = idx * self.embed_dim;
            let end = start + self.embed_dim;
            data[i * self.embed_dim..(i + 1) * self.embed_dim]
                .copy_from_slice(&self.weight.data[start..end]);
        }
        
        Tensor::new(data, vec![seq_len, self.embed_dim])
    }

    /// Forward pass for batched indices [batch, seq_len]
    pub fn forward_batch(&self, indices: &[Vec<usize>]) -> Tensor {
        let batch_size = indices.len();
        let seq_len = indices[0].len();
        let mut data = vec![0.0; batch_size * seq_len * self.embed_dim];
        
        for (b, seq) in indices.iter().enumerate() {
            for (s, &idx) in seq.iter().enumerate() {
                assert!(idx < self.vocab_size, "Token index out of bounds");
                let src_start = idx * self.embed_dim;
                let dst_start = (b * seq_len + s) * self.embed_dim;
                data[dst_start..dst_start + self.embed_dim]
                    .copy_from_slice(&self.weight.data[src_start..src_start + self.embed_dim]);
            }
        }
        
        Tensor::new(data, vec![batch_size, seq_len, self.embed_dim])
    }

    /// Get parameters
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }
}

/// 1D Convolution layer
#[derive(Debug, Clone)]
pub struct Conv1D {
    /// Weight [out_channels, in_channels, kernel_size]
    pub weight: Tensor,
    /// Bias [out_channels]
    pub bias: Option<Tensor>,
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
}

impl Conv1D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let weight = Tensor::kaiming(vec![out_channels, in_channels, kernel_size]).with_grad();
        let bias = if bias {
            Some(Tensor::zeros(vec![out_channels]).with_grad())
        } else {
            None
        };
        
        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    /// Forward pass: [batch, in_channels, seq_len] -> [batch, out_channels, out_len]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.shape.ndim(), 3, "Conv1D expects 3D input");
        
        let batch = x.shape.dim(0);
        let in_ch = x.shape.dim(1);
        let seq_len = x.shape.dim(2);
        
        assert_eq!(in_ch, self.in_channels);
        
        let out_len = (seq_len + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let mut output = vec![0.0; batch * self.out_channels * out_len];
        
        // Naive convolution implementation
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for o in 0..out_len {
                    let mut sum = 0.0;
                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let pos = o * self.stride + k;
                            let pos = pos as i32 - self.padding as i32;
                            
                            if pos >= 0 && (pos as usize) < seq_len {
                                let x_idx = b * in_ch * seq_len + ic * seq_len + pos as usize;
                                let w_idx = oc * self.in_channels * self.kernel_size 
                                    + ic * self.kernel_size + k;
                                sum += x.data[x_idx] * self.weight.data[w_idx];
                            }
                        }
                    }
                    
                    if let Some(ref bias) = self.bias {
                        sum += bias.data[oc];
                    }
                    
                    output[b * self.out_channels * out_len + oc * out_len + o] = sum;
                }
            }
        }
        
        Tensor::new(output, vec![batch, self.out_channels, out_len])
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let linear = Linear::new(4, 3, true);
        let x = Tensor::randn(vec![2, 4]);
        let y = linear.forward(&x);
        
        assert_eq!(y.shape.0, vec![2, 3]);
    }

    #[test]
    fn test_linear_3d() {
        let linear = Linear::new(4, 3, true);
        let x = Tensor::randn(vec![2, 5, 4]); // [batch, seq, features]
        let y = linear.forward(&x);
        
        assert_eq!(y.shape.0, vec![2, 5, 3]);
    }

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::randn(vec![2, 4]);
        let y = ln.forward(&x);
        
        assert_eq!(y.shape.0, vec![2, 4]);
    }

    #[test]
    fn test_embedding() {
        let emb = Embedding::new(100, 32);
        let indices = vec![1, 5, 10, 20];
        let y = emb.forward(&indices);
        
        assert_eq!(y.shape.0, vec![4, 32]);
    }

    #[test]
    fn test_conv1d() {
        let conv = Conv1D::new(3, 8, 3, 1, 1, true);
        let x = Tensor::randn(vec![2, 3, 10]);
        let y = conv.forward(&x);
        
        assert_eq!(y.shape.0, vec![2, 8, 10]);
    }
}
