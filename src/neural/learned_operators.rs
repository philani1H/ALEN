//! Learned Neural Operators
//!
//! Neural network-based reasoning operators that learn from data:
//! - GatedOperator: Uses gating mechanism for selective reasoning
//! - AttentionOperator: Uses self-attention for context-aware reasoning
//! - ResidualOperator: Deep residual networks for complex transformations

use super::tensor::Tensor;
use super::layers::{Linear, LayerNorm, Dropout};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for neural operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Input/output dimension
    pub d_model: usize,
    /// Hidden dimension
    pub d_hidden: usize,
    /// Number of layers
    pub n_layers: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Whether to use residual connections
    pub use_residual: bool,
    /// Whether to use layer normalization
    pub use_layer_norm: bool,
}

impl Default for OperatorConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            d_hidden: 1024,
            n_layers: 2,
            dropout: 0.1,
            use_residual: true,
            use_layer_norm: true,
        }
    }
}

/// Base trait for neural operators
pub trait NeuralOperatorTrait {
    /// Forward pass
    fn forward(&self, x: &Tensor) -> Tensor;
    
    /// Get all trainable parameters
    fn parameters(&self) -> Vec<&Tensor>;
    
    /// Get operator name
    fn name(&self) -> &str;
    
    /// Get operator type
    fn operator_type(&self) -> &str;
}

/// Gated Neural Operator
/// Uses a gating mechanism to selectively transform input
/// output = gate * transform(x) + (1 - gate) * x
#[derive(Debug, Clone)]
pub struct GatedOperator {
    /// Operator name
    pub name: String,
    /// Configuration
    pub config: OperatorConfig,
    /// Transform network
    pub transform_layers: Vec<Linear>,
    /// Gate network
    pub gate_layers: Vec<Linear>,
    /// Layer norms
    pub layer_norms: Vec<LayerNorm>,
    /// Dropout
    pub dropout: Dropout,
    /// Success count
    pub success_count: u64,
    /// Usage count
    pub usage_count: u64,
    /// Learned weight (for selection)
    pub weight: f32,
}

impl GatedOperator {
    pub fn new(name: &str, config: OperatorConfig) -> Self {
        let mut transform_layers = Vec::new();
        let mut gate_layers = Vec::new();
        let mut layer_norms = Vec::new();
        
        // Build transform network
        let mut in_dim = config.d_model;
        for i in 0..config.n_layers {
            let out_dim = if i == config.n_layers - 1 {
                config.d_model
            } else {
                config.d_hidden
            };
            transform_layers.push(Linear::new(in_dim, out_dim, true));
            gate_layers.push(Linear::new(in_dim, out_dim, true));
            
            if config.use_layer_norm {
                layer_norms.push(LayerNorm::new(vec![out_dim], 1e-5));
            }
            
            in_dim = out_dim;
        }
        
        let dropout_p = config.dropout;
        
        Self {
            name: name.to_string(),
            config,
            transform_layers,
            gate_layers,
            layer_norms,
            dropout: Dropout::new(dropout_p),
            success_count: 0,
            usage_count: 0,
            weight: 1.0,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut transform = x.clone();
        let mut gate = x.clone();
        
        for i in 0..self.transform_layers.len() {
            // Transform path
            transform = self.transform_layers[i].forward(&transform);
            if i < self.transform_layers.len() - 1 {
                transform = transform.gelu();
            }
            
            // Gate path
            gate = self.gate_layers[i].forward(&gate);
            gate = gate.sigmoid();
            
            // Apply layer norm if configured
            if self.config.use_layer_norm && i < self.layer_norms.len() {
                transform = self.layer_norms[i].forward(&transform);
            }
        }
        
        // Apply gating: output = gate * transform + (1 - gate) * x
        let gated = transform.mul(&gate);
        let one_minus_gate = gate.scale(-1.0).add(&Tensor::ones(gate.shape.clone()));
        let residual = x.mul(&one_minus_gate);
        
        let output = gated.add(&residual);
        self.dropout.forward(&output)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.transform_layers {
            params.extend(layer.parameters());
        }
        for layer in &self.gate_layers {
            params.extend(layer.parameters());
        }
        for ln in &self.layer_norms {
            params.extend(ln.parameters());
        }
        params
    }

    pub fn update_stats(&mut self, success: bool) {
        self.usage_count += 1;
        if success {
            self.success_count += 1;
            self.weight = (self.weight + 0.01).min(3.0);
        } else {
            self.weight = (self.weight - 0.005).max(0.1);
        }
    }

    pub fn success_rate(&self) -> f32 {
        if self.usage_count == 0 {
            0.5
        } else {
            self.success_count as f32 / self.usage_count as f32
        }
    }
}

/// Attention-based Neural Operator
/// Uses self-attention to reason about relationships in the input
#[derive(Debug, Clone)]
pub struct AttentionOperator {
    /// Operator name
    pub name: String,
    /// Configuration
    pub config: OperatorConfig,
    /// Query projection
    pub w_q: Linear,
    /// Key projection
    pub w_k: Linear,
    /// Value projection
    pub w_v: Linear,
    /// Output projection
    pub w_o: Linear,
    /// Feed-forward layers
    pub ff_layers: Vec<Linear>,
    /// Layer norms
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    /// Dropout
    pub dropout: Dropout,
    /// Number of attention heads
    pub n_heads: usize,
    /// Stats
    pub success_count: u64,
    pub usage_count: u64,
    pub weight: f32,
}

impl AttentionOperator {
    pub fn new(name: &str, config: OperatorConfig, n_heads: usize) -> Self {
        let d_model = config.d_model;
        
        Self {
            name: name.to_string(),
            config: config.clone(),
            w_q: Linear::new(d_model, d_model, false),
            w_k: Linear::new(d_model, d_model, false),
            w_v: Linear::new(d_model, d_model, false),
            w_o: Linear::new(d_model, d_model, false),
            ff_layers: vec![
                Linear::new(d_model, config.d_hidden, true),
                Linear::new(config.d_hidden, d_model, true),
            ],
            ln1: LayerNorm::new(vec![d_model], 1e-5),
            ln2: LayerNorm::new(vec![d_model], 1e-5),
            dropout: Dropout::new(config.dropout),
            n_heads,
            success_count: 0,
            usage_count: 0,
            weight: 1.0,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Ensure 3D input [batch, seq, d_model]
        let (x_3d, was_2d) = if x.shape.ndim() == 2 {
            (x.reshape(vec![1, x.shape.dim(0), x.shape.dim(1)]), true)
        } else {
            (x.clone(), false)
        };
        
        // Self-attention
        let normed = self.ln1.forward(&x_3d);
        let q = self.w_q.forward(&normed);
        let k = self.w_k.forward(&normed);
        let v = self.w_v.forward(&normed);
        
        // Simple attention (without multi-head for simplicity)
        let d_k = (self.config.d_model as f32).sqrt();
        let scores = self.compute_attention_scores(&q, &k, d_k);
        let attn_out = self.apply_attention(&scores, &v);
        let attn_out = self.w_o.forward(&attn_out);
        
        // Residual connection
        let x_3d = x_3d.add(&self.dropout.forward(&attn_out));
        
        // Feed-forward
        let normed = self.ln2.forward(&x_3d);
        let mut ff_out = self.ff_layers[0].forward(&normed).gelu();
        ff_out = self.ff_layers[1].forward(&ff_out);
        let output = x_3d.add(&self.dropout.forward(&ff_out));
        
        // Restore original shape if needed
        if was_2d {
            output.reshape(vec![output.shape.dim(1), output.shape.dim(2)])
        } else {
            output
        }
    }

    fn compute_attention_scores(&self, q: &Tensor, k: &Tensor, scale: f32) -> Tensor {
        let batch = q.shape.dim(0);
        let seq_len = q.shape.dim(1);
        let d_model = q.shape.dim(2);
        
        let mut scores = vec![0.0; batch * seq_len * seq_len];
        
        for b in 0..batch {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut dot = 0.0;
                    for d in 0..d_model {
                        let q_idx = b * seq_len * d_model + i * d_model + d;
                        let k_idx = b * seq_len * d_model + j * d_model + d;
                        dot += q.data[q_idx] * k.data[k_idx];
                    }
                    scores[b * seq_len * seq_len + i * seq_len + j] = dot / scale;
                }
            }
        }
        
        // Softmax
        let scores_tensor = Tensor::new(scores, vec![batch, seq_len, seq_len]);
        scores_tensor.softmax()
    }

    fn apply_attention(&self, scores: &Tensor, v: &Tensor) -> Tensor {
        let batch = v.shape.dim(0);
        let seq_len = v.shape.dim(1);
        let d_model = v.shape.dim(2);
        
        let mut output = vec![0.0; batch * seq_len * d_model];
        
        for b in 0..batch {
            for i in 0..seq_len {
                for d in 0..d_model {
                    let mut sum = 0.0;
                    for j in 0..seq_len {
                        let score_idx = b * seq_len * seq_len + i * seq_len + j;
                        let v_idx = b * seq_len * d_model + j * d_model + d;
                        sum += scores.data[score_idx] * v.data[v_idx];
                    }
                    output[b * seq_len * d_model + i * d_model + d] = sum;
                }
            }
        }
        
        Tensor::new(output, vec![batch, seq_len, d_model])
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.w_q.parameters());
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        for layer in &self.ff_layers {
            params.extend(layer.parameters());
        }
        params.extend(self.ln1.parameters());
        params.extend(self.ln2.parameters());
        params
    }

    pub fn update_stats(&mut self, success: bool) {
        self.usage_count += 1;
        if success {
            self.success_count += 1;
            self.weight = (self.weight + 0.01).min(3.0);
        } else {
            self.weight = (self.weight - 0.005).max(0.1);
        }
    }
}

/// Residual Neural Operator
/// Deep residual network for complex transformations
#[derive(Debug, Clone)]
pub struct ResidualOperator {
    /// Operator name
    pub name: String,
    /// Configuration
    pub config: OperatorConfig,
    /// Residual blocks
    pub blocks: Vec<ResidualBlock>,
    /// Input projection (if dimensions differ)
    pub input_proj: Option<Linear>,
    /// Output projection
    pub output_proj: Option<Linear>,
    /// Stats
    pub success_count: u64,
    pub usage_count: u64,
    pub weight: f32,
}

/// A single residual block
#[derive(Debug, Clone)]
pub struct ResidualBlock {
    pub linear1: Linear,
    pub linear2: Linear,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub dropout: Dropout,
}

impl ResidualBlock {
    pub fn new(dim: usize, hidden_dim: usize, dropout: f32) -> Self {
        Self {
            linear1: Linear::new(dim, hidden_dim, true),
            linear2: Linear::new(hidden_dim, dim, true),
            ln1: LayerNorm::new(vec![dim], 1e-5),
            ln2: LayerNorm::new(vec![dim], 1e-5),
            dropout: Dropout::new(dropout),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let normed = self.ln1.forward(x);
        let h = self.linear1.forward(&normed).gelu();
        let h = self.dropout.forward(&h);
        let h = self.linear2.forward(&h);
        let h = self.ln2.forward(&h);
        x.add(&self.dropout.forward(&h))
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.ln1.parameters());
        params.extend(self.ln2.parameters());
        params
    }
}

impl ResidualOperator {
    pub fn new(name: &str, config: OperatorConfig) -> Self {
        let blocks: Vec<ResidualBlock> = (0..config.n_layers)
            .map(|_| ResidualBlock::new(config.d_model, config.d_hidden, config.dropout))
            .collect();
        
        Self {
            name: name.to_string(),
            config,
            blocks,
            input_proj: None,
            output_proj: None,
            success_count: 0,
            usage_count: 0,
            weight: 1.0,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = if let Some(ref proj) = self.input_proj {
            proj.forward(x)
        } else {
            x.clone()
        };
        
        for block in &self.blocks {
            h = block.forward(&h);
        }
        
        if let Some(ref proj) = self.output_proj {
            proj.forward(&h)
        } else {
            h
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if let Some(ref proj) = self.input_proj {
            params.extend(proj.parameters());
        }
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        if let Some(ref proj) = self.output_proj {
            params.extend(proj.parameters());
        }
        params
    }

    pub fn update_stats(&mut self, success: bool) {
        self.usage_count += 1;
        if success {
            self.success_count += 1;
            self.weight = (self.weight + 0.01).min(3.0);
        } else {
            self.weight = (self.weight - 0.005).max(0.1);
        }
    }
}

/// Enum wrapper for different operator types
#[derive(Debug, Clone)]
pub enum NeuralOperator {
    Gated(GatedOperator),
    Attention(AttentionOperator),
    Residual(ResidualOperator),
}

impl NeuralOperator {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            NeuralOperator::Gated(op) => op.forward(x),
            NeuralOperator::Attention(op) => op.forward(x),
            NeuralOperator::Residual(op) => op.forward(x),
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        match self {
            NeuralOperator::Gated(op) => op.parameters(),
            NeuralOperator::Attention(op) => op.parameters(),
            NeuralOperator::Residual(op) => op.parameters(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            NeuralOperator::Gated(op) => &op.name,
            NeuralOperator::Attention(op) => &op.name,
            NeuralOperator::Residual(op) => &op.name,
        }
    }

    pub fn weight(&self) -> f32 {
        match self {
            NeuralOperator::Gated(op) => op.weight,
            NeuralOperator::Attention(op) => op.weight,
            NeuralOperator::Residual(op) => op.weight,
        }
    }

    pub fn update_stats(&mut self, success: bool) {
        match self {
            NeuralOperator::Gated(op) => op.update_stats(success),
            NeuralOperator::Attention(op) => op.update_stats(success),
            NeuralOperator::Residual(op) => op.update_stats(success),
        }
    }
}

/// Bank of neural operators for parallel reasoning
#[derive(Debug, Clone)]
pub struct NeuralOperatorBank {
    /// All operators
    pub operators: Vec<NeuralOperator>,
    /// Operator names to indices
    pub name_to_idx: HashMap<String, usize>,
    /// Configuration
    pub config: OperatorConfig,
}

impl NeuralOperatorBank {
    /// Create a new operator bank with default operators
    pub fn new(config: OperatorConfig) -> Self {
        let mut bank = Self {
            operators: Vec::new(),
            name_to_idx: HashMap::new(),
            config: config.clone(),
        };
        
        // Create diverse operators
        bank.add_operator(NeuralOperator::Gated(
            GatedOperator::new("logical", config.clone())
        ));
        bank.add_operator(NeuralOperator::Gated(
            GatedOperator::new("probabilistic", config.clone())
        ));
        bank.add_operator(NeuralOperator::Attention(
            AttentionOperator::new("analytical", config.clone(), 4)
        ));
        bank.add_operator(NeuralOperator::Attention(
            AttentionOperator::new("contextual", config.clone(), 8)
        ));
        bank.add_operator(NeuralOperator::Residual(
            ResidualOperator::new("deep_reasoning", config.clone())
        ));
        bank.add_operator(NeuralOperator::Residual(
            ResidualOperator::new("exploratory", config.clone())
        ));
        
        bank
    }

    /// Add an operator to the bank
    pub fn add_operator(&mut self, op: NeuralOperator) {
        let name = op.name().to_string();
        let idx = self.operators.len();
        self.operators.push(op);
        self.name_to_idx.insert(name, idx);
    }

    /// Get operator by name
    pub fn get(&self, name: &str) -> Option<&NeuralOperator> {
        self.name_to_idx.get(name).map(|&idx| &self.operators[idx])
    }

    /// Get mutable operator by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut NeuralOperator> {
        if let Some(&idx) = self.name_to_idx.get(name) {
            Some(&mut self.operators[idx])
        } else {
            None
        }
    }

    /// Generate candidates from all operators
    pub fn generate_candidates(&self, x: &Tensor) -> Vec<(String, Tensor)> {
        self.operators.iter()
            .map(|op| (op.name().to_string(), op.forward(x)))
            .collect()
    }

    /// Generate weighted candidates (more weight = more likely to be selected)
    pub fn generate_weighted_candidates(&self, x: &Tensor, count: usize) -> Vec<(String, Tensor)> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let total_weight: f32 = self.operators.iter().map(|op| op.weight()).sum();
        let mut candidates = Vec::new();
        
        for _ in 0..count {
            let mut threshold = rng.gen::<f32>() * total_weight;
            
            for op in &self.operators {
                threshold -= op.weight();
                if threshold <= 0.0 {
                    candidates.push((op.name().to_string(), op.forward(x)));
                    break;
                }
            }
        }
        
        candidates
    }

    /// Get all parameters from all operators
    pub fn parameters(&self) -> Vec<&Tensor> {
        self.operators.iter()
            .flat_map(|op| op.parameters())
            .collect()
    }

    /// Count total parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.shape.numel()).sum()
    }

    /// Update operator stats after selection
    pub fn update_operator(&mut self, name: &str, success: bool) {
        if let Some(op) = self.get_mut(name) {
            op.update_stats(success);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gated_operator() {
        let config = OperatorConfig {
            d_model: 64,
            d_hidden: 128,
            n_layers: 2,
            dropout: 0.0,
            use_residual: true,
            use_layer_norm: true,
        };
        
        let op = GatedOperator::new("test", config);
        let x = Tensor::randn(vec![2, 64]);
        let y = op.forward(&x);
        
        assert_eq!(y.shape.0, vec![2, 64]);
    }

    #[test]
    fn test_attention_operator() {
        let config = OperatorConfig {
            d_model: 64,
            d_hidden: 128,
            n_layers: 2,
            dropout: 0.0,
            use_residual: true,
            use_layer_norm: true,
        };
        
        let op = AttentionOperator::new("test", config, 4);
        let x = Tensor::randn(vec![2, 10, 64]);
        let y = op.forward(&x);
        
        assert_eq!(y.shape.0, vec![2, 10, 64]);
    }

    #[test]
    fn test_residual_operator() {
        let config = OperatorConfig {
            d_model: 64,
            d_hidden: 128,
            n_layers: 3,
            dropout: 0.0,
            use_residual: true,
            use_layer_norm: true,
        };
        
        let op = ResidualOperator::new("test", config);
        let x = Tensor::randn(vec![2, 64]);
        let y = op.forward(&x);
        
        assert_eq!(y.shape.0, vec![2, 64]);
    }

    #[test]
    fn test_operator_bank() {
        let config = OperatorConfig {
            d_model: 64,
            d_hidden: 128,
            n_layers: 2,
            dropout: 0.0,
            use_residual: true,
            use_layer_norm: true,
        };
        
        let bank = NeuralOperatorBank::new(config);
        let x = Tensor::randn(vec![2, 64]);
        let candidates = bank.generate_candidates(&x);
        
        assert_eq!(candidates.len(), 6);
        println!("Operator bank parameters: {}", bank.num_parameters());
    }
}
