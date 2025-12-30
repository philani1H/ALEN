//! Universal Expert Neural Network (UENN)
//!
//! Multi-branch architecture for solve-verify-explain:
//!
//! Mathematical Foundation:
//! Input: x̃ = concat(x, a, m) where:
//!   x = problem input ∈ ℝ^{d_x}
//!   a = audience profile ∈ ℝ^{d_a}
//!   m = memory retrieval ∈ ℝ^{d_m}
//!
//! Branches:
//! 1. Solve: y_s = f_s(x̃; θ_s) → solution embedding
//! 2. Verify: y_v = f_v(x̃, y_s; θ_v) → correctness probability
//! 3. Explain: y_e = f_e(x̃, y_s; θ_e) → explanation embedding
//!
//! Loss:
//! ℒ_total = α·ℒ_solution + β·ℒ_verify + γ·ℒ_explain
//!
//! Training:
//! θ_{t+1} = θ_t - η ∇_θ ℒ_total

use super::tensor::Tensor;
use super::layers::{Linear, LayerNorm, Dropout};
use super::transformer::{TransformerEncoder, TransformerConfig};
use serde::{Deserialize, Serialize};

// ============================================================================
// PART 1: CONFIGURATION
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalNetworkConfig {
    /// Input dimension (problem)
    pub input_dim: usize,
    
    /// Audience profile dimension
    pub audience_dim: usize,
    
    /// Memory dimension
    pub memory_dim: usize,
    
    /// Solution embedding dimension
    pub solution_dim: usize,
    
    /// Explanation embedding dimension
    pub explanation_dim: usize,
    
    /// Hidden dimensions for each branch
    pub solve_hidden: Vec<usize>,
    pub verify_hidden: Vec<usize>,
    pub explain_hidden: Vec<usize>,
    
    /// Transformer config
    pub transformer_config: TransformerConfig,
    
    /// Dropout
    pub dropout: f32,
    
    /// Loss weights
    pub alpha: f32,  // Solution weight
    pub beta: f32,   // Verification weight
    pub gamma: f32,  // Explanation weight
}

impl Default for UniversalNetworkConfig {
    fn default() -> Self {
        Self {
            input_dim: 128,
            audience_dim: 32,
            memory_dim: 64,
            solution_dim: 256,
            explanation_dim: 256,
            solve_hidden: vec![512, 512, 256],
            verify_hidden: vec![256, 128],
            explain_hidden: vec![512, 512, 256],
            transformer_config: TransformerConfig {
                d_model: 256,
                n_heads: 8,
                d_ff: 1024,
                n_layers: 6,
                dropout: 0.1,
                max_seq_len: 512,
                vocab_size: 10000,
                layer_norm_eps: 1e-5,
            },
            dropout: 0.1,
            alpha: 0.5,
            beta: 0.3,
            gamma: 0.2,
        }
    }
}

// ============================================================================
// PART 2: SOLVE BRANCH
// ============================================================================

pub struct SolveBranch {
    /// Input projection
    input_proj: Linear,
    
    /// Transformer encoder
    transformer: TransformerEncoder,
    
    /// Hidden layers
    hidden_layers: Vec<Linear>,
    layer_norms: Vec<LayerNorm>,
    dropouts: Vec<Dropout>,
    
    /// Output projection
    output_proj: Linear,
}

impl SolveBranch {
    pub fn new(config: &UniversalNetworkConfig) -> Self {
        let augmented_dim = config.input_dim + config.audience_dim + config.memory_dim;
        
        // Input projection to transformer dimension
        let input_proj = Linear::new(augmented_dim, config.transformer_config.d_model, true);
        
        // Transformer encoder
        let transformer = TransformerEncoder::new(config.transformer_config.clone());
        
        // Hidden layers
        let mut hidden_layers = Vec::new();
        let mut layer_norms = Vec::new();
        let mut dropouts = Vec::new();
        
        let mut prev_dim = config.transformer_config.d_model;
        for &hidden_dim in &config.solve_hidden {
            hidden_layers.push(Linear::new(prev_dim, hidden_dim, true));
            layer_norms.push(LayerNorm::new(vec![hidden_dim], 1e-5));
            dropouts.push(Dropout::new(config.dropout));
            prev_dim = hidden_dim;
        }
        
        // Output projection
        let output_proj = Linear::new(prev_dim, config.solution_dim, true);
        
        Self {
            input_proj,
            transformer,
            hidden_layers,
            layer_norms,
            dropouts,
            output_proj,
        }
    }
    
    /// Forward pass: x̃ → y_s
    pub fn forward(&self, augmented_input: &Tensor, training: bool) -> Tensor {
        // Project to transformer dimension
        let mut h = self.input_proj.forward(augmented_input);
        
        // Transformer encoding
        h = self.transformer.forward_embedded(&h, None);
        
        // Hidden layers with residual connections
        for i in 0..self.hidden_layers.len() {
            let h_prev = h.clone();
            h = self.hidden_layers[i].forward(&h);
            h = self.layer_norms[i].forward(&h);
            h = h.gelu(); // GELU activation
            h = self.dropouts[i].forward(&h);
            
            // Residual connection if dimensions match
            if h.shape() == h_prev.shape() {
                h = h.add(&h_prev);
            }
        }
        
        // Output projection
        self.output_proj.forward(&h)
    }
}

// ============================================================================
// PART 3: VERIFICATION BRANCH
// ============================================================================

pub struct VerificationBranch {
    /// Input layers (augmented_input + solution_embedding)
    hidden_layers: Vec<Linear>,
    layer_norms: Vec<LayerNorm>,
    dropouts: Vec<Dropout>,
    
    /// Output layer (probability)
    output_layer: Linear,
}

impl VerificationBranch {
    pub fn new(config: &UniversalNetworkConfig) -> Self {
        let augmented_dim = config.input_dim + config.audience_dim + config.memory_dim;
        let input_dim = augmented_dim + config.solution_dim;
        
        let mut hidden_layers = Vec::new();
        let mut layer_norms = Vec::new();
        let mut dropouts = Vec::new();
        
        let mut prev_dim = input_dim;
        for &hidden_dim in &config.verify_hidden {
            hidden_layers.push(Linear::new(prev_dim, hidden_dim, true));
            layer_norms.push(LayerNorm::new(vec![hidden_dim], 1e-5));
            dropouts.push(Dropout::new(config.dropout));
            prev_dim = hidden_dim;
        }
        
        // Output: single probability
        let output_layer = Linear::new(prev_dim, 1, true);
        
        Self {
            hidden_layers,
            layer_norms,
            dropouts,
            output_layer,
        }
    }
    
    /// Forward pass: (x̃, y_s) → y_v ∈ [0,1]
    pub fn forward(&self, augmented_input: &Tensor, solution_embedding: &Tensor, training: bool) -> Tensor {
        // Concatenate inputs
        let mut h = augmented_input.concat(solution_embedding, 1);
        
        // Hidden layers
        for i in 0..self.hidden_layers.len() {
            h = self.hidden_layers[i].forward(&h);
            h = self.layer_norms[i].forward(&h);
            h = h.relu();
            h = self.dropouts[i].forward(&h);
        }
        
        // Output with sigmoid
        let logit = self.output_layer.forward(&h);
        logit.sigmoid()
    }
}

// ============================================================================
// PART 4: EXPLANATION BRANCH
// ============================================================================

pub struct ExplanationBranch {
    /// Input layers (augmented_input + solution_embedding)
    hidden_layers: Vec<Linear>,
    layer_norms: Vec<LayerNorm>,
    dropouts: Vec<Dropout>,
    
    /// Attention layer for audience adaptation
    audience_attention: Linear,
    
    /// Output projection
    output_proj: Linear,
}

impl ExplanationBranch {
    pub fn new(config: &UniversalNetworkConfig) -> Self {
        let augmented_dim = config.input_dim + config.audience_dim + config.memory_dim;
        let input_dim = augmented_dim + config.solution_dim;
        
        let mut hidden_layers = Vec::new();
        let mut layer_norms = Vec::new();
        let mut dropouts = Vec::new();
        
        let mut prev_dim = input_dim;
        for &hidden_dim in &config.explain_hidden {
            hidden_layers.push(Linear::new(prev_dim, hidden_dim, true));
            layer_norms.push(LayerNorm::new(vec![hidden_dim], 1e-5));
            dropouts.push(Dropout::new(config.dropout));
            prev_dim = hidden_dim;
        }
        
        // Audience attention
        let audience_attention = Linear::new(config.audience_dim, prev_dim, true);
        
        // Output projection
        let output_proj = Linear::new(prev_dim, config.explanation_dim, true);
        
        Self {
            hidden_layers,
            layer_norms,
            dropouts,
            audience_attention,
            output_proj,
        }
    }
    
    /// Forward pass: (x̃, y_s, a) → y_e
    pub fn forward(
        &self,
        augmented_input: &Tensor,
        solution_embedding: &Tensor,
        audience_profile: &Tensor,
        training: bool,
    ) -> Tensor {
        // Concatenate inputs
        let mut h = augmented_input.concat(solution_embedding, 1);
        
        // Hidden layers
        for i in 0..self.hidden_layers.len() {
            h = self.hidden_layers[i].forward(&h);
            h = self.layer_norms[i].forward(&h);
            h = h.gelu();
            h = self.dropouts[i].forward(&h);
        }
        
        // Audience attention
        let audience_weights = self.audience_attention.forward(audience_profile);
        let audience_weights = audience_weights.softmax();
        h = h.mul(&audience_weights);
        
        // Output projection
        self.output_proj.forward(&h)
    }
}

// ============================================================================
// PART 5: UNIVERSAL EXPERT NEURAL NETWORK
// ============================================================================

pub struct UniversalExpertNetwork {
    config: UniversalNetworkConfig,
    
    /// Three branches
    solve_branch: SolveBranch,
    verify_branch: VerificationBranch,
    explain_branch: ExplanationBranch,
}

impl UniversalExpertNetwork {
    pub fn new(config: UniversalNetworkConfig) -> Self {
        let solve_branch = SolveBranch::new(&config);
        let verify_branch = VerificationBranch::new(&config);
        let explain_branch = ExplanationBranch::new(&config);
        
        Self {
            config,
            solve_branch,
            verify_branch,
            explain_branch,
        }
    }
    
    /// Forward pass through all branches
    pub fn forward(
        &self,
        problem_input: &Tensor,
        audience_profile: &Tensor,
        memory_retrieval: &Tensor,
        training: bool,
    ) -> UniversalNetworkOutput {
        // Augment input: x̃ = concat(x, a, m)
        let augmented_input = problem_input
            .concat(audience_profile, 1)
            .concat(memory_retrieval, 1);
        
        // Solve branch: y_s = f_s(x̃)
        let solution_embedding = self.solve_branch.forward(&augmented_input, training);
        
        // Verification branch: y_v = f_v(x̃, y_s)
        let verification_prob = self.verify_branch.forward(
            &augmented_input,
            &solution_embedding,
            training,
        );
        
        // Explanation branch: y_e = f_e(x̃, y_s, a)
        let explanation_embedding = self.explain_branch.forward(
            &augmented_input,
            &solution_embedding,
            audience_profile,
            training,
        );
        
        UniversalNetworkOutput {
            solution_embedding,
            verification_prob,
            explanation_embedding,
        }
    }
    
    /// Compute total loss
    pub fn compute_loss(
        &self,
        output: &UniversalNetworkOutput,
        target_solution: &Tensor,
        target_verification: f32,
        target_explanation: &Tensor,
    ) -> UniversalNetworkLoss {
        // Solution loss (MSE)
        let solution_loss = output.solution_embedding
            .sub(target_solution)
            .pow(2.0)
            .mean();
        
        // Verification loss (binary cross-entropy)
        let v = output.verification_prob.mean();
        let verification_loss = if target_verification > 0.5 {
            -(target_verification * v.ln() + (1.0 - target_verification) * (1.0 - v).ln())
        } else {
            -(target_verification * v.ln() + (1.0 - target_verification) * (1.0 - v).ln())
        };
        
        // Explanation loss (MSE)
        let explanation_loss = output.explanation_embedding
            .sub(target_explanation)
            .pow(2.0)
            .mean();
        
        // Total loss
        let total_loss = 
            self.config.alpha * solution_loss +
            self.config.beta * verification_loss +
            self.config.gamma * explanation_loss;
        
        UniversalNetworkLoss {
            total: total_loss,
            solution: solution_loss,
            verification: verification_loss,
            explanation: explanation_loss,
        }
    }
}

// ============================================================================
// PART 6: OUTPUT STRUCTURES
// ============================================================================

#[derive(Debug, Clone)]
pub struct UniversalNetworkOutput {
    pub solution_embedding: Tensor,
    pub verification_prob: Tensor,
    pub explanation_embedding: Tensor,
}

#[derive(Debug, Clone)]
pub struct UniversalNetworkLoss {
    pub total: f32,
    pub solution: f32,
    pub verification: f32,
    pub explanation: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_universal_network_creation() {
        let config = UniversalNetworkConfig::default();
        let _network = UniversalExpertNetwork::new(config);
    }
    
    #[test]
    fn test_forward_pass() {
        let config = UniversalNetworkConfig::default();
        let network = UniversalExpertNetwork::new(config.clone());
        
        let batch_size = 2;
        let problem_input = Tensor::randn(vec![batch_size, config.input_dim]);
        let audience_profile = Tensor::randn(vec![batch_size, config.audience_dim]);
        let memory_retrieval = Tensor::randn(vec![batch_size, config.memory_dim]);
        
        let output = network.forward(
            &problem_input,
            &audience_profile,
            &memory_retrieval,
            false,
        );
        
        assert_eq!(output.solution_embedding.shape()[1], config.solution_dim);
        assert_eq!(output.verification_prob.shape()[1], 1);
        assert_eq!(output.explanation_embedding.shape()[1], config.explanation_dim);
    }
}
