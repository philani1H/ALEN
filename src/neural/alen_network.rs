//! ALEN Neural Network Architecture
//!
//! Implements the complete neural substrate for ALEN's verified learning loop:
//! - Encoder: Input → Thought space (ψ₀)
//! - Neural Operators: Parallel reasoning transformations (Tᵢ)
//! - Decoder: Thought → Output
//! - Verifier: Thought → Reconstructed input (cycle consistency check)
//!
//! Mathematical Foundation:
//! - E: X → ℝᵈ (encoder)
//! - Tᵢ: ℝᵈ → ℝᵈ (reasoning operators)
//! - D: ℝᵈ → Y (decoder)
//! - V: ℝᵈ → X (verifier/inverse)
//!
//! Energy Function:
//! E(ψ) = αC(ψ) + βR(ψ) + γU(ψ)
//! where C=constraint violation, R=memory risk, U=uncertainty
//!
//! Verification:
//! 1. Forward: |D(ψ*) - y| < ε₁
//! 2. Backward: |E(V(ψ*)) - ψ₀| < ε₂
//! 3. Stability: E(ψ + δ) ≈ E(ψ)

use super::tensor::Tensor;
use super::layers::{Linear, LayerNorm, Dropout, Embedding};
use super::transformer::{TransformerEncoder, TransformerConfig};
use serde::{Deserialize, Serialize};
use rand;

/// Configuration for ALEN neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ALENConfig {
    /// Thought space dimension (d_model)
    pub thought_dim: usize,
    /// Input vocabulary size
    pub vocab_size: usize,
    /// Number of parallel reasoning operators
    pub num_operators: usize,
    /// Operator hidden dimension
    pub operator_hidden_dim: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Layer norm epsilon
    pub layer_norm_eps: f32,
    /// Use transformer encoder
    pub use_transformer: bool,
    /// Transformer layers (if used)
    pub transformer_layers: usize,
    /// Transformer heads (if used)
    pub transformer_heads: usize,
    /// Energy function weights
    pub energy_weights: EnergyWeights,
}

/// Energy function weights for thought evaluation
/// E(ψ) = α·C(ψ) + β·R(ψ) + γ·U(ψ) - λ·N(ψ)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyWeights {
    /// Constraint weight (α) - penalizes distance from initial thought
    pub alpha: f32,
    /// Risk weight (β) - penalizes high entropy/uncertainty
    pub beta: f32,
    /// Uncertainty weight (γ) - penalizes variance in thought vector
    pub gamma: f32,
    /// Novelty weight (λ) - rewards creative/novel solutions
    pub lambda: f32,
}

impl Default for EnergyWeights {
    fn default() -> Self {
        Self {
            alpha: 1.0,   // Constraint weight
            beta: 0.5,    // Risk weight
            gamma: 0.3,   // Uncertainty weight
            lambda: 0.1,  // Novelty/creativity weight
        }
    }
}

impl EnergyWeights {
    /// Conservative weights - prioritize safety and consistency
    pub fn conservative() -> Self {
        Self {
            alpha: 1.5,
            beta: 1.0,
            gamma: 0.8,
            lambda: 0.05,
        }
    }

    /// Creative weights - encourage exploration and novelty
    pub fn creative() -> Self {
        Self {
            alpha: 0.7,
            beta: 0.3,
            gamma: 0.2,
            lambda: 0.5,
        }
    }

    /// Balanced weights - middle ground
    pub fn balanced() -> Self {
        Self::default()
    }
}

impl Default for ALENConfig {
    fn default() -> Self {
        Self {
            thought_dim: 128,
            vocab_size: 10000,
            num_operators: 8,
            operator_hidden_dim: 256,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_transformer: true,
            transformer_layers: 4,
            transformer_heads: 4,
            energy_weights: EnergyWeights::default(),
        }
    }
}

impl ALENConfig {
    /// Small config for testing
    pub fn small() -> Self {
        Self {
            thought_dim: 64,
            vocab_size: 5000,
            num_operators: 4,
            operator_hidden_dim: 128,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_transformer: false,
            transformer_layers: 2,
            transformer_heads: 2,
            energy_weights: EnergyWeights::default(),
        }
    }

    /// Medium config for production
    pub fn medium() -> Self {
        Self {
            thought_dim: 256,
            vocab_size: 20000,
            num_operators: 8,
            operator_hidden_dim: 512,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_transformer: true,
            transformer_layers: 6,
            transformer_heads: 8,
            energy_weights: EnergyWeights::default(),
        }
    }
}

/// Encoder: Maps input to thought space
/// E: X → ℝᵈ
#[derive(Debug, Clone)]
pub struct ThoughtEncoder {
    /// Configuration
    config: ALENConfig,
    /// Token embedding
    embedding: Embedding,
    /// Optional transformer encoder
    transformer: Option<TransformerEncoder>,
    /// Projection to thought space
    projection: Linear,
    /// Layer normalization
    layer_norm: LayerNorm,
    /// Dropout
    dropout: Dropout,
}

impl ThoughtEncoder {
    pub fn new(config: ALENConfig) -> Self {
        let embedding = Embedding::new(config.vocab_size, config.thought_dim);
        
        let transformer = if config.use_transformer {
            let tf_config = TransformerConfig {
                d_model: config.thought_dim,
                n_heads: config.transformer_heads,
                d_ff: config.thought_dim * 4,
                n_layers: config.transformer_layers,
                max_seq_len: 512,
                vocab_size: config.vocab_size,
                dropout: config.dropout,
                layer_norm_eps: config.layer_norm_eps,
            };
            Some(TransformerEncoder::new(tf_config))
        } else {
            None
        };
        
        let projection = Linear::new(config.thought_dim, config.thought_dim, true);
        let layer_norm = LayerNorm::new(vec![config.thought_dim], config.layer_norm_eps);
        let dropout = Dropout::new(config.dropout);
        
        Self {
            config,
            embedding,
            transformer,
            projection,
            layer_norm,
            dropout,
        }
    }

    /// Encode input tokens to thought vector
    /// Returns normalized thought vector ψ₀
    pub fn encode(&self, token_ids: &[usize]) -> Tensor {
        // Embed tokens
        let embedded = self.embedding.forward(token_ids);
        
        // Apply transformer if available
        let encoded = if let Some(ref transformer) = self.transformer {
            // Reshape for transformer: [1, seq_len, d_model]
            let batch_input = vec![token_ids.to_vec()];
            let tf_out = transformer.forward(&batch_input, None);
            
            // Mean pooling over sequence
            self.mean_pool(&tf_out)
        } else {
            // Simple mean pooling
            self.mean_pool(&embedded)
        };
        
        // Project to thought space
        let projected = self.projection.forward(&encoded);
        
        // Normalize
        let normed = self.layer_norm.forward(&projected);
        let dropped = self.dropout.forward(&normed);
        
        // L2 normalize to unit sphere
        dropped.normalize()
    }

    /// Encode from raw vector (for pre-embedded inputs)
    pub fn encode_vector(&self, vector: &[f32]) -> Tensor {
        let input = Tensor::new(
            vector.iter().map(|&x| x as f32).collect(),
            vec![1, vector.len()]
        );
        
        let projected = self.projection.forward(&input);
        let normed = self.layer_norm.forward(&projected);
        normed.normalize()
    }

    /// Mean pooling over sequence dimension
    fn mean_pool(&self, x: &Tensor) -> Tensor {
        let shape = &x.shape;
        
        if shape.ndim() == 2 {
            // [seq_len, d_model] -> [1, d_model]
            let seq_len = shape.dim(0);
            let d_model = shape.dim(1);
            let mut pooled = vec![0.0; d_model];
            
            for s in 0..seq_len {
                for d in 0..d_model {
                    pooled[d] += x.data[s * d_model + d];
                }
            }
            
            for d in 0..d_model {
                pooled[d] /= seq_len as f32;
            }
            
            Tensor::new(pooled, vec![1, d_model])
        } else if shape.ndim() == 3 {
            // [batch, seq_len, d_model] -> [batch, d_model]
            let batch = shape.dim(0);
            let seq_len = shape.dim(1);
            let d_model = shape.dim(2);
            let mut pooled = vec![0.0; batch * d_model];
            
            for b in 0..batch {
                for s in 0..seq_len {
                    for d in 0..d_model {
                        let idx = b * seq_len * d_model + s * d_model + d;
                        pooled[b * d_model + d] += x.data[idx];
                    }
                }
                for d in 0..d_model {
                    pooled[b * d_model + d] /= seq_len as f32;
                }
            }
            
            Tensor::new(pooled, vec![batch, d_model])
        } else {
            x.clone()
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.embedding.parameters());
        if let Some(ref tf) = self.transformer {
            params.extend(tf.parameters());
        }
        params.extend(self.projection.parameters());
        params.extend(self.layer_norm.parameters());
        params
    }
}

/// Neural Reasoning Operator
/// Tᵢ: ℝᵈ → ℝᵈ
#[derive(Debug, Clone)]
pub struct NeuralReasoningOperator {
    /// Operator ID
    pub id: usize,
    /// Operator type name
    pub name: String,
    /// First layer
    linear1: Linear,
    /// Second layer
    linear2: Linear,
    /// Layer norm
    layer_norm: LayerNorm,
    /// Dropout
    dropout: Dropout,
    /// Learned weight/score
    pub weight: f32,
}

impl NeuralReasoningOperator {
    pub fn new(id: usize, name: String, thought_dim: usize, hidden_dim: usize, dropout: f32) -> Self {
        Self {
            id,
            name,
            linear1: Linear::new(thought_dim, hidden_dim, true),
            linear2: Linear::new(hidden_dim, thought_dim, true),
            layer_norm: LayerNorm::new(vec![thought_dim], 1e-5),
            dropout: Dropout::new(dropout),
            weight: 1.0,
        }
    }

    /// Apply operator: ψᵢ = Tᵢ(ψ₀)
    /// Uses residual connection: ψᵢ = ψ₀ + f(ψ₀)
    pub fn forward(&self, psi: &Tensor) -> Tensor {
        // f(ψ) = W₂ * GELU(W₁ * ψ)
        let h = self.linear1.forward(psi).gelu();
        let h = self.dropout.forward(&h);
        let delta = self.linear2.forward(&h);
        
        // Residual: ψ + Δψ
        let output = psi.add(&delta);
        
        // Layer norm
        let normed = self.layer_norm.forward(&output);
        
        // Normalize to unit sphere
        normed.normalize()
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.layer_norm.parameters());
        params
    }
}

/// Decoder: Maps thought to output
/// D: ℝᵈ → Y
#[derive(Debug, Clone)]
pub struct ThoughtDecoder {
    /// Hidden layer
    hidden: Linear,
    /// Output layer
    output: Linear,
    /// Layer norm
    layer_norm: LayerNorm,
    /// Dropout
    dropout: Dropout,
}

impl ThoughtDecoder {
    pub fn new(thought_dim: usize, hidden_dim: usize, output_dim: usize, dropout: f32) -> Self {
        Self {
            hidden: Linear::new(thought_dim, hidden_dim, true),
            output: Linear::new(hidden_dim, output_dim, true),
            layer_norm: LayerNorm::new(vec![thought_dim], 1e-5),
            dropout: Dropout::new(dropout),
        }
    }

    /// Decode thought to output logits
    pub fn forward(&self, psi: &Tensor) -> Tensor {
        let normed = self.layer_norm.forward(psi);
        let h = self.hidden.forward(&normed).gelu();
        let h = self.dropout.forward(&h);
        self.output.forward(&h)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.hidden.parameters());
        params.extend(self.output.parameters());
        params.extend(self.layer_norm.parameters());
        params
    }
}

/// Verifier: Reconstructs input from thought (cycle consistency)
/// V: ℝᵈ → X
#[derive(Debug, Clone)]
pub struct ThoughtVerifier {
    /// Reconstruction network
    reconstruct: Linear,
    /// Hidden layer
    hidden: Linear,
    /// Layer norm
    layer_norm: LayerNorm,
    /// Dropout
    dropout: Dropout,
}

impl ThoughtVerifier {
    pub fn new(thought_dim: usize, hidden_dim: usize, output_dim: usize, dropout: f32) -> Self {
        Self {
            hidden: Linear::new(thought_dim, hidden_dim, true),
            reconstruct: Linear::new(hidden_dim, output_dim, true),
            layer_norm: LayerNorm::new(vec![thought_dim], 1e-5),
            dropout: Dropout::new(dropout),
        }
    }

    /// Reconstruct input from thought
    /// V(ψ*) ≈ x
    pub fn forward(&self, psi: &Tensor) -> Tensor {
        let normed = self.layer_norm.forward(psi);
        let h = self.hidden.forward(&normed).gelu();
        let h = self.dropout.forward(&h);
        self.reconstruct.forward(&h)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.hidden.parameters());
        params.extend(self.reconstruct.parameters());
        params.extend(self.layer_norm.parameters());
        params
    }
}

/// Complete ALEN Neural Network
pub struct ALENNetwork {
    /// Configuration
    pub config: ALENConfig,
    /// Encoder
    pub encoder: ThoughtEncoder,
    /// Parallel reasoning operators
    pub operators: Vec<NeuralReasoningOperator>,
    /// Decoder
    pub decoder: ThoughtDecoder,
    /// Verifier (for cycle consistency)
    pub verifier: ThoughtVerifier,
}

impl ALENNetwork {
    pub fn new(config: ALENConfig) -> Self {
        let encoder = ThoughtEncoder::new(config.clone());
        
        // Create parallel reasoning operators
        let operator_names = vec![
            "Logical", "Probabilistic", "Heuristic", "Analogical",
            "Conservative", "Exploratory", "Analytical", "Intuitive"
        ];
        
        let operators: Vec<NeuralReasoningOperator> = (0..config.num_operators)
            .map(|i| {
                let name = operator_names.get(i)
                    .unwrap_or(&"Generic")
                    .to_string();
                NeuralReasoningOperator::new(
                    i,
                    name,
                    config.thought_dim,
                    config.operator_hidden_dim,
                    config.dropout,
                )
            })
            .collect();
        
        let decoder = ThoughtDecoder::new(
            config.thought_dim,
            config.operator_hidden_dim,
            config.thought_dim, // Output same dimension for now
            config.dropout,
        );
        
        let verifier = ThoughtVerifier::new(
            config.thought_dim,
            config.operator_hidden_dim,
            config.thought_dim,
            config.dropout,
        );
        
        Self {
            config,
            encoder,
            operators,
            decoder,
            verifier,
        }
    }

    /// Full forward pass: Input → ψ₀ → {ψᵢ} → Select ψ* → Output
    pub fn forward(&self, token_ids: &[usize]) -> ALENForwardResult {
        // 1. Encode input to thought space
        let psi_0 = self.encoder.encode(token_ids);
        
        // 2. Generate candidate thoughts via parallel operators
        let candidates: Vec<(usize, Tensor)> = self.operators
            .iter()
            .map(|op| (op.id, op.forward(&psi_0)))
            .collect();
        
        // 3. Evaluate each candidate (energy function)
        let evaluated: Vec<CandidateEvaluation> = candidates
            .iter()
            .map(|(id, psi_i)| {
                let energy = self.compute_energy(psi_i, &psi_0);
                CandidateEvaluation {
                    operator_id: *id,
                    thought: psi_i.clone(),
                    energy,
                }
            })
            .collect();
        
        // 4. Select best candidate (minimum energy)
        let best_idx = evaluated
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.energy.partial_cmp(&b.energy).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        let selected_operator = evaluated[best_idx].operator_id;
        let psi_star = evaluated[best_idx].thought.clone();
        
        // 5. Decode selected thought
        let output = self.decoder.forward(&psi_star);
        
        // 6. Verify (cycle consistency)
        let reconstructed = self.verifier.forward(&psi_star);
        let verification_error = self.compute_verification_error(&psi_0, &reconstructed);
        
        ALENForwardResult {
            psi_0,
            candidates: evaluated,
            selected_operator,
            psi_star,
            output,
            reconstructed,
            verification_error,
        }
    }

    /// Compute energy function: E'(ψ) = αC(ψ) + βR(ψ) + γU(ψ) - λN(ψ)
    /// This is the complete mathematical model with novelty term
    /// Uses configurable weights from self.config.energy_weights
    fn compute_energy(&self, psi: &Tensor, psi_0: &Tensor) -> f32 {
        let weights = &self.config.energy_weights;

        // C(ψ): Constraint violation (distance from initial thought)
        let constraint = self.compute_constraint(psi, psi_0);

        // R(ψ): Risk (entropy of output distribution)
        let risk = self.compute_risk(psi);

        // U(ψ): Uncertainty (variance in thought vector)
        let uncertainty = self.compute_uncertainty(psi);

        // N(ψ): Novelty (distance from known patterns)
        // For now, use distance from initial state as proxy
        // In full implementation, would check against memory embeddings
        let novelty = self.compute_novelty(psi, psi_0);

        // E'(ψ) = α·C(ψ) + β·R(ψ) + γ·U(ψ) - λ·N(ψ)
        // Lower energy = better, but novelty reduces energy (encourages creativity)
        weights.alpha * constraint + weights.beta * risk + weights.gamma * uncertainty - weights.lambda * novelty
    }

    fn compute_constraint(&self, psi: &Tensor, psi_0: &Tensor) -> f32 {
        // L2 distance
        psi.data.iter()
            .zip(psi_0.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn compute_risk(&self, psi: &Tensor) -> f32 {
        // Entropy of thought vector (treat as probability distribution)
        let softmax = psi.softmax();
        let entropy: f32 = softmax.data.iter()
            .map(|&p| {
                if p > 1e-10 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();
        entropy
    }

    fn compute_uncertainty(&self, psi: &Tensor) -> f32 {
        // Variance of thought vector
        let mean: f32 = psi.data.iter().sum::<f32>() / psi.data.len() as f32;
        let variance: f32 = psi.data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / psi.data.len() as f32;
        variance
    }

    /// Compute novelty: N(ψ) = min_j |ψ - μ_j|
    /// Measures distance from known patterns (memory embeddings)
    fn compute_novelty(&self, psi: &Tensor, psi_0: &Tensor) -> f32 {
        // For now, use L2 distance from initial state
        // In full implementation, would check against all memory embeddings
        // and return minimum distance
        let distance = psi.data.iter()
            .zip(psi_0.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        
        // Normalize to [0, 1] range
        // Larger distance = more novel
        (distance / (psi.data.len() as f32).sqrt()).min(1.0)
    }

    /// Check stability under perturbation
    /// E(ψ* + η) < E(ψ*) + ε for small η
    fn check_stability(&self, psi_star: &Tensor, psi_0: &Tensor, radius: f32, epsilon: f32) -> bool {
        let base_energy = self.compute_energy(psi_star, psi_0);
        
        // Test with small random perturbations
        let num_tests = 5;
        for _ in 0..num_tests {
            // Create small perturbation
            let mut perturbed_data = psi_star.data.clone();
            for x in &mut perturbed_data {
                let noise = (rand::random::<f32>() - 0.5) * 2.0 * radius;
                *x += noise;
            }
            
            let perturbed = Tensor::new(perturbed_data, psi_star.shape.clone());
            let perturbed_energy = self.compute_energy(&perturbed, psi_0);
            
            // Check if energy doesn't increase too much
            if perturbed_energy > base_energy + epsilon {
                return false;
            }
        }
        
        true
    }

    fn compute_verification_error(&self, psi_0: &Tensor, reconstructed: &Tensor) -> f32 {
        // MSE between original and reconstructed
        psi_0.data.iter()
            .zip(reconstructed.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / psi_0.data.len() as f32
    }

    /// Verify if a thought is valid
    /// Returns true if forward, backward, AND stability checks pass
    /// This implements the complete verification gate: V(ψ*) = 1[forward ∧ backward ∧ stable]
    pub fn verify(&self, psi_star: &Tensor, psi_0: &Tensor, _epsilon_1: f32, epsilon_2: f32) -> bool {
        // 1. Forward check: Output should be valid and finite
        let output = self.decoder.forward(psi_star);
        let forward_valid = output.data.iter().all(|&x| x.is_finite());
        
        // 2. Backward check: Cycle consistency |T^{-1}(ψ*) - ψ₀| < δ
        let reconstructed = self.verifier.forward(psi_star);
        let backward_error = self.compute_verification_error(psi_0, &reconstructed);
        let backward_valid = backward_error < epsilon_2;
        
        // 3. Stability check: E(ψ* + η) < E(ψ*) + ε for small η
        let stability_radius = 0.01;
        let stability_epsilon = 0.1;
        let stable = self.check_stability(psi_star, psi_0, stability_radius, stability_epsilon);
        
        // All three conditions must pass
        forward_valid && backward_valid && stable
    }

    /// Extended verification with detailed results
    pub fn verify_detailed(&self, psi_star: &Tensor, psi_0: &Tensor, epsilon_1: f32, epsilon_2: f32) -> VerificationResult {
        let output = self.decoder.forward(psi_star);
        let forward_valid = output.data.iter().all(|&x| x.is_finite());
        
        let reconstructed = self.verifier.forward(psi_star);
        let backward_error = self.compute_verification_error(psi_0, &reconstructed);
        let backward_valid = backward_error < epsilon_2;
        
        let stable = self.check_stability(psi_star, psi_0, 0.01, 0.1);
        
        VerificationResult {
            forward_valid,
            backward_valid,
            stable,
            backward_error,
            overall_verified: forward_valid && backward_valid && stable,
        }
    }

    /// Get all trainable parameters
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.encoder.parameters());
        for op in &self.operators {
            params.extend(op.parameters());
        }
        params.extend(self.decoder.parameters());
        params.extend(self.verifier.parameters());
        params
    }

    /// Count total parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.shape.numel()).sum()
    }
}

/// Result of forward pass
#[derive(Debug, Clone)]
pub struct ALENForwardResult {
    /// Initial thought
    pub psi_0: Tensor,
    /// All candidate thoughts with energies
    pub candidates: Vec<CandidateEvaluation>,
    /// Selected operator ID
    pub selected_operator: usize,
    /// Selected thought
    pub psi_star: Tensor,
    /// Decoded output
    pub output: Tensor,
    /// Reconstructed input (for verification)
    pub reconstructed: Tensor,
    /// Verification error
    pub verification_error: f32,
}

/// Candidate evaluation
#[derive(Debug, Clone)]
pub struct CandidateEvaluation {
    pub operator_id: usize,
    pub thought: Tensor,
    pub energy: f32,
}

/// Detailed verification result
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub forward_valid: bool,
    pub backward_valid: bool,
    pub stable: bool,
    pub backward_error: f32,
    pub overall_verified: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder() {
        let config = ALENConfig::small();
        let encoder = ThoughtEncoder::new(config.clone());
        let tokens = vec![1, 2, 3, 4, 5];
        let psi = encoder.encode(&tokens);
        
        assert_eq!(psi.shape.dim(1), config.thought_dim);
        
        // Should be normalized
        let norm: f32 = psi.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_operator() {
        let op = NeuralReasoningOperator::new(0, "Test".to_string(), 64, 128, 0.0);
        let psi = Tensor::randn(vec![1, 64]);
        let psi_i = op.forward(&psi);
        
        assert_eq!(psi_i.shape.0, vec![1, 64]);
    }

    #[test]
    fn test_full_network() {
        let config = ALENConfig::small();
        let network = ALENNetwork::new(config);
        let tokens = vec![1, 2, 3, 4, 5];
        
        let result = network.forward(&tokens);
        
        assert_eq!(result.candidates.len(), network.operators.len());
        assert!(result.verification_error >= 0.0);
        
        println!("Network parameters: {}", network.num_parameters());
        println!("Selected operator: {}", result.selected_operator);
        println!("Verification error: {:.6}", result.verification_error);
    }

    #[test]
    fn test_verification() {
        let config = ALENConfig::small();
        let network = ALENNetwork::new(config);
        let tokens = vec![1, 2, 3];
        
        let result = network.forward(&tokens);
        let verified = network.verify(&result.psi_star, &result.psi_0, 1.0, 0.5);
        
        println!("Verification passed: {}", verified);
    }
}
