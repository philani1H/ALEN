//! Unified Cognition Module
//!
//! Implements the mathematical framework that unifies:
//! - Human Higher Processing (HHP) - latent cause inference
//! - Transformer attention - statistical relevance weighting  
//! - ALEN verification - bidirectional consistency
//!
//! Core equation:
//! ψ* = argmin_ψ [ E[E_k(ψ)] + λ·Loss_reconstruction + μ·Novelty ]
//!
//! Understanding = stable compression that preserves predictive power
//! I(ψ; X) is high AND dim(ψ) << dim(X)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// SECTION 1: Context Understanding (Mathematical Definition)
// ============================================================================

/// Represents an input sequence X = (x_1, x_2, ..., x_T)
/// Could be conversation, document, or any sequential input
#[derive(Debug, Clone)]
pub struct InputSequence {
    /// Raw token embeddings
    pub tokens: Vec<Vec<f64>>,
    /// Sequence length T
    pub length: usize,
    /// Embedding dimension
    pub dim: usize,
    /// Metadata about the sequence
    pub metadata: SequenceMetadata,
}

#[derive(Debug, Clone, Default)]
pub struct SequenceMetadata {
    pub source: Option<String>,
    pub timestamp: Option<i64>,
    pub context_type: ContextType,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum ContextType {
    #[default]
    Conversation,
    Document,
    Query,
    Memory,
}

impl InputSequence {
    pub fn new(tokens: Vec<Vec<f64>>) -> Self {
        let length = tokens.len();
        let dim = tokens.first().map(|t| t.len()).unwrap_or(0);
        Self {
            tokens,
            length,
            dim,
            metadata: SequenceMetadata::default(),
        }
    }

    /// Compute total information content (proxy for mutual information)
    pub fn information_content(&self) -> f64 {
        // Approximate I(ψ; X) using variance as proxy
        if self.tokens.is_empty() {
            return 0.0;
        }
        
        let mut total_variance = 0.0;
        for d in 0..self.dim {
            let values: Vec<f64> = self.tokens.iter().map(|t| t[d]).collect();
            let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
            let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            total_variance += variance;
        }
        
        // Higher variance = more information
        total_variance.ln().max(0.0)
    }
}

// ============================================================================
// SECTION 5.1: Multi-Perspective Context Encoders
// ============================================================================

/// Trait for context encoders: ψ^(k) = f_k(X)
pub trait ContextEncoder: Send + Sync {
    /// Encode input sequence to thought state
    fn encode(&self, input: &InputSequence) -> ThoughtVector;
    
    /// Encoder name for identification
    fn name(&self) -> &str;
    
    /// Encoder type for weighting
    fn encoder_type(&self) -> EncoderType;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncoderType {
    /// f_1: Transformer attention (statistical)
    TransformerAttention,
    /// f_2: Graph abstraction (symbolic)
    GraphAbstraction,
    /// f_3: Causal inference
    CausalInference,
    /// f_4: Compression/min-description-length
    CompressionMDL,
    /// f_5: Analogical mapping
    AnalogicalMapping,
}

/// Thought vector ψ ∈ ℝ^d - the compressed representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtVector {
    /// The vector values
    pub values: Vec<f64>,
    /// Dimension
    pub dim: usize,
    /// Confidence in this encoding
    pub confidence: f64,
    /// Source encoder
    pub source: Option<String>,
}

impl ThoughtVector {
    pub fn new(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
            dim,
            confidence: 0.0,
            source: None,
        }
    }

    pub fn from_values(values: Vec<f64>) -> Self {
        let dim = values.len();
        Self {
            values,
            dim,
            confidence: 1.0,
            source: None,
        }
    }

    pub fn norm(&self) -> f64 {
        self.values.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > 1e-10 {
            for v in &mut self.values {
                *v /= n;
            }
        }
    }

    pub fn cosine_similarity(&self, other: &ThoughtVector) -> f64 {
        if self.dim != other.dim {
            return 0.0;
        }
        let dot: f64 = self.values.iter().zip(&other.values).map(|(a, b)| a * b).sum();
        let norm_a = self.norm();
        let norm_b = other.norm();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// Weighted combination: ψ = Σ w_k ψ^(k)
    pub fn weighted_combine(thoughts: &[(&ThoughtVector, f64)]) -> Self {
        if thoughts.is_empty() {
            return Self::new(0);
        }
        
        let dim = thoughts[0].0.dim;
        let mut combined = vec![0.0; dim];
        let mut total_weight = 0.0;
        
        for (thought, weight) in thoughts {
            if thought.dim != dim {
                continue;
            }
            for (i, &v) in thought.values.iter().enumerate() {
                combined[i] += v * weight;
            }
            total_weight += weight;
        }
        
        // Normalize by total weight
        if total_weight > 1e-10 {
            for v in &mut combined {
                *v /= total_weight;
            }
        }
        
        Self::from_values(combined)
    }
}

// ============================================================================
// SECTION 5.2: Consensus Integration (Human-like)
// ============================================================================

/// Multi-perspective encoder bank
/// ψ_0 = Σ w_k ψ^(k) with Σ w_k = 1
pub struct MultiPerspectiveEncoder {
    /// Individual encoders
    encoders: Vec<Box<dyn ContextEncoder>>,
    /// Adaptive weights for each encoder
    weights: HashMap<EncoderType, f64>,
    /// Weight learning rate
    weight_lr: f64,
    /// Dimension
    dim: usize,
}

impl MultiPerspectiveEncoder {
    pub fn new(dim: usize) -> Self {
        let mut weights = HashMap::new();
        // Initialize equal weights
        weights.insert(EncoderType::TransformerAttention, 0.3);
        weights.insert(EncoderType::GraphAbstraction, 0.2);
        weights.insert(EncoderType::CausalInference, 0.2);
        weights.insert(EncoderType::CompressionMDL, 0.15);
        weights.insert(EncoderType::AnalogicalMapping, 0.15);
        
        Self {
            encoders: Vec::new(),
            weights,
            weight_lr: 0.01,
            dim,
        }
    }

    /// Add an encoder to the bank
    pub fn add_encoder(&mut self, encoder: Box<dyn ContextEncoder>) {
        self.encoders.push(encoder);
    }

    /// Encode with all perspectives and integrate
    pub fn encode(&self, input: &InputSequence) -> ConsensusResult {
        let mut perspective_results = Vec::new();
        
        for encoder in &self.encoders {
            let thought = encoder.encode(input);
            let weight = self.weights.get(&encoder.encoder_type()).copied().unwrap_or(0.1);
            perspective_results.push(PerspectiveResult {
                encoder_type: encoder.encoder_type(),
                encoder_name: encoder.name().to_string(),
                thought: thought.clone(),
                weight,
            });
        }
        
        // Compute consensus: ψ_0 = Σ w_k ψ^(k)
        let weighted_thoughts: Vec<(&ThoughtVector, f64)> = perspective_results
            .iter()
            .map(|r| (&r.thought, r.weight))
            .collect();
        
        let consensus = ThoughtVector::weighted_combine(&weighted_thoughts);
        
        // Compute agreement score (how much encoders agree)
        let agreement = self.compute_agreement(&perspective_results);
        
        ConsensusResult {
            consensus_thought: consensus,
            perspectives: perspective_results,
            agreement_score: agreement,
        }
    }

    /// Compute agreement between encoders
    fn compute_agreement(&self, results: &[PerspectiveResult]) -> f64 {
        if results.len() < 2 {
            return 1.0;
        }
        
        let mut total_sim = 0.0;
        let mut count = 0;
        
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                total_sim += results[i].thought.cosine_similarity(&results[j].thought);
                count += 1;
            }
        }
        
        if count > 0 {
            total_sim / count as f64
        } else {
            1.0
        }
    }

    /// Update weights based on verification success
    pub fn update_weights(&mut self, encoder_type: EncoderType, reward: f64) {
        if let Some(weight) = self.weights.get_mut(&encoder_type) {
            *weight += self.weight_lr * reward;
            *weight = weight.clamp(0.05, 0.5);
        }
        
        // Renormalize weights to sum to 1
        let total: f64 = self.weights.values().sum();
        if total > 1e-10 {
            for w in self.weights.values_mut() {
                *w /= total;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerspectiveResult {
    pub encoder_type: EncoderType,
    pub encoder_name: String,
    pub thought: ThoughtVector,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct ConsensusResult {
    pub consensus_thought: ThoughtVector,
    pub perspectives: Vec<PerspectiveResult>,
    pub agreement_score: f64,
}

// ============================================================================
// SECTION 6: Rate-Distortion Summarization
// ============================================================================

/// Summarization loss: L(S) = D(X || X̂(S)) + λ|S|
/// Where X̂(S) = reconstruction from summary
#[derive(Debug, Clone)]
pub struct SummarizationObjective {
    /// Information loss weight
    pub distortion_weight: f64,
    /// Compression weight (λ)
    pub compression_weight: f64,
    /// Target compression ratio
    pub target_ratio: f64,
}

impl Default for SummarizationObjective {
    fn default() -> Self {
        Self {
            distortion_weight: 1.0,
            compression_weight: 0.1,
            target_ratio: 0.2, // 5x compression
        }
    }
}

impl SummarizationObjective {
    /// Compute summarization loss
    /// L(S) = D(X || X̂(S)) + λ|S|
    pub fn compute_loss(
        &self,
        original: &InputSequence,
        summary: &ThoughtVector,
        reconstruction: &InputSequence,
    ) -> SummarizationLoss {
        // Distortion: D(X || X̂)
        let distortion = self.compute_distortion(original, reconstruction);
        
        // Compression: |S| relative to |X|
        let compression_ratio = summary.dim as f64 / (original.length * original.dim) as f64;
        let compression_penalty = (compression_ratio - self.target_ratio).abs();
        
        let total = self.distortion_weight * distortion 
                  + self.compression_weight * compression_penalty;
        
        SummarizationLoss {
            total,
            distortion,
            compression_ratio,
            compression_penalty,
        }
    }

    /// Compute distortion between original and reconstruction
    fn compute_distortion(&self, original: &InputSequence, reconstruction: &InputSequence) -> f64 {
        if original.tokens.is_empty() || reconstruction.tokens.is_empty() {
            return 1.0;
        }
        
        // Use mean squared error as distortion measure
        let mut total_mse = 0.0;
        let min_len = original.length.min(reconstruction.length);
        
        for t in 0..min_len {
            let orig = &original.tokens[t];
            let recon = &reconstruction.tokens[t];
            let min_dim = orig.len().min(recon.len());
            
            for d in 0..min_dim {
                total_mse += (orig[d] - recon[d]).powi(2);
            }
        }
        
        total_mse / (min_len * original.dim).max(1) as f64
    }
}

#[derive(Debug, Clone)]
pub struct SummarizationLoss {
    pub total: f64,
    pub distortion: f64,
    pub compression_ratio: f64,
    pub compression_penalty: f64,
}

// ============================================================================
// SECTION 7: Bidirectional Verification
// ============================================================================

/// Verification constraint: X → ψ → X̂, where |X̂ - X| < ε
/// "I truly understand this" iff reconstruction is accurate
pub struct BidirectionalVerifier {
    /// Reconstruction threshold ε
    pub epsilon: f64,
    /// Encoder function
    pub encoder: Option<Box<dyn Fn(&InputSequence) -> ThoughtVector + Send + Sync>>,
    /// Decoder function
    pub decoder: Option<Box<dyn Fn(&ThoughtVector) -> InputSequence + Send + Sync>>,
}

impl BidirectionalVerifier {
    pub fn new(epsilon: f64) -> Self {
        Self {
            epsilon,
            encoder: None,
            decoder: None,
        }
    }

    /// Verify understanding: |X̂ - X| < ε
    pub fn verify(
        &self,
        original: &InputSequence,
        thought: &ThoughtVector,
        reconstruction: &InputSequence,
    ) -> VerificationResult {
        let reconstruction_error = self.compute_reconstruction_error(original, reconstruction);
        let verified = reconstruction_error < self.epsilon;
        
        // Compute understanding score (inverse of error, clamped)
        let understanding_score = (1.0 - reconstruction_error / self.epsilon).clamp(0.0, 1.0);
        
        VerificationResult {
            verified,
            reconstruction_error,
            understanding_score,
            epsilon: self.epsilon,
        }
    }

    /// Compute |X̂ - X|
    fn compute_reconstruction_error(&self, original: &InputSequence, reconstruction: &InputSequence) -> f64 {
        if original.tokens.is_empty() {
            return 0.0;
        }
        
        let mut total_error = 0.0;
        let min_len = original.length.min(reconstruction.length);
        
        for t in 0..min_len {
            let orig = &original.tokens[t];
            let recon = &reconstruction.tokens[t];
            let min_dim = orig.len().min(recon.len());
            
            for d in 0..min_dim {
                total_error += (orig[d] - recon[d]).abs();
            }
        }
        
        // Add penalty for length mismatch
        let length_penalty = (original.length as f64 - reconstruction.length as f64).abs() 
                           / original.length as f64;
        
        total_error / (min_len * original.dim).max(1) as f64 + length_penalty * 0.1
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub verified: bool,
    pub reconstruction_error: f64,
    pub understanding_score: f64,
    pub epsilon: f64,
}


// ============================================================================
// SECTION 8: Energy-Based Cognition with Context Coherence
// ============================================================================

/// Extended energy function with context coherence:
/// E(ψ) = αC(ψ) + βR(ψ) + γU(ψ) + δ|ψ - ψ_context|²
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveEnergy {
    /// Constraint energy weight (α)
    pub alpha: f64,
    /// Risk energy weight (β)
    pub beta: f64,
    /// Uncertainty energy weight (γ)
    pub gamma: f64,
    /// Context coherence weight (δ)
    pub delta: f64,
}

impl Default for CognitiveEnergy {
    fn default() -> Self {
        Self {
            alpha: 0.4,
            beta: 0.2,
            gamma: 0.2,
            delta: 0.2,
        }
    }
}

impl CognitiveEnergy {
    /// Compute total energy: E(ψ) = αC(ψ) + βR(ψ) + γU(ψ) + δ|ψ - ψ_context|²
    pub fn compute(
        &self,
        thought: &ThoughtVector,
        context: Option<&ThoughtVector>,
        constraints: &[Constraint],
    ) -> EnergyBreakdown {
        // C(ψ): Constraint satisfaction
        let constraint_energy = self.compute_constraint_energy(thought, constraints);
        
        // R(ψ): Risk (deviation from safe regions)
        let risk_energy = self.compute_risk_energy(thought);
        
        // U(ψ): Uncertainty (entropy of thought)
        let uncertainty_energy = self.compute_uncertainty_energy(thought);
        
        // δ|ψ - ψ_context|²: Context coherence
        let context_energy = if let Some(ctx) = context {
            self.compute_context_coherence(thought, ctx)
        } else {
            0.0
        };
        
        let total = self.alpha * constraint_energy
                  + self.beta * risk_energy
                  + self.gamma * uncertainty_energy
                  + self.delta * context_energy;
        
        EnergyBreakdown {
            total,
            constraint: constraint_energy,
            risk: risk_energy,
            uncertainty: uncertainty_energy,
            context_coherence: context_energy,
            weights: self.clone(),
        }
    }

    /// C(ψ): How well thought satisfies constraints
    fn compute_constraint_energy(&self, thought: &ThoughtVector, constraints: &[Constraint]) -> f64 {
        if constraints.is_empty() {
            return 0.0;
        }
        
        let mut total_violation = 0.0;
        for constraint in constraints {
            let violation = constraint.compute_violation(thought);
            total_violation += violation;
        }
        
        total_violation / constraints.len() as f64
    }

    /// R(ψ): Risk energy - penalize extreme values
    fn compute_risk_energy(&self, thought: &ThoughtVector) -> f64 {
        // Risk = how far from unit sphere
        let norm = thought.norm();
        (norm - 1.0).abs()
    }

    /// U(ψ): Uncertainty energy - entropy-like measure
    fn compute_uncertainty_energy(&self, thought: &ThoughtVector) -> f64 {
        if thought.values.is_empty() {
            return 1.0;
        }
        
        // Compute variance as uncertainty proxy
        let mean: f64 = thought.values.iter().sum::<f64>() / thought.dim as f64;
        let variance: f64 = thought.values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / thought.dim as f64;
        
        // High variance = high uncertainty
        variance.sqrt()
    }

    /// δ|ψ - ψ_context|²: Context coherence
    fn compute_context_coherence(&self, thought: &ThoughtVector, context: &ThoughtVector) -> f64 {
        if thought.dim != context.dim {
            return 1.0;
        }
        
        let mut squared_diff = 0.0;
        for (a, b) in thought.values.iter().zip(&context.values) {
            squared_diff += (a - b).powi(2);
        }
        
        squared_diff / thought.dim as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyBreakdown {
    pub total: f64,
    pub constraint: f64,
    pub risk: f64,
    pub uncertainty: f64,
    pub context_coherence: f64,
    pub weights: CognitiveEnergy,
}

/// A constraint on thought vectors
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Target value or vector
    pub target: Vec<f64>,
    /// Tolerance
    pub tolerance: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum ConstraintType {
    /// Must be similar to target
    Similarity,
    /// Must be different from target
    Dissimilarity,
    /// Specific dimensions must match
    DimensionMatch,
    /// Norm constraint
    NormBound,
}

impl Constraint {
    pub fn compute_violation(&self, thought: &ThoughtVector) -> f64 {
        match self.constraint_type {
            ConstraintType::Similarity => {
                let target_thought = ThoughtVector::from_values(self.target.clone());
                let sim = thought.cosine_similarity(&target_thought);
                (1.0 - sim).max(0.0)
            }
            ConstraintType::Dissimilarity => {
                let target_thought = ThoughtVector::from_values(self.target.clone());
                let sim = thought.cosine_similarity(&target_thought);
                sim.max(0.0)
            }
            ConstraintType::DimensionMatch => {
                let mut violation = 0.0;
                for (i, &target_val) in self.target.iter().enumerate() {
                    if i < thought.dim {
                        violation += (thought.values[i] - target_val).abs();
                    }
                }
                violation / self.target.len().max(1) as f64
            }
            ConstraintType::NormBound => {
                let norm = thought.norm();
                let target_norm = self.target.first().copied().unwrap_or(1.0);
                (norm - target_norm).abs()
            }
        }
    }
}

// ============================================================================
// SECTION 9: Audience-Aware Decoding
// ============================================================================

/// Answer = Decode(ψ*, audience_model a)
/// Same thought → different expressions based on audience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudienceModel {
    /// Complexity level (0 = simple, 1 = expert)
    pub complexity: f64,
    /// Formality level (0 = casual, 1 = formal)
    pub formality: f64,
    /// Technical depth (0 = layman, 1 = specialist)
    pub technical_depth: f64,
    /// Verbosity (0 = concise, 1 = detailed)
    pub verbosity: f64,
    /// Emotional tone (0 = neutral, 1 = empathetic)
    pub emotional_tone: f64,
}

impl Default for AudienceModel {
    fn default() -> Self {
        Self {
            complexity: 0.5,
            formality: 0.5,
            technical_depth: 0.5,
            verbosity: 0.5,
            emotional_tone: 0.5,
        }
    }
}

impl AudienceModel {
    /// Child audience
    pub fn child() -> Self {
        Self {
            complexity: 0.2,
            formality: 0.1,
            technical_depth: 0.1,
            verbosity: 0.6,
            emotional_tone: 0.8,
        }
    }

    /// Expert audience
    pub fn expert() -> Self {
        Self {
            complexity: 0.9,
            formality: 0.8,
            technical_depth: 0.95,
            verbosity: 0.4,
            emotional_tone: 0.2,
        }
    }

    /// General adult audience
    pub fn general() -> Self {
        Self::default()
    }

    /// Convert to modulation vector for decoding
    pub fn to_modulation_vector(&self, dim: usize) -> Vec<f64> {
        let mut modulation = vec![0.0; dim];
        let fifth = dim / 5;
        
        // Distribute audience parameters across dimensions
        for i in 0..fifth {
            modulation[i] = self.complexity - 0.5;
        }
        for i in fifth..(2 * fifth) {
            modulation[i] = self.formality - 0.5;
        }
        for i in (2 * fifth)..(3 * fifth) {
            modulation[i] = self.technical_depth - 0.5;
        }
        for i in (3 * fifth)..(4 * fifth) {
            modulation[i] = self.verbosity - 0.5;
        }
        for i in (4 * fifth)..dim {
            modulation[i] = self.emotional_tone - 0.5;
        }
        
        modulation
    }

    /// Modulate thought for this audience
    pub fn modulate_thought(&self, thought: &ThoughtVector) -> ThoughtVector {
        let modulation = self.to_modulation_vector(thought.dim);
        let mut modulated = thought.values.clone();
        
        for (i, m) in modulation.iter().enumerate() {
            if i < modulated.len() {
                modulated[i] *= 1.0 + m * 0.3;
            }
        }
        
        let mut result = ThoughtVector::from_values(modulated);
        result.normalize();
        result
    }
}

// ============================================================================
// SECTION 11: Master Optimization Objective
// ============================================================================

/// The unified ALEN objective:
/// ψ* = argmin_ψ [ E[E_k(ψ)] + λ·Loss_reconstruction + μ·Novelty ]
#[derive(Debug, Clone)]
pub struct UnifiedObjective {
    /// Energy function
    pub energy: CognitiveEnergy,
    /// Reconstruction loss weight (λ)
    pub lambda: f64,
    /// Novelty weight (μ) - negative for encouraging novelty
    pub mu: f64,
    /// Summarization objective
    pub summarization: SummarizationObjective,
}

impl Default for UnifiedObjective {
    fn default() -> Self {
        Self {
            energy: CognitiveEnergy::default(),
            lambda: 0.3,
            mu: -0.1, // Negative to encourage novelty
            summarization: SummarizationObjective::default(),
        }
    }
}

impl UnifiedObjective {
    /// Compute the master objective
    /// L(ψ) = E[E_k(ψ)] + λ·Loss_reconstruction + μ·Novelty
    pub fn compute(
        &self,
        thought: &ThoughtVector,
        context: Option<&ThoughtVector>,
        constraints: &[Constraint],
        reconstruction_loss: f64,
        novelty: f64,
    ) -> ObjectiveResult {
        // E[E_k(ψ)]: Expected energy across perspectives
        let energy_result = self.energy.compute(thought, context, constraints);
        
        // λ·Loss_reconstruction
        let reconstruction_term = self.lambda * reconstruction_loss;
        
        // μ·Novelty (negative μ encourages novelty)
        let novelty_term = self.mu * novelty;
        
        let total = energy_result.total + reconstruction_term + novelty_term;
        
        ObjectiveResult {
            total,
            energy: energy_result,
            reconstruction_loss,
            reconstruction_term,
            novelty,
            novelty_term,
            lambda: self.lambda,
            mu: self.mu,
        }
    }

    /// Find optimal thought via gradient descent
    pub fn optimize(
        &self,
        initial: &ThoughtVector,
        context: Option<&ThoughtVector>,
        constraints: &[Constraint],
        learning_rate: f64,
        max_iterations: usize,
    ) -> OptimizationResult {
        let mut current = initial.clone();
        let mut history = Vec::new();
        
        for iteration in 0..max_iterations {
            // Compute current objective (with dummy reconstruction/novelty)
            let result = self.compute(&current, context, constraints, 0.0, 0.0);
            history.push(result.total);
            
            // Check convergence
            if iteration > 0 && (history[iteration - 1] - result.total).abs() < 1e-6 {
                return OptimizationResult {
                    optimal_thought: current,
                    final_objective: result,
                    iterations: iteration + 1,
                    converged: true,
                    history,
                };
            }
            
            // Gradient step (simplified - numerical gradient)
            current = self.gradient_step(&current, context, constraints, learning_rate);
        }
        
        let final_result = self.compute(&current, context, constraints, 0.0, 0.0);
        
        OptimizationResult {
            optimal_thought: current,
            final_objective: final_result,
            iterations: max_iterations,
            converged: false,
            history,
        }
    }

    /// Single gradient descent step
    fn gradient_step(
        &self,
        thought: &ThoughtVector,
        context: Option<&ThoughtVector>,
        constraints: &[Constraint],
        lr: f64,
    ) -> ThoughtVector {
        let eps = 1e-5;
        let mut gradient = vec![0.0; thought.dim];
        
        let current_loss = self.compute(thought, context, constraints, 0.0, 0.0).total;
        
        // Numerical gradient
        for i in 0..thought.dim {
            let mut perturbed = thought.values.clone();
            perturbed[i] += eps;
            let perturbed_thought = ThoughtVector::from_values(perturbed);
            let perturbed_loss = self.compute(&perturbed_thought, context, constraints, 0.0, 0.0).total;
            gradient[i] = (perturbed_loss - current_loss) / eps;
        }
        
        // Update
        let mut new_values = thought.values.clone();
        for (i, g) in gradient.iter().enumerate() {
            new_values[i] -= lr * g;
        }
        
        let mut result = ThoughtVector::from_values(new_values);
        result.normalize();
        result
    }
}

#[derive(Debug, Clone)]
pub struct ObjectiveResult {
    pub total: f64,
    pub energy: EnergyBreakdown,
    pub reconstruction_loss: f64,
    pub reconstruction_term: f64,
    pub novelty: f64,
    pub novelty_term: f64,
    pub lambda: f64,
    pub mu: f64,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub optimal_thought: ThoughtVector,
    pub final_objective: ObjectiveResult,
    pub iterations: usize,
    pub converged: bool,
    pub history: Vec<f64>,
}

// ============================================================================
// SECTION 12: Concrete Encoder Implementations
// ============================================================================

/// Transformer-style attention encoder (f_1)
pub struct AttentionEncoder {
    dim: usize,
    num_heads: usize,
}

impl AttentionEncoder {
    pub fn new(dim: usize, num_heads: usize) -> Self {
        Self { dim, num_heads }
    }

    /// Compute attention: α_ij = softmax(Q_i K_j^T / √d)
    fn compute_attention(&self, query: &[f64], keys: &[Vec<f64>]) -> Vec<f64> {
        let d_k = (self.dim as f64).sqrt();
        
        let scores: Vec<f64> = keys.iter()
            .map(|key| {
                let dot: f64 = query.iter().zip(key).map(|(q, k)| q * k).sum();
                dot / d_k
            })
            .collect();
        
        // Softmax
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();
        
        exp_scores.iter().map(|e| e / sum).collect()
    }
}

impl ContextEncoder for AttentionEncoder {
    fn encode(&self, input: &InputSequence) -> ThoughtVector {
        if input.tokens.is_empty() {
            return ThoughtVector::new(self.dim);
        }
        
        // Use mean as query
        let mut query = vec![0.0; self.dim.min(input.dim)];
        for token in &input.tokens {
            for (i, &v) in token.iter().enumerate() {
                if i < query.len() {
                    query[i] += v;
                }
            }
        }
        for v in &mut query {
            *v /= input.length as f64;
        }
        
        // Compute attention weights
        let attention = self.compute_attention(&query, &input.tokens);
        
        // Weighted sum: ψ = Σ α_j V_j
        let mut result = vec![0.0; self.dim];
        for (j, &alpha) in attention.iter().enumerate() {
            for (i, &v) in input.tokens[j].iter().enumerate() {
                if i < self.dim {
                    result[i] += alpha * v;
                }
            }
        }
        
        let mut thought = ThoughtVector::from_values(result);
        thought.source = Some("attention".to_string());
        thought.confidence = 0.8;
        thought
    }

    fn name(&self) -> &str {
        "TransformerAttention"
    }

    fn encoder_type(&self) -> EncoderType {
        EncoderType::TransformerAttention
    }
}

/// Compression encoder (f_4) - minimum description length
pub struct CompressionEncoder {
    dim: usize,
}

impl CompressionEncoder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl ContextEncoder for CompressionEncoder {
    fn encode(&self, input: &InputSequence) -> ThoughtVector {
        if input.tokens.is_empty() {
            return ThoughtVector::new(self.dim);
        }
        
        // PCA-like compression: find principal components
        // Simplified: use variance-weighted mean
        let mut result = vec![0.0; self.dim];
        let mut variances = vec![0.0; self.dim.min(input.dim)];
        
        // Compute mean
        let mut mean = vec![0.0; self.dim.min(input.dim)];
        for token in &input.tokens {
            for (i, &v) in token.iter().enumerate() {
                if i < mean.len() {
                    mean[i] += v;
                }
            }
        }
        for v in &mut mean {
            *v /= input.length as f64;
        }
        
        // Compute variance
        for token in &input.tokens {
            for (i, &v) in token.iter().enumerate() {
                if i < variances.len() {
                    variances[i] += (v - mean[i]).powi(2);
                }
            }
        }
        for v in &mut variances {
            *v /= input.length as f64;
        }
        
        // Weight by inverse variance (focus on consistent dimensions)
        let total_var: f64 = variances.iter().sum::<f64>().max(1e-10);
        for (i, &var) in variances.iter().enumerate() {
            if i < self.dim {
                let weight = 1.0 - (var / total_var);
                result[i] = mean[i] * weight;
            }
        }
        
        let mut thought = ThoughtVector::from_values(result);
        thought.normalize();
        thought.source = Some("compression".to_string());
        thought.confidence = 0.7;
        thought
    }

    fn name(&self) -> &str {
        "CompressionMDL"
    }

    fn encoder_type(&self) -> EncoderType {
        EncoderType::CompressionMDL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thought_vector_operations() {
        let v1 = ThoughtVector::from_values(vec![1.0, 0.0, 0.0]);
        let v2 = ThoughtVector::from_values(vec![0.0, 1.0, 0.0]);
        
        assert!((v1.norm() - 1.0).abs() < 1e-6);
        assert!(v1.cosine_similarity(&v2).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_combine() {
        let v1 = ThoughtVector::from_values(vec![1.0, 0.0]);
        let v2 = ThoughtVector::from_values(vec![0.0, 1.0]);
        
        let combined = ThoughtVector::weighted_combine(&[(&v1, 0.5), (&v2, 0.5)]);
        
        assert!((combined.values[0] - 0.5).abs() < 1e-6);
        assert!((combined.values[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_energy_computation() {
        let energy = CognitiveEnergy::default();
        let thought = ThoughtVector::from_values(vec![0.5; 64]);
        
        let result = energy.compute(&thought, None, &[]);
        
        assert!(result.total >= 0.0);
    }

    #[test]
    fn test_attention_encoder() {
        let encoder = AttentionEncoder::new(64, 4);
        let input = InputSequence::new(vec![
            vec![0.1; 64],
            vec![0.2; 64],
            vec![0.3; 64],
        ]);
        
        let thought = encoder.encode(&input);
        
        assert_eq!(thought.dim, 64);
        assert!(thought.norm() > 0.0);
    }

    #[test]
    fn test_unified_objective() {
        let objective = UnifiedObjective::default();
        let thought = ThoughtVector::from_values(vec![0.5; 64]);
        
        let result = objective.compute(&thought, None, &[], 0.1, 0.5);
        
        assert!(result.total.is_finite());
    }

    #[test]
    fn test_optimization() {
        let objective = UnifiedObjective::default();
        let initial = ThoughtVector::from_values(vec![0.5; 32]);
        
        let result = objective.optimize(&initial, None, &[], 0.01, 100);
        
        assert!(result.iterations > 0);
        assert!(result.final_objective.total.is_finite());
    }
}
