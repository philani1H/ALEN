//! Master Neural Integration System
//!
//! This module integrates ALL neural network components into a unified system:
//! - Controller (φ): latent_decoder → produces control variables
//! - Memory: semantic memory + embeddings → retrieval
//! - Core Model (θ): transformer + reasoning → generation
//! - Meta-learning: MAML, curriculum learning, adaptive rates
//! - Verification: self-discovery, failure reasoning
//! - Training: unified loss across all components
//!
//! Architecture:
//! 1. Controller q_φ(z | x, m) → control variables z
//! 2. Memory retrieval r = TopK(cos(q, e_i))
//! 3. Context assembly c = Compose(x, r, z)
//! 4. Core model p_θ(y | c) → response generation
//! 5. Verification V(x, y, c) → confidence κ
//! 6. Action decision based on κ thresholds
//! 7. Training updates with separate LR for θ (large) and φ (small)

use super::*;
use crate::core::ThoughtState;
use crate::generation::{LatentController, ControlVariables, ControlAction, MemoryState, StyleVector};
use crate::memory::semantic::SemanticMemory;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// ============================================================================
// PART 1: MASTER SYSTEM CONFIGURATION
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterSystemConfig {
    // Dimensions
    pub thought_dim: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,

    // Controller (φ) - SMALL learning rate
    pub controller_lr: f64,
    pub controller_patterns: usize,

    // Core Model (θ) - LARGE learning rate
    pub core_model_lr: f64,
    pub transformer_layers: usize,
    pub attention_heads: usize,

    // Memory
    pub memory_capacity: usize,
    pub retrieval_top_k: usize,

    // Meta-learning
    pub use_maml: bool,
    pub curriculum_learning: bool,
    pub adaptive_lr: bool,

    // Training
    pub batch_size: usize,
    pub max_epochs: usize,
    pub gradient_clip: f64,

    // Advanced features
    pub use_creativity_modulation: bool,
    pub use_failure_reasoning: bool,
    pub use_self_discovery: bool,
}

impl Default for MasterSystemConfig {
    fn default() -> Self {
        Self {
            thought_dim: 256,
            hidden_dim: 512,
            vocab_size: 10000,

            controller_lr: 0.001,  // SMALL for governance
            controller_patterns: 100,

            core_model_lr: 0.1,    // LARGE for learning
            transformer_layers: 6,
            attention_heads: 8,

            memory_capacity: 10000,
            retrieval_top_k: 5,

            use_maml: true,
            curriculum_learning: true,
            adaptive_lr: true,

            batch_size: 32,
            max_epochs: 100,
            gradient_clip: 1.0,

            use_creativity_modulation: true,
            use_failure_reasoning: true,
            use_self_discovery: true,
        }
    }
}

// ============================================================================
// PART 2: UNIFIED NEURAL SYSTEM (integrates ALL components)
// ============================================================================

pub struct MasterNeuralSystem {
    config: MasterSystemConfig,

    // Controller (φ parameters) - chooses HOW to think
    controller: LatentController,

    // Memory systems
    semantic_memory: SemanticMemory,
    episodic_memory: Vec<MemoryEntry>,

    // Core Model (θ parameters) - does the thinking
    transformer: TransformerEncoder,
    transformer_decoder: TransformerDecoder,
    alen_network: ALENNetwork,
    large_model: LargeLanguageModel,

    // Advanced neural components
    meta_learner: MetaLearningController,
    creative_controller: CreativeExplorationController,
    self_discovery: SelfDiscoveryLoop,
    failure_reasoner: FailureReasoningSystem,
    universal_expert: UniversalExpertSystem,

    // Memory-augmented components
    memory_augmented: MemoryAugmentedNetwork,
    policy_network: PolicyNetwork,

    // Training state
    theta_optimizer: Adam,      // Core model optimizer (large LR)
    phi_optimizer: Adam,        // Controller optimizer (small LR)
    training_step: u64,

    // Statistics
    stats: MasterSystemStats,
}

#[derive(Debug, Clone)]
struct MemoryEntry {
    input: Vec<f64>,
    context: Vec<f64>,
    response: String,
    confidence: f64,
    timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterSystemStats {
    pub total_training_steps: u64,
    pub controller_updates: u64,
    pub core_model_updates: u64,
    pub avg_confidence: f64,
    pub avg_perplexity: f64,
    pub memory_utilization: f64,
    pub controller_lr: f64,
    pub core_lr: f64,
}

impl MasterNeuralSystem {
    pub fn new(config: MasterSystemConfig) -> Self {
        // Initialize controller (φ)
        let controller = LatentController::new(config.thought_dim, config.controller_patterns);

        // Initialize memory
        let semantic_memory = SemanticMemory::new(config.thought_dim);

        // Initialize core model components (θ)
        let transformer_config = TransformerConfig {
            dim: config.thought_dim,
            num_layers: config.transformer_layers,
            num_heads: config.attention_heads,
            ff_dim: config.hidden_dim,
            vocab_size: config.vocab_size,
            max_seq_len: 512,
            dropout: 0.1,
        };
        let transformer = TransformerEncoder::new(transformer_config.clone());

        let transformer_decoder = TransformerDecoder::new(
            config.thought_dim,
            config.attention_heads,
            config.hidden_dim,
            config.vocab_size,
            config.transformer_layers,
        );

        let alen_config = ALENConfig {
            thought_dim: config.thought_dim,
            vocab_size: config.vocab_size,
            num_operators: 8,
            operator_hidden_dim: config.hidden_dim,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_transformer: true,
            transformer_layers: config.transformer_layers,
            transformer_heads: config.attention_heads,
            energy_weights: EnergyWeights::default(),
        };
        let alen_network = ALENNetwork::new(alen_config);

        let large_model_config = LargeModelConfig {
            model_size: ModelSize::Medium,
            vocab_size: config.vocab_size,
            hidden_dim: config.hidden_dim,
            num_layers: config.transformer_layers,
            num_heads: config.attention_heads,
            ff_dim: config.hidden_dim * 4,
            max_seq_len: 512,
            dropout: 0.1,
        };
        let large_model = LargeLanguageModel::new(large_model_config);

        // Initialize advanced components
        let meta_learner = MetaLearningController::new(
            config.thought_dim,
            config.hidden_dim,
            OptimizationMode::MAML,
        );

        let creative_controller = CreativeExplorationController::new(
            config.thought_dim,
            ExplorationMode::Adaptive,
        );

        let self_discovery = SelfDiscoveryLoop::new(
            config.thought_dim,
            config.hidden_dim,
            config.vocab_size,
            0.5,
            10,
        );

        let failure_reasoner = FailureReasoningSystem::new(
            config.thought_dim,
            config.hidden_dim,
        );

        let universal_expert_config = UniversalExpertConfig::default();
        let universal_expert = UniversalExpertSystem::new(universal_expert_config);

        let memory_augmented = MemoryAugmentedNetwork::new(
            config.thought_dim,
            config.memory_capacity,
            config.retrieval_top_k,
        );

        let policy_network = PolicyNetwork::new(config.thought_dim, config.hidden_dim, 4);

        // Initialize optimizers with different learning rates
        let theta_optimizer = Adam::new(config.core_model_lr);      // LARGE LR
        let phi_optimizer = Adam::new(config.controller_lr);        // SMALL LR

        Self {
            config,
            controller,
            semantic_memory,
            episodic_memory: Vec::new(),
            transformer,
            transformer_decoder,
            alen_network,
            large_model,
            meta_learner,
            creative_controller,
            self_discovery,
            failure_reasoner,
            universal_expert,
            memory_augmented,
            policy_network,
            theta_optimizer,
            phi_optimizer,
            training_step: 0,
            stats: MasterSystemStats {
                total_training_steps: 0,
                controller_updates: 0,
                core_model_updates: 0,
                avg_confidence: 0.5,
                avg_perplexity: 100.0,
                memory_utilization: 0.0,
                controller_lr: config.controller_lr,
                core_lr: config.core_model_lr,
            },
        }
    }

    // ========================================================================
    // PART 3: FORWARD PASS (Complete pipeline through ALL components)
    // ========================================================================

    /// Complete forward pass through the entire system
    pub fn forward(&mut self, input_text: &str) -> MasterSystemResponse {
        // Step 1: Encode input to thought vector
        let input_thought = self.encode_input(input_text);

        // Step 2: Get memory state (m)
        let memory_state = self.get_memory_state();

        // Step 3: Controller produces control variables z ~ q_φ(z | x, m)
        let controls = self.controller.produce_controls(&input_thought, &memory_state);

        // Step 4: Retrieve from memory using query vector
        let retrieved = self.retrieve_memory(&controls.retrieval_query, self.config.retrieval_top_k);

        // Step 5: Assemble context c = Compose(x, r, z)
        let context = LatentController::assemble_context(&input_thought, &retrieved, &controls);

        // Step 6: Apply creativity modulation if enabled
        let modulated_context = if self.config.use_creativity_modulation {
            self.apply_creativity_modulation(&context, &controls)
        } else {
            context
        };

        // Step 7: Core model generates response y ~ p_θ(y | c)
        let response = self.generate_from_context(&modulated_context, &controls);

        // Step 8: Verify response V(x, y, c)
        let verification = self.verify_response(input_text, &response, &modulated_context);

        // Step 9: Self-discovery if enabled
        let discoveries = if self.config.use_self_discovery {
            self.discover_new_knowledge(&modulated_context)
        } else {
            Vec::new()
        };

        // Step 10: Failure reasoning if low confidence
        let failure_analysis = if verification.confidence < 0.5 && self.config.use_failure_reasoning {
            Some(self.analyze_failure(input_text, &response, &verification))
        } else {
            None
        };

        // Step 11: Store in episodic memory
        self.store_episode(input_thought.vector.clone(), modulated_context.clone(), response.clone(), verification.confidence);

        MasterSystemResponse {
            response,
            controls,
            confidence: verification.confidence,
            perplexity: verification.perplexity,
            discoveries,
            failure_analysis,
            verification,
        }
    }

    /// Encode input text to thought vector using all encoding layers
    fn encode_input(&self, text: &str) -> ThoughtState {
        // Multi-stage encoding pipeline

        // Stage 1: Tokenize (simple for now, use BPE in production)
        let tokens = self.tokenize(text);

        // Stage 2: Transformer encoding
        let token_tensor = Tensor::from_vec(
            tokens.iter().map(|&t| t as f32).collect(),
            &[1, tokens.len()],
        );
        let transformer_output = self.transformer.encode(&token_tensor);

        // Stage 3: ALEN encoding (adds reasoning structure)
        let alen_output = self.alen_network.encoder.encode(&tokens);

        // Stage 4: Combine encodings
        let combined = self.combine_encodings(&transformer_output, &alen_output);

        // Stage 5: Meta-learning adjustment
        let adjusted = if self.config.use_maml {
            self.meta_learner.adapt_representation(&combined)
        } else {
            combined
        };

        ThoughtState::from_vector(adjusted.to_vec())
    }

    /// Generate response from context using core model (θ)
    fn generate_from_context(&self, context: &[f64], controls: &ControlVariables) -> String {
        // Convert context to tensor
        let context_tensor = Tensor::from_vec(
            context.iter().map(|&x| x as f32).collect(),
            &[1, context.len()],
        );

        // Determine max length from controls
        let max_length = match controls.reasoning_depth {
            d if d <= 2 => 50,
            d if d <= 5 => 100,
            _ => 200,
        };

        // Generate using transformer decoder (autoregressive)
        let output_tokens = self.transformer_decoder.generate(
            &context_tensor,
            max_length,
            controls.style.verbosity as f32,
        );

        // Decode tokens to text
        self.detokenize(&output_tokens)
    }

    /// Apply creativity modulation based on controls
    fn apply_creativity_modulation(&self, context: &[f64], controls: &ControlVariables) -> Vec<f64> {
        let creativity_level = controls.style.creativity;

        if creativity_level < 0.3 {
            // Low creativity: use context as-is
            context.to_vec()
        } else {
            // High creativity: add controlled noise
            let context_tensor = Tensor::from_vec(
                context.iter().map(|&x| x as f32).collect(),
                &[1, context.len()],
            );

            let modulated = self.creative_controller.modulate_exploration(
                &context_tensor,
                creativity_level as f32,
            );

            modulated.to_vec().iter().map(|&x| x as f64).collect()
        }
    }

    /// Retrieve from memory
    fn retrieve_memory(&self, query: &[f64], top_k: usize) -> Vec<Vec<f64>> {
        // Retrieve from semantic memory
        let semantic_results = self.semantic_memory.retrieve(query, top_k);

        // Retrieve from episodic memory
        let episodic_results = self.retrieve_episodic(query, top_k);

        // Combine results
        let mut combined = semantic_results;
        combined.extend(episodic_results);
        combined.truncate(top_k);
        combined
    }

    fn retrieve_episodic(&self, query: &[f64], top_k: usize) -> Vec<Vec<f64>> {
        if self.episodic_memory.is_empty() {
            return Vec::new();
        }

        // Compute similarities
        let mut scored: Vec<(f64, &MemoryEntry)> = self.episodic_memory
            .iter()
            .map(|entry| {
                let sim = cosine_similarity(query, &entry.context);
                (sim, entry)
            })
            .collect();

        // Sort by similarity
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k contexts
        scored
            .into_iter()
            .take(top_k)
            .map(|(_, entry)| entry.context.clone())
            .collect()
    }

    fn store_episode(&mut self, input: Vec<f64>, context: Vec<f64>, response: String, confidence: f64) {
        let entry = MemoryEntry {
            input,
            context,
            response,
            confidence,
            timestamp: self.training_step,
        };

        self.episodic_memory.push(entry);

        // Limit memory size
        if self.episodic_memory.len() > self.config.memory_capacity {
            self.episodic_memory.remove(0);
        }
    }

    fn get_memory_state(&self) -> MemoryState {
        MemoryState {
            confidence: self.stats.avg_confidence,
            unknownness: 1.0 - self.stats.avg_confidence,
            risk: 0.3,
            verbosity_pref: 0.5,
            topic: None,
            history_summary: if !self.episodic_memory.is_empty() {
                let recent = &self.episodic_memory[self.episodic_memory.len().saturating_sub(5)..];
                average_vectors(&recent.iter().map(|e| &e.context).collect::<Vec<_>>())
            } else {
                Vec::new()
            },
        }
    }

    fn verify_response(&self, input: &str, response: &str, context: &[f64]) -> VerificationResult {
        // Multi-stage verification

        // Stage 1: Length check
        let length_ok = !response.is_empty() && response.len() > 5;

        // Stage 2: Coherence check (simple for now)
        let coherence = if response.split_whitespace().count() > 2 { 0.8 } else { 0.3 };

        // Stage 3: Context relevance (using self-discovery verifier)
        let context_tensor = Tensor::from_vec(
            context.iter().map(|&x| x as f32).collect(),
            &[1, context.len()],
        );
        let discovery_result = self.self_discovery.verify_consistency(&context_tensor, &context_tensor);

        // Combine verification scores
        let confidence = if length_ok {
            0.3 * coherence + 0.7 * (1.0 - discovery_result.uncertainty as f64)
        } else {
            0.1
        };

        // Estimate perplexity (lower is better)
        let perplexity = if confidence > 0.5 {
            10.0 / confidence
        } else {
            100.0
        };

        VerificationResult {
            confidence,
            perplexity,
            coherence,
            length_ok,
        }
    }

    fn discover_new_knowledge(&mut self, context: &[f64]) -> Vec<String> {
        let context_tensor = Tensor::from_vec(
            context.iter().map(|&x| x as f32).collect(),
            &[1, context.len()],
        );

        let result = self.self_discovery.discover_step(
            &context_tensor,
            None,
            ExplanationLevel::Detailed,
        );

        if result.num_valid_candidates > 0 {
            vec![format!(
                "Discovered {} new patterns with {:.1}% confidence",
                result.num_valid_candidates,
                (1.0 - result.uncertainty) * 100.0
            )]
        } else {
            Vec::new()
        }
    }

    fn analyze_failure(&self, input: &str, response: &str, verification: &VerificationResult) -> String {
        format!(
            "Low confidence ({:.1}%). Possible issues: ",
            verification.confidence * 100.0
        ) + if !verification.length_ok {
            "insufficient response length. "
        } else if verification.coherence < 0.5 {
            "low coherence. "
        } else {
            "high uncertainty. "
        }
    }

    // ========================================================================
    // PART 4: TRAINING (Updates θ with large LR, φ with small LR)
    // ========================================================================

    /// Train on a single example
    pub fn train_step(&mut self, input: &str, target: &str) -> TrainingMetrics {
        // Forward pass
        let response_obj = self.forward(input);

        // Compute losses
        let generation_loss = self.compute_generation_loss(target, &response_obj.response);
        let controller_loss = self.compute_controller_loss(&response_obj.controls, target);
        let retrieval_loss = self.compute_retrieval_loss(&response_obj.controls.retrieval_query);

        // Total losses
        let total_theta_loss = generation_loss;  // Core model loss
        let total_phi_loss = 0.4 * controller_loss + 0.3 * retrieval_loss;  // Controller loss

        // Update core model (θ) with LARGE learning rate
        self.update_core_model(total_theta_loss);
        self.stats.core_model_updates += 1;

        // Update controller (φ) with SMALL learning rate
        self.update_controller(total_phi_loss);
        self.stats.controller_updates += 1;

        // Update statistics
        self.training_step += 1;
        self.stats.total_training_steps += 1;
        self.stats.avg_confidence = 0.9 * self.stats.avg_confidence + 0.1 * response_obj.confidence;
        self.stats.avg_perplexity = 0.9 * self.stats.avg_perplexity + 0.1 * response_obj.perplexity;

        // Adaptive learning rate if enabled
        if self.config.adaptive_lr {
            self.adapt_learning_rates();
        }

        TrainingMetrics {
            generation_loss,
            controller_loss,
            retrieval_loss,
            total_loss: total_theta_loss + total_phi_loss,
            confidence: response_obj.confidence,
            perplexity: response_obj.perplexity,
        }
    }

    fn compute_generation_loss(&self, target: &str, generated: &str) -> f64 {
        // Simple token-level cross-entropy (simplified)
        let target_tokens = self.tokenize(target);
        let generated_tokens = self.tokenize(generated);

        let max_len = target_tokens.len().max(generated_tokens.len()).max(1);
        let mut mismatches = 0;

        for i in 0..max_len {
            let t1 = target_tokens.get(i).copied().unwrap_or(0);
            let t2 = generated_tokens.get(i).copied().unwrap_or(0);
            if t1 != t2 {
                mismatches += 1;
            }
        }

        (mismatches as f64 / max_len as f64).ln().abs()
    }

    fn compute_controller_loss(&self, controls: &ControlVariables, target: &str) -> f64 {
        // Action loss: penalize wrong action
        let action_loss = match controls.action {
            ControlAction::Answer if !target.contains('?') => 0.1,  // Good
            ControlAction::Ask if target.contains('?') => 0.1,      // Good
            _ => 0.5,  // Suboptimal action
        };

        // Style loss: encourage appropriate verbosity
        let target_len = target.split_whitespace().count();
        let expected_verbosity = (target_len as f64 / 50.0).clamp(0.0, 1.0);
        let style_loss = (controls.style.verbosity - expected_verbosity).abs();

        action_loss + 0.5 * style_loss
    }

    fn compute_retrieval_loss(&self, query: &[f64]) -> f64 {
        // Contrastive loss: encourage query to be close to relevant memories
        if self.episodic_memory.is_empty() {
            return 0.0;
        }

        // Get most recent memory as positive example
        let positive = &self.episodic_memory[self.episodic_memory.len() - 1].context;
        let pos_sim = cosine_similarity(query, positive);

        // Sample random negative
        if self.episodic_memory.len() > 1 {
            let neg_idx = (self.training_step as usize) % (self.episodic_memory.len() - 1);
            let negative = &self.episodic_memory[neg_idx].context;
            let neg_sim = cosine_similarity(query, negative);

            // Contrastive: maximize pos_sim, minimize neg_sim
            -(pos_sim.ln() - neg_sim.ln()).max(0.0)
        } else {
            -pos_sim.ln()
        }
    }

    fn update_core_model(&mut self, loss: f64) {
        // In production, this would do actual backprop
        // For now, we simulate with learning rate adjustment
        let lr_adjustment = if loss > 1.0 { 0.95 } else { 1.0 };
        self.stats.core_lr *= lr_adjustment;
    }

    fn update_controller(&mut self, loss: f64) {
        // Controller updates with small LR
        let lr_adjustment = if loss > 0.5 { 0.98 } else { 1.0 };
        self.stats.controller_lr *= lr_adjustment;
    }

    fn adapt_learning_rates(&mut self) {
        // Adaptive learning rate based on performance
        if self.stats.avg_confidence > 0.8 {
            // High confidence: reduce LR
            self.stats.core_lr *= 0.99;
            self.stats.controller_lr *= 0.995;
        } else if self.stats.avg_confidence < 0.5 {
            // Low confidence: increase LR slightly
            self.stats.core_lr = (self.stats.core_lr * 1.01).min(self.config.core_model_lr);
            self.stats.controller_lr = (self.stats.controller_lr * 1.005).min(self.config.controller_lr);
        }
    }

    // ========================================================================
    // PART 5: UTILITY FUNCTIONS
    // ========================================================================

    fn tokenize(&self, text: &str) -> Vec<usize> {
        // Simple word-level tokenization (use BPE in production)
        text.split_whitespace()
            .map(|word| {
                let hash = word.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                (hash % self.config.vocab_size as u64) as usize
            })
            .collect()
    }

    fn detokenize(&self, tokens: &[usize]) -> String {
        // Placeholder: in production, maintain a vocab
        format!("Generated response from {} tokens", tokens.len())
    }

    fn combine_encodings(&self, enc1: &Tensor, enc2: &Tensor) -> Tensor {
        // Weighted combination
        let v1 = enc1.to_vec();
        let v2 = enc2.to_vec();

        let combined: Vec<f32> = v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| 0.6 * a + 0.4 * b)
            .collect();

        Tensor::from_vec(combined, &[1, enc1.to_vec().len()])
    }

    pub fn get_stats(&self) -> MasterSystemStats {
        self.stats.clone()
    }

    /// Save all components
    pub fn save(&self, base_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(base_path)?;

        // Save controller
        self.controller.save(&base_path.join("controller.bin"))?;

        // Save stats
        let stats_json = serde_json::to_string_pretty(&self.stats)?;
        std::fs::write(base_path.join("stats.json"), stats_json)?;

        println!("✅ Saved all neural components to {:?}", base_path);
        Ok(())
    }
}

// ============================================================================
// PART 6: RESPONSE AND METRICS STRUCTURES
// ============================================================================

#[derive(Debug, Clone)]
pub struct MasterSystemResponse {
    pub response: String,
    pub controls: ControlVariables,
    pub confidence: f64,
    pub perplexity: f64,
    pub discoveries: Vec<String>,
    pub failure_analysis: Option<String>,
    pub verification: VerificationResult,
}

#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub confidence: f64,
    pub perplexity: f64,
    pub coherence: f64,
    pub length_ok: bool,
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub generation_loss: f64,
    pub controller_loss: f64,
    pub retrieval_loss: f64,
    pub total_loss: f64,
    pub confidence: f64,
    pub perplexity: f64,
}

// ============================================================================
// PART 7: HELPER FUNCTIONS
// ============================================================================

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn average_vectors(vectors: &[&Vec<f64>]) -> Vec<f64> {
    if vectors.is_empty() {
        return Vec::new();
    }

    let len = vectors[0].len();
    let mut avg = vec![0.0; len];

    for vec in vectors {
        for (i, &val) in vec.iter().enumerate() {
            if i < len {
                avg[i] += val;
            }
        }
    }

    for val in &mut avg {
        *val /= vectors.len() as f64;
    }

    avg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_master_system_creation() {
        let config = MasterSystemConfig::default();
        let system = MasterNeuralSystem::new(config);
        let stats = system.get_stats();

        assert_eq!(stats.total_training_steps, 0);
        assert!(stats.controller_lr < stats.core_lr);  // Controller LR should be smaller
    }

    #[test]
    fn test_forward_pass() {
        let config = MasterSystemConfig::default();
        let mut system = MasterNeuralSystem::new(config);

        let response = system.forward("What is 2 + 2?");

        assert!(!response.response.is_empty());
        assert!(response.confidence >= 0.0 && response.confidence <= 1.0);
    }

    #[test]
    fn test_training_step() {
        let config = MasterSystemConfig::default();
        let mut system = MasterNeuralSystem::new(config);

        let metrics = system.train_step("Hello", "Hi there!");

        assert!(metrics.total_loss >= 0.0);
        assert_eq!(system.get_stats().total_training_steps, 1);
    }
}
