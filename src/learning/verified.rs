//! ALEN Verified Learning System
//!
//! The core innovation: Learning by proving understanding.
//! Just like humans, ALEN must demonstrate it understands a problem
//! by verifying the solution backward to the question.
//!
//! Verification Criteria:
//! 1. Forward Consistency: Does the solution solve the problem?
//! 2. Backward Consistency: Does the solution lead back to the problem?
//! 3. Confidence Check: Is the model confident in its answer?
//! 4. Memory Coherence: Does this align with existing knowledge?
//!
//! Only when ALL conditions pass does learning commit to memory.

use crate::core::{ThoughtState, Evaluator, EnergyWeights, OperatorManager, OperatorType};
use crate::memory::{SemanticMemory, EmbeddingEngine, EmbeddingConfig};
use crate::knowledge::{KnowledgeBase, KnowledgeItem, KnowledgeCategory};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Verification thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationThresholds {
    /// Maximum forward error (output vs expected)
    pub forward_error: f64,
    /// Maximum backward error (reconstructed input vs original)
    pub backward_error: f64,
    /// Minimum confidence required
    pub min_confidence: f64,
    /// Maximum energy allowed
    pub max_energy: f64,
    /// Minimum memory coherence
    pub min_coherence: f64,
}

impl Default for VerificationThresholds {
    fn default() -> Self {
        Self {
            forward_error: 0.3,
            backward_error: 0.4,
            min_confidence: 0.6,
            max_energy: 1.5,
            min_coherence: 0.3,
        }
    }
}

/// Result of a verification attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Did the verification pass?
    pub passed: bool,
    /// Forward consistency error
    pub forward_error: f64,
    /// Backward consistency error
    pub backward_error: f64,
    /// Confidence score
    pub confidence: f64,
    /// Energy of the solution
    pub energy: f64,
    /// Coherence with existing memory
    pub coherence: f64,
    /// Which checks passed
    pub checks_passed: VerificationChecks,
    /// Detailed reasoning about the verification
    pub reasoning: String,
}

/// Individual verification checks
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VerificationChecks {
    pub forward_check: bool,
    pub backward_check: bool,
    pub confidence_check: bool,
    pub energy_check: bool,
    pub coherence_check: bool,
}

impl VerificationChecks {
    pub fn all_passed(&self) -> bool {
        self.forward_check && self.backward_check && self.confidence_check 
            && self.energy_check && self.coherence_check
    }
    
    pub fn count_passed(&self) -> usize {
        [self.forward_check, self.backward_check, self.confidence_check,
         self.energy_check, self.coherence_check]
            .iter()
            .filter(|&&x| x)
            .count()
    }
}

/// A learning session with verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedLearningSession {
    /// Problems attempted
    pub problems_attempted: usize,
    /// Problems verified successfully
    pub problems_verified: usize,
    /// Problems failed verification
    pub problems_failed: usize,
    /// Average forward error
    pub avg_forward_error: f64,
    /// Average backward error
    pub avg_backward_error: f64,
    /// Average confidence
    pub avg_confidence: f64,
    /// Categories learned
    pub categories_learned: HashMap<String, usize>,
}

impl Default for VerifiedLearningSession {
    fn default() -> Self {
        Self {
            problems_attempted: 0,
            problems_verified: 0,
            problems_failed: 0,
            avg_forward_error: 0.0,
            avg_backward_error: 0.0,
            avg_confidence: 0.0,
            categories_learned: HashMap::new(),
        }
    }
}

/// The Verified Learning Engine
pub struct VerifiedLearner {
    /// Dimension of thought vectors
    pub dimension: usize,
    /// Verification thresholds
    pub thresholds: VerificationThresholds,
    /// Operator manager for reasoning
    pub operators: OperatorManager,
    /// Evaluator for energy computation
    pub evaluator: Evaluator,
    /// Embedding engine
    pub embedder: EmbeddingEngine,
    /// Knowledge base for training data
    pub knowledge: KnowledgeBase,
    /// Current session stats
    pub session: VerifiedLearningSession,
    /// Inverse operator weights for backward inference
    inverse_weights: HashMap<OperatorType, DVector<f64>>,
}

impl VerifiedLearner {
    /// Create a new verified learner
    pub fn new(dimension: usize) -> Self {
        let mut inverse_weights = HashMap::new();
        
        // Initialize inverse weights for each operator type
        for op_type in [
            OperatorType::Logical, OperatorType::Exploratory, OperatorType::Analogical,
            OperatorType::Probabilistic, OperatorType::Heuristic, OperatorType::Conservative,
            OperatorType::Analytical, OperatorType::Intuitive,
        ] {
            inverse_weights.insert(op_type, DVector::from_element(dimension, 1.0));
        }
        
        Self {
            dimension,
            thresholds: VerificationThresholds::default(),
            operators: OperatorManager::new(dimension),
            evaluator: Evaluator::new(EnergyWeights::default(), 0.7),
            embedder: EmbeddingEngine::new(EmbeddingConfig {
                dimension,
                normalize: true,
                vocab_size: 10000,
            }),
            knowledge: KnowledgeBase::new(),
            session: VerifiedLearningSession::default(),
            inverse_weights,
        }
    }

    /// Perform backward inference: T⁻¹(output) ≈ input?
    /// This is how we verify the AI truly understood the problem.
    pub fn backward_inference(&self, output: &ThoughtState, operator_used: OperatorType) -> ThoughtState {
        // Get inverse weights for this operator
        let inv_weights = self.inverse_weights.get(&operator_used)
            .cloned()
            .unwrap_or_else(|| DVector::from_element(self.dimension, 1.0));
        
        // Apply inverse transformation
        // For linear operators T(x) = Wx, inverse is approximately x ≈ W⁺y
        // We approximate this with learned inverse weights
        let mut reconstructed = vec![0.0; self.dimension];
        
        for i in 0..self.dimension {
            // Inverse operation: multiply by inverse weight and apply inverse nonlinearity
            let v = output.vector[i] * inv_weights[i];
            // Approximate inverse of activation (for tanh: atanh, for relu: identity on positive)
            reconstructed[i] = if v.abs() < 0.999 {
                v.atanh().clamp(-3.0, 3.0)
            } else {
                v.signum() * 3.0
            };
        }
        
        // Normalize
        let norm: f64 = reconstructed.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for v in &mut reconstructed {
                *v /= norm;
            }
        }
        
        ThoughtState::from_vector(reconstructed, self.dimension)
    }

    /// Verify a candidate solution
    pub fn verify(&self, 
        input: &ThoughtState, 
        output: &ThoughtState, 
        expected: &ThoughtState,
        operator_used: OperatorType,
        energy: f64,
        memory: Option<&SemanticMemory>,
    ) -> VerificationResult {
        // 1. Forward Check: output ≈ expected?
        let forward_error = Self::vector_distance(&output.vector, &expected.vector);
        let forward_check = forward_error < self.thresholds.forward_error;
        
        // 2. Backward Check: T⁻¹(output) ≈ input?
        let reconstructed = self.backward_inference(output, operator_used);
        let backward_error = Self::vector_distance(&reconstructed.vector, &input.vector);
        let backward_check = backward_error < self.thresholds.backward_error;
        
        // 3. Confidence Check
        let confidence = output.confidence;
        let confidence_check = confidence >= self.thresholds.min_confidence;
        
        // 4. Energy Check
        let energy_check = energy < self.thresholds.max_energy;
        
        // 5. Coherence Check (if memory available)
        let coherence = if let Some(mem) = memory {
            self.compute_coherence(output, mem)
        } else {
            1.0 // Assume coherent if no memory to check against
        };
        let coherence_check = coherence >= self.thresholds.min_coherence;
        
        let checks = VerificationChecks {
            forward_check,
            backward_check,
            confidence_check,
            energy_check,
            coherence_check,
        };
        
        let passed = checks.all_passed();
        
        let reasoning = self.generate_verification_reasoning(
            forward_error, backward_error, confidence, energy, coherence, &checks
        );
        
        VerificationResult {
            passed,
            forward_error,
            backward_error,
            confidence,
            energy,
            coherence,
            checks_passed: checks,
            reasoning,
        }
    }

    /// Compute coherence with existing semantic memory
    fn compute_coherence(&self, output: &ThoughtState, memory: &SemanticMemory) -> f64 {
        // Search for similar facts
        match memory.find_similar(&output.vector, 5) {
            Ok(similar) => {
                if similar.is_empty() {
                    0.5 // Neutral coherence if no similar memories
                } else {
                    // Average similarity to top matches
                    similar.iter()
                        .map(|(fact, _sim)| Self::cosine_similarity(&output.vector, &fact.embedding))
                        .sum::<f64>() / similar.len() as f64
                }
            }
            Err(_) => 0.5,
        }
    }

    /// Generate human-readable reasoning about verification
    fn generate_verification_reasoning(
        &self,
        forward_error: f64,
        backward_error: f64,
        confidence: f64,
        energy: f64,
        coherence: f64,
        checks: &VerificationChecks,
    ) -> String {
        let mut reasons = Vec::new();
        
        if checks.forward_check {
            reasons.push(format!("✓ Solution matches expected (error: {:.3})", forward_error));
        } else {
            reasons.push(format!("✗ Solution differs from expected (error: {:.3} > {:.3})", 
                forward_error, self.thresholds.forward_error));
        }
        
        if checks.backward_check {
            reasons.push(format!("✓ Backward verification passed (error: {:.3})", backward_error));
        } else {
            reasons.push(format!("✗ Cannot reconstruct problem from solution (error: {:.3} > {:.3})", 
                backward_error, self.thresholds.backward_error));
        }
        
        if checks.confidence_check {
            reasons.push(format!("✓ Model is confident ({:.1}%)", confidence * 100.0));
        } else {
            reasons.push(format!("✗ Model uncertain ({:.1}% < {:.1}%)", 
                confidence * 100.0, self.thresholds.min_confidence * 100.0));
        }
        
        if checks.energy_check {
            reasons.push(format!("✓ Low energy state ({:.3})", energy));
        } else {
            reasons.push(format!("✗ High energy state ({:.3} > {:.3})", 
                energy, self.thresholds.max_energy));
        }
        
        if checks.coherence_check {
            reasons.push(format!("✓ Coherent with memory ({:.1}%)", coherence * 100.0));
        } else {
            reasons.push(format!("✗ Incoherent with memory ({:.1}% < {:.1}%)", 
                coherence * 100.0, self.thresholds.min_coherence * 100.0));
        }
        
        reasons.join("\n")
    }

    /// Train on a single knowledge item with verification
    pub fn train_verified(&mut self, item: &KnowledgeItem) -> VerificationResult {
        self.session.problems_attempted += 1;
        
        // Embed input and expected output
        let input_state = self.embedder.embed_text(&item.input);
        let expected_state = self.embedder.embed_text(&item.output);
        
        // Create a problem for evaluation
        let problem = crate::core::Problem::training(&item.input, &item.output, self.dimension);
        
        // Generate candidates using all operators
        let mut candidates: Vec<(OperatorType, ThoughtState)> = Vec::new();
        for op_type in OperatorType::all() {
            for op in self.operators.get_by_type(op_type) {
                let candidate = op.apply(&input_state);
                candidates.push((op_type, candidate));
            }
        }
        
        // If no candidates, create one using input directly
        if candidates.is_empty() {
            candidates.push((OperatorType::Logical, input_state.clone()));
        }
        
        // Evaluate all candidates
        let mut evaluated: Vec<(OperatorType, ThoughtState, f64)> = Vec::new();
        for (op_type, candidate) in candidates {
            let result = self.evaluator.evaluate(&candidate, &problem);
            evaluated.push((op_type, candidate, result.total));
        }
        
        // Find best candidate
        let (best_op, best_candidate, best_energy) = evaluated
            .into_iter()
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .unwrap_or_else(|| (OperatorType::Logical, input_state.clone(), 100.0));
        
        // Verify the best candidate
        let result = self.verify(
            &input_state,
            &best_candidate,
            &expected_state,
            best_op,
            best_energy,
            None, // TODO: pass memory
        );
        
        // Update session stats
        self.update_session_stats(&result, &item.category.to_string());
        
        // If verified, update operator weights (reward successful operator)
        if result.passed {
            let reward = 1.0 / (result.energy + 0.1);
            let learning_rate = 0.01;
            // Find the operator ID and update
            if let Some(op) = self.operators.get_by_type(best_op).first() {
                let op_id = op.id.clone();
                self.operators.update_weights(&op_id, reward, learning_rate);
            }
            
            // Update inverse weights for better backward inference
            self.update_inverse_weights(best_op, &input_state, &best_candidate);
        }
        
        result
    }

    /// Update inverse weights based on successful forward pass
    fn update_inverse_weights(&mut self, op_type: OperatorType, input: &ThoughtState, output: &ThoughtState) {
        let learning_rate = 0.01;
        
        if let Some(inv_weights) = self.inverse_weights.get_mut(&op_type) {
            for i in 0..self.dimension {
                if output.vector[i].abs() > 1e-10 {
                    // Approximate: if output[i] = f(input[i] * w[i]), then inv_w[i] ≈ input[i] / output[i]
                    let target_inv = input.vector[i] / (output.vector[i] + 1e-10);
                    let current = inv_weights[i];
                    inv_weights[i] = current + learning_rate * (target_inv.clamp(-10.0, 10.0) - current);
                }
            }
        }
    }

    /// Update session statistics
    fn update_session_stats(&mut self, result: &VerificationResult, category: &str) {
        if result.passed {
            self.session.problems_verified += 1;
            *self.session.categories_learned.entry(category.to_string()).or_insert(0) += 1;
        } else {
            self.session.problems_failed += 1;
        }
        
        // Update running averages
        let n = self.session.problems_attempted as f64;
        self.session.avg_forward_error = 
            (self.session.avg_forward_error * (n - 1.0) + result.forward_error) / n;
        self.session.avg_backward_error = 
            (self.session.avg_backward_error * (n - 1.0) + result.backward_error) / n;
        self.session.avg_confidence = 
            (self.session.avg_confidence * (n - 1.0) + result.confidence) / n;
    }

    /// Train on all knowledge in a category
    pub fn train_category(&mut self, category: KnowledgeCategory) -> Vec<VerificationResult> {
        let items: Vec<KnowledgeItem> = self.knowledge.by_category(category)
            .into_iter()
            .cloned()
            .collect();
        
        items.iter().map(|item| self.train_verified(item)).collect()
    }

    /// Train on entire knowledge base
    pub fn train_all(&mut self) -> VerifiedLearningSession {
        let all_items: Vec<KnowledgeItem> = self.knowledge.all_items().to_vec();
        
        for item in &all_items {
            self.train_verified(item);
        }
        
        self.session.clone()
    }

    /// Get the current session statistics
    pub fn get_session(&self) -> &VerifiedLearningSession {
        &self.session
    }

    /// Reset session statistics
    pub fn reset_session(&mut self) {
        self.session = VerifiedLearningSession::default();
    }

    /// Helper: Euclidean distance between vectors
    fn vector_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Helper: Cosine similarity
    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm_a > 1e-10 && norm_b > 1e-10 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Training curriculum - progressive learning with verification
pub struct TrainingCurriculum {
    /// Phases of training
    pub phases: Vec<CurriculumPhase>,
    /// Current phase index
    pub current_phase: usize,
    /// Overall progress (0.0 to 1.0)
    pub progress: f64,
}

/// A phase in the training curriculum
#[derive(Debug, Clone)]
pub struct CurriculumPhase {
    /// Phase name
    pub name: String,
    /// Categories to learn
    pub categories: Vec<KnowledgeCategory>,
    /// Maximum difficulty level
    pub max_difficulty: u8,
    /// Required verification rate to advance
    pub required_verification_rate: f64,
    /// Number of epochs
    pub epochs: usize,
}

impl TrainingCurriculum {
    /// Create a standard curriculum
    pub fn standard() -> Self {
        Self {
            phases: vec![
                // Phase 1: Basic Foundations
                CurriculumPhase {
                    name: "Foundations".into(),
                    categories: vec![
                        KnowledgeCategory::Mathematics,
                        KnowledgeCategory::Language,
                    ],
                    max_difficulty: 3,
                    required_verification_rate: 0.7,
                    epochs: 3,
                },
                // Phase 2: Core Sciences
                CurriculumPhase {
                    name: "Core Sciences".into(),
                    categories: vec![
                        KnowledgeCategory::Physics,
                        KnowledgeCategory::Chemistry,
                        KnowledgeCategory::Biology,
                    ],
                    max_difficulty: 5,
                    required_verification_rate: 0.6,
                    epochs: 3,
                },
                // Phase 3: Advanced Topics
                CurriculumPhase {
                    name: "Advanced Topics".into(),
                    categories: vec![
                        KnowledgeCategory::ComputerScience,
                        KnowledgeCategory::Logic,
                    ],
                    max_difficulty: 7,
                    required_verification_rate: 0.5,
                    epochs: 5,
                },
                // Phase 4: Mastery
                CurriculumPhase {
                    name: "Mastery".into(),
                    categories: vec![
                        KnowledgeCategory::Physics,
                        KnowledgeCategory::Mathematics,
                        KnowledgeCategory::ComputerScience,
                    ],
                    max_difficulty: 10,
                    required_verification_rate: 0.4,
                    epochs: 10,
                },
            ],
            current_phase: 0,
            progress: 0.0,
        }
    }

    /// Get current phase
    pub fn current(&self) -> Option<&CurriculumPhase> {
        self.phases.get(self.current_phase)
    }

    /// Advance to next phase if requirements met
    pub fn advance_if_ready(&mut self, verification_rate: f64) -> bool {
        if let Some(phase) = self.phases.get(self.current_phase) {
            if verification_rate >= phase.required_verification_rate {
                if self.current_phase < self.phases.len() - 1 {
                    self.current_phase += 1;
                    self.progress = self.current_phase as f64 / self.phases.len() as f64;
                    return true;
                }
            }
        }
        false
    }

    /// Check if curriculum is complete
    pub fn is_complete(&self) -> bool {
        self.current_phase >= self.phases.len() - 1 && self.progress >= 0.99
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verified_learner_creation() {
        let learner = VerifiedLearner::new(128);
        assert_eq!(learner.dimension, 128);
        assert!(!learner.knowledge.is_empty());
    }

    #[test]
    fn test_backward_inference() {
        let learner = VerifiedLearner::new(64);
        let output = ThoughtState::random(64);
        let reconstructed = learner.backward_inference(&output, OperatorType::Logical);
        assert_eq!(reconstructed.vector.len(), 64);
    }

    #[test]
    fn test_verification_checks() {
        let checks = VerificationChecks {
            forward_check: true,
            backward_check: true,
            confidence_check: true,
            energy_check: true,
            coherence_check: true,
        };
        assert!(checks.all_passed());
        assert_eq!(checks.count_passed(), 5);
        
        let partial = VerificationChecks {
            forward_check: true,
            backward_check: false,
            confidence_check: true,
            energy_check: true,
            coherence_check: false,
        };
        assert!(!partial.all_passed());
        assert_eq!(partial.count_passed(), 3);
    }

    #[test]
    fn test_train_single_item() {
        let mut learner = VerifiedLearner::new(64);
        let item = KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "arithmetic".into(),
            input: "What is 2+2?".into(),
            output: "4".into(),
            reasoning: "Basic addition".into(),
            backward_check: "What adds to 4?".into(),
            related: vec!["addition".into()],
            difficulty: 1,
            prerequisites: vec![],
        };
        
        let result = learner.train_verified(&item);
        assert!(result.forward_error >= 0.0);
        assert!(result.backward_error >= 0.0);
    }

    #[test]
    fn test_curriculum() {
        let mut curriculum = TrainingCurriculum::standard();
        assert_eq!(curriculum.current_phase, 0);
        assert!(!curriculum.is_complete());
        
        // Should not advance with low rate
        assert!(!curriculum.advance_if_ready(0.3));
        
        // Should advance with high rate
        assert!(curriculum.advance_if_ready(0.8));
        assert_eq!(curriculum.current_phase, 1);
    }

    #[test]
    fn test_session_stats() {
        let mut learner = VerifiedLearner::new(64);
        let items = learner.knowledge.by_category(KnowledgeCategory::Mathematics);
        
        if !items.is_empty() {
            learner.train_verified(&items[0].clone());
            assert_eq!(learner.session.problems_attempted, 1);
        }
    }
}
