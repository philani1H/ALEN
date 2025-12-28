//! Reasoning Engine - Multi-Modal Generation with Knowledge Anchoring
//!
//! Implements: h_latent = concept_vector + knowledge_vector + α * creativity_vector
//!
//! Key Features:
//! - Combines concept understanding, learned knowledge, and controlled creativity
//! - Knowledge anchoring: verifies generated latents against semantic memory
//! - Creativity injection: α parameter controls balance between faithful/creative
//! - Latent propagation: temporal consistency for video generation
//! - Multi-modal support: images, video, audio

use crate::core::{ThoughtState, BiasVector};
use crate::memory::SemanticMemory;
use crate::control::emotions::EmotionalState;
use crate::generation::factual_decoder::FactualThresholds;
use crate::reasoning::chain_of_thought::{ChainOfThoughtReasoner, ReasoningChain, ReasoningStep};
use serde::{Deserialize, Serialize};

/// Reasoning engine for multi-modal generation
pub struct ReasoningEngine {
    /// Vector dimension
    pub dimension: usize,
    /// Knowledge verification thresholds
    pub thresholds: FactualThresholds,
    /// Current emotional state for mood-aware generation
    pub emotional_state: Option<EmotionalState>,
    /// Chain-of-thought reasoner for multi-step reasoning
    pub chain_reasoner: ChainOfThoughtReasoner,
}

impl ReasoningEngine {
    pub fn new(dimension: usize, thresholds: FactualThresholds) -> Self {
        Self {
            dimension,
            thresholds,
            emotional_state: None,
            chain_reasoner: ChainOfThoughtReasoner::default(),
        }
    }

    /// Create with balanced verification
    pub fn balanced(dimension: usize) -> Self {
        Self::new(dimension, FactualThresholds::balanced())
    }

    /// Set emotional state for mood-aware generation
    pub fn with_emotion(mut self, emotion: EmotionalState) -> Self {
        self.emotional_state = Some(emotion);
        self
    }

    /// Compute latent vector for generation
    ///
    /// h_latent = concept_vector + knowledge_vector + α * creativity_vector
    ///
    /// Parameters:
    /// - concept_vector: What we want to generate (user intent)
    /// - knowledge_vector: What we know from memory (grounding)
    /// - creativity_vector: Creative exploration direction (bias)
    /// - alpha: Creativity injection weight (0.0 = purely factual, 1.0 = highly creative)
    pub fn compute_latent(
        &self,
        concept_vector: &[f64],
        knowledge_vector: &[f64],
        creativity_vector: &[f64],
        alpha: f64,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        if concept_vector.len() != self.dimension
            || knowledge_vector.len() != self.dimension
            || creativity_vector.len() != self.dimension
        {
            return Err("Vector dimension mismatch".into());
        }

        // Combine vectors: h_latent = concept + knowledge + α * creativity
        let mut latent = vec![0.0; self.dimension];

        for i in 0..self.dimension {
            latent[i] = concept_vector[i]
                      + knowledge_vector[i]
                      + alpha * creativity_vector[i];
        }

        // Apply emotional modulation if present
        if let Some(ref emotion) = self.emotional_state {
            self.apply_emotional_modulation(&mut latent, emotion);
        }

        // Normalize
        let norm: f64 = latent.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for val in latent.iter_mut() {
                *val /= norm;
            }
        }

        Ok(latent)
    }

    /// Compute latent from text concept with knowledge anchoring
    pub fn compute_latent_from_concept(
        &self,
        concept: &str,
        memory: &SemanticMemory,
        bias: &BiasVector,
    ) -> Result<LatentResult, Box<dyn std::error::Error>> {
        // 1. Convert concept to vector
        let concept_thought = ThoughtState::from_input(concept, self.dimension);
        let concept_vector = &concept_thought.vector;

        // 2. Search knowledge base for related concepts
        let knowledge_facts = memory.find_similar(concept_vector, 5)?;

        // 3. Compute knowledge vector (weighted average of similar facts)
        let knowledge_vector = if knowledge_facts.is_empty() {
            vec![0.0; self.dimension]
        } else {
            self.compute_knowledge_vector(&knowledge_facts)
        };

        // 4. Compute creativity vector from bias
        let creativity_vector = self.bias_to_creativity_vector(bias);

        // 5. Determine alpha from bias creativity parameter
        let alpha = bias.creativity;

        // 6. Compute final latent
        let latent = self.compute_latent(
            concept_vector,
            &knowledge_vector,
            &creativity_vector,
            alpha,
        )?;

        // 7. Verify latent against knowledge (if not purely creative)
        let verification = if alpha < 0.8 {
            self.verify_latent(&latent, memory)?
        } else {
            LatentVerification {
                verified: true,
                confidence: 1.0,
                max_similarity: 1.0,
                supporting_facts: Vec::new(),
                reason: "Creative mode - verification skipped".to_string(),
            }
        };

        Ok(LatentResult {
            latent,
            concept_vector: concept_vector.clone(),
            knowledge_vector,
            creativity_vector,
            alpha,
            verification,
            knowledge_facts_used: knowledge_facts.len(),
        })
    }

    /// Compute knowledge vector from retrieved facts
    fn compute_knowledge_vector(
        &self,
        facts: &[(crate::memory::SemanticFact, f64)],
    ) -> Vec<f64> {
        let mut knowledge = vec![0.0; self.dimension];
        let mut total_weight = 0.0;

        // Weighted average based on similarity scores
        for (fact, similarity) in facts.iter() {
            let weight = similarity * fact.confidence;
            total_weight += weight;

            for (i, &val) in fact.embedding.iter().enumerate() {
                if i < self.dimension {
                    knowledge[i] += val * weight;
                }
            }
        }

        // Normalize by total weight
        if total_weight > 1e-10 {
            for val in knowledge.iter_mut() {
                *val /= total_weight;
            }
        }

        knowledge
    }

    /// Convert bias vector to creativity direction
    fn bias_to_creativity_vector(&self, bias: &BiasVector) -> Vec<f64> {
        let mut creativity = vec![0.0; self.dimension];

        // Use bias parameters to create creativity direction
        let params = vec![
            bias.creativity,
            bias.exploration,
            bias.risk_tolerance,
            bias.urgency,
        ];

        // Distribute bias parameters across dimension
        let region_size = self.dimension / params.len();

        for (param_idx, &param_val) in params.iter().enumerate() {
            let start = param_idx * region_size;
            let end = ((param_idx + 1) * region_size).min(self.dimension);

            for i in start..end {
                // Create deterministic variation based on parameter
                let phase = (i as f64 * 0.1 + param_val * std::f64::consts::PI).sin();
                creativity[i] = phase * param_val;
            }
        }

        creativity
    }

    /// Apply emotional modulation to latent vector
    fn apply_emotional_modulation(&self, latent: &mut [f64], emotion: &EmotionalState) {
        let third = self.dimension / 3;

        // Modulate different regions based on emotional dimensions
        for (i, val) in latent.iter_mut().enumerate() {
            let modulation = if i < third {
                // Valence region
                emotion.valence - 0.5
            } else if i < 2 * third {
                // Arousal region
                emotion.arousal - 0.5
            } else {
                // Dominance region
                emotion.dominance - 0.5
            };

            *val *= 1.0 + modulation * 0.2;
        }
    }

    /// Verify latent vector against knowledge base
    pub fn verify_latent(
        &self,
        latent: &[f64],
        memory: &SemanticMemory,
    ) -> Result<LatentVerification, Box<dyn std::error::Error>> {
        let similar_facts = memory.find_similar(latent, 5)?;

        if similar_facts.is_empty() {
            return Ok(LatentVerification {
                verified: false,
                confidence: 0.0,
                max_similarity: 0.0,
                supporting_facts: Vec::new(),
                reason: "No knowledge found to verify latent".to_string(),
            });
        }

        let (best_fact, max_similarity) = &similar_facts[0];

        let verified = *max_similarity >= self.thresholds.min_knowledge_similarity
            && best_fact.confidence >= self.thresholds.min_token_confidence;

        let supporting_facts: Vec<String> = similar_facts.iter()
            .filter(|(_, sim)| *sim >= self.thresholds.min_knowledge_similarity)
            .map(|(fact, _)| fact.content.clone())
            .collect();

        Ok(LatentVerification {
            verified,
            confidence: best_fact.confidence,
            max_similarity: *max_similarity,
            supporting_facts,
            reason: if verified {
                format!("Latent verified: sim={:.3}, conf={:.3}", max_similarity, best_fact.confidence)
            } else {
                format!("Insufficient similarity: {:.3} < {:.3}", max_similarity, self.thresholds.min_knowledge_similarity)
            },
        })
    }

    /// Propagate latent for temporal consistency (video generation)
    ///
    /// h_{t+1} = h_t + Δh_temporal
    ///
    /// Maintains smooth transitions while allowing controlled evolution
    pub fn propagate_latent(
        &self,
        previous_latent: &[f64],
        temporal_delta: &[f64],
        propagation_strength: f64,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        if previous_latent.len() != self.dimension || temporal_delta.len() != self.dimension {
            return Err("Vector dimension mismatch".into());
        }

        let mut next_latent = vec![0.0; self.dimension];

        for i in 0..self.dimension {
            next_latent[i] = previous_latent[i] + propagation_strength * temporal_delta[i];
        }

        // Normalize
        let norm: f64 = next_latent.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for val in next_latent.iter_mut() {
                *val /= norm;
            }
        }

        Ok(next_latent)
    }

    /// Generate sequence of latents for video with temporal consistency
    pub fn generate_latent_sequence(
        &self,
        initial_concept: &str,
        memory: &SemanticMemory,
        bias: &BiasVector,
        num_frames: usize,
        propagation_strength: f64,
    ) -> Result<Vec<LatentResult>, Box<dyn std::error::Error>> {
        let mut sequence = Vec::new();

        // Generate initial latent
        let initial = self.compute_latent_from_concept(initial_concept, memory, bias)?;
        sequence.push(initial);

        // Propagate for subsequent frames
        for frame_idx in 1..num_frames {
            let previous_latent = &sequence[frame_idx - 1].latent;

            // Compute temporal delta (smooth evolution)
            let temporal_delta = self.compute_temporal_delta(frame_idx, bias);

            // Propagate latent
            let next_latent = self.propagate_latent(
                previous_latent,
                &temporal_delta,
                propagation_strength,
            )?;

            // Verify next latent
            let verification = if bias.creativity < 0.8 {
                self.verify_latent(&next_latent, memory)?
            } else {
                LatentVerification {
                    verified: true,
                    confidence: 1.0,
                    max_similarity: 1.0,
                    supporting_facts: Vec::new(),
                    reason: "Creative mode - verification skipped".to_string(),
                }
            };

            sequence.push(LatentResult {
                latent: next_latent,
                concept_vector: sequence[0].concept_vector.clone(),
                knowledge_vector: sequence[0].knowledge_vector.clone(),
                creativity_vector: sequence[0].creativity_vector.clone(),
                alpha: bias.creativity,
                verification,
                knowledge_facts_used: 0,
            });
        }

        Ok(sequence)
    }

    /// Compute temporal delta for smooth evolution
    fn compute_temporal_delta(&self, frame_idx: usize, bias: &BiasVector) -> Vec<f64> {
        let mut delta = vec![0.0; self.dimension];

        // Smooth sinusoidal evolution modulated by exploration bias
        let evolution_rate = bias.exploration * 0.05;

        for i in 0..self.dimension {
            let phase = (frame_idx as f64 * evolution_rate + i as f64 * 0.1).sin();
            delta[i] = phase * 0.02;
        }

        delta
    }

    /// Perform multi-step reasoning with knowledge anchoring at each step
    ///
    /// Combines chain-of-thought reasoning with knowledge verification.
    /// Each reasoning step generates a latent that is verified against knowledge.
    ///
    /// Returns: ReasoningChain with knowledge-verified steps
    pub fn reason_multi_step(
        &self,
        problem: &str,
        memory: &SemanticMemory,
        bias: &BiasVector,
    ) -> Result<MultiStepReasoning, Box<dyn std::error::Error>> {
        // 1. Decompose problem using chain-of-thought
        let chain = self.chain_reasoner.reason(problem);

        // 2. For each step, compute knowledge-anchored latent
        let mut verified_steps = Vec::new();
        let mut all_verified = true;

        for step in &chain.steps {
            // Compute latent for this reasoning step
            let step_latent = self.compute_latent_from_concept(
                &step.description,
                memory,
                bias,
            )?;

            // Create verified step with both reasoning and latent
            let verified_step = VerifiedReasoningStep {
                step: step.step,
                description: step.description.clone(),
                operator: step.operator.clone(),
                latent_result: step_latent.clone(),
                original_confidence: step.confidence,
            };

            if !step_latent.verification.verified {
                all_verified = false;
            }

            verified_steps.push(verified_step);
        }

        // 3. Compute overall confidence from all steps
        let avg_confidence = if !verified_steps.is_empty() {
            verified_steps.iter()
                .map(|s| s.latent_result.verification.confidence)
                .sum::<f64>() / verified_steps.len() as f64
        } else {
            0.0
        };

        Ok(MultiStepReasoning {
            problem: problem.to_string(),
            chain,
            verified_steps,
            all_steps_verified: all_verified,
            overall_confidence: avg_confidence,
        })
    }

    /// Generate latent from multi-step reasoning
    ///
    /// Combines all reasoning steps into a final latent vector
    pub fn latent_from_multi_step_reasoning(
        &self,
        multi_step: &MultiStepReasoning,
        bias: &BiasVector,
    ) -> Result<LatentResult, Box<dyn std::error::Error>> {
        if multi_step.verified_steps.is_empty() {
            return Err("No reasoning steps to combine".into());
        }

        // Combine all step latents with weighted averaging
        let mut combined_latent = vec![0.0; self.dimension];
        let mut total_weight = 0.0;

        for step in &multi_step.verified_steps {
            let weight = step.latent_result.verification.confidence;
            total_weight += weight;

            for (i, &val) in step.latent_result.latent.iter().enumerate() {
                combined_latent[i] += val * weight;
            }
        }

        // Normalize by total weight
        if total_weight > 1e-10 {
            for val in combined_latent.iter_mut() {
                *val /= total_weight;
            }
        }

        // Normalize vector
        let norm: f64 = combined_latent.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for val in combined_latent.iter_mut() {
                *val /= norm;
            }
        }

        // Get supporting facts from all steps
        let mut all_supporting_facts: Vec<String> = multi_step.verified_steps.iter()
            .flat_map(|s| s.latent_result.verification.supporting_facts.clone())
            .collect();
        all_supporting_facts.dedup();

        Ok(LatentResult {
            latent: combined_latent,
            concept_vector: multi_step.verified_steps[0].latent_result.concept_vector.clone(),
            knowledge_vector: multi_step.verified_steps[0].latent_result.knowledge_vector.clone(),
            creativity_vector: multi_step.verified_steps[0].latent_result.creativity_vector.clone(),
            alpha: bias.creativity,
            verification: LatentVerification {
                verified: multi_step.all_steps_verified,
                confidence: multi_step.overall_confidence,
                max_similarity: multi_step.verified_steps.iter()
                    .map(|s| s.latent_result.verification.max_similarity)
                    .fold(0.0, f64::max),
                supporting_facts: all_supporting_facts,
                reason: format!(
                    "Multi-step reasoning: {} steps, {} verified",
                    multi_step.verified_steps.len(),
                    multi_step.verified_steps.iter().filter(|s| s.latent_result.verification.verified).count()
                ),
            },
            knowledge_facts_used: multi_step.verified_steps.iter()
                .map(|s| s.latent_result.knowledge_facts_used)
                .sum(),
        })
    }
}

/// Result of latent computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentResult {
    /// Final latent vector for generation
    pub latent: Vec<f64>,
    /// Concept component
    pub concept_vector: Vec<f64>,
    /// Knowledge component
    pub knowledge_vector: Vec<f64>,
    /// Creativity component
    pub creativity_vector: Vec<f64>,
    /// Creativity injection weight used
    pub alpha: f64,
    /// Verification result
    pub verification: LatentVerification,
    /// Number of knowledge facts used
    pub knowledge_facts_used: usize,
}

/// Verification result for latent vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentVerification {
    /// Whether latent is verified against knowledge
    pub verified: bool,
    /// Confidence in verification
    pub confidence: f64,
    /// Maximum similarity to knowledge base
    pub max_similarity: f64,
    /// Supporting facts from knowledge
    pub supporting_facts: Vec<String>,
    /// Reason for verification result
    pub reason: String,
}

/// Single step in multi-step reasoning with knowledge verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedReasoningStep {
    /// Step number
    pub step: usize,
    /// Description of this step
    pub description: String,
    /// Operator used in reasoning
    pub operator: String,
    /// Knowledge-anchored latent for this step
    pub latent_result: LatentResult,
    /// Original confidence from chain-of-thought
    pub original_confidence: f64,
}

/// Result of multi-step reasoning with knowledge anchoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiStepReasoning {
    /// Problem being solved
    pub problem: String,
    /// Original reasoning chain
    pub chain: ReasoningChain,
    /// Verified steps with knowledge-anchored latents
    pub verified_steps: Vec<VerifiedReasoningStep>,
    /// Whether all steps passed verification
    pub all_steps_verified: bool,
    /// Overall confidence across all steps
    pub overall_confidence: f64,
}

impl MultiStepReasoning {
    /// Get summary of multi-step reasoning
    pub fn summary(&self) -> String {
        let mut summary = format!("Problem: {}\n", self.problem);
        summary.push_str(&format!("Total Steps: {}\n", self.verified_steps.len()));
        summary.push_str(&format!("All Verified: {}\n", self.all_steps_verified));
        summary.push_str(&format!("Overall Confidence: {:.3}\n\n", self.overall_confidence));

        for step in &self.verified_steps {
            summary.push_str(&format!("Step {}: {}\n", step.step, step.description));
            summary.push_str(&format!("  Operator: {}\n", step.operator));
            summary.push_str(&format!("  Verified: {}\n", step.latent_result.verification.verified));
            summary.push_str(&format!("  Confidence: {:.3}\n", step.latent_result.verification.confidence));
            summary.push_str(&format!("  Knowledge facts: {}\n", step.latent_result.knowledge_facts_used));

            if !step.latent_result.verification.supporting_facts.is_empty() {
                summary.push_str("  Supporting knowledge:\n");
                for fact in step.latent_result.verification.supporting_facts.iter().take(2) {
                    summary.push_str(&format!("    • {}\n", fact));
                }
            }
            summary.push('\n');
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_engine_creation() {
        let engine = ReasoningEngine::balanced(128);
        assert_eq!(engine.dimension, 128);
    }

    #[test]
    fn test_compute_latent() {
        let engine = ReasoningEngine::balanced(64);

        let concept = vec![1.0; 64];
        let knowledge = vec![0.5; 64];
        let creativity = vec![0.2; 64];
        let alpha = 0.5;

        let latent = engine.compute_latent(&concept, &knowledge, &creativity, alpha).unwrap();

        assert_eq!(latent.len(), 64);

        // Should be normalized
        let norm: f64 = latent.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_latent_propagation() {
        let engine = ReasoningEngine::balanced(64);

        let previous = vec![1.0; 64];
        let delta = vec![0.1; 64];
        let strength = 0.5;

        let next = engine.propagate_latent(&previous, &delta, strength).unwrap();

        assert_eq!(next.len(), 64);
    }

    #[test]
    fn test_bias_to_creativity() {
        let engine = ReasoningEngine::balanced(128);

        let bias = BiasVector {
            creativity: 0.8,
            exploration: 0.6,
            risk_tolerance: 0.5,
            urgency: 0.3,
        };

        let creativity = engine.bias_to_creativity_vector(&bias);
        assert_eq!(creativity.len(), 128);
    }
}
