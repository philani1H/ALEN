//! Episodic Memory Integration for Confidence
//!
//! Problem: Confidence decoder operates in isolation
//! - Ignores historical success patterns
//! - Doesn't learn from past episodes
//!
//! Solution: Integrate episodic memory signals into confidence
//!
//! Mathematical Foundation:
//! C_final = α · C_proof + β · ΔC_episodic + γ · C_concept
//!
//! Where:
//! - C_proof = current verification confidence
//! - ΔC_episodic = (1/k) Σ success_i · sim(e_x, e_i)
//! - C_concept = compressed rule confidence
//! - α + β + γ = 1 (normalized weights)

use crate::memory::input_embeddings::{InputEmbedder, EnhancedEpisode};

// ============================================================================
// PART 1: EPISODIC CONFIDENCE BOOST
// ============================================================================

/// Computes confidence boost from episodic memory
pub struct EpisodicConfidenceBooster {
    embedder: InputEmbedder,
    
    /// Weight for episodic signal (β)
    episodic_weight: f64,
    
    /// Minimum similarity to consider
    min_similarity: f64,
    
    /// Maximum number of episodes to consider
    top_k: usize,
}

impl EpisodicConfidenceBooster {
    pub fn new() -> Self {
        Self {
            embedder: InputEmbedder::new(128), // Default dimension
            episodic_weight: 0.3, // β = 0.3
            min_similarity: 0.1,
            top_k: 5,
        }
    }

    /// Compute episodic confidence boost
    /// ΔC_episodic = (1/k) Σ success_i · sim(e_x, e_i)
    pub fn compute_boost(&self, query: &str, episodes: &[EnhancedEpisode]) -> f64 {
        if episodes.is_empty() {
            return 0.0;
        }

        // Embed query in input space
        let query_embedding = self.embedder.embed(query);

        // Find similar episodes
        let mut similarities: Vec<(f64, &EnhancedEpisode)> = episodes
            .iter()
            .map(|ep| {
                let sim = self.embedder.similarity(&query_embedding, &ep.input_embedding);
                (sim, ep)
            })
            .filter(|(sim, _)| *sim >= self.min_similarity)
            .collect();

        if similarities.is_empty() {
            return 0.0;
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take top-k
        let top_episodes: Vec<_> = similarities.into_iter().take(self.top_k).collect();

        // Compute weighted average: (1/k) Σ success_i · sim(e_x, e_i)
        // Use verified status and confidence as success indicator
        let boost: f64 = top_episodes
            .iter()
            .map(|(sim, ep)| {
                let success = if ep.verified { ep.confidence_score } else { 0.0 };
                success * sim
            })
            .sum::<f64>() / top_episodes.len() as f64;

        boost
    }

    /// Get weighted boost (β · ΔC_episodic)
    pub fn get_weighted_boost(&self, query: &str, episodes: &[EnhancedEpisode]) -> f64 {
        let boost = self.compute_boost(query, episodes);
        self.episodic_weight * boost
    }
}

impl Default for EpisodicConfidenceBooster {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PART 2: CONCEPT CONFIDENCE
// ============================================================================

/// Represents confidence from compressed concepts/rules
#[derive(Debug, Clone)]
pub struct ConceptConfidence {
    /// Rule or concept that applies
    pub rule: String,
    
    /// Confidence in this rule (from compression)
    pub confidence: f64,
    
    /// Number of times this rule was successful
    pub success_count: usize,
    
    /// Total times this rule was applied
    pub total_count: usize,
}

impl ConceptConfidence {
    pub fn new(rule: String) -> Self {
        Self {
            rule,
            confidence: 0.5,
            success_count: 0,
            total_count: 0,
        }
    }

    /// Update confidence based on outcome
    pub fn update(&mut self, success: bool) {
        self.total_count += 1;
        if success {
            self.success_count += 1;
        }

        // Bayesian update
        self.confidence = self.success_count as f64 / self.total_count as f64;
    }

    /// Get current confidence
    pub fn get_confidence(&self) -> f64 {
        if self.total_count == 0 {
            0.5 // Prior
        } else {
            self.confidence
        }
    }
}

// ============================================================================
// PART 3: INTEGRATED CONFIDENCE CALCULATOR
// ============================================================================

/// Integrates multiple confidence signals
pub struct IntegratedConfidenceCalculator {
    /// Weight for proof confidence (α)
    proof_weight: f64,
    
    /// Weight for episodic confidence (β)
    episodic_weight: f64,
    
    /// Weight for concept confidence (γ)
    concept_weight: f64,
    
    /// Episodic booster
    episodic_booster: EpisodicConfidenceBooster,
}

impl IntegratedConfidenceCalculator {
    pub fn new() -> Self {
        Self {
            proof_weight: 0.5,      // α = 0.5
            episodic_weight: 0.3,   // β = 0.3
            concept_weight: 0.2,    // γ = 0.2
            episodic_booster: EpisodicConfidenceBooster::new(),
        }
    }

    /// Compute integrated confidence
    /// C_final = α · C_proof + β · ΔC_episodic + γ · C_concept
    pub fn compute_confidence(
        &self,
        proof_confidence: f64,
        query: &str,
        episodes: &[EnhancedEpisode],
        concept_confidence: Option<f64>,
    ) -> IntegratedConfidence {
        // Compute episodic boost
        let episodic_boost = self.episodic_booster.compute_boost(query, episodes);

        // Get concept confidence (default to 0.5 if not provided)
        let concept_conf = concept_confidence.unwrap_or(0.5);

        // Compute weighted sum
        let final_confidence = 
            self.proof_weight * proof_confidence +
            self.episodic_weight * episodic_boost +
            self.concept_weight * concept_conf;

        IntegratedConfidence {
            final_confidence,
            proof_confidence,
            episodic_boost,
            concept_confidence: concept_conf,
            weights: ConfidenceWeights {
                proof: self.proof_weight,
                episodic: self.episodic_weight,
                concept: self.concept_weight,
            },
        }
    }

    /// Set custom weights (must sum to 1.0)
    pub fn set_weights(&mut self, proof: f64, episodic: f64, concept: f64) {
        let sum = proof + episodic + concept;
        assert!((sum - 1.0).abs() < 0.001, "Weights must sum to 1.0");
        
        self.proof_weight = proof;
        self.episodic_weight = episodic;
        self.concept_weight = concept;
    }
}

impl Default for IntegratedConfidenceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of integrated confidence calculation
#[derive(Debug, Clone)]
pub struct IntegratedConfidence {
    /// Final integrated confidence
    pub final_confidence: f64,
    
    /// Proof verification confidence
    pub proof_confidence: f64,
    
    /// Episodic memory boost
    pub episodic_boost: f64,
    
    /// Concept/rule confidence
    pub concept_confidence: f64,
    
    /// Weights used
    pub weights: ConfidenceWeights,
}

impl IntegratedConfidence {
    /// Get breakdown as string for debugging
    pub fn breakdown(&self) -> String {
        format!(
            "Final: {:.3} = {:.3}×{:.3} (proof) + {:.3}×{:.3} (episodic) + {:.3}×{:.3} (concept)",
            self.final_confidence,
            self.weights.proof, self.proof_confidence,
            self.weights.episodic, self.episodic_boost,
            self.weights.concept, self.concept_confidence
        )
    }
}

#[derive(Debug, Clone)]
pub struct ConfidenceWeights {
    pub proof: f64,
    pub episodic: f64,
    pub concept: f64,
}

// ============================================================================
// PART 4: CONFIDENCE-AWARE RESPONSE GENERATOR
// ============================================================================

/// Generates responses based on integrated confidence
pub struct ConfidenceAwareResponder {
    calculator: IntegratedConfidenceCalculator,
}

impl ConfidenceAwareResponder {
    pub fn new() -> Self {
        Self {
            calculator: IntegratedConfidenceCalculator::new(),
        }
    }

    /// Generate response with confidence gating
    pub fn generate_response(
        &self,
        answer: String,
        proof_confidence: f64,
        query: &str,
        episodes: &[EnhancedEpisode],
        concept_confidence: Option<f64>,
        threshold: f64,
    ) -> ConfidenceGatedResponse {
        // Compute integrated confidence
        let integrated = self.calculator.compute_confidence(
            proof_confidence,
            query,
            episodes,
            concept_confidence,
        );

        // Check threshold
        let should_answer = integrated.final_confidence >= threshold;

        let final_confidence = integrated.final_confidence;
        
        ConfidenceGatedResponse {
            answer: if should_answer {
                Some(answer)
            } else {
                None
            },
            confidence: final_confidence,
            confidence_breakdown: integrated,
            threshold,
            refused: !should_answer,
            refusal_reason: if !should_answer {
                Some(format!(
                    "Confidence {:.3} below threshold {:.3}",
                    final_confidence, threshold
                ))
            } else {
                None
            },
        }
    }
}

impl Default for ConfidenceAwareResponder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ConfidenceGatedResponse {
    /// Answer (None if refused)
    pub answer: Option<String>,
    
    /// Final confidence
    pub confidence: f64,
    
    /// Confidence breakdown
    pub confidence_breakdown: IntegratedConfidence,
    
    /// Threshold used
    pub threshold: f64,
    
    /// Whether response was refused
    pub refused: bool,
    
    /// Reason for refusal (if refused)
    pub refusal_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_episode(input: &str, verified: bool) -> EnhancedEpisode {
        let embedder = InputEmbedder::new(128);
        EnhancedEpisode::new(
            input.to_string(),
            "test output".to_string(),
            embedder.embed(input),
            vec![0.0; 128], // Dummy thought vector
            verified,
            0.8,
            0.5,
            "test_op".to_string(),
        )
    }

    #[test]
    fn test_episodic_boost() {
        let booster = EpisodicConfidenceBooster::new();

        let episodes = vec![
            create_test_episode("What is 2+2?", true),
            create_test_episode("Calculate 3+3", true),
            create_test_episode("Solve 5+5", false),
        ];

        let boost = booster.compute_boost("What is 4+4?", &episodes);
        
        // Should get positive boost from similar successful episodes
        assert!(boost > 0.0);
        assert!(boost <= 1.0);
    }

    #[test]
    fn test_integrated_confidence() {
        let calculator = IntegratedConfidenceCalculator::new();

        let episodes = vec![
            create_test_episode("What is 2+2?", true),
            create_test_episode("Calculate 3+3", true),
        ];

        let integrated = calculator.compute_confidence(
            0.8,  // proof confidence
            "What is 4+4?",
            &episodes,
            Some(0.9), // concept confidence
        );

        // Final confidence should be weighted combination
        assert!(integrated.final_confidence > 0.0);
        assert!(integrated.final_confidence <= 1.0);
        
        // Should be influenced by all three components
        assert!(integrated.proof_confidence == 0.8);
        assert!(integrated.episodic_boost > 0.0);
        assert!(integrated.concept_confidence == 0.9);
    }

    #[test]
    fn test_confidence_gating() {
        let responder = ConfidenceAwareResponder::new();

        let episodes = vec![
            create_test_episode("What is 2+2?", true),
        ];

        // High confidence - should answer
        let response = responder.generate_response(
            "The answer is 4".to_string(),
            0.9,
            "What is 2+2?",
            &episodes,
            Some(0.9),
            0.7, // threshold
        );

        assert!(!response.refused);
        assert!(response.answer.is_some());

        // Low confidence - should refuse
        let response = responder.generate_response(
            "The answer is 4".to_string(),
            0.3,
            "What is 2+2?",
            &episodes,
            Some(0.3),
            0.7, // threshold
        );

        assert!(response.refused);
        assert!(response.answer.is_none());
    }

    #[test]
    fn test_concept_confidence_update() {
        let mut concept = ConceptConfidence::new("addition_rule".to_string());

        // Initially 0.5 (prior)
        assert_eq!(concept.get_confidence(), 0.5);

        // Update with successes
        concept.update(true);
        concept.update(true);
        concept.update(false);

        // Should be 2/3 ≈ 0.667
        assert!((concept.get_confidence() - 0.667).abs() < 0.01);
    }
}
