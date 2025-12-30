//! Uncertainty Handler
//!
//! Handles cases where the AI doesn't know the answer.
//! Generates honest "I don't know" responses with reasoning.

use serde::{Deserialize, Serialize};
use crate::core::ThoughtState;
use crate::memory::input_embeddings::EnhancedEpisode;

/// Uncertainty assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyAssessment {
    /// Is the AI uncertain about this?
    pub is_uncertain: bool,
    /// Confidence level (0.0 = completely uncertain, 1.0 = completely certain)
    pub confidence: f64,
    /// Reasons for uncertainty
    pub uncertainty_reasons: Vec<String>,
    /// Should refuse to answer?
    pub should_refuse: bool,
    /// Suggested response if refusing
    pub refusal_response: Option<String>,
}

/// Uncertainty handler
pub struct UncertaintyHandler {
    /// Minimum confidence threshold for answering
    pub min_confidence: f64,
    /// Minimum episodes needed for confidence
    pub min_episodes: usize,
}

impl UncertaintyHandler {
    pub fn new(min_confidence: f64, min_episodes: usize) -> Self {
        Self {
            min_confidence,
            min_episodes,
        }
    }

    /// Assess uncertainty for a given query and context
    pub fn assess_uncertainty(
        &self,
        query: &str,
        thought: &ThoughtState,
        confidence: f64,
        similar_episodes: &[EnhancedEpisode],
    ) -> UncertaintyAssessment {
        let mut reasons = Vec::new();
        let mut is_uncertain = false;

        // Check 1: Low confidence from neural network
        if confidence < self.min_confidence {
            reasons.push(format!(
                "Neural network confidence is low ({:.1}% < {:.1}% threshold)",
                confidence * 100.0,
                self.min_confidence * 100.0
            ));
            is_uncertain = true;
        }

        // Check 2: No similar training examples
        if similar_episodes.is_empty() {
            reasons.push("No similar examples found in training data".to_string());
            is_uncertain = true;
        } else if similar_episodes.len() < self.min_episodes {
            reasons.push(format!(
                "Only {} similar examples found (need at least {})",
                similar_episodes.len(),
                self.min_episodes
            ));
            is_uncertain = true;
        }

        // Check 3: High entropy in thought vector (confused/uncertain state)
        let entropy = self.calculate_entropy(thought);
        if entropy > 0.8 {
            reasons.push(format!(
                "High uncertainty in reasoning process (entropy: {:.2})",
                entropy
            ));
            is_uncertain = true;
        }

        // Check 4: Low similarity to training examples
        if !similar_episodes.is_empty() {
            let max_similarity = similar_episodes.iter()
                .map(|ep| self.calculate_similarity(&thought.vector, &ep.thought_vector))
                .fold(0.0f64, f64::max);

            if max_similarity < 0.5 {
                reasons.push(format!(
                    "Query is quite different from training examples (max similarity: {:.1}%)",
                    max_similarity * 100.0
                ));
                is_uncertain = true;
            }
        }

        // Decide if we should refuse to answer
        let should_refuse = is_uncertain && (
            confidence < self.min_confidence * 0.8 ||
            similar_episodes.is_empty() ||
            entropy > 0.9
        );

        // Generate refusal response if needed
        let refusal_response = if should_refuse {
            Some(self.generate_honest_refusal(query, &reasons, confidence))
        } else {
            None
        };

        UncertaintyAssessment {
            is_uncertain,
            confidence,
            uncertainty_reasons: reasons,
            should_refuse,
            refusal_response,
        }
    }

    /// Generate an honest "I don't know" response with reasoning
    fn generate_honest_refusal(
        &self,
        query: &str,
        reasons: &[String],
        confidence: f64,
    ) -> String {
        // Start with honest admission
        let mut response = format!(
            "I don't have enough confidence to answer that question (confidence: {:.1}%). ",
            confidence * 100.0
        );

        // Explain why
        if !reasons.is_empty() {
            response.push_str("Here's why:\n");
            for (i, reason) in reasons.iter().enumerate() {
                response.push_str(&format!("{}. {}\n", i + 1, reason));
            }
        }

        // Offer to learn
        response.push_str("\nI'd be happy to learn about this topic if you can provide some training examples. ");
        response.push_str("You can teach me by providing input-output pairs related to this question.");

        response
    }

    /// Calculate entropy of thought vector (measure of uncertainty)
    fn calculate_entropy(&self, thought: &ThoughtState) -> f64 {
        let mut entropy = 0.0;
        let n = thought.vector.len() as f64;

        for &val in &thought.vector {
            let abs_val = val.abs();
            if abs_val > 1e-10 {
                entropy -= abs_val * abs_val.ln();
            }
        }

        // Normalize to [0, 1]
        let max_entropy = n.ln();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Calculate cosine similarity between two vectors
    fn calculate_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            (dot / (norm_a * norm_b)).max(-1.0).min(1.0)
        } else {
            0.0
        }
    }

    /// Generate a response that acknowledges uncertainty but still tries to help
    pub fn generate_uncertain_response(
        &self,
        query: &str,
        partial_answer: &str,
        confidence: f64,
    ) -> String {
        format!(
            "I'm not entirely certain about this (confidence: {:.1}%), but based on what I know: {}\n\n\
            Please note that this answer may not be completely accurate. If you have more information or can correct me, I'd appreciate learning from you.",
            confidence * 100.0,
            partial_answer
        )
    }
}

impl Default for UncertaintyHandler {
    fn default() -> Self {
        Self::new(0.6, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertainty_assessment() {
        let handler = UncertaintyHandler::default();
        let thought = ThoughtState::random(64);
        let episodes = Vec::new();

        let assessment = handler.assess_uncertainty(
            "What is quantum chromodynamics?",
            &thought,
            0.3,
            &episodes,
        );

        assert!(assessment.is_uncertain);
        assert!(assessment.should_refuse);
        assert!(assessment.refusal_response.is_some());
        assert!(!assessment.uncertainty_reasons.is_empty());
    }

    #[test]
    fn test_high_confidence_no_refusal() {
        let handler = UncertaintyHandler::new(0.7, 2);
        // Create a low-entropy thought vector (confident state)
        // Use a peaked distribution (one dominant component)
        let mut thought = ThoughtState::new(64);
        thought.vector[0] = 1.0; // Peaked at first dimension
        for i in 1..64 {
            thought.vector[i] = 0.01; // Small values elsewhere
        }
        
        // Create mock episodes with similar thought vectors
        let episodes = vec![
            EnhancedEpisode::new(
                "test".to_string(),
                "answer".to_string(),
                vec![0.0; 64],
                thought.vector.clone(), // Use similar thought vector
                true,
                0.9,
                0.1,
                "test".to_string(),
            ),
            EnhancedEpisode::new(
                "test2".to_string(),
                "answer2".to_string(),
                vec![0.0; 64],
                thought.vector.clone(), // Use similar thought vector
                true,
                0.9,
                0.1,
                "test".to_string(),
            ),
        ];

        let assessment = handler.assess_uncertainty(
            "What is 2+2?",
            &thought,
            0.95,
            &episodes,
        );
        
        // High confidence with episodes should not refuse
        assert!(!assessment.should_refuse, "Should not refuse with high confidence and episodes. Reasons: {:?}", assessment.uncertainty_reasons);
    }
}
