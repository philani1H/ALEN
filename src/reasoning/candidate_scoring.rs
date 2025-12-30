//! Candidate Scoring System - Neural Network + Memory + Confidence + Style
//!
//! Implements: S_i = P_θ(Y_i) · P_memory(Y_i) · C(Y_i) · V(Y_i)
//!
//! This ensures:
//! 1. Neural network is ALWAYS active
//! 2. Memory provides soft guidance (not retrieval)
//! 3. Confidence verification for multi-step reasoning
//! 4. Style/novelty factor for personalization

use crate::core::{ThoughtState, EnergyResult};
use crate::memory::{SemanticMemory, Episode};
use serde::{Deserialize, Serialize};

/// Candidate response with full scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredCandidate {
    /// The candidate answer
    pub answer: String,
    /// Thought vector that generated this
    pub thought: ThoughtState,
    /// Neural network probability P_θ(Y_i | h_X, u, c)
    pub neural_probability: f64,
    /// Memory guidance P_memory(Y_i) - soft influence
    pub memory_guidance: f64,
    /// Confidence verification C(Y_i)
    pub confidence: f64,
    /// Style/novelty factor V(Y_i)
    pub style_factor: f64,
    /// Final combined score
    pub final_score: f64,
    /// Explanation of reasoning
    pub explanation: Vec<String>,
}

/// User embedding for personalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserEmbedding {
    /// Verbosity preference (0.0 = concise, 1.0 = detailed)
    pub verbosity: f64,
    /// Technical level (0.0 = simple, 1.0 = expert)
    pub technical_level: f64,
    /// Creativity preference (0.0 = factual, 1.0 = creative)
    pub creativity: f64,
    /// Formality (0.0 = casual, 1.0 = formal)
    pub formality: f64,
}

impl Default for UserEmbedding {
    fn default() -> Self {
        Self {
            verbosity: 0.5,
            technical_level: 0.5,
            creativity: 0.5,
            formality: 0.5,
        }
    }
}

/// Candidate scoring system
pub struct CandidateScorer {
    /// Dimension
    dimension: usize,
    /// Confidence threshold for flagging
    confidence_threshold: f64,
    /// Memory influence weight (0.0 = ignore, 1.0 = strong)
    memory_weight: f64,
}

impl CandidateScorer {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            confidence_threshold: 0.5,
            memory_weight: 0.3, // Soft influence, not hard retrieval
        }
    }

    /// Score a candidate using full formula:
    /// S_i = P_θ(Y_i) · P_memory(Y_i) · C(Y_i) · V(Y_i)
    pub fn score_candidate(
        &self,
        thought: &ThoughtState,
        answer: &str,
        energy: &EnergyResult,
        memory: &SemanticMemory,
        user: &UserEmbedding,
        context: &[String],
    ) -> ScoredCandidate {
        // 1. Neural probability P_θ(Y_i | h_X, u, c)
        let neural_probability = self.compute_neural_probability(thought, energy);
        
        // 2. Memory guidance P_memory(Y_i) - SOFT INFLUENCE
        let memory_guidance = self.compute_memory_guidance(thought, answer, memory);
        
        // 3. Confidence verification C(Y_i)
        let confidence = energy.confidence_score;
        
        // 4. Style/novelty factor V(Y_i)
        let style_factor = self.compute_style_factor(answer, user);
        
        // 5. Final score: S_i = P_θ · P_memory · C · V
        let final_score = neural_probability 
            * (1.0 - self.memory_weight + self.memory_weight * memory_guidance)
            * confidence 
            * style_factor;
        
        // 6. Generate explanation
        let explanation = self.generate_explanation(
            neural_probability,
            memory_guidance,
            confidence,
            style_factor,
            context,
        );
        
        ScoredCandidate {
            answer: answer.to_string(),
            thought: thought.clone(),
            neural_probability,
            memory_guidance,
            confidence,
            style_factor,
            final_score,
            explanation,
        }
    }

    /// Compute neural network probability
    /// P_θ(Y_i | h_X, u, c) based on energy and thought quality
    fn compute_neural_probability(&self, thought: &ThoughtState, energy: &EnergyResult) -> f64 {
        // Lower energy = higher probability
        // Use softmax-like transformation
        let energy_score = (-energy.total).exp();
        
        // Normalize by thought norm (quality indicator)
        let norm = thought.norm();
        let norm_score = if norm > 0.0 { 1.0 / (1.0 + (norm - 1.0).abs()) } else { 0.5 };
        
        // Combine
        (energy_score * norm_score).min(1.0)
    }

    /// Compute memory guidance - SOFT INFLUENCE, not retrieval
    /// P_memory(Y_i) = 1.0 if matches, 0.5 if similar, 0.0 if novel
    fn compute_memory_guidance(
        &self,
        thought: &ThoughtState,
        _answer: &str,
        memory: &SemanticMemory,
    ) -> f64 {
        // Find similar patterns in memory
        let similar = memory.find_similar(&thought.vector, 3).unwrap_or_default();
        
        if similar.is_empty() {
            // Novel pattern - no guidance
            return 0.0;
        }
        
        // Calculate average similarity
        let avg_similarity: f64 = similar.iter()
            .map(|(_, sim)| sim)
            .sum::<f64>() / similar.len() as f64;
        
        // Soft guidance based on similarity
        if avg_similarity > 0.8 {
            1.0 // Strong pattern match
        } else if avg_similarity > 0.5 {
            0.5 // Similar pattern
        } else {
            0.0 // Novel pattern
        }
    }

    /// Compute style/novelty factor V(Y_i)
    /// Adapts to user preferences
    fn compute_style_factor(&self, answer: &str, user: &UserEmbedding) -> f64 {
        let word_count = answer.split_whitespace().count();
        
        // Verbosity match
        let expected_words = 10.0 + user.verbosity * 40.0; // 10-50 words
        let verbosity_score = 1.0 - ((word_count as f64 - expected_words).abs() / expected_words).min(1.0);
        
        // Technical level (simple heuristic: longer words = more technical)
        let avg_word_len: f64 = answer.split_whitespace()
            .map(|w| w.len() as f64)
            .sum::<f64>() / word_count.max(1) as f64;
        let technical_score = if avg_word_len > 5.0 {
            user.technical_level
        } else {
            1.0 - user.technical_level
        };
        
        // Combine
        (verbosity_score + technical_score) / 2.0
    }

    /// Generate explanation of reasoning process
    fn generate_explanation(
        &self,
        neural_prob: f64,
        memory_guidance: f64,
        confidence: f64,
        style: f64,
        context: &[String],
    ) -> Vec<String> {
        let mut explanation = Vec::new();
        
        explanation.push(format!(
            "Neural network probability: {:.1}%",
            neural_prob * 100.0
        ));
        
        if memory_guidance > 0.0 {
            explanation.push(format!(
                "Memory guidance: {:.1}% (similar patterns found)",
                memory_guidance * 100.0
            ));
        } else {
            explanation.push("Memory guidance: Novel pattern (no similar examples)".to_string());
        }
        
        explanation.push(format!(
            "Confidence verification: {:.1}%",
            confidence * 100.0
        ));
        
        explanation.push(format!(
            "Style match: {:.1}%",
            style * 100.0
        ));
        
        if !context.is_empty() {
            explanation.push(format!(
                "Context: {} previous messages considered",
                context.len()
            ));
        }
        
        explanation
    }

    /// Check if candidate needs clarification
    pub fn needs_clarification(&self, candidate: &ScoredCandidate) -> bool {
        candidate.confidence < self.confidence_threshold
    }

    /// Generate follow-up question Q' ~ P_θ(Q' | X, A, u, e)
    pub fn generate_followup_question(
        &self,
        candidate: &ScoredCandidate,
        user: &UserEmbedding,
    ) -> Option<String> {
        // Only generate if confidence is medium (not too low, not too high)
        if candidate.confidence < 0.4 {
            Some("Could you provide more details or rephrase your question?".to_string())
        } else if candidate.confidence < 0.7 && user.verbosity > 0.5 {
            Some("Would you like me to explain this in more detail?".to_string())
        } else if candidate.memory_guidance < 0.3 && user.creativity > 0.6 {
            Some("This is a novel question for me. Would you like to explore related topics?".to_string())
        } else {
            None
        }
    }
}

/// Self-correction signal for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionSignal {
    /// Original thought
    pub thought: ThoughtState,
    /// Original answer
    pub answer: String,
    /// Corrected answer
    pub correction: String,
    /// User feedback
    pub feedback: String,
    /// Confidence of original
    pub original_confidence: f64,
}

impl CorrectionSignal {
    pub fn new(
        thought: ThoughtState,
        answer: String,
        correction: String,
        feedback: String,
        confidence: f64,
    ) -> Self {
        Self {
            thought,
            answer,
            correction,
            feedback,
            original_confidence: confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::EnergyWeights;

    #[test]
    fn test_candidate_scoring() {
        let scorer = CandidateScorer::new(64);
        let thought = ThoughtState::random(64);
        let energy = EnergyResult {
            total: 0.3,
            constraint_energy: 0.1,
            risk_energy: 0.1,
            uncertainty_energy: 0.1,
            confidence_score: 0.8,
            verified: true,
        };
        
        let memory = SemanticMemory::in_memory(64).unwrap();
        let user = UserEmbedding::default();
        let context = vec![];
        
        let candidate = scorer.score_candidate(
            &thought,
            "This is a test answer",
            &energy,
            &memory,
            &user,
            &context,
        );
        
        assert!(candidate.final_score > 0.0);
        assert!(candidate.final_score <= 1.0);
        assert!(!candidate.explanation.is_empty());
    }

    #[test]
    fn test_memory_guidance_soft_influence() {
        let scorer = CandidateScorer::new(64);
        let thought = ThoughtState::random(64);
        let memory = SemanticMemory::in_memory(64).unwrap();
        
        // Memory guidance should be between 0 and 1
        let guidance = scorer.compute_memory_guidance(&thought, "test", &memory);
        assert!(guidance >= 0.0 && guidance <= 1.0);
    }

    #[test]
    fn test_followup_generation() {
        let scorer = CandidateScorer::new(64);
        let candidate = ScoredCandidate {
            answer: "test".to_string(),
            thought: ThoughtState::random(64),
            neural_probability: 0.7,
            memory_guidance: 0.5,
            confidence: 0.5,
            style_factor: 0.8,
            final_score: 0.6,
            explanation: vec![],
        };
        
        let user = UserEmbedding {
            verbosity: 0.8,
            ..Default::default()
        };
        
        let followup = scorer.generate_followup_question(&candidate, &user);
        assert!(followup.is_some());
    }
}
