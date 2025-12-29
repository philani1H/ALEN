//! Universal Expert System
//!
//! Complete solve-verify-explain pipeline that:
//! 1. Solves problems
//! 2. Verifies correctness
//! 3. Explains at any comprehension level
//! 4. Tracks teaching effectiveness
//! 5. Continuously improves

use crate::api::user_modeling::{UserModelingManager, UserState};
use crate::core::{Problem, ThoughtState, EnergyResult};
use crate::learning::verification_loop::{VerificationLoop, VerificationResult};
use crate::memory::{EpisodicMemory, SemanticMemory};
use crate::verification::FormalVerifier;
use super::cognitive_distance::{CognitiveDistanceCalculator, CognitiveDistance};
use super::multimodal_generator::{MultiModalExplanationGenerator, CompleteExplanation};
use super::effectiveness_tracker::{TeachingEffectivenessTracker, UserFeedback};
use serde::{Deserialize, Serialize};

// ============================================================================
// PART 1: SOLUTION STRUCTURE
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    /// Answer text
    pub answer: String,
    
    /// Thought vector
    pub thought: ThoughtState,
    
    /// Energy result
    pub energy: EnergyResult,
    
    /// Confidence score
    pub confidence: f64,
}

// ============================================================================
// PART 2: UNIVERSAL EXPERT RESPONSE
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalExpertResponse {
    /// Solution
    pub solution: Solution,
    
    /// Verification result
    pub verification: VerificationResult,
    
    /// Complete explanation
    pub explanation: CompleteExplanation,
    
    /// Cognitive distance
    pub cognitive_distance: CognitiveDistance,
    
    /// Teaching quality estimate
    pub teaching_quality: f64,
    
    /// Was answer refused?
    pub refused: bool,
    
    /// Refusal reason (if refused)
    pub refusal_reason: Option<String>,
}

// ============================================================================
// PART 3: UNIVERSAL EXPERT SYSTEM
// ============================================================================

pub struct UniversalExpertSystem {
    /// Verification loop
    verification_loop: VerificationLoop,
    
    /// Formal verifier
    formal_verifier: FormalVerifier,
    
    /// Explanation generator
    explanation_generator: MultiModalExplanationGenerator,
    
    /// Cognitive distance calculator
    cognitive_distance: CognitiveDistanceCalculator,
    
    /// Teaching effectiveness tracker
    effectiveness_tracker: TeachingEffectivenessTracker,
    
    /// User modeling
    user_modeling: UserModelingManager,
}

impl UniversalExpertSystem {
    pub fn new(verification_loop: VerificationLoop) -> Self {
        Self {
            verification_loop,
            formal_verifier: FormalVerifier::new(),
            explanation_generator: MultiModalExplanationGenerator::new(),
            cognitive_distance: CognitiveDistanceCalculator::new(),
            effectiveness_tracker: TeachingEffectivenessTracker::new(),
            user_modeling: UserModelingManager::new(128),
        }
    }
    
    /// Solve, verify, and explain a problem
    pub fn solve_verify_explain(
        &mut self,
        problem: &Problem,
        solution: Solution,
        user_id: &str,
    ) -> UniversalExpertResponse {
        // 1. Get user profile
        let user = self.user_modeling.get_or_create(user_id);
        
        // 2. Verify solution
        let verification = self.verification_loop.verify_and_store(
            problem,
            &solution.thought,
            &solution.energy,
            &solution.answer,
            "general", // TODO: extract domain from problem
        );
        
        // 3. Formal verification (if applicable)
        let formal_verification = if verification.verified {
            self.formal_verifier.verify_math(&problem.input, &solution.answer)
        } else {
            // Skip formal verification if already failed
            crate::verification::MathVerificationResult {
                verified: false,
                symbolic_solution: None,
                neural_solution: solution.answer.clone(),
                match_score: 0.0,
                reasoning: "Skipped due to failed verification".to_string(),
            }
        };
        
        // 4. Generate explanation (if verified)
        // Clone user to avoid borrow issues
        let user_clone = user.clone();
        
        let (explanation, refused, refusal_reason) = if verification.verified && formal_verification.verified {
            let exp = self.explanation_generator.generate_complete_explanation(
                &problem.input,
                &solution.answer,
                &user_clone,
            );
            (exp, false, None)
        } else {
            // Explain why we can't answer
            let refusal = self.explain_refusal(&verification, &formal_verification, &user_clone);
            (refusal.0, true, Some(refusal.1))
        };
        
        // 5. Compute cognitive distance
        let cognitive_distance = self.cognitive_distance.compute_distance(
            &explanation.text,
            &user_clone,
        );
        
        // 6. If distance too high, regenerate with simpler style
        let final_explanation = if cognitive_distance.total > 0.7 && !refused {
            self.simplify_explanation(&explanation, &user_clone)
        } else {
            explanation
        };
        
        // 7. Estimate teaching quality
        let teaching_quality = self.estimate_teaching_quality(&final_explanation, &cognitive_distance);
        
        UniversalExpertResponse {
            solution,
            verification,
            explanation: final_explanation,
            cognitive_distance,
            teaching_quality,
            refused,
            refusal_reason,
        }
    }
    
    /// Record user feedback
    pub fn record_user_feedback(
        &mut self,
        user_id: &str,
        explanation_id: &str,
        message: &str,
        is_followup: bool,
        latency: f64,
        engaged: bool,
    ) {
        // Update user model
        self.user_modeling.update_user(
            user_id,
            message,
            is_followup,
            latency,
            engaged,
        );
        
        // Create feedback
        let feedback = UserFeedback::from_interaction(
            "general".to_string(), // TODO: extract concept
            message.to_string(),
            is_followup,
            latency,
            engaged,
        );
        
        // Update teaching effectiveness
        let user = self.user_modeling.get(user_id).unwrap();
        self.effectiveness_tracker.record_outcome(
            &feedback.concept,
            user.archetype(),
            explanation_id,
            &feedback,
        );
    }
    
    /// Explain why we're refusing to answer
    fn explain_refusal(
        &self,
        verification: &VerificationResult,
        formal_verification: &crate::verification::MathVerificationResult,
        user: &UserState,
    ) -> (CompleteExplanation, String) {
        let reason = if !verification.verified {
            format!(
                "I don't have enough confidence to answer this question. {}",
                verification.reasoning
            )
        } else if !formal_verification.verified {
            format!(
                "My answer doesn't match the formal verification. {}",
                formal_verification.reasoning
            )
        } else {
            "I cannot provide a reliable answer at this time.".to_string()
        };
        
        let explanation = CompleteExplanation {
            text: reason.clone(),
            visual: None,
            analogies: vec![],
            steps: vec![],
            examples: vec![],
            audience_adapted: true,
            id: format!("refusal_{}", Self::current_timestamp()),
        };
        
        (explanation, reason)
    }
    
    /// Simplify explanation for better understanding
    fn simplify_explanation(
        &self,
        explanation: &CompleteExplanation,
        user: &UserState,
    ) -> CompleteExplanation {
        // Simplify text by breaking into shorter sentences
        let simplified_text = self.simplify_text(&explanation.text);
        
        // Add more examples if user prefers them
        let mut new_explanation = explanation.clone();
        new_explanation.text = simplified_text;
        
        // Add visual if not present
        if new_explanation.visual.is_none() && user.preferences.depth.mean() > 0.5 {
            // Could add a simple visual here
        }
        
        new_explanation
    }
    
    /// Simplify text
    fn simplify_text(&self, text: &str) -> String {
        // Break long sentences into shorter ones
        let sentences: Vec<&str> = text.split(&['.', '!', '?'][..]).collect();
        
        let simplified: Vec<String> = sentences.iter()
            .filter(|s| !s.trim().is_empty())
            .map(|s| {
                let words: Vec<&str> = s.split_whitespace().collect();
                if words.len() > 15 {
                    // Break into two sentences
                    let mid = words.len() / 2;
                    format!("{}. {}", 
                        words[..mid].join(" "),
                        words[mid..].join(" ")
                    )
                } else {
                    s.to_string()
                }
            })
            .collect();
        
        simplified.join(". ")
    }
    
    /// Estimate teaching quality
    fn estimate_teaching_quality(
        &self,
        explanation: &CompleteExplanation,
        cognitive_distance: &CognitiveDistance,
    ) -> f64 {
        // Quality factors
        let distance_quality = 1.0 - cognitive_distance.total;
        let completeness = self.measure_completeness(explanation);
        let engagement = self.measure_engagement(explanation);
        
        // Weighted average
        0.4 * distance_quality + 0.3 * completeness + 0.3 * engagement
    }
    
    /// Measure completeness of explanation
    fn measure_completeness(&self, explanation: &CompleteExplanation) -> f64 {
        let mut score = 0.0;
        
        // Has text
        if !explanation.text.is_empty() {
            score += 0.3;
        }
        
        // Has steps
        if !explanation.steps.is_empty() {
            score += 0.3;
        }
        
        // Has examples
        if !explanation.examples.is_empty() {
            score += 0.2;
        }
        
        // Has visual or analogies
        if explanation.visual.is_some() || !explanation.analogies.is_empty() {
            score += 0.2;
        }
        
        score
    }
    
    /// Measure engagement potential
    fn measure_engagement(&self, explanation: &CompleteExplanation) -> f64 {
        let mut score: f64 = 0.5; // Base score
        
        // Analogies increase engagement
        if !explanation.analogies.is_empty() {
            score += 0.2;
        }
        
        // Visual increases engagement
        if explanation.visual.is_some() {
            score += 0.2;
        }
        
        // Examples increase engagement
        if !explanation.examples.is_empty() {
            score += 0.1;
        }
        
        score.min(1.0)
    }
    
    /// Get teaching effectiveness statistics
    pub fn get_effectiveness_stats(&self) -> super::effectiveness_tracker::EffectivenessStats {
        self.effectiveness_tracker.get_overall_stats()
    }
    
    /// Get concept effectiveness
    pub fn get_concept_effectiveness(&self, concept: &str) -> Option<&super::effectiveness_tracker::ConceptEffectiveness> {
        self.effectiveness_tracker.get_concept_effectiveness(concept)
    }
    
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Evaluator;
    
    #[test]
    fn test_universal_expert_system() {
        let evaluator = Evaluator::new(crate::core::EnergyWeights::default(), 0.6);
        let verification_loop = VerificationLoop::new(evaluator);
        let mut expert = UniversalExpertSystem::new(verification_loop);
        
        let problem = Problem::new("What is 2+2?", 128);
        let solution = Solution {
            answer: "4".to_string(),
            thought: ThoughtState::random(128),
            energy: EnergyResult {
                total: 0.3,
                constraint_energy: 0.1,
                risk_energy: 0.1,
                uncertainty_energy: 0.1,
                verified: true,
                confidence_score: 0.8,
            },
            confidence: 0.8,
        };
        
        let response = expert.solve_verify_explain(&problem, solution, "test_user");
        
        assert!(!response.explanation.text.is_empty());
        assert!(response.teaching_quality > 0.0);
    }
}
