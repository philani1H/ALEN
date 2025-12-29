//! Epistemic Reward System
//!
//! Implements the anti-hallucination reward function:
//! R = w₁V + w₂P + w₃C - w₄H
//!
//! Where:
//! - V = Verification success (forward + backward)
//! - P = Proof consistency (energy-based)
//! - C = Confidence calibration (accuracy)
//! - H = Hallucination penalty (guessing without proof)
//!
//! This reward function ensures:
//! 1. AI is rewarded for BEING right, not SOUNDING right
//! 2. Confidence grows only with proof success
//! 3. Silence is preferred over guessing
//! 4. Epistemic humility is maintained

use crate::core::{ThoughtState, Problem, EnergyResult};
use serde::{Deserialize, Serialize};

/// Reward function weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardWeights {
    /// Weight for verification success (w₁)
    pub verification: f64,
    /// Weight for proof consistency (w₂)
    pub proof: f64,
    /// Weight for confidence calibration (w₃)
    pub confidence: f64,
    /// Weight for hallucination penalty (w₄)
    pub hallucination: f64,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            verification: 1.0,      // Highest priority: correctness
            proof: 0.5,             // Medium: proof quality
            confidence: 0.3,        // Lower: confidence accuracy
            hallucination: 2.0,     // High penalty: guessing without proof
        }
    }
}

/// Complete reward calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicReward {
    /// Total reward: R = w₁V + w₂P + w₃C - w₄H
    pub total: f64,
    
    /// Verification score V ∈ [0, 1]
    pub verification_score: f64,
    
    /// Proof consistency score P ∈ [0, 1]
    pub proof_score: f64,
    
    /// Confidence calibration score C ∈ [0, 1]
    pub confidence_score: f64,
    
    /// Hallucination penalty H ∈ [0, 1]
    pub hallucination_penalty: f64,
    
    /// Whether this should be committed to memory
    pub should_commit: bool,
    
    /// Breakdown for debugging
    pub breakdown: RewardBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardBreakdown {
    pub verification_contribution: f64,
    pub proof_contribution: f64,
    pub confidence_contribution: f64,
    pub hallucination_contribution: f64,
}

/// Verification result for reward calculation
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Forward check: Does solution match expected?
    pub forward_passed: bool,
    pub forward_error: f64,
    
    /// Backward check: Can we reconstruct problem from solution?
    pub backward_passed: bool,
    pub backward_error: f64,
    
    /// Overall verification success
    pub verified: bool,
}

/// Epistemic reward calculator
pub struct EpistemicRewardCalculator {
    pub weights: RewardWeights,
    
    /// Maximum energy for normalization
    pub max_energy: f64,
    
    /// Verification thresholds
    pub forward_threshold: f64,
    pub backward_threshold: f64,
    
    /// Confidence calibration threshold
    pub confidence_threshold: f64,
}

impl Default for EpistemicRewardCalculator {
    fn default() -> Self {
        Self {
            weights: RewardWeights::default(),
            max_energy: 2.0,
            forward_threshold: 0.3,
            backward_threshold: 0.3,
            confidence_threshold: 0.7,
        }
    }
}

impl EpistemicRewardCalculator {
    pub fn new(weights: RewardWeights) -> Self {
        Self {
            weights,
            ..Default::default()
        }
    }

    /// Calculate complete epistemic reward
    /// R = w₁V + w₂P + w₃C - w₄H
    pub fn calculate_reward(
        &self,
        candidate: &ThoughtState,
        problem: &Problem,
        energy: &EnergyResult,
        claimed_confidence: f64,
        has_proof: bool,
    ) -> EpistemicReward {
        // 1. Verification score V
        let verification = self.calculate_verification_score(candidate, problem);
        let v_score = if verification.verified { 1.0 } else { 0.0 };

        // 2. Proof consistency score P
        let p_score = self.calculate_proof_score(energy);

        // 3. Confidence calibration score C
        let c_score = self.calculate_confidence_score(
            claimed_confidence,
            verification.verified,
        );

        // 4. Hallucination penalty H
        let h_penalty = self.calculate_hallucination_penalty(
            has_proof,
            verification.verified,
            claimed_confidence,
        );

        // Calculate total reward
        let v_contrib = self.weights.verification * v_score;
        let p_contrib = self.weights.proof * p_score;
        let c_contrib = self.weights.confidence * c_score;
        let h_contrib = self.weights.hallucination * h_penalty;

        let total = v_contrib + p_contrib + c_contrib - h_contrib;

        // Should commit only if verified and not hallucinating
        let should_commit = verification.verified && h_penalty < 0.5;

        EpistemicReward {
            total,
            verification_score: v_score,
            proof_score: p_score,
            confidence_score: c_score,
            hallucination_penalty: h_penalty,
            should_commit,
            breakdown: RewardBreakdown {
                verification_contribution: v_contrib,
                proof_contribution: p_contrib,
                confidence_contribution: c_contrib,
                hallucination_contribution: -h_contrib,
            },
        }
    }

    /// Calculate verification score V
    /// V = 1 if T⁻¹(T(ψ)) ≈ ψ, else 0
    fn calculate_verification_score(
        &self,
        candidate: &ThoughtState,
        problem: &Problem,
    ) -> VerificationResult {
        // Forward check: Does candidate match target?
        let (forward_passed, forward_error) = if let Some(ref target) = problem.target_state {
            let error = candidate.distance(target);
            (error < self.forward_threshold, error)
        } else {
            // No target provided - can't verify forward
            (false, 1.0)
        };

        // Backward check: Can we reconstruct problem from candidate?
        let (backward_passed, backward_error) = {
            // Simplified backward check: measure structural similarity
            let similarity = candidate.cosine_similarity(&problem.state);
            let error = 1.0 - similarity;
            (error < self.backward_threshold, error)
        };

        // Both checks must pass for verification
        let verified = forward_passed && backward_passed;

        VerificationResult {
            forward_passed,
            forward_error,
            backward_passed,
            backward_error,
            verified,
        }
    }

    /// Calculate proof consistency score P
    /// P = 1 - E(proof) / E_max
    /// Lower energy = higher score
    fn calculate_proof_score(&self, energy: &EnergyResult) -> f64 {
        let normalized_energy = (energy.total / self.max_energy).min(1.0);
        (1.0 - normalized_energy).max(0.0)
    }

    /// Calculate confidence calibration score C
    /// C = 1 - |claimed_confidence - actual_correctness|
    /// Punishes false confidence
    fn calculate_confidence_score(
        &self,
        claimed_confidence: f64,
        is_correct: bool,
    ) -> f64 {
        let actual_correctness = if is_correct { 1.0 } else { 0.0 };
        let calibration_error = (claimed_confidence - actual_correctness).abs();
        (1.0 - calibration_error).max(0.0)
    }

    /// Calculate hallucination penalty H
    /// H = 1 if answer given without proof, else 0
    /// Forces silence over guessing
    fn calculate_hallucination_penalty(
        &self,
        has_proof: bool,
        is_verified: bool,
        claimed_confidence: f64,
    ) -> f64 {
        // High penalty if:
        // 1. No proof provided but high confidence claimed
        // 2. Not verified but answered anyway
        
        if !has_proof && claimed_confidence > self.confidence_threshold {
            // Claiming high confidence without proof = hallucination
            return 1.0;
        }

        if !is_verified && claimed_confidence > 0.5 {
            // Moderate confidence without verification = guessing
            return 0.5;
        }

        // No penalty if properly calibrated
        0.0
    }

    /// Calculate earned confidence (not claimed)
    /// Confidence grows only with successful verifications
    pub fn calculate_earned_confidence(
        &self,
        successful_verifications: u64,
        total_attempts: u64,
    ) -> f64 {
        if total_attempts == 0 {
            return 0.0;
        }

        let success_rate = successful_verifications as f64 / total_attempts as f64;
        
        // Apply confidence curve: slower growth at extremes
        // This prevents overconfidence
        let calibrated = if success_rate < 0.5 {
            // Low success: confidence grows slowly
            success_rate * 0.8
        } else if success_rate > 0.9 {
            // High success: cap confidence to maintain humility
            0.9 + (success_rate - 0.9) * 0.5
        } else {
            // Middle range: linear growth
            success_rate
        };

        calibrated.clamp(0.0, 0.95) // Never 100% confident
    }
}

/// Operator performance tracker with epistemic metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicOperatorStats {
    pub operator_id: String,
    pub operator_type: String,
    
    /// Total attempts
    pub attempts: u64,
    
    /// Successful verifications
    pub verified_successes: u64,
    
    /// Earned confidence (not claimed)
    pub earned_confidence: f64,
    
    /// Average reward received
    pub average_reward: f64,
    
    /// Hallucination count
    pub hallucinations: u64,
    
    /// Current weight
    pub weight: f64,
}

impl EpistemicOperatorStats {
    pub fn new(operator_id: String, operator_type: String) -> Self {
        Self {
            operator_id,
            operator_type,
            attempts: 0,
            verified_successes: 0,
            earned_confidence: 0.0,
            average_reward: 0.0,
            hallucinations: 0,
            weight: 1.0,
        }
    }

    /// Update stats after receiving reward
    pub fn update(&mut self, reward: &EpistemicReward, calculator: &EpistemicRewardCalculator) {
        self.attempts += 1;

        if reward.verification_score > 0.5 {
            self.verified_successes += 1;
        }

        if reward.hallucination_penalty > 0.5 {
            self.hallucinations += 1;
        }

        // Update average reward (exponential moving average)
        let alpha = 0.1;
        self.average_reward = alpha * reward.total + (1.0 - alpha) * self.average_reward;

        // Update earned confidence
        self.earned_confidence = calculator.calculate_earned_confidence(
            self.verified_successes,
            self.attempts,
        );

        // Update weight based on reward
        let learning_rate = 0.01;
        self.weight += learning_rate * reward.total;
        self.weight = self.weight.clamp(0.1, 2.0);
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.attempts == 0 {
            return 0.0;
        }
        self.verified_successes as f64 / self.attempts as f64
    }

    /// Get hallucination rate
    pub fn hallucination_rate(&self) -> f64 {
        if self.attempts == 0 {
            return 0.0;
        }
        self.hallucinations as f64 / self.attempts as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_score() {
        let calculator = EpistemicRewardCalculator::default();
        let problem = Problem::training("test", "answer", 128);
        let candidate = problem.target_state.as_ref().unwrap().clone();
        
        let verification = calculator.calculate_verification_score(&candidate, &problem);
        assert!(verification.verified);
        assert!(verification.forward_passed);
    }

    #[test]
    fn test_proof_score() {
        let calculator = EpistemicRewardCalculator::default();
        
        // Low energy = high score
        let low_energy = EnergyResult {
            total: 0.2,
            constraint_energy: 0.1,
            risk_energy: 0.05,
            uncertainty_energy: 0.05,
            verified: true,
            confidence_score: 0.8,
        };
        let score = calculator.calculate_proof_score(&low_energy);
        assert!(score > 0.8);

        // High energy = low score
        let high_energy = EnergyResult {
            total: 1.8,
            constraint_energy: 0.6,
            risk_energy: 0.6,
            uncertainty_energy: 0.6,
            verified: false,
            confidence_score: 0.2,
        };
        let score = calculator.calculate_proof_score(&high_energy);
        assert!(score < 0.2);
    }

    #[test]
    fn test_confidence_calibration() {
        let calculator = EpistemicRewardCalculator::default();
        
        // Well-calibrated: high confidence + correct
        let score = calculator.calculate_confidence_score(0.9, true);
        assert!(score > 0.8);

        // Poorly calibrated: high confidence + incorrect
        let score = calculator.calculate_confidence_score(0.9, false);
        assert!(score < 0.2);

        // Well-calibrated: low confidence + incorrect
        let score = calculator.calculate_confidence_score(0.1, false);
        assert!(score > 0.8);
    }

    #[test]
    fn test_hallucination_penalty() {
        let calculator = EpistemicRewardCalculator::default();
        
        // No proof + high confidence = hallucination
        let penalty = calculator.calculate_hallucination_penalty(false, false, 0.9);
        assert_eq!(penalty, 1.0);

        // Has proof + verified = no penalty
        let penalty = calculator.calculate_hallucination_penalty(true, true, 0.9);
        assert_eq!(penalty, 0.0);

        // No verification + moderate confidence = guessing
        let penalty = calculator.calculate_hallucination_penalty(false, false, 0.6);
        assert_eq!(penalty, 0.5);
    }

    #[test]
    fn test_earned_confidence_growth() {
        let calculator = EpistemicRewardCalculator::default();
        
        // No attempts = no confidence
        let conf = calculator.calculate_earned_confidence(0, 0);
        assert_eq!(conf, 0.0);

        // 50% success = moderate confidence
        let conf = calculator.calculate_earned_confidence(5, 10);
        assert!(conf > 0.4 && conf < 0.6);

        // 90% success = high but capped confidence
        let conf = calculator.calculate_earned_confidence(9, 10);
        assert!(conf < 0.95); // Never 100%
    }

    #[test]
    fn test_complete_reward_calculation() {
        let calculator = EpistemicRewardCalculator::default();
        let problem = Problem::training("test", "answer", 128);
        let candidate = problem.target_state.as_ref().unwrap().clone();
        
        let energy = EnergyResult {
            total: 0.3,
            constraint_energy: 0.1,
            risk_energy: 0.1,
            uncertainty_energy: 0.1,
            verified: true,
            confidence_score: 0.7,
        };

        // Good case: verified, low energy, calibrated confidence
        let reward = calculator.calculate_reward(
            &candidate,
            &problem,
            &energy,
            0.7,  // claimed confidence
            true, // has proof
        );

        assert!(reward.total > 0.5);
        assert!(reward.should_commit);
        assert_eq!(reward.hallucination_penalty, 0.0);

        // Bad case: not verified, high confidence, no proof
        let bad_candidate = ThoughtState::random(128);
        let reward = calculator.calculate_reward(
            &bad_candidate,
            &problem,
            &energy,
            0.9,   // high claimed confidence
            false, // no proof
        );

        assert!(reward.total < 0.0); // Negative reward
        assert!(!reward.should_commit);
        assert_eq!(reward.hallucination_penalty, 1.0);
    }
}
