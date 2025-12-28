//! Evaluator Module - Energy Function E(ψ)
//!
//! The "self-test" mechanism that calculates how well a thought matches
//! the input and known facts. Lower energy = better thought.
//! E(ψ) = αC(ψ) + βR(ψ) + γU(ψ)

use crate::core::state::{ThoughtState, Problem};
use serde::{Deserialize, Serialize};

/// Energy function weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyWeights {
    /// Weight for constraint violations (α)
    pub alpha: f64,
    /// Weight for risk/inconsistency (β)
    pub beta: f64,
    /// Weight for uncertainty (γ)
    pub gamma: f64,
}

impl Default for EnergyWeights {
    fn default() -> Self {
        Self {
            alpha: 0.4,
            beta: 0.3,
            gamma: 0.3,
        }
    }
}

/// Result of energy calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyResult {
    /// Total energy (lower is better)
    pub total: f64,
    /// Constraint violation component C(ψ)
    pub constraint_energy: f64,
    /// Risk/inconsistency component R(ψ)
    pub risk_energy: f64,
    /// Uncertainty component U(ψ)
    pub uncertainty_energy: f64,
    /// Whether this passes the confidence threshold
    pub verified: bool,
    /// Confidence score (inverse of energy, normalized)
    pub confidence_score: f64,
}

/// The Energy Evaluator - scores candidate thoughts
#[derive(Debug, Clone)]
pub struct Evaluator {
    /// Weights for the energy function
    pub weights: EnergyWeights,
    /// Confidence threshold for verification
    pub confidence_threshold: f64,
    /// Maximum allowed energy for acceptance
    pub energy_threshold: f64,
}

impl Default for Evaluator {
    fn default() -> Self {
        Self {
            weights: EnergyWeights::default(),
            confidence_threshold: 0.5,
            energy_threshold: 0.8,
        }
    }
}

impl Evaluator {
    /// Create a new evaluator with custom weights
    pub fn new(weights: EnergyWeights, confidence_threshold: f64) -> Self {
        Self {
            weights,
            confidence_threshold,
            energy_threshold: 1.0 - confidence_threshold,
        }
    }

    /// Calculate the total energy of a thought state
    pub fn evaluate(&self, thought: &ThoughtState, problem: &Problem) -> EnergyResult {
        let constraint_energy = self.calculate_constraint_energy(thought, problem);
        let risk_energy = self.calculate_risk_energy(thought, problem);
        let uncertainty_energy = self.calculate_uncertainty_energy(thought);

        let total = self.weights.alpha * constraint_energy
            + self.weights.beta * risk_energy
            + self.weights.gamma * uncertainty_energy;

        let confidence_score = (1.0 - total.min(1.0)).max(0.0);
        let verified = total < self.energy_threshold && confidence_score > self.confidence_threshold;

        EnergyResult {
            total,
            constraint_energy,
            risk_energy,
            uncertainty_energy,
            verified,
            confidence_score,
        }
    }

    /// Calculate constraint violation energy C(ψ)
    /// Higher value = more constraint violations
    fn calculate_constraint_energy(&self, thought: &ThoughtState, problem: &Problem) -> f64 {
        let mut energy = 0.0;

        // Check similarity to problem state (should be related)
        let problem_similarity = thought.cosine_similarity(&problem.state);
        if problem_similarity < 0.3 {
            // Penalty for drifting too far from the problem
            energy += (0.3 - problem_similarity) * 0.5;
        }

        // If we have a target state (training mode), check alignment
        if let Some(ref target) = problem.target_state {
            let target_similarity = thought.cosine_similarity(target);
            // Higher similarity to target = lower energy
            energy += (1.0 - target_similarity) * 0.5;
        }

        // Check for vector magnitude constraints (should be normalized)
        let norm = thought.norm();
        if (norm - 1.0).abs() > 0.1 {
            energy += (norm - 1.0).abs() * 0.3;
        }

        // Check for NaN or Inf values
        if thought.vector.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            energy += 1.0;
        }

        energy.min(1.0)
    }

    /// Calculate risk/inconsistency energy R(ψ)
    /// Higher value = more inconsistent with past patterns
    fn calculate_risk_energy(&self, thought: &ThoughtState, problem: &Problem) -> f64 {
        let mut energy = 0.0;

        // Check variance of thought vector (shouldn't be too uniform or too varied)
        let mean: f64 = thought.vector.iter().sum::<f64>() / thought.dimension as f64;
        let variance: f64 = thought.vector.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / thought.dimension as f64;
        
        // Variance should be in reasonable range
        if variance < 0.001 {
            // Too uniform - suspiciously collapsed
            energy += 0.3;
        } else if variance > 0.5 {
            // Too varied - possibly unstable
            energy += (variance - 0.5) * 0.2;
        }

        // Check for extreme values
        let max_abs = thought.vector.iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        if max_abs > 3.0 {
            energy += (max_abs - 3.0) * 0.1;
        }

        // Check context alignment if available
        for context in &problem.context {
            let context_state = ThoughtState::from_input(context, thought.dimension);
            let context_similarity = thought.cosine_similarity(&context_state);
            if context_similarity < 0.1 {
                energy += 0.1; // Penalty for ignoring context
            }
        }

        energy.min(1.0)
    }

    /// Calculate uncertainty energy U(ψ)
    /// Higher value = less confident in this path
    fn calculate_uncertainty_energy(&self, thought: &ThoughtState) -> f64 {
        // Base uncertainty from thought's confidence
        let base_uncertainty = 1.0 - thought.confidence;

        // Entropy-like measure of the thought vector
        let positive_values: Vec<f64> = thought.vector.iter()
            .map(|&x| (x.abs() + 0.001).min(1.0))
            .collect();
        let sum: f64 = positive_values.iter().sum();
        let normalized: Vec<f64> = positive_values.iter()
            .map(|x| x / sum)
            .collect();
        let entropy: f64 = -normalized.iter()
            .map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 })
            .sum::<f64>();
        
        // Normalize entropy to [0, 1]
        let max_entropy = (thought.dimension as f64).ln();
        let normalized_entropy = entropy / max_entropy;

        // Combine base uncertainty with entropy
        (base_uncertainty * 0.5 + normalized_entropy * 0.5).min(1.0)
    }

    /// Evaluate a candidate for training verification
    pub fn evaluate_training(&self, candidate: &ThoughtState, problem: &Problem) -> TrainingEvaluation {
        let energy_result = self.evaluate(candidate, problem);
        
        // Calculate backward inference check
        let backward_check = if let Some(ref target) = problem.target_state {
            self.backward_inference_check(candidate, target, &problem.state)
        } else {
            BackwardCheck {
                passes: false,
                reconstruction_error: 1.0,
                path_consistency: 0.0,
            }
        };

        // Calculate should_commit before moving values
        let should_commit = energy_result.verified && backward_check.passes;

        TrainingEvaluation {
            energy: energy_result,
            backward_check,
            should_commit,
        }
    }

    /// Backward inference check: T^{-1}ψ* ≈ ψ
    /// "If this answer is correct, does it lead back to the original question?"
    fn backward_inference_check(
        &self,
        candidate: &ThoughtState,
        target: &ThoughtState,
        original: &ThoughtState,
    ) -> BackwardCheck {
        // Check if candidate is close to target
        let target_similarity = candidate.cosine_similarity(target);
        
        // Check if we can "trace back" from candidate to original
        // This is a simplified check - in full implementation would use inverse operators
        let path_consistency = self.calculate_path_consistency(candidate, original);
        
        // Reconstruction error - how far is candidate from target
        let reconstruction_error = candidate.distance(target) / (candidate.dimension as f64).sqrt();
        
        BackwardCheck {
            passes: target_similarity > 0.7 && path_consistency > 0.5,
            reconstruction_error,
            path_consistency,
        }
    }

    /// Calculate path consistency between two states
    fn calculate_path_consistency(&self, from: &ThoughtState, to: &ThoughtState) -> f64 {
        // Check structural similarity - do they share similar patterns?
        let cosine = from.cosine_similarity(to);
        
        // Check that dimensions have similar relative magnitudes
        let from_ranks: Vec<usize> = {
            let mut indexed: Vec<(usize, f64)> = from.vector.iter()
                .cloned()
                .enumerate()
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.iter().map(|(i, _)| *i).collect()
        };
        
        let to_ranks: Vec<usize> = {
            let mut indexed: Vec<(usize, f64)> = to.vector.iter()
                .cloned()
                .enumerate()
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.iter().map(|(i, _)| *i).collect()
        };
        
        // Spearman-like rank correlation (simplified)
        let rank_correlation = {
            let n = from_ranks.len().min(to_ranks.len());
            let top_n = n / 4; // Compare top quarter
            
            let from_top: std::collections::HashSet<_> = from_ranks[..top_n].iter().collect();
            let to_top: std::collections::HashSet<_> = to_ranks[..top_n].iter().collect();
            
            let overlap = from_top.intersection(&to_top).count();
            overlap as f64 / top_n as f64
        };
        
        (cosine * 0.6 + rank_correlation * 0.4).max(0.0)
    }

    /// Compare multiple candidates and rank them
    pub fn rank_candidates(
        &self,
        candidates: &[(String, ThoughtState)],
        problem: &Problem,
    ) -> Vec<RankedCandidate> {
        let mut ranked: Vec<RankedCandidate> = candidates
            .iter()
            .map(|(op_id, thought)| {
                let energy = self.evaluate(thought, problem);
                RankedCandidate {
                    operator_id: op_id.clone(),
                    thought: thought.clone(),
                    energy,
                    rank: 0, // Will be set below
                }
            })
            .collect();

        // Sort by energy (ascending - lower is better)
        ranked.sort_by(|a, b| {
            a.energy.total.partial_cmp(&b.energy.total).unwrap()
        });

        // Assign ranks
        for (i, candidate) in ranked.iter_mut().enumerate() {
            candidate.rank = i + 1;
        }

        ranked
    }

    /// Select the best candidate (minimum energy principle)
    pub fn select_best(
        &self,
        candidates: &[(String, ThoughtState)],
        problem: &Problem,
    ) -> Option<(String, ThoughtState, EnergyResult)> {
        let ranked = self.rank_candidates(candidates, problem);
        ranked.into_iter()
            .next()
            .map(|rc| (rc.operator_id, rc.thought, rc.energy))
    }
}

/// Result of backward inference check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackwardCheck {
    /// Whether the backward check passes
    pub passes: bool,
    /// Error in reconstructing the target
    pub reconstruction_error: f64,
    /// Consistency of the reasoning path
    pub path_consistency: f64,
}

/// Complete training evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEvaluation {
    /// Energy evaluation
    pub energy: EnergyResult,
    /// Backward inference check
    pub backward_check: BackwardCheck,
    /// Whether this should be committed to memory
    pub should_commit: bool,
}

/// A ranked candidate thought
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedCandidate {
    /// ID of the operator that produced this
    pub operator_id: String,
    /// The candidate thought state
    pub thought: ThoughtState,
    /// Energy evaluation
    pub energy: EnergyResult,
    /// Rank (1 = best)
    pub rank: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_evaluation() {
        let evaluator = Evaluator::default();
        let thought = ThoughtState::from_input("answer", 64);
        let problem = Problem::new("question", 64);

        let energy = evaluator.evaluate(&thought, &problem);
        assert!(energy.total >= 0.0);
        assert!(energy.total <= 2.0); // Should be reasonable
    }

    #[test]
    fn test_training_evaluation() {
        let evaluator = Evaluator::default();
        let problem = Problem::training("2 + 2", "4", 64);
        let candidate = ThoughtState::from_input("4", 64);

        let eval = evaluator.evaluate_training(&candidate, &problem);
        // Should have high confidence since candidate matches target
        assert!(eval.energy.confidence_score > 0.5);
    }

    #[test]
    fn test_candidate_ranking() {
        let evaluator = Evaluator::default();
        let problem = Problem::training("test", "answer", 64);
        
        let candidates = vec![
            ("op1".to_string(), ThoughtState::from_input("answer", 64)),
            ("op2".to_string(), ThoughtState::from_input("wrong", 64)),
            ("op3".to_string(), ThoughtState::from_input("different", 64)),
        ];

        let ranked = evaluator.rank_candidates(&candidates, &problem);
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].rank, 1);
        assert_eq!(ranked[2].rank, 3);
    }
}
