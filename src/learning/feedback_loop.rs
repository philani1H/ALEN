//! Feedback Loop Module
//!
//! Implements the "try again" logic by adjusting operator weights
//! based on whether the verification succeeded.
//! 
//! Now uses epistemic reward function:
//! R = w₁V + w₂P + w₃C - w₄H
//!
//! Where:
//! - V = Verification success (forward + backward)
//! - P = Proof consistency (energy-based)
//! - C = Confidence calibration (accuracy)
//! - H = Hallucination penalty (guessing without proof)

use crate::core::{
    ThoughtState, Problem, OperatorManager,
    Evaluator, EnergyResult, TrainingEvaluation, Selector,
    SelectionStrategy,
};
use crate::memory::{EpisodicMemory, Episode};
use crate::learning::epistemic_reward::{
    EpistemicRewardCalculator, EpistemicReward, RewardWeights,
    EpistemicOperatorStats,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Learning rate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Base learning rate (η)
    pub learning_rate: f64,
    /// Minimum learning rate
    pub min_learning_rate: f64,
    /// Learning rate decay factor
    pub decay_factor: f64,
    /// Number of candidates to generate per iteration
    pub num_candidates: usize,
    /// Maximum iterations for retry
    pub max_iterations: usize,
    /// Confidence threshold for accepting an answer
    pub confidence_threshold: f64,
    /// Energy threshold for verification
    pub energy_threshold: f64,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            min_learning_rate: 0.001,
            decay_factor: 0.99,
            num_candidates: 5,
            max_iterations: 10,
            confidence_threshold: 0.55,
            energy_threshold: 0.55,
        }
    }
}

/// Result of a training attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Whether training was successful
    pub success: bool,
    /// The best candidate found
    pub best_candidate: Option<ThoughtState>,
    /// Operator that produced the best candidate
    pub best_operator_id: Option<String>,
    /// Energy of the best candidate
    pub best_energy: Option<EnergyResult>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Training evaluation details
    pub evaluation: Option<TrainingEvaluation>,
    /// Rewards applied to operators
    pub rewards: HashMap<String, f64>,
}

/// Result of an inference (thinking) attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// The selected thought
    pub thought: ThoughtState,
    /// Operator that produced it
    pub operator_id: String,
    /// Energy evaluation
    pub energy: EnergyResult,
    /// Confidence in this result
    pub confidence: f64,
    /// Number of candidates considered
    pub candidates_considered: usize,
    /// Was synthesis used
    pub is_synthesis: bool,
}

/// The Feedback Loop - core learning mechanism
pub struct FeedbackLoop {
    /// Operator manager
    operators: OperatorManager,
    /// Evaluator
    evaluator: Evaluator,
    /// Selector
    selector: Selector,
    /// Learning configuration
    config: LearningConfig,
    /// Current learning rate
    current_learning_rate: f64,
    /// Training iteration counter
    iteration_count: u64,
    /// Epistemic reward calculator (anti-hallucination)
    epistemic_reward: EpistemicRewardCalculator,
    /// Operator statistics with epistemic metrics
    operator_stats: HashMap<String, EpistemicOperatorStats>,
}

impl FeedbackLoop {
    /// Create a new feedback loop
    pub fn new(
        operators: OperatorManager,
        evaluator: Evaluator,
        config: LearningConfig,
    ) -> Self {
        let current_learning_rate = config.learning_rate;
        let selector = Selector::with_strategy(
            evaluator.clone(),
            SelectionStrategy::Softmax { temperature: 0.5 },
        );

        // Initialize epistemic reward calculator
        let epistemic_reward = EpistemicRewardCalculator::default();

        // Initialize operator stats
        let mut operator_stats = HashMap::new();
        for (id, op) in &operators.operators {
            operator_stats.insert(
                id.clone(),
                EpistemicOperatorStats::new(
                    id.clone(),
                    format!("{:?}", op.operator_type),
                ),
            );
        }

        Self {
            operators,
            evaluator,
            selector,
            config,
            current_learning_rate,
            iteration_count: 0,
            epistemic_reward,
            operator_stats,
        }
    }

    /// Perform a single training step with verification
    /// This is the "Verification-First Training Loop"
    pub fn train_step(&mut self, problem: &Problem) -> TrainingResult {
        let mut best_candidate: Option<ThoughtState> = None;
        let mut best_operator_id: Option<String> = None;
        let mut best_energy: Option<EnergyResult> = None;
        let mut best_evaluation: Option<TrainingEvaluation> = None;
        let mut rewards: HashMap<String, f64> = HashMap::new();
        let mut iterations = 0;

        // Iterate until we find a verified answer or max iterations
        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            // Step 1: Generate candidates using different operators
            let candidates = self.operators.generate_weighted_candidates(
                &problem.state,
                self.config.num_candidates,
            );

            // Step 2: Evaluate each candidate
            for (op_id, candidate) in &candidates {
                // Step 3: Self-check (backward inference + energy)
                let evaluation = self.evaluator.evaluate_training(candidate, problem);

                // Track best so far
                if best_energy.is_none() || evaluation.energy.total < best_energy.as_ref().unwrap().total {
                    best_candidate = Some(candidate.clone());
                    best_operator_id = Some(op_id.clone());
                    best_energy = Some(evaluation.energy.clone());
                    best_evaluation = Some(evaluation.clone());
                }

                // Calculate epistemic reward: R = w₁V + w₂P + w₃C - w₄H
                let has_proof = evaluation.backward_check.passes;
                let claimed_confidence = candidate.confidence;
                
                let epistemic_reward = self.epistemic_reward.calculate_reward(
                    candidate,
                    problem,
                    &evaluation.energy,
                    claimed_confidence,
                    has_proof,
                );

                // Use epistemic reward (not just energy)
                let reward = epistemic_reward.total;

                // Step 5: Update operator weight with epistemic reward
                self.operators.update_weights(op_id, reward, self.current_learning_rate);
                *rewards.entry(op_id.clone()).or_insert(0.0) += reward;

                // Update operator stats with epistemic metrics
                if let Some(stats) = self.operator_stats.get_mut(op_id) {
                    stats.update(&epistemic_reward, &self.epistemic_reward);
                }

                // Check if this candidate should be committed (Step 4)
                if evaluation.should_commit {
                    // Found a verified answer!
                    self.iteration_count += 1;
                    self.decay_learning_rate();

                    return TrainingResult {
                        success: true,
                        best_candidate: Some(candidate.clone()),
                        best_operator_id: Some(op_id.clone()),
                        best_energy: Some(evaluation.energy.clone()),
                        iterations,
                        evaluation: Some(evaluation),
                        rewards,
                    };
                }
            }
        }

        // Didn't find a verified answer
        self.iteration_count += 1;
        self.decay_learning_rate();

        TrainingResult {
            success: false,
            best_candidate,
            best_operator_id,
            best_energy,
            iterations,
            evaluation: best_evaluation,
            rewards,
        }
    }

    /// Perform inference (thinking) on a problem without known answer
    pub fn infer(&self, problem: &Problem) -> InferenceResult {
        // Generate candidates
        let candidates = self.operators.generate_weighted_candidates(
            &problem.state,
            self.config.num_candidates,
        );

        // Select the best using current strategy
        let selection = self.selector.select(&candidates, problem);

        match selection {
            Some(result) => {
                let confidence = result.energy.confidence_score;
                InferenceResult {
                    thought: result.thought,
                    operator_id: result.operator_id,
                    energy: result.energy,
                    confidence,
                    candidates_considered: result.candidates_considered,
                    is_synthesis: result.is_synthesis,
                }
            },
            None => {
                // Fallback - return problem state with low confidence
                InferenceResult {
                    thought: problem.state.clone(),
                    operator_id: "none".to_string(),
                    energy: EnergyResult {
                        total: 1.0,
                        constraint_energy: 0.5,
                        risk_energy: 0.25,
                        uncertainty_energy: 0.25,
                        verified: false,
                        confidence_score: 0.0,
                    },
                    confidence: 0.0,
                    candidates_considered: 0,
                    is_synthesis: false,
                }
            }
        }
    }

    /// Train on multiple problems (batch training)
    pub fn train_batch(&mut self, problems: &[Problem]) -> BatchTrainingResult {
        let mut successes = 0;
        let mut failures = 0;
        let mut total_iterations = 0;
        let mut results = Vec::new();

        for problem in problems {
            let result = self.train_step(problem);
            total_iterations += result.iterations;

            if result.success {
                successes += 1;
            } else {
                failures += 1;
            }

            results.push(result);
        }

        BatchTrainingResult {
            total_problems: problems.len(),
            successes,
            failures,
            total_iterations,
            average_iterations: total_iterations as f64 / problems.len() as f64,
            success_rate: successes as f64 / problems.len() as f64,
            results,
        }
    }

    /// Decay learning rate
    fn decay_learning_rate(&mut self) {
        self.current_learning_rate = (self.current_learning_rate * self.config.decay_factor)
            .max(self.config.min_learning_rate);
    }

    /// Reset learning rate
    pub fn reset_learning_rate(&mut self) {
        self.current_learning_rate = self.config.learning_rate;
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.current_learning_rate
    }

    /// Get operator statistics
    pub fn get_operator_stats(&self) -> Vec<crate::core::OperatorStats> {
        self.operators.get_statistics()
    }

    /// Get epistemic operator statistics (with hallucination metrics)
    pub fn get_epistemic_stats(&self) -> Vec<EpistemicOperatorStats> {
        self.operator_stats.values().cloned().collect()
    }

    /// Get hallucination rate across all operators
    pub fn get_hallucination_rate(&self) -> f64 {
        let total_attempts: u64 = self.operator_stats.values()
            .map(|s| s.attempts)
            .sum();
        let total_hallucinations: u64 = self.operator_stats.values()
            .map(|s| s.hallucinations)
            .sum();

        if total_attempts == 0 {
            return 0.0;
        }

        total_hallucinations as f64 / total_attempts as f64
    }

    /// Get average earned confidence (not claimed)
    pub fn get_earned_confidence(&self) -> f64 {
        if self.operator_stats.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.operator_stats.values()
            .map(|s| s.earned_confidence)
            .sum();

        sum / self.operator_stats.len() as f64
    }

    /// Get reference to operators
    pub fn operators(&self) -> &OperatorManager {
        &self.operators
    }

    /// Get mutable reference to operators
    pub fn operators_mut(&mut self) -> &mut OperatorManager {
        &mut self.operators
    }

    /// Get iteration count
    pub fn iteration_count(&self) -> u64 {
        self.iteration_count
    }

    /// Set selection strategy
    pub fn set_selection_strategy(&mut self, strategy: SelectionStrategy) {
        self.selector.set_strategy(strategy);
    }
}

/// Result of batch training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTrainingResult {
    pub total_problems: usize,
    pub successes: usize,
    pub failures: usize,
    pub total_iterations: usize,
    pub average_iterations: f64,
    pub success_rate: f64,
    pub results: Vec<TrainingResult>,
}

/// Training session that combines feedback loop with memory
pub struct TrainingSession {
    /// The feedback loop
    pub feedback: FeedbackLoop,
    /// Episodic memory for storing verified results
    pub memory: EpisodicMemory,
    /// Session statistics
    stats: SessionStats,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionStats {
    pub total_training_attempts: u64,
    pub successful_commits: u64,
    pub failed_attempts: u64,
}

impl TrainingSession {
    /// Create a new training session
    pub fn new(feedback: FeedbackLoop, memory: EpisodicMemory) -> Self {
        Self {
            feedback,
            memory,
            stats: SessionStats::default(),
        }
    }

    /// Train on a problem and commit to memory if verified
    pub fn train_and_commit(&mut self, problem: &Problem) -> TrainingResult {
        self.stats.total_training_attempts += 1;

        let result = self.feedback.train_step(problem);

        if result.success {
            // Commit to episodic memory
            if let (Some(ref thought), Some(ref energy), Some(ref op_id)) = 
                (&result.best_candidate, &result.best_energy, &result.best_operator_id) 
            {
                let episode = Episode::from_training(problem, thought, energy, op_id);
                
                if let Ok(stored) = self.memory.store(&episode) {
                    if stored {
                        self.stats.successful_commits += 1;
                    }
                }
            }
        } else {
            self.stats.failed_attempts += 1;
        }

        result
    }

    /// Get session statistics
    pub fn stats(&self) -> &SessionStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{OperatorManager, Evaluator, EnergyWeights};

    fn create_test_feedback() -> FeedbackLoop {
        let operators = OperatorManager::new(64);
        let evaluator = Evaluator::new(EnergyWeights::default(), 0.5);
        let config = LearningConfig {
            max_iterations: 3,
            num_candidates: 3,
            ..Default::default()
        };
        FeedbackLoop::new(operators, evaluator, config)
    }

    #[test]
    fn test_training_step() {
        let mut feedback = create_test_feedback();
        let problem = Problem::training("2 + 2", "4", 64);

        let result = feedback.train_step(&problem);
        assert!(result.iterations > 0);
        assert!(result.best_candidate.is_some());
    }

    #[test]
    fn test_inference() {
        let feedback = create_test_feedback();
        let problem = Problem::new("what is 2 + 2", 64);

        let result = feedback.infer(&problem);
        assert!(result.candidates_considered > 0);
    }

    #[test]
    fn test_batch_training() {
        let mut feedback = create_test_feedback();
        let problems = vec![
            Problem::training("1 + 1", "2", 64),
            Problem::training("2 + 2", "4", 64),
            Problem::training("3 + 3", "6", 64),
        ];

        let result = feedback.train_batch(&problems);
        assert_eq!(result.total_problems, 3);
        assert!(result.total_iterations > 0);
    }

    #[test]
    fn test_learning_rate_decay() {
        let mut feedback = create_test_feedback();
        let initial_lr = feedback.learning_rate();
        
        let problem = Problem::training("test", "answer", 64);
        feedback.train_step(&problem);
        
        let new_lr = feedback.learning_rate();
        assert!(new_lr < initial_lr);
    }
}
