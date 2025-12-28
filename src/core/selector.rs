//! Selector Module - Executes the argmin logic
//!
//! ψ* = argmin_i E(ψ_i)
//! Selects the best thought from candidates based on energy minimization.

use crate::core::state::{ThoughtState, Problem};
use crate::core::evaluator::{Evaluator, EnergyResult, RankedCandidate};
use serde::{Deserialize, Serialize};
use rand::Rng;

/// Selection strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Always pick the minimum energy candidate
    Greedy,
    /// Softmax selection with temperature
    Softmax { temperature: f64 },
    /// Epsilon-greedy: random with probability epsilon
    EpsilonGreedy { epsilon: f64 },
    /// Tournament selection: pick best from random subset
    Tournament { size: usize },
    /// Top-k synthesis: combine top k candidates
    TopKSynthesis { k: usize },
}

impl Default for SelectionStrategy {
    fn default() -> Self {
        SelectionStrategy::Greedy
    }
}

/// Selection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResult {
    /// The selected thought state
    pub thought: ThoughtState,
    /// ID of the operator that produced it
    pub operator_id: String,
    /// Energy of the selected thought
    pub energy: EnergyResult,
    /// All candidates that were considered
    pub candidates_considered: usize,
    /// The strategy used for selection
    pub strategy_used: SelectionStrategy,
    /// Whether this was a synthesized result
    pub is_synthesis: bool,
}

/// The Selector - picks the best thought from candidates
#[derive(Debug, Clone)]
pub struct Selector {
    /// The evaluator to use for scoring
    pub evaluator: Evaluator,
    /// Selection strategy
    pub strategy: SelectionStrategy,
}

impl Selector {
    /// Create a new selector with default strategy
    pub fn new(evaluator: Evaluator) -> Self {
        Self {
            evaluator,
            strategy: SelectionStrategy::default(),
        }
    }

    /// Create selector with specific strategy
    pub fn with_strategy(evaluator: Evaluator, strategy: SelectionStrategy) -> Self {
        Self {
            evaluator,
            strategy,
        }
    }

    /// Set the selection strategy
    pub fn set_strategy(&mut self, strategy: SelectionStrategy) {
        self.strategy = strategy;
    }

    /// Select the best thought from candidates
    pub fn select(
        &self,
        candidates: &[(String, ThoughtState)],
        problem: &Problem,
    ) -> Option<SelectionResult> {
        if candidates.is_empty() {
            return None;
        }

        match self.strategy {
            SelectionStrategy::Greedy => self.greedy_select(candidates, problem),
            SelectionStrategy::Softmax { temperature } => {
                self.softmax_select(candidates, problem, temperature)
            }
            SelectionStrategy::EpsilonGreedy { epsilon } => {
                self.epsilon_greedy_select(candidates, problem, epsilon)
            }
            SelectionStrategy::Tournament { size } => {
                self.tournament_select(candidates, problem, size)
            }
            SelectionStrategy::TopKSynthesis { k } => {
                self.top_k_synthesis(candidates, problem, k)
            }
        }
    }

    /// Greedy selection: always pick minimum energy
    fn greedy_select(
        &self,
        candidates: &[(String, ThoughtState)],
        problem: &Problem,
    ) -> Option<SelectionResult> {
        let ranked = self.evaluator.rank_candidates(candidates, problem);
        ranked.into_iter().next().map(|rc| SelectionResult {
            thought: rc.thought,
            operator_id: rc.operator_id,
            energy: rc.energy,
            candidates_considered: candidates.len(),
            strategy_used: SelectionStrategy::Greedy,
            is_synthesis: false,
        })
    }

    /// Softmax selection: probabilistic based on energy
    fn softmax_select(
        &self,
        candidates: &[(String, ThoughtState)],
        problem: &Problem,
        temperature: f64,
    ) -> Option<SelectionResult> {
        let ranked = self.evaluator.rank_candidates(candidates, problem);
        if ranked.is_empty() {
            return None;
        }

        // Calculate softmax probabilities
        let energies: Vec<f64> = ranked.iter().map(|rc| -rc.energy.total / temperature).collect();
        let max_energy = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_energies: Vec<f64> = energies.iter().map(|e| (e - max_energy).exp()).collect();
        let sum_exp: f64 = exp_energies.iter().sum();
        let probabilities: Vec<f64> = exp_energies.iter().map(|e| e / sum_exp).collect();

        // Sample from distribution
        let mut rng = rand::thread_rng();
        let threshold: f64 = rng.gen();
        let mut cumulative = 0.0;

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if threshold <= cumulative {
                let rc = &ranked[i];
                return Some(SelectionResult {
                    thought: rc.thought.clone(),
                    operator_id: rc.operator_id.clone(),
                    energy: rc.energy.clone(),
                    candidates_considered: candidates.len(),
                    strategy_used: SelectionStrategy::Softmax { temperature },
                    is_synthesis: false,
                });
            }
        }

        // Fallback to last candidate
        ranked.last().map(|rc| SelectionResult {
            thought: rc.thought.clone(),
            operator_id: rc.operator_id.clone(),
            energy: rc.energy.clone(),
            candidates_considered: candidates.len(),
            strategy_used: SelectionStrategy::Softmax { temperature },
            is_synthesis: false,
        })
    }

    /// Epsilon-greedy: random exploration with probability epsilon
    fn epsilon_greedy_select(
        &self,
        candidates: &[(String, ThoughtState)],
        problem: &Problem,
        epsilon: f64,
    ) -> Option<SelectionResult> {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < epsilon {
            // Random selection
            let idx = rng.gen_range(0..candidates.len());
            let (op_id, thought) = candidates[idx].clone();
            let energy = self.evaluator.evaluate(&thought, problem);

            Some(SelectionResult {
                thought,
                operator_id: op_id,
                energy,
                candidates_considered: candidates.len(),
                strategy_used: SelectionStrategy::EpsilonGreedy { epsilon },
                is_synthesis: false,
            })
        } else {
            // Greedy selection
            let mut result = self.greedy_select(candidates, problem)?;
            result.strategy_used = SelectionStrategy::EpsilonGreedy { epsilon };
            Some(result)
        }
    }

    /// Tournament selection: pick best from random subset
    fn tournament_select(
        &self,
        candidates: &[(String, ThoughtState)],
        problem: &Problem,
        size: usize,
    ) -> Option<SelectionResult> {
        let mut rng = rand::thread_rng();
        let tournament_size = size.min(candidates.len());

        // Select random subset
        let mut indices: Vec<usize> = (0..candidates.len()).collect();
        let tournament_indices: Vec<usize> = {
            let mut selected = Vec::new();
            for _ in 0..tournament_size {
                let idx = rng.gen_range(0..indices.len());
                selected.push(indices.remove(idx));
            }
            selected
        };

        // Evaluate tournament candidates
        let tournament_candidates: Vec<(String, ThoughtState)> = tournament_indices
            .iter()
            .map(|&i| candidates[i].clone())
            .collect();

        let mut result = self.greedy_select(&tournament_candidates, problem)?;
        result.candidates_considered = candidates.len();
        result.strategy_used = SelectionStrategy::Tournament { size };
        Some(result)
    }

    /// Top-K synthesis: combine best k candidates
    fn top_k_synthesis(
        &self,
        candidates: &[(String, ThoughtState)],
        problem: &Problem,
        k: usize,
    ) -> Option<SelectionResult> {
        let ranked = self.evaluator.rank_candidates(candidates, problem);
        if ranked.is_empty() {
            return None;
        }

        let top_k: Vec<&RankedCandidate> = ranked.iter().take(k).collect();

        if top_k.is_empty() {
            return None;
        }

        if top_k.len() == 1 {
            let rc = top_k[0];
            return Some(SelectionResult {
                thought: rc.thought.clone(),
                operator_id: rc.operator_id.clone(),
                energy: rc.energy.clone(),
                candidates_considered: candidates.len(),
                strategy_used: SelectionStrategy::TopKSynthesis { k },
                is_synthesis: false,
            });
        }

        // Synthesize by weighted average based on inverse energy
        let weights: Vec<f64> = top_k.iter()
            .map(|rc| 1.0 / (rc.energy.total + 0.001))
            .collect();
        let total_weight: f64 = weights.iter().sum();
        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / total_weight).collect();

        // Combine thought vectors
        let dimension = top_k[0].thought.dimension;
        let mut combined_vector = vec![0.0; dimension];

        for (i, rc) in top_k.iter().enumerate() {
            for (j, &val) in rc.thought.vector.iter().enumerate() {
                combined_vector[j] += val * normalized_weights[i];
            }
        }

        // Create synthesized thought
        let mut synthesized = ThoughtState {
            vector: combined_vector,
            dimension,
            confidence: top_k.iter()
                .zip(normalized_weights.iter())
                .map(|(rc, w)| rc.thought.confidence * w)
                .sum(),
            metadata: top_k[0].thought.metadata.clone(),
        };
        synthesized.normalize();

        // Evaluate the synthesized thought
        let energy = self.evaluator.evaluate(&synthesized, problem);

        Some(SelectionResult {
            thought: synthesized,
            operator_id: format!("synthesis({})", 
                top_k.iter().map(|rc| rc.operator_id.clone()).collect::<Vec<_>>().join(",")),
            energy,
            candidates_considered: candidates.len(),
            strategy_used: SelectionStrategy::TopKSynthesis { k },
            is_synthesis: true,
        })
    }

    /// Get the evaluator reference
    pub fn evaluator(&self) -> &Evaluator {
        &self.evaluator
    }
}

/// Builder for configuring selection
pub struct SelectorBuilder {
    evaluator: Evaluator,
    strategy: SelectionStrategy,
}

impl SelectorBuilder {
    pub fn new() -> Self {
        Self {
            evaluator: Evaluator::default(),
            strategy: SelectionStrategy::default(),
        }
    }

    pub fn evaluator(mut self, evaluator: Evaluator) -> Self {
        self.evaluator = evaluator;
        self
    }

    pub fn strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn build(self) -> Selector {
        Selector::with_strategy(self.evaluator, self.strategy)
    }
}

impl Default for SelectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_selection() {
        let selector = Selector::new(Evaluator::default());
        let problem = Problem::training("test", "answer", 64);
        
        let candidates = vec![
            ("op1".to_string(), ThoughtState::from_input("answer", 64)),
            ("op2".to_string(), ThoughtState::from_input("wrong", 64)),
        ];

        let result = selector.select(&candidates, &problem);
        assert!(result.is_some());
    }

    #[test]
    fn test_softmax_selection() {
        let selector = Selector::with_strategy(
            Evaluator::default(),
            SelectionStrategy::Softmax { temperature: 1.0 },
        );
        let problem = Problem::new("test", 64);
        
        let candidates = vec![
            ("op1".to_string(), ThoughtState::from_input("a", 64)),
            ("op2".to_string(), ThoughtState::from_input("b", 64)),
        ];

        let result = selector.select(&candidates, &problem);
        assert!(result.is_some());
    }

    #[test]
    fn test_top_k_synthesis() {
        let selector = Selector::with_strategy(
            Evaluator::default(),
            SelectionStrategy::TopKSynthesis { k: 2 },
        );
        let problem = Problem::new("test", 64);
        
        let candidates = vec![
            ("op1".to_string(), ThoughtState::from_input("a", 64)),
            ("op2".to_string(), ThoughtState::from_input("b", 64)),
            ("op3".to_string(), ThoughtState::from_input("c", 64)),
        ];

        let result = selector.select(&candidates, &problem);
        assert!(result.is_some());
        assert!(result.unwrap().is_synthesis);
    }
}
