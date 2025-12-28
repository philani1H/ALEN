//! Operators Module - Reasoning Operators (T_i)
//!
//! These are the different ways the AI "thinks". Each operator transforms
//! a thought state into a new state representing a different perspective.
//! |ψ_i⟩ = T_i |ψ⟩

use crate::core::state::ThoughtState;
use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::{Normal, Distribution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Types of reasoning operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperatorType {
    /// Logical deduction - follows strict rules
    Logical,
    /// Probabilistic reasoning - considers likelihoods
    Probabilistic,
    /// Heuristic shortcuts - fast approximations
    Heuristic,
    /// Analogical reasoning - finds patterns from similar problems
    Analogical,
    /// Conservative - risk-averse thinking
    Conservative,
    /// Exploratory - creative, risk-tolerant thinking
    Exploratory,
    /// Analytical - deep, thorough analysis
    Analytical,
    /// Intuitive - fast, gut-feeling based
    Intuitive,
}

impl OperatorType {
    /// Get all operator types
    pub fn all() -> Vec<OperatorType> {
        vec![
            OperatorType::Logical,
            OperatorType::Probabilistic,
            OperatorType::Heuristic,
            OperatorType::Analogical,
            OperatorType::Conservative,
            OperatorType::Exploratory,
            OperatorType::Analytical,
            OperatorType::Intuitive,
        ]
    }
}

/// A Reasoning Operator - transforms thought states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningOperator {
    /// Unique identifier
    pub id: String,
    /// Type of operator
    pub operator_type: OperatorType,
    /// Weight (learned over time) - higher = more trusted
    pub weight: f64,
    /// The transformation matrix
    #[serde(skip)]
    transformation: Option<DMatrix<f64>>,
    /// Serializable transformation data
    transformation_data: Vec<f64>,
    /// Dimension this operator works with
    pub dimension: usize,
    /// Noise level for this operator
    pub noise_level: f64,
    /// Success count for this operator
    pub success_count: u64,
    /// Total usage count
    pub usage_count: u64,
}

impl ReasoningOperator {
    /// Create a new reasoning operator
    pub fn new(operator_type: OperatorType, dimension: usize) -> Self {
        let transformation = Self::generate_transformation(operator_type, dimension);
        let transformation_data: Vec<f64> = transformation.iter().cloned().collect();
        
        Self {
            id: Uuid::new_v4().to_string(),
            operator_type,
            weight: 1.0, // Initial weight
            transformation: Some(transformation),
            transformation_data,
            dimension,
            noise_level: Self::default_noise(operator_type),
            success_count: 0,
            usage_count: 0,
        }
    }

    /// Generate the transformation matrix based on operator type
    fn generate_transformation(operator_type: OperatorType, dimension: usize) -> DMatrix<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();
        
        match operator_type {
            OperatorType::Logical => {
                // Near-identity with small structured perturbations
                let mut matrix = DMatrix::identity(dimension, dimension);
                for i in 0..dimension {
                    for j in 0..dimension {
                        if i == j {
                            matrix[(i, j)] = 0.95 + normal.sample(&mut rng) * 0.05;
                        } else if (i as i32 - j as i32).abs() <= 2 {
                            matrix[(i, j)] = normal.sample(&mut rng) * 0.1;
                        }
                    }
                }
                matrix
            }
            OperatorType::Probabilistic => {
                // Softmax-like transformation
                let mut matrix = DMatrix::from_fn(dimension, dimension, |i, j| {
                    if i == j {
                        0.7 + normal.sample(&mut rng) * 0.1
                    } else {
                        0.3 / (dimension as f64) + normal.sample(&mut rng) * 0.05
                    }
                });
                matrix
            }
            OperatorType::Heuristic => {
                // Sparse transformation - picks key features
                let mut matrix: DMatrix<f64> = DMatrix::zeros(dimension, dimension);
                let key_indices: Vec<usize> = (0..dimension/4)
                    .map(|i| i * 4)
                    .collect();
                for &i in &key_indices {
                    matrix[(i, i)] = 1.5;
                    if i + 1 < dimension {
                        matrix[(i, i + 1)] = 0.3;
                    }
                }
                // Fill diagonal for stability
                for i in 0..dimension {
                    if matrix[(i, i)].abs() < 0.1 {
                        matrix[(i, i)] = 0.5;
                    }
                }
                matrix
            }
            OperatorType::Analogical => {
                // Pattern-matching transformation - connects distant features
                let mut matrix = DMatrix::from_fn(dimension, dimension, |i, j| {
                    let dist = ((i as f64 - j as f64).abs() / dimension as f64 * std::f64::consts::PI).cos();
                    dist * 0.5 + normal.sample(&mut rng) * 0.1
                });
                matrix
            }
            OperatorType::Conservative => {
                // Dampening transformation - reduces extreme values
                DMatrix::from_fn(dimension, dimension, |i, j| {
                    if i == j {
                        0.8 + normal.sample(&mut rng) * 0.05
                    } else {
                        normal.sample(&mut rng) * 0.02
                    }
                })
            }
            OperatorType::Exploratory => {
                // Amplifying transformation - increases variance
                DMatrix::from_fn(dimension, dimension, |i, j| {
                    if i == j {
                        1.2 + normal.sample(&mut rng) * 0.1
                    } else {
                        normal.sample(&mut rng) * 0.15
                    }
                })
            }
            OperatorType::Analytical => {
                // Structured decomposition - emphasizes patterns
                let mut matrix = DMatrix::zeros(dimension, dimension);
                for i in 0..dimension {
                    matrix[(i, i)] = 1.0;
                    // Create structured connections
                    for k in 1..=3 {
                        let j = (i + k * dimension / 4) % dimension;
                        matrix[(i, j)] = 0.2 / k as f64;
                    }
                }
                matrix
            }
            OperatorType::Intuitive => {
                // Random but coherent transformation
                let mut matrix = DMatrix::from_fn(dimension, dimension, |i, j| {
                    if i == j {
                        1.0 + normal.sample(&mut rng) * 0.2
                    } else {
                        normal.sample(&mut rng) * 0.1
                    }
                });
                // Apply SVD-like regularization by clamping singular values
                matrix
            }
        }
    }

    /// Default noise level for each operator type
    fn default_noise(operator_type: OperatorType) -> f64 {
        match operator_type {
            OperatorType::Logical => 0.01,
            OperatorType::Probabilistic => 0.05,
            OperatorType::Heuristic => 0.02,
            OperatorType::Analogical => 0.08,
            OperatorType::Conservative => 0.01,
            OperatorType::Exploratory => 0.15,
            OperatorType::Analytical => 0.03,
            OperatorType::Intuitive => 0.1,
        }
    }

    /// Apply this operator to a thought state
    pub fn apply(&self, thought: &ThoughtState) -> ThoughtState {
        let transformation = self.get_transformation();
        let input = thought.to_dvector();
        
        // Apply transformation: |ψ'⟩ = T|ψ⟩
        let mut output = &transformation * &input;
        
        // Add noise based on operator type
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, self.noise_level).unwrap();
        for i in 0..output.len() {
            output[i] += normal.sample(&mut rng);
        }
        
        // Create the transformed thought state
        let mut new_thought = ThoughtState::from_dvector(
            output,
            thought.confidence * self.weight.min(1.0),
        );
        new_thought.normalize();
        new_thought.metadata.operator_id = Some(self.id.clone());
        new_thought.metadata.iteration = thought.metadata.iteration + 1;
        
        new_thought
    }

    /// Get the transformation matrix
    fn get_transformation(&self) -> DMatrix<f64> {
        if let Some(ref t) = self.transformation {
            t.clone()
        } else {
            // Reconstruct from serialized data
            DMatrix::from_vec(
                self.dimension,
                self.dimension,
                self.transformation_data.clone(),
            )
        }
    }

    /// Compute inverse transformation for backward inference
    pub fn inverse(&self) -> Option<DMatrix<f64>> {
        let transformation = self.get_transformation();
        transformation.try_inverse()
    }

    /// Apply inverse transformation (backward inference)
    pub fn apply_inverse(&self, thought: &ThoughtState) -> Option<ThoughtState> {
        let inverse = self.inverse()?;
        let input = thought.to_dvector();
        let output = &inverse * &input;
        
        let mut new_thought = ThoughtState::from_dvector(
            output,
            thought.confidence,
        );
        new_thought.normalize();
        
        Some(new_thought)
    }

    /// Update weight based on success/failure
    pub fn update_weight(&mut self, reward: f64, learning_rate: f64) {
        // w_i ← w_i + η(reward - E(ψ_i))
        self.weight = (self.weight + learning_rate * reward).clamp(0.1, 3.0);
        self.usage_count += 1;
        if reward > 0.0 {
            self.success_count += 1;
        }
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.usage_count == 0 {
            0.5 // Default success rate
        } else {
            self.success_count as f64 / self.usage_count as f64
        }
    }
}

/// Manager for all reasoning operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorManager {
    /// All available operators
    pub operators: HashMap<String, ReasoningOperator>,
    /// Operators indexed by type
    pub by_type: HashMap<OperatorType, Vec<String>>,
    /// Default dimension
    pub dimension: usize,
}

impl OperatorManager {
    /// Create a new operator manager with default operators
    pub fn new(dimension: usize) -> Self {
        let mut manager = Self {
            operators: HashMap::new(),
            by_type: HashMap::new(),
            dimension,
        };
        
        // Create one operator of each type
        for op_type in OperatorType::all() {
            let op = ReasoningOperator::new(op_type, dimension);
            manager.add_operator(op);
        }
        
        manager
    }

    /// Add an operator
    pub fn add_operator(&mut self, operator: ReasoningOperator) {
        let id = operator.id.clone();
        let op_type = operator.operator_type;
        
        self.by_type
            .entry(op_type)
            .or_insert_with(Vec::new)
            .push(id.clone());
        
        self.operators.insert(id, operator);
    }

    /// Get operator by ID
    pub fn get(&self, id: &str) -> Option<&ReasoningOperator> {
        self.operators.get(id)
    }

    /// Get mutable operator by ID
    pub fn get_mut(&mut self, id: &str) -> Option<&mut ReasoningOperator> {
        self.operators.get_mut(id)
    }

    /// Get operators of a specific type
    pub fn get_by_type(&self, op_type: OperatorType) -> Vec<&ReasoningOperator> {
        self.by_type
            .get(&op_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.operators.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Apply all operators to generate candidate thoughts
    pub fn generate_candidates(&self, thought: &ThoughtState) -> Vec<(String, ThoughtState)> {
        self.operators
            .iter()
            .map(|(id, op)| (id.clone(), op.apply(thought)))
            .collect()
    }

    /// Apply weighted selection of operators
    pub fn generate_weighted_candidates(
        &self,
        thought: &ThoughtState,
        count: usize,
    ) -> Vec<(String, ThoughtState)> {
        let mut rng = rand::thread_rng();
        
        // Calculate total weight
        let total_weight: f64 = self.operators.values().map(|op| op.weight).sum();
        
        let mut candidates = Vec::new();
        
        for _ in 0..count {
            // Weighted random selection
            let mut threshold = rng.gen::<f64>() * total_weight;
            
            for (id, op) in &self.operators {
                threshold -= op.weight;
                if threshold <= 0.0 {
                    candidates.push((id.clone(), op.apply(thought)));
                    break;
                }
            }
        }
        
        candidates
    }

    /// Update operator weights based on feedback
    pub fn update_weights(&mut self, operator_id: &str, reward: f64, learning_rate: f64) {
        if let Some(op) = self.operators.get_mut(operator_id) {
            op.update_weight(reward, learning_rate);
        }
    }

    /// Get operator statistics
    pub fn get_statistics(&self) -> Vec<OperatorStats> {
        self.operators
            .values()
            .map(|op| OperatorStats {
                id: op.id.clone(),
                operator_type: op.operator_type,
                weight: op.weight,
                success_rate: op.success_rate(),
                usage_count: op.usage_count,
            })
            .collect()
    }
}

/// Statistics for an operator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorStats {
    pub id: String,
    pub operator_type: OperatorType,
    pub weight: f64,
    pub success_rate: f64,
    pub usage_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_creation() {
        let op = ReasoningOperator::new(OperatorType::Logical, 64);
        assert_eq!(op.dimension, 64);
        assert!((op.weight - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_operator_application() {
        let op = ReasoningOperator::new(OperatorType::Logical, 64);
        let thought = ThoughtState::from_input("test input", 64);
        
        let result = op.apply(&thought);
        assert_eq!(result.dimension, 64);
        assert!(result.metadata.operator_id.is_some());
    }

    #[test]
    fn test_operator_manager() {
        let manager = OperatorManager::new(64);
        assert_eq!(manager.operators.len(), OperatorType::all().len());
    }

    #[test]
    fn test_candidate_generation() {
        let manager = OperatorManager::new(64);
        let thought = ThoughtState::from_input("test", 64);
        
        let candidates = manager.generate_candidates(&thought);
        assert_eq!(candidates.len(), OperatorType::all().len());
    }
}
