//! Meta-Learning Optimizer
//!
//! Implements "learning how to learn" through:
//! - MAML-style meta-learning
//! - Operator selection optimization
//! - Adaptive strategy selection
//!
//! Mathematical Foundation:
//! θ* = arg min_θ 𝔼_{T ~ 𝒯} [L_T(θ)]
//! Inner loop: θ_i' = θ - α ∇_θ L_{T_i}(θ)
//! Outer loop: θ ← θ - β ∇_θ L_{T_i}(θ_i')

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// PART 1: TASK REPRESENTATION
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub domain: String,
    pub difficulty: f64,
    pub problem: String,
    pub solution: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDistribution {
    pub domain: String,
    pub tasks: Vec<Task>,
    pub difficulty_range: (f64, f64),
}

// ============================================================================
// PART 2: META-PARAMETERS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaParameters {
    /// Operator selection weights
    pub operator_weights: HashMap<String, f64>,
    
    /// Learning rate per domain
    pub domain_learning_rates: HashMap<String, f64>,
    
    /// Verification threshold per domain
    pub domain_thresholds: HashMap<String, f64>,
    
    /// Exploration vs exploitation balance
    pub exploration_rate: f64,
}

impl Default for MetaParameters {
    fn default() -> Self {
        Self {
            operator_weights: HashMap::new(),
            domain_learning_rates: HashMap::new(),
            domain_thresholds: HashMap::new(),
            exploration_rate: 0.1,
        }
    }
}

// ============================================================================
// PART 3: ADAPTATION RECORD
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationRecord {
    pub task_id: String,
    pub domain: String,
    pub initial_loss: f64,
    pub adapted_loss: f64,
    pub improvement: f64,
    pub steps_taken: usize,
    pub timestamp: u64,
}

// ============================================================================
// PART 4: META-LEARNING OPTIMIZER
// ============================================================================

pub struct MetaLearningOptimizer {
    /// Meta-parameters
    pub meta_params: MetaParameters,
    
    /// Inner loop learning rate (α)
    pub inner_lr: f64,
    
    /// Outer loop learning rate (β)
    pub outer_lr: f64,
    
    /// Adaptation history
    pub adaptation_history: Vec<AdaptationRecord>,
    
    /// Task distributions
    pub task_distributions: HashMap<String, TaskDistribution>,
    
    /// Operator performance tracking
    pub operator_performance: HashMap<String, OperatorPerformance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorPerformance {
    pub operator_id: String,
    pub success_count: usize,
    pub failure_count: usize,
    pub avg_confidence: f64,
    pub avg_energy: f64,
    pub domain_performance: HashMap<String, DomainPerformance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainPerformance {
    pub success_count: usize,
    pub total_count: usize,
    pub success_rate: f64,
}

impl MetaLearningOptimizer {
    pub fn new(inner_lr: f64, outer_lr: f64) -> Self {
        Self {
            meta_params: MetaParameters::default(),
            inner_lr,
            outer_lr,
            adaptation_history: Vec::new(),
            task_distributions: HashMap::new(),
            operator_performance: HashMap::new(),
        }
    }
    
    /// Meta-update from a batch of tasks
    pub fn meta_update(&mut self, task_batch: &[Task]) -> MetaUpdateResult {
        let mut total_improvement = 0.0;
        let mut successful_adaptations = 0;
        
        for task in task_batch {
            // Inner loop: adapt to task
            let adaptation = self.adapt_to_task(task);
            
            if adaptation.improvement > 0.0 {
                successful_adaptations += 1;
                total_improvement += adaptation.improvement;
            }
            
            // Store adaptation record
            self.adaptation_history.push(adaptation);
        }
        
        // Outer loop: update meta-parameters based on adaptations
        self.update_meta_parameters();
        
        MetaUpdateResult {
            tasks_processed: task_batch.len(),
            successful_adaptations,
            avg_improvement: total_improvement / task_batch.len() as f64,
            meta_params: self.meta_params.clone(),
        }
    }
    
    /// Adapt to a single task (inner loop)
    fn adapt_to_task(&mut self, task: &Task) -> AdaptationRecord {
        let initial_loss = self.compute_task_loss(task);
        
        // Simulate adaptation steps
        let mut current_loss = initial_loss;
        let mut steps = 0;
        let max_steps = 10;
        
        while steps < max_steps && current_loss > 0.1 {
            // Gradient descent step (simplified)
            let gradient = self.compute_task_gradient(task, current_loss);
            current_loss -= self.inner_lr * gradient;
            steps += 1;
        }
        
        let adapted_loss = current_loss.max(0.0);
        let improvement = initial_loss - adapted_loss;
        
        AdaptationRecord {
            task_id: task.id.clone(),
            domain: task.domain.clone(),
            initial_loss,
            adapted_loss,
            improvement,
            steps_taken: steps,
            timestamp: Self::current_timestamp(),
        }
    }
    
    /// Update meta-parameters based on adaptation history
    fn update_meta_parameters(&mut self) {
        // Update domain learning rates based on adaptation success
        let mut domain_improvements: HashMap<String, Vec<f64>> = HashMap::new();
        
        for record in &self.adaptation_history {
            domain_improvements
                .entry(record.domain.clone())
                .or_default()
                .push(record.improvement);
        }
        
        for (domain, improvements) in domain_improvements {
            let avg_improvement = improvements.iter().sum::<f64>() / improvements.len() as f64;
            
            // Adjust learning rate based on improvement
            let current_lr = self.meta_params.domain_learning_rates
                .get(&domain)
                .copied()
                .unwrap_or(0.01);
            
            let new_lr = if avg_improvement > 0.1 {
                current_lr * 1.1 // Increase if improving well
            } else if avg_improvement < 0.01 {
                current_lr * 0.9 // Decrease if not improving
            } else {
                current_lr
            };
            
            self.meta_params.domain_learning_rates.insert(domain, new_lr.clamp(0.001, 0.1));
        }
    }
    
    /// Compute loss for a task
    fn compute_task_loss(&self, task: &Task) -> f64 {
        // Simplified loss computation
        // In practice, this would involve actual problem solving
        if task.solution.is_some() {
            0.5 // Has solution, moderate loss
        } else {
            1.0 // No solution, high loss
        }
    }
    
    /// Compute gradient for a task
    fn compute_task_gradient(&self, task: &Task, current_loss: f64) -> f64 {
        // Simplified gradient
        current_loss * 0.1
    }
    
    /// Select best operator for a task
    pub fn select_operator(&self, task: &Task) -> String {
        // Get operator weights for this domain
        let domain_operators: Vec<_> = self.operator_performance
            .iter()
            .filter_map(|(op_id, perf)| {
                perf.domain_performance.get(&task.domain).map(|dp| {
                    let weight = self.meta_params.operator_weights
                        .get(op_id)
                        .copied()
                        .unwrap_or(1.0);
                    (op_id.clone(), weight * dp.success_rate)
                })
            })
            .collect();
        
        if domain_operators.is_empty() {
            // No history, return default
            return "default".to_string();
        }
        
        // Epsilon-greedy selection
        if rand::random::<f64>() < self.meta_params.exploration_rate {
            // Explore: random operator
            let idx = rand::random::<usize>() % domain_operators.len();
            domain_operators[idx].0.clone()
        } else {
            // Exploit: best operator
            domain_operators
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(id, _)| id.clone())
                .unwrap_or_else(|| "default".to_string())
        }
    }
    
    /// Update operator performance
    pub fn update_operator_performance(
        &mut self,
        operator_id: &str,
        domain: &str,
        success: bool,
        confidence: f64,
        energy: f64,
    ) {
        let perf = self.operator_performance
            .entry(operator_id.to_string())
            .or_insert_with(|| OperatorPerformance {
                operator_id: operator_id.to_string(),
                success_count: 0,
                failure_count: 0,
                avg_confidence: 0.0,
                avg_energy: 0.0,
                domain_performance: HashMap::new(),
            });
        
        if success {
            perf.success_count += 1;
        } else {
            perf.failure_count += 1;
        }
        
        // Update averages
        let total = (perf.success_count + perf.failure_count) as f64;
        perf.avg_confidence = (perf.avg_confidence * (total - 1.0) + confidence) / total;
        perf.avg_energy = (perf.avg_energy * (total - 1.0) + energy) / total;
        
        // Update domain performance
        let domain_perf = perf.domain_performance
            .entry(domain.to_string())
            .or_insert_with(|| DomainPerformance {
                success_count: 0,
                total_count: 0,
                success_rate: 0.0,
            });
        
        domain_perf.total_count += 1;
        if success {
            domain_perf.success_count += 1;
        }
        domain_perf.success_rate = domain_perf.success_count as f64 / domain_perf.total_count as f64;
    }
    
    /// Get meta-learning statistics
    pub fn get_stats(&self) -> MetaLearningStats {
        let total_adaptations = self.adaptation_history.len();
        let successful = self.adaptation_history.iter()
            .filter(|a| a.improvement > 0.0)
            .count();
        
        let avg_improvement = if total_adaptations > 0 {
            self.adaptation_history.iter()
                .map(|a| a.improvement)
                .sum::<f64>() / total_adaptations as f64
        } else {
            0.0
        };
        
        MetaLearningStats {
            total_adaptations,
            successful_adaptations: successful,
            success_rate: successful as f64 / total_adaptations.max(1) as f64,
            avg_improvement,
            meta_params: self.meta_params.clone(),
            operator_performance: self.operator_performance.clone(),
        }
    }
    
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

impl Default for MetaLearningOptimizer {
    fn default() -> Self {
        Self::new(0.01, 0.001)
    }
}

// ============================================================================
// PART 5: RESULTS AND STATISTICS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaUpdateResult {
    pub tasks_processed: usize,
    pub successful_adaptations: usize,
    pub avg_improvement: f64,
    pub meta_params: MetaParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningStats {
    pub total_adaptations: usize,
    pub successful_adaptations: usize,
    pub success_rate: f64,
    pub avg_improvement: f64,
    pub meta_params: MetaParameters,
    pub operator_performance: HashMap<String, OperatorPerformance>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_meta_optimizer() {
        let mut optimizer = MetaLearningOptimizer::new(0.01, 0.001);
        
        let tasks = vec![
            Task {
                id: "task1".to_string(),
                domain: "math".to_string(),
                difficulty: 0.5,
                problem: "2+2".to_string(),
                solution: Some("4".to_string()),
            },
        ];
        
        let result = optimizer.meta_update(&tasks);
        assert_eq!(result.tasks_processed, 1);
    }
    
    #[test]
    fn test_operator_selection() {
        let mut optimizer = MetaLearningOptimizer::new(0.01, 0.001);
        
        // Add some operator performance data
        optimizer.update_operator_performance("op1", "math", true, 0.9, 0.2);
        optimizer.update_operator_performance("op1", "math", true, 0.8, 0.3);
        optimizer.update_operator_performance("op2", "math", false, 0.5, 0.7);
        
        let task = Task {
            id: "test".to_string(),
            domain: "math".to_string(),
            difficulty: 0.5,
            problem: "test".to_string(),
            solution: None,
        };
        
        let selected = optimizer.select_operator(&task);
        assert!(!selected.is_empty());
    }
}
