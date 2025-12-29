//! Meta-Learning Optimizer
//!
//! Learning how to learn - adapts optimization strategy based on task:
//!
//! Mathematical Foundation:
//! Meta-objective: min_θ 𝔼_τ~p(τ) [L_τ(f_θ)]
//! MAML: θ' = θ - α∇_θ L_τ(f_θ)
//!       θ ← θ - β∇_θ L_τ(f_θ')
//!
//! Learned Optimizer:
//! g_t = ∇_θ L(θ_t)
//! θ_{t+1} = θ_t + Δθ_t
//! Δθ_t = f_φ(g_t, θ_t, m_t)  // learned update rule

use super::tensor::Tensor;
use std::collections::HashMap;

// ============================================================================
// PART 1: MAML (Model-Agnostic Meta-Learning)
// ============================================================================

pub struct MAML {
    /// Inner loop learning rate (α)
    inner_lr: f32,
    
    /// Outer loop learning rate (β)
    outer_lr: f32,
    
    /// Number of inner loop steps
    inner_steps: usize,
    
    /// First-order approximation (faster, less accurate)
    first_order: bool,
}

impl MAML {
    pub fn new(inner_lr: f32, outer_lr: f32, inner_steps: usize, first_order: bool) -> Self {
        Self {
            inner_lr,
            outer_lr,
            inner_steps,
            first_order,
        }
    }
    
    /// Meta-training step
    /// tasks: batch of tasks, each with support and query sets
    pub fn meta_train_step(&self, tasks: &[Task]) -> MetaTrainMetrics {
        let mut total_loss = 0.0;
        let mut task_losses = Vec::new();
        
        for task in tasks {
            // Inner loop: adapt to task
            let adapted_params = self.inner_loop_adapt(task);
            
            // Outer loop: evaluate on query set
            let query_loss = self.evaluate_on_query(task, &adapted_params);
            
            total_loss += query_loss;
            task_losses.push(query_loss);
        }
        
        MetaTrainMetrics {
            meta_loss: total_loss / tasks.len() as f32,
            task_losses,
            num_tasks: tasks.len(),
        }
    }
    
    /// Inner loop adaptation
    fn inner_loop_adapt(&self, task: &Task) -> HashMap<String, Tensor> {
        let mut params = task.initial_params.clone();
        
        for _ in 0..self.inner_steps {
            // Compute loss on support set
            let loss = self.compute_loss(&params, &task.support_set);
            
            // Compute gradients
            let grads = self.compute_gradients(&params, loss);
            
            // Update parameters: θ' = θ - α∇L
            for (name, param) in params.iter_mut() {
                if let Some(grad) = grads.get(name) {
                    *param = param.sub(&grad.mul_scalar(self.inner_lr));
                }
            }
        }
        
        params
    }
    
    /// Evaluate on query set
    fn evaluate_on_query(&self, task: &Task, params: &HashMap<String, Tensor>) -> f32 {
        self.compute_loss(params, &task.query_set)
    }
    
    /// Compute loss (placeholder - would use actual model)
    fn compute_loss(&self, params: &HashMap<String, Tensor>, data: &DataSet) -> f32 {
        // Simplified: MSE between predictions and targets
        let mut total_loss = 0.0;
        
        for (input, target) in data.samples.iter() {
            // Forward pass (simplified)
            let pred = self.forward(params, input);
            let loss = pred.sub(target).pow(2.0).mean();
            total_loss += loss.item();
        }
        
        total_loss / data.samples.len() as f32
    }
    
    /// Forward pass (placeholder)
    fn forward(&self, params: &HashMap<String, Tensor>, input: &Tensor) -> Tensor {
        // Simplified linear model
        if let Some(weight) = params.get("weight") {
            input.matmul(weight)
        } else {
            input.clone()
        }
    }
    
    /// Compute gradients (placeholder - would use autograd)
    fn compute_gradients(&self, params: &HashMap<String, Tensor>, loss: f32) -> HashMap<String, Tensor> {
        let mut grads = HashMap::new();
        
        for (name, param) in params {
            // Simplified: random gradient
            let grad = Tensor::randn(param.shape()) * loss;
            grads.insert(name.clone(), grad);
        }
        
        grads
    }
}

#[derive(Debug, Clone)]
pub struct Task {
    pub initial_params: HashMap<String, Tensor>,
    pub support_set: DataSet,
    pub query_set: DataSet,
}

#[derive(Debug, Clone)]
pub struct DataSet {
    pub samples: Vec<(Tensor, Tensor)>,
}

#[derive(Debug, Clone)]
pub struct MetaTrainMetrics {
    pub meta_loss: f32,
    pub task_losses: Vec<f32>,
    pub num_tasks: usize,
}

// ============================================================================
// PART 2: LEARNED OPTIMIZER
// ============================================================================

pub struct LearnedOptimizer {
    /// Optimizer parameters
    optimizer_params: HashMap<String, Tensor>,
    
    /// Hidden state for recurrent optimizer
    hidden_state: Option<Tensor>,
    
    /// Learning rate
    learning_rate: f32,
}

impl LearnedOptimizer {
    pub fn new(param_dim: usize, hidden_dim: usize, learning_rate: f32) -> Self {
        let mut optimizer_params = HashMap::new();
        
        // Input projection: [gradient, parameter, momentum] -> hidden
        optimizer_params.insert(
            "input_proj".to_string(),
            Tensor::randn(&[param_dim * 3, hidden_dim]) * 0.01,
        );
        
        // Recurrent weights
        optimizer_params.insert(
            "recurrent".to_string(),
            Tensor::randn(&[hidden_dim, hidden_dim]) * 0.01,
        );
        
        // Output projection: hidden -> update
        optimizer_params.insert(
            "output_proj".to_string(),
            Tensor::randn(&[hidden_dim, param_dim]) * 0.01,
        );
        
        Self {
            optimizer_params,
            hidden_state: None,
            learning_rate,
        }
    }
    
    /// Compute parameter update
    /// Δθ = f_φ(g, θ, m)
    pub fn compute_update(
        &mut self,
        gradient: &Tensor,
        parameter: &Tensor,
        momentum: &Tensor,
    ) -> Tensor {
        // Concatenate inputs
        let input = gradient.concat(parameter, 1).concat(momentum, 1);
        
        // Project to hidden
        let input_proj = self.optimizer_params.get("input_proj").unwrap();
        let mut hidden = input.matmul(input_proj);
        
        // Add recurrent connection
        if let Some(prev_hidden) = &self.hidden_state {
            let recurrent = self.optimizer_params.get("recurrent").unwrap();
            let recurrent_contrib = prev_hidden.matmul(recurrent);
            hidden = hidden.add(&recurrent_contrib);
        }
        
        // Apply activation
        hidden = hidden.tanh();
        
        // Store hidden state
        self.hidden_state = Some(hidden.clone());
        
        // Project to update
        let output_proj = self.optimizer_params.get("output_proj").unwrap();
        let update = hidden.matmul(output_proj);
        
        update.mul_scalar(self.learning_rate)
    }
    
    /// Reset hidden state
    pub fn reset(&mut self) {
        self.hidden_state = None;
    }
}

// ============================================================================
// PART 3: ADAPTIVE LEARNING RATE
// ============================================================================

pub struct AdaptiveLearningRate {
    /// Base learning rate
    base_lr: f32,
    
    /// Per-parameter learning rates
    param_lrs: HashMap<String, f32>,
    
    /// Gradient statistics
    grad_stats: HashMap<String, GradientStats>,
}

impl AdaptiveLearningRate {
    pub fn new(base_lr: f32) -> Self {
        Self {
            base_lr,
            param_lrs: HashMap::new(),
            grad_stats: HashMap::new(),
        }
    }
    
    /// Update learning rate based on gradient statistics
    pub fn update_lr(&mut self, param_name: &str, gradient: &Tensor) {
        let stats = self.grad_stats
            .entry(param_name.to_string())
            .or_insert_with(GradientStats::new);
        
        stats.update(gradient);
        
        // Adapt learning rate based on gradient variance
        let adapted_lr = self.base_lr / (1.0 + stats.variance.sqrt());
        self.param_lrs.insert(param_name.to_string(), adapted_lr);
    }
    
    /// Get learning rate for parameter
    pub fn get_lr(&self, param_name: &str) -> f32 {
        self.param_lrs.get(param_name).copied().unwrap_or(self.base_lr)
    }
}

#[derive(Debug, Clone)]
struct GradientStats {
    mean: f32,
    variance: f32,
    count: usize,
}

impl GradientStats {
    fn new() -> Self {
        Self {
            mean: 0.0,
            variance: 0.0,
            count: 0,
        }
    }
    
    fn update(&mut self, gradient: &Tensor) {
        let grad_mean = gradient.mean().item();
        let grad_var = gradient.var().item();
        
        self.count += 1;
        let alpha = 1.0 / self.count as f32;
        
        // Update running statistics
        self.mean = (1.0 - alpha) * self.mean + alpha * grad_mean;
        self.variance = (1.0 - alpha) * self.variance + alpha * grad_var;
    }
}

// ============================================================================
// PART 4: CURRICULUM LEARNING
// ============================================================================

pub struct CurriculumLearning {
    /// Current difficulty level
    difficulty: f32,
    
    /// Difficulty increase rate
    increase_rate: f32,
    
    /// Performance threshold for difficulty increase
    performance_threshold: f32,
    
    /// Recent performance history
    performance_history: Vec<f32>,
    
    /// History window size
    window_size: usize,
}

impl CurriculumLearning {
    pub fn new(
        initial_difficulty: f32,
        increase_rate: f32,
        performance_threshold: f32,
        window_size: usize,
    ) -> Self {
        Self {
            difficulty: initial_difficulty,
            increase_rate,
            performance_threshold,
            performance_history: Vec::new(),
            window_size,
        }
    }
    
    /// Update curriculum based on performance
    pub fn update(&mut self, performance: f32) {
        self.performance_history.push(performance);
        
        if self.performance_history.len() > self.window_size {
            self.performance_history.remove(0);
        }
        
        // Check if ready to increase difficulty
        if self.should_increase_difficulty() {
            self.difficulty = (self.difficulty + self.increase_rate).min(1.0);
        }
    }
    
    fn should_increase_difficulty(&self) -> bool {
        if self.performance_history.len() < self.window_size {
            return false;
        }
        
        let avg_performance: f32 = self.performance_history.iter().sum::<f32>()
            / self.performance_history.len() as f32;
        
        avg_performance >= self.performance_threshold
    }
    
    /// Get current difficulty
    pub fn get_difficulty(&self) -> f32 {
        self.difficulty
    }
    
    /// Sample task based on difficulty
    pub fn sample_task(&self, tasks: &[Task]) -> Option<&Task> {
        if tasks.is_empty() {
            return None;
        }
        
        // Filter tasks by difficulty
        let suitable_tasks: Vec<&Task> = tasks
            .iter()
            .filter(|task| {
                let task_difficulty = self.estimate_task_difficulty(task);
                (task_difficulty - self.difficulty).abs() < 0.2
            })
            .collect();
        
        if suitable_tasks.is_empty() {
            tasks.first()
        } else {
            let idx = rand::random::<usize>() % suitable_tasks.len();
            Some(suitable_tasks[idx])
        }
    }
    
    fn estimate_task_difficulty(&self, task: &Task) -> f32 {
        // Simplified: based on support set size
        let support_size = task.support_set.samples.len();
        (support_size as f32 / 100.0).min(1.0)
    }
}

// ============================================================================
// PART 5: META-LEARNING CONTROLLER
// ============================================================================

pub struct MetaLearningController {
    /// MAML optimizer
    maml: MAML,
    
    /// Learned optimizer
    learned_optimizer: LearnedOptimizer,
    
    /// Adaptive learning rate
    adaptive_lr: AdaptiveLearningRate,
    
    /// Curriculum learning
    curriculum: CurriculumLearning,
    
    /// Optimization mode
    mode: OptimizationMode,
}

impl MetaLearningController {
    pub fn new(
        inner_lr: f32,
        outer_lr: f32,
        inner_steps: usize,
        param_dim: usize,
        hidden_dim: usize,
        base_lr: f32,
    ) -> Self {
        Self {
            maml: MAML::new(inner_lr, outer_lr, inner_steps, false),
            learned_optimizer: LearnedOptimizer::new(param_dim, hidden_dim, base_lr),
            adaptive_lr: AdaptiveLearningRate::new(base_lr),
            curriculum: CurriculumLearning::new(0.1, 0.05, 0.8, 10),
            mode: OptimizationMode::MAML,
        }
    }
    
    /// Meta-train on batch of tasks
    pub fn meta_train(&mut self, tasks: &[Task]) -> MetaTrainMetrics {
        match self.mode {
            OptimizationMode::MAML => self.maml.meta_train_step(tasks),
            OptimizationMode::Learned => {
                // Use learned optimizer
                self.learned_optimizer_train(tasks)
            }
            OptimizationMode::Hybrid => {
                // Combine both approaches
                let maml_metrics = self.maml.meta_train_step(tasks);
                let learned_metrics = self.learned_optimizer_train(tasks);
                
                // Average metrics
                MetaTrainMetrics {
                    meta_loss: (maml_metrics.meta_loss + learned_metrics.meta_loss) / 2.0,
                    task_losses: maml_metrics.task_losses,
                    num_tasks: tasks.len(),
                }
            }
        }
    }
    
    fn learned_optimizer_train(&mut self, tasks: &[Task]) -> MetaTrainMetrics {
        let mut total_loss = 0.0;
        let mut task_losses = Vec::new();
        
        for task in tasks {
            // Simplified training with learned optimizer
            let loss = self.train_with_learned_optimizer(task);
            total_loss += loss;
            task_losses.push(loss);
        }
        
        MetaTrainMetrics {
            meta_loss: total_loss / tasks.len() as f32,
            task_losses,
            num_tasks: tasks.len(),
        }
    }
    
    fn train_with_learned_optimizer(&mut self, task: &Task) -> f32 {
        // Simplified: compute loss on support set
        let mut params = task.initial_params.clone();
        let mut momentum = HashMap::new();
        
        for (name, param) in params.iter() {
            momentum.insert(name.clone(), Tensor::zeros(param.shape()));
        }
        
        // Training steps
        for _ in 0..10 {
            for (name, param) in params.iter_mut() {
                // Compute gradient (simplified)
                let grad = Tensor::randn(param.shape()) * 0.01;
                
                // Get momentum
                let mom = momentum.get(name).unwrap();
                
                // Compute update using learned optimizer
                let update = self.learned_optimizer.compute_update(&grad, param, mom);
                
                // Apply update
                *param = param.add(&update);
                
                // Update momentum
                momentum.insert(name.clone(), mom.mul_scalar(0.9).add(&grad.mul_scalar(0.1)));
            }
        }
        
        // Evaluate on query set
        self.maml.evaluate_on_query(task, &params)
    }
    
    /// Set optimization mode
    pub fn set_mode(&mut self, mode: OptimizationMode) {
        self.mode = mode;
    }
    
    /// Update curriculum
    pub fn update_curriculum(&mut self, performance: f32) {
        self.curriculum.update(performance);
    }
    
    /// Get current difficulty
    pub fn get_difficulty(&self) -> f32 {
        self.curriculum.get_difficulty()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizationMode {
    MAML,
    Learned,
    Hybrid,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_maml() {
        let maml = MAML::new(0.01, 0.001, 5, false);
        
        let task = Task {
            initial_params: {
                let mut params = HashMap::new();
                params.insert("weight".to_string(), Tensor::randn(&[10, 10]));
                params
            },
            support_set: DataSet {
                samples: vec![
                    (Tensor::randn(&[1, 10]), Tensor::randn(&[1, 10])),
                ],
            },
            query_set: DataSet {
                samples: vec![
                    (Tensor::randn(&[1, 10]), Tensor::randn(&[1, 10])),
                ],
            },
        };
        
        let metrics = maml.meta_train_step(&[task]);
        assert!(metrics.meta_loss.is_finite());
    }
    
    #[test]
    fn test_curriculum_learning() {
        let mut curriculum = CurriculumLearning::new(0.1, 0.05, 0.8, 5);
        
        // Simulate good performance
        for _ in 0..10 {
            curriculum.update(0.9);
        }
        
        assert!(curriculum.get_difficulty() > 0.1);
    }
}
