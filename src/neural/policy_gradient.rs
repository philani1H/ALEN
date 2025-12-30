//! Policy Gradient Training
//!
//! For discrete outputs (code, formulas, symbolic expressions):
//!
//! Mathematical Foundation:
//! Policy: π_θ(y|x) = P(y|x; θ)
//! Objective: J(θ) = 𝔼_{y~π_θ}[R(y)]
//! Gradient: ∇_θ J(θ) = 𝔼_{y~π_θ}[R(y) ∇_θ log π_θ(y|x)]
//!
//! REINFORCE Algorithm:
//! 1. Sample y ~ π_θ(·|x)
//! 2. Compute reward R(y)
//! 3. Update: θ ← θ + α R(y) ∇_θ log π_θ(y|x)
//!
//! Variance Reduction:
//! - Baseline: b(x) = 𝔼[R(y)]
//! - Advantage: A(y) = R(y) - b(x)
//! - Update: θ ← θ + α A(y) ∇_θ log π_θ(y|x)

use super::tensor::Tensor;
use std::collections::VecDeque;

// ============================================================================
// PART 1: POLICY NETWORK
// ============================================================================

pub struct PolicyNetwork {
    /// Action space size (vocabulary size for discrete actions)
    action_space_size: usize,
    
    /// Temperature for sampling
    temperature: f32,
    
    /// Baseline for variance reduction
    baseline: ExponentialMovingAverage,
}

impl PolicyNetwork {
    pub fn new(action_space_size: usize, temperature: f32) -> Self {
        Self {
            action_space_size,
            temperature,
            baseline: ExponentialMovingAverage::new(0.99),
        }
    }
    
    /// Sample action from policy
    /// logits: [batch_size, action_space_size]
    /// Returns: (actions, log_probs)
    pub fn sample(&self, logits: &Tensor) -> (Vec<usize>, Vec<f32>) {
        let batch_size = logits.shape()[0];
        let mut actions = Vec::new();
        let mut log_probs = Vec::new();
        
        for b in 0..batch_size {
            // Get logits for this batch
            let batch_logits = logits.slice(b, 0);
            
            // Apply temperature
            let scaled_logits: Vec<f32> = batch_logits
                .iter()
                .map(|&l| l / self.temperature)
                .collect();
            
            // Softmax
            let probs = self.softmax(&scaled_logits);
            
            // Sample
            let action = self.categorical_sample(&probs);
            let log_prob = probs[action].ln();
            
            actions.push(action);
            log_probs.push(log_prob);
        }
        
        (actions, log_probs)
    }
    
    /// Compute policy gradient loss
    /// log_probs: log π_θ(y|x)
    /// rewards: R(y)
    /// Returns: -𝔼[A(y) log π_θ(y|x)]
    pub fn compute_loss(&mut self, log_probs: &[f32], rewards: &[f32]) -> f32 {
        assert_eq!(log_probs.len(), rewards.len());
        
        let mut total_loss = 0.0;
        
        for (&log_prob, &reward) in log_probs.iter().zip(rewards.iter()) {
            // Update baseline
            self.baseline.update(reward);
            
            // Compute advantage
            let advantage = reward - self.baseline.get();
            
            // Policy gradient: -A(y) log π_θ(y|x)
            total_loss -= advantage * log_prob;
        }
        
        total_loss / log_probs.len() as f32
    }
    
    /// Softmax
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = exp_logits.iter().sum();
        
        exp_logits.iter().map(|&e| e / sum).collect()
    }
    
    /// Categorical sampling
    fn categorical_sample(&self, probs: &[f32]) -> usize {
        let mut rng = rand::thread_rng();
        let u: f32 = rand::Rng::gen(&mut rng);
        
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if u < cumsum {
                return i;
            }
        }
        
        probs.len() - 1
    }
}

// ============================================================================
// PART 2: EXPONENTIAL MOVING AVERAGE (BASELINE)
// ============================================================================

struct ExponentialMovingAverage {
    value: f32,
    alpha: f32,
    initialized: bool,
}

impl ExponentialMovingAverage {
    fn new(alpha: f32) -> Self {
        Self {
            value: 0.0,
            alpha,
            initialized: false,
        }
    }
    
    fn update(&mut self, new_value: f32) {
        if !self.initialized {
            self.value = new_value;
            self.initialized = true;
        } else {
            self.value = self.alpha * self.value + (1.0 - self.alpha) * new_value;
        }
    }
    
    fn get(&self) -> f32 {
        self.value
    }
}

// ============================================================================
// PART 3: ACTOR-CRITIC
// ============================================================================

pub struct ActorCritic {
    /// Policy network (actor)
    policy: PolicyNetwork,
    
    /// Value network (critic) - estimates V(x)
    value_baseline: ExponentialMovingAverage,
    
    /// Discount factor
    gamma: f32,
}

impl ActorCritic {
    pub fn new(action_space_size: usize, temperature: f32, gamma: f32) -> Self {
        Self {
            policy: PolicyNetwork::new(action_space_size, temperature),
            value_baseline: ExponentialMovingAverage::new(0.99),
            gamma,
        }
    }
    
    /// Compute actor-critic loss
    /// trajectory: sequence of (log_prob, reward) pairs
    pub fn compute_loss(&mut self, trajectory: &[(f32, f32)]) -> (f32, f32) {
        let mut actor_loss = 0.0;
        let mut critic_loss = 0.0;
        
        // Compute returns (discounted cumulative rewards)
        let returns = self.compute_returns(trajectory);
        
        for (i, &(log_prob, _)) in trajectory.iter().enumerate() {
            let return_val = returns[i];
            
            // Update value baseline
            self.value_baseline.update(return_val);
            let value_estimate = self.value_baseline.get();
            
            // Advantage
            let advantage = return_val - value_estimate;
            
            // Actor loss: -A(y) log π_θ(y|x)
            actor_loss -= advantage * log_prob;
            
            // Critic loss: (V(x) - G)²
            critic_loss += (value_estimate - return_val).powi(2);
        }
        
        let n = trajectory.len() as f32;
        (actor_loss / n, critic_loss / n)
    }
    
    /// Compute discounted returns
    fn compute_returns(&self, trajectory: &[(f32, f32)]) -> Vec<f32> {
        let mut returns = vec![0.0; trajectory.len()];
        let mut g = 0.0;
        
        for i in (0..trajectory.len()).rev() {
            let (_, reward) = trajectory[i];
            g = reward + self.gamma * g;
            returns[i] = g;
        }
        
        returns
    }
}

// ============================================================================
// PART 4: REWARD FUNCTIONS
// ============================================================================

pub struct RewardFunction;

impl RewardFunction {
    /// Reward for code generation
    pub fn code_reward(
        code: &str,
        compiles: bool,
        passes_tests: bool,
        correctness_score: f32,
    ) -> f32 {
        let mut reward = 0.0;
        
        // Compilation
        if compiles {
            reward += 0.3;
        }
        
        // Tests
        if passes_tests {
            reward += 0.4;
        }
        
        // Correctness
        reward += 0.3 * correctness_score;
        
        // Length penalty (prefer concise code)
        let length_penalty = (code.len() as f32 / 1000.0).min(0.1);
        reward -= length_penalty;
        
        reward
    }
    
    /// Reward for mathematical formula
    pub fn formula_reward(
        formula: &str,
        is_valid: bool,
        simplicity_score: f32,
        correctness_score: f32,
    ) -> f32 {
        let mut reward = 0.0;
        
        // Validity
        if is_valid {
            reward += 0.2;
        }
        
        // Correctness
        reward += 0.6 * correctness_score;
        
        // Simplicity
        reward += 0.2 * simplicity_score;
        
        reward
    }
    
    /// Reward for explanation quality
    pub fn explanation_reward(
        explanation: &str,
        clarity_score: f32,
        completeness_score: f32,
        audience_match_score: f32,
    ) -> f32 {
        let mut reward = 0.0;
        
        // Clarity
        reward += 0.3 * clarity_score;
        
        // Completeness
        reward += 0.3 * completeness_score;
        
        // Audience adaptation
        reward += 0.4 * audience_match_score;
        
        reward
    }
}

// ============================================================================
// PART 5: POLICY GRADIENT TRAINER
// ============================================================================

pub struct PolicyGradientTrainer {
    /// Actor-critic
    actor_critic: ActorCritic,
    
    /// Learning rate
    learning_rate: f32,
    
    /// Trajectory buffer
    trajectory_buffer: VecDeque<(f32, f32)>,
    
    /// Max trajectory length
    max_trajectory_length: usize,
}

impl PolicyGradientTrainer {
    pub fn new(
        action_space_size: usize,
        temperature: f32,
        gamma: f32,
        learning_rate: f32,
        max_trajectory_length: usize,
    ) -> Self {
        Self {
            actor_critic: ActorCritic::new(action_space_size, temperature, gamma),
            learning_rate,
            trajectory_buffer: VecDeque::new(),
            max_trajectory_length,
        }
    }
    
    /// Add experience to trajectory
    pub fn add_experience(&mut self, log_prob: f32, reward: f32) {
        self.trajectory_buffer.push_back((log_prob, reward));
        
        if self.trajectory_buffer.len() > self.max_trajectory_length {
            self.trajectory_buffer.pop_front();
        }
    }
    
    /// Train on accumulated trajectory
    pub fn train(&mut self) -> TrainingMetrics {
        if self.trajectory_buffer.is_empty() {
            return TrainingMetrics::default();
        }
        
        let trajectory: Vec<(f32, f32)> = self.trajectory_buffer.iter().copied().collect();
        let (actor_loss, critic_loss) = self.actor_critic.compute_loss(&trajectory);
        
        // Clear buffer
        self.trajectory_buffer.clear();
        
        TrainingMetrics {
            actor_loss,
            critic_loss,
            total_loss: actor_loss + critic_loss,
            trajectory_length: trajectory.len(),
        }
    }
    
    /// Get current baseline value
    pub fn get_baseline(&self) -> f32 {
        self.actor_critic.value_baseline.get()
    }
}

#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    pub actor_loss: f32,
    pub critic_loss: f32,
    pub total_loss: f32,
    pub trajectory_length: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_policy_network() {
        let policy = PolicyNetwork::new(10, 1.0);
        let logits = Tensor::randn(vec![2, 10]);
        
        let (actions, log_probs) = policy.sample(&logits);
        
        assert_eq!(actions.len(), 2);
        assert_eq!(log_probs.len(), 2);
    }
    
    #[test]
    fn test_actor_critic() {
        let mut ac = ActorCritic::new(10, 1.0, 0.99);
        
        let trajectory = vec![
            (-1.0, 0.5),
            (-1.2, 0.7),
            (-0.8, 0.9),
        ];
        
        let (actor_loss, critic_loss) = ac.compute_loss(&trajectory);
        
        assert!(actor_loss.is_finite());
        assert!(critic_loss.is_finite());
    }
    
    #[test]
    fn test_reward_functions() {
        let code_reward = RewardFunction::code_reward("fn main() {}", true, true, 1.0);
        assert!(code_reward > 0.0);
        
        let formula_reward = RewardFunction::formula_reward("x^2 + 1", true, 0.8, 0.9);
        assert!(formula_reward > 0.0);
        
        let explanation_reward = RewardFunction::explanation_reward(
            "This is a clear explanation",
            0.9,
            0.8,
            0.85,
        );
        assert!(explanation_reward > 0.0);
    }
}
