//! Gradient-Based Personalization
//!
//! Implements: u_t+1 = u_t + η·∇_u log P_θ(Y_t | Z_t, u_t)
//!
//! User embedding adapts based on gradient of generation likelihood,
//! making responses more personalized over time.

use serde::{Deserialize, Serialize};

/// User embedding for personalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserEmbedding {
    /// Embedding vector u ∈ ℝ^d_u
    pub vector: Vec<f64>,
    /// Dimension
    pub dimension: usize,
    /// Learning rate η
    pub learning_rate: f64,
    /// Momentum for smoothing updates
    pub momentum: f64,
    /// Velocity for momentum
    velocity: Vec<f64>,
}

impl UserEmbedding {
    /// Create new user embedding
    pub fn new(dimension: usize, learning_rate: f64) -> Self {
        Self {
            vector: vec![0.0; dimension],
            dimension,
            learning_rate,
            momentum: 0.9,
            velocity: vec![0.0; dimension],
        }
    }
    
    /// Create with initial values
    pub fn with_initial(initial: Vec<f64>, learning_rate: f64) -> Self {
        let dimension = initial.len();
        Self {
            vector: initial,
            dimension,
            learning_rate,
            momentum: 0.9,
            velocity: vec![0.0; dimension],
        }
    }
    
    /// Update embedding using gradient: u_t+1 = u_t + η·∇_u log P_θ(Y|Z,u)
    pub fn update(&mut self, gradient: &[f64]) {
        assert_eq!(gradient.len(), self.dimension);
        
        for i in 0..self.dimension {
            // Momentum update: v = β·v + (1-β)·∇
            self.velocity[i] = self.momentum * self.velocity[i] + (1.0 - self.momentum) * gradient[i];
            
            // Parameter update: u = u + η·v
            self.vector[i] += self.learning_rate * self.velocity[i];
            
            // Clamp to prevent explosion
            self.vector[i] = self.vector[i].max(-10.0).min(10.0);
        }
    }
    
    /// Compute gradient of log-likelihood w.r.t. user embedding
    /// ∇_u log P_θ(Y|Z,u) ≈ (Y - Ŷ) · ∂Ŷ/∂u
    pub fn compute_gradient(
        &self,
        predicted: &[f64],
        target: &[f64],
        output_weights: &[Vec<f64>],
    ) -> Vec<f64> {
        assert_eq!(predicted.len(), target.len());
        
        let mut gradient = vec![0.0; self.dimension];
        
        // Error signal: δ = Y - Ŷ
        let mut error = vec![0.0; predicted.len()];
        for i in 0..predicted.len() {
            error[i] = target[i] - predicted[i];
        }
        
        // Backpropagate through output layer: ∇_u = Σ_i δ_i · W_i
        for i in 0..self.dimension {
            let mut sum = 0.0;
            for j in 0..predicted.len() {
                if i < output_weights[j].len() {
                    sum += error[j] * output_weights[j][i];
                }
            }
            gradient[i] = sum;
        }
        
        gradient
    }
    
    /// Get current embedding
    pub fn get(&self) -> &[f64] {
        &self.vector
    }
    
    /// Reset to zero
    pub fn reset(&mut self) {
        self.vector.fill(0.0);
        self.velocity.fill(0.0);
    }
    
    /// Decay learning rate over time
    pub fn decay_learning_rate(&mut self, decay_factor: f64) {
        self.learning_rate *= decay_factor;
    }
}

/// Personalization manager for multiple users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationManager {
    /// User embeddings by user ID
    embeddings: std::collections::HashMap<String, UserEmbedding>,
    /// Default embedding dimension
    dimension: usize,
    /// Default learning rate
    learning_rate: f64,
}

impl PersonalizationManager {
    pub fn new(dimension: usize, learning_rate: f64) -> Self {
        Self {
            embeddings: std::collections::HashMap::new(),
            dimension,
            learning_rate,
        }
    }
    
    /// Get or create user embedding
    pub fn get_or_create(&mut self, user_id: &str) -> &mut UserEmbedding {
        self.embeddings
            .entry(user_id.to_string())
            .or_insert_with(|| UserEmbedding::new(self.dimension, self.learning_rate))
    }
    
    /// Update user embedding
    pub fn update_user(
        &mut self,
        user_id: &str,
        predicted: &[f64],
        target: &[f64],
        output_weights: &[Vec<f64>],
    ) {
        let embedding = self.get_or_create(user_id);
        let gradient = embedding.compute_gradient(predicted, target, output_weights);
        embedding.update(&gradient);
    }
    
    /// Get user embedding vector
    pub fn get_user_embedding(&mut self, user_id: &str) -> Vec<f64> {
        self.get_or_create(user_id).get().to_vec()
    }
}

/// Personalization feedback for reinforcement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationFeedback {
    /// User acceptance (0-1)
    pub acceptance: f64,
    /// User engagement (0-1)
    pub engagement: f64,
    /// Explicit rating (0-1)
    pub rating: Option<f64>,
}

impl PersonalizationFeedback {
    /// Compute reward signal for personalization
    pub fn compute_reward(&self) -> f64 {
        let base_reward = 0.5 * self.acceptance + 0.3 * self.engagement;
        if let Some(rating) = self.rating {
            base_reward + 0.2 * rating
        } else {
            base_reward
        }
    }
    
    /// Convert to gradient signal
    pub fn to_gradient(&self, embedding_dim: usize) -> Vec<f64> {
        let reward = self.compute_reward();
        // Simple gradient: scale embedding by reward
        vec![reward; embedding_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_user_embedding() {
        let mut embedding = UserEmbedding::new(10, 0.01);
        
        let gradient = vec![0.1; 10];
        embedding.update(&gradient);
        
        // Should have moved in direction of gradient
        assert!(embedding.vector[0] > 0.0);
    }
    
    #[test]
    fn test_gradient_computation() {
        let embedding = UserEmbedding::new(5, 0.01);
        let predicted = vec![0.5, 0.6];
        let target = vec![0.7, 0.8];
        let weights = vec![vec![0.1; 5], vec![0.2; 5]];
        
        let gradient = embedding.compute_gradient(&predicted, &target, &weights);
        
        assert_eq!(gradient.len(), 5);
        // Gradient should be non-zero when there's error
        assert!(gradient.iter().any(|&g| g != 0.0));
    }
    
    #[test]
    fn test_personalization_manager() {
        let mut manager = PersonalizationManager::new(10, 0.01);
        
        let user1_emb = manager.get_user_embedding("user1");
        assert_eq!(user1_emb.len(), 10);
        
        let predicted = vec![0.5; 10];
        let target = vec![0.6; 10];
        let weights = vec![vec![0.1; 10]; 10];
        
        manager.update_user("user1", &predicted, &target, &weights);
        
        let updated_emb = manager.get_user_embedding("user1");
        // Should have changed
        assert_ne!(user1_emb, updated_emb);
    }
    
    #[test]
    fn test_feedback_reward() {
        let feedback = PersonalizationFeedback {
            acceptance: 0.8,
            engagement: 0.7,
            rating: Some(0.9),
        };
        
        let reward = feedback.compute_reward();
        assert!(reward > 0.0 && reward <= 1.0);
    }
}
