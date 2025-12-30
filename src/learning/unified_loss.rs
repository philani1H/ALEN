//! Unified Loss Function
//!
//! Implements: L = -log P_θ(Y|Z,u) + λ·KL(q_φ(Z|X) || p(Z)) - γ·log V(X,Y)
//!
//! This combines:
//! 1. Generation loss (negative log-likelihood)
//! 2. KL divergence (latent regularization)
//! 3. Verification reward (correctness incentive)

use serde::{Deserialize, Serialize};

/// Unified loss configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedLossConfig {
    /// Weight for KL divergence term (λ)
    pub lambda: f64,
    /// Weight for verification reward term (γ)
    pub gamma: f64,
    /// Weight for generation loss (typically 1.0)
    pub alpha: f64,
}

impl Default for UnifiedLossConfig {
    fn default() -> Self {
        Self {
            lambda: 0.01,  // Small weight for KL to avoid posterior collapse
            gamma: 1.0,    // Equal weight for verification
            alpha: 1.0,    // Standard weight for generation
        }
    }
}

/// Complete loss breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedLoss {
    /// Generation loss: -log P_θ(Y|Z,u)
    pub generation_loss: f64,
    /// KL divergence: KL(q_φ(Z|X) || p(Z))
    pub kl_divergence: f64,
    /// Verification reward: -log V(X,Y) (negative because we want to maximize V)
    pub verification_loss: f64,
    /// Total loss: α·gen + λ·KL - γ·log(V)
    pub total: f64,
    /// Individual components for monitoring
    pub weighted_generation: f64,
    pub weighted_kl: f64,
    pub weighted_verification: f64,
}

impl UnifiedLoss {
    /// Compute unified loss
    pub fn compute(
        generation_loss: f64,
        kl_divergence: f64,
        verification_score: f64,
        config: &UnifiedLossConfig,
    ) -> Self {
        // Clamp verification score to prevent log(0)
        let v_clamped = verification_score.max(1e-10).min(1.0);
        
        // Verification loss: -log V(X,Y)
        // We want to maximize V, so we minimize -log V
        let verification_loss = -v_clamped.ln();
        
        // Weighted components
        let weighted_generation = config.alpha * generation_loss;
        let weighted_kl = config.lambda * kl_divergence;
        let weighted_verification = config.gamma * verification_loss;
        
        // Total loss
        let total = weighted_generation + weighted_kl + weighted_verification;
        
        Self {
            generation_loss,
            kl_divergence,
            verification_loss,
            total,
            weighted_generation,
            weighted_kl,
            weighted_verification,
        }
    }
    
    /// Compute with default config
    pub fn compute_default(
        generation_loss: f64,
        kl_divergence: f64,
        verification_score: f64,
    ) -> Self {
        Self::compute(
            generation_loss,
            kl_divergence,
            verification_score,
            &UnifiedLossConfig::default(),
        )
    }
    
    /// Get breakdown as string for logging
    pub fn breakdown(&self) -> String {
        format!(
            "Total: {:.4} = Gen: {:.4} + KL: {:.4} + Ver: {:.4}",
            self.total,
            self.weighted_generation,
            self.weighted_kl,
            self.weighted_verification
        )
    }
}

/// Loss computer for training
pub struct UnifiedLossComputer {
    config: UnifiedLossConfig,
}

impl UnifiedLossComputer {
    pub fn new(config: UnifiedLossConfig) -> Self {
        Self { config }
    }
    
    pub fn default() -> Self {
        Self::new(UnifiedLossConfig::default())
    }
    
    /// Compute loss for a training example
    pub fn compute_loss(
        &self,
        predicted: &[f64],
        target: &[f64],
        kl_divergence: f64,
        verification_score: f64,
    ) -> UnifiedLoss {
        // Generation loss: MSE between predicted and target
        let generation_loss = self.mse_loss(predicted, target);
        
        UnifiedLoss::compute(
            generation_loss,
            kl_divergence,
            verification_score,
            &self.config,
        )
    }
    
    /// Mean Squared Error loss
    fn mse_loss(&self, predicted: &[f64], target: &[f64]) -> f64 {
        assert_eq!(predicted.len(), target.len());
        
        let mut sum = 0.0;
        for i in 0..predicted.len() {
            let diff = predicted[i] - target[i];
            sum += diff * diff;
        }
        sum / predicted.len() as f64
    }
    
    /// Cross-entropy loss for classification
    pub fn cross_entropy_loss(&self, predicted: &[f64], target: &[f64]) -> f64 {
        assert_eq!(predicted.len(), target.len());
        
        let mut sum = 0.0;
        for i in 0..predicted.len() {
            let p = predicted[i].max(1e-10).min(1.0 - 1e-10);
            sum -= target[i] * p.ln();
        }
        sum / predicted.len() as f64
    }
}

/// Gradient computation for backpropagation
pub struct UnifiedLossGradient {
    /// Gradient w.r.t. generation parameters
    pub generation_grad: Vec<f64>,
    /// Gradient w.r.t. latent mean
    pub mean_grad: Vec<f64>,
    /// Gradient w.r.t. latent log-variance
    pub logvar_grad: Vec<f64>,
}

impl UnifiedLossGradient {
    /// Compute gradients for backpropagation
    pub fn compute(
        predicted: &[f64],
        target: &[f64],
        mean: &[f64],
        logvar: &[f64],
        config: &UnifiedLossConfig,
    ) -> Self {
        // Generation gradient: ∂L_gen/∂pred = 2(pred - target)/n
        let mut generation_grad = vec![0.0; predicted.len()];
        for i in 0..predicted.len() {
            generation_grad[i] = config.alpha * 2.0 * (predicted[i] - target[i]) / predicted.len() as f64;
        }
        
        // KL gradient w.r.t. mean: ∂KL/∂μ = μ
        let mut mean_grad = vec![0.0; mean.len()];
        for i in 0..mean.len() {
            mean_grad[i] = config.lambda * mean[i];
        }
        
        // KL gradient w.r.t. log-variance: ∂KL/∂log(σ²) = 0.5(exp(log(σ²)) - 1)
        let mut logvar_grad = vec![0.0; logvar.len()];
        for i in 0..logvar.len() {
            logvar_grad[i] = config.lambda * 0.5 * (logvar[i].exp() - 1.0);
        }
        
        Self {
            generation_grad,
            mean_grad,
            logvar_grad,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_unified_loss() {
        let loss = UnifiedLoss::compute_default(
            1.0,  // generation loss
            0.5,  // KL divergence
            0.8,  // verification score
        );
        
        assert!(loss.total > 0.0);
        assert_eq!(loss.generation_loss, 1.0);
        assert_eq!(loss.kl_divergence, 0.5);
        println!("{}", loss.breakdown());
    }
    
    #[test]
    fn test_loss_computer() {
        let computer = UnifiedLossComputer::default();
        
        let predicted = vec![0.5, 0.6, 0.7];
        let target = vec![0.4, 0.5, 0.8];
        
        let loss = computer.compute_loss(&predicted, &target, 0.3, 0.9);
        
        assert!(loss.total > 0.0);
        assert!(loss.generation_loss > 0.0);
    }
    
    #[test]
    fn test_gradient_computation() {
        let predicted = vec![0.5, 0.6];
        let target = vec![0.4, 0.5];
        let mean = vec![0.1, 0.2];
        let logvar = vec![0.0, 0.0];
        
        let grad = UnifiedLossGradient::compute(
            &predicted,
            &target,
            &mean,
            &logvar,
            &UnifiedLossConfig::default(),
        );
        
        assert_eq!(grad.generation_grad.len(), 2);
        assert_eq!(grad.mean_grad.len(), 2);
        assert_eq!(grad.logvar_grad.len(), 2);
    }
}
