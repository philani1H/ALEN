//! Variational Encoder - Implements q_φ(Z|X)
//!
//! Probabilistic encoding with reparameterization trick for VAE-style learning.
//! This enables sampling from latent distributions and KL regularization.

use crate::neural::tensor::Tensor;
use crate::core::ThoughtState;
use serde::{Deserialize, Serialize};

/// Variational Encoder implementing q_φ(Z|X) ≈ p_θ(Z|X)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationalEncoder {
    /// Input dimension
    input_dim: usize,
    /// Latent dimension
    latent_dim: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Mean network weights (input -> hidden -> latent)
    mean_weights_1: Vec<Vec<f64>>,
    mean_weights_2: Vec<Vec<f64>>,
    /// Log-variance network weights
    logvar_weights_1: Vec<Vec<f64>>,
    logvar_weights_2: Vec<Vec<f64>>,
    /// Biases
    mean_bias_1: Vec<f64>,
    mean_bias_2: Vec<f64>,
    logvar_bias_1: Vec<f64>,
    logvar_bias_2: Vec<f64>,
}

/// Variational encoding result
#[derive(Debug, Clone)]
pub struct VariationalEncoding {
    /// Mean of latent distribution
    pub mean: Vec<f64>,
    /// Log-variance of latent distribution
    pub logvar: Vec<f64>,
    /// Sampled latent vector (z = μ + σ ⊙ ε)
    pub z: Vec<f64>,
    /// KL divergence from prior
    pub kl_divergence: f64,
}

impl VariationalEncoder {
    /// Create new variational encoder
    pub fn new(input_dim: usize, latent_dim: usize, hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Xavier initialization
        let xavier_in = (6.0 / (input_dim + hidden_dim) as f64).sqrt();
        let xavier_hidden = (6.0 / (hidden_dim + latent_dim) as f64).sqrt();
        
        Self {
            input_dim,
            latent_dim,
            hidden_dim,
            mean_weights_1: (0..hidden_dim)
                .map(|_| (0..input_dim).map(|_| rng.gen_range(-xavier_in..xavier_in)).collect())
                .collect(),
            mean_weights_2: (0..latent_dim)
                .map(|_| (0..hidden_dim).map(|_| rng.gen_range(-xavier_hidden..xavier_hidden)).collect())
                .collect(),
            logvar_weights_1: (0..hidden_dim)
                .map(|_| (0..input_dim).map(|_| rng.gen_range(-xavier_in..xavier_in)).collect())
                .collect(),
            logvar_weights_2: (0..latent_dim)
                .map(|_| (0..hidden_dim).map(|_| rng.gen_range(-xavier_hidden..xavier_hidden)).collect())
                .collect(),
            mean_bias_1: vec![0.0; hidden_dim],
            mean_bias_2: vec![0.0; latent_dim],
            logvar_bias_1: vec![0.0; hidden_dim],
            logvar_bias_2: vec![0.0; latent_dim],
        }
    }
    
    /// Encode input into variational latent distribution
    /// Returns: (mean, logvar, sampled_z, kl_divergence)
    pub fn encode(&self, input: &[f64]) -> VariationalEncoding {
        assert_eq!(input.len(), self.input_dim, "Input dimension mismatch");
        
        // Compute mean: μ = f_mean(x)
        let mean = self.compute_mean(input);
        
        // Compute log-variance: log(σ²) = f_logvar(x)
        let logvar = self.compute_logvar(input);
        
        // Sample using reparameterization trick: z = μ + σ ⊙ ε, where ε ~ N(0,I)
        let z = self.reparameterize(&mean, &logvar);
        
        // Compute KL divergence: KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        let kl_divergence = self.compute_kl_divergence(&mean, &logvar);
        
        VariationalEncoding {
            mean,
            logvar,
            z,
            kl_divergence,
        }
    }
    
    /// Compute mean vector μ
    fn compute_mean(&self, input: &[f64]) -> Vec<f64> {
        // Hidden layer: h = ReLU(W1 * x + b1)
        let mut hidden = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut sum = self.mean_bias_1[i];
            for j in 0..self.input_dim {
                sum += self.mean_weights_1[i][j] * input[j];
            }
            hidden[i] = sum.max(0.0); // ReLU
        }
        
        // Output layer: μ = W2 * h + b2
        let mut mean = vec![0.0; self.latent_dim];
        for i in 0..self.latent_dim {
            let mut sum = self.mean_bias_2[i];
            for j in 0..self.hidden_dim {
                sum += self.mean_weights_2[i][j] * hidden[j];
            }
            mean[i] = sum;
        }
        
        mean
    }
    
    /// Compute log-variance vector log(σ²)
    fn compute_logvar(&self, input: &[f64]) -> Vec<f64> {
        // Hidden layer: h = ReLU(W1 * x + b1)
        let mut hidden = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut sum = self.logvar_bias_1[i];
            for j in 0..self.input_dim {
                sum += self.logvar_weights_1[i][j] * input[j];
            }
            hidden[i] = sum.max(0.0); // ReLU
        }
        
        // Output layer: log(σ²) = W2 * h + b2
        let mut logvar = vec![0.0; self.latent_dim];
        for i in 0..self.latent_dim {
            let mut sum = self.logvar_bias_2[i];
            for j in 0..self.hidden_dim {
                sum += self.logvar_weights_2[i][j] * hidden[j];
            }
            // Clamp to prevent numerical instability
            logvar[i] = sum.max(-10.0).min(10.0);
        }
        
        logvar
    }
    
    /// Reparameterization trick: z = μ + σ ⊙ ε, where ε ~ N(0,I)
    fn reparameterize(&self, mean: &[f64], logvar: &[f64]) -> Vec<f64> {
        use rand::Rng;
        use rand_distr::{Normal, Distribution};
        
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        let mut z = vec![0.0; self.latent_dim];
        for i in 0..self.latent_dim {
            let epsilon = normal.sample(&mut rng);
            let std = (logvar[i] / 2.0).exp(); // σ = exp(log(σ²) / 2)
            z[i] = mean[i] + std * epsilon;
        }
        
        z
    }
    
    /// Compute KL divergence: KL(q(z|x) || p(z))
    /// For Gaussian: KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    fn compute_kl_divergence(&self, mean: &[f64], logvar: &[f64]) -> f64 {
        let mut kl = 0.0;
        for i in 0..self.latent_dim {
            let mu = mean[i];
            let logvar_i = logvar[i];
            let var = logvar_i.exp();
            kl += -0.5 * (1.0 + logvar_i - mu * mu - var);
        }
        kl
    }
    
    /// Encode deterministically (use mean without sampling)
    pub fn encode_deterministic(&self, input: &[f64]) -> Vec<f64> {
        self.compute_mean(input)
    }
    
    /// Convert to ThoughtState
    pub fn to_thought_state(&self, encoding: &VariationalEncoding) -> ThoughtState {
        ThoughtState::from_vector(encoding.z.clone(), self.latent_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_variational_encoder() {
        let encoder = VariationalEncoder::new(128, 64, 256);
        let input = vec![0.5; 128];
        
        let encoding = encoder.encode(&input);
        
        assert_eq!(encoding.mean.len(), 64);
        assert_eq!(encoding.logvar.len(), 64);
        assert_eq!(encoding.z.len(), 64);
        assert!(encoding.kl_divergence >= 0.0);
    }
    
    #[test]
    fn test_reparameterization() {
        let encoder = VariationalEncoder::new(10, 5, 20);
        let input = vec![1.0; 10];
        
        // Sample multiple times - should get different results
        let z1 = encoder.encode(&input).z;
        let z2 = encoder.encode(&input).z;
        
        // Should be different due to sampling
        assert_ne!(z1, z2);
    }
    
    #[test]
    fn test_kl_divergence() {
        let encoder = VariationalEncoder::new(10, 5, 20);
        
        // Zero mean and unit variance should give KL ≈ 0
        let mean = vec![0.0; 5];
        let logvar = vec![0.0; 5]; // log(1) = 0
        
        let kl = encoder.compute_kl_divergence(&mean, &logvar);
        assert!(kl.abs() < 0.1);
    }
}
