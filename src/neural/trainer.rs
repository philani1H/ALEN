//! Training Infrastructure
//!
//! Implements:
//! - Optimizers (Adam, SGD)
//! - Loss functions
//! - Training loop with backpropagation
//! - Learning rate scheduling

use super::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Optimizer trait
pub trait Optimizer {
    /// Update parameters given gradients
    fn step(&mut self, params: &mut [&mut Tensor], grads: &[Tensor]);
    
    /// Zero all gradients
    fn zero_grad(&mut self);
    
    /// Get current learning rate
    fn get_lr(&self) -> f32;
    
    /// Set learning rate
    fn set_lr(&mut self, lr: f32);
}

/// Stochastic Gradient Descent optimizer
#[derive(Debug, Clone)]
pub struct SGD {
    /// Learning rate
    pub lr: f32,
    /// Momentum factor
    pub momentum: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Velocity buffers for momentum
    velocities: HashMap<usize, Vec<f32>>,
}

impl SGD {
    pub fn new(lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            velocities: HashMap::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [&mut Tensor], grads: &[Tensor]) {
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let numel = param.shape.numel();
            
            // Get or create velocity buffer
            let velocity = self.velocities.entry(i).or_insert_with(|| vec![0.0; numel]);
            
            // Update with momentum
            let mut new_data = param.to_vec();
            for j in 0..numel {
                // Apply weight decay
                let grad_with_decay = grad.data[j] + self.weight_decay * new_data[j];
                
                // Update velocity
                velocity[j] = self.momentum * velocity[j] + grad_with_decay;
                
                // Update parameter
                new_data[j] -= self.lr * velocity[j];
            }
            
            *param = Tensor::new(new_data, param.shape.clone());
            if param.requires_grad {
                *param = param.clone().with_grad();
            }
        }
    }

    fn zero_grad(&mut self) {
        // Velocities are preserved across steps
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Adam optimizer
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate
    pub lr: f32,
    /// Beta1 (first moment decay)
    pub beta1: f32,
    /// Beta2 (second moment decay)
    pub beta2: f32,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// First moment estimates
    m: HashMap<usize, Vec<f32>>,
    /// Second moment estimates
    v: HashMap<usize, Vec<f32>>,
    /// Step count
    t: usize,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [&mut Tensor], grads: &[Tensor]) {
        self.t += 1;
        
        // Bias correction
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
        
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let numel = param.shape.numel();
            
            // Get or create moment buffers
            let m = self.m.entry(i).or_insert_with(|| vec![0.0; numel]);
            let v = self.v.entry(i).or_insert_with(|| vec![0.0; numel]);
            
            let mut new_data = param.to_vec();
            
            for j in 0..numel {
                let g = grad.data[j];
                
                // Update biased first moment estimate
                m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * g;
                
                // Update biased second moment estimate
                v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * g * g;
                
                // Compute bias-corrected estimates
                let m_hat = m[j] / bias_correction1;
                let v_hat = v[j] / bias_correction2;
                
                // Update parameter
                let update = self.lr * m_hat / (v_hat.sqrt() + self.eps);
                new_data[j] -= update;
                
                // Apply weight decay (AdamW style)
                if self.weight_decay > 0.0 {
                    new_data[j] -= self.lr * self.weight_decay * new_data[j];
                }
            }
            
            *param = Tensor::new(new_data, param.shape.clone());
            if param.requires_grad {
                *param = param.clone().with_grad();
            }
        }
    }

    fn zero_grad(&mut self) {
        // Moment estimates are preserved
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Loss function trait
pub trait LossFunction {
    /// Compute loss and gradient
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> (f32, Tensor);
}

/// Mean Squared Error loss
#[derive(Debug, Clone, Default)]
pub struct MSELoss;

impl LossFunction for MSELoss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> (f32, Tensor) {
        assert_eq!(predictions.shape, targets.shape);
        
        let n = predictions.shape.numel() as f32;
        let mut loss = 0.0;
        let mut grad = vec![0.0; predictions.shape.numel()];
        
        for i in 0..predictions.shape.numel() {
            let diff = predictions.data[i] - targets.data[i];
            loss += diff * diff;
            grad[i] = 2.0 * diff / n;
        }
        
        (loss / n, Tensor::new(grad, predictions.shape.clone()))
    }
}

/// Cross-Entropy loss (for classification)
#[derive(Debug, Clone, Default)]
pub struct CrossEntropyLoss;

impl LossFunction for CrossEntropyLoss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> (f32, Tensor) {
        // predictions: [batch, num_classes] (logits)
        // targets: [batch] (class indices as floats)
        
        let batch_size = predictions.shape.dim(0);
        let num_classes = predictions.shape.dim(1);
        
        // Softmax
        let probs = predictions.softmax();
        
        let mut loss = 0.0;
        let mut grad = vec![0.0; predictions.shape.numel()];
        
        for b in 0..batch_size {
            let target_class = targets.data[b] as usize;
            
            // Cross-entropy: -log(p[target])
            let prob = probs.data[b * num_classes + target_class].max(1e-10);
            loss -= prob.ln();
            
            // Gradient: p - one_hot(target)
            for c in 0..num_classes {
                let idx = b * num_classes + c;
                grad[idx] = probs.data[idx];
                if c == target_class {
                    grad[idx] -= 1.0;
                }
                grad[idx] /= batch_size as f32;
            }
        }
        
        (loss / batch_size as f32, Tensor::new(grad, predictions.shape.clone()))
    }
}

/// Contrastive loss for embedding learning
#[derive(Debug, Clone)]
pub struct ContrastiveLoss {
    /// Temperature parameter
    pub temperature: f32,
}

impl Default for ContrastiveLoss {
    fn default() -> Self {
        Self { temperature: 0.07 }
    }
}

impl LossFunction for ContrastiveLoss {
    fn compute(&self, anchor: &Tensor, positive: &Tensor) -> (f32, Tensor) {
        // Simplified contrastive loss: maximize similarity between anchor and positive
        let batch_size = anchor.shape.dim(0);
        let dim = anchor.shape.dim(1);
        
        let mut loss = 0.0;
        let mut grad = vec![0.0; anchor.shape.numel()];
        
        for b in 0..batch_size {
            // Compute cosine similarity
            let mut dot = 0.0;
            let mut norm_a = 0.0;
            let mut norm_p = 0.0;
            
            for d in 0..dim {
                let a = anchor.data[b * dim + d];
                let p = positive.data[b * dim + d];
                dot += a * p;
                norm_a += a * a;
                norm_p += p * p;
            }
            
            norm_a = norm_a.sqrt().max(1e-10);
            norm_p = norm_p.sqrt().max(1e-10);
            let sim = dot / (norm_a * norm_p);
            
            // Loss: -log(exp(sim/temp) / sum(exp(sims/temp)))
            // Simplified: just maximize similarity
            loss -= sim / self.temperature;
            
            // Gradient w.r.t. anchor
            for d in 0..dim {
                let a = anchor.data[b * dim + d];
                let p = positive.data[b * dim + d];
                // d(sim)/d(a) = p/(norm_a * norm_p) - a * dot / (norm_a^3 * norm_p)
                grad[b * dim + d] = -(p / (norm_a * norm_p) - a * dot / (norm_a.powi(3) * norm_p)) 
                    / self.temperature / batch_size as f32;
            }
        }
        
        (loss / batch_size as f32, Tensor::new(grad, anchor.shape.clone()))
    }
}

/// Learning rate scheduler
#[derive(Debug, Clone)]
pub enum LRScheduler {
    /// Constant learning rate
    Constant(f32),
    /// Step decay: lr = lr * gamma every step_size steps
    StepLR { initial_lr: f32, step_size: usize, gamma: f32 },
    /// Cosine annealing
    CosineAnnealing { initial_lr: f32, min_lr: f32, total_steps: usize },
    /// Linear warmup then decay
    WarmupLinear { initial_lr: f32, warmup_steps: usize, total_steps: usize },
}

impl LRScheduler {
    pub fn get_lr(&self, step: usize) -> f32 {
        match self {
            LRScheduler::Constant(lr) => *lr,
            
            LRScheduler::StepLR { initial_lr, step_size, gamma } => {
                let num_decays = step / step_size;
                initial_lr * gamma.powi(num_decays as i32)
            }
            
            LRScheduler::CosineAnnealing { initial_lr, min_lr, total_steps } => {
                let progress = (step as f32 / *total_steps as f32).min(1.0);
                let cosine = (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
                min_lr + (initial_lr - min_lr) * cosine
            }
            
            LRScheduler::WarmupLinear { initial_lr, warmup_steps, total_steps } => {
                if step < *warmup_steps {
                    // Linear warmup
                    initial_lr * (step as f32 / *warmup_steps as f32)
                } else {
                    // Linear decay
                    let decay_steps = total_steps - warmup_steps;
                    let decay_progress = (step - warmup_steps) as f32 / decay_steps as f32;
                    initial_lr * (1.0 - decay_progress).max(0.0)
                }
            }
        }
    }
}

/// Training batch
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// Input tensors
    pub inputs: Tensor,
    /// Target tensors
    pub targets: Tensor,
    /// Optional attention mask
    pub mask: Option<Tensor>,
}

/// Training metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Total loss
    pub loss: f32,
    /// Number of samples
    pub num_samples: usize,
    /// Accuracy (if applicable)
    pub accuracy: Option<f32>,
    /// Learning rate
    pub learning_rate: f32,
    /// Step number
    pub step: usize,
}

/// Trainer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub num_epochs: usize,
    /// Gradient clipping threshold
    pub max_grad_norm: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// Warmup steps
    pub warmup_steps: usize,
    /// Log every N steps
    pub log_interval: usize,
    /// Evaluate every N steps
    pub eval_interval: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 32,
            num_epochs: 10,
            max_grad_norm: 1.0,
            weight_decay: 0.01,
            warmup_steps: 1000,
            log_interval: 100,
            eval_interval: 1000,
        }
    }
}

/// Main trainer struct
pub struct Trainer {
    /// Configuration
    pub config: TrainerConfig,
    /// Optimizer
    pub optimizer: Adam,
    /// Learning rate scheduler
    pub scheduler: LRScheduler,
    /// Loss function
    pub loss_fn: Box<dyn LossFunction + Send + Sync>,
    /// Current step
    pub step: usize,
    /// Training history
    pub history: Vec<TrainingMetrics>,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        let optimizer = Adam::new(config.learning_rate)
            .with_weight_decay(config.weight_decay);
        
        let scheduler = LRScheduler::WarmupLinear {
            initial_lr: config.learning_rate,
            warmup_steps: config.warmup_steps,
            total_steps: config.num_epochs * 1000, // Approximate
        };
        
        Self {
            config,
            optimizer,
            scheduler,
            loss_fn: Box::new(MSELoss),
            step: 0,
            history: Vec::new(),
        }
    }

    pub fn with_loss<L: LossFunction + Send + Sync + 'static>(mut self, loss: L) -> Self {
        self.loss_fn = Box::new(loss);
        self
    }

    /// Perform a single training step
    pub fn train_step(&mut self, params: &mut [&mut Tensor], batch: &TrainingBatch) -> TrainingMetrics {
        // Forward pass would be done by the model
        // Here we just compute loss and update
        
        // Compute loss and gradient
        let (loss, grad) = self.loss_fn.compute(&batch.inputs, &batch.targets);
        
        // Clip gradients
        let grad = self.clip_grad(&grad);
        
        // Update learning rate
        let lr = self.scheduler.get_lr(self.step);
        self.optimizer.set_lr(lr);
        
        // Update parameters
        self.optimizer.step(params, &[grad]);
        
        self.step += 1;
        
        let metrics = TrainingMetrics {
            loss,
            num_samples: batch.inputs.shape.dim(0),
            accuracy: None,
            learning_rate: lr,
            step: self.step,
        };
        
        self.history.push(metrics.clone());
        metrics
    }

    /// Clip gradients by norm
    fn clip_grad(&self, grad: &Tensor) -> Tensor {
        let norm: f32 = grad.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm > self.config.max_grad_norm {
            let scale = self.config.max_grad_norm / norm;
            grad.scale(scale)
        } else {
            grad.clone()
        }
    }

    /// Get training summary
    pub fn summary(&self) -> String {
        let avg_loss: f32 = self.history.iter().map(|m| m.loss).sum::<f32>() 
            / self.history.len().max(1) as f32;
        
        format!(
            "Training Summary:\n  Steps: {}\n  Avg Loss: {:.4}\n  Final LR: {:.6}",
            self.step,
            avg_loss,
            self.scheduler.get_lr(self.step)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd() {
        let mut sgd = SGD::new(0.1, 0.9, 0.0);
        let mut param = Tensor::ones(vec![2, 2]).with_grad();
        let grad = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
        
        sgd.step(&mut [&mut param], &[grad]);
        
        // Parameter should have decreased
        assert!(param.data[0] < 1.0);
    }

    #[test]
    fn test_adam() {
        let mut adam = Adam::new(0.001);
        let mut param = Tensor::ones(vec![2, 2]).with_grad();
        let grad = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
        
        adam.step(&mut [&mut param], &[grad]);
        
        assert!(param.data[0] < 1.0);
    }

    #[test]
    fn test_mse_loss() {
        let loss_fn = MSELoss;
        let pred = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let target = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        
        let (loss, _grad) = loss_fn.compute(&pred, &target);
        assert!(loss < 1e-6);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let loss_fn = CrossEntropyLoss;
        let pred = Tensor::new(vec![2.0, 1.0, 0.1, 0.1, 2.0, 1.0], vec![2, 3]);
        let target = Tensor::new(vec![0.0, 1.0], vec![2]); // Class 0 and 1
        
        let (loss, grad) = loss_fn.compute(&pred, &target);
        assert!(loss > 0.0);
        assert_eq!(grad.shape.0, vec![2, 3]);
    }

    #[test]
    fn test_lr_scheduler() {
        let scheduler = LRScheduler::CosineAnnealing {
            initial_lr: 0.1,
            min_lr: 0.001,
            total_steps: 100,
        };
        
        assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-5);
        assert!(scheduler.get_lr(50) < 0.1);
        assert!(scheduler.get_lr(100) < 0.01);
    }

    #[test]
    fn test_warmup_scheduler() {
        let scheduler = LRScheduler::WarmupLinear {
            initial_lr: 0.1,
            warmup_steps: 10,
            total_steps: 100,
        };
        
        // During warmup
        assert!(scheduler.get_lr(5) < 0.1);
        // After warmup
        assert!((scheduler.get_lr(10) - 0.1).abs() < 1e-5);
        // Decay
        assert!(scheduler.get_lr(50) < 0.1);
    }
}
