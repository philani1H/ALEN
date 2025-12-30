//! Creative Latent Space Exploration
//!
//! Enables creative problem-solving through controlled noise injection:
//!
//! Mathematical Foundation:
//! Base latent: z = Encoder(x)
//! Creative latent: z_creative = z + ε, where ε ~ N(0, σ²I)
//! Temperature sampling: P(y|z) ∝ exp(f(z, y) / τ)
//!
//! Exploration Strategies:
//! 1. Gaussian noise injection
//! 2. Temperature-based sampling
//! 3. Diversity-promoting regularization
//! 4. Novelty search

use super::tensor::Tensor;
use rand::Rng;
use std::collections::HashMap;

// ============================================================================
// PART 1: NOISE INJECTION
// ============================================================================

#[derive(Debug, Clone)]
pub struct NoiseInjector {
    /// Noise standard deviation
    sigma: f32,
    
    /// Noise schedule (annealing)
    schedule: NoiseSchedule,
}

impl NoiseInjector {
    pub fn new(sigma: f32, schedule: NoiseSchedule) -> Self {
        Self { sigma, schedule }
    }
    
    /// Inject Gaussian noise into latent representation
    /// z_creative = z + ε, where ε ~ N(0, σ²I)
    pub fn inject(&self, latent: &Tensor, step: usize) -> Tensor {
        let current_sigma = self.schedule.get_sigma(self.sigma, step);
        let noise = Tensor::randn(latent.shape()) .scale(current_sigma);
        latent.add(&noise)
    }
    
    /// Inject structured noise (correlated dimensions)
    pub fn inject_structured(&self, latent: &Tensor, step: usize, correlation: f32) -> Tensor {
        let current_sigma = self.schedule.get_sigma(self.sigma, step);
        let shape = latent.shape();
        
        // Generate base noise
        let base_noise = Tensor::randn(vec![shape[0], 1]).scale(current_sigma);
        
        // Generate independent noise
        let independent_noise = Tensor::randn(shape) .scale(current_sigma);
        
        // Combine: ε = α * base + (1-α) * independent
        let structured = base_noise.broadcast(shape).mul_scalar(correlation);
        let independent = independent_noise.mul_scalar(1.0 - correlation);
        
        latent.add(&structured.add(&independent))
    }
}

#[derive(Debug, Clone)]
pub enum NoiseSchedule {
    /// Constant noise
    Constant,
    
    /// Linear annealing: σ(t) = σ₀ * (1 - t/T)
    LinearAnneal { total_steps: usize },
    
    /// Exponential decay: σ(t) = σ₀ * exp(-λt)
    ExponentialDecay { decay_rate: f32 },
    
    /// Cosine annealing: σ(t) = σ₀ * (1 + cos(πt/T)) / 2
    CosineAnneal { total_steps: usize },
}

impl NoiseSchedule {
    fn get_sigma(&self, base_sigma: f32, step: usize) -> f32 {
        match self {
            NoiseSchedule::Constant => base_sigma,
            
            NoiseSchedule::LinearAnneal { total_steps } => {
                let progress = (step as f32 / *total_steps as f32).min(1.0);
                base_sigma * (1.0 - progress)
            }
            
            NoiseSchedule::ExponentialDecay { decay_rate } => {
                base_sigma * (-decay_rate * step as f32).exp()
            }
            
            NoiseSchedule::CosineAnneal { total_steps } => {
                let progress = (step as f32 / *total_steps as f32).min(1.0);
                base_sigma * (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0
            }
        }
    }
}

// ============================================================================
// PART 2: TEMPERATURE SAMPLING
// ============================================================================

#[derive(Debug, Clone)]
pub struct TemperatureSampler {
    /// Base temperature
    base_temperature: f32,
    
    /// Temperature schedule
    schedule: TemperatureSchedule,
}

impl TemperatureSampler {
    pub fn new(base_temperature: f32, schedule: TemperatureSchedule) -> Self {
        Self {
            base_temperature,
            schedule,
        }
    }
    
    /// Sample from distribution with temperature
    /// P(y|z) ∝ exp(logits / τ)
    pub fn sample(&self, logits: &Tensor, step: usize) -> Vec<usize> {
        let temperature = self.schedule.get_temperature(self.base_temperature, step);
        let batch_size = logits.shape()[0];
        let vocab_size = logits.shape()[1];
        
        let mut samples = Vec::new();
        
        for b in 0..batch_size {
            let batch_logits = logits.slice(b, 0);
            
            // Scale by temperature
            let scaled_logits: Vec<f32> = batch_logits
                .iter()
                .map(|&l| l / temperature)
                .collect();
            
            // Softmax
            let probs = self.softmax(&scaled_logits);
            
            // Sample
            let sample = self.categorical_sample(&probs);
            samples.push(sample);
        }
        
        samples
    }
    
    /// Top-k sampling with temperature
    pub fn sample_top_k(&self, logits: &Tensor, k: usize, step: usize) -> Vec<usize> {
        let temperature = self.schedule.get_temperature(self.base_temperature, step);
        let batch_size = logits.shape()[0];
        
        let mut samples = Vec::new();
        
        for b in 0..batch_size {
            let batch_logits = logits.slice(b, 0);
            
            // Scale by temperature
            let mut scaled_logits: Vec<(usize, f32)> = batch_logits
                .iter()
                .enumerate()
                .map(|(i, &l)| (i, l / temperature))
                .collect();
            
            // Sort by logit (descending)
            scaled_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // Take top-k
            let top_k = scaled_logits.iter().take(k).collect::<Vec<_>>();
            let top_k_logits: Vec<f32> = top_k.iter().map(|(_, l)| *l).collect();
            
            // Softmax over top-k
            let probs = self.softmax(&top_k_logits);
            
            // Sample
            let sample_idx = self.categorical_sample(&probs);
            let sample = top_k[sample_idx].0;
            samples.push(sample);
        }
        
        samples
    }
    
    /// Nucleus (top-p) sampling with temperature
    pub fn sample_nucleus(&self, logits: &Tensor, p: f32, step: usize) -> Vec<usize> {
        let temperature = self.schedule.get_temperature(self.base_temperature, step);
        let batch_size = logits.shape()[0];
        
        let mut samples = Vec::new();
        
        for b in 0..batch_size {
            let batch_logits = logits.slice(b, 0);
            
            // Scale by temperature
            let mut scaled_logits: Vec<(usize, f32)> = batch_logits
                .iter()
                .enumerate()
                .map(|(i, &l)| (i, l / temperature))
                .collect();
            
            // Sort by logit (descending)
            scaled_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // Compute probabilities
            let logits_only: Vec<f32> = scaled_logits.iter().map(|(_, l)| *l).collect();
            let all_probs = self.softmax(&logits_only);
            
            // Find nucleus (cumulative probability >= p)
            let mut cumsum = 0.0;
            let mut nucleus_size = 0;
            for prob in all_probs.iter() {
                cumsum += prob;
                nucleus_size += 1;
                if cumsum >= p {
                    break;
                }
            }
            
            // Sample from nucleus
            let nucleus_probs = &all_probs[..nucleus_size];
            let sample_idx = self.categorical_sample(nucleus_probs);
            let sample = scaled_logits[sample_idx].0;
            samples.push(sample);
        }
        
        samples
    }
    
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = exp_logits.iter().sum();
        
        exp_logits.iter().map(|&e| e / sum).collect()
    }
    
    fn categorical_sample(&self, probs: &[f32]) -> usize {
        let mut rng = rand::thread_rng();
        let u: f32 = rng.gen();
        
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

#[derive(Debug, Clone)]
pub enum TemperatureSchedule {
    /// Constant temperature
    Constant,
    
    /// Linear cooling: τ(t) = τ₀ * (1 - t/T)
    LinearCooling { total_steps: usize },
    
    /// Exponential cooling: τ(t) = τ₀ * exp(-λt)
    ExponentialCooling { decay_rate: f32 },
}

impl TemperatureSchedule {
    fn get_temperature(&self, base_temp: f32, step: usize) -> f32 {
        match self {
            TemperatureSchedule::Constant => base_temp,
            
            TemperatureSchedule::LinearCooling { total_steps } => {
                let progress = (step as f32 / *total_steps as f32).min(1.0);
                base_temp * (1.0 - 0.9 * progress).max(0.1)
            }
            
            TemperatureSchedule::ExponentialCooling { decay_rate } => {
                (base_temp * (-decay_rate * step as f32).exp()).max(0.1)
            }
        }
    }
}

// ============================================================================
// PART 3: DIVERSITY PROMOTION
// ============================================================================

#[derive(Debug, Clone)]
pub struct DiversityPromoter {
    /// Diversity weight
    diversity_weight: f32,
    
    /// History of generated samples
    sample_history: Vec<Vec<f32>>,
    
    /// Maximum history size
    max_history_size: usize,
}

impl DiversityPromoter {
    pub fn new(diversity_weight: f32, max_history_size: usize) -> Self {
        Self {
            diversity_weight,
            sample_history: Vec::new(),
            max_history_size,
        }
    }
    
    /// Add sample to history
    pub fn add_sample(&mut self, sample: Vec<f32>) {
        self.sample_history.push(sample);
        
        if self.sample_history.len() > self.max_history_size {
            self.sample_history.remove(0);
        }
    }
    
    /// Compute diversity loss (encourages different outputs)
    /// L_diversity = -∑_i ∑_j d(z_i, z_j)
    pub fn compute_diversity_loss(&self, current_samples: &[Vec<f32>]) -> f32 {
        if self.sample_history.is_empty() {
            return 0.0;
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for current in current_samples {
            for historical in &self.sample_history {
                let distance = self.euclidean_distance(current, historical);
                total_distance += distance;
                count += 1;
            }
        }
        
        if count > 0 {
            -self.diversity_weight * (total_distance / count as f32)
        } else {
            0.0
        }
    }
    
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

// ============================================================================
// PART 4: NOVELTY SEARCH
// ============================================================================

#[derive(Debug, Clone)]
pub struct NoveltySearch {
    /// Archive of novel behaviors
    archive: Vec<BehaviorDescriptor>,
    
    /// K-nearest neighbors for novelty computation
    k_nearest: usize,
    
    /// Novelty threshold for archive addition
    novelty_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct BehaviorDescriptor {
    pub features: Vec<f32>,
    pub fitness: f32,
}

impl NoveltySearch {
    pub fn new(k_nearest: usize, novelty_threshold: f32) -> Self {
        Self {
            archive: Vec::new(),
            k_nearest,
            novelty_threshold,
        }
    }
    
    /// Compute novelty score
    /// Novelty = average distance to k-nearest neighbors
    pub fn compute_novelty(&self, behavior: &BehaviorDescriptor) -> f32 {
        if self.archive.is_empty() {
            return f32::INFINITY;
        }
        
        // Compute distances to all archived behaviors
        let mut distances: Vec<f32> = self.archive
            .iter()
            .map(|archived| self.behavior_distance(behavior, archived))
            .collect();
        
        // Sort distances
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Average of k-nearest
        let k = self.k_nearest.min(distances.len());
        distances.iter().take(k).sum::<f32>() / k as f32
    }
    
    /// Add behavior to archive if novel enough
    pub fn maybe_add_to_archive(&mut self, behavior: BehaviorDescriptor) -> bool {
        let novelty = self.compute_novelty(&behavior);
        
        if novelty >= self.novelty_threshold {
            self.archive.push(behavior);
            true
        } else {
            false
        }
    }
    
    fn behavior_distance(&self, a: &BehaviorDescriptor, b: &BehaviorDescriptor) -> f32 {
        if a.features.len() != b.features.len() {
            return f32::INFINITY;
        }
        
        a.features
            .iter()
            .zip(b.features.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    pub fn archive_size(&self) -> usize {
        self.archive.len()
    }
}

// ============================================================================
// PART 5: CREATIVE EXPLORATION CONTROLLER
// ============================================================================

#[derive(Debug, Clone)]
pub struct CreativeExplorationController {
    /// Noise injector
    noise_injector: NoiseInjector,
    
    /// Temperature sampler
    temperature_sampler: TemperatureSampler,
    
    /// Diversity promoter
    diversity_promoter: DiversityPromoter,
    
    /// Novelty search
    novelty_search: NoveltySearch,
    
    /// Current step
    step: usize,
}

impl CreativeExplorationController {
    pub fn new(
        sigma: f32,
        noise_schedule: NoiseSchedule,
        temperature: f32,
        temp_schedule: TemperatureSchedule,
        diversity_weight: f32,
        novelty_k: usize,
        novelty_threshold: f32,
    ) -> Self {
        Self {
            noise_injector: NoiseInjector::new(sigma, noise_schedule),
            temperature_sampler: TemperatureSampler::new(temperature, temp_schedule),
            diversity_promoter: DiversityPromoter::new(diversity_weight, 100),
            novelty_search: NoveltySearch::new(novelty_k, novelty_threshold),
            step: 0,
        }
    }
    
    /// Apply creative exploration to latent
    pub fn explore(&mut self, latent: &Tensor, exploration_mode: ExplorationMode) -> Tensor {
        match exploration_mode {
            ExplorationMode::Gaussian => {
                self.noise_injector.inject(latent, self.step)
            }
            
            ExplorationMode::Structured { correlation } => {
                self.noise_injector.inject_structured(latent, self.step, correlation)
            }
            
            ExplorationMode::None => latent.clone(),
        }
    }
    
    /// Sample with temperature
    pub fn sample(&self, logits: &Tensor, sampling_mode: SamplingMode) -> Vec<usize> {
        match sampling_mode {
            SamplingMode::Greedy => {
                // Argmax
                let batch_size = logits.shape()[0];
                (0..batch_size)
                    .map(|b| {
                        let batch_logits = logits.slice(b, 0);
                        batch_logits
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(i, _)| i)
                            .unwrap_or(0)
                    })
                    .collect()
            }
            
            SamplingMode::Temperature => {
                self.temperature_sampler.sample(logits, self.step)
            }
            
            SamplingMode::TopK { k } => {
                self.temperature_sampler.sample_top_k(logits, k, self.step)
            }
            
            SamplingMode::Nucleus { p } => {
                self.temperature_sampler.sample_nucleus(logits, p, self.step)
            }
        }
    }
    
    /// Increment step
    pub fn step(&mut self) {
        self.step += 1;
    }
    
    /// Get current step
    pub fn current_step(&self) -> usize {
        self.step
    }
}

#[derive(Debug, Clone)]
pub enum ExplorationMode {
    None,
    Gaussian,
    Structured { correlation: f32 },
}

#[derive(Debug, Clone)]
pub enum SamplingMode {
    Greedy,
    Temperature,
    TopK { k: usize },
    Nucleus { p: f32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_noise_injection() {
        let injector = NoiseInjector::new(0.1, NoiseSchedule::Constant);
        let latent = Tensor::randn(vec![2, 128]);
        
        let noisy = injector.inject(&latent, 0);
        
        assert_eq!(noisy.shape(), latent.shape());
    }
    
    #[test]
    fn test_temperature_sampling() {
        let sampler = TemperatureSampler::new(1.0, TemperatureSchedule::Constant);
        let logits = Tensor::randn(vec![2, 10]);
        
        let samples = sampler.sample(&logits, 0);
        
        assert_eq!(samples.len(), 2);
    }
    
    #[test]
    fn test_novelty_search() {
        let mut novelty = NoveltySearch::new(3, 0.5);
        
        let behavior1 = BehaviorDescriptor {
            features: vec![1.0, 2.0, 3.0],
            fitness: 0.8,
        };
        
        let added = novelty.maybe_add_to_archive(behavior1);
        assert!(added);
        
        assert_eq!(novelty.archive_size(), 1);
    }
}
