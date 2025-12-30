//! Probabilistic Decoder
//!
//! Implements: P_θ(Y | Z, u) = Π_t P_θ(y_t | y_<t, Z, u)
//!
//! Autoregressive generation with temperature-scaled sampling
//! for probabilistic text generation.

use crate::core::ThoughtState;
use crate::memory::SemanticMemory;
use serde::{Deserialize, Serialize};

/// Probabilistic decoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticDecoderConfig {
    /// Temperature for sampling (higher = more random)
    pub temperature: f64,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) sampling (0.0 = disabled)
    pub top_p: f64,
    /// Maximum tokens to generate
    pub max_tokens: usize,
}

impl Default for ProbabilisticDecoderConfig {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_k: 50,
            top_p: 0.95,
            max_tokens: 100,
        }
    }
}

/// Probabilistic decoder
pub struct ProbabilisticDecoder {
    config: ProbabilisticDecoderConfig,
    dimension: usize,
}

impl ProbabilisticDecoder {
    pub fn new(dimension: usize, config: ProbabilisticDecoderConfig) -> Self {
        Self { config, dimension }
    }
    
    pub fn default(dimension: usize) -> Self {
        Self::new(dimension, ProbabilisticDecoderConfig::default())
    }
    
    /// Generate text probabilistically: P_θ(Y | Z, u)
    pub fn generate(
        &self,
        thought: &ThoughtState,
        user_embedding: Option<&[f64]>,
        memory: &SemanticMemory,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut tokens = Vec::new();
        let mut current_state = thought.vector.clone();
        
        // Combine with user embedding if provided
        if let Some(u) = user_embedding {
            current_state = self.combine_with_user(&current_state, u);
        }
        
        // Autoregressive generation
        for _ in 0..self.config.max_tokens {
            // Get next token distribution
            let logits = self.compute_logits(&current_state, memory)?;
            
            // Sample from distribution
            let token = self.sample_token(&logits);
            
            // Check for end token
            if token == "<END>" {
                break;
            }
            
            tokens.push(token.clone());
            
            // Update state for next token
            current_state = self.update_state(&current_state, &token);
        }
        
        Ok(tokens.join(" "))
    }
    
    /// Compute logits for next token: f_θ(y_<t, Z, u)
    fn compute_logits(
        &self,
        state: &[f64],
        memory: &SemanticMemory,
    ) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
        // Find similar concepts in memory
        let similar = memory.find_similar(state, 20)?;
        
        // Convert similarities to logits
        let mut logits = Vec::new();
        for (fact, similarity) in similar {
            // Extract words from fact content
            for word in fact.content.split_whitespace() {
                logits.push((word.to_string(), similarity));
            }
        }
        
        // Add end token
        logits.push(("<END>".to_string(), 0.1));
        
        Ok(logits)
    }
    
    /// Sample token from distribution with temperature
    fn sample_token(&self, logits: &[(String, f64)]) -> String {
        if logits.is_empty() {
            return "<END>".to_string();
        }
        
        // Apply temperature scaling: p_i = exp(logit_i / T) / Σ exp(logit_j / T)
        let scaled_logits: Vec<f64> = logits
            .iter()
            .map(|(_, logit)| (logit / self.config.temperature).exp())
            .collect();
        
        let sum: f64 = scaled_logits.iter().sum();
        let probabilities: Vec<f64> = scaled_logits.iter().map(|x| x / sum).collect();
        
        // Apply top-k filtering
        let filtered = if self.config.top_k > 0 {
            self.top_k_filter(logits, &probabilities)
        } else {
            (logits.to_vec(), probabilities)
        };
        
        // Apply top-p (nucleus) filtering
        let (tokens, probs) = if self.config.top_p > 0.0 {
            self.top_p_filter(&filtered.0, &filtered.1)
        } else {
            filtered
        };
        
        // Sample from filtered distribution
        self.sample_from_distribution(&tokens, &probs)
    }
    
    /// Top-k filtering: keep only top k tokens
    fn top_k_filter(
        &self,
        tokens: &[(String, f64)],
        probabilities: &[f64],
    ) -> (Vec<(String, f64)>, Vec<f64>) {
        let mut indexed: Vec<(usize, f64)> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        // Sort by probability descending
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top k
        let k = self.config.top_k.min(indexed.len());
        let top_indices: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
        
        let filtered_tokens: Vec<(String, f64)> = top_indices
            .iter()
            .map(|&i| tokens[i].clone())
            .collect();
        
        let filtered_probs: Vec<f64> = top_indices.iter().map(|&i| probabilities[i]).collect();
        
        // Renormalize
        let sum: f64 = filtered_probs.iter().sum();
        let normalized: Vec<f64> = filtered_probs.iter().map(|p| p / sum).collect();
        
        (filtered_tokens, normalized)
    }
    
    /// Top-p (nucleus) filtering: keep tokens with cumulative probability >= p
    fn top_p_filter(
        &self,
        tokens: &[(String, f64)],
        probabilities: &[f64],
    ) -> (Vec<(String, f64)>, Vec<f64>) {
        let mut indexed: Vec<(usize, f64)> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        // Sort by probability descending
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Accumulate until reaching top_p
        let mut cumsum = 0.0;
        let mut cutoff = indexed.len();
        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= self.config.top_p {
                cutoff = i + 1;
                break;
            }
        }
        
        let top_indices: Vec<usize> = indexed.iter().take(cutoff).map(|(i, _)| *i).collect();
        
        let filtered_tokens: Vec<(String, f64)> = top_indices
            .iter()
            .map(|&i| tokens[i].clone())
            .collect();
        
        let filtered_probs: Vec<f64> = top_indices.iter().map(|&i| probabilities[i]).collect();
        
        // Renormalize
        let sum: f64 = filtered_probs.iter().sum();
        let normalized: Vec<f64> = filtered_probs.iter().map(|p| p / sum).collect();
        
        (filtered_tokens, normalized)
    }
    
    /// Sample from probability distribution
    fn sample_from_distribution(&self, tokens: &[(String, f64)], probabilities: &[f64]) -> String {
        use rand::Rng;
        
        if tokens.is_empty() {
            return "<END>".to_string();
        }
        
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        
        let mut cumsum = 0.0;
        for (i, &p) in probabilities.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return tokens[i].0.clone();
            }
        }
        
        // Fallback to last token
        tokens.last().unwrap().0.clone()
    }
    
    /// Combine thought state with user embedding
    fn combine_with_user(&self, thought: &[f64], user: &[f64]) -> Vec<f64> {
        let mut combined = thought.to_vec();
        let min_len = combined.len().min(user.len());
        
        // Add user embedding (element-wise)
        for i in 0..min_len {
            combined[i] += 0.1 * user[i]; // Small weight for user influence
        }
        
        combined
    }
    
    /// Update state after generating token
    fn update_state(&self, state: &[f64], token: &str) -> Vec<f64> {
        // Simple update: add small perturbation based on token
        let mut new_state = state.to_vec();
        let token_hash = token.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64));
        
        for i in 0..new_state.len() {
            let perturbation = ((token_hash.wrapping_mul(i as u64)) % 1000) as f64 / 10000.0;
            new_state[i] += perturbation - 0.05; // Center around 0
        }
        
        new_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_probabilistic_decoder() {
        let decoder = ProbabilisticDecoder::default(128);
        let thought = ThoughtState::new(128);
        let memory = SemanticMemory::in_memory(128).unwrap();
        
        // Should generate something (even if empty memory)
        let result = decoder.generate(&thought, None, &memory);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_temperature_scaling() {
        let decoder = ProbabilisticDecoder::new(
            128,
            ProbabilisticDecoderConfig {
                temperature: 0.1, // Low temperature = more deterministic
                ..Default::default()
            },
        );
        
        let logits = vec![
            ("a".to_string(), 2.0),
            ("b".to_string(), 1.0),
            ("c".to_string(), 0.5),
        ];
        
        // With low temperature, should mostly pick "a"
        let mut counts = std::collections::HashMap::new();
        for _ in 0..100 {
            let token = decoder.sample_token(&logits);
            *counts.entry(token).or_insert(0) += 1;
        }
        
        // "a" should be most common
        assert!(counts.get("a").unwrap_or(&0) > counts.get("b").unwrap_or(&0));
    }
}
