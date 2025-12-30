//! Latent Space Decoder - UNDERSTANDING, NOT MEMORIZATION
//!
//! Generates text from thought vectors using learned patterns in latent space.
//! NO RETRIEVAL. NO LOOKUP. NO STORED ANSWERS.
//!
//! Architecture:
//! 1. Thought vector → Latent pattern recognition
//! 2. Pattern → Concept activation (learned weights)
//! 3. Concept activation → Token probabilities (neural network)
//! 4. Token probabilities → Generated text (sampling)
//!
//! This implements the mathematical framework:
//! Y = Decoder_φ(z) where z = latent reasoning context

use crate::core::ThoughtState;
use crate::memory::SemanticMemory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Latent pattern in thought space
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LatentPattern {
    /// Pattern weights in thought space
    weights: Vec<f64>,
    /// Concept activations this pattern triggers
    concept_activations: HashMap<String, f64>,
    /// How often this pattern was reinforced
    reinforcement: f64,
}

impl LatentPattern {
    fn new(dimension: usize) -> Self {
        Self {
            weights: vec![0.0; dimension],
            concept_activations: HashMap::new(),
            reinforcement: 0.0,
        }
    }

    /// Calculate activation for a given thought
    fn activation(&self, thought: &[f64]) -> f64 {
        if self.weights.len() != thought.len() {
            return 0.0;
        }

        let dot: f64 = self.weights.iter()
            .zip(thought.iter())
            .map(|(w, t)| w * t)
            .sum();

        // Sigmoid activation
        1.0 / (1.0 + (-dot).exp())
    }

    /// Update pattern from thought-concept pair
    fn learn(&mut self, thought: &[f64], concept: &str, learning_rate: f64) {
        // Update weights toward thought
        for (w, t) in self.weights.iter_mut().zip(thought.iter()) {
            *w += learning_rate * (t - *w);
        }

        // Update concept activation
        let current = self.concept_activations.get(concept).copied().unwrap_or(0.0);
        self.concept_activations.insert(concept.to_string(), current + learning_rate);

        self.reinforcement += 1.0;
    }
}

/// Token generator from concept activations
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConceptToTokenNetwork {
    /// Concept → Token probability weights (learned)
    weights: HashMap<String, HashMap<String, f64>>,
    /// Token vocabulary (dynamically built)
    vocabulary: Vec<String>,
    /// Token frequencies (for smoothing)
    token_frequencies: HashMap<String, f64>,
}

impl ConceptToTokenNetwork {
    fn new() -> Self {
        Self {
            weights: HashMap::new(),
            vocabulary: Vec::new(),
            token_frequencies: HashMap::new(),
        }
    }

    /// Learn concept → token association
    fn learn_association(&mut self, concept: &str, tokens: &[String], learning_rate: f64) {
        let concept_weights = self.weights.entry(concept.to_string())
            .or_insert_with(HashMap::new);

        for token in tokens {
            // Add to vocabulary if new
            if !self.vocabulary.contains(token) {
                self.vocabulary.push(token.clone());
            }

            // Update weight
            let current = concept_weights.get(token).copied().unwrap_or(0.0);
            concept_weights.insert(token.clone(), current + learning_rate);

            // Update frequency
            let freq = self.token_frequencies.get(token).copied().unwrap_or(0.0);
            self.token_frequencies.insert(token.clone(), freq + 1.0);
        }
    }

    /// Generate token probabilities from concept activations
    fn generate_probabilities(&self, concept_activations: &HashMap<String, f64>) -> HashMap<String, f64> {
        let mut token_probs: HashMap<String, f64> = HashMap::new();

        // Accumulate probabilities from all activated concepts
        for (concept, activation) in concept_activations {
            if let Some(concept_weights) = self.weights.get(concept) {
                for (token, weight) in concept_weights {
                    let current = token_probs.get(token).copied().unwrap_or(0.0);
                    token_probs.insert(token.clone(), current + activation * weight);
                }
            }
        }

        // Normalize to probabilities
        let total: f64 = token_probs.values().sum();
        if total > 1e-10 {
            for prob in token_probs.values_mut() {
                *prob /= total;
            }
        }

        token_probs
    }

    /// Sample a token from probabilities with temperature
    fn sample_token(&self, probabilities: &HashMap<String, f64>, temperature: f64) -> Option<String> {
        if probabilities.is_empty() {
            return None;
        }

        // Apply temperature
        let mut scaled_probs: Vec<(String, f64)> = probabilities.iter()
            .map(|(token, prob)| {
                let scaled = if temperature > 0.0 {
                    (prob.ln() / temperature).exp()
                } else {
                    *prob
                };
                (token.clone(), scaled)
            })
            .collect();

        // Normalize
        let total: f64 = scaled_probs.iter().map(|(_, p)| p).sum();
        if total < 1e-10 {
            return None;
        }

        for (_, p) in scaled_probs.iter_mut() {
            *p /= total;
        }

        // Sample
        let rng_val = rand::random::<f64>();
        let mut cumulative = 0.0;

        for (token, prob) in scaled_probs {
            cumulative += prob;
            if rng_val <= cumulative {
                return Some(token);
            }
        }

        // Fallback to most probable
        probabilities.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(token, _)| token.clone())
    }
}

/// Latent Space Decoder - generates text from understanding
#[derive(Serialize, Deserialize)]
pub struct LatentDecoder {
    /// Learned patterns in latent space
    patterns: Vec<LatentPattern>,
    /// Concept to token network
    token_network: ConceptToTokenNetwork,
    /// Dimension of thought space
    dimension: usize,
    /// Learning rate
    learning_rate: f64,
    /// Temperature for generation
    temperature: f64,
    /// Maximum tokens to generate
    max_tokens: usize,
}

impl LatentDecoder {
    pub fn new(dimension: usize, num_patterns: usize) -> Self {
        let mut patterns = Vec::new();
        for _ in 0..num_patterns {
            patterns.push(LatentPattern::new(dimension));
        }

        Self {
            patterns,
            token_network: ConceptToTokenNetwork::new(),
            dimension,
            learning_rate: 0.01,
            temperature: 0.7,
            max_tokens: 100,
        }
    }

    /// Learn from thought-text pair (stores patterns, not answers)
    pub fn learn(&mut self, thought: &ThoughtState, text: &str) {
        let tokens: Vec<String> = text.split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();

        if tokens.is_empty() {
            return;
        }

        // Find or create pattern for this thought
        let mut best_pattern_idx = 0;
        let mut best_activation = 0.0;

        for (idx, pattern) in self.patterns.iter().enumerate() {
            let activation = pattern.activation(&thought.vector);
            if activation > best_activation {
                best_activation = activation;
                best_pattern_idx = idx;
            }
        }

        // If no pattern is activated, use the one with lowest reinforcement
        if best_activation < 0.01 {
            best_pattern_idx = self.patterns.iter()
                .enumerate()
                .min_by(|a, b| a.1.reinforcement.partial_cmp(&b.1.reinforcement).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
        }

        // Learn pattern → concept associations with HIGHER learning rate
        for token in &tokens {
            self.patterns[best_pattern_idx].learn(
                &thought.vector,
                token,
                self.learning_rate * 5.0  // Increase learning rate
            );

            // Learn concept → token associations with HIGHER learning rate
            self.token_network.learn_association(token, &tokens, self.learning_rate * 5.0);
        }
    }

    /// Generate text from thought (NO RETRIEVAL)
    pub fn generate(&self, thought: &ThoughtState) -> (String, f64) {
        // Step 1: Activate patterns based on thought
        let mut concept_activations: HashMap<String, f64> = HashMap::new();

        for pattern in &self.patterns {
            let activation = pattern.activation(&thought.vector);
            
            // Lower threshold to activate more patterns
            if activation > 0.01 {
                // Accumulate concept activations from this pattern
                for (concept, weight) in &pattern.concept_activations {
                    let current = concept_activations.get(concept).copied().unwrap_or(0.0);
                    concept_activations.insert(concept.clone(), current + activation * weight);
                }
            }
        }

        // If no patterns activated, use all patterns with lower weight
        if concept_activations.is_empty() {
            for pattern in &self.patterns {
                if pattern.reinforcement > 0.0 {
                    for (concept, weight) in &pattern.concept_activations {
                        let current = concept_activations.get(concept).copied().unwrap_or(0.0);
                        concept_activations.insert(concept.clone(), current + weight * 0.1);
                    }
                }
            }
        }

        // Still empty? Return empty string (let caller handle)
        if concept_activations.is_empty() {
            return (String::new(), 0.0);
        }

        // Step 2: Generate tokens from concept activations
        let mut generated_tokens = Vec::new();
        let mut total_confidence = 0.0;

        for _ in 0..self.max_tokens {
            let token_probs = self.token_network.generate_probabilities(&concept_activations);
            
            if token_probs.is_empty() {
                break;
            }

            // Sample next token
            if let Some(token) = self.token_network.sample_token(&token_probs, self.temperature) {
                let confidence = token_probs.get(&token).copied().unwrap_or(0.0);
                total_confidence += confidence;
                generated_tokens.push(token);

                // Stop if we generated end-of-sequence indicators
                if generated_tokens.len() > 5 && confidence < 0.1 {
                    break;
                }
            } else {
                break;
            }
        }

        if generated_tokens.is_empty() {
            // Return empty string, let caller handle
            return (String::new(), 0.0);
        }

        // Join tokens and calculate confidence
        let text = generated_tokens.join(" ");
        let confidence = if generated_tokens.len() > 0 {
            (total_confidence / generated_tokens.len() as f64).min(1.0)
        } else {
            0.0
        };

        (text, confidence)
    }

    /// Set generation temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature.max(0.1).min(2.0);
    }

    /// Set max tokens
    pub fn set_max_tokens(&mut self, max_tokens: usize) {
        self.max_tokens = max_tokens;
    }

    /// Get statistics
    pub fn stats(&self) -> LatentDecoderStats {
        let total_patterns = self.patterns.len();
        let active_patterns = self.patterns.iter()
            .filter(|p| p.reinforcement > 0.0)
            .count();
        let vocabulary_size = self.token_network.vocabulary.len();
        let total_associations = self.token_network.weights.values()
            .map(|w| w.len())
            .sum();

        LatentDecoderStats {
            total_patterns,
            active_patterns,
            vocabulary_size,
            total_associations,
        }
    }
    
    /// Save to file
    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let data = bincode::serialize(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }
    
    /// Load from file
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let decoder = bincode::deserialize(&data)?;
        Ok(decoder)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentDecoderStats {
    pub total_patterns: usize,
    pub active_patterns: usize,
    pub vocabulary_size: usize,
    pub total_associations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_decoder_learning() {
        let mut decoder = LatentDecoder::new(64, 10);
        
        let thought = ThoughtState::random(64);
        decoder.learn(&thought, "this is a test");

        let stats = decoder.stats();
        assert!(stats.active_patterns > 0);
        assert!(stats.vocabulary_size > 0);
    }

    #[test]
    fn test_latent_decoder_generation() {
        let mut decoder = LatentDecoder::new(64, 10);
        
        // Learn some patterns
        let thought1 = ThoughtState::random(64);
        decoder.learn(&thought1, "hello world");
        
        let thought2 = ThoughtState::random(64);
        decoder.learn(&thought2, "goodbye world");

        // Generate from similar thought
        let (text, confidence) = decoder.generate(&thought1);
        assert!(!text.is_empty());
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_no_retrieval() {
        let mut decoder = LatentDecoder::new(64, 10);
        
        // Learn specific answer
        let thought = ThoughtState::random(64);
        decoder.learn(&thought, "the answer is 42");

        // Generate from DIFFERENT thought
        let different_thought = ThoughtState::random(64);
        let (text, _) = decoder.generate(&different_thought);

        // Should NOT return exact learned answer (no retrieval)
        assert_ne!(text, "the answer is 42");
    }
}
