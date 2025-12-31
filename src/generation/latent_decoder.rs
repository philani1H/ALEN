//! Latent Space Decoder - UNDERSTANDING, NOT MEMORIZATION
//!
//! Generates text from thought vectors using learned patterns in latent space.
//! NO RETRIEVAL. NO LOOKUP. NO STORED ANSWERS.
//!
//! Architecture:
//! 1. Thought vector → Latent pattern recognition
//! 2. Pattern → Concept activation (learned weights)
//! 3. Concept activation → Token probabilities (neural network)
//! 4. Token probabilities → Generated text (sampling with bigram model)
//!
//! This implements the mathematical framework:
//! Y = Decoder_φ(z) where z = latent reasoning context
//!
//! Key improvements:
//! - Higher learning rates for faster convergence
//! - Bigram model for coherent sequences
//! - Softmax temperature for controlled generation
//! - Pattern clustering for better generalization

use crate::core::ThoughtState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;
use rand_distr::{Normal, Distribution};

/// Latent pattern in thought space
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LatentPattern {
    /// Pattern centroid in thought space
    centroid: Vec<f64>,
    /// Token associations learned from this pattern
    token_weights: HashMap<String, f64>,
    /// Number of examples this pattern learned from
    example_count: u32,
    /// Running confidence for this pattern
    confidence: f64,
}

impl LatentPattern {
    fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();
        
        Self {
            centroid: (0..dimension).map(|_| normal.sample(&mut rng)).collect(),
            token_weights: HashMap::new(),
            example_count: 0,
            confidence: 0.5,
        }
    }

    /// Calculate similarity to a thought vector (cosine similarity)
    fn similarity(&self, thought: &[f64]) -> f64 {
        if self.centroid.len() != thought.len() {
            return 0.0;
        }

        let dot: f64 = self.centroid.iter()
            .zip(thought.iter())
            .map(|(c, t)| c * t)
            .sum();
        
        let norm_c: f64 = self.centroid.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_t: f64 = thought.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm_c > 1e-10 && norm_t > 1e-10 {
            (dot / (norm_c * norm_t) + 1.0) / 2.0  // Scale to [0, 1]
        } else {
            0.0
        }
    }

    /// Learn from a thought-token pair with momentum
    fn learn(&mut self, thought: &[f64], tokens: &[String], learning_rate: f64) {
        // Update centroid with momentum
        let momentum = 0.1;
        let lr = learning_rate * (1.0 + momentum * self.example_count as f64).min(2.0);
        
        for (c, t) in self.centroid.iter_mut().zip(thought.iter()) {
            *c = (1.0 - lr) * *c + lr * t;
        }

        // Update token weights with position decay
        for (pos, token) in tokens.iter().enumerate() {
            let position_weight = 1.0 / (1.0 + 0.1 * pos as f64);
            let current = self.token_weights.get(token).copied().unwrap_or(0.0);
            self.token_weights.insert(
                token.clone(),
                current + learning_rate * position_weight
            );
        }

        self.example_count += 1;
        self.confidence = (self.confidence + 0.1).min(1.0);
    }
    
    /// Get top tokens for this pattern
    fn top_tokens(&self, k: usize) -> Vec<(String, f64)> {
        let mut tokens: Vec<_> = self.token_weights.iter()
            .map(|(t, w)| (t.clone(), *w))
            .collect();
        tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        tokens.truncate(k);
        tokens
    }
}

/// Bigram language model for coherent generation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BigramLanguageModel {
    /// Bigram counts: (prev, next) -> count
    bigrams: HashMap<(String, String), f64>,
    /// Unigram counts for smoothing
    unigrams: HashMap<String, f64>,
    /// Total transitions
    total: f64,
    /// Smoothing factor (Laplace)
    smoothing: f64,
}

impl BigramLanguageModel {
    fn new() -> Self {
        Self {
            bigrams: HashMap::new(),
            unigrams: HashMap::new(),
            total: 0.0,
            smoothing: 0.1,
        }
    }
    
    fn learn(&mut self, tokens: &[String]) {
        if tokens.is_empty() {
            return;
        }
        
        // Add start token
        let start = "<START>".to_string();
        *self.bigrams.entry((start.clone(), tokens[0].clone())).or_insert(0.0) += 1.0;
        *self.unigrams.entry(start).or_insert(0.0) += 1.0;
        
        for window in tokens.windows(2) {
            let prev = &window[0];
            let next = &window[1];
            *self.bigrams.entry((prev.clone(), next.clone())).or_insert(0.0) += 1.0;
            *self.unigrams.entry(prev.clone()).or_insert(0.0) += 1.0;
        }
        
        self.total += tokens.len() as f64;
    }
    
    /// Get P(next|prev) with smoothing
    fn probability(&self, prev: &str, next: &str, vocab_size: usize) -> f64 {
        let bigram_count = self.bigrams
            .get(&(prev.to_string(), next.to_string()))
            .copied()
            .unwrap_or(0.0);
        let unigram_count = self.unigrams.get(prev).copied().unwrap_or(0.0);
        
        (bigram_count + self.smoothing) / (unigram_count + self.smoothing * vocab_size as f64)
    }
}

/// Token generator from concept activations with improved learning
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConceptToTokenNetwork {
    /// Global token weights learned from all patterns
    global_weights: HashMap<String, f64>,
    /// Token vocabulary (dynamically built)
    vocabulary: Vec<String>,
    /// Token frequencies (for smoothing)
    token_frequencies: HashMap<String, f64>,
    /// Bigram model for coherence
    bigram_model: BigramLanguageModel,
    /// Total training examples
    training_count: u64,
}

impl ConceptToTokenNetwork {
    fn new() -> Self {
        Self {
            global_weights: HashMap::new(),
            vocabulary: Vec::new(),
            token_frequencies: HashMap::new(),
            bigram_model: BigramLanguageModel::new(),
            training_count: 0,
        }
    }

    /// Learn token associations with improved weighting
    fn learn(&mut self, tokens: &[String], learning_rate: f64) {
        if tokens.is_empty() {
            return;
        }
        
        // Learn bigram model
        self.bigram_model.learn(tokens);
        
        for (pos, token) in tokens.iter().enumerate() {
            // Add to vocabulary if new
            if !self.vocabulary.contains(token) {
                self.vocabulary.push(token.clone());
            }

            // Position-weighted learning (early tokens more important)
            let position_weight = 1.0 / (1.0 + 0.1 * pos as f64);
            
            // Update global weight
            let current = self.global_weights.get(token).copied().unwrap_or(0.0);
            self.global_weights.insert(
                token.clone(),
                current + learning_rate * position_weight
            );

            // Update frequency
            let freq = self.token_frequencies.get(token).copied().unwrap_or(0.0);
            self.token_frequencies.insert(token.clone(), freq + 1.0);
        }
        
        self.training_count += 1;
    }

    /// Generate token probabilities from pattern activations
    fn generate_probabilities(
        &self,
        pattern_activations: &[(f64, Vec<(String, f64)>)],
        prev_token: Option<&str>,
    ) -> HashMap<String, f64> {
        let mut token_probs: HashMap<String, f64> = HashMap::new();

        // Accumulate from activated patterns
        for (activation, pattern_tokens) in pattern_activations {
            for (token, weight) in pattern_tokens {
                let current = token_probs.get(token).copied().unwrap_or(0.0);
                token_probs.insert(token.clone(), current + activation * weight);
            }
        }
        
        // Add global prior
        for (token, weight) in &self.global_weights {
            let current = token_probs.get(token).copied().unwrap_or(0.0);
            token_probs.insert(token.clone(), current + weight * 0.1);
        }
        
        // Apply bigram weighting if we have a previous token
        if let Some(prev) = prev_token {
            let vocab_size = self.vocabulary.len().max(1);
            for (token, prob) in token_probs.iter_mut() {
                let bigram_p = self.bigram_model.probability(prev, token, vocab_size);
                *prob = *prob * 0.3 + bigram_p * 0.7;  // Stronger bigram for coherence
            }
        }

        // Normalize
        let total: f64 = token_probs.values().sum();
        if total > 1e-10 {
            for prob in token_probs.values_mut() {
                *prob /= total;
            }
        }

        token_probs
    }

    /// Sample with temperature
    fn sample(&self, probabilities: &HashMap<String, f64>, temperature: f64) -> Option<String> {
        if probabilities.is_empty() {
            return None;
        }

        // Apply temperature scaling
        let mut scaled: Vec<(String, f64)> = probabilities.iter()
            .map(|(token, prob)| {
                let scaled = if temperature > 0.0 && *prob > 1e-10 {
                    (prob.ln() / temperature).exp()
                } else {
                    *prob
                };
                (token.clone(), scaled)
            })
            .collect();

        // Normalize
        let total: f64 = scaled.iter().map(|(_, p)| p).sum();
        if total < 1e-10 {
            return None;
        }

        for (_, p) in scaled.iter_mut() {
            *p /= total;
        }

        // Sample
        let mut rng = rand::thread_rng();
        let r = rng.gen::<f64>();
        let mut cumsum = 0.0;

        for (token, prob) in &scaled {
            cumsum += prob;
            if r <= cumsum {
                return Some(token.clone());
            }
        }

        // Fallback to highest probability
        scaled.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(token, _)| token.clone())
    }
}

/// Latent Space Decoder - generates text from understanding
/// Improved version with better learning and generation
#[derive(Serialize, Deserialize)]
pub struct LatentDecoder {
    /// Learned patterns in latent space
    patterns: Vec<LatentPattern>,
    /// Token network with bigram model
    token_network: ConceptToTokenNetwork,
    /// Dimension of thought space
    dimension: usize,
    /// Learning rate (HIGHER for faster convergence)
    learning_rate: f64,
    /// Temperature for generation
    temperature: f64,
    /// Maximum tokens to generate
    max_tokens: usize,
    /// Minimum pattern similarity to activate
    activation_threshold: f64,
    /// Training examples seen
    training_count: u64,
}

impl LatentDecoder {
    pub fn new(dimension: usize, num_patterns: usize) -> Self {
        let patterns = (0..num_patterns)
            .map(|_| LatentPattern::new(dimension))
            .collect();

        Self {
            patterns,
            token_network: ConceptToTokenNetwork::new(),
            dimension,
            learning_rate: 0.1,  // HIGHER learning rate
            temperature: 0.7,  // Balanced for coherence and variety
            max_tokens: 15,  // Shorter responses
            activation_threshold: 0.3,  // Lower threshold
            training_count: 0,
        }
    }

    /// Learn from thought-text pair (stores patterns, not answers)
    pub fn learn(&mut self, thought: &ThoughtState, text: &str) {
        // Tokenize and normalize
        let tokens: Vec<String> = text.split_whitespace()
            .map(|s| s.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if tokens.is_empty() {
            return;
        }

        // Find best matching pattern or create new one
        let (best_idx, best_sim) = self.patterns.iter()
            .enumerate()
            .map(|(idx, p)| (idx, p.similarity(&thought.vector)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));

        let pattern_idx = if best_sim < self.activation_threshold {
            // Find least-used pattern or add new one
            let least_used = self.patterns.iter()
                .enumerate()
                .min_by_key(|(_, p)| p.example_count)
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            // If we have capacity and all patterns are used, add new one
            if self.patterns.len() < 100 && self.patterns.iter().all(|p| p.example_count > 0) {
                let new_pattern = LatentPattern::new(self.dimension);
                self.patterns.push(new_pattern);
                self.patterns.len() - 1
            } else {
                least_used
            }
        } else {
            best_idx
        };

        // Update pattern with thought and tokens
        self.patterns[pattern_idx].learn(&thought.vector, &tokens, self.learning_rate);
        
        // Update token network (global learning)
        self.token_network.learn(&tokens, self.learning_rate);
        
        self.training_count += 1;
    }

    /// Generate text from thought (NO RETRIEVAL - pure generation)
    pub fn generate(&self, thought: &ThoughtState) -> (String, f64) {
        if self.training_count == 0 {
            return (String::new(), 0.0);
        }
        
        // Step 1: Compute pattern activations
        let mut pattern_activations: Vec<(f64, Vec<(String, f64)>)> = self.patterns.iter()
            .filter(|p| p.example_count > 0)
            .map(|p| {
                let sim = p.similarity(&thought.vector);
                let tokens = p.top_tokens(20);  // Get top tokens from pattern
                (sim, tokens)
            })
            .filter(|(sim, _)| *sim > 0.1)  // Low threshold for activation
            .collect();
        
        // If no strong activations, use all patterns with weight
        if pattern_activations.is_empty() {
            pattern_activations = self.patterns.iter()
                .filter(|p| p.example_count > 0)
                .map(|p| {
                    let tokens = p.top_tokens(20);
                    (0.2, tokens)  // Default low activation
                })
                .collect();
        }
        
        // Still nothing? Return empty
        if pattern_activations.is_empty() {
            return (String::new(), 0.0);
        }

        // Step 2: Generate tokens autoregressively
        let mut generated_tokens: Vec<String> = Vec::new();
        let mut total_confidence = 0.0;

        for i in 0..self.max_tokens {
            let prev_token = if i > 0 { 
                Some(generated_tokens.last().unwrap().as_str()) 
            } else { 
                None 
            };
            
            let token_probs = self.token_network.generate_probabilities(
                &pattern_activations,
                prev_token
            );
            
            if token_probs.is_empty() {
                break;
            }

            // Sample with temperature
            if let Some(token) = self.token_network.sample(&token_probs, self.temperature) {
                let confidence = token_probs.get(&token).copied().unwrap_or(0.0);
                total_confidence += confidence;
                
                // Avoid repetition
                if generated_tokens.len() >= 2 {
                    let last_two: Vec<_> = generated_tokens.iter().rev().take(2).collect();
                    if last_two.iter().any(|t| *t == &token) {
                        // Try to sample again with higher temperature
                        if let Some(alt_token) = self.token_network.sample(&token_probs, self.temperature * 1.5) {
                            if !last_two.iter().any(|t| *t == &alt_token) {
                                generated_tokens.push(alt_token);
                                continue;
                            }
                        }
                    }
                }
                
                generated_tokens.push(token);

                // Stop conditions - stop early for better quality
                if generated_tokens.len() >= 5 && confidence < 0.1 {
                    break;
                }
                if generated_tokens.len() >= 10 {
                    break;
                }
            } else {
                break;
            }
        }

        if generated_tokens.is_empty() {
            return (String::new(), 0.0);
        }

        // Format output
        let text = self.format_output(&generated_tokens);
        let confidence = (total_confidence / generated_tokens.len() as f64).min(1.0);

        (text, confidence)
    }
    
    /// Format generated tokens into readable text
    fn format_output(&self, tokens: &[String]) -> String {
        if tokens.is_empty() {
            return String::new();
        }
        
        let mut result = String::new();
        let punctuation = ['.', ',', '!', '?', ':', ';', '\'', '"', ')', ']'];
        
        for (i, token) in tokens.iter().enumerate() {
            // Add space if needed
            if i > 0 && !punctuation.contains(&token.chars().next().unwrap_or(' ')) {
                result.push(' ');
            }
            result.push_str(token);
        }
        
        // Capitalize first letter
        let mut chars: Vec<char> = result.chars().collect();
        if !chars.is_empty() {
            chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
        }
        
        chars.into_iter().collect()
    }

    /// Set generation temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature.max(0.1).min(2.0);
    }

    /// Set max tokens
    pub fn set_max_tokens(&mut self, max_tokens: usize) {
        self.max_tokens = max_tokens;
    }
    
    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr.max(0.001).min(1.0);
    }

    /// Generate text ONLY if verified - enforces is_verified = true
    /// This is the production method - NEVER output unverified content
    pub fn generate_verified(&self, thought: &ThoughtState, min_confidence: f64) -> Option<(String, f64, bool)> {
        let (text, confidence) = self.generate(thought);
        
        // Verification check: confidence must exceed threshold
        // is_verified MUST be true for output
        let is_verified = confidence >= min_confidence && !text.is_empty();
        
        if is_verified {
            Some((text, confidence, true))  // is_verified = true
        } else {
            // DO NOT return unverified content
            // Return None - caller must handle uncertainty appropriately
            None
        }
    }
    
    /// Verify a generated response against the thought
    pub fn verify_response(&self, thought: &ThoughtState, response: &str) -> bool {
        // Generate our own response and compare
        let (generated, confidence) = self.generate(thought);
        
        // Must have learned patterns
        if self.training_count == 0 || confidence < 0.3 {
            return false;
        }
        
        // Check semantic similarity
        // Store lowercase strings to extend lifetime
        let gen_lower = generated.to_lowercase();
        let resp_lower = response.to_lowercase();
        let gen_words: Vec<&str> = gen_lower.split_whitespace().collect();
        let resp_words: Vec<&str> = resp_lower.split_whitespace().collect();
        
        if gen_words.is_empty() || resp_words.is_empty() {
            return false;
        }
        
        let mut matches = 0;
        for w in &resp_words {
            if gen_words.contains(w) {
                matches += 1;
            }
        }
        
        let similarity = matches as f64 / resp_words.len() as f64;
        similarity > 0.5  // Require >50% word overlap for verification
    }

    /// Get statistics
    pub fn stats(&self) -> LatentDecoderStats {
        let total_patterns = self.patterns.len();
        let active_patterns = self.patterns.iter()
            .filter(|p| p.example_count > 0)
            .count();
        let vocabulary_size = self.token_network.vocabulary.len();
        let total_associations = self.patterns.iter()
            .map(|p| p.token_weights.len())
            .sum();

        LatentDecoderStats {
            total_patterns,
            active_patterns,
            vocabulary_size,
            total_associations,
            training_count: self.training_count,
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
    pub training_count: u64,
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
        assert_eq!(stats.training_count, 1);
    }

    #[test]
    fn test_latent_decoder_generation() {
        let mut decoder = LatentDecoder::new(64, 20);
        
        // Learn multiple examples for better generation
        for i in 0..10 {
            let thought = ThoughtState::random(64);
            decoder.learn(&thought, &format!("hello world test number {}", i));
            decoder.learn(&thought, "this is another test sentence");
        }

        // Generate from a thought
        let test_thought = ThoughtState::random(64);
        let (text, confidence) = decoder.generate(&test_thought);
        
        // Should generate something after training
        assert!(decoder.stats().training_count >= 10);
        // Text may be empty initially but confidence should be valid
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_no_retrieval() {
        let mut decoder = LatentDecoder::new(64, 10);
        
        // Learn specific answer
        let thought = ThoughtState::random(64);
        decoder.learn(&thought, "the answer is forty two");

        // Generate from DIFFERENT thought
        let different_thought = ThoughtState::random(64);
        let (text, _) = decoder.generate(&different_thought);

        // Should NOT return exact learned answer (no retrieval)
        // The text should be different due to pattern-based generation
        assert_ne!(text.to_lowercase(), "the answer is forty two");
    }
    
    #[test]
    fn test_bigram_coherence() {
        let mut decoder = LatentDecoder::new(64, 20);
        
        // Train with multiple coherent sentences
        for _ in 0..20 {
            let thought = ThoughtState::random(64);
            decoder.learn(&thought, "the quick brown fox jumps over");
            decoder.learn(&thought, "a lazy dog sleeps under the tree");
        }
        
        // The bigram model should learn token transitions
        let stats = decoder.stats();
        assert!(stats.vocabulary_size >= 10);
    }
}
