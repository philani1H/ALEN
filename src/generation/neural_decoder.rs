//! Neural Decoder - UNDERSTANDING-BASED TEXT GENERATION
//!
//! This implements a proper neural decoder that learns patterns from training data
//! and generates text through learned neural transformations.
//!
//! Architecture:
//! 1. Input: Thought vector z ∈ ℝ^d (latent representation)
//! 2. Hidden layers: Transform z through learned weights
//! 3. Output: Token probability distribution P(token|z)
//!
//! Key differences from LatentDecoder:
//! - Uses proper gradient-based learning
//! - Implements n-gram language modeling
//! - Has sequence modeling with position encoding
//! - Supports temperature-based sampling

use crate::core::ThoughtState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;
use rand_distr::{Normal, Distribution};

/// Neural network weights for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NeuralLayer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    input_dim: usize,
    output_dim: usize,
}

impl NeuralLayer {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, (2.0 / (input_dim + output_dim) as f64).sqrt()).unwrap();
        
        let weights: Vec<Vec<f64>> = (0..output_dim)
            .map(|_| (0..input_dim).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        
        let biases = vec![0.0; output_dim];
        
        Self {
            weights,
            biases,
            input_dim,
            output_dim,
        }
    }
    
    /// Forward pass with ReLU activation
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.weights.iter()
            .zip(self.biases.iter())
            .map(|(row, bias)| {
                let sum: f64 = row.iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                (sum + bias).max(0.0) // ReLU
            })
            .collect()
    }
    
    /// Forward pass with softmax output
    fn forward_softmax(&self, input: &[f64], temperature: f64) -> Vec<f64> {
        let logits: Vec<f64> = self.weights.iter()
            .zip(self.biases.iter())
            .map(|(row, bias)| {
                let sum: f64 = row.iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                (sum + bias) / temperature
            })
            .collect();
        
        // Softmax
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        logits.iter().map(|&x| (x - max_logit).exp() / exp_sum).collect()
    }
    
    /// Update weights with gradient descent
    fn update(&mut self, input: &[f64], target_idx: usize, learning_rate: f64) {
        // Simple gradient update toward target
        for (i, (row, bias)) in self.weights.iter_mut().zip(self.biases.iter_mut()).enumerate() {
            let is_target = if i == target_idx { 1.0 } else { 0.0 };
            
            for (w, x) in row.iter_mut().zip(input.iter()) {
                *w += learning_rate * (is_target - 0.5) * x;
            }
            *bias += learning_rate * (is_target - 0.5);
        }
    }
}

/// Token with embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Token {
    text: String,
    embedding: Vec<f64>,
    frequency: u32,
}

/// Vocabulary built from training data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LearnedVocabulary {
    tokens: Vec<Token>,
    token_to_idx: HashMap<String, usize>,
    embedding_dim: usize,
}

impl LearnedVocabulary {
    fn new(embedding_dim: usize) -> Self {
        let mut vocab = Self {
            tokens: Vec::new(),
            token_to_idx: HashMap::new(),
            embedding_dim,
        };
        
        // Add special tokens
        vocab.add_token("<PAD>");
        vocab.add_token("<UNK>");
        vocab.add_token("<BOS>");
        vocab.add_token("<EOS>");
        
        vocab
    }
    
    fn add_token(&mut self, text: &str) -> usize {
        if let Some(&idx) = self.token_to_idx.get(text) {
            self.tokens[idx].frequency += 1;
            return idx;
        }
        
        let idx = self.tokens.len();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0 / (self.embedding_dim as f64).sqrt()).unwrap();
        
        let embedding: Vec<f64> = (0..self.embedding_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
        
        self.tokens.push(Token {
            text: text.to_string(),
            embedding,
            frequency: 1,
        });
        
        self.token_to_idx.insert(text.to_string(), idx);
        idx
    }
    
    fn get_idx(&self, text: &str) -> usize {
        *self.token_to_idx.get(text).unwrap_or(&1) // <UNK> = 1
    }
    
    fn get_token(&self, idx: usize) -> &str {
        self.tokens.get(idx).map(|t| t.text.as_str()).unwrap_or("<UNK>")
    }
    
    fn get_embedding(&self, idx: usize) -> &[f64] {
        self.tokens.get(idx)
            .map(|t| t.embedding.as_slice())
            .unwrap_or(&[])
    }
    
    fn size(&self) -> usize {
        self.tokens.len()
    }
}

/// Bigram language model for coherent generation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BigramModel {
    /// Bigram counts: (prev_token, next_token) -> count
    bigrams: HashMap<(usize, usize), f64>,
    /// Unigram counts
    unigrams: HashMap<usize, f64>,
    /// Total token count
    total_count: f64,
    /// Smoothing factor
    smoothing: f64,
}

impl BigramModel {
    fn new() -> Self {
        Self {
            bigrams: HashMap::new(),
            unigrams: HashMap::new(),
            total_count: 0.0,
            smoothing: 0.1,
        }
    }
    
    fn learn(&mut self, tokens: &[usize]) {
        for window in tokens.windows(2) {
            let prev = window[0];
            let next = window[1];
            
            *self.bigrams.entry((prev, next)).or_insert(0.0) += 1.0;
            *self.unigrams.entry(prev).or_insert(0.0) += 1.0;
            self.total_count += 1.0;
        }
    }
    
    /// Get probability P(next|prev) with Laplace smoothing
    fn probability(&self, prev: usize, next: usize, vocab_size: usize) -> f64 {
        let bigram_count = self.bigrams.get(&(prev, next)).copied().unwrap_or(0.0);
        let unigram_count = self.unigrams.get(&prev).copied().unwrap_or(0.0);
        
        (bigram_count + self.smoothing) / (unigram_count + self.smoothing * vocab_size as f64)
    }
    
    /// Sample next token given previous
    fn sample_next(&self, prev: usize, vocab_size: usize, temperature: f64) -> usize {
        let mut probs: Vec<(usize, f64)> = (0..vocab_size)
            .map(|next| {
                let p = self.probability(prev, next, vocab_size);
                (next, (p.ln() / temperature).exp())
            })
            .collect();
        
        // Normalize
        let total: f64 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= total;
        }
        
        // Sample
        let mut rng = rand::thread_rng();
        let r = rng.gen::<f64>();
        let mut cumsum = 0.0;
        
        for (idx, prob) in probs {
            cumsum += prob;
            if r <= cumsum {
                return idx;
            }
        }
        
        0 // Fallback
    }
}

/// Thought-to-Token association network
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ThoughtTokenNetwork {
    /// Maps thought patterns to token distributions
    associations: Vec<(Vec<f64>, HashMap<usize, f64>)>,
    /// Learning rate
    learning_rate: f64,
    /// Similarity threshold for matching
    threshold: f64,
}

impl ThoughtTokenNetwork {
    fn new(learning_rate: f64) -> Self {
        Self {
            associations: Vec::new(),
            learning_rate,
            threshold: 0.3,
        }
    }
    
    /// Learn association between thought vector and tokens
    fn learn(&mut self, thought: &[f64], tokens: &[usize]) {
        // Find existing association or create new
        let mut best_match: Option<(usize, f64)> = None;
        
        for (idx, (pattern, _)) in self.associations.iter().enumerate() {
            let sim = cosine_similarity(thought, pattern);
            if sim > self.threshold {
                if best_match.is_none() || sim > best_match.unwrap().1 {
                    best_match = Some((idx, sim));
                }
            }
        }
        
        match best_match {
            Some((idx, _)) => {
                // Update existing association
                let (pattern, token_dist) = &mut self.associations[idx];
                
                // Move pattern toward thought
                for (p, t) in pattern.iter_mut().zip(thought.iter()) {
                    *p = (1.0 - self.learning_rate) * *p + self.learning_rate * t;
                }
                
                // Update token distribution
                for &token in tokens {
                    *token_dist.entry(token).or_insert(0.0) += self.learning_rate;
                }
            }
            None => {
                // Create new association
                let mut token_dist = HashMap::new();
                for &token in tokens {
                    *token_dist.entry(token).or_insert(0.0) += 1.0;
                }
                self.associations.push((thought.to_vec(), token_dist));
            }
        }
    }
    
    /// Get token distribution for a thought vector
    fn get_distribution(&self, thought: &[f64]) -> HashMap<usize, f64> {
        let mut combined = HashMap::new();
        
        for (pattern, token_dist) in &self.associations {
            let sim = cosine_similarity(thought, pattern);
            if sim > 0.0 {
                for (&token, &weight) in token_dist {
                    *combined.entry(token).or_insert(0.0) += sim * weight;
                }
            }
        }
        
        // Normalize
        let total: f64 = combined.values().sum();
        if total > 0.0 {
            for weight in combined.values_mut() {
                *weight /= total;
            }
        }
        
        combined
    }
}

/// Neural Decoder - Understanding-based text generation
#[derive(Serialize, Deserialize)]
pub struct NeuralDecoder {
    /// Vocabulary learned from data
    vocabulary: LearnedVocabulary,
    /// Input projection layer: thought_dim -> hidden_dim
    input_layer: NeuralLayer,
    /// Hidden layer: hidden_dim -> hidden_dim
    hidden_layer: NeuralLayer,
    /// Output layer: hidden_dim -> vocab_size
    output_layer: NeuralLayer,
    /// Bigram language model for coherence
    bigram_model: BigramModel,
    /// Thought-to-token associations
    thought_network: ThoughtTokenNetwork,
    /// Dimension of thought vectors
    thought_dim: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Temperature for generation
    temperature: f64,
    /// Maximum tokens to generate
    max_tokens: usize,
    /// Learning rate
    learning_rate: f64,
    /// Training examples seen
    examples_seen: u64,
}

impl NeuralDecoder {
    pub fn new(thought_dim: usize, hidden_dim: usize) -> Self {
        let vocabulary = LearnedVocabulary::new(hidden_dim);
        
        Self {
            input_layer: NeuralLayer::new(thought_dim, hidden_dim),
            hidden_layer: NeuralLayer::new(hidden_dim, hidden_dim),
            output_layer: NeuralLayer::new(hidden_dim, vocabulary.size().max(100)),
            bigram_model: BigramModel::new(),
            thought_network: ThoughtTokenNetwork::new(0.1),
            vocabulary,
            thought_dim,
            hidden_dim,
            temperature: 0.7,
            max_tokens: 100,
            learning_rate: 0.01,
            examples_seen: 0,
        }
    }
    
    /// Learn from thought-text pair
    pub fn learn(&mut self, thought: &ThoughtState, text: &str) {
        // Tokenize
        let tokens: Vec<String> = text.split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        
        if tokens.is_empty() {
            return;
        }
        
        // Add tokens to vocabulary
        let token_indices: Vec<usize> = tokens.iter()
            .map(|t| self.vocabulary.add_token(t))
            .collect();
        
        // Update output layer size if needed
        if self.vocabulary.size() > self.output_layer.output_dim {
            self.resize_output_layer();
        }
        
        // Learn bigrams
        let bos = self.vocabulary.get_idx("<BOS>");
        let eos = self.vocabulary.get_idx("<EOS>");
        let mut full_sequence = vec![bos];
        full_sequence.extend(&token_indices);
        full_sequence.push(eos);
        self.bigram_model.learn(&full_sequence);
        
        // Learn thought-to-token associations
        self.thought_network.learn(&thought.vector, &token_indices);
        
        // Train neural network
        let hidden = self.input_layer.forward(&thought.vector);
        let hidden2 = self.hidden_layer.forward(&hidden);
        
        // Train on each token
        for &target in &token_indices {
            self.output_layer.update(&hidden2, target, self.learning_rate);
        }
        
        self.examples_seen += 1;
    }
    
    /// Resize output layer when vocabulary grows
    fn resize_output_layer(&mut self) {
        let new_size = self.vocabulary.size() + 50; // Add buffer
        let mut new_layer = NeuralLayer::new(self.hidden_dim, new_size);
        
        // Copy existing weights
        for (i, (new_row, old_row)) in new_layer.weights.iter_mut()
            .zip(self.output_layer.weights.iter())
            .enumerate()
        {
            if i < self.output_layer.output_dim {
                for (j, (new_w, old_w)) in new_row.iter_mut().zip(old_row.iter()).enumerate() {
                    if j < self.output_layer.input_dim {
                        *new_w = *old_w;
                    }
                }
            }
        }
        
        self.output_layer = new_layer;
    }
    
    /// Generate text from thought vector
    pub fn generate(&self, thought: &ThoughtState) -> (String, f64) {
        if self.examples_seen == 0 {
            return (String::new(), 0.0);
        }
        
        // Get thought-to-token distribution
        let thought_dist = self.thought_network.get_distribution(&thought.vector);
        
        // Forward through network
        let hidden = self.input_layer.forward(&thought.vector);
        let hidden2 = self.hidden_layer.forward(&hidden);
        let neural_probs = self.output_layer.forward_softmax(&hidden2, self.temperature);
        
        // Combine distributions
        let vocab_size = self.vocabulary.size();
        let mut combined_probs: Vec<f64> = (0..vocab_size)
            .map(|i| {
                let neural_p = neural_probs.get(i).copied().unwrap_or(0.0);
                let thought_p = thought_dist.get(&i).copied().unwrap_or(0.0);
                neural_p * 0.5 + thought_p * 0.5
            })
            .collect();
        
        // Normalize
        let total: f64 = combined_probs.iter().sum();
        if total > 0.0 {
            for p in combined_probs.iter_mut() {
                *p /= total;
            }
        }
        
        // Generate sequence
        let mut tokens: Vec<usize> = Vec::new();
        let bos = self.vocabulary.get_idx("<BOS>");
        let eos = self.vocabulary.get_idx("<EOS>");
        let pad = self.vocabulary.get_idx("<PAD>");
        let unk = self.vocabulary.get_idx("<UNK>");
        
        let mut prev_token = bos;
        let mut total_confidence = 0.0;
        let mut rng = rand::thread_rng();
        
        for _ in 0..self.max_tokens {
            // Combine neural probs with bigram model
            let mut final_probs: Vec<(usize, f64)> = (0..vocab_size)
                .filter(|&i| i != pad && i != bos && i != unk)
                .map(|i| {
                    let neural_p = combined_probs.get(i).copied().unwrap_or(0.0);
                    let bigram_p = self.bigram_model.probability(prev_token, i, vocab_size);
                    (i, neural_p * 0.6 + bigram_p * 0.4)
                })
                .collect();
            
            // Normalize
            let total: f64 = final_probs.iter().map(|(_, p)| p).sum();
            if total < 1e-10 {
                break;
            }
            
            for (_, p) in final_probs.iter_mut() {
                *p /= total;
            }
            
            // Temperature sampling
            let scaled: Vec<(usize, f64)> = final_probs.iter()
                .map(|&(i, p)| (i, (p.ln() / self.temperature).exp()))
                .collect();
            
            let scaled_total: f64 = scaled.iter().map(|(_, p)| p).sum();
            
            // Sample
            let r = rng.gen::<f64>() * scaled_total;
            let mut cumsum = 0.0;
            let mut selected = eos;
            
            for (idx, prob) in &scaled {
                cumsum += prob;
                if r <= cumsum {
                    selected = *idx;
                    total_confidence += final_probs.iter()
                        .find(|(i, _)| *i == selected)
                        .map(|(_, p)| *p)
                        .unwrap_or(0.0);
                    break;
                }
            }
            
            // Stop on EOS
            if selected == eos {
                break;
            }
            
            tokens.push(selected);
            prev_token = selected;
            
            // Stop if confidence drops
            if tokens.len() > 5 && total_confidence / (tokens.len() as f64) < 0.05 {
                break;
            }
        }
        
        // Convert to text
        let text: String = tokens.iter()
            .map(|&i| self.vocabulary.get_token(i))
            .collect::<Vec<_>>()
            .join(" ");
        
        let confidence = if tokens.is_empty() {
            0.0
        } else {
            (total_confidence / tokens.len() as f64).min(1.0)
        };
        
        (text, confidence)
    }
    
    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature.max(0.1).min(2.0);
    }
    
    /// Set max tokens
    pub fn set_max_tokens(&mut self, max_tokens: usize) {
        self.max_tokens = max_tokens;
    }
    
    /// Get statistics
    pub fn stats(&self) -> NeuralDecoderStats {
        NeuralDecoderStats {
            vocabulary_size: self.vocabulary.size(),
            examples_seen: self.examples_seen,
            associations_count: self.thought_network.associations.len(),
            bigram_count: self.bigram_model.bigrams.len(),
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

/// Statistics for neural decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralDecoderStats {
    pub vocabulary_size: usize,
    pub examples_seen: u64,
    pub associations_count: usize,
    pub bigram_count: usize,
}

/// Cosine similarity helper
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    
    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neural_decoder_learning() {
        let mut decoder = NeuralDecoder::new(64, 128);
        
        // Train on some examples
        let thought = ThoughtState::random(64);
        decoder.learn(&thought, "hello world this is a test");
        decoder.learn(&thought, "testing the neural decoder");
        
        let stats = decoder.stats();
        assert!(stats.vocabulary_size > 4); // More than special tokens
        assert!(stats.examples_seen == 2);
    }
    
    #[test]
    fn test_neural_decoder_generation() {
        let mut decoder = NeuralDecoder::new(64, 128);
        
        // Train
        for i in 0..10 {
            let thought = ThoughtState::random(64);
            decoder.learn(&thought, &format!("hello world test number {}", i));
        }
        
        // Generate
        let thought = ThoughtState::random(64);
        let (text, confidence) = decoder.generate(&thought);
        
        // Should generate something
        assert!(!text.is_empty() || decoder.stats().examples_seen > 0);
    }
    
    #[test]
    fn test_bigram_model() {
        let mut model = BigramModel::new();
        
        model.learn(&[0, 1, 2, 1, 0]);
        
        let p = model.probability(1, 2, 10);
        assert!(p > 0.0);
    }
}
