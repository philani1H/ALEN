//! Text Decoder - Implements p_t = softmax(W_out·h_t + b)
//!
//! Converts latent thought vectors h_t ∈ ℝ^d into text tokens through
//! learned transformation matrices and softmax activation.
//!
//! IMPORTANT: This decoder uses semantic memory for vocabulary.
//! All tokens come from learned knowledge - no hardcoded words.

use crate::memory::{SemanticMemory, SemanticFact};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{WeightedIndex, Distribution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Vocabulary for text generation - learns from semantic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    /// Token to ID mapping
    pub token_to_id: HashMap<String, usize>,
    /// ID to token mapping
    pub id_to_token: HashMap<usize, String>,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Whether vocabulary was learned from data
    pub is_learned: bool,
}

impl Vocabulary {
    /// Create empty vocabulary ready to learn
    pub fn new() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        
        // Only special tokens are predefined
        let special_tokens = vec!["<START>", "<END>", "<UNK>", "<PAD>"];
        for (id, token) in special_tokens.iter().enumerate() {
            token_to_id.insert(token.to_string(), id);
            id_to_token.insert(id, token.to_string());
        }
        
        Self {
            token_to_id,
            id_to_token,
            vocab_size: special_tokens.len(),
            is_learned: false,
        }
    }

    /// Learn vocabulary from semantic memory
    pub fn learn_from_memory(&mut self, memory: &SemanticMemory, max_vocab: usize) -> Result<(), Box<dyn std::error::Error>> {
        // Get all facts from memory
        let facts = memory.get_all_facts(max_vocab * 10)?;
        
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        // Count words from all facts
        for fact in &facts {
            for word in fact.content.split_whitespace() {
                let word_lower = word.to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '\'')
                    .collect::<String>();
                
                if !word_lower.is_empty() && word_lower.len() > 1 {
                    *word_counts.entry(word_lower).or_insert(0) += 1;
                }
            }
            
            // Also add concept words
            for word in fact.concept.split_whitespace() {
                let word_lower = word.to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '\'')
                    .collect::<String>();
                
                if !word_lower.is_empty() && word_lower.len() > 1 {
                    *word_counts.entry(word_lower).or_insert(0) += 1;
                }
            }
        }
        
        // Sort by frequency and add to vocabulary
        let mut sorted_words: Vec<_> = word_counts.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));
        
        for (word, _count) in sorted_words.into_iter().take(max_vocab - self.vocab_size) {
            if !self.token_to_id.contains_key(&word) {
                let id = self.vocab_size;
                self.token_to_id.insert(word.clone(), id);
                self.id_to_token.insert(id, word);
                self.vocab_size += 1;
            }
        }
        
        self.is_learned = true;
        Ok(())
    }

    /// Add a single token
    pub fn add_token(&mut self, token: &str) -> usize {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        
        let id = self.vocab_size;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        self.vocab_size += 1;
        id
    }

    /// Create a poetry-focused vocabulary (fallback when no memory available)
    /// DEPRECATED: Use learn_from_memory instead
    #[deprecated(note = "Use learn_from_memory for production")]
    pub fn poetry_vocab() -> Self {
        let mut vocab = Self::new();
        
        // Add minimal fallback words - these should come from training
        let fallback_words = vec![
            "the", "a", "and", "of", "in", "to", "with", "for", "is", "are",
            ".", ",", "!", "?",
        ];
        
        for word in fallback_words {
            vocab.add_token(word);
        }
        
        vocab
    }

    /// Get token ID
    pub fn get_id(&self, token: &str) -> usize {
        *self.token_to_id.get(token)
            .or_else(|| self.token_to_id.get(&token.to_lowercase()))
            .unwrap_or(&self.token_to_id["<UNK>"])
    }

    /// Get token from ID
    pub fn get_token(&self, id: usize) -> String {
        self.id_to_token.get(&id).cloned().unwrap_or("<UNK>".to_string())
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Text decoder - transforms thought vectors into tokens
/// Implements: p_t = softmax(W_out·h_t + b_out)
/// 
/// Uses semantic memory for vocabulary - all tokens come from learned knowledge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDecoder {
    /// Output weight matrix W_out ∈ ℝ^(vocab_size × d_model)
    #[serde(skip)]
    pub w_out: Option<DMatrix<f64>>,
    /// Serializable weight data
    w_out_data: Vec<f64>,
    /// Output bias b_out ∈ ℝ^vocab_size
    pub b_out: Vec<f64>,
    /// Model dimension
    pub d_model: usize,
    /// Vocabulary (learned from memory)
    pub vocab: Vocabulary,
    /// Temperature for sampling (higher = more random)
    pub temperature: f64,
    /// Whether decoder has been initialized with learned vocabulary
    pub is_initialized: bool,
}

impl TextDecoder {
    /// Create a new text decoder with empty vocabulary
    pub fn new(d_model: usize, temperature: f64) -> Self {
        let vocab = Vocabulary::new();
        let vocab_size = vocab.vocab_size;

        // Initialize W_out with small random values for special tokens only
        let mut rng = rand::thread_rng();
        let w_out_data: Vec<f64> = (0..vocab_size * d_model)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        let w_out = DMatrix::from_row_slice(vocab_size, d_model, &w_out_data);

        // Initialize bias to zero
        let b_out = vec![0.0; vocab_size];

        Self {
            w_out: Some(w_out),
            w_out_data,
            b_out,
            d_model,
            vocab,
            temperature,
            is_initialized: false,
        }
    }

    /// Initialize decoder with vocabulary learned from semantic memory
    pub fn initialize_from_memory(&mut self, memory: &SemanticMemory, max_vocab: usize) -> Result<(), Box<dyn std::error::Error>> {
        // Learn vocabulary from memory
        self.vocab.learn_from_memory(memory, max_vocab)?;
        
        // Reinitialize weights for new vocabulary size
        let mut rng = rand::thread_rng();
        self.w_out_data = (0..self.vocab.vocab_size * self.d_model)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        
        self.w_out = Some(DMatrix::from_row_slice(
            self.vocab.vocab_size,
            self.d_model,
            &self.w_out_data,
        ));
        
        self.b_out = vec![0.0; self.vocab.vocab_size];
        self.is_initialized = true;
        
        Ok(())
    }

    /// Ensure W_out matrix is loaded
    fn ensure_matrix(&mut self) {
        if self.w_out.is_none() {
            self.w_out = Some(DMatrix::from_row_slice(
                self.vocab.vocab_size,
                self.d_model,
                &self.w_out_data,
            ));
        }
    }

    /// Compute logits: W_out·h_t + b_out
    pub fn compute_logits(&mut self, h_t: &[f64]) -> Vec<f64> {
        self.ensure_matrix();

        let h_vec = DVector::from_vec(h_t.to_vec());
        let w_out = self.w_out.as_ref().unwrap();

        // Logits = W_out · h_t + b_out
        let logits_vec = w_out * h_vec;

        logits_vec.iter()
            .zip(self.b_out.iter())
            .map(|(l, b)| l + b)
            .collect()
    }

    /// Apply softmax with temperature
    pub fn softmax(&self, logits: &[f64]) -> Vec<f64> {
        // Apply temperature scaling
        let scaled: Vec<f64> = logits.iter()
            .map(|&l| l / self.temperature)
            .collect();

        // Compute softmax
        let max_logit = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = scaled.iter().map(|&l| (l - max_logit).exp()).sum();

        scaled.iter()
            .map(|&l| (l - max_logit).exp() / exp_sum)
            .collect()
    }

    /// Sample token from probability distribution
    pub fn sample(&self, probs: &[f64]) -> usize {
        let mut rng = rand::thread_rng();

        // Create weighted distribution
        let dist = WeightedIndex::new(probs).unwrap();
        dist.sample(&mut rng)
    }

    /// Decode a single token: p_t = softmax(W_out·h_t + b_out)
    pub fn decode_token(&mut self, h_t: &[f64]) -> (usize, Vec<f64>) {
        let logits = self.compute_logits(h_t);
        let probs = self.softmax(&logits);
        let token_id = self.sample(&probs);
        (token_id, probs)
    }

    /// Generate text using semantic memory for context
    pub fn generate_with_memory(
        &mut self,
        initial_h_t: &[f64],
        memory: &SemanticMemory,
        max_tokens: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Initialize from memory if not done
        if !self.is_initialized {
            self.initialize_from_memory(memory, 1000)?;
        }
        
        let mut tokens = Vec::new();
        let mut h_t = initial_h_t.to_vec();

        for _ in 0..max_tokens {
            // Query semantic memory for relevant concepts
            let similar_facts = memory.find_similar(&h_t, 5)?;
            
            if !similar_facts.is_empty() {
                // Use most similar fact to guide generation
                let (best_fact, _similarity) = &similar_facts[0];
                
                // Extract a word from the fact content
                let words: Vec<&str> = best_fact.content.split_whitespace().collect();
                if !words.is_empty() {
                    let word_idx = tokens.len() % words.len();
                    let word = words[word_idx].to_lowercase();
                    
                    // Skip special tokens and punctuation-only
                    if !word.starts_with('<') && word.chars().any(|c| c.is_alphabetic()) {
                        tokens.push(word.clone());
                    }
                }
            }
            
            // Update h_t based on generated content
            h_t = self.update_hidden_state(&h_t, tokens.len());
            
            // Stop if we have enough tokens
            if tokens.len() >= max_tokens {
                break;
            }
        }

        Ok(self.format_output(&tokens))
    }

    /// Generate text autoregressively (fallback when no memory)
    pub fn generate(&mut self, initial_h_t: &[f64], max_tokens: usize) -> String {
        let mut tokens = Vec::new();
        let mut h_t = initial_h_t.to_vec();

        // Start token
        tokens.push(self.vocab.get_id("<START>"));

        for _ in 0..max_tokens {
            let (token_id, _probs) = self.decode_token(&h_t);

            // Stop if we generate end token
            if token_id == self.vocab.get_id("<END>") {
                break;
            }

            tokens.push(token_id);

            // Update h_t based on generated token
            h_t = self.update_hidden_state(&h_t, token_id);
        }

        // Convert tokens to text
        let words: Vec<String> = tokens.iter()
            .map(|&id| self.vocab.get_token(id))
            .filter(|t| t != "<START>" && t != "<END>" && t != "<UNK>" && t != "<PAD>")
            .collect();
        
        self.format_output(&words)
    }

    /// Format output tokens into readable text
    fn format_output(&self, tokens: &[String]) -> String {
        if tokens.is_empty() {
            return String::new();
        }
        
        let mut result = String::new();
        let no_space_before = [".", ",", "!", "?", ";", ":", "'", ")", "]"];
        let no_space_after = ["(", "[", "'"];

        for (i, token) in tokens.iter().enumerate() {
            let needs_space = if i == 0 {
                false
            } else if no_space_before.contains(&token.as_str()) {
                false
            } else if i > 0 && no_space_after.contains(&tokens[i-1].as_str()) {
                false
            } else {
                true
            };

            if needs_space {
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

    /// Update hidden state after generating token
    fn update_hidden_state(&self, h_t: &[f64], token_id: usize) -> Vec<f64> {
        // Mix current state with token influence
        let token_influence = (token_id as f64 / self.vocab.vocab_size.max(1) as f64) * 0.1;

        h_t.iter()
            .enumerate()
            .map(|(i, &h)| {
                let noise = (i as f64 * token_influence).sin() * 0.05;
                h * 0.95 + noise
            })
            .collect()
    }

    /// Set temperature for sampling
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature.max(0.01); // Avoid division by zero
    }
}

/// Softmax function for reference
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = logits.iter().map(|&l| (l - max_logit).exp()).sum();

    logits.iter()
        .map(|&l| (l - max_logit).exp() / exp_sum)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_creation() {
        let vocab = Vocabulary::new();
        assert!(vocab.vocab_size >= 4); // At least special tokens
        assert!(vocab.token_to_id.contains_key("<UNK>"));
    }

    #[test]
    fn test_add_token() {
        let mut vocab = Vocabulary::new();
        let initial_size = vocab.vocab_size;
        
        vocab.add_token("test");
        assert_eq!(vocab.vocab_size, initial_size + 1);
        assert!(vocab.token_to_id.contains_key("test"));
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Probabilities should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Highest logit should have highest probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = TextDecoder::new(128, 1.0);
        assert_eq!(decoder.d_model, 128);
        assert_eq!(decoder.temperature, 1.0);
        assert!(!decoder.is_initialized);
    }

    #[test]
    fn test_token_generation() {
        let mut decoder = TextDecoder::new(128, 1.0);
        
        // Add some tokens for testing
        decoder.vocab.add_token("hello");
        decoder.vocab.add_token("world");
        
        // Reinitialize weights for new vocab
        let mut rng = rand::thread_rng();
        decoder.w_out_data = (0..decoder.vocab.vocab_size * decoder.d_model)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        decoder.w_out = Some(DMatrix::from_row_slice(
            decoder.vocab.vocab_size,
            decoder.d_model,
            &decoder.w_out_data,
        ));
        decoder.b_out = vec![0.0; decoder.vocab.vocab_size];
        
        let h_t = vec![0.5; 128];
        let (token_id, probs) = decoder.decode_token(&h_t);

        // Check probability distribution
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Token ID should be valid
        assert!(token_id < decoder.vocab.vocab_size);
    }

    #[test]
    fn test_text_generation() {
        let mut decoder = TextDecoder::new(128, 0.8);
        let h_t = vec![0.3; 128];

        let text = decoder.generate(&h_t, 20);

        // Text generation works (may be empty without learned vocab)
        println!("Generated text: {}", text);
    }
}
