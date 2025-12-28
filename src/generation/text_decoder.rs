//! Text Decoder - Implements p_t = softmax(W_out·h_t + b)
//!
//! Converts latent thought vectors h_t ∈ ℝ^d into text tokens through
//! learned transformation matrices and softmax activation.

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{WeightedIndex, Distribution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Vocabulary for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    /// Token to ID mapping
    pub token_to_id: HashMap<String, usize>,
    /// ID to token mapping
    pub id_to_token: HashMap<usize, String>,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl Vocabulary {
    /// Create a poetry-focused vocabulary
    pub fn poetry_vocab() -> Self {
        let tokens = vec![
            // Poetic words
            "the", "a", "an", "and", "of", "in", "to", "with", "for",
            "love", "heart", "soul", "dream", "night", "day", "light", "dark",
            "moon", "sun", "star", "sky", "sea", "ocean", "wind", "fire",
            "time", "life", "death", "hope", "fear", "joy", "pain", "peace",
            "beauty", "truth", "lie", "whisper", "silence", "voice", "song",
            "dance", "tears", "smile", "eyes", "hand", "touch", "kiss",
            "rose", "flower", "garden", "spring", "summer", "fall", "winter",
            "eternal", "fleeting", "gentle", "fierce", "wild", "soft", "warm",
            "cold", "bright", "dim", "sweet", "bitter", "pure", "broken",
            "lost", "found", "alone", "together", "forever", "never", "always",
            "flows", "burns", "fades", "shines", "whispers", "echoes", "dances",
            "beneath", "above", "beyond", "through", "across", "within",
            "like", "as", "though", "yet", "still", "once", "now", "then",
            // Punctuation
            ".", ",", "!", "?", ";", ":",
            // Special tokens
            "<START>", "<END>", "<UNK>",
        ];

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (id, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.to_string(), id);
            id_to_token.insert(id, token.to_string());
        }

        Self {
            token_to_id,
            id_to_token,
            vocab_size: tokens.len(),
        }
    }

    /// Get token ID
    pub fn get_id(&self, token: &str) -> usize {
        *self.token_to_id.get(token).unwrap_or(&self.token_to_id["<UNK>"])
    }

    /// Get token from ID
    pub fn get_token(&self, id: usize) -> String {
        self.id_to_token.get(&id).cloned().unwrap_or("<UNK>".to_string())
    }
}

/// Text decoder - transforms thought vectors into tokens
/// Implements: p_t = softmax(W_out·h_t + b_out)
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
    /// Vocabulary
    pub vocab: Vocabulary,
    /// Temperature for sampling (higher = more random)
    pub temperature: f64,
}

impl TextDecoder {
    /// Create a new text decoder
    pub fn new(d_model: usize, temperature: f64) -> Self {
        let vocab = Vocabulary::poetry_vocab();
        let vocab_size = vocab.vocab_size;

        // Initialize W_out with small random values
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
        }
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

    /// Generate text autoregressively
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

            // Update h_t based on generated token (simplified)
            // In full implementation, this would use attention/RNN
            h_t = self.update_hidden_state(&h_t, token_id);
        }

        // Convert tokens to text
        tokens.iter()
            .map(|&id| self.vocab.get_token(id))
            .filter(|t| t != "<START>" && t != "<END>")
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Update hidden state after generating token (simplified)
    fn update_hidden_state(&self, h_t: &[f64], token_id: usize) -> Vec<f64> {
        // Simplified: mix current state with token embedding
        let token_influence = (token_id as f64 / self.vocab.vocab_size as f64) * 0.1;

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
        let vocab = Vocabulary::poetry_vocab();
        assert!(vocab.vocab_size > 0);
        assert_eq!(vocab.get_id("love"), vocab.get_id("love")); // Deterministic
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
    }

    #[test]
    fn test_token_generation() {
        let mut decoder = TextDecoder::new(128, 1.0);
        let h_t = vec![0.5; 128]; // Random hidden state

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

        // Should generate some text
        assert!(!text.is_empty());
        println!("Generated text: {}", text);
    }
}
