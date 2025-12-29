//! Embeddings Module
//!
//! Converts raw data (text, etc.) into the vector space ℝⁿ
//! This is the bridge between symbolic and vector representations.
//! 
//! Uses BPE tokenization for production-grade subword handling.

use crate::core::ThoughtState;
use crate::generation::{BPETokenizer, BPETrainer};
use rand::SeedableRng;
use rand_distr::{Normal, Distribution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Dimension of embeddings
    pub dimension: usize,
    /// Whether to normalize embeddings
    pub normalize: bool,
    /// Vocabulary size for token embeddings
    pub vocab_size: usize,
    /// Use BPE tokenization (recommended for production)
    pub use_bpe: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            normalize: true,
            vocab_size: 10000,
            use_bpe: true,
        }
    }
}

/// The Embedding Engine - transforms raw data to vectors
#[derive(Debug, Clone)]
pub struct EmbeddingEngine {
    /// Configuration
    config: EmbeddingConfig,
    /// Token embeddings (vocabulary)
    token_embeddings: HashMap<String, Vec<f64>>,
    /// Token ID embeddings (for BPE)
    id_embeddings: Vec<Vec<f64>>,
    /// Position encoding
    position_encodings: Vec<Vec<f64>>,
    /// Max sequence length for position encoding
    max_seq_length: usize,
    /// BPE tokenizer (learned from data)
    bpe_tokenizer: Option<BPETokenizer>,
    /// Whether BPE has been trained
    bpe_trained: bool,
}

impl EmbeddingEngine {
    /// Create a new embedding engine
    pub fn new(config: EmbeddingConfig) -> Self {
        let max_seq_length = 512;
        
        // Initialize position encodings (sinusoidal)
        let position_encodings = Self::generate_position_encodings(
            max_seq_length,
            config.dimension,
        );

        // Initialize BPE tokenizer if enabled
        let bpe_tokenizer = if config.use_bpe {
            Some(BPETokenizer::new(config.vocab_size))
        } else {
            None
        };

        // Initialize ID embeddings for BPE vocab
        let id_embeddings = Self::initialize_id_embeddings(config.vocab_size, config.dimension);

        Self {
            config,
            token_embeddings: HashMap::new(),
            id_embeddings,
            position_encodings,
            max_seq_length,
            bpe_tokenizer,
            bpe_trained: false,
        }
    }

    /// Initialize embeddings for token IDs
    fn initialize_id_embeddings(vocab_size: usize, dim: usize) -> Vec<Vec<f64>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0 / (dim as f64).sqrt()).unwrap();
        
        (0..vocab_size)
            .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
            .collect()
    }

    /// Train BPE tokenizer on corpus
    pub fn train_bpe(&mut self, texts: &[String], num_merges: usize) {
        if let Some(ref mut tokenizer) = self.bpe_tokenizer {
            tokenizer.train(texts, num_merges);
            self.bpe_trained = true;
            
            // Reinitialize embeddings for new vocab size
            self.id_embeddings = Self::initialize_id_embeddings(
                tokenizer.vocab_size(),
                self.config.dimension,
            );
        }
    }

    /// Train BPE from a corpus using the trainer
    pub fn train_bpe_from_corpus(&mut self, texts: &[String]) {
        let trainer = BPETrainer::new(self.config.vocab_size)
            .with_min_frequency(2);
        
        self.bpe_tokenizer = Some(trainer.train(texts));
        self.bpe_trained = true;
        
        // Reinitialize embeddings for new vocab size
        if let Some(ref tokenizer) = self.bpe_tokenizer {
            self.id_embeddings = Self::initialize_id_embeddings(
                tokenizer.vocab_size(),
                self.config.dimension,
            );
        }
    }

    /// Generate sinusoidal position encodings
    fn generate_position_encodings(max_len: usize, dim: usize) -> Vec<Vec<f64>> {
        let mut encodings = Vec::with_capacity(max_len);
        
        for pos in 0..max_len {
            let mut encoding = Vec::with_capacity(dim);
            for i in 0..dim {
                let angle = pos as f64 / 10000_f64.powf(2.0 * (i / 2) as f64 / dim as f64);
                if i % 2 == 0 {
                    encoding.push(angle.sin());
                } else {
                    encoding.push(angle.cos());
                }
            }
            encodings.push(encoding);
        }
        
        encodings
    }

    /// Get embedding for token ID (BPE mode)
    fn get_id_embedding(&self, id: usize) -> Vec<f64> {
        if id < self.id_embeddings.len() {
            self.id_embeddings[id].clone()
        } else {
            // Return UNK embedding (ID 1)
            self.id_embeddings.get(1).cloned().unwrap_or_else(|| vec![0.0; self.config.dimension])
        }
    }

    /// Get or create token embedding (fallback for non-BPE mode)
    fn get_token_embedding(&mut self, token: &str) -> Vec<f64> {
        if let Some(emb) = self.token_embeddings.get(token) {
            return emb.clone();
        }

        // Generate deterministic embedding based on token hash
        let hash: u64 = token.bytes()
            .fold(5381u64, |acc, b| acc.wrapping_mul(33).wrapping_add(b as u64));
        
        let mut rng = rand::rngs::StdRng::seed_from_u64(hash);
        let normal = Normal::new(0.0, 1.0 / (self.config.dimension as f64).sqrt()).unwrap();
        
        let embedding: Vec<f64> = (0..self.config.dimension)
            .map(|_| normal.sample(&mut rng))
            .collect();

        // Cache if vocabulary isn't full
        if self.token_embeddings.len() < self.config.vocab_size {
            self.token_embeddings.insert(token.to_string(), embedding.clone());
        }

        embedding
    }

    /// Tokenize text using BPE or fallback
    fn tokenize(&self, text: &str) -> Vec<usize> {
        if let Some(ref tokenizer) = self.bpe_tokenizer {
            tokenizer.encode(text)
        } else {
            // Fallback: character-level tokenization with hash
            text.to_lowercase()
                .chars()
                .map(|c| (c as usize) % self.config.vocab_size)
                .collect()
        }
    }

    /// Tokenize to strings (for backward compatibility)
    fn tokenize_to_strings(&self, text: &str) -> Vec<String> {
        if let Some(ref tokenizer) = self.bpe_tokenizer {
            let ids = tokenizer.encode(text);
            ids.iter()
                .map(|&id| tokenizer.id_to_token(id).to_string())
                .collect()
        } else {
            text.to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        }
    }

    /// Embed a single token
    pub fn embed_token(&mut self, token: &str) -> Vec<f64> {
        if let Some(ref tokenizer) = self.bpe_tokenizer {
            let ids = tokenizer.encode(token);
            if ids.is_empty() {
                return vec![0.0; self.config.dimension];
            }
            // Average embeddings for subword tokens
            let mut result = vec![0.0; self.config.dimension];
            for id in &ids {
                let emb = self.get_id_embedding(*id);
                for (i, v) in emb.iter().enumerate() {
                    result[i] += v;
                }
            }
            for v in &mut result {
                *v /= ids.len() as f64;
            }
            result
        } else {
            self.get_token_embedding(token)
        }
    }

    /// Embed text to a thought state using BPE tokenization
    pub fn embed_text(&mut self, text: &str) -> ThoughtState {
        let token_ids = self.tokenize(text);
        
        if token_ids.is_empty() {
            return ThoughtState::new(self.config.dimension);
        }

        // Average token embeddings with position encoding
        let mut result = vec![0.0; self.config.dimension];
        let num_tokens = token_ids.len().min(self.max_seq_length);

        for (pos, &token_id) in token_ids.iter().take(num_tokens).enumerate() {
            let token_emb = self.get_id_embedding(token_id);
            let pos_enc = &self.position_encodings[pos];

            for i in 0..self.config.dimension {
                // Add token embedding + position encoding
                result[i] += token_emb[i] + pos_enc[i] * 0.1;
            }
        }

        // Average
        for val in &mut result {
            *val /= num_tokens as f64;
        }

        // Normalize if configured
        if self.config.normalize {
            let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for val in &mut result {
                    *val /= norm;
                }
            }
        }

        let mut thought = ThoughtState::new(self.config.dimension);
        thought.vector = result;
        thought.confidence = 0.5; // Base confidence
        thought.metadata.source = Some(text.to_string());
        thought.metadata.timestamp = Some(chrono::Utc::now().timestamp());

        thought
    }

    /// Update embedding for a token ID during training
    pub fn update_embedding(&mut self, token_id: usize, gradient: &[f64], learning_rate: f64) {
        if token_id < self.id_embeddings.len() {
            for (i, &g) in gradient.iter().enumerate() {
                if i < self.config.dimension {
                    self.id_embeddings[token_id][i] -= learning_rate * g;
                }
            }
        }
    }

    /// Get the BPE tokenizer (if available)
    pub fn get_tokenizer(&self) -> Option<&BPETokenizer> {
        self.bpe_tokenizer.as_ref()
    }

    /// Check if BPE is trained
    pub fn is_bpe_trained(&self) -> bool {
        self.bpe_trained
    }

    /// Embed multiple texts and average them (for context)
    pub fn embed_context(&mut self, texts: &[&str]) -> ThoughtState {
        if texts.is_empty() {
            return ThoughtState::new(self.config.dimension);
        }

        let embeddings: Vec<ThoughtState> = texts
            .iter()
            .map(|t| self.embed_text(t))
            .collect();

        // Average all embeddings
        let mut result = vec![0.0; self.config.dimension];
        for emb in &embeddings {
            for (i, val) in emb.vector.iter().enumerate() {
                result[i] += val;
            }
        }

        for val in &mut result {
            *val /= embeddings.len() as f64;
        }

        // Normalize
        let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for val in &mut result {
                *val /= norm;
            }
        }

        let mut thought = ThoughtState::new(self.config.dimension);
        thought.vector = result;
        thought.confidence = 0.5;

        thought
    }

    /// Calculate similarity between two texts
    pub fn similarity(&mut self, text1: &str, text2: &str) -> f64 {
        let emb1 = self.embed_text(text1);
        let emb2 = self.embed_text(text2);
        emb1.cosine_similarity(&emb2)
    }

    /// Find most similar text from candidates
    pub fn find_most_similar(&mut self, query: &str, candidates: &[&str]) -> Option<(usize, f64)> {
        if candidates.is_empty() {
            return None;
        }

        let query_emb = self.embed_text(query);
        
        candidates
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let emb = self.embed_text(text);
                (i, query_emb.cosine_similarity(&emb))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// Get vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.token_embeddings.len()
    }

    /// Get configuration
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

/// Batch embedding helper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingBatch {
    pub texts: Vec<String>,
    pub embeddings: Vec<Vec<f64>>,
}

impl EmbeddingBatch {
    pub fn new() -> Self {
        Self {
            texts: Vec::new(),
            embeddings: Vec::new(),
        }
    }

    pub fn add(&mut self, text: String, embedding: Vec<f64>) {
        self.texts.push(text);
        self.embeddings.push(embedding);
    }

    pub fn len(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }
}

impl Default for EmbeddingBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_creation() {
        let mut engine = EmbeddingEngine::new(EmbeddingConfig::default());
        let emb = engine.embed_text("hello world");
        
        assert_eq!(emb.dimension, 128);
        assert!((emb.norm() - 1.0).abs() < 0.01); // Should be normalized
    }

    #[test]
    fn test_embedding_similarity() {
        let mut engine = EmbeddingEngine::new(EmbeddingConfig::default());
        
        // Same text should have similarity of 1.0
        let sim = engine.similarity("hello", "hello");
        assert!((sim - 1.0).abs() < 0.01);
        
        // Different texts should have lower similarity
        let sim2 = engine.similarity("hello", "goodbye");
        assert!(sim2 < 1.0);
    }

    #[test]
    fn test_deterministic_embeddings() {
        let config = EmbeddingConfig::default();
        
        let mut engine1 = EmbeddingEngine::new(config.clone());
        let mut engine2 = EmbeddingEngine::new(config);
        
        let emb1 = engine1.embed_text("test");
        let emb2 = engine2.embed_text("test");
        
        // Should be identical (deterministic)
        for (a, b) in emb1.vector.iter().zip(emb2.vector.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_context_embedding() {
        let mut engine = EmbeddingEngine::new(EmbeddingConfig::default());
        
        let context = engine.embed_context(&["math is fun", "addition is basic"]);
        assert_eq!(context.dimension, 128);
    }
}
