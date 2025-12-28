//! Embeddings Module
//!
//! Converts raw data (text, etc.) into the vector space ℝⁿ
//! This is the bridge between symbolic and vector representations.

use crate::core::ThoughtState;
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
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            normalize: true,
            vocab_size: 10000,
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
    /// Position encoding
    position_encodings: Vec<Vec<f64>>,
    /// Max sequence length for position encoding
    max_seq_length: usize,
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

        Self {
            config,
            token_embeddings: HashMap::new(),
            position_encodings,
            max_seq_length,
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

    /// Get or create token embedding
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

    /// Tokenize text (simple whitespace + basic cleaning)
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Embed a single token
    pub fn embed_token(&mut self, token: &str) -> Vec<f64> {
        self.get_token_embedding(token)
    }

    /// Embed text to a thought state
    pub fn embed_text(&mut self, text: &str) -> ThoughtState {
        let tokens = self.tokenize(text);
        
        if tokens.is_empty() {
            return ThoughtState::new(self.config.dimension);
        }

        // Average token embeddings with position encoding
        let mut result = vec![0.0; self.config.dimension];
        let num_tokens = tokens.len().min(self.max_seq_length);

        for (pos, token) in tokens.iter().take(num_tokens).enumerate() {
            let token_emb = self.get_token_embedding(token);
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
