//! ALEN Generation Module
//!
//! Generates outputs from thought states:
//! - Text Generation: Vocabulary-based decoding
//! - Poetry Generation: Mood-aware creative text (p_t = softmax(W_out·h_t + b))
//! - Image Generation: Simple diffusion-like process
//! - Video Generation: Temporal sequence generation
//! - Content synthesis and controlled generation

pub mod video;
pub mod text_decoder;
pub mod poetry;
pub mod learned_decoder;
pub mod factual_decoder;
pub mod explanation_decoder;
pub mod reasoning_engine;
pub mod knowledge_visual;

pub use video::{VideoGenerator, VideoGenConfig, GeneratedVideo, MotionType};
pub use text_decoder::{TextDecoder, Vocabulary as DecoderVocabulary};
pub use poetry::{PoetryGenerator, PoetryStyle, PoetryTheme};
pub use learned_decoder::{LearnedDecoder, GenerationResult, ConceptSource};
pub use factual_decoder::{
    FactualDecoder, FactualResponse, FactualThresholds, TokenVerification,
    VerificationResult, GenerationMode,
};
pub use explanation_decoder::{
    ExplanationDecoder, ExplanationAudience, ExplanationResponse, StyleVector,
};
pub use reasoning_engine::{
    ReasoningEngine, LatentResult, LatentVerification,
};
pub use knowledge_visual::{
    KnowledgeImageGenerator, KnowledgeImage, KnowledgeVideo, KnowledgeVisualConfig,
};

use crate::core::{ThoughtState, Activation, DenseLayer};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Temperature for sampling (higher = more random)
    pub temperature: f64,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f64,
    /// Maximum generation length
    pub max_length: usize,
    /// Repetition penalty
    pub repetition_penalty: f64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            max_length: 256,
            repetition_penalty: 1.1,
        }
    }
}

/// Generated content wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedContent {
    pub text: Option<String>,
    pub image_data: Option<Vec<f64>>,
    pub confidence: f64,
    pub tokens_generated: usize,
}

// ============================================================================
// TEXT GENERATION
// ============================================================================

/// Simple vocabulary for text generation
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub words: Vec<String>,
    pub word_to_idx: HashMap<String, usize>,
    pub embeddings: DMatrix<f64>,
    pub dim: usize,
}

impl Vocabulary {
    /// Create a basic vocabulary
    pub fn new(dim: usize) -> Self {
        // Basic vocabulary for demonstration
        let words: Vec<String> = vec![
            // Common words
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "can", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "just", "also", "now", "and", "but", "or", "if", "because", "until",
            // Technical/AI words
            "system", "data", "learning", "model", "algorithm", "neural", "network", "training",
            "input", "output", "function", "process", "analysis", "result", "value", "parameter",
            "optimization", "gradient", "weight", "bias", "layer", "activation", "loss", "error",
            "accuracy", "precision", "recall", "metric", "evaluation", "validation", "test", "train",
            "feature", "embedding", "vector", "matrix", "tensor", "dimension", "space", "representation",
            "attention", "transformer", "encoder", "decoder", "sequence", "token", "vocabulary", "context",
            // General concepts
            "information", "knowledge", "understanding", "reasoning", "thinking", "intelligence", "artificial",
            "machine", "computer", "software", "hardware", "memory", "storage", "processing", "computation",
            "solution", "problem", "answer", "question", "query", "response", "request", "action",
            "state", "transition", "change", "update", "modify", "create", "delete", "read", "write",
            "time", "space", "energy", "matter", "force", "motion", "physics", "mathematics", "science",
            "technology", "engineering", "design", "architecture", "structure", "pattern", "rule", "logic",
            // Punctuation and special
            ".", ",", "!", "?", ":", ";", "-", "'", "\"", "(", ")", "[", "]",
            "<START>", "<END>", "<PAD>", "<UNK>",
        ].iter().map(|s| s.to_string()).collect();

        let word_to_idx: HashMap<String, usize> = words.iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();

        // Initialize embeddings randomly
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0 / (dim as f64).sqrt()).unwrap();
        let embeddings = DMatrix::from_fn(dim, words.len(), |_, _| {
            normal.sample(&mut rng)
        });

        Self { words, word_to_idx, embeddings, dim }
    }

    /// Get embedding for a word
    pub fn get_embedding(&self, word: &str) -> DVector<f64> {
        let idx = self.word_to_idx.get(word).copied().unwrap_or(
            *self.word_to_idx.get("<UNK>").unwrap_or(&0)
        );
        self.embeddings.column(idx).into()
    }

    /// Find closest word to an embedding
    pub fn nearest_word(&self, embedding: &DVector<f64>) -> String {
        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;

        for (i, col) in self.embeddings.column_iter().enumerate() {
            let col_vec: DVector<f64> = col.into();
            let sim = embedding.dot(&col_vec) / (embedding.norm() * col_vec.norm() + 1e-10);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        self.words.get(best_idx).cloned().unwrap_or_else(|| "<UNK>".to_string())
    }

    /// Find top-k nearest words
    pub fn top_k_words(&self, embedding: &DVector<f64>, k: usize) -> Vec<(String, f64)> {
        let mut scores: Vec<(usize, f64)> = self.embeddings
            .column_iter()
            .enumerate()
            .map(|(i, col)| {
                let col_vec: DVector<f64> = col.into();
                let sim = embedding.dot(&col_vec) / (embedding.norm() * col_vec.norm() + 1e-10);
                (i, sim)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores.into_iter()
            .take(k)
            .map(|(i, s)| (self.words[i].clone(), s))
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.words.len()
    }
}

/// Text generator using thought states
#[derive(Debug, Clone)]
pub struct TextGenerator {
    pub config: GenerationConfig,
    pub vocab: Vocabulary,
    pub projection: DenseLayer,
    pub dim: usize,
}

impl TextGenerator {
    pub fn new(config: GenerationConfig, dim: usize) -> Self {
        let vocab = Vocabulary::new(dim);
        let projection = DenseLayer::new(dim, vocab.vocab_size(), Activation::Softmax);

        Self { config, vocab, projection, dim }
    }

    /// Generate text from a thought state
    pub fn generate(&self, thought: &ThoughtState, max_tokens: usize) -> String {
        let mut tokens = Vec::new();
        let mut current = DVector::from_column_slice(&thought.vector);
        let max_len = max_tokens.min(self.config.max_length);

        for _ in 0..max_len {
            // Project to vocabulary
            let logits = self.projection.forward(&current);
            
            // Apply temperature
            let scaled: Vec<f64> = logits.iter()
                .map(|x| x / self.config.temperature)
                .collect();
            
            // Apply softmax
            let probs = Activation::Softmax.apply_vector(&scaled);
            
            // Sample from distribution
            let token_idx = self.sample_token(&probs);
            let token = self.vocab.words.get(token_idx)
                .cloned()
                .unwrap_or_else(|| "<UNK>".to_string());

            // Stop conditions
            if token == "<END>" || token == "<PAD>" {
                break;
            }

            if token != "<START>" && token != "<UNK>" {
                tokens.push(token.clone());
            }

            // Update state for next token
            let token_emb = self.vocab.get_embedding(&token);
            current = (&current + &token_emb) / 2.0;
            let norm = current.norm();
            if norm > 1e-10 {
                current /= norm;
            }
        }

        self.join_tokens(&tokens)
    }

    /// Sample a token index from probability distribution
    fn sample_token(&self, probs: &[f64]) -> usize {
        let mut rng = rand::thread_rng();

        if self.config.top_k > 0 {
            // Top-k sampling
            let mut indexed: Vec<(usize, f64)> = probs.iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            let top_k = indexed.into_iter().take(self.config.top_k).collect::<Vec<_>>();
            let sum: f64 = top_k.iter().map(|(_, p)| p).sum();
            
            let mut r = rng.gen::<f64>() * sum;
            for (idx, p) in top_k {
                r -= p;
                if r <= 0.0 {
                    return idx;
                }
            }
            return 0;
        }

        // Regular sampling
        let mut r = rng.gen::<f64>();
        for (i, &p) in probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        0
    }

    /// Join tokens into readable text
    fn join_tokens(&self, tokens: &[String]) -> String {
        let mut result = String::new();
        let punctuation = vec![".", ",", "!", "?", ":", ";", "'", "\"", ")", "]"];
        let no_space_before = vec![".", ",", "!", "?", ":", ";", "'", ")", "]"];
        let no_space_after = vec!["(", "[", "'", "\""];

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

    /// Generate with constraints
    pub fn generate_constrained(&self, thought: &ThoughtState, max_tokens: usize, must_include: &[&str]) -> String {
        let mut base = self.generate(thought, max_tokens);
        
        // Simple constraint enforcement - append missing required words
        for word in must_include {
            if !base.to_lowercase().contains(&word.to_lowercase()) {
                base.push_str(&format!(" {}", word));
            }
        }
        
        base
    }
}

// ============================================================================
// IMAGE GENERATION
// ============================================================================

/// Simple image generator using thought states
#[derive(Debug, Clone)]
pub struct ImageGenerator {
    pub config: GenerationConfig,
    pub dim: usize,
    pub image_size: usize,
    pub channels: usize,
    /// Upsampling layers
    pub upsample_layers: Vec<DenseLayer>,
}

impl ImageGenerator {
    pub fn new(dim: usize, image_size: usize) -> Self {
        // Create upsampling network
        let channels = 3;
        let final_size = image_size * image_size * channels;
        
        let mut upsample_layers = Vec::new();
        let mut current_dim = dim;
        
        // Progressive upsampling
        let intermediate_sizes = vec![dim * 2, dim * 4, final_size / 4, final_size];
        
        for target_dim in intermediate_sizes {
            upsample_layers.push(DenseLayer::new(current_dim, target_dim, Activation::ReLU));
            current_dim = target_dim;
        }

        Self {
            config: GenerationConfig::default(),
            dim,
            image_size,
            channels,
            upsample_layers,
        }
    }

    /// Generate image data from thought state
    pub fn generate(&self, thought: &ThoughtState) -> Vec<f64> {
        let mut current = DVector::from_column_slice(&thought.vector);

        // Pass through upsampling layers
        for layer in &self.upsample_layers {
            current = layer.forward(&current);
        }

        // Normalize to [0, 1]
        let min_val = current.min();
        let max_val = current.max();
        let range = (max_val - min_val).max(1e-10);

        current.iter()
            .map(|x| ((x - min_val) / range).clamp(0.0, 1.0))
            .collect()
    }

    /// Generate image with noise for variation
    pub fn generate_with_noise(&self, thought: &ThoughtState, noise_level: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, noise_level).unwrap();

        let mut noisy_thought = thought.clone();
        for v in &mut noisy_thought.vector {
            *v += normal.sample(&mut rng);
        }

        // Re-normalize
        let norm: f64 = noisy_thought.vector.iter().map(|x| x*x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for v in &mut noisy_thought.vector {
                *v /= norm;
            }
        }

        self.generate(&noisy_thought)
    }

    /// Simple diffusion-like denoising
    pub fn denoise_step(&self, image: &[f64], noise_level: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, noise_level * 0.1).unwrap();

        // Simple denoising by averaging with small noise
        image.iter()
            .map(|&x| {
                let noise = normal.sample(&mut rng);
                (x + noise).clamp(0.0, 1.0)
            })
            .collect()
    }
}

// ============================================================================
// UNIFIED CONTENT GENERATOR
// ============================================================================

/// Unified content generator that can produce text and images
#[derive(Debug, Clone)]
pub struct ContentGenerator {
    pub text_gen: TextGenerator,
    pub image_gen: ImageGenerator,
    pub dim: usize,
}

impl ContentGenerator {
    pub fn new(dim: usize) -> Self {
        Self {
            text_gen: TextGenerator::new(GenerationConfig::default(), dim),
            image_gen: ImageGenerator::new(dim, 64),
            dim,
        }
    }

    /// Generate text from thought
    pub fn generate_text(&self, thought: &ThoughtState, max_tokens: usize) -> GeneratedContent {
        let text = self.text_gen.generate(thought, max_tokens);
        GeneratedContent {
            text: Some(text.clone()),
            image_data: None,
            confidence: thought.confidence,
            tokens_generated: text.split_whitespace().count(),
        }
    }

    /// Generate image from thought
    pub fn generate_image(&self, thought: &ThoughtState) -> GeneratedContent {
        let image = self.image_gen.generate(thought);
        GeneratedContent {
            text: None,
            image_data: Some(image),
            confidence: thought.confidence,
            tokens_generated: 0,
        }
    }

    /// Generate both text and image
    pub fn generate_multimodal(&self, thought: &ThoughtState, max_tokens: usize) -> GeneratedContent {
        let text = self.text_gen.generate(thought, max_tokens);
        let image = self.image_gen.generate(thought);
        GeneratedContent {
            text: Some(text.clone()),
            image_data: Some(image),
            confidence: thought.confidence,
            tokens_generated: text.split_whitespace().count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary() {
        let vocab = Vocabulary::new(64);
        assert!(vocab.vocab_size() > 100);
        
        let emb = vocab.get_embedding("the");
        assert_eq!(emb.len(), 64);
    }

    #[test]
    fn test_text_generation() {
        let config = GenerationConfig::default();
        let gen = TextGenerator::new(config, 64);
        let thought = ThoughtState::random(64);
        let text = gen.generate(&thought, 20);
        assert!(!text.is_empty());
    }

    #[test]
    fn test_image_generation() {
        let gen = ImageGenerator::new(64, 32);
        let thought = ThoughtState::random(64);
        let image = gen.generate(&thought);
        assert_eq!(image.len(), 32 * 32 * 3);
        
        // Check all values in [0, 1]
        assert!(image.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_content_generator() {
        let gen = ContentGenerator::new(64);
        let thought = ThoughtState::random(64);
        
        let text_content = gen.generate_text(&thought, 10);
        assert!(text_content.text.is_some());
        
        let image_content = gen.generate_image(&thought);
        assert!(image_content.image_data.is_some());
        
        let multi = gen.generate_multimodal(&thought, 10);
        assert!(multi.text.is_some());
        assert!(multi.image_data.is_some());
    }

    #[test]
    fn test_top_k_words() {
        let vocab = Vocabulary::new(64);
        let emb = vocab.get_embedding("system");
        let top_k = vocab.top_k_words(&emb, 5);
        assert_eq!(top_k.len(), 5);
        // First result should be "system" itself (highest similarity)
        assert_eq!(top_k[0].0, "system");
    }

    #[test]
    fn test_generation_config() {
        let config = GenerationConfig {
            temperature: 0.5,
            top_k: 10,
            top_p: 0.8,
            max_length: 100,
            repetition_penalty: 1.2,
        };
        let gen = TextGenerator::new(config.clone(), 64);
        assert!((gen.config.temperature - 0.5).abs() < 1e-10);
    }
}
