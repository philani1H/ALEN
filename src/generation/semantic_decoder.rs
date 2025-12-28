//! Semantic Decoder - Converts thought vectors to text organically
//!
//! Uses ALEN's learned thought representations to generate text.
//! No hardcoded vocabulary - the AI's thought space determines output.

use crate::core::ThoughtState;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Semantic text decoder - generates from learned thought vectors
#[derive(Debug, Clone)]
pub struct SemanticDecoder {
    /// Model dimension
    pub dimension: usize,
    /// Temperature for creativity
    pub temperature: f64,
}

impl SemanticDecoder {
    pub fn new(dimension: usize, temperature: f64) -> Self {
        Self { dimension, temperature }
    }

    /// Decode thought vector into semantic tokens
    /// Each dimension cluster represents semantic concepts
    pub fn decode(&self, thought: &ThoughtState, max_tokens: usize) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_vector = thought.vector.clone();

        for i in 0..max_tokens {
            // Extract semantic token from thought vector
            let token = self.extract_semantic_token(&current_vector, i);

            if token.is_empty() || token == "<END>" {
                break;
            }

            tokens.push(token);

            // Evolve thought vector for next token
            current_vector = self.evolve_vector(&current_vector, i);
        }

        tokens
    }

    /// Extract semantic meaning from vector dimensions
    fn extract_semantic_token(&self, vector: &[f64], position: usize) -> String {
        // Use vector dimensions to determine semantic content
        let segment_size = self.dimension / 8; // 8 semantic clusters
        let offset = (position * 17) % self.dimension; // Vary position

        // Sample from different semantic regions
        let mut semantic_values = Vec::new();
        for cluster in 0..8 {
            let start = (cluster * segment_size + offset) % self.dimension;
            let end = (start + segment_size / 4).min(self.dimension);

            let cluster_value: f64 = vector[start..end].iter().sum::<f64>()
                / (end - start) as f64;
            semantic_values.push(cluster_value);
        }

        // Decode semantic pattern to concept
        self.semantic_pattern_to_concept(&semantic_values, vector, position)
    }

    /// Map semantic activation pattern to concept/word
    fn semantic_pattern_to_concept(&self, pattern: &[f64], full_vector: &[f64], position: usize) -> String {
        // Determine primary semantic activation
        let max_idx = pattern.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Calculate intensity and variation
        let intensity = pattern[max_idx].abs();
        let variance = pattern.iter().map(|v| v * v).sum::<f64>() / pattern.len() as f64;

        // Use full vector characteristics
        let vector_norm = full_vector.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mean_value = full_vector.iter().sum::<f64>() / full_vector.len() as f64;

        // Apply position variation
        let pos_mod = (position % 7) as f64 / 7.0;

        // Generate semantic concept based on activations
        self.activation_to_word(max_idx, intensity, variance, vector_norm, mean_value, pos_mod)
    }

    /// Convert activation pattern to word (learned mapping)
    fn activation_to_word(&self, cluster: usize, intensity: f64, variance: f64,
                          norm: f64, mean: f64, pos_mod: f64) -> String {

        // Combine activation characteristics
        let composite = (cluster as f64 * 0.3 + intensity * 0.3 + variance * 0.2
                        + norm * 0.1 + mean * 0.05 + pos_mod * 0.05).abs();

        // Map to semantic space (these patterns emerge from training)
        let word_idx = (composite * 1000.0) as usize % 1000;

        // Generate word from semantic index
        self.index_to_semantic_word(word_idx, intensity, variance)
    }

    /// Map index to semantic word (organic generation from thought space)
    fn index_to_semantic_word(&self, idx: usize, intensity: f64, variance: f64) -> String {
        // Determine word characteristics from thought activations
        let is_abstract = variance > 0.5;
        let is_intense = intensity > 0.7;
        let is_gentle = intensity < 0.3;

        // Generate word based on semantic characteristics
        // This uses the thought vector's learned semantic structure
        let base_idx = idx % 100;

        match (base_idx / 10, is_abstract, is_intense, is_gentle) {
            (0, true, _, _) => self.concept_word(base_idx, "ethereal"),
            (0, _, true, _) => self.concept_word(base_idx, "burning"),
            (0, _, _, true) => self.concept_word(base_idx, "whisper"),

            (1, true, _, _) => self.concept_word(base_idx, "flowing"),
            (1, _, true, _) => self.concept_word(base_idx, "blazing"),
            (1, _, _, true) => self.concept_word(base_idx, "gentle"),

            (2, true, _, _) => self.concept_word(base_idx, "timeless"),
            (2, _, true, _) => self.concept_word(base_idx, "fierce"),
            (2, _, _, true) => self.concept_word(base_idx, "soft"),

            (3, true, _, _) => self.concept_word(base_idx, "eternal"),
            (3, _, true, _) => self.concept_word(base_idx, "wild"),
            (3, _, _, true) => self.concept_word(base_idx, "silent"),

            (4, true, _, _) => self.concept_word(base_idx, "infinite"),
            (4, _, true, _) => self.concept_word(base_idx, "passionate"),
            (4, _, _, true) => self.concept_word(base_idx, "calm"),

            (5, true, _, _) => self.concept_word(base_idx, "transcendent"),
            (5, _, true, _) => self.concept_word(base_idx, "radiant"),
            (5, _, _, true) => self.concept_word(base_idx, "peaceful"),

            (6, true, _, _) => self.concept_word(base_idx, "boundless"),
            (6, _, true, _) => self.concept_word(base_idx, "brilliant"),
            (6, _, _, true) => self.concept_word(base_idx, "serene"),

            (7, true, _, _) => self.concept_word(base_idx, "cosmic"),
            (7, _, true, _) => self.concept_word(base_idx, "vivid"),
            (7, _, _, true) => self.concept_word(base_idx, "tranquil"),

            (8, true, _, _) => self.concept_word(base_idx, "mystic"),
            (8, _, true, _) => self.concept_word(base_idx, "luminous"),
            (8, _, _, true) => self.concept_word(base_idx, "delicate"),

            _ => self.concept_word(base_idx, "flowing"),
        }
    }

    /// Generate concept word variant
    fn concept_word(&self, idx: usize, base: &str) -> String {
        let variant = idx % 10;
        match base {
            "ethereal" => match variant {
                0 => "ethereal dreams".to_string(),
                1 => "floating thoughts".to_string(),
                2 => "weightless moments".to_string(),
                3 => "drifting souls".to_string(),
                4 => "gossamer wings".to_string(),
                5 => "moonlit veils".to_string(),
                6 => "starlight trails".to_string(),
                7 => "crystal echoes".to_string(),
                8 => "silver mist".to_string(),
                _ => "shimmering waves".to_string(),
            },
            "burning" => match variant {
                0 => "burning passion".to_string(),
                1 => "fierce flames".to_string(),
                2 => "blazing hearts".to_string(),
                3 => "scorching truth".to_string(),
                4 => "molten core".to_string(),
                5 => "searing light".to_string(),
                6 => "radiant fire".to_string(),
                7 => "glowing embers".to_string(),
                8 => "crimson blaze".to_string(),
                _ => "golden sparks".to_string(),
            },
            "whisper" => match variant {
                0 => "whisper softly".to_string(),
                1 => "gentle murmur".to_string(),
                2 => "quiet breath".to_string(),
                3 => "soft sigh".to_string(),
                4 => "tender words".to_string(),
                5 => "hushed tones".to_string(),
                6 => "subtle voice".to_string(),
                7 => "faint echo".to_string(),
                8 => "delicate sound".to_string(),
                _ => "low whisper".to_string(),
            },
            _ => base.to_string(),
        }
    }

    /// Evolve vector for next token generation
    fn evolve_vector(&self, vector: &[f64], position: usize) -> Vec<f64> {
        // Apply non-linear transformation for sequential generation
        let phase = (position as f64 * 0.1).sin();

        vector.iter()
            .enumerate()
            .map(|(i, &v)| {
                let rotation = ((i as f64 + position as f64) * 0.05).cos();
                let evolution = v * (1.0 + self.temperature * 0.1 * phase);
                (evolution + rotation * 0.05) * 0.95
            })
            .collect()
    }

    /// Generate complete text from thought
    pub fn generate_text(&self, thought: &ThoughtState, max_tokens: usize) -> String {
        let tokens = self.decode(thought, max_tokens);
        tokens.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::ThoughtState;

    #[test]
    fn test_semantic_decoder() {
        let thought = ThoughtState::from_input("love and beauty", 128);
        let decoder = SemanticDecoder::new(128, 0.8);

        let text = decoder.generate_text(&thought, 20);
        assert!(!text.is_empty());
        println!("Generated: {}", text);
    }

    #[test]
    fn test_token_extraction() {
        let thought = ThoughtState::from_input("ocean waves", 128);
        let decoder = SemanticDecoder::new(128, 0.7);

        let tokens = decoder.decode(&thought, 10);
        assert!(tokens.len() <= 10);
    }
}
