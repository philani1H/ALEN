//! Semantic Decoder - Converts thought vectors to text using learned representations
//!
//! Uses ALEN's learned thought representations to generate text.
//! All output is derived from learned semantic memory - no hardcoded vocabulary.
//! The decoder queries semantic memory to find concepts that match thought activations.

use crate::core::ThoughtState;
use crate::memory::{SemanticMemory, SemanticFact};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Semantic text decoder - generates from learned thought vectors
#[derive(Debug, Clone)]
pub struct SemanticDecoder {
    /// Model dimension
    pub dimension: usize,
    /// Temperature for creativity (affects diversity of output)
    pub temperature: f64,
    /// Number of semantic clusters to analyze
    pub num_clusters: usize,
    /// Minimum similarity threshold for concept retrieval
    pub min_similarity: f64,
}

impl SemanticDecoder {
    pub fn new(dimension: usize, temperature: f64) -> Self {
        Self {
            dimension,
            temperature,
            num_clusters: 8,
            min_similarity: 0.1,
        }
    }

    /// Decode thought vector into text using semantic memory
    pub fn decode_with_memory(
        &self,
        thought: &ThoughtState,
        memory: &SemanticMemory,
        max_tokens: usize,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut tokens = Vec::new();
        let mut current_vector = thought.vector.clone();
        let mut used_concepts: HashMap<String, usize> = HashMap::new();

        for i in 0..max_tokens {
            // Query semantic memory for matching concepts
            let candidates = memory.find_similar(&current_vector, 10)?;

            if candidates.is_empty() {
                break;
            }

            // Select concept based on similarity and diversity
            let selected = self.select_concept(&candidates, &used_concepts, i);

            if let Some((fact, _similarity)) = selected {
                // Extract meaningful token from concept
                let token = self.extract_token_from_concept(&fact, i);

                if token.is_empty() || token == "<END>" {
                    break;
                }

                tokens.push(token.clone());
                *used_concepts.entry(fact.id.clone()).or_insert(0) += 1;

                // Evolve thought vector for next token
                current_vector = self.evolve_vector(&current_vector, &fact.embedding, i);
            } else {
                break;
            }
        }

        Ok(tokens)
    }

    /// Select concept from candidates with diversity consideration
    fn select_concept<'a>(
        &self,
        candidates: &'a [(SemanticFact, f64)],
        used: &HashMap<String, usize>,
        position: usize,
    ) -> Option<&'a (SemanticFact, f64)> {
        // Score candidates based on similarity and novelty
        let mut scored: Vec<(&(SemanticFact, f64), f64)> = candidates.iter()
            .map(|c| {
                let usage_penalty = used.get(&c.0.id).copied().unwrap_or(0) as f64 * 0.3;
                let position_bonus = (position as f64 * 0.05).sin().abs() * self.temperature;
                let score = c.1 - usage_penalty + position_bonus;
                (c, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.first().map(|(c, _)| *c)
    }

    /// Extract meaningful token from semantic concept
    fn extract_token_from_concept(&self, fact: &SemanticFact, position: usize) -> String {
        let words: Vec<&str> = fact.content.split_whitespace().collect();

        if words.is_empty() {
            return String::new();
        }

        // Use position to select different parts of the concept
        let idx = position % words.len();
        words[idx].to_string()
    }

    /// Evolve vector for next token generation using learned concept
    fn evolve_vector(&self, current: &[f64], concept_embedding: &[f64], position: usize) -> Vec<f64> {
        let decay = 0.9 - (position as f64 * 0.01).min(0.3);
        let blend = self.temperature * 0.2;

        current.iter()
            .zip(concept_embedding.iter())
            .enumerate()
            .map(|(i, (&c, &e))| {
                let phase = ((i + position) as f64 * 0.1).cos() * 0.05;
                c * decay + e * blend + phase
            })
            .collect()
    }

    /// Generate complete text from thought using memory
    pub fn generate_text_with_memory(
        &self,
        thought: &ThoughtState,
        memory: &SemanticMemory,
        max_tokens: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let tokens = self.decode_with_memory(thought, memory, max_tokens)?;
        Ok(self.join_tokens(&tokens))
    }

    /// Join tokens into readable text
    fn join_tokens(&self, tokens: &[String]) -> String {
        if tokens.is_empty() {
            return String::new();
        }

        let mut result = String::new();
        let no_space_before = [".", ",", "!", "?", ":", ";", "'", ")", "]"];
        let no_space_after = ["(", "[", "'"];

        for (i, token) in tokens.iter().enumerate() {
            let needs_space = if i == 0 {
                false
            } else if no_space_before.contains(&token.as_str()) {
                false
            } else if i > 0 && no_space_after.contains(&tokens[i - 1].as_str()) {
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

    /// Fallback: Generate from thought vector when no memory available
    /// Uses vector characteristics to describe the semantic space
    pub fn describe_thought(&self, thought: &ThoughtState) -> ThoughtDescription {
        let norm = thought.norm();
        let mean: f64 = thought.vector.iter().sum::<f64>() / thought.vector.len() as f64;
        let variance: f64 = thought.vector.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>() / thought.vector.len() as f64;
        let std_dev = variance.sqrt();

        // Analyze activation patterns in different regions
        let quarter = self.dimension / 4;
        let region_activations: Vec<f64> = (0..4)
            .map(|r| {
                let start = r * quarter;
                let end = ((r + 1) * quarter).min(thought.vector.len());
                thought.vector[start..end].iter().sum::<f64>() / (end - start) as f64
            })
            .collect();

        // Find dominant region
        let dominant_region = region_activations.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        ThoughtDescription {
            norm,
            mean,
            std_dev,
            dominant_region,
            region_activations,
            confidence: thought.confidence,
        }
    }

    /// Generate semantic clusters from thought vector
    pub fn extract_semantic_clusters(&self, thought: &ThoughtState) -> Vec<SemanticCluster> {
        let cluster_size = self.dimension / self.num_clusters;
        let mut clusters = Vec::new();

        for i in 0..self.num_clusters {
            let start = i * cluster_size;
            let end = ((i + 1) * cluster_size).min(thought.vector.len());

            let values: Vec<f64> = thought.vector[start..end].to_vec();
            let activation: f64 = values.iter().sum::<f64>() / values.len() as f64;
            let variance: f64 = values.iter()
                .map(|&v| (v - activation).powi(2))
                .sum::<f64>() / values.len() as f64;

            clusters.push(SemanticCluster {
                index: i,
                activation,
                variance,
                values,
            });
        }

        clusters
    }
}

/// Description of thought vector characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtDescription {
    pub norm: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub dominant_region: usize,
    pub region_activations: Vec<f64>,
    pub confidence: f64,
}

/// Semantic cluster extracted from thought vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCluster {
    pub index: usize,
    pub activation: f64,
    pub variance: f64,
    pub values: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_decoder_creation() {
        let decoder = SemanticDecoder::new(128, 0.8);
        assert_eq!(decoder.dimension, 128);
        assert_eq!(decoder.num_clusters, 8);
    }

    #[test]
    fn test_thought_description() {
        let thought = ThoughtState::from_input("test thought", 128);
        let decoder = SemanticDecoder::new(128, 0.7);

        let description = decoder.describe_thought(&thought);
        assert!(description.norm > 0.0);
        assert_eq!(description.region_activations.len(), 4);
    }

    #[test]
    fn test_semantic_clusters() {
        let thought = ThoughtState::from_input("semantic analysis", 128);
        let decoder = SemanticDecoder::new(128, 0.7);

        let clusters = decoder.extract_semantic_clusters(&thought);
        assert_eq!(clusters.len(), 8);
    }

    #[test]
    fn test_token_joining() {
        let decoder = SemanticDecoder::new(128, 0.7);
        let tokens = vec!["hello".to_string(), "world".to_string(), ".".to_string()];
        let text = decoder.join_tokens(&tokens);

        assert_eq!(text, "Hello world.");
    }
}
