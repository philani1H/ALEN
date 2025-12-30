//! Learned Text Decoder - DEPRECATED FOR GENERATION
//!
//! CRITICAL: This module is DEPRECATED for text generation.
//! Use LatentDecoder instead for all text generation.
//!
//! This module does RETRIEVAL (fact.content.clone()) which is MEMORIZATION.
//! For understanding-based generation, use: LatentDecoder
//!
//! This is kept only for backward compatibility.

use crate::core::ThoughtState;
use crate::memory::SemanticMemory;
use serde::{Deserialize, Serialize};

/// Decoder that uses learned semantic knowledge
#[derive(Debug, Clone)]
pub struct LearnedDecoder {
    /// Dimension
    pub dimension: usize,
    /// Temperature for diversity
    pub temperature: f64,
    /// Top-k concepts to consider
    pub top_k: usize,
}

impl LearnedDecoder {
    pub fn new(dimension: usize, temperature: f64) -> Self {
        Self {
            dimension,
            temperature,
            top_k: 5,
        }
    }

    /// Generate text from thought using learned semantic memory
    pub fn generate_from_memory(
        &self,
        thought: &ThoughtState,
        memory: &SemanticMemory,
        max_concepts: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Search semantic memory for similar learned concepts
        let similar_concepts = memory.find_similar(&thought.vector, max_concepts)?;

        if similar_concepts.is_empty() {
            // No learned concepts yet - return thought interpretation
            return Ok(self.interpret_raw_thought(thought));
        }

        // DEPRECATED: This does RETRIEVAL which is MEMORIZATION
        // Use LatentDecoder.generate() instead for understanding-based generation
        Ok("[DEPRECATED: Use LatentDecoder for generation]".to_string())
    }

    /// Fallback: Interpret thought vector directly when no memory available
    pub fn interpret_raw_thought(&self, thought: &ThoughtState) -> String {
        // Use thought vector characteristics to generate semantic description
        let norm = thought.norm();
        let mean: f64 = thought.vector.iter().sum::<f64>() / thought.vector.len() as f64;
        let variance: f64 = thought.vector.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>() / thought.vector.len() as f64;

        let std_dev = variance.sqrt();

        // Describe the thought's semantic characteristics
        let intensity = if norm > 1.5 {
            "intense"
        } else if norm > 1.0 {
            "strong"
        } else if norm > 0.7 {
            "moderate"
        } else {
            "subtle"
        };

        let distribution = if std_dev > 0.3 {
            "complex"
        } else if std_dev > 0.15 {
            "varied"
        } else {
            "focused"
        };

        let orientation = if mean > 0.1 {
            "positive"
        } else if mean < -0.1 {
            "negative"
        } else {
            "neutral"
        };

        format!("{} {} {} thought patterns", intensity, distribution, orientation)
    }

    /// Generate poem using learned concepts
    pub fn generate_poem_from_memory(
        &self,
        thought: &ThoughtState,
        memory: &SemanticMemory,
        theme: &str,
        lines: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut poem_lines = Vec::new();

        // Vary the thought vector slightly for each line
        for i in 0..lines {
            let varied_thought = self.vary_thought_for_line(thought, i);

            // Search for concepts related to this variation
            let concepts = memory.find_similar(&varied_thought.vector, 3)?;

            if !concepts.is_empty() {
                // Combine concepts into a poetic line
                let line_concepts: Vec<String> = concepts.iter()
                    .map(|(fact, _)| fact.content.clone())
                    .collect();

                poem_lines.push(line_concepts.join(" "));
            } else {
                // Use theme if no learned concepts
                poem_lines.push(format!("{} patterns", theme));
            }
        }

        Ok(poem_lines.join("\n"))
    }

    /// Vary thought vector for diversity across lines
    fn vary_thought_for_line(&self, thought: &ThoughtState, line_index: usize) -> ThoughtState {
        let mut varied = thought.clone();

        // Apply rotation based on line index
        let angle = (line_index as f64 * 0.3).sin() * self.temperature;

        for (i, value) in varied.vector.iter_mut().enumerate() {
            let phase = ((i + line_index) as f64 * 0.1).cos();
            *value = *value * (1.0 + angle * 0.1) + phase * 0.05;
        }

        varied.normalize();
        varied
    }

    /// Generate with metadata about learned concepts used
    pub fn generate_with_provenance(
        &self,
        thought: &ThoughtState,
        memory: &SemanticMemory,
        max_concepts: usize,
    ) -> Result<GenerationResult, Box<dyn std::error::Error>> {
        let similar_concepts = memory.find_similar(&thought.vector, max_concepts)?;

        let mut text_parts = Vec::new();
        let mut concept_sources = Vec::new();

        for (fact, similarity) in similar_concepts.iter().take(max_concepts) {
            text_parts.push(fact.content.clone());
            concept_sources.push(ConceptSource {
                concept: fact.content.clone(),
                similarity: *similarity,
                confidence: fact.confidence,
            });
        }

        Ok(GenerationResult {
            text: text_parts.join(" "),
            sources: concept_sources,
            thought_norm: thought.norm(),
            learned_concepts_used: !similar_concepts.is_empty(),
        })
    }
}

/// Result of generation with provenance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// Generated text
    pub text: String,
    /// Learned concepts that contributed
    pub sources: Vec<ConceptSource>,
    /// Thought vector norm
    pub thought_norm: f64,
    /// Whether learned concepts were used
    pub learned_concepts_used: bool,
}

/// Source concept used in generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptSource {
    /// Concept content
    pub concept: String,
    /// Similarity to query thought
    pub similarity: f64,
    /// Original confidence when learned
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_thought_interpretation() {
        let thought = ThoughtState::from_input("test thought", 128);
        let decoder = LearnedDecoder::new(128, 0.8);

        let interpretation = decoder.interpret_raw_thought(&thought);
        assert!(!interpretation.is_empty());
        println!("Interpretation: {}", interpretation);
    }

    #[test]
    fn test_thought_variation() {
        let thought = ThoughtState::from_input("base thought", 128);
        let decoder = LearnedDecoder::new(128, 0.7);

        let varied1 = decoder.vary_thought_for_line(&thought, 0);
        let varied2 = decoder.vary_thought_for_line(&thought, 1);

        // Variations should be different
        let similarity = varied1.cosine_similarity(&varied2);
        assert!(similarity < 1.0);
        assert!(similarity > 0.5); // But still related
    }
}
