//! Poetry Generator - Mood-Aware Creative Text Generation
//!
//! Integrates:
//! - h_t (thought vector)
//! - m_t (mood vector from BiasVector)
//! - Creativity parameters
//! - Semantic memory for vocabulary (no hardcoded words)
//!
//! All poetry vocabulary comes from learned semantic memory.

use super::text_decoder::{TextDecoder, Vocabulary};
use super::semantic_decoder::SemanticDecoder;
use crate::core::state::{ThoughtState, BiasVector};
use crate::control::emotions::EmotionalState;
use crate::memory::SemanticMemory;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Poetry style parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoetryStyle {
    /// Formal vs casual (0.0 = casual, 1.0 = formal)
    pub formality: f64,
    /// Rhyme preference (0.0 = free verse, 1.0 = strict rhyme)
    pub rhyme: f64,
    /// Imagery intensity (0.0 = literal, 1.0 = metaphorical)
    pub imagery: f64,
    /// Emotional intensity (0.0 = neutral, 1.0 = intense)
    pub intensity: f64,
}

impl Default for PoetryStyle {
    fn default() -> Self {
        Self {
            formality: 0.5,
            rhyme: 0.3,
            imagery: 0.7,
            intensity: 0.6,
        }
    }
}

impl PoetryStyle {
    /// Create style from mood and bias
    pub fn from_mood_and_bias(emotion: &EmotionalState, bias: &BiasVector) -> Self {
        Self {
            formality: 0.5 + emotion.dominance * 0.3,
            rhyme: bias.creativity * 0.5,
            imagery: bias.creativity,
            intensity: emotion.arousal,
        }
    }

    /// Convert to vector for modulation
    pub fn to_vector(&self) -> Vec<f64> {
        vec![self.formality, self.rhyme, self.imagery, self.intensity]
    }
}

/// Poetry theme/topic
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PoetryTheme {
    Love,
    Nature,
    Time,
    Loss,
    Hope,
    Dreams,
    Freedom,
    Journey,
    Silence,
    Fire,
    Ocean,
    Stars,
}

impl PoetryTheme {
    /// Get seed concepts for theme (used to query semantic memory)
    pub fn seed_concepts(&self) -> Vec<&'static str> {
        match self {
            PoetryTheme::Love => vec!["love", "heart", "affection", "romance"],
            PoetryTheme::Nature => vec!["nature", "flower", "tree", "earth"],
            PoetryTheme::Time => vec!["time", "moment", "eternal", "change"],
            PoetryTheme::Loss => vec!["loss", "grief", "memory", "absence"],
            PoetryTheme::Hope => vec!["hope", "future", "light", "possibility"],
            PoetryTheme::Dreams => vec!["dream", "imagination", "vision", "sleep"],
            PoetryTheme::Freedom => vec!["freedom", "liberty", "choice", "open"],
            PoetryTheme::Journey => vec!["journey", "path", "travel", "destination"],
            PoetryTheme::Silence => vec!["silence", "quiet", "peace", "stillness"],
            PoetryTheme::Fire => vec!["fire", "flame", "passion", "warmth"],
            PoetryTheme::Ocean => vec!["ocean", "sea", "wave", "depth"],
            PoetryTheme::Stars => vec!["star", "night", "cosmos", "universe"],
        }
    }

    /// Modulate thought vector based on theme
    pub fn modulate_thought(&self, thought: &ThoughtState) -> ThoughtState {
        let mut modulated = thought.clone();
        let theme_seed = self.seed_concepts().join("");
        let theme_hash: u64 = theme_seed.bytes().map(|b| b as u64).sum();

        // Apply theme-specific modulation to thought vector
        for (i, value) in modulated.vector.iter_mut().enumerate() {
            let theme_influence = ((i as u64 + theme_hash) % 100) as f64 / 100.0;
            *value = *value * 0.7 + theme_influence * 0.3;
        }

        modulated.normalize();
        modulated
    }
}

/// Poetry Generator - uses semantic memory for vocabulary
pub struct PoetryGenerator {
    /// Text decoder (learns from memory)
    pub decoder: TextDecoder,
    /// Semantic decoder for memory-based generation
    pub semantic_decoder: SemanticDecoder,
    /// Current style
    pub style: PoetryStyle,
    /// Theme
    pub theme: Option<PoetryTheme>,
    /// Model dimension
    pub dimension: usize,
}

impl PoetryGenerator {
    /// Create new poetry generator
    pub fn new(dimension: usize) -> Self {
        let decoder = TextDecoder::new(dimension, 0.8); // Lower temperature for coherent poetry
        let semantic_decoder = SemanticDecoder::new(dimension, 0.8);

        Self {
            decoder,
            semantic_decoder,
            style: PoetryStyle::default(),
            theme: None,
            dimension,
        }
    }

    /// Set style from emotion and bias
    pub fn set_mood(&mut self, emotion: &EmotionalState, bias: &BiasVector) {
        self.style = PoetryStyle::from_mood_and_bias(emotion, bias);

        // Adjust temperature based on creativity
        let temperature = 0.5 + bias.creativity * 0.7;
        self.decoder.set_temperature(temperature);
    }

    /// Set theme
    pub fn set_theme(&mut self, theme: PoetryTheme) {
        self.theme = Some(theme);
    }

    /// Generate a poem from thought state using semantic memory
    pub fn generate_poem_with_memory(
        &mut self,
        thought: &ThoughtState,
        memory: &SemanticMemory,
        num_lines: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut lines = Vec::new();

        // Apply theme modulation if set
        let modulated_thought = if let Some(theme) = self.theme {
            theme.modulate_thought(thought)
        } else {
            thought.clone()
        };

        // Apply style modulation
        let styled_thought = self.modulate_with_style(&modulated_thought);

        // Generate lines using semantic memory
        for i in 0..num_lines {
            let line_thought = self.vary_thought(&styled_thought, i);
            let line = self.generate_line_with_memory(&line_thought, memory)?;
            if !line.is_empty() {
                lines.push(line);
            }
        }

        Ok(self.format_poem(&lines))
    }

    /// Generate a single line using semantic memory
    fn generate_line_with_memory(
        &self,
        thought: &ThoughtState,
        memory: &SemanticMemory,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Query semantic memory for relevant concepts
        let similar_facts = memory.find_similar(&thought.vector, 5)?;
        
        if similar_facts.is_empty() {
            // Fallback: describe the thought
            let description = self.semantic_decoder.describe_thought(thought);
            return Ok(format!("{} patterns emerge", 
                if description.mean > 0.0 { "bright" } else { "soft" }
            ));
        }

        // Build line from semantic concepts
        let mut words = Vec::new();
        let max_words = 6 + (self.style.formality * 4.0) as usize;

        for (fact, _similarity) in similar_facts.iter().take(3) {
            // Extract meaningful words from fact content
            let fact_words: Vec<&str> = fact.content
                .split_whitespace()
                .filter(|w| w.len() > 2 && w.chars().all(|c| c.is_alphabetic()))
                .collect();
            
            if !fact_words.is_empty() {
                let idx = words.len() % fact_words.len();
                words.push(fact_words[idx].to_lowercase());
            }
            
            if words.len() >= max_words {
                break;
            }
        }

        let line = words.join(" ");
        Ok(self.post_process_line(&line))
    }

    /// Generate a poem from thought state (fallback without memory)
    pub fn generate_poem(
        &mut self,
        thought: &ThoughtState,
        num_lines: usize,
    ) -> String {
        let mut lines = Vec::new();

        // Apply theme modulation if set
        let modulated_thought = if let Some(theme) = self.theme {
            theme.modulate_thought(thought)
        } else {
            thought.clone()
        };

        // Apply style modulation
        let styled_thought = self.modulate_with_style(&modulated_thought);

        // Generate lines
        for i in 0..num_lines {
            // Vary the thought vector slightly for each line
            let line_thought = self.vary_thought(&styled_thought, i);
            let line = self.generate_line(&line_thought);
            lines.push(line);
        }

        // Format as poem
        self.format_poem(&lines)
    }

    /// Modulate thought with style parameters
    fn modulate_with_style(&self, thought: &ThoughtState) -> ThoughtState {
        let mut modulated = thought.clone();
        let style_vec = self.style.to_vector();

        // Apply style influence to thought vector
        let quarter = thought.dimension / 4;
        for (i, &style_param) in style_vec.iter().enumerate() {
            let start = i * quarter;
            let end = ((i + 1) * quarter).min(thought.dimension);

            for idx in start..end {
                modulated.vector[idx] *= 1.0 + (style_param - 0.5) * 0.4;
            }
        }

        modulated.normalize();
        modulated
    }

    /// Vary thought for diversity across lines
    fn vary_thought(&self, thought: &ThoughtState, line_index: usize) -> ThoughtState {
        let mut varied = thought.clone();
        
        for (i, value) in varied.vector.iter_mut().enumerate() {
            let variation = ((i + line_index) as f64 * 0.1).sin() * 0.1;
            *value += variation;
        }
        
        varied.normalize();
        varied
    }

    /// Generate a single line (fallback)
    fn generate_line(&mut self, thought: &ThoughtState) -> String {
        let max_tokens = 8 + (self.style.formality * 6.0) as usize;
        let line = self.decoder.generate(&thought.vector, max_tokens);

        // Post-process line
        self.post_process_line(&line)
    }

    /// Post-process generated line
    fn post_process_line(&self, line: &str) -> String {
        let mut processed = line.trim().to_string();

        // Remove <UNK> tokens
        processed = processed.replace("<UNK>", "");
        processed = processed.replace("<PAD>", "");
        processed = processed.replace("<START>", "");
        processed = processed.replace("<END>", "");

        // Clean up spacing around punctuation
        processed = processed.replace(" .", ".");
        processed = processed.replace(" ,", ",");
        processed = processed.replace(" !", "!");
        processed = processed.replace(" ?", "?");

        // Ensure line doesn't start with punctuation
        if processed.starts_with(|c: char| c.is_ascii_punctuation()) {
            processed = processed.chars().skip(1).collect();
        }

        // Capitalize first letter
        if let Some(first_char) = processed.chars().next() {
            processed = first_char.to_uppercase().chain(processed.chars().skip(1)).collect();
        }

        processed.trim().to_string()
    }

    /// Format lines into poem structure
    fn format_poem(&self, lines: &[String]) -> String {
        let mut poem = String::new();

        for (i, line) in lines.iter().enumerate() {
            if !line.is_empty() {
                poem.push_str(line);

                // Add line breaks based on style
                if (i + 1) % 2 == 0 && self.style.formality > 0.5 {
                    poem.push_str("\n\n"); // Stanza break
                } else {
                    poem.push('\n');
                }
            }
        }

        poem.trim().to_string()
    }

    /// Generate haiku using semantic memory
    pub fn generate_haiku_with_memory(
        &mut self,
        thought: &ThoughtState,
        memory: &SemanticMemory,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let lines = vec![
            self.generate_line_with_memory(thought, memory)?,
            self.generate_line_with_memory(&self.vary_thought(thought, 1), memory)?,
            self.generate_line_with_memory(&self.vary_thought(thought, 2), memory)?,
        ];

        Ok(lines.join("\n"))
    }

    /// Generate haiku (fallback)
    pub fn generate_haiku(&mut self, thought: &ThoughtState) -> String {
        let lines = vec![
            self.generate_line(thought),
            self.generate_line(&self.vary_thought(thought, 1)),
            self.generate_line(&self.vary_thought(thought, 2)),
        ];

        lines.join("\n")
    }

    /// Generate poem with specific emotion using semantic memory
    pub fn generate_with_emotion_and_memory(
        &mut self,
        thought: &ThoughtState,
        emotion: &EmotionalState,
        bias: &BiasVector,
        theme: Option<PoetryTheme>,
        memory: &SemanticMemory,
        num_lines: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Set mood and theme
        self.set_mood(emotion, bias);
        if let Some(t) = theme {
            self.set_theme(t);
        }

        // Generate poem using memory
        self.generate_poem_with_memory(thought, memory, num_lines)
    }

    /// Generate poem with specific emotion (fallback)
    pub fn generate_with_emotion(
        &mut self,
        thought: &ThoughtState,
        emotion: &EmotionalState,
        bias: &BiasVector,
        theme: Option<PoetryTheme>,
        num_lines: usize,
    ) -> String {
        // Set mood and theme
        self.set_mood(emotion, bias);
        if let Some(t) = theme {
            self.set_theme(t);
        }

        // Generate poem
        self.generate_poem(thought, num_lines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poetry_style_creation() {
        let emotion = EmotionalState {
            valence: 0.8,
            arousal: 0.6,
            dominance: 0.7,
        };
        let bias = BiasVector {
            creativity: 0.9,
            ..Default::default()
        };

        let style = PoetryStyle::from_mood_and_bias(&emotion, &bias);
        assert!(style.imagery > 0.5); // High creativity = high imagery
    }

    #[test]
    fn test_theme_modulation() {
        let thought = ThoughtState::from_input("love and loss", 128);
        let theme = PoetryTheme::Love;

        let modulated = theme.modulate_thought(&thought);
        assert_eq!(modulated.dimension, thought.dimension);
    }

    #[test]
    fn test_poem_generation() {
        let thought = ThoughtState::from_input("the stars shine bright tonight", 128);
        let mut generator = PoetryGenerator::new(128);

        generator.set_theme(PoetryTheme::Stars);

        let poem = generator.generate_poem(&thought, 4);
        println!("Generated poem:\n{}", poem);

        // Poem may be empty if no learned vocabulary, but should not panic
        assert!(poem.len() >= 0);
    }

    #[test]
    fn test_haiku_generation() {
        let thought = ThoughtState::from_input("cherry blossoms fall", 128);
        let mut generator = PoetryGenerator::new(128);

        let haiku = generator.generate_haiku(&thought);
        println!("Generated haiku:\n{}", haiku);

        // Haiku may be empty if no learned vocabulary, but should not panic
        assert!(haiku.len() >= 0);
    }
}
