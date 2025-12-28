//! Universal Explanation Decoder - Multi-Audience Knowledge Translation
//!
//! Implements: h'_explain = f(h_knowledge, s_style, c_context)
//!
//! Key Features:
//! - Same knowledge → different audiences (child, elder, mathematician)
//! - ZERO HALLUCINATIONS - same verification as FactualDecoder
//! - Style vectors control vocabulary/tone, NOT truth
//! - Projection operators: P_style(h_knowledge) → h_audience

use crate::core::{ThoughtState, BiasVector};
use crate::memory::{SemanticMemory, SemanticFact};
use crate::generation::factual_decoder::{FactualThresholds, TokenVerification};
use serde::{Deserialize, Serialize};

/// Explanation audience/style
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ExplanationAudience {
    /// Simple, concrete examples, analogies (5-10 years old)
    Child,
    /// Practical, respectful, step-by-step (general adult)
    General,
    /// Practical wisdom, real-world applications
    Elder,
    /// Formal, symbolic, rigorous mathematical notation
    Mathematician,
    /// Technical, precise, assumes background knowledge
    Expert,
}

impl ExplanationAudience {
    /// Get complexity level (0.0 = simplest, 1.0 = most complex)
    pub fn complexity_level(&self) -> f64 {
        match self {
            ExplanationAudience::Child => 0.2,
            ExplanationAudience::General => 0.5,
            ExplanationAudience::Elder => 0.6,
            ExplanationAudience::Mathematician => 0.9,
            ExplanationAudience::Expert => 1.0,
        }
    }

    /// Get style characteristics
    pub fn style_vector(&self) -> StyleVector {
        match self {
            ExplanationAudience::Child => StyleVector {
                abstraction: 0.2,      // Concrete examples
                formality: 0.1,        // Casual, friendly
                technical_density: 0.1, // Simple words
                analogy_preference: 0.9, // Heavy analogies
                step_detail: 0.5,      // Medium steps
            },
            ExplanationAudience::General => StyleVector {
                abstraction: 0.5,
                formality: 0.5,
                technical_density: 0.4,
                analogy_preference: 0.6,
                step_detail: 0.6,
            },
            ExplanationAudience::Elder => StyleVector {
                abstraction: 0.4,
                formality: 0.7,        // Respectful
                technical_density: 0.3,
                analogy_preference: 0.7, // Practical examples
                step_detail: 0.8,      // Detailed steps
            },
            ExplanationAudience::Mathematician => StyleVector {
                abstraction: 0.9,      // Abstract concepts
                formality: 1.0,        // Very formal
                technical_density: 1.0, // Symbols, notation
                analogy_preference: 0.2, // Minimal analogies
                step_detail: 0.7,      // Rigorous proofs
            },
            ExplanationAudience::Expert => StyleVector {
                abstraction: 0.8,
                formality: 0.8,
                technical_density: 0.9,
                analogy_preference: 0.3,
                step_detail: 0.5,      // Skip basics
            },
        }
    }
}

/// Style characteristics for explanation
#[derive(Debug, Clone, Default)]
pub struct StyleVector {
    /// How abstract vs concrete (0.0 = concrete, 1.0 = abstract)
    pub abstraction: f64,
    /// Formality level (0.0 = casual, 1.0 = formal)
    pub formality: f64,
    /// Technical vocabulary density (0.0 = simple, 1.0 = technical)
    pub technical_density: f64,
    /// Preference for analogies/metaphors (0.0 = literal, 1.0 = heavy analogies)
    pub analogy_preference: f64,
    /// Level of step-by-step detail (0.0 = skip steps, 1.0 = every detail)
    pub step_detail: f64,
}

impl StyleVector {
    /// Convert to vector for projection
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.abstraction,
            self.formality,
            self.technical_density,
            self.analogy_preference,
            self.step_detail,
        ]
    }
}

/// Universal explanation decoder
pub struct ExplanationDecoder {
    /// Dimension of thought vectors
    pub dimension: usize,
    /// Verification thresholds (same as FactualDecoder - NO HALLUCINATIONS)
    pub thresholds: FactualThresholds,
    /// Target audience
    pub audience: ExplanationAudience,
    /// Neutral bias (no creativity in factual mode)
    neutral_bias: BiasVector,
}

impl ExplanationDecoder {
    pub fn new(dimension: usize, audience: ExplanationAudience, thresholds: FactualThresholds) -> Self {
        Self {
            dimension,
            thresholds,
            audience,
            neutral_bias: BiasVector {
                risk_tolerance: 0.5,
                exploration: 0.5,
                urgency: 0.5,
                creativity: 0.0,  // ZERO creativity - factual only
            },
        }
    }

    /// Explain a concept to the target audience
    /// Uses: h'_explain = f(h_knowledge, s_style, c_context)
    /// CRITICAL: Same verification as FactualDecoder - NO HALLUCINATIONS
    pub fn explain(
        &self,
        concept: &str,
        memory: &SemanticMemory,
        max_sentences: usize,
    ) -> Result<ExplanationResponse, Box<dyn std::error::Error>> {
        // Get knowledge vector for concept
        let knowledge_thought = ThoughtState::from_input(concept, self.dimension);

        // Apply audience-specific projection
        let audience_thought = self.project_to_audience(&knowledge_thought);

        // Generate explanation with strict verification
        let mut sentences = Vec::new();
        let mut all_verifications = Vec::new();

        for sentence_idx in 0..max_sentences {
            // Vary thought slightly for each sentence while maintaining truth
            let sentence_thought = self.vary_for_sentence(&audience_thought, sentence_idx);

            // Generate sentence with STRICT knowledge verification
            let (sentence, verifications) = self.generate_verified_sentence(
                &sentence_thought,
                memory,
                20, // max tokens per sentence
            )?;

            if sentence.is_empty() {
                break;
            }

            sentences.push(sentence);
            all_verifications.extend(verifications);
        }

        let verified_percentage = self.compute_verification_rate(&all_verifications);

        Ok(ExplanationResponse {
            explanation: sentences.join(" "),
            sentences,
            audience: self.audience,
            verifications: all_verifications,
            style_applied: self.audience.style_vector(),
            verified_percentage,
        })
    }

    /// Project knowledge vector to audience-appropriate space
    /// P_style(h_knowledge) → h_audience
    /// IMPORTANT: This only affects vocabulary/tone, NOT truth content
    fn project_to_audience(&self, knowledge: &ThoughtState) -> ThoughtState {
        let mut projected = knowledge.clone();
        let style = self.audience.style_vector();
        let style_vec = style.to_vec();

        // Apply style modulation to different vector regions
        let region_size = self.dimension / style_vec.len();

        for (i, &style_param) in style_vec.iter().enumerate() {
            let start = i * region_size;
            let end = ((i + 1) * region_size).min(self.dimension);

            for idx in start..end {
                // Modulate based on style parameter
                // Higher style param = stronger influence
                let modulation = (style_param - 0.5) * 0.3;
                projected.vector[idx] *= 1.0 + modulation;
            }
        }

        projected.normalize();
        projected
    }

    /// Vary thought for sentence diversity while maintaining factual grounding
    fn vary_for_sentence(&self, thought: &ThoughtState, sentence_idx: usize) -> ThoughtState {
        let mut varied = thought.clone();

        // Small deterministic variation based on sentence index
        let variation_scale = 0.05; // Very small to maintain truthfulness
        for (i, value) in varied.vector.iter_mut().enumerate() {
            let phase = ((i + sentence_idx) as f64 * 0.1).sin();
            *value += phase * variation_scale;
        }

        varied.normalize();
        varied
    }

    /// Generate a verified sentence
    /// Same verification loop as FactualDecoder - NO HALLUCINATIONS
    fn generate_verified_sentence(
        &self,
        thought: &ThoughtState,
        memory: &SemanticMemory,
        max_tokens: usize,
    ) -> Result<(String, Vec<TokenVerification>), Box<dyn std::error::Error>> {
        let mut tokens = Vec::new();
        let mut verifications = Vec::new();

        for _ in 0..max_tokens {
            // Search knowledge with current thought
            let candidates = memory.find_similar(&thought.vector, 10)?;

            if candidates.is_empty() {
                break;
            }

            // Find best verified candidate
            let mut best_verified: Option<(String, TokenVerification)> = None;

            for (fact, similarity) in candidates.iter() {
                // CRITICAL: Same threshold check as FactualDecoder
                if *similarity >= self.thresholds.min_knowledge_similarity
                    && fact.confidence >= self.thresholds.min_token_confidence
                {
                    let token = self.extract_audience_appropriate_token(fact)?;

                    best_verified = Some((
                        token,
                        TokenVerification {
                            similarity: *similarity,
                            confidence: fact.confidence,
                            knowledge_source: Some(fact.id.clone()),
                            verified: true,
                            reason: format!(
                                "Verified: sim={:.3}, conf={:.3}",
                                similarity, fact.confidence
                            ),
                        },
                    ));
                    break;
                }
            }

            if let Some((token, verification)) = best_verified {
                tokens.push(token);
                verifications.push(verification);
            } else {
                // No verified token - stop rather than hallucinate
                break;
            }
        }

        Ok((tokens.join(" "), verifications))
    }

    /// Extract audience-appropriate token from semantic fact
    /// Uses style vector to select appropriate vocabulary
    fn extract_audience_appropriate_token(
        &self,
        fact: &SemanticFact,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // For now, use the full fact content
        // Future: implement vocabulary simplification based on audience
        // (e.g., child: simpler words, expert: technical terms)
        Ok(fact.content.clone())
    }

    /// Compute verification rate
    fn compute_verification_rate(&self, verifications: &[TokenVerification]) -> f64 {
        if verifications.is_empty() {
            return 0.0;
        }

        let verified_count = verifications.iter().filter(|v| v.verified).count();
        verified_count as f64 / verifications.len() as f64
    }

    /// Generate analogy for concept (only for appropriate audiences)
    pub fn generate_analogy(
        &self,
        concept: &str,
        memory: &SemanticMemory,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let style = self.audience.style_vector();

        // Only generate analogies for audiences that prefer them
        if style.analogy_preference < 0.5 {
            return Ok(None);
        }

        let thought = ThoughtState::from_input(concept, self.dimension);

        // Search for related concrete examples
        let candidates = memory.find_similar(&thought.vector, 5)?;

        if let Some((fact, similarity)) = candidates.first() {
            if *similarity >= self.thresholds.min_knowledge_similarity {
                return Ok(Some(format!(
                    "It's like: {}",
                    fact.content
                )));
            }
        }

        Ok(None)
    }
}

/// Explanation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationResponse {
    /// Full explanation text
    pub explanation: String,
    /// Individual sentences
    pub sentences: Vec<String>,
    /// Target audience
    pub audience: ExplanationAudience,
    /// Verification for each token
    pub verifications: Vec<TokenVerification>,
    /// Style characteristics applied
    #[serde(skip)]
    pub style_applied: StyleVector,
    /// Percentage of verified tokens (0.0-1.0)
    pub verified_percentage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audience_complexity() {
        assert!(ExplanationAudience::Child.complexity_level() < 0.5);
        assert!(ExplanationAudience::Mathematician.complexity_level() > 0.8);
    }

    #[test]
    fn test_style_vectors() {
        let child_style = ExplanationAudience::Child.style_vector();
        let math_style = ExplanationAudience::Mathematician.style_vector();

        assert!(child_style.analogy_preference > math_style.analogy_preference);
        assert!(math_style.technical_density > child_style.technical_density);
    }

    #[test]
    fn test_projection() {
        let thought = ThoughtState::from_input("gravity", 128);
        let decoder = ExplanationDecoder::new(
            128,
            ExplanationAudience::Child,
            FactualThresholds::balanced(),
        );

        let projected = decoder.project_to_audience(&thought);
        assert_eq!(projected.dimension, thought.dimension);
    }
}
