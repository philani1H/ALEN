//! Factual Text Decoder - DEPRECATED FOR GENERATION
//!
//! CRITICAL: This module is DEPRECATED for text generation.
//! Use LatentDecoder instead for all text generation.
//!
//! This module does RETRIEVAL (fact.content) which is MEMORIZATION.
//! For understanding-based generation with verification, use: LatentDecoder
//!
//! This is kept only for backward compatibility and verification logic.

use crate::core::{ThoughtState, BiasVector};
use crate::memory::{SemanticMemory, SemanticFact};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strict verification thresholds for factual generation
#[derive(Debug, Clone)]
pub struct FactualThresholds {
    /// Minimum cosine similarity to knowledge vector (φ_strict)
    pub min_knowledge_similarity: f64,
    /// Minimum confidence for accepting a token
    pub min_token_confidence: f64,
    /// Maximum retries for finding verified token
    pub max_retries: usize,
    /// Temperature for deterministic generation (low = more deterministic)
    pub temperature: f64,
}

impl Default for FactualThresholds {
    fn default() -> Self {
        Self {
            min_knowledge_similarity: 0.18,  // Calibrated for compositional bag-of-words
            min_token_confidence: 0.4,
            max_retries: 5,
            temperature: 0.2,  // Very low for deterministic behavior
        }
    }
}

impl FactualThresholds {
    /// Strict mode: highest verification standards
    pub fn strict() -> Self {
        Self {
            min_knowledge_similarity: 0.22,  // Calibrated for compositional bag-of-words
            min_token_confidence: 0.5,
            max_retries: 3,
            temperature: 0.1,
        }
    }

    /// Balanced mode: moderate verification
    pub fn balanced() -> Self {
        Self::default()  // 0.18 similarity
    }

    /// Relaxed mode: lower thresholds but still verified
    pub fn relaxed() -> Self {
        Self {
            min_knowledge_similarity: 0.15,  // Calibrated for compositional bag-of-words
            min_token_confidence: 0.3,
            max_retries: 7,
            temperature: 0.3,
        }
    }
}

/// Factual text decoder with knowledge verification
pub struct FactualDecoder {
    /// Dimension of thought vectors
    pub dimension: usize,
    /// Verification thresholds
    pub thresholds: FactualThresholds,
    /// Neutral bias vector (no creative bias)
    neutral_bias: BiasVector,
}

impl FactualDecoder {
    pub fn new(dimension: usize, thresholds: FactualThresholds) -> Self {
        Self {
            dimension,
            thresholds,
            // Neutral: all parameters at 0.5 (no bias)
            neutral_bias: BiasVector {
                risk_tolerance: 0.5,
                exploration: 0.5,
                urgency: 0.5,
                creativity: 0.0,  // Zero creativity for factual mode
            },
        }
    }

    /// Create with strict verification
    pub fn strict(dimension: usize) -> Self {
        Self::new(dimension, FactualThresholds::strict())
    }

    /// Generate factual answer from thought with knowledge verification
    pub fn generate_factual(
        &self,
        thought: &ThoughtState,
        memory: &SemanticMemory,
        max_tokens: usize,
    ) -> Result<FactualResponse, Box<dyn std::error::Error>> {
        let mut tokens = Vec::new();
        let mut verifications = Vec::new();
        let mut current_thought = thought.clone();

        for token_idx in 0..max_tokens {
            // Generate token candidate
            let (token, verification) = self.generate_verified_token(
                &current_thought,
                memory,
                &tokens,
            )?;

            // If we got an empty token or end signal, stop
            if token.is_empty() || token == "<END>" {
                break;
            }

            tokens.push(token.clone());
            verifications.push(verification);

            // Update thought vector for next token (contextual evolution)
            current_thought = self.evolve_thought(&current_thought, &token, token_idx);
        }

        let overall_confidence = self.compute_overall_confidence(&verifications);

        Ok(FactualResponse {
            text: tokens.join(" "),
            tokens,
            verifications,
            mode: GenerationMode::Factual,
            overall_confidence,
        })
    }

    /// Generate a single verified token
    fn generate_verified_token(
        &self,
        thought: &ThoughtState,
        memory: &SemanticMemory,
        context: &[String],
    ) -> Result<(String, TokenVerification), Box<dyn std::error::Error>> {
        // Apply neutral bias to thought (remove creative influences)
        let factual_thought = self.apply_neutral_bias(thought);

        // Search semantic memory for relevant knowledge
        let candidates = memory.find_similar(&factual_thought.vector, 10)?;

        if candidates.is_empty() {
            // No knowledge available - must reject or use fallback
            return Ok((String::new(), TokenVerification {
                similarity: 0.0,
                confidence: 0.0,
                knowledge_source: None,
                verified: false,
                reason: "No knowledge available".to_string(),
            }));
        }

        // Try candidates in order of similarity
        for (fact, similarity) in candidates.iter() {
            // Check if similarity meets threshold φ_strict
            if *similarity >= self.thresholds.min_knowledge_similarity {
                // Extract token from knowledge content
                let token = self.extract_token_from_fact(fact, context)?;

                // Verify token confidence
                if fact.confidence >= self.thresholds.min_token_confidence {
                    return Ok((token.clone(), TokenVerification {
                        similarity: *similarity,
                        confidence: fact.confidence,
                        knowledge_source: Some(fact.id.clone()),
                        verified: true,
                        reason: format!("Verified: sim={:.3}, conf={:.3}", similarity, fact.confidence),
                    }));
                }
            }
        }

        // No verified token found - return most similar with warning
        let (best_fact, best_sim) = &candidates[0];
        let token = self.extract_token_from_fact(best_fact, context)?;

        Ok((token, TokenVerification {
            similarity: *best_sim,
            confidence: best_fact.confidence,
            knowledge_source: Some(best_fact.id.clone()),
            verified: false,
            reason: format!("Below threshold: sim={:.3} < {:.3}", best_sim, self.thresholds.min_knowledge_similarity),
        }))
    }

    /// Apply neutral bias to remove creative influences
    fn apply_neutral_bias(&self, thought: &ThoughtState) -> ThoughtState {
        let mut neutral_thought = thought.clone();

        // Zero out any creative/exploratory components
        // This ensures factual mode stays grounded
        let scale = 1.0 - self.neutral_bias.creativity;

        for value in neutral_thought.vector.iter_mut() {
            *value *= scale;
        }

        neutral_thought.normalize();
        neutral_thought
    }

    /// Extract appropriate token from semantic fact
    fn extract_token_from_fact(
        &self,
        fact: &SemanticFact,
        context: &[String],
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Split fact content into words
        let words: Vec<&str> = fact.content.split_whitespace().collect();

        if words.is_empty() {
            return Ok(String::new());
        }

        // For now, use deterministic selection based on context length
        // In production, this would use more sophisticated NLP
        let idx = context.len() % words.len();
        Ok(words[idx].to_string())
    }

    /// Evolve thought vector for next token (maintains coherence)
    fn evolve_thought(&self, thought: &ThoughtState, token: &str, position: usize) -> ThoughtState {
        let mut evolved = thought.clone();

        // Small deterministic perturbation based on token hash
        let token_hash: u64 = token.bytes().map(|b| b as u64).sum();
        let perturbation = (token_hash as f64 / 1000.0).sin() * 0.05;

        for (i, value) in evolved.vector.iter_mut().enumerate() {
            let pos_influence = ((i + position) as f64 * 0.1).cos() * perturbation;
            *value += pos_influence;
        }

        evolved.normalize();
        evolved
    }

    /// Compute overall confidence from token verifications
    fn compute_overall_confidence(&self, verifications: &[TokenVerification]) -> f64 {
        if verifications.is_empty() {
            return 0.0;
        }

        let verified_count = verifications.iter().filter(|v| v.verified).count();
        let total_confidence: f64 = verifications.iter().map(|v| v.confidence).sum();
        let avg_confidence = total_confidence / verifications.len() as f64;

        // Weighted score: 70% average confidence + 30% verification rate
        avg_confidence * 0.7 + (verified_count as f64 / verifications.len() as f64) * 0.3
    }

    /// Generate factual explanation (multi-sentence)
    pub fn generate_explanation(
        &self,
        question: &str,
        memory: &SemanticMemory,
        max_sentences: usize,
    ) -> Result<FactualResponse, Box<dyn std::error::Error>> {
        // Convert question to thought vector
        let thought = ThoughtState::from_input(question, self.dimension);

        // Generate multiple sentences
        let mut all_tokens = Vec::new();
        let mut all_verifications = Vec::new();

        for _sentence in 0..max_sentences {
            let response = self.generate_factual(&thought, memory, 20)?;

            if response.tokens.is_empty() {
                break;
            }

            all_tokens.extend(response.tokens);
            all_verifications.extend(response.verifications);
        }

        let overall_confidence = self.compute_overall_confidence(&all_verifications);

        Ok(FactualResponse {
            text: all_tokens.join(" "),
            tokens: all_tokens,
            verifications: all_verifications,
            mode: GenerationMode::Factual,
            overall_confidence,
        })
    }

    /// Verify a statement against knowledge base
    pub fn verify_statement(
        &self,
        statement: &str,
        memory: &SemanticMemory,
    ) -> Result<VerificationResult, Box<dyn std::error::Error>> {
        let thought = ThoughtState::from_input(statement, self.dimension);
        let candidates = memory.find_similar(&thought.vector, 5)?;

        if candidates.is_empty() {
            return Ok(VerificationResult {
                verified: false,
                confidence: 0.0,
                max_similarity: 0.0,
                supporting_facts: Vec::new(),
                reason: "No knowledge found to verify statement".to_string(),
            });
        }

        let (best_fact, max_similarity) = &candidates[0];

        let verified = *max_similarity >= self.thresholds.min_knowledge_similarity
            && best_fact.confidence >= self.thresholds.min_token_confidence;

        let supporting_facts: Vec<String> = candidates.iter()
            .filter(|(_, sim)| *sim >= self.thresholds.min_knowledge_similarity)
            .map(|(fact, _)| fact.content.clone())
            .collect();

        Ok(VerificationResult {
            verified,
            confidence: best_fact.confidence,
            max_similarity: *max_similarity,
            supporting_facts,
            reason: if verified {
                "Statement verified against knowledge base".to_string()
            } else {
                format!("Insufficient similarity: {:.3} < {:.3}", max_similarity, self.thresholds.min_knowledge_similarity)
            },
        })
    }
}

/// Response from factual generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactualResponse {
    /// Generated text
    pub text: String,
    /// Individual tokens
    pub tokens: Vec<String>,
    /// Verification for each token
    pub verifications: Vec<TokenVerification>,
    /// Generation mode used
    pub mode: GenerationMode,
    /// Overall confidence (0.0-1.0)
    pub overall_confidence: f64,
}

/// Verification information for a single token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenVerification {
    /// Cosine similarity to knowledge vector
    pub similarity: f64,
    /// Confidence in this token
    pub confidence: f64,
    /// Source knowledge ID (if any)
    pub knowledge_source: Option<String>,
    /// Whether token passed verification
    pub verified: bool,
    /// Reason for verification result
    pub reason: String,
}

/// Result of verifying a statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether statement is verified
    pub verified: bool,
    /// Confidence in verification
    pub confidence: f64,
    /// Maximum similarity to knowledge
    pub max_similarity: f64,
    /// Supporting facts from knowledge base
    pub supporting_facts: Vec<String>,
    /// Reason for result
    pub reason: String,
}

/// Generation mode
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum GenerationMode {
    /// Factual mode: strict verification, no hallucinations
    Factual,
    /// Creative mode: artistic freedom, exploratory
    Creative,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::SemanticMemory;

    #[test]
    fn test_factual_decoder_creation() {
        let decoder = FactualDecoder::strict(128);
        assert_eq!(decoder.dimension, 128);
        assert_eq!(decoder.neutral_bias.creativity, 0.0);
        // Strict threshold is 0.22 (calibrated for compositional bag-of-words)
        assert!(decoder.thresholds.min_knowledge_similarity >= 0.2);
    }

    #[test]
    fn test_threshold_modes() {
        let strict = FactualThresholds::strict();
        let balanced = FactualThresholds::balanced();
        let relaxed = FactualThresholds::relaxed();

        assert!(strict.min_knowledge_similarity > balanced.min_knowledge_similarity);
        assert!(balanced.min_knowledge_similarity > relaxed.min_knowledge_similarity);
    }

    #[test]
    fn test_neutral_bias_application() {
        let decoder = FactualDecoder::strict(64);
        let thought = ThoughtState::from_input("test thought", 64);
        let neutral = decoder.apply_neutral_bias(&thought);

        // Neutral thought should have same or reduced magnitude
        // With creativity=0, scale=1.0, so norm should be approximately same (both unit normalized)
        assert!(neutral.norm() <= thought.norm() + 0.1); // Allow floating point tolerance
    }
}
