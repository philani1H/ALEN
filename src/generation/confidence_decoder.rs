//! Confidence-Calibrated Decoder
//!
//! Ensures output confidence matches actual accuracy.
//! Key principle: Only answer when confidence justifies it.
//!
//! Mathematical Foundation:
//! - Calibration: P(correct | confidence = c) ≈ c
//! - Refusal threshold: τ below which system says "I don't know"
//! - Uncertainty quantification: Multiple sources of uncertainty
//!
//! This prevents hallucination by refusing to answer when uncertain.

use serde::{Deserialize, Serialize};
use crate::memory::Episode;

// ============================================================================
// PART 1: CONFIDENCE CALIBRATION
// ============================================================================

/// Confidence-calibrated decoder
/// Ensures system only answers when confidence is justified
#[derive(Debug, Clone)]
pub struct ConfidenceDecoder {
    /// Minimum confidence to provide an answer
    pub refusal_threshold: f64,
    /// Minimum similarity for retrieval
    pub similarity_threshold: f64,
    /// Calibration parameters
    pub calibration: CalibrationParams,
    /// Refusal messages by uncertainty type
    pub refusal_messages: RefusalMessages,
}

#[derive(Debug, Clone)]
pub struct CalibrationParams {
    /// Temperature for confidence scaling
    pub temperature: f64,
    /// Bias correction (if system is over/under-confident)
    pub bias: f64,
    /// Platt scaling parameters (learned from data)
    pub platt_a: f64,
    pub platt_b: f64,
}

impl Default for CalibrationParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            bias: 0.0,
            platt_a: 1.0,
            platt_b: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RefusalMessages {
    pub low_confidence: String,
    pub no_knowledge: String,
    pub ambiguous: String,
    pub needs_clarification: String,
}

impl Default for RefusalMessages {
    fn default() -> Self {
        Self {
            low_confidence: "I'm not confident enough to answer that accurately. Could you rephrase or provide more context?".to_string(),
            no_knowledge: "I don't have enough knowledge about that topic yet. I'm still learning!".to_string(),
            ambiguous: "Your question could mean several things. Could you clarify what you're asking about?".to_string(),
            needs_clarification: "I need more information to answer that properly. Could you tell me more?".to_string(),
        }
    }
}

impl ConfidenceDecoder {
    pub fn new() -> Self {
        Self {
            refusal_threshold: 0.3,      // Only answer if >30% confident
            similarity_threshold: 0.1,    // Only retrieve if >10% similar (low for demo)
            calibration: CalibrationParams::default(),
            refusal_messages: RefusalMessages::default(),
        }
    }

    /// Decode with confidence calibration
    /// Returns answer only if confidence is above threshold
    pub fn decode(
        &self,
        retrieved_episodes: &[Episode],
        query_embedding: &[f64],
    ) -> DecoderOutput {
        if retrieved_episodes.is_empty() {
            return DecoderOutput::refuse(
                RefusalReason::NoKnowledge,
                &self.refusal_messages.no_knowledge,
                0.0,
            );
        }

        // Calculate similarity for best match
        let best_episode = &retrieved_episodes[0];
        let similarity = self.calculate_similarity(query_embedding, &best_episode.thought_vector);

        // Check similarity threshold
        if similarity < self.similarity_threshold {
            return DecoderOutput::refuse(
                RefusalReason::LowSimilarity,
                &self.refusal_messages.no_knowledge,
                similarity,
            );
        }

        // Calibrate confidence
        let raw_confidence = best_episode.confidence_score * similarity;
        let calibrated_confidence = self.calibrate_confidence(raw_confidence);

        // Check confidence threshold
        if calibrated_confidence < self.refusal_threshold {
            return DecoderOutput::refuse(
                RefusalReason::LowConfidence,
                &self.refusal_messages.low_confidence,
                calibrated_confidence,
            );
        }

        // Check for ambiguity (multiple high-confidence answers)
        if self.is_ambiguous(retrieved_episodes, similarity) {
            return DecoderOutput::refuse(
                RefusalReason::Ambiguous,
                &self.refusal_messages.ambiguous,
                calibrated_confidence,
            );
        }

        // Provide answer with calibrated confidence
        DecoderOutput::answer(
            best_episode.answer_output.clone(),
            calibrated_confidence,
            similarity,
            UncertaintyBreakdown {
                epistemic: 1.0 - best_episode.confidence_score,
                aleatoric: 1.0 - similarity,
                total: 1.0 - calibrated_confidence,
            },
        )
    }

    /// Calibrate confidence using temperature scaling and Platt scaling
    /// Ensures P(correct | confidence = c) ≈ c
    fn calibrate_confidence(&self, raw_confidence: f64) -> f64 {
        // Temperature scaling
        let temp_scaled = raw_confidence / self.calibration.temperature;
        
        // Platt scaling: σ(a * logit(p) + b)
        let logit = (temp_scaled / (1.0 - temp_scaled + 1e-10)).ln();
        let platt_scaled = 1.0 / (1.0 + (-self.calibration.platt_a * logit - self.calibration.platt_b).exp());
        
        // Apply bias correction
        let calibrated = platt_scaled + self.calibration.bias;
        
        calibrated.max(0.0).min(1.0)
    }

    /// Calculate cosine similarity
    fn calculate_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        (dot / (norm_a * norm_b)).max(-1.0).min(1.0)
    }

    /// Check if query is ambiguous (multiple good answers)
    fn is_ambiguous(&self, episodes: &[Episode], best_similarity: f64) -> bool {
        if episodes.len() < 2 {
            return false;
        }

        // Check if second-best is also high confidence
        let second_best_conf = episodes[1].confidence_score;
        let best_conf = episodes[0].confidence_score;

        // Ambiguous if second answer is within 20% of best
        (second_best_conf / best_conf) > 0.8 && second_best_conf > 0.6
    }

    /// Update calibration parameters based on feedback
    /// This is called when we get ground truth about whether answer was correct
    pub fn update_calibration(&mut self, predicted_confidence: f64, was_correct: bool) {
        // Simple exponential moving average for bias correction
        let error = if was_correct { 0.0 } else { predicted_confidence };
        self.calibration.bias = 0.9 * self.calibration.bias - 0.1 * error;
    }
}

// ============================================================================
// PART 2: DECODER OUTPUT
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderOutput {
    /// Whether system provided an answer
    pub answered: bool,
    /// The answer (if provided)
    pub answer: Option<String>,
    /// Calibrated confidence
    pub confidence: f64,
    /// Similarity to retrieved knowledge
    pub similarity: f64,
    /// Reason for refusal (if refused)
    pub refusal_reason: Option<RefusalReason>,
    /// Refusal message (if refused)
    pub refusal_message: Option<String>,
    /// Uncertainty breakdown
    pub uncertainty: UncertaintyBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefusalReason {
    NoKnowledge,
    LowConfidence,
    LowSimilarity,
    Ambiguous,
    NeedsClarification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyBreakdown {
    /// Epistemic uncertainty (knowledge uncertainty)
    pub epistemic: f64,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric: f64,
    /// Total uncertainty
    pub total: f64,
}

impl DecoderOutput {
    pub fn answer(
        answer: String,
        confidence: f64,
        similarity: f64,
        uncertainty: UncertaintyBreakdown,
    ) -> Self {
        Self {
            answered: true,
            answer: Some(answer),
            confidence,
            similarity,
            refusal_reason: None,
            refusal_message: None,
            uncertainty,
        }
    }

    pub fn refuse(reason: RefusalReason, message: &str, confidence: f64) -> Self {
        Self {
            answered: false,
            answer: None,
            confidence,
            similarity: 0.0,
            refusal_reason: Some(reason),
            refusal_message: Some(message.to_string()),
            uncertainty: UncertaintyBreakdown {
                epistemic: 1.0,
                aleatoric: 1.0,
                total: 1.0,
            },
        }
    }
}

// ============================================================================
// PART 3: CONFIDENCE CALIBRATION METRICS
// ============================================================================

/// Tracks calibration quality over time
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Bins for calibration curve
    pub bins: Vec<CalibrationBin>,
    /// Expected Calibration Error (ECE)
    pub ece: f64,
    /// Maximum Calibration Error (MCE)
    pub mce: f64,
}

#[derive(Debug, Clone)]
pub struct CalibrationBin {
    /// Confidence range [lower, upper)
    pub confidence_range: (f64, f64),
    /// Average predicted confidence in this bin
    pub avg_confidence: f64,
    /// Actual accuracy in this bin
    pub accuracy: f64,
    /// Number of samples in this bin
    pub count: usize,
}

impl CalibrationMetrics {
    /// Calculate calibration metrics from predictions and outcomes
    pub fn calculate(predictions: &[(f64, bool)]) -> Self {
        let num_bins = 10;
        let mut bins = vec![CalibrationBin {
            confidence_range: (0.0, 0.0),
            avg_confidence: 0.0,
            accuracy: 0.0,
            count: 0,
        }; num_bins];

        // Initialize bins
        for i in 0..num_bins {
            bins[i].confidence_range = (i as f64 / num_bins as f64, (i + 1) as f64 / num_bins as f64);
        }

        // Assign predictions to bins
        for &(confidence, correct) in predictions {
            let bin_idx = ((confidence * num_bins as f64).floor() as usize).min(num_bins - 1);
            bins[bin_idx].avg_confidence += confidence;
            bins[bin_idx].accuracy += if correct { 1.0 } else { 0.0 };
            bins[bin_idx].count += 1;
        }

        // Calculate averages
        let mut ece: f64 = 0.0;
        let mut mce: f64 = 0.0;
        let total_samples = predictions.len() as f64;

        for bin in &mut bins {
            if bin.count > 0 {
                bin.avg_confidence /= bin.count as f64;
                bin.accuracy /= bin.count as f64;

                let calibration_error = (bin.avg_confidence - bin.accuracy).abs();
                ece += (bin.count as f64 / total_samples) * calibration_error;
                mce = mce.max(calibration_error as f64);
            }
        }

        Self { bins, ece, mce }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_decoder() {
        let decoder = ConfidenceDecoder::new();
        
        // Test with empty episodes (should refuse)
        let query_embedding = vec![0.5; 128];
        let output = decoder.decode(&[], &query_embedding);
        
        assert!(!output.answered);
        assert!(output.refusal_reason.is_some());
    }

    #[test]
    fn test_calibration() {
        let decoder = ConfidenceDecoder::new();
        
        // Test confidence calibration
        let raw_confidence = 0.8;
        let calibrated = decoder.calibrate_confidence(raw_confidence);
        
        assert!(calibrated >= 0.0 && calibrated <= 1.0);
    }

    #[test]
    fn test_similarity_threshold() {
        let decoder = ConfidenceDecoder::new();
        
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        
        let similarity = decoder.calculate_similarity(&a, &b);
        assert!(similarity < decoder.similarity_threshold);
    }

    #[test]
    fn test_calibration_metrics() {
        let predictions = vec![
            (0.9, true),
            (0.8, true),
            (0.7, false),
            (0.6, true),
            (0.5, false),
        ];

        let metrics = CalibrationMetrics::calculate(&predictions);
        assert!(metrics.ece >= 0.0 && metrics.ece <= 1.0);
        assert!(metrics.mce >= 0.0 && metrics.mce <= 1.0);
    }
}
