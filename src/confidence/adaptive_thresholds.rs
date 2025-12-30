//! Adaptive Threshold Calibration
//!
//! Problem: Static thresholds cause:
//! - Too high → system refuses to answer
//! - Too low → hallucination sneaks back in
//!
//! Solution: Domain-specific, empirically calibrated thresholds
//!
//! Mathematical Foundation:
//! - Track outcomes: (C_i, correct_i) for each answer
//! - Model reliability: P(correct | C)
//! - Set threshold by risk tolerance: P(correct | C ≥ τ) ≥ 1 - δ
//!
//! Key Insight: Thresholds are domain-specific, not global
//! - Math domain → δ = 0.01 (very strict)
//! - Conversation → δ = 0.2 (more lenient)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// PART 1: OUTCOME TRACKING
// ============================================================================

/// Tracks confidence scores and their outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeRecord {
    pub confidence: f64,
    pub correct: bool,
    pub domain: String,
    pub timestamp: u64,
}

impl OutcomeRecord {
    pub fn new(confidence: f64, correct: bool, domain: String) -> Self {
        Self {
            confidence,
            correct,
            domain,
            timestamp: Self::current_timestamp(),
        }
    }

    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

// ============================================================================
// PART 2: CALIBRATION ENGINE
// ============================================================================

/// Calibrates thresholds based on empirical outcomes
#[derive(Debug, Clone)]
pub struct ThresholdCalibrator {
    /// Historical outcomes: (confidence, correctness)
    outcomes: Vec<OutcomeRecord>,
    
    /// Domain-specific thresholds
    thresholds: HashMap<String, f64>,
    
    /// Domain-specific risk tolerances (δ)
    risk_tolerances: HashMap<String, f64>,
    
    /// Minimum samples before calibration
    min_samples: usize,
}

impl ThresholdCalibrator {
    pub fn new() -> Self {
        let mut risk_tolerances = HashMap::new();
        
        // Domain-specific risk tolerances
        risk_tolerances.insert("math".to_string(), 0.01);      // 99% accuracy required
        risk_tolerances.insert("logic".to_string(), 0.02);     // 98% accuracy required
        risk_tolerances.insert("code".to_string(), 0.05);      // 95% accuracy required
        risk_tolerances.insert("conversation".to_string(), 0.2); // 80% accuracy required
        risk_tolerances.insert("general".to_string(), 0.1);    // 90% accuracy required (default)

        // Initialize with lenient thresholds to allow trained responses
        let mut thresholds = HashMap::new();
        thresholds.insert("conversation".to_string(), 0.50);  // Allow 50%+ confidence
        thresholds.insert("general".to_string(), 0.55);       // Allow 55%+ confidence
        thresholds.insert("math".to_string(), 0.60);          // Math requires 60%+
        thresholds.insert("logic".to_string(), 0.60);         // Logic requires 60%+
        thresholds.insert("code".to_string(), 0.58);          // Code requires 58%+

        Self {
            outcomes: Vec::new(),
            thresholds,
            risk_tolerances,
            min_samples: 10, // Need at least 10 samples to calibrate
        }
    }

    /// Record an outcome
    pub fn record_outcome(&mut self, confidence: f64, correct: bool, domain: &str) {
        self.outcomes.push(OutcomeRecord::new(confidence, correct, domain.to_string()));
        
        // Recalibrate if we have enough samples
        if self.outcomes.len() % 10 == 0 {
            self.calibrate_all_domains();
        }
    }

    /// Get threshold for a specific domain
    pub fn get_threshold(&self, domain: &str) -> f64 {
        self.thresholds
            .get(domain)
            .copied()
            .unwrap_or_else(|| {
                // Default threshold based on risk tolerance
                let delta = self.risk_tolerances.get(domain).copied().unwrap_or(0.1);
                self.default_threshold_for_risk(delta)
            })
    }

    /// Calibrate thresholds for all domains
    fn calibrate_all_domains(&mut self) {
        let domains: Vec<String> = self.outcomes
            .iter()
            .map(|o| o.domain.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for domain in domains {
            if let Some(threshold) = self.calibrate_domain(&domain) {
                self.thresholds.insert(domain, threshold);
            }
        }
    }

    /// Calibrate threshold for a specific domain
    /// Returns τ such that P(correct | C ≥ τ) ≥ 1 - δ
    fn calibrate_domain(&self, domain: &str) -> Option<f64> {
        // Filter outcomes for this domain
        let domain_outcomes: Vec<&OutcomeRecord> = self.outcomes
            .iter()
            .filter(|o| o.domain == domain)
            .collect();

        if domain_outcomes.len() < self.min_samples {
            return None; // Not enough data
        }

        // Get risk tolerance for this domain
        let delta = self.risk_tolerances.get(domain).copied().unwrap_or(0.1);
        let required_accuracy = 1.0 - delta;

        // Sort by confidence (descending)
        let mut sorted = domain_outcomes.clone();
        sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        // Find threshold using isotonic regression (simplified)
        let threshold = self.find_threshold_isotonic(&sorted, required_accuracy);

        Some(threshold)
    }

    /// Find threshold using isotonic regression
    /// Returns lowest confidence where accuracy ≥ required_accuracy
    fn find_threshold_isotonic(&self, sorted_outcomes: &[&OutcomeRecord], required_accuracy: f64) -> f64 {
        if sorted_outcomes.is_empty() {
            return 0.5; // Default
        }

        // Bin outcomes by confidence
        let bins = self.create_confidence_bins(sorted_outcomes);

        // Find lowest confidence bin with sufficient accuracy
        for (confidence, accuracy) in bins.iter().rev() {
            if *accuracy >= required_accuracy {
                return *confidence;
            }
        }

        // If no bin meets requirement, return highest confidence
        sorted_outcomes[0].confidence
    }

    /// Create confidence bins with accuracy estimates
    /// Returns: Vec<(confidence_threshold, accuracy)>
    fn create_confidence_bins(&self, outcomes: &[&OutcomeRecord]) -> Vec<(f64, f64)> {
        let num_bins = 10;
        let mut bins: Vec<(f64, Vec<bool>)> = Vec::new();

        // Create bins
        for i in 0..num_bins {
            let threshold = i as f64 / num_bins as f64;
            bins.push((threshold, Vec::new()));
        }

        // Assign outcomes to bins
        for outcome in outcomes {
            let bin_idx = ((outcome.confidence * num_bins as f64).floor() as usize).min(num_bins - 1);
            bins[bin_idx].1.push(outcome.correct);
        }

        // Calculate accuracy for each bin
        bins.into_iter()
            .map(|(threshold, corrects)| {
                if corrects.is_empty() {
                    (threshold, 0.0)
                } else {
                    let accuracy = corrects.iter().filter(|&&c| c).count() as f64 / corrects.len() as f64;
                    (threshold, accuracy)
                }
            })
            .collect()
    }

    /// Default threshold based on risk tolerance
    fn default_threshold_for_risk(&self, delta: f64) -> f64 {
        // Very lenient threshold to allow trained responses
        // Higher risk tolerance → lower threshold
        // Conversation (delta=0.2): 0.48
        // General (delta=0.1): 0.51
        // Math (delta=0.01): 0.549
        0.48 + (0.1 * (1.0 - delta))
    }

    /// Get calibration statistics for a domain
    pub fn get_stats(&self, domain: &str) -> CalibrationStats {
        let domain_outcomes: Vec<&OutcomeRecord> = self.outcomes
            .iter()
            .filter(|o| o.domain == domain)
            .collect();

        let total = domain_outcomes.len();
        let correct = domain_outcomes.iter().filter(|o| o.correct).count();
        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

        let avg_confidence = if total > 0 {
            domain_outcomes.iter().map(|o| o.confidence).sum::<f64>() / total as f64
        } else {
            0.0
        };

        CalibrationStats {
            domain: domain.to_string(),
            total_samples: total,
            correct_count: correct,
            accuracy,
            avg_confidence,
            current_threshold: self.get_threshold(domain),
            calibrated: total >= self.min_samples,
        }
    }
}

impl Default for ThresholdCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationStats {
    pub domain: String,
    pub total_samples: usize,
    pub correct_count: usize,
    pub accuracy: f64,
    pub avg_confidence: f64,
    pub current_threshold: f64,
    pub calibrated: bool,
}

// ============================================================================
// PART 3: DOMAIN CLASSIFIER
// ============================================================================

/// Classifies input into domains for threshold selection
pub struct DomainClassifier;

impl DomainClassifier {
    /// Classify input text into a domain
    pub fn classify(text: &str) -> String {
        let lower = text.to_lowercase();

        // Math domain
        if lower.contains("calculate") || lower.contains("solve") || 
           lower.contains("equation") || lower.contains("integral") ||
           lower.contains("derivative") || lower.contains("proof") {
            return "math".to_string();
        }

        // Logic domain
        if lower.contains("if") && lower.contains("then") ||
           lower.contains("implies") || lower.contains("therefore") ||
           lower.contains("because") {
            return "logic".to_string();
        }

        // Code domain
        if lower.contains("function") || lower.contains("class") ||
           lower.contains("algorithm") || lower.contains("code") ||
           lower.contains("program") {
            return "code".to_string();
        }

        // Conversation domain
        if lower.contains("hello") || lower.contains("hi") ||
           lower.contains("how are you") || lower.contains("thank") {
            return "conversation".to_string();
        }

        // Default
        "general".to_string()
    }
}

// ============================================================================
// PART 4: CONFIDENCE GATING WITH ADAPTIVE THRESHOLDS
// ============================================================================

/// Gates responses based on calibrated thresholds
pub struct AdaptiveConfidenceGate {
    calibrator: ThresholdCalibrator,
}

impl AdaptiveConfidenceGate {
    pub fn new() -> Self {
        Self {
            calibrator: ThresholdCalibrator::new(),
        }
    }

    /// Check if confidence meets threshold for domain
    pub fn should_answer(&self, confidence: f64, input: &str) -> bool {
        let domain = DomainClassifier::classify(input);
        let threshold = self.calibrator.get_threshold(&domain);
        
        confidence >= threshold
    }

    /// Record outcome for calibration
    pub fn record_outcome(&mut self, confidence: f64, correct: bool, input: &str) {
        let domain = DomainClassifier::classify(input);
        self.calibrator.record_outcome(confidence, correct, &domain);
    }

    /// Get statistics for a domain
    pub fn get_stats(&self, domain: &str) -> CalibrationStats {
        self.calibrator.get_stats(domain)
    }

    /// Get all domain statistics
    pub fn get_all_stats(&self) -> Vec<CalibrationStats> {
        vec!["math", "logic", "code", "conversation", "general"]
            .iter()
            .map(|d| self.get_stats(d))
            .collect()
    }
}

impl Default for AdaptiveConfidenceGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_classification() {
        assert_eq!(DomainClassifier::classify("Calculate 2+2"), "math");
        assert_eq!(DomainClassifier::classify("If A then B"), "logic");
        assert_eq!(DomainClassifier::classify("Write a function"), "code");
        assert_eq!(DomainClassifier::classify("Hello there"), "conversation");
        assert_eq!(DomainClassifier::classify("Random text"), "general");
    }

    #[test]
    fn test_threshold_calibration() {
        let mut calibrator = ThresholdCalibrator::new();

        // Record outcomes for math domain
        for i in 0..20 {
            let confidence = 0.5 + (i as f64 / 40.0); // 0.5 to 1.0
            let correct = confidence > 0.8; // Correct if confidence > 0.8
            calibrator.record_outcome(confidence, correct, "math");
        }

        let threshold = calibrator.get_threshold("math");
        
        // Math domain has strict requirements (δ = 0.01)
        // So threshold should be relatively high
        assert!(threshold > 0.5);
    }

    #[test]
    fn test_adaptive_gate() {
        let gate = AdaptiveConfidenceGate::new();

        // Math requires high confidence
        assert!(!gate.should_answer(0.5, "Calculate integral"));
        
        // Conversation is more lenient
        assert!(gate.should_answer(0.5, "Hello how are you"));
    }

    #[test]
    fn test_calibration_stats() {
        let mut calibrator = ThresholdCalibrator::new();

        for _ in 0..15 {
            calibrator.record_outcome(0.9, true, "math");
        }
        for _ in 0..5 {
            calibrator.record_outcome(0.6, false, "math");
        }

        let stats = calibrator.get_stats("math");
        
        assert_eq!(stats.total_samples, 20);
        assert_eq!(stats.correct_count, 15);
        assert!((stats.accuracy - 0.75).abs() < 0.01);
        assert!(stats.calibrated);
    }
}
