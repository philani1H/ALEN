//! Calibration Tracking System
//!
//! Tracks and measures calibration quality over time using:
//! - Expected Calibration Error (ECE)
//! - Maximum Calibration Error (MCE)
//! - Brier Score
//!
//! Mathematical Foundation:
//! ECE = ∑_{m=1}^{M} (|B_m|/n) |acc(B_m) - conf(B_m)|
//! MCE = max_{m=1,...,M} |acc(B_m) - conf(B_m)|
//! Brier = (1/N) ∑_{i=1}^{N} (p_i - y_i)²

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// PART 1: CALIBRATION BIN
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBin {
    /// Bin range: [lower, upper)
    pub lower: f64,
    pub upper: f64,
    
    /// Number of predictions in this bin
    pub count: usize,
    
    /// Number of correct predictions
    pub correct_count: usize,
    
    /// Sum of confidences
    pub confidence_sum: f64,
    
    /// Average confidence
    pub avg_confidence: f64,
    
    /// Empirical accuracy
    pub accuracy: f64,
}

impl CalibrationBin {
    pub fn new(lower: f64, upper: f64) -> Self {
        Self {
            lower,
            upper,
            count: 0,
            correct_count: 0,
            confidence_sum: 0.0,
            avg_confidence: 0.0,
            accuracy: 0.0,
        }
    }
    
    pub fn add_outcome(&mut self, confidence: f64, correct: bool) {
        self.count += 1;
        if correct {
            self.correct_count += 1;
        }
        self.confidence_sum += confidence;
        
        // Update averages
        self.avg_confidence = self.confidence_sum / self.count as f64;
        self.accuracy = self.correct_count as f64 / self.count as f64;
    }
    
    pub fn calibration_error(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.accuracy - self.avg_confidence).abs()
        }
    }
}

// ============================================================================
// PART 2: CALIBRATION TRACKER
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationTracker {
    /// Number of bins for calibration
    pub num_bins: usize,
    
    /// Calibration bins
    pub bins: Vec<CalibrationBin>,
    
    /// All outcomes: (confidence, correct)
    pub outcomes: Vec<(f64, bool)>,
    
    /// ECE history over time
    pub ece_history: Vec<f64>,
    
    /// MCE history over time
    pub mce_history: Vec<f64>,
    
    /// Brier score history
    pub brier_history: Vec<f64>,
    
    /// Domain-specific tracking
    pub domain_trackers: HashMap<String, DomainCalibrationTracker>,
}

impl CalibrationTracker {
    pub fn new(num_bins: usize) -> Self {
        let mut bins = Vec::new();
        for i in 0..num_bins {
            let lower = i as f64 / num_bins as f64;
            let upper = (i + 1) as f64 / num_bins as f64;
            bins.push(CalibrationBin::new(lower, upper));
        }
        
        Self {
            num_bins,
            bins,
            outcomes: Vec::new(),
            ece_history: Vec::new(),
            mce_history: Vec::new(),
            brier_history: Vec::new(),
            domain_trackers: HashMap::new(),
        }
    }
    
    /// Record an outcome
    pub fn record_outcome(&mut self, confidence: f64, correct: bool, domain: Option<&str>) {
        // Clamp confidence to [0, 1]
        let conf = confidence.max(0.0).min(1.0);
        
        // Add to outcomes
        self.outcomes.push((conf, correct));
        
        // Add to appropriate bin
        let bin_idx = ((conf * self.num_bins as f64).floor() as usize).min(self.num_bins - 1);
        self.bins[bin_idx].add_outcome(conf, correct);
        
        // Update domain tracker if provided
        if let Some(d) = domain {
            self.domain_trackers
                .entry(d.to_string())
                .or_insert_with(|| DomainCalibrationTracker::new(self.num_bins))
                .record_outcome(conf, correct);
        }
        
        // Recompute metrics periodically
        if self.outcomes.len() % 100 == 0 {
            self.recompute_metrics();
        }
    }
    
    /// Compute Expected Calibration Error (ECE)
    pub fn compute_ece(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        
        let total = self.outcomes.len() as f64;
        let mut ece = 0.0;
        
        for bin in &self.bins {
            if bin.count > 0 {
                let weight = bin.count as f64 / total;
                ece += weight * bin.calibration_error();
            }
        }
        
        ece
    }
    
    /// Compute Maximum Calibration Error (MCE)
    pub fn compute_mce(&self) -> f64 {
        self.bins.iter()
            .filter(|b| b.count > 0)
            .map(|b| b.calibration_error())
            .fold(0.0, f64::max)
    }
    
    /// Compute Brier Score
    pub fn compute_brier(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        
        self.outcomes.iter()
            .map(|(conf, correct)| {
                let y = if *correct { 1.0 } else { 0.0 };
                (conf - y).powi(2)
            })
            .sum::<f64>() / self.outcomes.len() as f64
    }
    
    /// Recompute all metrics and update history
    fn recompute_metrics(&mut self) {
        let ece = self.compute_ece();
        let mce = self.compute_mce();
        let brier = self.compute_brier();
        
        self.ece_history.push(ece);
        self.mce_history.push(mce);
        self.brier_history.push(brier);
    }
    
    /// Get calibration statistics
    pub fn get_stats(&self) -> CalibrationStats {
        CalibrationStats {
            total_outcomes: self.outcomes.len(),
            ece: self.compute_ece(),
            mce: self.compute_mce(),
            brier: self.compute_brier(),
            ece_history: self.ece_history.clone(),
            mce_history: self.mce_history.clone(),
            brier_history: self.brier_history.clone(),
            bins: self.bins.clone(),
        }
    }
    
    /// Get domain-specific statistics
    pub fn get_domain_stats(&self, domain: &str) -> Option<CalibrationStats> {
        self.domain_trackers.get(domain).map(|dt| dt.get_stats())
    }
    
    /// Get reliability diagram data
    pub fn get_reliability_diagram(&self) -> Vec<(f64, f64)> {
        self.bins.iter()
            .filter(|b| b.count > 0)
            .map(|b| (b.avg_confidence, b.accuracy))
            .collect()
    }
}

impl Default for CalibrationTracker {
    fn default() -> Self {
        Self::new(10)
    }
}

// ============================================================================
// PART 3: DOMAIN-SPECIFIC CALIBRATION
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainCalibrationTracker {
    pub num_bins: usize,
    pub bins: Vec<CalibrationBin>,
    pub outcomes: Vec<(f64, bool)>,
}

impl DomainCalibrationTracker {
    pub fn new(num_bins: usize) -> Self {
        let mut bins = Vec::new();
        for i in 0..num_bins {
            let lower = i as f64 / num_bins as f64;
            let upper = (i + 1) as f64 / num_bins as f64;
            bins.push(CalibrationBin::new(lower, upper));
        }
        
        Self {
            num_bins,
            bins,
            outcomes: Vec::new(),
        }
    }
    
    pub fn record_outcome(&mut self, confidence: f64, correct: bool) {
        let conf = confidence.max(0.0).min(1.0);
        self.outcomes.push((conf, correct));
        
        let bin_idx = ((conf * self.num_bins as f64).floor() as usize).min(self.num_bins - 1);
        self.bins[bin_idx].add_outcome(conf, correct);
    }
    
    pub fn compute_ece(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        
        let total = self.outcomes.len() as f64;
        let mut ece = 0.0;
        
        for bin in &self.bins {
            if bin.count > 0 {
                let weight = bin.count as f64 / total;
                ece += weight * bin.calibration_error();
            }
        }
        
        ece
    }
    
    pub fn compute_mce(&self) -> f64 {
        self.bins.iter()
            .filter(|b| b.count > 0)
            .map(|b| b.calibration_error())
            .fold(0.0, f64::max)
    }
    
    pub fn compute_brier(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        
        self.outcomes.iter()
            .map(|(conf, correct)| {
                let y = if *correct { 1.0 } else { 0.0 };
                (conf - y).powi(2)
            })
            .sum::<f64>() / self.outcomes.len() as f64
    }
    
    pub fn get_stats(&self) -> CalibrationStats {
        CalibrationStats {
            total_outcomes: self.outcomes.len(),
            ece: self.compute_ece(),
            mce: self.compute_mce(),
            brier: self.compute_brier(),
            ece_history: vec![],
            mce_history: vec![],
            brier_history: vec![],
            bins: self.bins.clone(),
        }
    }
}

// ============================================================================
// PART 4: CALIBRATION STATISTICS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationStats {
    pub total_outcomes: usize,
    pub ece: f64,
    pub mce: f64,
    pub brier: f64,
    pub ece_history: Vec<f64>,
    pub mce_history: Vec<f64>,
    pub brier_history: Vec<f64>,
    pub bins: Vec<CalibrationBin>,
}

impl CalibrationStats {
    pub fn is_well_calibrated(&self) -> bool {
        self.ece < 0.05 && self.mce < 0.10
    }
    
    pub fn calibration_quality(&self) -> &str {
        if self.ece < 0.02 {
            "Excellent"
        } else if self.ece < 0.05 {
            "Good"
        } else if self.ece < 0.10 {
            "Fair"
        } else {
            "Poor"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calibration_tracker() {
        let mut tracker = CalibrationTracker::new(10);
        
        // Perfect calibration: confidence matches accuracy
        for i in 0..100 {
            let conf = (i % 10) as f64 / 10.0 + 0.05;
            let correct = (i % 10) >= 5; // 50% accuracy for conf >= 0.5
            tracker.record_outcome(conf, correct, Some("test"));
        }
        
        let stats = tracker.get_stats();
        assert!(stats.total_outcomes == 100);
    }
    
    #[test]
    fn test_ece_computation() {
        let mut tracker = CalibrationTracker::new(10);
        
        // Add perfectly calibrated outcomes
        for _ in 0..10 {
            tracker.record_outcome(0.9, true, None);
        }
        for _ in 0..10 {
            tracker.record_outcome(0.1, false, None);
        }
        
        let ece = tracker.compute_ece();
        assert!(ece < 0.15); // Should be well-calibrated
    }
    
    #[test]
    fn test_brier_score() {
        let mut tracker = CalibrationTracker::new(10);
        
        // Perfect predictions
        tracker.record_outcome(1.0, true, None);
        tracker.record_outcome(0.0, false, None);
        
        let brier = tracker.compute_brier();
        assert!(brier < 0.01); // Should be near 0
    }
}
