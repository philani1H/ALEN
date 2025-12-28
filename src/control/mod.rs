//! Control Module
//!
//! Contains:
//! - Emotion/bias control (modulates reasoning)
//! - Meta tracking (confidence, uncertainty)
//! - Curiosity and self-supervised learning
//! - Biologically-inspired emotion system
//! - Mood system (persistent emotional state)

pub mod curiosity;
pub mod emotions;
pub mod mood;

pub use curiosity::{
    CuriosityEngine, Prediction, Observation, Surprise,
    SelfSupervisedLoop, CuriosityStats,
};
pub use emotions::{
    EmotionSystem, Emotion, EmotionalStimulus, StimulusType,
    EmotionalResponse, RegulatedResponse, Neurotransmitters,
    EmotionalState, LimbicSystem, PrefrontalCortex,
};
pub use mood::{
    MoodEngine, MoodState, Mood, MoodStatistics,
};

use crate::core::{ThoughtState, BiasVector};
use serde::{Deserialize, Serialize};

/// Meta-cognitive state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaState {
    /// Overall system confidence (0.0 - 1.0)
    pub confidence: f64,
    /// Uncertainty level (0.0 - 1.0)
    pub uncertainty: f64,
    /// Number of reasoning cycles performed
    pub reasoning_cycles: u64,
    /// Current attention focus
    pub attention_focus: Option<String>,
    /// Cognitive load estimate
    pub cognitive_load: f64,
    /// Historical confidence values
    pub confidence_history: Vec<f64>,
}

impl Default for MetaState {
    fn default() -> Self {
        Self {
            confidence: 0.5,
            uncertainty: 0.5,
            reasoning_cycles: 0,
            attention_focus: None,
            cognitive_load: 0.0,
            confidence_history: Vec::new(),
        }
    }
}

impl MetaState {
    /// Update confidence based on new evaluation
    pub fn update_confidence(&mut self, new_confidence: f64) {
        // Exponential moving average
        self.confidence = 0.7 * self.confidence + 0.3 * new_confidence;
        self.uncertainty = 1.0 - self.confidence;
        
        // Track history (keep last 100)
        self.confidence_history.push(new_confidence);
        if self.confidence_history.len() > 100 {
            self.confidence_history.remove(0);
        }
    }

    /// Increment reasoning cycles
    pub fn increment_cycles(&mut self) {
        self.reasoning_cycles += 1;
        // Cognitive load increases with cycles
        self.cognitive_load = (self.reasoning_cycles as f64 / 100.0).min(1.0);
    }

    /// Set attention focus
    pub fn set_focus(&mut self, focus: &str) {
        self.attention_focus = Some(focus.to_string());
    }

    /// Clear attention focus
    pub fn clear_focus(&mut self) {
        self.attention_focus = None;
    }

    /// Get confidence trend (positive = improving, negative = declining)
    pub fn confidence_trend(&self) -> f64 {
        if self.confidence_history.len() < 2 {
            return 0.0;
        }

        let recent: f64 = self.confidence_history.iter()
            .rev()
            .take(10)
            .sum::<f64>() / 10.0;
        
        let older: f64 = self.confidence_history.iter()
            .rev()
            .skip(10)
            .take(10)
            .sum::<f64>() / 10.0;

        recent - older
    }

    /// Should we switch strategy based on meta state?
    pub fn should_switch_strategy(&self) -> bool {
        // Switch if confidence is low and declining
        self.confidence < 0.3 && self.confidence_trend() < -0.1
    }

    /// Get recommended exploration level based on state
    pub fn recommended_exploration(&self) -> f64 {
        // High uncertainty -> more exploration
        // High confidence -> more exploitation
        self.uncertainty * 0.8 + 0.1
    }
}

/// Emotion/bias controller
#[derive(Debug, Clone)]
pub struct BiasController {
    /// Current bias vector
    pub current_bias: BiasVector,
    /// Default bias
    pub default_bias: BiasVector,
    /// Meta state
    pub meta: MetaState,
}

impl Default for BiasController {
    fn default() -> Self {
        Self {
            current_bias: BiasVector::default(),
            default_bias: BiasVector::default(),
            meta: MetaState::default(),
        }
    }
}

impl BiasController {
    /// Create a new bias controller
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom default bias
    pub fn with_default(bias: BiasVector) -> Self {
        Self {
            current_bias: bias.clone(),
            default_bias: bias,
            meta: MetaState::default(),
        }
    }

    /// Set risk tolerance
    pub fn set_risk_tolerance(&mut self, value: f64) {
        self.current_bias.risk_tolerance = value.clamp(0.0, 1.0);
    }

    /// Set exploration level
    pub fn set_exploration(&mut self, value: f64) {
        self.current_bias.exploration = value.clamp(0.0, 1.0);
    }

    /// Set urgency
    pub fn set_urgency(&mut self, value: f64) {
        self.current_bias.urgency = value.clamp(0.0, 1.0);
    }

    /// Set creativity
    pub fn set_creativity(&mut self, value: f64) {
        self.current_bias.creativity = value.clamp(0.0, 1.0);
    }

    /// Reset to default bias
    pub fn reset(&mut self) {
        self.current_bias = self.default_bias.clone();
    }

    /// Modulate a thought with current bias
    pub fn modulate(&self, thought: &ThoughtState) -> ThoughtState {
        self.current_bias.modulate(thought)
    }

    /// Auto-adjust bias based on meta state
    pub fn auto_adjust(&mut self) {
        // Increase exploration when uncertain
        if self.meta.uncertainty > 0.7 {
            self.current_bias.exploration = (self.current_bias.exploration + 0.1).min(1.0);
        }

        // Decrease risk when cognitive load is high
        if self.meta.cognitive_load > 0.7 {
            self.current_bias.risk_tolerance = (self.current_bias.risk_tolerance - 0.1).max(0.0);
        }

        // Increase urgency if many cycles without progress
        if self.meta.confidence_trend() < -0.05 && self.meta.reasoning_cycles > 10 {
            self.current_bias.urgency = (self.current_bias.urgency + 0.05).min(1.0);
        }
    }

    /// Update meta state with new evaluation
    pub fn update_meta(&mut self, confidence: f64) {
        self.meta.update_confidence(confidence);
        self.meta.increment_cycles();
    }

    /// Get current state summary
    pub fn state_summary(&self) -> ControlStateSummary {
        ControlStateSummary {
            bias: self.current_bias.clone(),
            confidence: self.meta.confidence,
            uncertainty: self.meta.uncertainty,
            cognitive_load: self.meta.cognitive_load,
            reasoning_cycles: self.meta.reasoning_cycles,
            confidence_trend: self.meta.confidence_trend(),
            recommended_exploration: self.meta.recommended_exploration(),
        }
    }
}

/// Summary of control state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlStateSummary {
    pub bias: BiasVector,
    pub confidence: f64,
    pub uncertainty: f64,
    pub cognitive_load: f64,
    pub reasoning_cycles: u64,
    pub confidence_trend: f64,
    pub recommended_exploration: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_state_update() {
        let mut meta = MetaState::default();
        
        meta.update_confidence(0.8);
        assert!(meta.confidence > 0.5);
        assert!(meta.uncertainty < 0.5);
    }

    #[test]
    fn test_bias_controller() {
        let mut controller = BiasController::new();
        
        controller.set_risk_tolerance(0.8);
        assert!((controller.current_bias.risk_tolerance - 0.8).abs() < 0.01);
        
        controller.reset();
        assert!((controller.current_bias.risk_tolerance - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_thought_modulation() {
        let controller = BiasController::new();
        let thought = ThoughtState::from_input("test", 64);
        
        let modulated = controller.modulate(&thought);
        assert_eq!(modulated.dimension, thought.dimension);
    }
}
