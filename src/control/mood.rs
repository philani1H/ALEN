//! Mood System - Persistent Emotional State
//!
//! Mood is NOT emotion. Mood is:
//! - Slow-changing internal state
//! - Biases interpretation of inputs
//! - Changes reaction thresholds
//! - Emerges from feedback loops
//!
//! This is NOT metaphorical - it's functional.

use serde::{Deserialize, Serialize};
use super::emotions::{Emotion, EmotionalResponse};

/// Mood state - slow-changing background chemistry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodState {
    /// Reward level (dopamine baseline) [0, 1]
    pub reward_level: f64,
    /// Stress level (cortisol baseline) [0, 1]
    pub stress_level: f64,
    /// Trust level (oxytocin baseline) [0, 1]
    pub trust_level: f64,
    /// Curiosity level (novelty seeking) [0, 1]
    pub curiosity_level: f64,
    /// Energy level (general activation) [0, 1]
    pub energy_level: f64,
    /// Stability (resistance to mood change) [0, 1]
    pub stability: f64,
}

impl Default for MoodState {
    fn default() -> Self {
        Self {
            reward_level: 0.5,
            stress_level: 0.3,
            trust_level: 0.5,
            curiosity_level: 0.5,
            energy_level: 0.6,
            stability: 0.5,
        }
    }
}

impl MoodState {
    /// Classify current mood
    pub fn classify(&self) -> Mood {
        // Mood emerges from combination of levels
        match (self.reward_level, self.stress_level, self.energy_level) {
            (r, s, e) if r > 0.7 && s < 0.3 && e > 0.6 => Mood::Optimistic,
            (r, s, e) if r < 0.3 && s > 0.7 && e < 0.4 => Mood::Depressed,
            (r, s, e) if r < 0.4 && s > 0.6 && e > 0.5 => Mood::Anxious,
            (r, s, _) if r > 0.6 && s < 0.4 => Mood::Content,
            (r, s, e) if r < 0.4 && s < 0.4 && e < 0.4 => Mood::Apathetic,
            (_, s, e) if s > 0.7 && e > 0.7 => Mood::Stressed,
            (r, _, e) if r > 0.6 && e > 0.7 => Mood::Energized,
            _ => Mood::Neutral,
        }
    }

    /// Decay toward baseline (homeostasis)
    pub fn decay(&mut self, rate: f64) {
        let baseline_reward = 0.5;
        let baseline_stress = 0.3;
        let baseline_trust = 0.5;
        let baseline_curiosity = 0.5;
        let baseline_energy = 0.6;

        self.reward_level += (baseline_reward - self.reward_level) * rate;
        self.stress_level += (baseline_stress - self.stress_level) * rate;
        self.trust_level += (baseline_trust - self.trust_level) * rate;
        self.curiosity_level += (baseline_curiosity - self.curiosity_level) * rate;
        self.energy_level += (baseline_energy - self.energy_level) * rate;

        self.clamp();
    }

    /// Clamp all values to [0, 1]
    fn clamp(&mut self) {
        self.reward_level = self.reward_level.clamp(0.0, 1.0);
        self.stress_level = self.stress_level.clamp(0.0, 1.0);
        self.trust_level = self.trust_level.clamp(0.0, 1.0);
        self.curiosity_level = self.curiosity_level.clamp(0.0, 1.0);
        self.energy_level = self.energy_level.clamp(0.0, 1.0);
    }

    /// Get mood bias for perception
    /// This is how mood filters interpretation
    pub fn perception_bias(&self) -> f64 {
        // Positive mood → positive bias
        // Negative mood → negative bias
        let positive = self.reward_level + self.trust_level;
        let negative = self.stress_level;
        (positive - negative) / 2.0
    }

    /// Get reaction threshold modifier
    /// High stress → lower threshold (more reactive)
    /// High stability → higher threshold (less reactive)
    pub fn reaction_threshold(&self) -> f64 {
        0.5 - (self.stress_level * 0.3) + (self.stability * 0.2)
    }
}

/// Discrete mood categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mood {
    Optimistic,   // High reward, low stress
    Content,      // Balanced positive
    Neutral,      // Baseline
    Anxious,      // High stress, moderate energy
    Stressed,     // High stress, high energy
    Depressed,    // Low reward, high stress, low energy
    Apathetic,    // Low everything
    Energized,    // High reward, high energy
}

impl Mood {
    pub fn as_str(&self) -> &str {
        match self {
            Mood::Optimistic => "Optimistic",
            Mood::Content => "Content",
            Mood::Neutral => "Neutral",
            Mood::Anxious => "Anxious",
            Mood::Stressed => "Stressed",
            Mood::Depressed => "Depressed",
            Mood::Apathetic => "Apathetic",
            Mood::Energized => "Energized",
        }
    }
}

/// Mood engine - manages persistent emotional state
pub struct MoodEngine {
    /// Current mood state
    state: MoodState,
    /// Decay rate (how fast mood returns to baseline)
    decay_rate: f64,
    /// Mood history for tracking
    history: Vec<(Mood, f64)>, // (mood, timestamp)
    /// Accumulation factor (how much emotions affect mood)
    accumulation_factor: f64,
}

impl MoodEngine {
    pub fn new() -> Self {
        Self {
            state: MoodState::default(),
            decay_rate: 0.01, // Slow decay (moods persist)
            history: Vec::new(),
            accumulation_factor: 0.05, // Emotions slowly accumulate into mood
        }
    }

    /// Update mood based on emotional response
    /// This is the key: emotions → mood accumulation
    pub fn update_from_emotion(&mut self, emotion_response: &EmotionalResponse) {
        let factor = self.accumulation_factor;

        match emotion_response.emotion {
            Emotion::Joy | Emotion::Excitement => {
                self.state.reward_level += factor * emotion_response.intensity;
                self.state.energy_level += factor * emotion_response.intensity * 0.5;
                self.state.stress_level -= factor * emotion_response.intensity * 0.3;
            }
            Emotion::Sadness => {
                self.state.reward_level -= factor * emotion_response.intensity;
                self.state.energy_level -= factor * emotion_response.intensity * 0.5;
                self.state.stress_level += factor * emotion_response.intensity * 0.2;
            }
            Emotion::Fear => {
                self.state.stress_level += factor * emotion_response.intensity;
                self.state.trust_level -= factor * emotion_response.intensity * 0.3;
                self.state.energy_level += factor * emotion_response.intensity * 0.3;
            }
            Emotion::Anger => {
                self.state.stress_level += factor * emotion_response.intensity * 0.7;
                self.state.energy_level += factor * emotion_response.intensity * 0.5;
            }
            Emotion::Curiosity => {
                self.state.curiosity_level += factor * emotion_response.intensity;
                self.state.energy_level += factor * emotion_response.intensity * 0.3;
            }
            Emotion::Contentment => {
                self.state.reward_level += factor * emotion_response.intensity * 0.5;
                self.state.stress_level -= factor * emotion_response.intensity * 0.5;
            }
            _ => {}
        }

        self.state.clamp();
    }

    /// Decay mood toward baseline (homeostasis)
    pub fn decay(&mut self) {
        self.state.decay(self.decay_rate);
    }

    /// Get current mood
    pub fn current_mood(&self) -> Mood {
        self.state.classify()
    }

    /// Get mood state
    pub fn get_state(&self) -> &MoodState {
        &self.state
    }

    /// Set mood state directly (for external adjustment)
    pub fn set_state(&mut self, state: MoodState) {
        self.state = state;
    }

    /// Record mood in history
    pub fn record(&mut self) {
        let mood = self.current_mood();
        let timestamp = Self::timestamp();
        self.history.push((mood, timestamp));

        // Keep only recent history
        if self.history.len() > 1000 {
            self.history.drain(0..500);
        }
    }

    /// Get mood trend (recent moods)
    pub fn mood_trend(&self) -> Vec<Mood> {
        self.history.iter()
            .rev()
            .take(10)
            .map(|(m, _)| *m)
            .collect()
    }

    /// Check if mood is stable
    pub fn is_stable(&self) -> bool {
        if self.history.len() < 5 {
            return true;
        }

        let recent: Vec<Mood> = self.history.iter()
            .rev()
            .take(5)
            .map(|(m, _)| *m)
            .collect();

        // Stable if all recent moods are the same
        recent.windows(2).all(|w| w[0] == w[1])
    }

    /// Get mood statistics
    pub fn statistics(&self) -> MoodStatistics {
        let current = self.current_mood();
        let trend = self.mood_trend();
        
        let mut mood_counts = std::collections::HashMap::new();
        for mood in &trend {
            *mood_counts.entry(*mood).or_insert(0) += 1;
        }

        MoodStatistics {
            current_mood: current,
            reward_level: self.state.reward_level,
            stress_level: self.state.stress_level,
            trust_level: self.state.trust_level,
            curiosity_level: self.state.curiosity_level,
            energy_level: self.state.energy_level,
            perception_bias: self.state.perception_bias(),
            reaction_threshold: self.state.reaction_threshold(),
            is_stable: self.is_stable(),
            history_length: self.history.len(),
        }
    }

    fn timestamp() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }
}

impl Default for MoodEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Mood statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodStatistics {
    pub current_mood: Mood,
    pub reward_level: f64,
    pub stress_level: f64,
    pub trust_level: f64,
    pub curiosity_level: f64,
    pub energy_level: f64,
    pub perception_bias: f64,
    pub reaction_threshold: f64,
    pub is_stable: bool,
    pub history_length: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::emotions::{EmotionalResponse, Emotion, Neurotransmitters};

    #[test]
    fn test_mood_classification() {
        let state = MoodState {
            reward_level: 0.8,
            stress_level: 0.2,
            trust_level: 0.7,
            curiosity_level: 0.6,
            energy_level: 0.7,
            stability: 0.5,
        };
        assert_eq!(state.classify(), Mood::Optimistic);
    }

    #[test]
    fn test_mood_decay() {
        let mut state = MoodState::default();
        state.reward_level = 1.0;
        state.decay(0.5);
        assert!(state.reward_level < 1.0);
        assert!(state.reward_level > 0.5);
    }

    #[test]
    fn test_mood_engine() {
        let mut engine = MoodEngine::new();
        
        // Simulate joy
        let response = EmotionalResponse {
            emotion: Emotion::Joy,
            valence: 0.8,
            arousal: 0.7,
            intensity: 0.9,
            neurotransmitters: Neurotransmitters::default(),
        };
        
        engine.update_from_emotion(&response);
        assert!(engine.get_state().reward_level > 0.5);
    }

    #[test]
    fn test_mood_accumulation() {
        let mut engine = MoodEngine::new();
        
        // Multiple positive emotions should improve mood
        for _ in 0..10 {
            let response = EmotionalResponse {
                emotion: Emotion::Joy,
                valence: 0.8,
                arousal: 0.7,
                intensity: 0.8,
                neurotransmitters: Neurotransmitters::default(),
            };
            engine.update_from_emotion(&response);
        }
        
        let mood = engine.current_mood();
        assert!(mood == Mood::Optimistic || mood == Mood::Content || mood == Mood::Energized);
    }
}
