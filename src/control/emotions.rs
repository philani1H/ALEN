//! Biologically-Inspired Emotion System
//!
//! Models emotional processing based on human neuroscience:
//! - Limbic system (amygdala, hippocampus, hypothalamus)
//! - Neurotransmitter dynamics (dopamine, serotonin, etc.)
//! - Prefrontal cortex evaluation
//! - Feedback loops between sensation, chemistry, and behavior
//!
//! Emotions are NOT hardcoded responses - they emerge from
//! network activation patterns, just like in biological brains.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Neurotransmitter levels (0.0 - 1.0)
/// These modulate emotional intensity and cognitive function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neurotransmitters {
    /// Dopamine: reward, motivation, pleasure
    pub dopamine: f64,
    /// Serotonin: mood stability, well-being
    pub serotonin: f64,
    /// Norepinephrine: alertness, stress response
    pub norepinephrine: f64,
    /// Oxytocin: bonding, trust
    pub oxytocin: f64,
    /// Cortisol: stress hormone
    pub cortisol: f64,
    /// GABA: inhibitory, calming
    pub gaba: f64,
    /// Glutamate: excitatory, learning
    pub glutamate: f64,
}

impl Default for Neurotransmitters {
    fn default() -> Self {
        Self {
            dopamine: 0.5,
            serotonin: 0.5,
            norepinephrine: 0.3,
            oxytocin: 0.5,
            cortisol: 0.2,
            gaba: 0.5,
            glutamate: 0.5,
        }
    }
}

impl Neurotransmitters {
    /// Decay neurotransmitters over time (homeostasis)
    pub fn decay(&mut self, rate: f64) {
        let target = 0.5; // Homeostatic baseline
        self.dopamine += (target - self.dopamine) * rate;
        self.serotonin += (target - self.serotonin) * rate;
        self.norepinephrine += (target - self.norepinephrine) * rate;
        self.oxytocin += (target - self.oxytocin) * rate;
        self.cortisol += (0.2 - self.cortisol) * rate; // Lower baseline for stress
        self.gaba += (target - self.gaba) * rate;
        self.glutamate += (target - self.glutamate) * rate;
    }

    /// Clamp all values to [0, 1]
    pub fn clamp(&mut self) {
        self.dopamine = self.dopamine.clamp(0.0, 1.0);
        self.serotonin = self.serotonin.clamp(0.0, 1.0);
        self.norepinephrine = self.norepinephrine.clamp(0.0, 1.0);
        self.oxytocin = self.oxytocin.clamp(0.0, 1.0);
        self.cortisol = self.cortisol.clamp(0.0, 1.0);
        self.gaba = self.gaba.clamp(0.0, 1.0);
        self.glutamate = self.glutamate.clamp(0.0, 1.0);
    }
}

/// Emotional valence (positive/negative) and arousal (calm/excited)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EmotionalState {
    /// Valence: -1.0 (negative) to +1.0 (positive)
    pub valence: f64,
    /// Arousal: 0.0 (calm) to 1.0 (excited)
    pub arousal: f64,
    /// Dominance: 0.0 (submissive) to 1.0 (dominant)
    pub dominance: f64,
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.3,
            dominance: 0.5,
        }
    }
}

impl EmotionalState {
    /// Classify into discrete emotion
    pub fn classify(&self) -> Emotion {
        match (self.valence, self.arousal) {
            (v, a) if v > 0.5 && a > 0.6 => Emotion::Joy,
            (v, a) if v > 0.3 && a < 0.4 => Emotion::Contentment,
            (v, a) if v < -0.5 && a > 0.6 => Emotion::Fear,
            (v, a) if v < -0.5 && a < 0.4 => Emotion::Sadness,
            (v, a) if v < -0.3 && a > 0.5 => Emotion::Anger,
            (v, a) if v > 0.2 && a > 0.5 => Emotion::Excitement,
            (v, _) if v.abs() < 0.2 => Emotion::Neutral,
            _ => Emotion::Neutral,
        }
    }
}

/// Discrete emotion categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Emotion {
    Joy,
    Sadness,
    Fear,
    Anger,
    Surprise,
    Disgust,
    Contentment,
    Excitement,
    Curiosity,
    Neutral,
}

impl Emotion {
    pub fn as_str(&self) -> &str {
        match self {
            Emotion::Joy => "Joy",
            Emotion::Sadness => "Sadness",
            Emotion::Fear => "Fear",
            Emotion::Anger => "Anger",
            Emotion::Surprise => "Surprise",
            Emotion::Disgust => "Disgust",
            Emotion::Contentment => "Contentment",
            Emotion::Excitement => "Excitement",
            Emotion::Curiosity => "Curiosity",
            Emotion::Neutral => "Neutral",
        }
    }
}

/// Stimulus that triggers emotional response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalStimulus {
    /// Type of stimulus
    pub stimulus_type: StimulusType,
    /// Intensity (0.0 - 1.0)
    pub intensity: f64,
    /// Valence (positive/negative)
    pub valence: f64,
    /// Context information
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StimulusType {
    Success,      // Task completed successfully
    Failure,      // Task failed
    Surprise,     // Unexpected outcome
    Threat,       // Potential danger/error
    Reward,       // Positive feedback
    Punishment,   // Negative feedback
    Novel,        // New information
    Familiar,     // Known pattern
}

/// Limbic System - emotional processing center
/// Models: amygdala, hippocampus, hypothalamus
#[derive(Debug, Clone)]
pub struct LimbicSystem {
    /// Current neurotransmitter levels
    neurotransmitters: Neurotransmitters,
    /// Current emotional state
    emotional_state: EmotionalState,
    /// Emotional memory (stimulus -> response associations)
    emotional_memory: HashMap<String, EmotionalState>,
    /// Decay rate for neurotransmitters
    decay_rate: f64,
}

impl LimbicSystem {
    pub fn new() -> Self {
        Self {
            neurotransmitters: Neurotransmitters::default(),
            emotional_state: EmotionalState::default(),
            emotional_memory: HashMap::new(),
            decay_rate: 0.1,
        }
    }

    /// Process stimulus through limbic system
    /// This is the "amygdala response" - fast, automatic
    pub fn process_stimulus(&mut self, stimulus: &EmotionalStimulus) -> EmotionalResponse {
        // 1. Amygdala: Detect emotional salience
        let salience = self.compute_salience(stimulus);
        
        // 2. Hippocampus: Check emotional memory
        let memory_influence = self.recall_emotional_memory(&stimulus.context);
        
        // 3. Hypothalamus: Trigger neurotransmitter release
        self.release_neurotransmitters(stimulus);
        
        // 4. Update emotional state based on chemistry
        self.update_emotional_state(stimulus, salience, memory_influence);
        
        // 5. Store in emotional memory
        self.store_emotional_memory(&stimulus.context, self.emotional_state);
        
        EmotionalResponse {
            emotion: self.emotional_state.classify(),
            valence: self.emotional_state.valence,
            arousal: self.emotional_state.arousal,
            intensity: salience * stimulus.intensity,
            neurotransmitters: self.neurotransmitters.clone(),
        }
    }

    /// Compute emotional salience (how important is this stimulus?)
    fn compute_salience(&self, stimulus: &EmotionalStimulus) -> f64 {
        match stimulus.stimulus_type {
            StimulusType::Threat => 0.9,      // High salience
            StimulusType::Reward => 0.8,
            StimulusType::Surprise => 0.7,
            StimulusType::Success => 0.6,
            StimulusType::Failure => 0.7,
            StimulusType::Novel => 0.6,
            StimulusType::Familiar => 0.3,
            StimulusType::Punishment => 0.8,
        }
    }

    /// Recall emotional memory (hippocampus function)
    fn recall_emotional_memory(&self, context: &str) -> f64 {
        self.emotional_memory
            .get(context)
            .map(|state| state.valence)
            .unwrap_or(0.0)
    }

    /// Release neurotransmitters based on stimulus (hypothalamus function)
    fn release_neurotransmitters(&mut self, stimulus: &EmotionalStimulus) {
        match stimulus.stimulus_type {
            StimulusType::Success | StimulusType::Reward => {
                self.neurotransmitters.dopamine += 0.3 * stimulus.intensity;
                self.neurotransmitters.serotonin += 0.2 * stimulus.intensity;
            }
            StimulusType::Failure | StimulusType::Punishment => {
                self.neurotransmitters.dopamine -= 0.2 * stimulus.intensity;
                self.neurotransmitters.cortisol += 0.3 * stimulus.intensity;
            }
            StimulusType::Threat => {
                self.neurotransmitters.norepinephrine += 0.4 * stimulus.intensity;
                self.neurotransmitters.cortisol += 0.4 * stimulus.intensity;
                self.neurotransmitters.dopamine -= 0.1 * stimulus.intensity;
            }
            StimulusType::Surprise => {
                self.neurotransmitters.norepinephrine += 0.3 * stimulus.intensity;
                self.neurotransmitters.glutamate += 0.2 * stimulus.intensity;
            }
            StimulusType::Novel => {
                self.neurotransmitters.dopamine += 0.2 * stimulus.intensity;
                self.neurotransmitters.glutamate += 0.3 * stimulus.intensity;
            }
            StimulusType::Familiar => {
                self.neurotransmitters.gaba += 0.1 * stimulus.intensity;
            }
        }
        
        self.neurotransmitters.clamp();
    }

    /// Update emotional state based on neurotransmitter levels
    fn update_emotional_state(&mut self, stimulus: &EmotionalStimulus, salience: f64, memory: f64) {
        // Valence influenced by dopamine, serotonin, cortisol
        let valence_delta = 
            self.neurotransmitters.dopamine * 0.4 +
            self.neurotransmitters.serotonin * 0.3 -
            self.neurotransmitters.cortisol * 0.3 +
            stimulus.valence * 0.5 +
            memory * 0.2;
        
        // Arousal influenced by norepinephrine, glutamate
        let arousal_delta = 
            self.neurotransmitters.norepinephrine * 0.5 +
            self.neurotransmitters.glutamate * 0.3 -
            self.neurotransmitters.gaba * 0.2 +
            salience * 0.3;
        
        // Update with momentum
        self.emotional_state.valence = 
            0.7 * self.emotional_state.valence + 0.3 * valence_delta.clamp(-1.0, 1.0);
        self.emotional_state.arousal = 
            0.7 * self.emotional_state.arousal + 0.3 * arousal_delta.clamp(0.0, 1.0);
    }

    /// Store emotional association in memory
    fn store_emotional_memory(&mut self, context: &str, state: EmotionalState) {
        self.emotional_memory.insert(context.to_string(), state);
        
        // Limit memory size
        if self.emotional_memory.len() > 1000 {
            // Remove oldest entries (simplified)
            let keys: Vec<String> = self.emotional_memory.keys().take(100).cloned().collect();
            for key in keys {
                self.emotional_memory.remove(&key);
            }
        }
    }

    /// Decay neurotransmitters (homeostasis)
    pub fn decay(&mut self) {
        self.neurotransmitters.decay(self.decay_rate);
    }

    /// Get current emotional state
    pub fn get_state(&self) -> EmotionalState {
        self.emotional_state
    }

    /// Get current emotion
    pub fn get_emotion(&self) -> Emotion {
        self.emotional_state.classify()
    }
}

/// Prefrontal cortex - rational evaluation of emotions
#[derive(Debug, Clone)]
pub struct PrefrontalCortex {
    /// Emotional regulation strength
    regulation_strength: f64,
    /// Cognitive reappraisal ability
    reappraisal_ability: f64,
}

impl PrefrontalCortex {
    pub fn new() -> Self {
        Self {
            regulation_strength: 0.5,
            reappraisal_ability: 0.5,
        }
    }

    /// Evaluate and potentially regulate emotional response
    /// This is "thinking yourself out of" an emotion
    pub fn evaluate(&self, response: &EmotionalResponse, context: &str) -> RegulatedResponse {
        // Rational evaluation
        let should_regulate = self.should_regulate(response);
        
        if should_regulate {
            // Apply cognitive reappraisal
            let regulated_valence = response.valence * (1.0 - self.regulation_strength * 0.5);
            let regulated_arousal = response.arousal * (1.0 - self.regulation_strength * 0.3);
            
            RegulatedResponse {
                original_emotion: response.emotion,
                regulated_emotion: EmotionalState {
                    valence: regulated_valence,
                    arousal: regulated_arousal,
                    dominance: 0.6, // Increased sense of control
                }.classify(),
                regulation_applied: true,
                regulation_strength: self.regulation_strength,
                rationale: format!("Regulated {} response in context: {}", 
                    response.emotion.as_str(), context),
            }
        } else {
            RegulatedResponse {
                original_emotion: response.emotion,
                regulated_emotion: response.emotion,
                regulation_applied: false,
                regulation_strength: 0.0,
                rationale: "No regulation needed".to_string(),
            }
        }
    }

    /// Decide if emotion should be regulated
    fn should_regulate(&self, response: &EmotionalResponse) -> bool {
        // Regulate if arousal is very high or valence is very negative
        response.arousal > 0.8 || response.valence < -0.7
    }
}

/// Emotional response from limbic system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalResponse {
    pub emotion: Emotion,
    pub valence: f64,
    pub arousal: f64,
    pub intensity: f64,
    pub neurotransmitters: Neurotransmitters,
}

/// Regulated response from prefrontal cortex
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatedResponse {
    pub original_emotion: Emotion,
    pub regulated_emotion: Emotion,
    pub regulation_applied: bool,
    pub regulation_strength: f64,
    pub rationale: String,
}

/// Complete emotion system
pub struct EmotionSystem {
    limbic: LimbicSystem,
    prefrontal: PrefrontalCortex,
    /// Emotion history
    history: Vec<(Emotion, f64)>, // (emotion, timestamp)
}

impl EmotionSystem {
    pub fn new() -> Self {
        Self {
            limbic: LimbicSystem::new(),
            prefrontal: PrefrontalCortex::new(),
            history: Vec::new(),
        }
    }

    /// Process stimulus through complete emotional system
    pub fn process(&mut self, stimulus: EmotionalStimulus) -> RegulatedResponse {
        // 1. Limbic system: automatic emotional response
        let limbic_response = self.limbic.process_stimulus(&stimulus);
        
        // 2. Prefrontal cortex: rational evaluation
        let regulated = self.prefrontal.evaluate(&limbic_response, &stimulus.context);
        
        // 3. Record in history
        self.history.push((regulated.regulated_emotion, Self::timestamp()));
        
        // 4. Decay neurotransmitters
        self.limbic.decay();
        
        regulated
    }

    /// Get current emotional state
    pub fn current_emotion(&self) -> Emotion {
        self.limbic.get_emotion()
    }

    /// Get emotional trend (recent emotions)
    pub fn emotional_trend(&self) -> Vec<Emotion> {
        self.history.iter()
            .rev()
            .take(10)
            .map(|(e, _)| *e)
            .collect()
    }

    fn timestamp() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }
}

impl Default for EmotionSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neurotransmitter_decay() {
        let mut nt = Neurotransmitters::default();
        nt.dopamine = 1.0;
        nt.decay(0.5);
        assert!(nt.dopamine < 1.0);
        assert!(nt.dopamine > 0.5);
    }

    #[test]
    fn test_emotional_state_classification() {
        let state = EmotionalState {
            valence: 0.8,
            arousal: 0.8,
            dominance: 0.5,
        };
        assert_eq!(state.classify(), Emotion::Joy);
    }

    #[test]
    fn test_limbic_system() {
        let mut limbic = LimbicSystem::new();
        
        let stimulus = EmotionalStimulus {
            stimulus_type: StimulusType::Success,
            intensity: 0.8,
            valence: 0.7,
            context: "test_success".to_string(),
        };
        
        let response = limbic.process_stimulus(&stimulus);
        assert!(response.valence > 0.0);
        assert!(response.neurotransmitters.dopamine > 0.5);
    }

    #[test]
    fn test_emotion_system() {
        let mut system = EmotionSystem::new();
        
        let stimulus = EmotionalStimulus {
            stimulus_type: StimulusType::Reward,
            intensity: 0.9,
            valence: 0.8,
            context: "task_completed".to_string(),
        };
        
        let response = system.process(stimulus);
        assert!(response.original_emotion == Emotion::Joy || 
                response.original_emotion == Emotion::Excitement);
    }
}
