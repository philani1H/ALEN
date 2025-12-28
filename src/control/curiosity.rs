//! Curiosity and Self-Supervised Learning
//!
//! Implements:
//! - Self-prediction loops
//! - Surprise-based learning
//! - Internal question generation
//! - Free energy minimization

use serde::{Deserialize, Serialize};
use crate::core::ThoughtState;
use rand::Rng;

/// Prediction made by the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// What was predicted
    pub predicted_state: Vec<f64>,
    /// Confidence in prediction
    pub confidence: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Observation (actual outcome)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// What was actually observed
    pub observed_state: Vec<f64>,
    /// Timestamp
    pub timestamp: u64,
}

/// Surprise signal (prediction error)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Surprise {
    /// Magnitude of surprise
    pub magnitude: f64,
    /// Direction of error
    pub error_vector: Vec<f64>,
    /// Whether this was expected
    pub expected: bool,
}

impl Surprise {
    /// Compute surprise from prediction and observation
    pub fn compute(prediction: &Prediction, observation: &Observation) -> Self {
        let error_vector: Vec<f64> = prediction.predicted_state.iter()
            .zip(observation.observed_state.iter())
            .map(|(p, o)| o - p)
            .collect();

        let magnitude: f64 = error_vector.iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        // Expected if magnitude is small
        let expected = magnitude < 0.1;

        Self {
            magnitude,
            error_vector,
            expected,
        }
    }

    /// Is this a significant surprise?
    pub fn is_significant(&self) -> bool {
        self.magnitude > 0.5
    }
}

/// Curiosity-driven learning engine
pub struct CuriosityEngine {
    /// Dimension of thought space
    dimension: usize,
    /// Recent predictions
    predictions: Vec<Prediction>,
    /// Recent observations
    observations: Vec<Observation>,
    /// Surprise history
    surprises: Vec<Surprise>,
    /// Curiosity threshold
    curiosity_threshold: f64,
    /// Learning rate for surprise
    surprise_learning_rate: f64,
}

impl CuriosityEngine {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            predictions: Vec::new(),
            observations: Vec::new(),
            surprises: Vec::new(),
            curiosity_threshold: 0.3,
            surprise_learning_rate: 0.01,
        }
    }

    /// Generate a self-directed question
    /// This is how ALEN explores without external input
    pub fn generate_question(&self) -> String {
        let templates = vec![
            "What happens if I combine concept A with concept B?",
            "Can I predict the next state from this pattern?",
            "What is the relationship between X and Y?",
            "How does this constraint affect the outcome?",
            "What would happen if I negate this assumption?",
            "Can I find a simpler explanation?",
            "What are the edge cases?",
            "Is there a pattern I'm missing?",
        ];

        let mut rng = rand::thread_rng();
        templates[rng.gen_range(0..templates.len())].to_string()
    }

    /// Make a prediction about next state
    pub fn predict(&mut self, current_state: &ThoughtState) -> Prediction {
        // Simple prediction: assume state will remain stable
        // In full implementation, would use learned dynamics model
        let predicted_state = current_state.vector.clone();
        
        let confidence = 0.7; // Base confidence
        
        let prediction = Prediction {
            predicted_state,
            confidence,
            timestamp: Self::current_timestamp(),
        };

        self.predictions.push(prediction.clone());
        prediction
    }

    /// Record an observation
    pub fn observe(&mut self, state: &ThoughtState) -> Observation {
        let observation = Observation {
            observed_state: state.vector.clone(),
            timestamp: Self::current_timestamp(),
        };

        self.observations.push(observation.clone());
        observation
    }

    /// Compute surprise and update learning
    pub fn compute_surprise(&mut self, prediction: &Prediction, observation: &Observation) -> Surprise {
        let surprise = Surprise::compute(prediction, observation);
        
        self.surprises.push(surprise.clone());
        
        // Keep only recent history
        if self.surprises.len() > 1000 {
            self.surprises.drain(0..500);
        }

        surprise
    }

    /// Should we explore more? (high surprise = high curiosity)
    pub fn should_explore(&self) -> bool {
        if self.surprises.is_empty() {
            return true; // Always explore if no history
        }

        // Average recent surprise
        let recent_surprise: f64 = self.surprises.iter()
            .rev()
            .take(10)
            .map(|s| s.magnitude)
            .sum::<f64>() / 10.0_f64.min(self.surprises.len() as f64);

        recent_surprise > self.curiosity_threshold
    }

    /// Get curiosity score (0-1, higher = more curious)
    pub fn curiosity_score(&self) -> f64 {
        if self.surprises.is_empty() {
            return 1.0;
        }

        let recent_surprise: f64 = self.surprises.iter()
            .rev()
            .take(10)
            .map(|s| s.magnitude)
            .sum::<f64>() / 10.0_f64.min(self.surprises.len() as f64);

        (recent_surprise / 2.0_f64).min(1.0_f64)
    }

    /// Compute free energy (prediction error + complexity)
    pub fn free_energy(&self, prediction: &Prediction, observation: &Observation) -> f64 {
        let surprise = Surprise::compute(prediction, observation);
        
        // Free energy = prediction error + complexity penalty
        let prediction_error = surprise.magnitude;
        let complexity = self.estimate_complexity(&prediction.predicted_state);
        
        prediction_error + 0.1 * complexity
    }

    fn estimate_complexity(&self, state: &[f64]) -> f64 {
        // Simple complexity: variance of state
        let mean = state.iter().sum::<f64>() / state.len() as f64;
        let variance = state.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / state.len() as f64;
        variance
    }

    /// Get statistics
    pub fn statistics(&self) -> CuriosityStats {
        let total_predictions = self.predictions.len();
        let total_surprises = self.surprises.len();
        
        let avg_surprise = if !self.surprises.is_empty() {
            self.surprises.iter().map(|s| s.magnitude).sum::<f64>() / total_surprises as f64
        } else {
            0.0
        };

        let significant_surprises = self.surprises.iter()
            .filter(|s| s.is_significant())
            .count();

        CuriosityStats {
            total_predictions,
            total_surprises,
            avg_surprise,
            significant_surprises,
            curiosity_score: self.curiosity_score(),
        }
    }

    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

/// Statistics about curiosity-driven learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriosityStats {
    pub total_predictions: usize,
    pub total_surprises: usize,
    pub avg_surprise: f64,
    pub significant_surprises: usize,
    pub curiosity_score: f64,
}

/// Self-supervised learning loop
pub struct SelfSupervisedLoop {
    curiosity: CuriosityEngine,
    /// Number of self-generated questions per cycle
    questions_per_cycle: usize,
    /// Minimum surprise to trigger learning
    min_surprise_threshold: f64,
}

impl SelfSupervisedLoop {
    pub fn new(dimension: usize, questions_per_cycle: usize) -> Self {
        Self {
            curiosity: CuriosityEngine::new(dimension),
            questions_per_cycle,
            min_surprise_threshold: 0.3,
        }
    }

    /// Run one cycle of self-supervised learning
    pub fn run_cycle(&mut self, current_state: &ThoughtState) -> Vec<(String, Surprise)> {
        let mut results = Vec::new();

        for _ in 0..self.questions_per_cycle {
            // Generate question
            let question = self.curiosity.generate_question();

            // Make prediction
            let prediction = self.curiosity.predict(current_state);

            // Simulate observation (in full implementation, would actually test)
            let observation = self.curiosity.observe(current_state);

            // Compute surprise
            let surprise = self.curiosity.compute_surprise(&prediction, &observation);

            results.push((question, surprise));
        }

        results
    }

    /// Should we continue exploring?
    pub fn should_continue(&self) -> bool {
        self.curiosity.should_explore()
    }

    /// Get curiosity statistics
    pub fn statistics(&self) -> CuriosityStats {
        self.curiosity.statistics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surprise_computation() {
        let prediction = Prediction {
            predicted_state: vec![1.0, 2.0, 3.0],
            confidence: 0.8,
            timestamp: 0,
        };

        let observation = Observation {
            observed_state: vec![1.1, 2.1, 3.1],
            timestamp: 1,
        };

        let surprise = Surprise::compute(&prediction, &observation);
        assert!(surprise.magnitude > 0.0);
        assert!(surprise.magnitude < 0.5); // Small error
    }

    #[test]
    fn test_curiosity_engine() {
        let mut engine = CuriosityEngine::new(64);
        let state = ThoughtState::new(64);

        let prediction = engine.predict(&state);
        assert_eq!(prediction.predicted_state.len(), 64);

        let observation = engine.observe(&state);
        let surprise = engine.compute_surprise(&prediction, &observation);

        assert!(surprise.magnitude >= 0.0);
    }

    #[test]
    fn test_self_supervised_loop() {
        let mut loop_engine = SelfSupervisedLoop::new(64, 5);
        let state = ThoughtState::new(64);

        let results = loop_engine.run_cycle(&state);
        assert_eq!(results.len(), 5);
    }
}
