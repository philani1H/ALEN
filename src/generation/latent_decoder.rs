//! Latent Controller (φ parameters) - CONTROLS HOW TO THINK, NOT WHAT TO SAY
//!
//! This is the latent decoder / controller component in the architecture:
//! - Produces control variables z (intent, style, retrieval, action, reasoning depth)
//! - Queries memory and assembles context
//! - Does NOT generate text responses
//! - The core model (θ) generates responses from the assembled context
//!
//! Architecture (CORRECT):
//! 1. Controller reads input and state → produces z ~ q_φ(z | x, m)
//! 2. Controller queries memory → retrieves context r
//! 3. Compose context c = Compose(x, r, z)
//! 4. Core model generates y ~ p_θ(y | c)  ← BRAIN GENERATES, NOT CONTROLLER!
//!
//! This implements the mathematical framework:
//! z ~ q_φ(z | x, m) where:
//!   - x = input
//!   - m = memory/state (confidence, unknownness, style, intent, history)
//!   - z = control variables (intent, style, retrieval query, action, reasoning depth)
//!   - φ = controller parameters (mostly stable, small LR)
//!
//! The controller is your brainstem + executive function, NOT the brain.

use crate::core::ThoughtState;
use crate::memory::semantic::SemanticMemory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;
use rand_distr::{Normal, Distribution};

// ============================================================================
// PART 1: CONTROL VARIABLES (z)
// ============================================================================

/// Action the controller decides to take
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ControlAction {
    /// Answer with current knowledge
    Answer,
    /// Ask clarifying question
    Ask,
    /// Verify/rethink before answering
    VerifyMore,
    /// Regenerate with different approach
    Regenerate,
}

/// Intent classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentClass {
    Question,
    Statement,
    Command,
    Greeting,
    Unknown,
}

/// Style/pacing vector for response generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleVector {
    /// Verbosity level (0.0 = terse, 1.0 = verbose)
    pub verbosity: f64,
    /// Formality level (0.0 = casual, 1.0 = formal)
    pub formality: f64,
    /// Technical depth (0.0 = simple, 1.0 = technical)
    pub technical_depth: f64,
    /// Creativity level (0.0 = factual, 1.0 = creative)
    pub creativity: f64,
}

impl Default for StyleVector {
    fn default() -> Self {
        Self {
            verbosity: 0.5,
            formality: 0.5,
            technical_depth: 0.5,
            creativity: 0.3,
        }
    }
}

/// Complete control variables produced by the controller
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlVariables {
    /// Intent classification (z_I)
    pub intent: IntentClass,

    /// Style/pacing vector (z_S)
    pub style: StyleVector,

    /// Retrieval query vector (z_K) for memory lookup
    pub retrieval_query: Vec<f64>,

    /// Reasoning depth budget (z_R)
    pub reasoning_depth: usize,

    /// Action to take (z_A)
    pub action: ControlAction,

    /// Confidence in this control decision
    pub confidence: f64,
}

// ============================================================================
// PART 2: MEMORY STATE (m)
// ============================================================================

/// Memory state used by controller to make decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryState {
    /// Confidence in current knowledge
    pub confidence: f64,

    /// Unknownness estimate
    pub unknownness: f64,

    /// Risk level
    pub risk: f64,

    /// Preferred verbosity
    pub verbosity_pref: f64,

    /// Topic context
    pub topic: Option<String>,

    /// Conversation history summary
    pub history_summary: Vec<f64>,
}

impl Default for MemoryState {
    fn default() -> Self {
        Self {
            confidence: 0.5,
            unknownness: 0.5,
            risk: 0.3,
            verbosity_pref: 0.5,
            topic: None,
            history_summary: Vec::new(),
        }
    }
}

// ============================================================================
// PART 3: LATENT CONTROLLER (THE DECODER φ)
// ============================================================================

/// Latent Controller - produces control variables, NOT text responses
/// This is the φ network in the architecture
#[derive(Serialize, Deserialize)]
pub struct LatentController {
    /// Learned patterns for control decision-making
    control_patterns: Vec<ControlPattern>,

    /// Dimension of thought space
    dimension: usize,

    /// Learning rate (SMALL - controller is governance, stays stable)
    learning_rate: f64,

    /// Confidence thresholds for action decisions
    action_thresholds: ActionThresholds,

    /// Training count
    training_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ControlPattern {
    /// Pattern centroid in thought space
    centroid: Vec<f64>,

    /// Learned intent distribution
    intent_weights: HashMap<String, f64>,

    /// Learned style preferences
    style_preferences: StyleVector,

    /// Typical reasoning depth for this pattern
    typical_depth: usize,

    /// Example count
    example_count: u32,
}

impl ControlPattern {
    fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        Self {
            centroid: (0..dimension).map(|_| normal.sample(&mut rng)).collect(),
            intent_weights: HashMap::new(),
            style_preferences: StyleVector::default(),
            typical_depth: 3,
            example_count: 0,
        }
    }

    fn similarity(&self, thought: &[f64]) -> f64 {
        if self.centroid.len() != thought.len() {
            return 0.0;
        }

        let dot: f64 = self.centroid.iter()
            .zip(thought.iter())
            .map(|(c, t)| c * t)
            .sum();

        let norm_c: f64 = self.centroid.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_t: f64 = thought.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_c > 1e-10 && norm_t > 1e-10 {
            (dot / (norm_c * norm_t) + 1.0) / 2.0
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionThresholds {
    /// τ_1: Below this, ASK for clarification
    pub ask_threshold: f64,
    /// τ_2: Between τ_1 and τ_2, VERIFY_MORE (Above τ_2: ANSWER)
    pub verify_threshold: f64,
}

impl Default for ActionThresholds {
    fn default() -> Self {
        Self {
            ask_threshold: 0.3,
            verify_threshold: 0.6,
        }
    }
}

impl LatentController {
    pub fn new(dimension: usize, num_patterns: usize) -> Self {
        let patterns = (0..num_patterns)
            .map(|_| ControlPattern::new(dimension))
            .collect();

        Self {
            control_patterns: patterns,
            dimension,
            learning_rate: 0.01, // SMALL LR - controller is governance
            action_thresholds: ActionThresholds::default(),
            training_count: 0,
        }
    }

    /// Main controller function: q_φ(z | x, m)
    /// Produces control variables from input and memory state
    /// DOES NOT GENERATE TEXT!
    pub fn produce_controls(
        &self,
        thought: &ThoughtState,
        memory_state: &MemoryState,
    ) -> ControlVariables {
        // Step 1: Find relevant control patterns
        let pattern_activations: Vec<(usize, f64)> = self.control_patterns
            .iter()
            .enumerate()
            .map(|(idx, p)| (idx, p.similarity(&thought.vector)))
            .filter(|(_, sim)| *sim > 0.1)
            .collect();

        // Step 2: Classify intent (z_I)
        let intent = self.classify_intent(&pattern_activations);

        // Step 3: Determine style (z_S)
        let style = self.determine_style(&pattern_activations, memory_state);

        // Step 4: Generate retrieval query (z_K)
        let retrieval_query = self.generate_retrieval_query(thought, memory_state);

        // Step 5: Decide reasoning depth (z_R)
        let reasoning_depth = self.decide_reasoning_depth(&pattern_activations, memory_state);

        // Step 6: Decide action (z_A) based on confidence
        let confidence = self.estimate_confidence(&pattern_activations, memory_state);
        let action = self.decide_action(confidence);

        ControlVariables {
            intent,
            style,
            retrieval_query,
            reasoning_depth,
            action,
            confidence,
        }
    }

    /// Classify intent from pattern activations
    fn classify_intent(&self, activations: &[(usize, f64)]) -> IntentClass {
        if activations.is_empty() {
            return IntentClass::Unknown;
        }

        let mut intent_scores: HashMap<String, f64> = HashMap::new();

        for (idx, activation) in activations {
            let pattern = &self.control_patterns[*idx];
            for (intent, weight) in &pattern.intent_weights {
                let score = intent_scores.get(intent).copied().unwrap_or(0.0);
                intent_scores.insert(intent.clone(), score + activation * weight);
            }
        }

        // Get top intent
        let top_intent = intent_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.as_str());

        match top_intent {
            Some("question") => IntentClass::Question,
            Some("statement") => IntentClass::Statement,
            Some("command") => IntentClass::Command,
            Some("greeting") => IntentClass::Greeting,
            _ => IntentClass::Unknown,
        }
    }

    /// Determine style from patterns and memory state
    fn determine_style(
        &self,
        activations: &[(usize, f64)],
        memory_state: &MemoryState,
    ) -> StyleVector {
        if activations.is_empty() {
            return StyleVector::default();
        }

        // Weighted average of pattern styles
        let mut style = StyleVector {
            verbosity: 0.0,
            formality: 0.0,
            technical_depth: 0.0,
            creativity: 0.0,
        };

        let total_activation: f64 = activations.iter().map(|(_, a)| a).sum();

        for (idx, activation) in activations {
            let pattern = &self.control_patterns[*idx];
            let weight = activation / total_activation;

            style.verbosity += pattern.style_preferences.verbosity * weight;
            style.formality += pattern.style_preferences.formality * weight;
            style.technical_depth += pattern.style_preferences.technical_depth * weight;
            style.creativity += pattern.style_preferences.creativity * weight;
        }

        // Modulate with memory state preferences
        style.verbosity = (style.verbosity + memory_state.verbosity_pref) / 2.0;

        style
    }

    /// Generate retrieval query vector for memory lookup
    fn generate_retrieval_query(
        &self,
        thought: &ThoughtState,
        memory_state: &MemoryState,
    ) -> Vec<f64> {
        // Combine thought vector with history context
        let mut query = thought.vector.clone();

        // If we have history, blend it in
        if !memory_state.history_summary.is_empty() {
            for (i, val) in query.iter_mut().enumerate() {
                if i < memory_state.history_summary.len() {
                    *val = 0.7 * *val + 0.3 * memory_state.history_summary[i];
                }
            }
        }

        query
    }

    /// Decide reasoning depth based on patterns and uncertainty
    fn decide_reasoning_depth(
        &self,
        activations: &[(usize, f64)],
        memory_state: &MemoryState,
    ) -> usize {
        if activations.is_empty() {
            return 3; // default
        }

        // Average depth from patterns
        let total_activation: f64 = activations.iter().map(|(_, a)| a).sum();
        let avg_depth: f64 = activations
            .iter()
            .map(|(idx, a)| {
                self.control_patterns[*idx].typical_depth as f64 * (a / total_activation)
            })
            .sum();

        // Increase depth if high unknownness
        let depth_boost = if memory_state.unknownness > 0.7 { 2 } else { 0 };

        (avg_depth as usize + depth_boost).clamp(1, 10)
    }

    /// Estimate confidence from patterns and memory
    fn estimate_confidence(
        &self,
        activations: &[(usize, f64)],
        memory_state: &MemoryState,
    ) -> f64 {
        if activations.is_empty() {
            return memory_state.confidence * 0.5; // Low confidence if no patterns
        }

        // Max activation as confidence
        let max_activation = activations
            .iter()
            .map(|(_, a)| a)
            .fold(0.0f64, |a, b| a.max(*b));

        // Combine with memory state confidence
        let combined = 0.6 * max_activation + 0.4 * memory_state.confidence;

        // Penalize if high unknownness
        combined * (1.0 - 0.3 * memory_state.unknownness)
    }

    /// Decide action based on confidence (implements the threshold logic)
    fn decide_action(&self, confidence: f64) -> ControlAction {
        if confidence < self.action_thresholds.ask_threshold {
            ControlAction::Ask
        } else if confidence < self.action_thresholds.verify_threshold {
            ControlAction::VerifyMore
        } else {
            ControlAction::Answer
        }
    }

    /// Learn from supervision (update controller with small LR)
    /// This is L_ctrl(φ) = α*L_A + β*L_ret + γ*L_style
    pub fn learn(
        &mut self,
        thought: &ThoughtState,
        true_action: ControlAction,
        true_style: Option<StyleVector>,
    ) {
        // Find best matching pattern
        let (best_idx, _) = self.control_patterns
            .iter()
            .enumerate()
            .map(|(idx, p)| (idx, p.similarity(&thought.vector)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));

        let pattern = &mut self.control_patterns[best_idx];

        // Update centroid (small LR!)
        for (c, t) in pattern.centroid.iter_mut().zip(thought.vector.iter()) {
            *c = (1.0 - self.learning_rate) * *c + self.learning_rate * t;
        }

        // Update intent weights
        let intent_str = match true_action {
            ControlAction::Ask => "question",
            _ => "statement",
        };
        let current = pattern.intent_weights.get(intent_str).copied().unwrap_or(0.0);
        pattern.intent_weights.insert(
            intent_str.to_string(),
            current + self.learning_rate,
        );

        // Update style if provided
        if let Some(style) = true_style {
            pattern.style_preferences.verbosity =
                (1.0 - self.learning_rate) * pattern.style_preferences.verbosity
                + self.learning_rate * style.verbosity;
            pattern.style_preferences.formality =
                (1.0 - self.learning_rate) * pattern.style_preferences.formality
                + self.learning_rate * style.formality;
        }

        pattern.example_count += 1;
        self.training_count += 1;
    }

    /// Assemble context for the core model
    /// c = Compose(x, r, z)
    pub fn assemble_context(
        input: &ThoughtState,
        retrieved_memory: &[Vec<f64>],
        controls: &ControlVariables,
    ) -> Vec<f64> {
        let mut context = input.vector.clone();

        // Add retrieved memory (average)
        if !retrieved_memory.is_empty() {
            let memory_avg: Vec<f64> = (0..context.len())
                .map(|i| {
                    retrieved_memory
                        .iter()
                        .map(|m| m.get(i).copied().unwrap_or(0.0))
                        .sum::<f64>()
                        / retrieved_memory.len() as f64
                })
                .collect();

            // Blend input with memory
            for (i, c_val) in context.iter_mut().enumerate() {
                if i < memory_avg.len() {
                    *c_val = 0.6 * *c_val + 0.4 * memory_avg[i];
                }
            }
        }

        // Add control signal (style, intent) as modulation
        let style_signal = (controls.style.verbosity
            + controls.style.formality
            + controls.style.technical_depth
            + controls.style.creativity) / 4.0;

        // Modulate context based on style
        for val in context.iter_mut() {
            *val *= 1.0 + 0.2 * (style_signal - 0.5);
        }

        context
    }

    /// Get statistics
    pub fn stats(&self) -> LatentControllerStats {
        let active_patterns = self.control_patterns
            .iter()
            .filter(|p| p.example_count > 0)
            .count();

        LatentControllerStats {
            total_patterns: self.control_patterns.len(),
            active_patterns,
            training_count: self.training_count,
            learning_rate: self.learning_rate,
        }
    }

    /// Save to file
    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let data = bincode::serialize(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load from file
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let controller = bincode::deserialize(&data)?;
        Ok(controller)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentControllerStats {
    pub total_patterns: usize,
    pub active_patterns: usize,
    pub training_count: u64,
    pub learning_rate: f64,
}

// ============================================================================
// BACKWARD COMPATIBILITY SHIMS (for existing code that calls generate())
// ============================================================================

/// OLD API SHIM - For backward compatibility
/// This delegates to the controller but doesn't generate text!
/// The calling code must use the controls to invoke the core model.
#[derive(Serialize, Deserialize)]
pub struct LatentDecoder {
    controller: LatentController,
}

impl LatentDecoder {
    pub fn new(dimension: usize, num_patterns: usize) -> Self {
        Self {
            controller: LatentController::new(dimension, num_patterns),
        }
    }

    /// OLD API: learn from thought-text pair
    /// Now learns control patterns instead of generating text
    pub fn learn(&mut self, thought: &ThoughtState, _text: &str) {
        // Infer controls from the training example
        // (In production, you'd have supervised control labels)
        self.controller.learn(
            thought,
            ControlAction::Answer, // default
            None,
        );
    }

    /// OLD API: generate text
    /// NOW RETURNS EMPTY - caller must use controller + core model!
    pub fn generate(&self, _thought: &ThoughtState) -> (String, f64) {
        // DO NOT GENERATE TEXT!
        // Return empty string to signal caller must use proper flow
        (String::new(), 0.0)
    }

    /// NEW API: Get controls instead of generating
    pub fn get_controls(
        &self,
        thought: &ThoughtState,
        memory_state: &MemoryState,
    ) -> ControlVariables {
        self.controller.produce_controls(thought, memory_state)
    }

    pub fn stats(&self) -> LatentControllerStats {
        self.controller.stats()
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        self.controller.save(path)
    }

    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            controller: LatentController::load(path)?,
        })
    }

    // Compatibility methods that do nothing
    pub fn set_temperature(&mut self, _temperature: f64) {}
    pub fn set_max_tokens(&mut self, _max_tokens: usize) {}
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.controller.learning_rate = lr.max(0.001).min(0.1);
    }
    pub fn generate_verified(&self, _thought: &ThoughtState, _min_confidence: f64) -> Option<(String, f64, bool)> {
        None // Must use new API!
    }
    pub fn verify_response(&self, _thought: &ThoughtState, _response: &str) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_production() {
        let controller = LatentController::new(64, 10);
        let thought = ThoughtState::random(64);
        let memory = MemoryState::default();

        let controls = controller.produce_controls(&thought, &memory);

        // Should produce valid controls
        assert!(controls.confidence >= 0.0 && controls.confidence <= 1.0);
        assert!(controls.reasoning_depth >= 1);
    }

    #[test]
    fn test_action_decision() {
        let controller = LatentController::new(64, 10);

        // Low confidence → Ask
        assert_eq!(controller.decide_action(0.2), ControlAction::Ask);

        // Medium confidence → VerifyMore
        assert_eq!(controller.decide_action(0.5), ControlAction::VerifyMore);

        // High confidence → Answer
        assert_eq!(controller.decide_action(0.8), ControlAction::Answer);
    }

    #[test]
    fn test_no_text_generation() {
        let decoder = LatentDecoder::new(64, 10);
        let thought = ThoughtState::random(64);

        // Old API should return empty!
        let (text, _) = decoder.generate(&thought);
        assert_eq!(text, "");
    }

    #[test]
    fn test_context_assembly() {
        let input = ThoughtState::random(64);
        let memory = vec![vec![0.5; 64]];
        let controls = ControlVariables {
            intent: IntentClass::Question,
            style: StyleVector::default(),
            retrieval_query: vec![0.0; 64],
            reasoning_depth: 3,
            action: ControlAction::Answer,
            confidence: 0.8,
        };

        let context = LatentController::assemble_context(&input, &memory, &controls);
        assert_eq!(context.len(), 64);
    }
}
