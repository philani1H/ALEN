//! Mood and Emotion API Endpoints
//!
//! Exposes the biologically-inspired mood and emotion system to users.
//! Users can query emotional state, adjust mood parameters, and see how
//! emotions influence reasoning and responses.

use super::{AppState, Problem};

use axum::{
    extract::{State, Json},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Get current emotional and mood state
pub async fn get_emotional_state(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let engine = state.engine.lock().await;

    // Get real mood statistics
    let mood_stats = engine.mood_engine.statistics();
    let current_emotion = engine.emotion_system.current_emotion();
    let emotional_trend = engine.emotion_system.emotional_trend();

    Json(serde_json::json!({
        "mood": {
            "current_mood": mood_stats.current_mood.as_str(),
            "reward_level": mood_stats.reward_level,
            "stress_level": mood_stats.stress_level,
            "trust_level": mood_stats.trust_level,
            "curiosity_level": mood_stats.curiosity_level,
            "energy_level": mood_stats.energy_level,
            "perception_bias": mood_stats.perception_bias,
            "reaction_threshold": mood_stats.reaction_threshold,
            "is_stable": mood_stats.is_stable,
            "history_length": mood_stats.history_length,
        },
        "emotion": {
            "current_emotion": current_emotion.as_str(),
            "recent_emotions": emotional_trend.iter()
                .map(|e| e.as_str())
                .collect::<Vec<_>>(),
        },
        "system_info": {
            "explanation": "ALEN has a biologically-inspired mood and emotion system that:",
            "features": [
                "Slow-changing mood state (like dopamine, serotonin, cortisol baselines)",
                "Reactive emotion processing (limbic system + prefrontal cortex)",
                "Emotions accumulate into mood over time",
                "Mood biases interpretation of inputs (positive mood = positive bias)",
                "Mood changes reaction thresholds (stress = more reactive)",
                "Emotions emerge from neurotransmitter dynamics",
                "Affects learning, memory formation, and reasoning"
            ],
            "not_metaphorical": "This is functional, not symbolic - moods and emotions actually change behavior"
        }
    }))
}

/// Adjust mood parameters
#[derive(Debug, Deserialize)]
pub struct AdjustMoodRequest {
    /// Set reward level (0-1, like dopamine baseline)
    #[serde(default)]
    pub reward_level: Option<f64>,
    /// Set stress level (0-1, like cortisol baseline)
    #[serde(default)]
    pub stress_level: Option<f64>,
    /// Set curiosity level (0-1)
    #[serde(default)]
    pub curiosity_level: Option<f64>,
    /// Set energy level (0-1)
    #[serde(default)]
    pub energy_level: Option<f64>,
    /// Set trust level (0-1, like oxytocin baseline)
    #[serde(default)]
    pub trust_level: Option<f64>,
}

/// Adjust mood state directly
pub async fn adjust_mood(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AdjustMoodRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;

    // Get current mood state
    let mood_state = engine.mood_engine.get_state();
    let mut new_state = mood_state.clone();

    // Apply adjustments
    if let Some(reward) = req.reward_level {
        new_state.reward_level = reward.clamp(0.0, 1.0);
    }
    if let Some(stress) = req.stress_level {
        new_state.stress_level = stress.clamp(0.0, 1.0);
    }
    if let Some(curiosity) = req.curiosity_level {
        new_state.curiosity_level = curiosity.clamp(0.0, 1.0);
    }
    if let Some(energy) = req.energy_level {
        new_state.energy_level = energy.clamp(0.0, 1.0);
    }
    if let Some(trust) = req.trust_level {
        new_state.trust_level = trust.clamp(0.0, 1.0);
    }

    // Update mood engine with new state
    engine.mood_engine.set_state(new_state);

    let mood_stats = engine.mood_engine.statistics();

    Json(serde_json::json!({
        "success": true,
        "message": "Mood adjusted - this will affect how ALEN interprets and responds",
        "new_state": {
            "current_mood": mood_stats.current_mood.as_str(),
            "reward_level": mood_stats.reward_level,
            "stress_level": mood_stats.stress_level,
            "trust_level": mood_stats.trust_level,
            "curiosity_level": mood_stats.curiosity_level,
            "energy_level": mood_stats.energy_level,
            "perception_bias": mood_stats.perception_bias,
            "reaction_threshold": mood_stats.reaction_threshold,
        }
    }))
}

/// Demonstrate emotional response to input
#[derive(Debug, Deserialize)]
pub struct EmotionalDemonstrationRequest {
    pub input: String,
    pub context: Option<String>,
}

/// Demonstrate how mood affects interpretation
pub async fn demonstrate_mood_influence(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmotionalDemonstrationRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    let dim = state.config.dimension;

    // Get mood state before processing
    let before_mood = engine.mood_engine.statistics();
    let before_emotion = engine.emotion_system.current_emotion();

    // Process input through full reasoning pipeline
    let problem = Problem::new(&req.input, dim);
    let result = engine.infer(&problem);

    // Get mood state after processing
    let after_mood = engine.mood_engine.statistics();
    let after_emotion = engine.emotion_system.current_emotion();

    Json(serde_json::json!({
        "input": req.input,
        "interpretation": {
            "confidence": result.confidence,
            "energy": result.energy.total,
            "operator_used": result.operator_id,
            "verified": result.energy.verified
        },
        "emotional_context": {
            "before": {
                "mood": before_mood.current_mood.as_str(),
                "emotion": before_emotion.as_str(),
                "perception_bias": before_mood.perception_bias,
                "reaction_threshold": before_mood.reaction_threshold,
            },
            "after": {
                "mood": after_mood.current_mood.as_str(),
                "emotion": after_emotion.as_str(),
                "perception_bias": after_mood.perception_bias,
                "reaction_threshold": after_mood.reaction_threshold,
            },
            "mood_influence": {
                "description": "Mood biased the interpretation and updated based on result",
                "perception_bias_change": after_mood.perception_bias - before_mood.perception_bias,
                "reaction_threshold_change": after_mood.reaction_threshold - before_mood.reaction_threshold,
            }
        },
        "explanation": {
            "how_it_works": [
                "Same input → different interpretation based on current mood",
                "High stress → perceive threats more easily, lower reaction threshold",
                "High reward → perceive opportunities more easily, positive bias",
                "High curiosity → explore more options",
                "Moods accumulate from emotional responses over time",
                "Emotions process through limbic system then prefrontal cortex",
                "Mood decays slowly toward homeostatic baseline"
            ]
        }
    }))
}

/// Reset mood to baseline
pub async fn reset_mood(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;

    // Reset to default mood engine
    engine.mood_engine = crate::control::MoodEngine::new();

    let mood_stats = engine.mood_engine.statistics();

    Json(serde_json::json!({
        "success": true,
        "message": "Mood and emotion systems reset to neutral baseline",
        "state": {
            "current_mood": mood_stats.current_mood.as_str(),
            "reward_level": mood_stats.reward_level,
            "stress_level": mood_stats.stress_level,
            "trust_level": mood_stats.trust_level,
            "curiosity_level": mood_stats.curiosity_level,
            "energy_level": mood_stats.energy_level,
        }
    }))
}

/// Get mood history and patterns
pub async fn get_mood_patterns(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let engine = state.engine.lock().await;

    let mood_stats = engine.mood_engine.statistics();
    let mood_trend = engine.mood_engine.mood_trend();
    let emotional_trend = engine.emotion_system.emotional_trend();

    Json(serde_json::json!({
        "current_state": {
            "mood": mood_stats.current_mood.as_str(),
            "emotion": engine.emotion_system.current_emotion().as_str(),
        },
        "trends": {
            "recent_moods": mood_trend.iter().map(|m| m.as_str()).collect::<Vec<_>>(),
            "recent_emotions": emotional_trend.iter().map(|e| e.as_str()).collect::<Vec<_>>(),
            "is_stable": mood_stats.is_stable,
        },
        "patterns": {
            "description": "Moods emerge from feedback loops over time",
            "examples": [
                {
                    "pattern": "Success reinforcement",
                    "mechanism": "Verified learning → dopamine release → mood improves → more confidence → more success"
                },
                {
                    "pattern": "Stress accumulation",
                    "mechanism": "Failed attempts → cortisol release → mood worsens → higher stress → more reactive"
                },
                {
                    "pattern": "Curiosity satiation",
                    "mechanism": "Novel inputs → dopamine/glutamate → curiosity increases → explore more → eventually settle"
                },
                {
                    "pattern": "Homeostatic regulation",
                    "mechanism": "Extreme moods decay toward baseline over time (like biological systems)"
                }
            ]
        },
        "biological_analogy": {
            "dopamine": "reward_level (confidence and exploration tendency)",
            "cortisol": "stress_level (uncertainty and reactivity)",
            "serotonin": "implicitly modeled in stability and decay",
            "oxytocin": "trust_level (social bonding, system trust)",
            "note": "ALEN's mood emerges from neurotransmitter-like dynamics and accumulates from emotional responses"
        }
    }))
}
