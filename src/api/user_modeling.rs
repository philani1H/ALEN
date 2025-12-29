//! User Modeling System
//!
//! Implements the clean architecture for learning users:
//! - Preferences (depth, math, verbosity)
//! - Interests (topics, domains)
//! - Skill estimation (from corrections)
//! - User embedding (who is this person?)
//!
//! This is SEPARATE from problem-solving - it only models the human.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Beta distribution for Bayesian preference learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaDist {
    pub alpha: f64,
    pub beta: f64,
}

impl BetaDist {
    pub fn new() -> Self {
        Self { alpha: 1.0, beta: 1.0 }
    }

    /// Get mean of distribution
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Get confidence (inverse of variance)
    pub fn confidence(&self) -> f64 {
        self.alpha + self.beta
    }

    /// Update with success/failure
    pub fn update(&mut self, success: bool) {
        if success {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
    }
}

/// User preferences (learned from behavior)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Depth preference: 0 = concise, 1 = detailed
    pub depth: BetaDist,
    
    /// Math preference: 0 = avoid, 1 = include
    pub math: BetaDist,
    
    /// Verbosity preference: 0 = terse, 1 = verbose
    pub verbosity: BetaDist,
    
    /// Technical level: 0 = beginner, 1 = expert
    pub technical_level: BetaDist,
    
    /// Formality: 0 = casual, 1 = formal
    pub formality: BetaDist,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            depth: BetaDist::new(),
            math: BetaDist::new(),
            verbosity: BetaDist::new(),
            technical_level: BetaDist::new(),
            formality: BetaDist::new(),
        }
    }
}

/// Interest in a topic (evidence-based)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicInterest {
    pub topic_id: String,
    pub score: f64,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
    pub followup_count: u32,
    pub correction_count: u32,
}

impl TopicInterest {
    pub fn new(topic_id: String) -> Self {
        Self {
            topic_id,
            score: 0.1,
            last_accessed: Utc::now(),
            access_count: 1,
            followup_count: 0,
            correction_count: 0,
        }
    }

    /// Update interest with decay
    pub fn update(&mut self, lambda: f64, weight: f64) {
        self.score = lambda * self.score + weight;
        self.last_accessed = Utc::now();
        self.access_count += 1;
    }

    /// Increase weight when user follows up
    pub fn followup(&mut self) {
        self.followup_count += 1;
        self.score += 0.2;
    }

    /// Increase weight when user corrects
    pub fn correction(&mut self) {
        self.correction_count += 1;
        self.score += 0.3;
    }
}

/// Skill estimation for a domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillEstimate {
    pub domain: String,
    pub skill_level: BetaDist,
    pub correct_count: u32,
    pub incorrect_count: u32,
}

impl SkillEstimate {
    pub fn new(domain: String) -> Self {
        Self {
            domain,
            skill_level: BetaDist::new(),
            correct_count: 0,
            incorrect_count: 0,
        }
    }

    /// Update from user behavior
    pub fn update_from_behavior(&mut self, correct: bool) {
        if correct {
            self.correct_count += 1;
            self.skill_level.update(true);
        } else {
            self.incorrect_count += 1;
            self.skill_level.update(false);
        }
    }

    /// Get estimated skill level
    pub fn level(&self) -> f64 {
        self.skill_level.mean()
    }
}

/// Interaction features extracted from behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionFeatures {
    pub topic_id: String,
    pub depth_requested: f64,
    pub math_density: f64,
    pub correction: bool,
    pub followup: bool,
    pub followup_latency: f64,
    pub message_length: usize,
    pub question_count: usize,
}

impl InteractionFeatures {
    pub fn extract(message: &str, is_followup: bool, latency: f64) -> Self {
        let lower = message.to_lowercase();
        
        // Detect depth request
        let depth_requested = if lower.contains("explain") || lower.contains("detail") || lower.contains("how") {
            0.8
        } else if lower.contains("brief") || lower.contains("short") || lower.contains("quick") {
            0.2
        } else {
            0.5
        };

        // Detect math density
        let math_keywords = ["equation", "formula", "calculate", "prove", "derive"];
        let math_density = math_keywords.iter()
            .filter(|k| lower.contains(*k))
            .count() as f64 / math_keywords.len() as f64;

        // Detect correction
        let correction = lower.contains("no") || lower.contains("wrong") || lower.contains("actually") || lower.contains("correct");

        // Count questions
        let question_count = message.matches('?').count();

        // Extract topic (simplified)
        let topic_id = lower.split_whitespace()
            .filter(|w| w.len() > 3)
            .take(2)
            .collect::<Vec<_>>()
            .join("_");

        Self {
            topic_id,
            depth_requested,
            math_density,
            correction,
            followup: is_followup,
            followup_latency: latency,
            message_length: message.len(),
            question_count,
        }
    }
}

/// Complete user state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserState {
    pub user_id: String,
    pub preferences: UserPreferences,
    pub interests: HashMap<String, TopicInterest>,
    pub skills: HashMap<String, SkillEstimate>,
    pub embedding: Vec<f64>,
    pub interaction_count: u32,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

impl UserState {
    pub fn new(user_id: String, dimension: usize) -> Self {
        Self {
            user_id,
            preferences: UserPreferences::default(),
            interests: HashMap::new(),
            skills: HashMap::new(),
            embedding: vec![0.0; dimension],
            interaction_count: 0,
            created_at: Utc::now(),
            last_updated: Utc::now(),
        }
    }

    /// Update from interaction
    pub fn update_from_interaction(&mut self, features: &InteractionFeatures, response_engaged: bool) {
        self.interaction_count += 1;
        self.last_updated = Utc::now();

        // Update preferences
        if features.depth_requested > 0.6 {
            self.preferences.depth.update(response_engaged);
        } else if features.depth_requested < 0.4 {
            self.preferences.depth.update(!response_engaged);
        }

        if features.math_density > 0.3 {
            self.preferences.math.update(response_engaged);
        }

        if features.message_length > 100 {
            self.preferences.verbosity.update(true);
        } else if features.message_length < 30 {
            self.preferences.verbosity.update(false);
        }

        // Update interests
        let interest = self.interests
            .entry(features.topic_id.clone())
            .or_insert_with(|| TopicInterest::new(features.topic_id.clone()));

        let weight = if features.followup { 0.3 } else { 0.1 };
        interest.update(0.95, weight);

        if features.followup {
            interest.followup();
        }

        if features.correction {
            interest.correction();
        }

        // Update embedding (exponential moving average)
        let alpha = 0.1;
        for i in 0..self.embedding.len().min(10) {
            let feature_val = match i {
                0 => features.depth_requested,
                1 => features.math_density,
                2 => if features.correction { 1.0 } else { 0.0 },
                3 => if features.followup { 1.0 } else { 0.0 },
                4 => features.followup_latency / 100.0,
                5 => features.message_length as f64 / 500.0,
                6 => features.question_count as f64 / 5.0,
                7 => self.preferences.depth.mean(),
                8 => self.preferences.math.mean(),
                9 => self.preferences.technical_level.mean(),
                _ => 0.0,
            };
            self.embedding[i] = alpha * feature_val + (1.0 - alpha) * self.embedding[i];
        }
    }

    /// Get user archetype
    pub fn archetype(&self) -> UserArchetype {
        let depth = self.preferences.depth.mean();
        let math = self.preferences.math.mean();
        let technical = self.preferences.technical_level.mean();

        if depth > 0.7 && math > 0.6 {
            UserArchetype::Analytical
        } else if depth > 0.7 && technical < 0.4 {
            UserArchetype::Curious
        } else if math > 0.7 {
            UserArchetype::Technical
        } else if depth < 0.3 {
            UserArchetype::Pragmatic
        } else {
            UserArchetype::Balanced
        }
    }

    /// Should include math in response?
    pub fn wants_math(&self) -> bool {
        self.preferences.math.mean() > 0.5
    }

    /// Preferred response depth
    pub fn preferred_depth(&self) -> ResponseDepth {
        let depth = self.preferences.depth.mean();
        if depth > 0.7 {
            ResponseDepth::Detailed
        } else if depth < 0.3 {
            ResponseDepth::Concise
        } else {
            ResponseDepth::Moderate
        }
    }
}

/// User archetype (cluster)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UserArchetype {
    Analytical,    // Deep + Math
    Curious,       // Deep + Low Technical
    Technical,     // High Math
    Pragmatic,     // Low Depth
    Balanced,      // Middle ground
}

/// Response depth level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseDepth {
    Concise,
    Moderate,
    Detailed,
}

/// User modeling manager
pub struct UserModelingManager {
    users: HashMap<String, UserState>,
    dimension: usize,
}

impl UserModelingManager {
    pub fn new(dimension: usize) -> Self {
        Self {
            users: HashMap::new(),
            dimension,
        }
    }

    /// Get or create user state
    pub fn get_or_create(&mut self, user_id: &str) -> &mut UserState {
        self.users
            .entry(user_id.to_string())
            .or_insert_with(|| UserState::new(user_id.to_string(), self.dimension))
    }

    /// Get user state
    pub fn get(&self, user_id: &str) -> Option<&UserState> {
        self.users.get(user_id)
    }

    /// Update user from interaction
    pub fn update_user(
        &mut self,
        user_id: &str,
        message: &str,
        is_followup: bool,
        latency: f64,
        response_engaged: bool,
    ) {
        let features = InteractionFeatures::extract(message, is_followup, latency);
        let user = self.get_or_create(user_id);
        user.update_from_interaction(&features, response_engaged);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_dist() {
        let mut dist = BetaDist::new();
        assert_eq!(dist.mean(), 0.5);

        dist.update(true);
        assert!(dist.mean() > 0.5);

        dist.update(false);
        // Should be closer to 0.5 again
    }

    #[test]
    fn test_user_state() {
        let mut user = UserState::new("test_user".to_string(), 128);
        
        let features = InteractionFeatures {
            topic_id: "math".to_string(),
            depth_requested: 0.8,
            math_density: 0.7,
            correction: false,
            followup: true,
            followup_latency: 5.0,
            message_length: 150,
            question_count: 2,
        };

        user.update_from_interaction(&features, true);
        
        assert_eq!(user.interaction_count, 1);
        assert!(user.preferences.depth.mean() > 0.5);
    }

    #[test]
    fn test_archetype_detection() {
        let mut user = UserState::new("test".to_string(), 128);
        
        // Simulate analytical user
        user.preferences.depth.alpha = 8.0;
        user.preferences.depth.beta = 2.0;
        user.preferences.math.alpha = 7.0;
        user.preferences.math.beta = 3.0;

        assert_eq!(user.archetype(), UserArchetype::Analytical);
    }
}
