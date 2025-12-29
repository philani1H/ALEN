//! Teaching Effectiveness Tracker
//!
//! Measures and improves teaching quality through:
//! - Comprehension tracking
//! - Engagement measurement
//! - Feedback integration
//! - Adaptive improvement

use crate::api::user_modeling::UserArchetype;
use std::hash::{Hash, Hasher};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// PART 1: USER FEEDBACK
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedback {
    /// Concept being explained
    pub concept: String,
    
    /// User's message (for context)
    pub message: String,
    
    /// Comprehension score (0-1, from user rating or inferred)
    pub comprehension_score: f64,
    
    /// Engagement score (0-1, from interaction patterns)
    pub engagement_score: f64,
    
    /// Number of follow-up questions
    pub followup_count: usize,
    
    /// Did user need correction?
    pub correction_needed: bool,
    
    /// Time to understanding (seconds)
    pub time_to_understanding: f64,
    
    /// Was this a follow-up question?
    pub is_followup: bool,
    
    /// User engagement indicators
    pub engaged: bool,
}

impl UserFeedback {
    pub fn from_interaction(
        concept: String,
        message: String,
        is_followup: bool,
        latency: f64,
        engaged: bool,
    ) -> Self {
        // Infer comprehension from interaction patterns
        let comprehension_score = if is_followup {
            0.5 // Follow-up suggests partial understanding
        } else {
            0.8 // No follow-up suggests good understanding
        };
        
        // Infer engagement from latency and engagement flag
        let engagement_score = if engaged && latency < 60.0 {
            0.9
        } else if engaged {
            0.7
        } else {
            0.4
        };
        
        Self {
            concept,
            message,
            comprehension_score,
            engagement_score,
            followup_count: if is_followup { 1 } else { 0 },
            correction_needed: false,
            time_to_understanding: latency,
            is_followup,
            engaged,
        }
    }
}

// ============================================================================
// PART 2: EFFECTIVENESS RECORD
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectivenessRecord {
    /// Concept explained
    pub concept: String,
    
    /// Audience type
    pub audience: UserArchetype,
    
    /// Explanation ID
    pub explanation_id: String,
    
    /// Comprehension score
    pub comprehension_score: f64,
    
    /// Engagement score
    pub engagement_score: f64,
    
    /// Follow-up count
    pub followup_count: usize,
    
    /// Correction needed
    pub correction_needed: bool,
    
    /// Time to understanding
    pub time_to_understanding: f64,
    
    /// Timestamp
    pub timestamp: u64,
}

// ============================================================================
// PART 3: CONCEPT EFFECTIVENESS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptEffectiveness {
    /// Concept name
    pub concept: String,
    
    /// Total explanations
    pub total_explanations: usize,
    
    /// Average comprehension
    pub avg_comprehension: f64,
    
    /// Average engagement
    pub avg_engagement: f64,
    
    /// Correction rate
    pub correction_rate: f64,
    
    /// Average time to understanding
    pub avg_time_to_understanding: f64,
}

impl Default for ConceptEffectiveness {
    fn default() -> Self {
        Self {
            concept: String::new(),
            total_explanations: 0,
            avg_comprehension: 0.0,
            avg_engagement: 0.0,
            correction_rate: 0.0,
            avg_time_to_understanding: 0.0,
        }
    }
}

// ============================================================================
// PART 4: AUDIENCE EFFECTIVENESS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudienceEffectiveness {
    /// Audience type
    pub audience: UserArchetype,
    
    /// Total explanations
    pub total_explanations: usize,
    
    /// Average comprehension
    pub avg_comprehension: f64,
    
    /// Average engagement
    pub avg_engagement: f64,
    
    /// Success rate (comprehension > 0.7)
    pub success_rate: f64,
}

impl Default for AudienceEffectiveness {
    fn default() -> Self {
        Self {
            audience: UserArchetype::Balanced,
            total_explanations: 0,
            avg_comprehension: 0.0,
            avg_engagement: 0.0,
            success_rate: 0.0,
        }
    }
}

// ============================================================================
// PART 5: EFFECTIVENESS STATISTICS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectivenessStats {
    /// Total explanations given
    pub total_explanations: usize,
    
    /// Average comprehension score
    pub avg_comprehension: f64,
    
    /// Average engagement score
    pub avg_engagement: f64,
    
    /// Correction rate
    pub correction_rate: f64,
    
    /// Average time to understanding
    pub avg_time_to_understanding: f64,
    
    /// Success rate (comprehension > 0.7)
    pub success_rate: f64,
}

impl Default for EffectivenessStats {
    fn default() -> Self {
        Self {
            total_explanations: 0,
            avg_comprehension: 0.0,
            avg_engagement: 0.0,
            correction_rate: 0.0,
            avg_time_to_understanding: 0.0,
            success_rate: 0.0,
        }
    }
}

// ============================================================================
// PART 6: TEACHING EFFECTIVENESS TRACKER
// ============================================================================

pub struct TeachingEffectivenessTracker {
    /// Historical effectiveness data
    effectiveness_history: Vec<EffectivenessRecord>,
    
    /// Per-concept effectiveness
    concept_effectiveness: HashMap<String, ConceptEffectiveness>,
    
    /// Per-audience effectiveness
    audience_effectiveness: HashMap<UserArchetype, AudienceEffectiveness>,
}

impl TeachingEffectivenessTracker {
    pub fn new() -> Self {
        Self {
            effectiveness_history: Vec::new(),
            concept_effectiveness: HashMap::new(),
            audience_effectiveness: HashMap::new(),
        }
    }
    
    /// Record an outcome
    pub fn record_outcome(
        &mut self,
        concept: &str,
        audience: UserArchetype,
        explanation_id: &str,
        user_feedback: &UserFeedback,
    ) {
        let record = EffectivenessRecord {
            concept: concept.to_string(),
            audience,
            explanation_id: explanation_id.to_string(),
            comprehension_score: user_feedback.comprehension_score,
            engagement_score: user_feedback.engagement_score,
            followup_count: user_feedback.followup_count,
            correction_needed: user_feedback.correction_needed,
            time_to_understanding: user_feedback.time_to_understanding,
            timestamp: Self::current_timestamp(),
        };
        
        self.effectiveness_history.push(record.clone());
        
        // Update concept effectiveness
        self.update_concept_effectiveness(concept, &record);
        
        // Update audience effectiveness
        self.update_audience_effectiveness(audience, &record);
    }
    
    /// Update concept effectiveness
    fn update_concept_effectiveness(&mut self, concept: &str, record: &EffectivenessRecord) {
        let stats = self.concept_effectiveness
            .entry(concept.to_string())
            .or_insert_with(|| ConceptEffectiveness {
                concept: concept.to_string(),
                ..Default::default()
            });
        
        let n = stats.total_explanations as f64;
        stats.total_explanations += 1;
        let new_n = stats.total_explanations as f64;
        
        // Update running averages
        stats.avg_comprehension = (stats.avg_comprehension * n + record.comprehension_score) / new_n;
        stats.avg_engagement = (stats.avg_engagement * n + record.engagement_score) / new_n;
        stats.avg_time_to_understanding = (stats.avg_time_to_understanding * n + record.time_to_understanding) / new_n;
        
        // Update correction rate
        let corrections = if record.correction_needed { 1.0 } else { 0.0 };
        stats.correction_rate = (stats.correction_rate * n + corrections) / new_n;
    }
    
    /// Update audience effectiveness
    fn update_audience_effectiveness(&mut self, audience: UserArchetype, record: &EffectivenessRecord) {
        let stats = self.audience_effectiveness
            .entry(audience)
            .or_insert_with(|| AudienceEffectiveness {
                audience,
                ..Default::default()
            });
        
        let n = stats.total_explanations as f64;
        stats.total_explanations += 1;
        let new_n = stats.total_explanations as f64;
        
        // Update running averages
        stats.avg_comprehension = (stats.avg_comprehension * n + record.comprehension_score) / new_n;
        stats.avg_engagement = (stats.avg_engagement * n + record.engagement_score) / new_n;
        
        // Update success rate
        let success = if record.comprehension_score > 0.7 { 1.0 } else { 0.0 };
        stats.success_rate = (stats.success_rate * n + success) / new_n;
    }
    
    /// Get effectiveness statistics for a concept
    pub fn get_effectiveness_stats(&self, concept: &str) -> EffectivenessStats {
        let records: Vec<_> = self.effectiveness_history
            .iter()
            .filter(|r| r.concept == concept)
            .collect();
        
        if records.is_empty() {
            return EffectivenessStats::default();
        }
        
        let total = records.len();
        let avg_comprehension = records.iter()
            .map(|r| r.comprehension_score)
            .sum::<f64>() / total as f64;
        
        let avg_engagement = records.iter()
            .map(|r| r.engagement_score)
            .sum::<f64>() / total as f64;
        
        let correction_rate = records.iter()
            .filter(|r| r.correction_needed)
            .count() as f64 / total as f64;
        
        let avg_time = records.iter()
            .map(|r| r.time_to_understanding)
            .sum::<f64>() / total as f64;
        
        let success_rate = records.iter()
            .filter(|r| r.comprehension_score > 0.7)
            .count() as f64 / total as f64;
        
        EffectivenessStats {
            total_explanations: total,
            avg_comprehension,
            avg_engagement,
            correction_rate,
            avg_time_to_understanding: avg_time,
            success_rate,
        }
    }
    
    /// Get overall effectiveness statistics
    pub fn get_overall_stats(&self) -> EffectivenessStats {
        if self.effectiveness_history.is_empty() {
            return EffectivenessStats::default();
        }
        
        let total = self.effectiveness_history.len();
        let avg_comprehension = self.effectiveness_history.iter()
            .map(|r| r.comprehension_score)
            .sum::<f64>() / total as f64;
        
        let avg_engagement = self.effectiveness_history.iter()
            .map(|r| r.engagement_score)
            .sum::<f64>() / total as f64;
        
        let correction_rate = self.effectiveness_history.iter()
            .filter(|r| r.correction_needed)
            .count() as f64 / total as f64;
        
        let avg_time = self.effectiveness_history.iter()
            .map(|r| r.time_to_understanding)
            .sum::<f64>() / total as f64;
        
        let success_rate = self.effectiveness_history.iter()
            .filter(|r| r.comprehension_score > 0.7)
            .count() as f64 / total as f64;
        
        EffectivenessStats {
            total_explanations: total,
            avg_comprehension,
            avg_engagement,
            correction_rate,
            avg_time_to_understanding: avg_time,
            success_rate,
        }
    }
    
    /// Get concept effectiveness
    pub fn get_concept_effectiveness(&self, concept: &str) -> Option<&ConceptEffectiveness> {
        self.concept_effectiveness.get(concept)
    }
    
    /// Get audience effectiveness
    pub fn get_audience_effectiveness(&self, audience: UserArchetype) -> Option<&AudienceEffectiveness> {
        self.audience_effectiveness.get(&audience)
    }
    
    /// Get all concept effectiveness
    pub fn get_all_concept_effectiveness(&self) -> Vec<ConceptEffectiveness> {
        self.concept_effectiveness.values().cloned().collect()
    }
    
    /// Get all audience effectiveness
    pub fn get_all_audience_effectiveness(&self) -> Vec<AudienceEffectiveness> {
        self.audience_effectiveness.values().cloned().collect()
    }
    
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

impl Default for TeachingEffectivenessTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_effectiveness_tracker() {
        let mut tracker = TeachingEffectivenessTracker::new();
        
        let feedback = UserFeedback::from_interaction(
            "addition".to_string(),
            "What is 2+2?".to_string(),
            false,
            10.0,
            true,
        );
        
        tracker.record_outcome(
            "addition",
            UserArchetype::Curious,
            "exp_123",
            &feedback,
        );
        
        let stats = tracker.get_effectiveness_stats("addition");
        assert_eq!(stats.total_explanations, 1);
        assert!(stats.avg_comprehension > 0.0);
    }
    
    #[test]
    fn test_concept_effectiveness() {
        let mut tracker = TeachingEffectivenessTracker::new();
        
        for i in 0..5 {
            let feedback = UserFeedback::from_interaction(
                "math".to_string(),
                format!("Question {}", i),
                i % 2 == 0,
                10.0 + i as f64,
                true,
            );
            
            tracker.record_outcome(
                "math",
                UserArchetype::Analytical,
                &format!("exp_{}", i),
                &feedback,
            );
        }
        
        let concept_eff = tracker.get_concept_effectiveness("math").unwrap();
        assert_eq!(concept_eff.total_explanations, 5);
    }
}
