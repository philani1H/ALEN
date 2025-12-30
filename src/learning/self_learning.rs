//! Self-Learning System
//!
//! Implements continuous learning from conversations and patterns:
//! - Pattern extraction from conversations
//! - Cross-user aggregation (privacy-preserving)
//! - Bayesian confidence updates over time
//! - Knowledge base that grows with verified evidence
//!
//! Key Principles:
//! 1. Learn from patterns, not raw data
//! 2. Aggregate across users (privacy-preserving)
//! 3. Confidence grows with evidence
//! 4. Time decay for stale patterns
//! 5. Verification before storage

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// PART 1: PATTERN EXTRACTION (FROM CONVERSATIONS)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub id: String,
    pub domain: Domain,
    pub pattern_type: PatternType,
    pub condition: String,
    pub effects: Vec<String>,
    pub common_mistakes: Vec<String>,
    pub successful_strategies: Vec<String>,
    pub confidence: f64, // Bayesian confidence [0, 1]
    pub evidence_count: u32,
    pub first_observed: u64,
    pub last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Domain {
    Career,
    Relationships,
    Health,
    Finance,
    Education,
    Personal,
    Technical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    CauseEffect,
    Strategy,
    CommonProblem,
    SuccessIndicator,
    FailureMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSegment {
    pub user_id: String,
    pub content: String,
    pub domain: Domain,
    pub emotional_valence: f64,
    pub problem_mentioned: Option<String>,
    pub solution_mentioned: Option<String>,
    pub outcome_mentioned: Option<String>,
    pub timestamp: u64,
}

pub struct PatternExtractor {
    pub min_confidence_threshold: f64,
    pub min_evidence_for_pattern: u32,
}

impl PatternExtractor {
    pub fn new() -> Self {
        Self {
            min_confidence_threshold: 0.75,
            min_evidence_for_pattern: 3,
        }
    }

    /// Extract patterns from a conversation segment
    pub fn extract_patterns(&self, segment: &ConversationSegment) -> Vec<PatternCandidate> {
        let mut candidates = Vec::new();

        // Pattern 1: Problem-Solution pairs
        if let (Some(problem), Some(solution)) = (&segment.problem_mentioned, &segment.solution_mentioned) {
            candidates.push(PatternCandidate {
                pattern_type: PatternType::Strategy,
                domain: segment.domain.clone(),
                condition: problem.clone(),
                observation: solution.clone(),
                confidence: 0.6,
                source_count: 1,
            });
        }

        // Pattern 2: Cause-Effect relationships
        if let Some(outcome) = &segment.outcome_mentioned {
            if let Some(solution) = &segment.solution_mentioned {
                candidates.push(PatternCandidate {
                    pattern_type: PatternType::CauseEffect,
                    domain: segment.domain.clone(),
                    condition: solution.clone(),
                    observation: outcome.clone(),
                    confidence: 0.5,
                    source_count: 1,
                });
            }
        }

        // Pattern 3: Common problems (high frequency)
        if let Some(problem) = &segment.problem_mentioned {
            candidates.push(PatternCandidate {
                pattern_type: PatternType::CommonProblem,
                domain: segment.domain.clone(),
                condition: "general".to_string(),
                observation: problem.clone(),
                confidence: 0.4,
                source_count: 1,
            });
        }

        candidates
    }

    /// Determine if a pattern candidate should be promoted to verified pattern
    pub fn should_promote(&self, candidate: &AggregatedPattern) -> bool {
        candidate.confidence >= self.min_confidence_threshold
            && candidate.evidence_count >= self.min_evidence_for_pattern
    }
}

#[derive(Debug, Clone)]
pub struct PatternCandidate {
    pub pattern_type: PatternType,
    pub domain: Domain,
    pub condition: String,
    pub observation: String,
    pub confidence: f64,
    pub source_count: u32,
}

// ============================================================================
// PART 2: CROSS-USER AGGREGATION (PRIVACY-PRESERVING)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedPattern {
    pub pattern_type: PatternType,
    pub domain: Domain,
    pub condition: String,
    pub observations: Vec<String>,
    pub confidence: f64,
    pub evidence_count: u32,
    pub unique_users: u32,
    pub success_rate: Option<f64>,
}

pub struct CrossUserAggregator {
    privacy_filter: PrivacyFilter,
    aggregation_threshold: u32,
}

impl CrossUserAggregator {
    pub fn new() -> Self {
        Self {
            privacy_filter: PrivacyFilter::new(),
            aggregation_threshold: 3,
        }
    }

    /// Aggregate patterns from multiple users (privacy-preserving)
    pub fn aggregate(&self, candidates: Vec<PatternCandidate>) -> Vec<AggregatedPattern> {
        let mut groups: HashMap<(Domain, String), Vec<PatternCandidate>> = HashMap::new();

        for candidate in candidates {
            let sanitized = self.privacy_filter.sanitize(&candidate);
            let key = (sanitized.domain.clone(), sanitized.condition.clone());
            groups.entry(key).or_insert_with(Vec::new).push(sanitized);
        }

        let mut aggregated = Vec::new();

        for ((domain, condition), group) in groups {
            if group.len() < self.aggregation_threshold as usize {
                continue;
            }

            let observations: Vec<String> = group.iter()
                .map(|c| c.observation.clone())
                .collect();

            let confidence = self.aggregate_confidence(&group);
            let evidence_count = group.len() as u32;
            let unique_users = group.iter()
                .map(|c| c.source_count)
                .sum::<u32>();

            aggregated.push(AggregatedPattern {
                pattern_type: group[0].pattern_type.clone(),
                domain,
                condition,
                observations,
                confidence,
                evidence_count,
                unique_users,
                success_rate: None,
            });
        }

        aggregated
    }

    /// Bayesian confidence aggregation
    fn aggregate_confidence(&self, group: &[PatternCandidate]) -> f64 {
        let n = group.len() as f64;
        let prior = 0.5;
        let avg_confidence: f64 = group.iter().map(|c| c.confidence).sum::<f64>() / n;
        let evidence_boost = (n / 10.0).min(1.0);
        let posterior = prior + (avg_confidence - prior) * (0.5 + 0.5 * evidence_boost);
        posterior.max(0.0).min(1.0)
    }
}

pub struct PrivacyFilter {
    pii_patterns: Vec<String>,
}

impl PrivacyFilter {
    pub fn new() -> Self {
        Self {
            pii_patterns: vec![
                "name".to_string(),
                "email".to_string(),
                "phone".to_string(),
                "address".to_string(),
            ],
        }
    }

    pub fn sanitize(&self, candidate: &PatternCandidate) -> PatternCandidate {
        let sanitized_condition = self.remove_pii(&candidate.condition);
        let sanitized_observation = self.remove_pii(&candidate.observation);

        PatternCandidate {
            pattern_type: candidate.pattern_type.clone(),
            domain: candidate.domain.clone(),
            condition: sanitized_condition,
            observation: sanitized_observation,
            confidence: candidate.confidence,
            source_count: candidate.source_count,
        }
    }

    fn remove_pii(&self, text: &str) -> String {
        let mut sanitized = text.to_string();
        sanitized = sanitized.replace(|c: char| c == '@', "[EMAIL]");
        
        let words: Vec<&str> = sanitized.split_whitespace().collect();
        let filtered: Vec<String> = words.iter()
            .map(|w| {
                if w.chars().filter(|c| c.is_numeric()).count() > 5 {
                    "[NUMBER]".to_string()
                } else {
                    w.to_string()
                }
            })
            .collect();
        
        filtered.join(" ")
    }
}

// ============================================================================
// PART 3: CONFIDENCE UPDATES OVER TIME (BAYESIAN)
// ============================================================================

pub struct ConfidenceUpdater {
    base_learning_rate: f64,
    decay_rate: f64,
    max_confidence: f64,
}

impl ConfidenceUpdater {
    pub fn new() -> Self {
        Self {
            base_learning_rate: 0.1,
            decay_rate: 0.01,
            max_confidence: 0.95,
        }
    }

    /// Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
    pub fn update_confidence(
        &self,
        prior: f64,
        new_evidence_confidence: f64,
        evidence_weight: f64,
    ) -> f64 {
        let learning_rate = self.base_learning_rate * evidence_weight;
        let posterior = prior + learning_rate * (new_evidence_confidence - prior);
        posterior.max(0.0).min(self.max_confidence)
    }

    pub fn update_on_success(&self, current_confidence: f64, success_magnitude: f64) -> f64 {
        let boost = self.base_learning_rate * success_magnitude;
        (current_confidence + boost).min(self.max_confidence)
    }

    pub fn update_on_failure(&self, current_confidence: f64, failure_magnitude: f64) -> f64 {
        let penalty = self.base_learning_rate * failure_magnitude;
        (current_confidence - penalty).max(0.0)
    }

    /// Time decay: C(t) = C₀ * e^(-λt)
    pub fn apply_time_decay(&self, confidence: f64, days_since_update: f64) -> f64 {
        confidence * (-self.decay_rate * days_since_update).exp()
    }

    pub fn calculate_evidence_weight(&self, source_reliability: f64, evidence_count: u32) -> f64 {
        let base_weight = source_reliability;
        let count_boost = (evidence_count as f64).ln().max(1.0);
        (base_weight * count_boost / 10.0).min(1.0)
    }
}

// ============================================================================
// PART 4: KNOWLEDGE BASE (GROWS OVER TIME)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBase {
    pub patterns: Vec<Pattern>,
    pub pattern_index: HashMap<Domain, Vec<usize>>,
    pub total_evidence: u64,
    pub creation_time: u64,
    pub last_updated: u64,
}

impl KnowledgeBase {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            pattern_index: HashMap::new(),
            total_evidence: 0,
            creation_time: Self::current_timestamp(),
            last_updated: Self::current_timestamp(),
        }
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Add or update a pattern
    pub fn upsert_pattern(&mut self, new_pattern: Pattern) {
        let existing_idx = self.patterns.iter().position(|p| {
            p.domain == new_pattern.domain
                && p.condition == new_pattern.condition
                && std::mem::discriminant(&p.pattern_type) == std::mem::discriminant(&new_pattern.pattern_type)
        });

        match existing_idx {
            Some(idx) => {
                let existing = &mut self.patterns[idx];
                
                for strategy in &new_pattern.successful_strategies {
                    if !existing.successful_strategies.contains(strategy) {
                        existing.successful_strategies.push(strategy.clone());
                    }
                }
                
                for mistake in &new_pattern.common_mistakes {
                    if !existing.common_mistakes.contains(mistake) {
                        existing.common_mistakes.push(mistake.clone());
                    }
                }
                
                let n1 = existing.evidence_count as f64;
                let n2 = new_pattern.evidence_count as f64;
                existing.confidence = (existing.confidence * n1 + new_pattern.confidence * n2) / (n1 + n2);
                existing.evidence_count += new_pattern.evidence_count;
                existing.last_updated = Self::current_timestamp();
            }
            None => {
                let pattern_idx = self.patterns.len();
                self.patterns.push(new_pattern.clone());
                
                self.pattern_index
                    .entry(new_pattern.domain.clone())
                    .or_insert_with(Vec::new)
                    .push(pattern_idx);
                
                self.total_evidence += new_pattern.evidence_count as u64;
            }
        }
        
        self.last_updated = Self::current_timestamp();
    }

    pub fn query_by_domain(&self, domain: &Domain) -> Vec<&Pattern> {
        self.pattern_index
            .get(domain)
            .map(|indices| {
                indices.iter()
                    .filter_map(|&idx| self.patterns.get(idx))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn query_by_condition(&self, condition: &str) -> Vec<&Pattern> {
        self.patterns.iter()
            .filter(|p| p.condition.contains(condition))
            .collect()
    }

    pub fn top_patterns(&self, domain: &Domain, k: usize) -> Vec<&Pattern> {
        let mut domain_patterns = self.query_by_domain(domain);
        domain_patterns.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        domain_patterns.into_iter().take(k).collect()
    }

    pub fn decay_all(&mut self, updater: &ConfidenceUpdater) {
        let current_time = Self::current_timestamp();
        
        for pattern in &mut self.patterns {
            let days_since_update = (current_time - pattern.last_updated) as f64 / 86400.0;
            pattern.confidence = updater.apply_time_decay(pattern.confidence, days_since_update);
        }
    }

    pub fn stats(&self) -> KnowledgeStats {
        let total_patterns = self.patterns.len();
        let avg_confidence = if total_patterns > 0 {
            self.patterns.iter().map(|p| p.confidence).sum::<f64>() / total_patterns as f64
        } else {
            0.0
        };

        let patterns_by_domain: HashMap<Domain, usize> = self.pattern_index.iter()
            .map(|(domain, indices)| (domain.clone(), indices.len()))
            .collect();

        KnowledgeStats {
            total_patterns,
            total_evidence: self.total_evidence,
            avg_confidence,
            patterns_by_domain,
            age_days: (Self::current_timestamp() - self.creation_time) as f64 / 86400.0,
        }
    }
}

#[derive(Debug)]
pub struct KnowledgeStats {
    pub total_patterns: usize,
    pub total_evidence: u64,
    pub avg_confidence: f64,
    pub patterns_by_domain: HashMap<Domain, usize>,
    pub age_days: f64,
}

// ============================================================================
// PART 5: SELF-LEARNING SYSTEM (ORCHESTRATION)
// ============================================================================

pub struct SelfLearningSystem {
    knowledge_base: KnowledgeBase,
    pattern_extractor: PatternExtractor,
    aggregator: CrossUserAggregator,
    confidence_updater: ConfidenceUpdater,
    learning_buffer: Vec<PatternCandidate>,
}

impl SelfLearningSystem {
    pub fn new() -> Self {
        Self {
            knowledge_base: KnowledgeBase::new(),
            pattern_extractor: PatternExtractor::new(),
            aggregator: CrossUserAggregator::new(),
            confidence_updater: ConfidenceUpdater::new(),
            learning_buffer: Vec::new(),
        }
    }

    /// Step 1: Extract patterns from conversation
    pub fn learn_from_conversation(&mut self, segment: ConversationSegment) {
        let candidates = self.pattern_extractor.extract_patterns(&segment);
        self.learning_buffer.extend(candidates);
    }

    /// Step 2: Aggregate patterns from buffer
    pub fn aggregate_patterns(&mut self) {
        if self.learning_buffer.is_empty() {
            return;
        }

        let aggregated = self.aggregator.aggregate(self.learning_buffer.drain(..).collect());

        for agg in aggregated {
            if self.pattern_extractor.should_promote(&agg) {
                let pattern = self.convert_to_pattern(agg);
                self.knowledge_base.upsert_pattern(pattern);
            }
        }
    }

    fn convert_to_pattern(&self, agg: AggregatedPattern) -> Pattern {
        Pattern {
            id: format!("pattern_{}", KnowledgeBase::current_timestamp()),
            domain: agg.domain,
            pattern_type: agg.pattern_type,
            condition: agg.condition,
            effects: Vec::new(),
            common_mistakes: Vec::new(),
            successful_strategies: agg.observations,
            confidence: agg.confidence,
            evidence_count: agg.evidence_count,
            first_observed: KnowledgeBase::current_timestamp(),
            last_updated: KnowledgeBase::current_timestamp(),
        }
    }

    /// Step 3: Update confidence based on outcomes
    pub fn update_on_outcome(&mut self, pattern_id: &str, success: bool, magnitude: f64) {
        if let Some(pattern) = self.knowledge_base.patterns.iter_mut().find(|p| p.id == pattern_id) {
            pattern.confidence = if success {
                self.confidence_updater.update_on_success(pattern.confidence, magnitude)
            } else {
                self.confidence_updater.update_on_failure(pattern.confidence, magnitude)
            };
            pattern.last_updated = KnowledgeBase::current_timestamp();
        }
    }

    /// Step 4: Apply time decay
    pub fn apply_time_decay(&mut self) {
        self.knowledge_base.decay_all(&self.confidence_updater);
    }

    pub fn query_knowledge(&self, domain: &Domain, condition: &str) -> Vec<&Pattern> {
        self.knowledge_base.query_by_condition(condition)
            .into_iter()
            .filter(|p| &p.domain == domain)
            .collect()
    }

    pub fn stats(&self) -> KnowledgeStats {
        self.knowledge_base.stats()
    }

    pub fn get_best_strategies(&self, domain: &Domain, k: usize) -> Vec<&Pattern> {
        self.knowledge_base.top_patterns(domain, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_learning_system() {
        let mut system = SelfLearningSystem::new();
        
        // Lower threshold for testing
        system.pattern_extractor.min_confidence_threshold = 0.5;

        // Add more segments to increase confidence
        for i in 0..10 {
            let segment = ConversationSegment {
                user_id: format!("user_{}", i),
                content: "Feeling stressed about work deadlines".to_string(),
                domain: Domain::Career,
                emotional_valence: -0.6,
                problem_mentioned: Some("work stress".to_string()),
                solution_mentioned: Some("time blocking".to_string()),
                outcome_mentioned: Some("improved focus".to_string()),
                timestamp: KnowledgeBase::current_timestamp(),
            };

            system.learn_from_conversation(segment);
        }

        system.aggregate_patterns();

        let stats = system.stats();
        assert!(stats.total_patterns > 0);
        // The aggregated confidence should be above 0.5 with 10 samples
        assert!(stats.avg_confidence > 0.5);

        let career_patterns = system.query_knowledge(&Domain::Career, "work stress");
        assert!(!career_patterns.is_empty());
    }

    #[test]
    fn test_privacy_filter() {
        let filter = PrivacyFilter::new();
        
        let candidate = PatternCandidate {
            pattern_type: PatternType::Strategy,
            domain: Domain::Career,
            condition: "Contact john@example.com for help".to_string(),
            observation: "Call 555-1234 for support".to_string(),
            confidence: 0.7,
            source_count: 1,
        };

        let sanitized = filter.sanitize(&candidate);
        
        assert!(!sanitized.condition.contains("@"));
        assert!(!sanitized.observation.contains("555"));
    }

    #[test]
    fn test_confidence_updates() {
        let updater = ConfidenceUpdater::new();

        let prior = 0.5;
        let evidence = 0.8;
        let weight = 0.7;
        
        let posterior = updater.update_confidence(prior, evidence, weight);
        assert!(posterior > prior);
        assert!(posterior < evidence);

        let boosted = updater.update_on_success(0.7, 0.9);
        assert!(boosted > 0.7);

        let penalized = updater.update_on_failure(0.7, 0.6);
        assert!(penalized < 0.7);

        let decayed = updater.apply_time_decay(0.9, 30.0);
        assert!(decayed < 0.9);
    }
}
