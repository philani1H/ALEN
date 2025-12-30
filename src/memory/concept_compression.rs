//! Concept Compression and Forgetting
//!
//! Implements controlled forgetting and concept compression:
//! - Replace many specific experiences with smaller, provable rules
//! - Decay episodic memories over time
//! - Preserve verified concepts permanently
//! - Extract invariants and patterns
//!
//! Mathematical Foundation:
//! - Compression: C: K → K̃ where |K̃| << |K|
//! - Retention: R(m) = confidence(m) × usage(m)
//! - Decay: w(t) = w₀ × e^(-λt)
//!
//! Key Principle: Keep truths, discard instances.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// PART 1: CONCEPT MEMORY (PERSISTENT)
// ============================================================================

/// Compressed concept - persistent knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    pub id: String,
    pub concept_type: ConceptType,
    pub rule: String,
    pub confidence: f64,
    pub evidence_count: u32,
    pub last_verified: u64,
    pub usage_count: u32,
    pub invariants: Vec<String>,
    pub proof_skeleton: Option<ProofSkeleton>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConceptType {
    /// General rule extracted from examples
    Rule,
    /// Invariant property that always holds
    Invariant,
    /// Verified theorem with proof
    Theorem,
    /// User preference or trait
    UserTrait,
    /// Abstraction from multiple cases
    Abstraction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofSkeleton {
    /// High-level proof structure
    pub structure: Vec<String>,
    /// Key assumptions
    pub assumptions: Vec<String>,
    /// Transformation steps (abstract)
    pub steps: Vec<String>,
}

impl Concept {
    pub fn new_rule(rule: String, confidence: f64, evidence_count: u32) -> Self {
        Self {
            id: format!("concept_{}", Self::current_timestamp()),
            concept_type: ConceptType::Rule,
            rule,
            confidence,
            evidence_count,
            last_verified: Self::current_timestamp(),
            usage_count: 0,
            invariants: Vec::new(),
            proof_skeleton: None,
        }
    }

    pub fn new_invariant(invariant: String, confidence: f64) -> Self {
        Self {
            id: format!("invariant_{}", Self::current_timestamp()),
            concept_type: ConceptType::Invariant,
            rule: invariant.clone(),
            confidence,
            evidence_count: 1,
            last_verified: Self::current_timestamp(),
            usage_count: 0,
            invariants: vec![invariant],
            proof_skeleton: None,
        }
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

// ============================================================================
// PART 2: EPISODIC MEMORY DECAY
// ============================================================================

/// Manages decay of episodic memories
#[derive(Debug, Clone)]
pub struct MemoryDecay {
    /// Decay rate λ
    pub decay_rate: f64,
    /// Minimum retention threshold
    pub retention_threshold: f64,
    /// Usage boost factor
    pub usage_boost: f64,
}

impl MemoryDecay {
    pub fn new() -> Self {
        Self {
            decay_rate: 0.01,           // 1% per day
            retention_threshold: 0.2,    // Keep if >20% retention
            usage_boost: 0.1,            // 10% boost per use
        }
    }

    /// Calculate retention score: R(m) = confidence(m) × usage(m) × decay(t)
    pub fn retention_score(
        &self,
        confidence: f64,
        usage_count: u32,
        days_since_last_use: f64,
    ) -> f64 {
        let usage_factor = 1.0 + (usage_count as f64 * self.usage_boost);
        let decay_factor = (-self.decay_rate * days_since_last_use).exp();
        
        confidence * usage_factor * decay_factor
    }

    /// Check if memory should be kept
    pub fn should_keep(&self, retention: f64) -> bool {
        retention >= self.retention_threshold
    }

    /// Check if memory should be compressed into concept
    pub fn should_compress(&self, retention: f64, evidence_count: u32) -> bool {
        retention > 0.7 && evidence_count >= 3
    }
}

// ============================================================================
// PART 3: CONCEPT EXTRACTION
// ============================================================================

/// Extracts concepts from episodic memories
pub struct ConceptExtractor {
    min_evidence: u32,
    min_confidence: f64,
}

impl ConceptExtractor {
    pub fn new() -> Self {
        Self {
            min_evidence: 3,
            min_confidence: 0.75,
        }
    }

    /// Extract rule from multiple similar episodes
    pub fn extract_rule(&self, episodes: &[EpisodeData]) -> Option<Concept> {
        if episodes.len() < self.min_evidence as usize {
            return None;
        }

        // Find common pattern
        let pattern = self.find_common_pattern(episodes);
        if pattern.is_empty() {
            return None;
        }

        // Calculate average confidence
        let avg_confidence: f64 = episodes.iter()
            .map(|e| e.confidence)
            .sum::<f64>() / episodes.len() as f64;

        if avg_confidence < self.min_confidence {
            return None;
        }

        // Create rule
        let rule = format!("When {}, then {}", pattern, self.extract_outcome(episodes));
        
        Some(Concept::new_rule(rule, avg_confidence, episodes.len() as u32))
    }

    /// Find invariant across episodes
    pub fn extract_invariant(&self, episodes: &[EpisodeData]) -> Option<Concept> {
        // Look for properties that hold across all episodes
        let invariant = self.find_invariant_property(episodes)?;
        
        Some(Concept::new_invariant(invariant, 0.9))
    }

    fn find_common_pattern(&self, episodes: &[EpisodeData]) -> String {
        // Simple implementation: find common words in inputs
        let words: Vec<Vec<&str>> = episodes.iter()
            .map(|e| e.input.split_whitespace().collect())
            .collect();

        if words.is_empty() {
            return String::new();
        }

        // Find words that appear in all episodes
        let first = &words[0];
        let common: Vec<&str> = first.iter()
            .filter(|word| words.iter().all(|w| w.contains(word)))
            .copied()
            .collect();

        common.join(" ")
    }

    fn extract_outcome(&self, episodes: &[EpisodeData]) -> String {
        // Use most common outcome
        let mut outcomes: HashMap<String, usize> = HashMap::new();
        for episode in episodes {
            *outcomes.entry(episode.output.clone()).or_insert(0) += 1;
        }

        outcomes.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(outcome, _)| outcome)
            .unwrap_or_default()
    }

    fn find_invariant_property(&self, episodes: &[EpisodeData]) -> Option<String> {
        // Check if all episodes share a property
        if episodes.is_empty() {
            return None;
        }

        // Example: Check if all have same domain
        let first_domain = &episodes[0].domain;
        if episodes.iter().all(|e| &e.domain == first_domain) {
            return Some(format!("Domain: {}", first_domain));
        }

        None
    }
}

#[derive(Debug, Clone)]
pub struct EpisodeData {
    pub input: String,
    pub output: String,
    pub confidence: f64,
    pub domain: String,
    pub timestamp: u64,
    pub usage_count: u32,
}

// ============================================================================
// PART 4: COMPRESSION MANAGER
// ============================================================================

/// Manages concept compression and forgetting
pub struct CompressionManager {
    /// Persistent concepts
    pub concepts: HashMap<String, Concept>,
    /// Decay manager
    pub decay: MemoryDecay,
    /// Concept extractor
    pub extractor: ConceptExtractor,
    /// Protected concepts (never forget)
    pub protected: Vec<String>,
}

impl CompressionManager {
    pub fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            decay: MemoryDecay::new(),
            extractor: ConceptExtractor::new(),
            protected: Vec::new(),
        }
    }

    /// Compress episodes into concepts
    pub fn compress(&mut self, episodes: Vec<EpisodeData>) -> CompressionResult {
        let mut compressed_count = 0;
        let mut kept_count = 0;
        let mut forgotten_count = 0;

        // Group similar episodes
        let groups = self.group_similar_episodes(&episodes);

        for group in groups {
            // Try to extract concept
            if let Some(concept) = self.extractor.extract_rule(&group) {
                self.concepts.insert(concept.id.clone(), concept);
                compressed_count += group.len();
            } else {
                // Check retention for each episode
                for episode in group {
                    let days_since = self.days_since(episode.timestamp);
                    let retention = self.decay.retention_score(
                        episode.confidence,
                        episode.usage_count,
                        days_since,
                    );

                    if self.decay.should_keep(retention) {
                        kept_count += 1;
                    } else {
                        forgotten_count += 1;
                    }
                }
            }
        }

        CompressionResult {
            compressed_count,
            kept_count,
            forgotten_count,
            concepts_created: self.concepts.len(),
        }
    }

    /// Apply forgetting to episodic memories
    pub fn apply_forgetting(&self, episodes: &[EpisodeData]) -> Vec<EpisodeData> {
        episodes.iter()
            .filter(|episode| {
                let days_since = self.days_since(episode.timestamp);
                let retention = self.decay.retention_score(
                    episode.confidence,
                    episode.usage_count,
                    days_since,
                );
                self.decay.should_keep(retention)
            })
            .cloned()
            .collect()
    }

    /// Get concept by query
    pub fn query_concept(&mut self, query: &str) -> Option<&Concept> {
        // Simple matching - in production use embeddings
        for concept in self.concepts.values_mut() {
            if concept.rule.contains(query) {
                concept.usage_count += 1;
                return Some(concept);
            }
        }
        None
    }

    /// Protect concept from forgetting
    pub fn protect(&mut self, concept_id: String) {
        if !self.protected.contains(&concept_id) {
            self.protected.push(concept_id);
        }
    }

    fn group_similar_episodes(&self, episodes: &[EpisodeData]) -> Vec<Vec<EpisodeData>> {
        // Simple grouping by domain
        let mut groups: HashMap<String, Vec<EpisodeData>> = HashMap::new();
        
        for episode in episodes {
            groups.entry(episode.domain.clone())
                .or_insert_with(Vec::new)
                .push(episode.clone());
        }

        groups.into_values().collect()
    }

    fn days_since(&self, timestamp: u64) -> f64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        (now - timestamp) as f64 / 86400.0
    }

    /// Get statistics
    pub fn stats(&self) -> CompressionStats {
        CompressionStats {
            total_concepts: self.concepts.len(),
            protected_concepts: self.protected.len(),
            avg_confidence: if !self.concepts.is_empty() {
                self.concepts.values().map(|c| c.confidence).sum::<f64>() 
                    / self.concepts.len() as f64
            } else {
                0.0
            },
            total_evidence: self.concepts.values().map(|c| c.evidence_count).sum(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionResult {
    pub compressed_count: usize,
    pub kept_count: usize,
    pub forgotten_count: usize,
    pub concepts_created: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub total_concepts: usize,
    pub protected_concepts: usize,
    pub avg_confidence: f64,
    pub total_evidence: u32,
}

// ============================================================================
// PART 5: NEVER FORGET (PROTECTED KNOWLEDGE)
// ============================================================================

/// Knowledge that should never be forgotten
#[derive(Debug, Clone)]
pub struct ProtectedKnowledge {
    /// Axioms
    pub axioms: Vec<String>,
    /// Validated rules
    pub validated_rules: Vec<String>,
    /// High-confidence concepts
    pub high_confidence: Vec<String>,
    /// User-confirmed preferences
    pub user_preferences: Vec<String>,
}

impl ProtectedKnowledge {
    pub fn new() -> Self {
        Self {
            axioms: Vec::new(),
            validated_rules: Vec::new(),
            high_confidence: Vec::new(),
            user_preferences: Vec::new(),
        }
    }

    pub fn is_protected(&self, concept_id: &str) -> bool {
        self.axioms.contains(&concept_id.to_string())
            || self.validated_rules.contains(&concept_id.to_string())
            || self.high_confidence.contains(&concept_id.to_string())
            || self.user_preferences.contains(&concept_id.to_string())
    }

    pub fn add_axiom(&mut self, concept_id: String) {
        if !self.axioms.contains(&concept_id) {
            self.axioms.push(concept_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_decay() {
        let decay = MemoryDecay::new();
        
        // High confidence, recent, frequently used
        let retention1 = decay.retention_score(0.9, 10, 1.0);
        assert!(retention1 > 0.7);
        
        // Low confidence, old, very rarely used (0 uses, very old)
        let retention2 = decay.retention_score(0.1, 0, 100.0);
        // With decay_rate=0.01, 100 days -> decay_factor = exp(-1.0) ≈ 0.37
        // retention = 0.1 * 1.0 * 0.37 ≈ 0.037 < 0.2 threshold
        assert!(retention2 < 0.2);
        
        // Should keep high retention
        assert!(decay.should_keep(retention1));
        
        // Should forget low retention
        assert!(!decay.should_keep(retention2));
    }

    #[test]
    fn test_concept_extraction() {
        let extractor = ConceptExtractor::new();
        
        let episodes = vec![
            EpisodeData {
                input: "work stress problem".to_string(),
                output: "time blocking helps".to_string(),
                confidence: 0.8,
                domain: "career".to_string(),
                timestamp: 0,
                usage_count: 5,
            },
            EpisodeData {
                input: "work stress issue".to_string(),
                output: "time blocking helps".to_string(),
                confidence: 0.85,
                domain: "career".to_string(),
                timestamp: 0,
                usage_count: 3,
            },
            EpisodeData {
                input: "work stress challenge".to_string(),
                output: "time blocking helps".to_string(),
                confidence: 0.9,
                domain: "career".to_string(),
                timestamp: 0,
                usage_count: 7,
            },
        ];

        let concept = extractor.extract_rule(&episodes);
        assert!(concept.is_some());
        
        let concept = concept.unwrap();
        assert!(concept.confidence > 0.8);
        assert_eq!(concept.evidence_count, 3);
    }

    #[test]
    fn test_compression_manager() {
        let mut manager = CompressionManager::new();
        
        let episodes = vec![
            EpisodeData {
                input: "test".to_string(),
                output: "result".to_string(),
                confidence: 0.9,
                domain: "test".to_string(),
                timestamp: 0,
                usage_count: 1,
            },
        ];

        let result = manager.compress(episodes);
        assert!(result.compressed_count > 0 || result.kept_count > 0 || result.forgotten_count > 0);
    }
}
