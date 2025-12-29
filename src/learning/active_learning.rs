//! Active Learning System
//!
//! Implements human-like learning through:
//! 1. Active recall - prove understanding by reconstruction
//! 2. Context inference - understand what conversations are REALLY about
//! 3. Reasoning - derive new knowledge from existing patterns
//! 4. Independent problem solving - figure things out without prompts

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// PART 1: ACTIVE RECALL (PROVE UNDERSTANDING)
// ============================================================================

/// Active recall forces the system to reconstruct knowledge without prompts
/// This proves understanding, not just storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallChallenge {
    pub id: String,
    pub pattern_id: String,
    pub challenge_type: ChallengeType,
    pub question: String,
    pub expected_reconstruction: String,
    pub difficulty: f64, // [0, 1]
    pub last_attempted: Option<u64>,
    pub success_count: u32,
    pub failure_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeType {
    RecallCondition,      // Given effect, recall cause
    RecallEffect,         // Given cause, recall effect
    RecallStrategy,       // Given problem, recall solution
    InferRelationship,    // Given A and B, infer connection
    TransferKnowledge,    // Apply pattern to new domain
}

pub struct ActiveRecallEngine {
    spaced_repetition: SpacedRepetition,
    min_understanding_score: f64,
}

impl ActiveRecallEngine {
    pub fn new() -> Self {
        Self {
            spaced_repetition: SpacedRepetition::new(),
            min_understanding_score: 0.7,
        }
    }

    /// Generate a recall challenge for a pattern
    pub fn generate_challenge(&self, pattern: &Pattern) -> RecallChallenge {
        let challenge_type = self.select_challenge_type(&pattern.pattern_type);
        
        let (question, expected) = match challenge_type {
            ChallengeType::RecallCondition => {
                let effect = pattern.effects.first().unwrap_or(&pattern.condition);
                (
                    format!("What causes: '{}'?", effect),
                    pattern.condition.clone(),
                )
            }
            ChallengeType::RecallEffect => {
                (
                    format!("What happens when: '{}'?", pattern.condition),
                    pattern.effects.join(", "),
                )
            }
            ChallengeType::RecallStrategy => {
                (
                    format!("How to handle: '{}'?", pattern.condition),
                    pattern.successful_strategies.join(", "),
                )
            }
            _ => (String::new(), String::new()),
        };

        RecallChallenge {
            id: format!("challenge_{}", Self::current_timestamp()),
            pattern_id: pattern.id.clone(),
            challenge_type,
            question,
            expected_reconstruction: expected,
            difficulty: self.calculate_difficulty(pattern),
            last_attempted: None,
            success_count: 0,
            failure_count: 0,
        }
    }

    fn select_challenge_type(&self, pattern_type: &PatternType) -> ChallengeType {
        match pattern_type {
            PatternType::CauseEffect => ChallengeType::RecallEffect,
            PatternType::Strategy => ChallengeType::RecallStrategy,
            _ => ChallengeType::RecallCondition,
        }
    }

    fn calculate_difficulty(&self, pattern: &Pattern) -> f64 {
        let evidence_strength = (pattern.evidence_count as f64).ln() / 10.0;
        (1.0 - pattern.confidence * evidence_strength).max(0.2).min(1.0)
    }

    /// Verify reconstruction using semantic similarity
    pub fn verify_reconstruction(&self, reconstructed: &str, expected: &str) -> f64 {
        let reconstructed_tokens: Vec<&str> = reconstructed.split_whitespace().collect();
        let expected_tokens: Vec<&str> = expected.split_whitespace().collect();
        
        let overlap = reconstructed_tokens.iter()
            .filter(|t| expected_tokens.contains(t))
            .count();
        
        let precision = overlap as f64 / reconstructed_tokens.len().max(1) as f64;
        let recall = overlap as f64 / expected_tokens.len().max(1) as f64;
        
        // F1 score
        if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        }
    }

    /// Update pattern confidence based on recall success
    pub fn update_understanding(&self, 
                                 pattern_confidence: f64, 
                                 recall_score: f64,
                                 difficulty: f64) -> f64 {
        let boost = recall_score * difficulty * 0.15;
        (pattern_confidence + boost).min(0.95)
    }

    /// Calculate when to test again (spaced repetition)
    pub fn next_review_interval(&self, challenge: &RecallChallenge) -> u64 {
        self.spaced_repetition.calculate_interval(
            challenge.success_count,
            challenge.failure_count,
            challenge.difficulty,
        )
    }

    /// Calculate understanding score for a pattern
    pub fn understanding_score(&self, challenge: &RecallChallenge, pattern_confidence: f64) -> f64 {
        let total_attempts = challenge.success_count + challenge.failure_count;
        if total_attempts == 0 {
            return 0.0;
        }
        
        let success_rate = challenge.success_count as f64 / total_attempts as f64;
        success_rate * pattern_confidence
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

/// Spaced repetition algorithm (similar to Anki/SuperMemo)
pub struct SpacedRepetition {
    base_interval: u64,
    multiplier: f64,
    difficulty_factor: f64,
}

impl SpacedRepetition {
    pub fn new() -> Self {
        Self {
            base_interval: 86400,
            multiplier: 2.5,
            difficulty_factor: 0.8,
        }
    }

    pub fn calculate_interval(&self, 
                               success_count: u32, 
                               failure_count: u32,
                               difficulty: f64) -> u64 {
        if success_count == 0 {
            return self.base_interval;
        }
        
        let ease = 2.5 - (failure_count as f64 * 0.2);
        let ease = ease.max(1.3);
        
        let interval = self.base_interval as f64 
                     * ease.powi(success_count as i32)
                     * (1.0 + difficulty * self.difficulty_factor);
        
        interval as u64
    }
}

// ============================================================================
// PART 2: CONTEXT INFERENCE
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationContext {
    pub explicit_topic: Option<String>,
    pub inferred_topic: Vec<String>,
    pub emotional_state: EmotionalState,
    pub underlying_need: Option<UnderlyingNeed>,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub valence: f64,
    pub arousal: f64,
    pub dominant_emotion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnderlyingNeed {
    Validation,
    Guidance,
    Venting,
    ProblemSolving,
    Connection,
    Information,
}

pub struct ContextInferenceEngine {
    domain_signals: HashMap<Domain, Vec<String>>,
    emotional_lexicon: HashMap<String, (f64, f64)>,
}

impl ContextInferenceEngine {
    pub fn new() -> Self {
        let mut emotional_lexicon = HashMap::new();
        
        emotional_lexicon.insert("stressed".to_string(), (-0.7, 0.8));
        emotional_lexicon.insert("anxious".to_string(), (-0.6, 0.9));
        emotional_lexicon.insert("happy".to_string(), (0.8, 0.6));
        emotional_lexicon.insert("calm".to_string(), (0.5, 0.2));
        emotional_lexicon.insert("excited".to_string(), (0.7, 0.9));
        emotional_lexicon.insert("confused".to_string(), (-0.4, 0.5));
        emotional_lexicon.insert("sad".to_string(), (-0.7, 0.3));
        emotional_lexicon.insert("worried".to_string(), (-0.6, 0.7));
        
        Self {
            domain_signals: Self::build_domain_signals(),
            emotional_lexicon,
        }
    }

    fn build_domain_signals() -> HashMap<Domain, Vec<String>> {
        let mut signals = HashMap::new();
        
        signals.insert(Domain::Career, vec![
            "job".to_string(),
            "work".to_string(),
            "boss".to_string(),
            "career".to_string(),
            "promotion".to_string(),
            "deadline".to_string(),
        ]);
        
        signals.insert(Domain::Relationships, vec![
            "partner".to_string(),
            "relationship".to_string(),
            "dating".to_string(),
            "love".to_string(),
            "breakup".to_string(),
            "trust".to_string(),
        ]);
        
        signals.insert(Domain::Health, vec![
            "health".to_string(),
            "sleep".to_string(),
            "exercise".to_string(),
            "mental".to_string(),
            "therapy".to_string(),
        ]);
        
        signals
    }

    /// Infer what the conversation is REALLY about
    pub fn infer_context(&self, conversation: &[String]) -> ConversationContext {
        let combined_text = conversation.join(" ").to_lowercase();
        
        let inferred_topics = self.infer_topics(&combined_text);
        let emotional_state = self.infer_emotion(&combined_text);
        let underlying_need = self.infer_need(&combined_text, &emotional_state);
        let confidence = self.calculate_inference_confidence(&inferred_topics, &combined_text);
        
        ConversationContext {
            explicit_topic: None,
            inferred_topic: inferred_topics,
            emotional_state,
            underlying_need: Some(underlying_need),
            confidence,
            evidence: self.extract_evidence(&combined_text),
        }
    }

    fn infer_topics(&self, text: &str) -> Vec<String> {
        let mut topic_scores: HashMap<Domain, f64> = HashMap::new();
        
        for (domain, signals) in &self.domain_signals {
            let score = signals.iter()
                .filter(|signal| text.contains(signal.as_str()))
                .count() as f64;
            
            if score > 0.0 {
                topic_scores.insert(domain.clone(), score);
            }
        }
        
        let mut topics: Vec<(Domain, f64)> = topic_scores.into_iter().collect();
        topics.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        topics.into_iter()
            .take(2)
            .map(|(domain, _)| format!("{:?}", domain))
            .collect()
    }

    fn infer_emotion(&self, text: &str) -> EmotionalState {
        let words: Vec<&str> = text.split_whitespace().collect();
        
        let mut valence_sum = 0.0;
        let mut arousal_sum = 0.0;
        let mut count = 0;
        
        for word in words {
            if let Some(&(v, a)) = self.emotional_lexicon.get(word) {
                valence_sum += v;
                arousal_sum += a;
                count += 1;
            }
        }
        
        let (valence, arousal) = if count > 0 {
            (valence_sum / count as f64, arousal_sum / count as f64)
        } else {
            (0.0, 0.5)
        };
        
        let dominant_emotion = self.classify_emotion(valence, arousal);
        
        EmotionalState {
            valence,
            arousal,
            dominant_emotion,
        }
    }

    fn classify_emotion(&self, valence: f64, arousal: f64) -> String {
        match (valence > 0.3, arousal > 0.6) {
            (true, true) => "excited".to_string(),
            (true, false) => "content".to_string(),
            (false, true) => "anxious".to_string(),
            (false, false) => "sad".to_string(),
        }
    }

    fn infer_need(&self, text: &str, emotion: &EmotionalState) -> UnderlyingNeed {
        if text.contains("how") || text.contains("what") || text.contains("why") {
            return UnderlyingNeed::Information;
        }
        
        if text.contains("help") || text.contains("advice") || text.contains("should") {
            return UnderlyingNeed::ProblemSolving;
        }
        
        if emotion.arousal > 0.7 && emotion.valence < 0.0 {
            return UnderlyingNeed::Venting;
        }
        
        UnderlyingNeed::Guidance
    }

    fn calculate_inference_confidence(&self, topics: &[String], text: &str) -> f64 {
        let signal_strength = topics.len() as f64 / 3.0;
        let text_length_factor = (text.len() as f64 / 100.0).min(1.0);
        
        (signal_strength * text_length_factor).min(0.95)
    }

    fn extract_evidence(&self, text: &str) -> Vec<String> {
        text.split('.')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .take(3)
            .collect()
    }
}

// ============================================================================
// PART 3: PATTERN DEFINITIONS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Domain {
    Career,
    Relationships,
    Health,
    Learning,
    Finance,
    General,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    CauseEffect,
    Strategy,
    Principle,
    Heuristic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub id: String,
    pub domain: Domain,
    pub pattern_type: PatternType,
    pub condition: String,
    pub effects: Vec<String>,
    pub common_mistakes: Vec<String>,
    pub successful_strategies: Vec<String>,
    pub confidence: f64,
    pub evidence_count: u32,
    pub first_observed: u64,
    pub last_updated: u64,
}

// ============================================================================
// PART 4: REASONING ENGINE
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedKnowledge {
    pub id: String,
    pub source_patterns: Vec<String>,
    pub reasoning_type: ReasoningType,
    pub conclusion: String,
    pub confidence: f64,
    pub derivation_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningType {
    Deduction,
    Induction,
    Analogy,
    Abduction,
}

pub struct ReasoningEngine {
    min_source_confidence: f64,
    max_inference_chain: usize,
}

impl ReasoningEngine {
    pub fn new() -> Self {
        Self {
            min_source_confidence: 0.7,
            max_inference_chain: 3,
        }
    }

    /// Apply deductive reasoning: If A→B and B→C, then A→C
    pub fn deduce_transitive(&self, 
                             pattern1: &Pattern, 
                             pattern2: &Pattern) -> Option<DerivedKnowledge> {
        if !self.can_chain(pattern1, pattern2) {
            return None;
        }

        if pattern1.confidence < self.min_source_confidence 
           || pattern2.confidence < self.min_source_confidence {
            return None;
        }

        let confidence = pattern1.confidence.min(pattern2.confidence) * 0.9;

        Some(DerivedKnowledge {
            id: format!("derived_{}", Self::current_timestamp()),
            source_patterns: vec![pattern1.id.clone(), pattern2.id.clone()],
            reasoning_type: ReasoningType::Deduction,
            conclusion: format!(
                "If {}, then {} (via {})",
                pattern1.condition,
                self.extract_effect(pattern2),
                self.extract_effect(pattern1)
            ),
            confidence,
            derivation_steps: vec![
                format!("Given: {}", pattern1.condition),
                format!("Leads to: {}", self.extract_effect(pattern1)),
                format!("Which leads to: {}", self.extract_effect(pattern2)),
            ],
        })
    }

    fn can_chain(&self, pattern1: &Pattern, pattern2: &Pattern) -> bool {
        let effect1 = self.extract_effect(pattern1);
        let condition2 = &pattern2.condition;
        
        effect1.contains(condition2) || condition2.contains(&effect1)
    }

    fn extract_effect(&self, pattern: &Pattern) -> String {
        pattern.effects.first()
            .or_else(|| pattern.successful_strategies.first())
            .cloned()
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Apply inductive reasoning
    pub fn induce_general_pattern(&self, patterns: &[Pattern]) -> Option<DerivedKnowledge> {
        if patterns.len() < 3 {
            return None;
        }

        let common_domain = patterns[0].domain.clone();
        if !patterns.iter().all(|p| p.domain == common_domain) {
            return None;
        }

        let avg_confidence: f64 = patterns.iter()
            .map(|p| p.confidence)
            .sum::<f64>() / patterns.len() as f64;

        let conclusion = format!(
            "General pattern in {:?}: {} observations support common outcome",
            common_domain,
            patterns.len()
        );

        Some(DerivedKnowledge {
            id: format!("induced_{}", Self::current_timestamp()),
            source_patterns: patterns.iter().map(|p| p.id.clone()).collect(),
            reasoning_type: ReasoningType::Induction,
            conclusion,
            confidence: avg_confidence * 0.85,
            derivation_steps: patterns.iter()
                .map(|p| format!("Case: {}", p.condition))
                .collect(),
        })
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

// ============================================================================
// PART 5: UNIFIED ACTIVE LEARNING SYSTEM
// ============================================================================

pub struct ActiveLearningSystem {
    recall_engine: ActiveRecallEngine,
    context_engine: ContextInferenceEngine,
    reasoning_engine: ReasoningEngine,
    recall_challenges: HashMap<String, RecallChallenge>,
    derived_knowledge: Vec<DerivedKnowledge>,
    patterns: HashMap<String, Pattern>,
}

impl ActiveLearningSystem {
    pub fn new() -> Self {
        Self {
            recall_engine: ActiveRecallEngine::new(),
            context_engine: ContextInferenceEngine::new(),
            reasoning_engine: ReasoningEngine::new(),
            recall_challenges: HashMap::new(),
            derived_knowledge: Vec::new(),
            patterns: HashMap::new(),
        }
    }

    /// Store a pattern
    pub fn store_pattern(&mut self, pattern: Pattern) {
        self.patterns.insert(pattern.id.clone(), pattern);
    }

    /// Test understanding through active recall
    pub fn test_understanding(&mut self, pattern_id: &str) -> Option<RecallChallenge> {
        let pattern = self.patterns.get(pattern_id)?;
        let challenge = self.recall_engine.generate_challenge(pattern);
        self.recall_challenges.insert(challenge.id.clone(), challenge.clone());
        Some(challenge)
    }

    /// Verify a reconstruction attempt
    pub fn verify_recall(&mut self, 
                         challenge_id: &str, 
                         reconstructed: &str) -> Option<(f64, f64)> {
        let challenge = self.recall_challenges.get_mut(challenge_id)?;
        let pattern = self.patterns.get(&challenge.pattern_id)?;
        
        let score = self.recall_engine.verify_reconstruction(
            reconstructed,
            &challenge.expected_reconstruction
        );

        if score > 0.7 {
            challenge.success_count += 1;
        } else {
            challenge.failure_count += 1;
        }
        challenge.last_attempted = Some(Self::current_timestamp());

        let new_confidence = self.recall_engine.update_understanding(
            pattern.confidence,
            score,
            challenge.difficulty
        );

        // Update pattern confidence
        if let Some(pattern) = self.patterns.get_mut(&challenge.pattern_id) {
            pattern.confidence = new_confidence;
        }

        Some((score, new_confidence))
    }

    /// Infer conversation context
    pub fn understand_context(&self, conversation: &[String]) -> ConversationContext {
        self.context_engine.infer_context(conversation)
    }

    /// Derive new knowledge through reasoning
    pub fn derive_knowledge(&mut self) -> Vec<DerivedKnowledge> {
        let patterns: Vec<Pattern> = self.patterns.values().cloned().collect();
        let mut derived = Vec::new();

        // Try deductive reasoning (pairwise)
        for i in 0..patterns.len() {
            for j in (i+1)..patterns.len() {
                if let Some(deduction) = self.reasoning_engine.deduce_transitive(
                    &patterns[i],
                    &patterns[j]
                ) {
                    derived.push(deduction);
                }
            }
        }

        // Try inductive reasoning
        if let Some(induction) = self.reasoning_engine.induce_general_pattern(&patterns) {
            derived.push(induction);
        }

        self.derived_knowledge.extend(derived.clone());
        derived
    }

    /// Figure things out independently
    pub fn solve_independently(&self, problem: &str) -> Option<String> {
        let context = self.context_engine.infer_context(&[problem.to_string()]);

        let solution = format!(
            "Problem analysis (confidence {:.2}):\n\
             Context: {}\n\
             Underlying need: {:?}\n\
             Emotional state: {} (valence: {:.2}, arousal: {:.2})",
            context.confidence,
            context.inferred_topic.join(", "),
            context.underlying_need,
            context.emotional_state.dominant_emotion,
            context.emotional_state.valence,
            context.emotional_state.arousal
        );

        Some(solution)
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}
