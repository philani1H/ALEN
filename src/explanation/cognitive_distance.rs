//! Cognitive Distance Calculator
//!
//! Measures how understandable an explanation is for a specific audience.
//!
//! Mathematical Foundation:
//! CognitiveDistance(L, a) = α·Complexity(L, a) + β·Relevance(L, a) + γ·Clarity(L, a)
//!
//! Where:
//! - Complexity: Vocabulary difficulty, sentence structure
//! - Relevance: Topic alignment, example appropriateness
//! - Clarity: Logical flow, step coherence

use crate::api::user_modeling::{UserState, UserPreferences, TopicInterest};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ============================================================================
// PART 1: VOCABULARY DATABASE
// ============================================================================

pub struct VocabularyDatabase {
    /// Word difficulty levels (0-1, where 1 is most difficult)
    word_difficulty: HashMap<String, f64>,
    
    /// Common words (difficulty < 0.3)
    common_words: HashSet<String>,
    
    /// Technical terms (difficulty > 0.7)
    technical_terms: HashSet<String>,
}

impl VocabularyDatabase {
    pub fn new() -> Self {
        let mut word_difficulty = HashMap::new();
        let mut common_words = HashSet::new();
        let mut technical_terms = HashSet::new();
        
        // Common words (child-friendly)
        let common = vec![
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "can", "may", "might", "must", "shall", "this", "that", "these", "those",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
            "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
            "and", "or", "but", "if", "because", "so", "then", "than", "as", "like",
            "in", "on", "at", "to", "for", "of", "with", "from", "by", "about",
            "up", "down", "out", "over", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "can", "will",
            "just", "should", "now", "one", "two", "three", "four", "five",
            "add", "plus", "minus", "times", "divide", "equal", "number", "count",
        ];
        
        for word in common {
            word_difficulty.insert(word.to_string(), 0.1);
            common_words.insert(word.to_string());
        }
        
        // Technical terms
        let technical = vec![
            "algorithm", "function", "variable", "parameter", "derivative", "integral",
            "polynomial", "coefficient", "theorem", "lemma", "corollary", "axiom",
            "bijection", "isomorphism", "homomorphism", "topology", "manifold",
            "eigenvalue", "eigenvector", "matrix", "determinant", "vector", "scalar",
            "differential", "equation", "inequality", "optimization", "convergence",
            "asymptotic", "complexity", "recursion", "iteration", "induction",
        ];
        
        for word in technical {
            word_difficulty.insert(word.to_string(), 0.9);
            technical_terms.insert(word.to_string());
        }
        
        Self {
            word_difficulty,
            common_words,
            technical_terms,
        }
    }
    
    pub fn get_difficulty(&self, word: &str) -> f64 {
        let normalized = word.to_lowercase();
        self.word_difficulty.get(&normalized).copied().unwrap_or(0.5)
    }
    
    pub fn is_common(&self, word: &str) -> bool {
        self.common_words.contains(&word.to_lowercase())
    }
    
    pub fn is_technical(&self, word: &str) -> bool {
        self.technical_terms.contains(&word.to_lowercase())
    }
}

impl Default for VocabularyDatabase {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PART 2: COMPLEXITY ANALYZER
// ============================================================================

pub struct ComplexityAnalyzer {
    vocabulary: VocabularyDatabase,
}

impl ComplexityAnalyzer {
    pub fn new() -> Self {
        Self {
            vocabulary: VocabularyDatabase::new(),
        }
    }
    
    /// Measure complexity of explanation for given audience
    pub fn measure(&self, explanation: &str, preferences: &UserPreferences) -> f64 {
        // Tokenize
        let words: Vec<&str> = explanation.split_whitespace().collect();
        
        if words.is_empty() {
            return 0.0;
        }
        
        // Vocabulary complexity
        let vocab_complexity = self.measure_vocabulary_complexity(&words);
        
        // Sentence complexity
        let sentence_complexity = self.measure_sentence_complexity(explanation);
        
        // Technical density
        let technical_density = self.measure_technical_density(&words);
        
        // User's technical level preference
        let user_technical_level = preferences.technical_level.mean();
        
        // Compute mismatch
        let vocab_mismatch = (vocab_complexity - user_technical_level).abs();
        let technical_mismatch = (technical_density - user_technical_level).abs();
        
        // Combined complexity score
        0.4 * vocab_mismatch + 0.3 * sentence_complexity + 0.3 * technical_mismatch
    }
    
    fn measure_vocabulary_complexity(&self, words: &[&str]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }
        
        let total_difficulty: f64 = words.iter()
            .map(|w| self.vocabulary.get_difficulty(w))
            .sum();
        
        total_difficulty / words.len() as f64
    }
    
    fn measure_sentence_complexity(&self, text: &str) -> f64 {
        let sentences: Vec<&str> = text.split(&['.', '!', '?'][..]).collect();
        
        if sentences.is_empty() {
            return 0.0;
        }
        
        // Average words per sentence
        let total_words: usize = sentences.iter()
            .map(|s| s.split_whitespace().count())
            .sum();
        
        let avg_words_per_sentence = total_words as f64 / sentences.len() as f64;
        
        // Normalize: 10 words = 0.0, 30 words = 1.0
        ((avg_words_per_sentence - 10.0) / 20.0).clamp(0.0, 1.0)
    }
    
    fn measure_technical_density(&self, words: &[&str]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }
        
        let technical_count = words.iter()
            .filter(|w| self.vocabulary.is_technical(w))
            .count();
        
        technical_count as f64 / words.len() as f64
    }
}

impl Default for ComplexityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PART 3: RELEVANCE SCORER
// ============================================================================

pub struct RelevanceScorer;

impl RelevanceScorer {
    pub fn new() -> Self {
        Self
    }
    
    /// Score relevance of explanation to user's interests
    pub fn score(&self, explanation: &str, interests: &HashMap<String, TopicInterest>) -> f64 {
        if interests.is_empty() {
            return 0.0; // No known interests, assume neutral
        }
        
        let words: Vec<String> = explanation.to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        
        // Check how many interest topics appear in explanation
        let mut relevance_sum = 0.0;
        let mut count = 0;
        
        for (topic, interest) in interests {
            let topic_lower = topic.to_lowercase();
            let appears = words.iter().any(|w| w.contains(&topic_lower));
            
            if appears {
                relevance_sum += interest.score;
                count += 1;
            }
        }
        
        if count == 0 {
            // No topics matched, return moderate irrelevance
            0.5
        } else {
            // Average relevance of matched topics
            // Invert: high interest = low distance
            1.0 - (relevance_sum / count as f64)
        }
    }
}

impl Default for RelevanceScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PART 4: CLARITY ASSESSOR
// ============================================================================

pub struct ClarityAssessor;

impl ClarityAssessor {
    pub fn new() -> Self {
        Self
    }
    
    /// Assess clarity of explanation
    pub fn assess(&self, explanation: &str) -> f64 {
        // Logical flow indicators
        let has_transitions = self.has_transition_words(explanation);
        let has_structure = self.has_clear_structure(explanation);
        let has_examples = self.has_examples(explanation);
        
        // Compute clarity score (lower is better for distance)
        let clarity_score = 
            (if has_transitions { 0.3 } else { 0.0 }) +
            (if has_structure { 0.4 } else { 0.0 }) +
            (if has_examples { 0.3 } else { 0.0 });
        
        // Invert: high clarity = low distance
        1.0 - clarity_score
    }
    
    fn has_transition_words(&self, text: &str) -> bool {
        let transitions = [
            "first", "second", "third", "next", "then", "finally",
            "therefore", "thus", "hence", "because", "since",
            "however", "although", "while", "whereas",
            "for example", "for instance", "such as",
        ];
        
        let lower = text.to_lowercase();
        transitions.iter().any(|t| lower.contains(t))
    }
    
    fn has_clear_structure(&self, text: &str) -> bool {
        // Check for numbered lists or bullet points
        let has_numbers = text.contains("1.") || text.contains("2.") || text.contains("3.");
        let has_bullets = text.contains("- ") || text.contains("* ");
        let has_steps = text.to_lowercase().contains("step");
        
        has_numbers || has_bullets || has_steps
    }
    
    fn has_examples(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        lower.contains("example") || lower.contains("for instance") || 
        lower.contains("such as") || lower.contains("like")
    }
}

impl Default for ClarityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PART 5: COGNITIVE DISTANCE CALCULATOR
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceWeights {
    pub complexity: f64,
    pub relevance: f64,
    pub clarity: f64,
}

impl Default for DistanceWeights {
    fn default() -> Self {
        Self {
            complexity: 0.4,
            relevance: 0.3,
            clarity: 0.3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveDistance {
    /// Total cognitive distance (0-1, lower is better)
    pub total: f64,
    
    /// Complexity component
    pub complexity: f64,
    
    /// Relevance component
    pub relevance: f64,
    
    /// Clarity component
    pub clarity: f64,
    
    /// Is explanation understandable? (distance < 0.5)
    pub understandable: bool,
    
    /// Quality rating
    pub quality: String,
}

impl CognitiveDistance {
    pub fn quality_rating(&self) -> &str {
        if self.total < 0.2 {
            "Excellent"
        } else if self.total < 0.4 {
            "Good"
        } else if self.total < 0.6 {
            "Fair"
        } else if self.total < 0.8 {
            "Poor"
        } else {
            "Very Poor"
        }
    }
}

pub struct CognitiveDistanceCalculator {
    /// Complexity analyzer
    complexity_analyzer: ComplexityAnalyzer,
    
    /// Relevance scorer
    relevance_scorer: RelevanceScorer,
    
    /// Clarity assessor
    clarity_assessor: ClarityAssessor,
    
    /// Weights for combined metric
    weights: DistanceWeights,
}

impl CognitiveDistanceCalculator {
    pub fn new() -> Self {
        Self {
            complexity_analyzer: ComplexityAnalyzer::new(),
            relevance_scorer: RelevanceScorer::new(),
            clarity_assessor: ClarityAssessor::new(),
            weights: DistanceWeights::default(),
        }
    }
    
    pub fn with_weights(weights: DistanceWeights) -> Self {
        Self {
            complexity_analyzer: ComplexityAnalyzer::new(),
            relevance_scorer: RelevanceScorer::new(),
            clarity_assessor: ClarityAssessor::new(),
            weights,
        }
    }
    
    /// Compute cognitive distance between explanation and audience
    pub fn compute_distance(
        &self,
        explanation: &str,
        user: &UserState,
    ) -> CognitiveDistance {
        // Measure complexity
        let complexity = self.complexity_analyzer.measure(
            explanation,
            &user.preferences,
        );
        
        // Measure relevance
        let relevance = self.relevance_scorer.score(
            explanation,
            &user.interests,
        );
        
        // Measure clarity
        let clarity = self.clarity_assessor.assess(explanation);
        
        // Compute weighted distance
        let total = 
            self.weights.complexity * complexity +
            self.weights.relevance * relevance +
            self.weights.clarity * clarity;
        
        let understandable = total < 0.5;
        let quality = Self::quality_rating(total);
        
        CognitiveDistance {
            total,
            complexity,
            relevance,
            clarity,
            understandable,
            quality,
        }
    }
    
    fn quality_rating(distance: f64) -> String {
        if distance < 0.2 {
            "Excellent".to_string()
        } else if distance < 0.4 {
            "Good".to_string()
        } else if distance < 0.6 {
            "Fair".to_string()
        } else if distance < 0.8 {
            "Poor".to_string()
        } else {
            "Very Poor".to_string()
        }
    }
}

impl Default for CognitiveDistanceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::user_modeling::BetaDist;
    
    #[test]
    fn test_vocabulary_database() {
        let vocab = VocabularyDatabase::new();
        
        assert!(vocab.is_common("the"));
        assert!(vocab.is_technical("algorithm"));
        assert!(vocab.get_difficulty("the") < 0.3);
        assert!(vocab.get_difficulty("algorithm") > 0.7);
    }
    
    #[test]
    fn test_complexity_analyzer() {
        let analyzer = ComplexityAnalyzer::new();
        
        let simple = "The cat sat on the mat.";
        let complex = "The bijective homomorphism preserves algebraic structure.";
        
        let prefs = UserPreferences {
            depth: BetaDist { alpha: 2.0, beta: 2.0 },
            math: BetaDist { alpha: 2.0, beta: 2.0 },
            verbosity: BetaDist { alpha: 2.0, beta: 2.0 },
            technical_level: BetaDist { alpha: 2.0, beta: 8.0 }, // Low technical
            formality: BetaDist { alpha: 2.0, beta: 2.0 },
        };
        
        let simple_complexity = analyzer.measure(simple, &prefs);
        let complex_complexity = analyzer.measure(complex, &prefs);
        
        assert!(complex_complexity > simple_complexity);
    }
    
    #[test]
    fn test_clarity_assessor() {
        let assessor = ClarityAssessor::new();
        
        let clear = "First, we add 2 and 2. Then, we get 4. For example, 2 apples plus 2 apples equals 4 apples.";
        let unclear = "Addition operation yields sum.";
        
        let clear_score = assessor.assess(clear);
        let unclear_score = assessor.assess(unclear);
        
        assert!(clear_score < unclear_score); // Lower distance = clearer
    }
}
