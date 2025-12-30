//! Frequency Governor
//!
//! Controls adaptive thinking speed based on:
//! - Problem difficulty
//! - Current confidence
//! - Risk level
//! - User context
//!
//! Implements: B = f(difficulty, confidence, risk)
//!
//! This makes the system think:
//! - Fast for simple questions (1+1)
//! - Deep for complex problems (proofs)
//! - Careful for emotional/sensitive topics

use serde::{Deserialize, Serialize};

/// Frequency allocation for different thinking modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyAllocation {
    /// Number of reasoning cycles (parallel operators)
    pub reasoning_cycles: usize,
    
    /// Number of verification passes
    pub verification_passes: usize,
    
    /// Learning update rate (0.0 to 1.0)
    pub learning_rate: f64,
    
    /// Attention refresh rate (how often to re-evaluate context)
    pub attention_refresh: usize,
    
    /// Total thinking budget allocated
    pub total_budget: f64,
}

/// Problem characteristics for frequency allocation
#[derive(Debug, Clone)]
pub struct ProblemCharacteristics {
    /// Estimated difficulty (0.0 = trivial, 1.0 = very hard)
    pub difficulty: f64,
    
    /// Current confidence in solution (0.0 to 1.0)
    pub confidence: f64,
    
    /// Risk level (0.0 = safe, 1.0 = high stakes)
    pub risk: f64,
    
    /// Problem type
    pub problem_type: ProblemType,
    
    /// Emotional sensitivity (0.0 = factual, 1.0 = highly emotional)
    pub emotional_sensitivity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProblemType {
    /// Simple arithmetic or lookup
    Trivial,
    
    /// Standard question with known answer
    Standard,
    
    /// Requires reasoning but not proof
    Moderate,
    
    /// Requires proof or deep analysis
    Complex,
    
    /// Novel problem, no known solution
    Novel,
    
    /// Emotional support or sensitive topic
    Emotional,
}

impl ProblemType {
    /// Base difficulty for this problem type
    pub fn base_difficulty(&self) -> f64 {
        match self {
            ProblemType::Trivial => 0.1,
            ProblemType::Standard => 0.3,
            ProblemType::Moderate => 0.5,
            ProblemType::Complex => 0.7,
            ProblemType::Novel => 0.9,
            ProblemType::Emotional => 0.6, // Moderate difficulty but high sensitivity
        }
    }

    /// Base risk for this problem type
    pub fn base_risk(&self) -> f64 {
        match self {
            ProblemType::Trivial => 0.1,
            ProblemType::Standard => 0.2,
            ProblemType::Moderate => 0.4,
            ProblemType::Complex => 0.6,
            ProblemType::Novel => 0.8,
            ProblemType::Emotional => 0.9, // High risk - can cause harm if wrong
        }
    }
}

/// Frequency Governor - controls adaptive thinking speed
pub struct FrequencyGovernor {
    /// Maximum reasoning cycles allowed
    pub max_reasoning_cycles: usize,
    
    /// Maximum verification passes
    pub max_verification_passes: usize,
    
    /// Base learning rate
    pub base_learning_rate: f64,
    
    /// Minimum thinking budget
    pub min_budget: f64,
    
    /// Maximum thinking budget
    pub max_budget: f64,
}

impl Default for FrequencyGovernor {
    fn default() -> Self {
        Self {
            max_reasoning_cycles: 8,
            max_verification_passes: 5,
            base_learning_rate: 0.01,
            min_budget: 1.0,
            max_budget: 10.0,
        }
    }
}

impl FrequencyGovernor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate thinking budget based on problem characteristics
    /// B = f(difficulty, confidence, risk)
    pub fn calculate_budget(&self, chars: &ProblemCharacteristics) -> f64 {
        // Base budget from difficulty
        let difficulty_factor = chars.difficulty;
        
        // Confidence factor: lower confidence = more thinking needed
        let confidence_factor = 1.0 - chars.confidence;
        
        // Risk factor: higher risk = more careful thinking
        let risk_factor = chars.risk;
        
        // Emotional sensitivity: emotional topics need careful handling
        let emotional_factor = chars.emotional_sensitivity;
        
        // Combined budget calculation
        let budget = self.min_budget + 
            (self.max_budget - self.min_budget) * (
                0.4 * difficulty_factor +
                0.3 * confidence_factor +
                0.2 * risk_factor +
                0.1 * emotional_factor
            );
        
        budget.clamp(self.min_budget, self.max_budget)
    }

    /// Allocate frequency based on budget
    pub fn allocate_frequency(
        &self,
        chars: &ProblemCharacteristics,
    ) -> FrequencyAllocation {
        let budget = self.calculate_budget(chars);
        
        // Allocate reasoning cycles (more for complex problems)
        let reasoning_cycles = match chars.problem_type {
            ProblemType::Trivial => 1,
            ProblemType::Standard => 2,
            ProblemType::Moderate => 4,
            ProblemType::Complex => 6,
            ProblemType::Novel => 8,
            ProblemType::Emotional => 3, // Moderate cycles but high sensitivity
        };
        
        // Allocate verification passes (more for high-risk problems)
        let verification_passes = if chars.risk > 0.7 {
            5
        } else if chars.risk > 0.5 {
            3
        } else if chars.risk > 0.3 {
            2
        } else {
            1
        };
        
        // Adjust learning rate based on confidence
        let learning_rate = if chars.confidence > 0.8 {
            self.base_learning_rate * 1.5 // Learn faster when confident
        } else if chars.confidence < 0.5 {
            self.base_learning_rate * 0.5 // Learn slower when uncertain
        } else {
            self.base_learning_rate
        };
        
        // Attention refresh (more for emotional/contextual problems)
        let attention_refresh = if chars.emotional_sensitivity > 0.5 {
            3 // Refresh context frequently for emotional topics
        } else if chars.difficulty > 0.7 {
            2 // Refresh for complex problems
        } else {
            1 // Minimal refresh for simple problems
        };
        
        FrequencyAllocation {
            reasoning_cycles: reasoning_cycles.min(self.max_reasoning_cycles),
            verification_passes: verification_passes.min(self.max_verification_passes),
            learning_rate,
            attention_refresh,
            total_budget: budget,
        }
    }

    /// Estimate problem characteristics from input
    pub fn estimate_characteristics(&self, input: &str) -> ProblemCharacteristics {
        let lower = input.to_lowercase();
        
        // Detect problem type
        let problem_type = self.detect_problem_type(&lower);
        
        // Estimate difficulty
        let difficulty = self.estimate_difficulty(&lower, problem_type);
        
        // Estimate risk
        let risk = self.estimate_risk(&lower, problem_type);
        
        // Detect emotional sensitivity
        let emotional_sensitivity = self.detect_emotional_sensitivity(&lower);
        
        // Initial confidence (will be updated during reasoning)
        let confidence = 0.5;
        
        ProblemCharacteristics {
            difficulty,
            confidence,
            risk,
            problem_type,
            emotional_sensitivity,
        }
    }

    /// Detect problem type from input
    fn detect_problem_type(&self, input: &str) -> ProblemType {
        // Emotional keywords
        let emotional_keywords = [
            "crying", "sad", "depressed", "anxious", "worried", "scared",
            "angry", "frustrated", "hurt", "lonely", "help", "support",
            "comfort", "advice", "feel", "feeling", "emotion",
        ];
        
        if emotional_keywords.iter().any(|k| input.contains(k)) {
            return ProblemType::Emotional;
        }
        
        // Trivial (simple arithmetic)
        if input.matches('+').count() > 0 || input.matches('*').count() > 0 {
            if input.split_whitespace().count() < 10 {
                return ProblemType::Trivial;
            }
        }
        
        // Complex (proof, theorem, deep analysis)
        let complex_keywords = ["prove", "theorem", "derive", "analyze deeply"];
        if complex_keywords.iter().any(|k| input.contains(k)) {
            return ProblemType::Complex;
        }
        
        // Novel (new, unknown, research)
        let novel_keywords = ["new", "novel", "unknown", "research", "discover"];
        if novel_keywords.iter().any(|k| input.contains(k)) {
            return ProblemType::Novel;
        }
        
        // Moderate (explain, compare, summarize)
        let moderate_keywords = ["explain", "compare", "summarize", "how", "why"];
        if moderate_keywords.iter().any(|k| input.contains(k)) {
            return ProblemType::Moderate;
        }
        
        // Default to standard
        ProblemType::Standard
    }

    /// Estimate difficulty from input
    fn estimate_difficulty(&self, input: &str, problem_type: ProblemType) -> f64 {
        let base = problem_type.base_difficulty();
        
        // Adjust based on length (longer = potentially more complex)
        let length_factor = (input.len() as f64 / 500.0).min(0.3);
        
        // Adjust based on technical terms
        let technical_keywords = ["algorithm", "equation", "formula", "theorem", "proof"];
        let technical_count = technical_keywords.iter()
            .filter(|k| input.contains(*k))
            .count();
        let technical_factor = (technical_count as f64 * 0.1).min(0.2);
        
        (base + length_factor + technical_factor).min(1.0)
    }

    /// Estimate risk from input
    fn estimate_risk(&self, input: &str, problem_type: ProblemType) -> f64 {
        let base = problem_type.base_risk();
        
        // High-risk keywords
        let risk_keywords = ["medical", "legal", "financial", "safety", "danger", "harm"];
        let risk_count = risk_keywords.iter()
            .filter(|k| input.contains(*k))
            .count();
        let risk_factor = (risk_count as f64 * 0.2).min(0.3);
        
        (base + risk_factor).min(1.0)
    }

    /// Detect emotional sensitivity
    fn detect_emotional_sensitivity(&self, input: &str) -> f64 {
        let emotional_keywords = [
            "crying", "sad", "depressed", "anxious", "worried", "scared",
            "angry", "frustrated", "hurt", "lonely", "help", "support",
            "comfort", "advice", "feel", "feeling", "emotion", "upset",
            "devastated", "heartbroken", "suffering", "pain",
        ];
        
        let count = emotional_keywords.iter()
            .filter(|k| input.contains(*k))
            .count();
        
        (count as f64 * 0.2).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_problem() {
        let governor = FrequencyGovernor::new();
        let chars = governor.estimate_characteristics("What is 2 + 2?");
        
        assert_eq!(chars.problem_type, ProblemType::Trivial);
        assert!(chars.difficulty < 0.3);
        
        let allocation = governor.allocate_frequency(&chars);
        assert_eq!(allocation.reasoning_cycles, 1);
        assert_eq!(allocation.verification_passes, 1);
    }

    #[test]
    fn test_emotional_problem() {
        let governor = FrequencyGovernor::new();
        // Use input with multiple emotional keywords to exceed 0.5 threshold
        let chars = governor.estimate_characteristics("My friend is crying and feeling sad, I'm worried about her. Help!");
        
        assert_eq!(chars.problem_type, ProblemType::Emotional);
        // With 4+ emotional keywords (crying, feeling, sad, worried, help), sensitivity should be > 0.5
        assert!(chars.emotional_sensitivity > 0.5);
        assert!(chars.risk > 0.7);
        
        let allocation = governor.allocate_frequency(&chars);
        assert!(allocation.attention_refresh >= 3);
        assert!(allocation.verification_passes >= 3);
    }

    #[test]
    fn test_complex_problem() {
        let governor = FrequencyGovernor::new();
        let chars = governor.estimate_characteristics("Prove that the square root of 2 is irrational");
        
        assert_eq!(chars.problem_type, ProblemType::Complex);
        assert!(chars.difficulty > 0.6);
        
        let allocation = governor.allocate_frequency(&chars);
        assert!(allocation.reasoning_cycles >= 6);
        assert!(allocation.verification_passes >= 3);
    }

    #[test]
    fn test_budget_calculation() {
        let governor = FrequencyGovernor::new();
        
        // Low difficulty, high confidence = low budget
        let chars1 = ProblemCharacteristics {
            difficulty: 0.2,
            confidence: 0.9,
            risk: 0.1,
            problem_type: ProblemType::Trivial,
            emotional_sensitivity: 0.0,
        };
        let budget1 = governor.calculate_budget(&chars1);
        assert!(budget1 < 3.0);
        
        // High difficulty, low confidence, high risk = high budget
        let chars2 = ProblemCharacteristics {
            difficulty: 0.9,
            confidence: 0.3,
            risk: 0.8,
            problem_type: ProblemType::Novel,
            emotional_sensitivity: 0.0,
        };
        let budget2 = governor.calculate_budget(&chars2);
        assert!(budget2 > 7.0);
    }
}
