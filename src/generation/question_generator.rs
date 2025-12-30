//! Question Generation Module
//!
//! Implements: Q' ~ P_θ(Q' | S, A, u)
//!
//! Generates questions to:
//! - Clarify user understanding
//! - Guide learning
//! - Test comprehension
//! - Encourage interaction
//!
//! This makes the AI interactive and teaching-capable.

use crate::core::ThoughtState;
use crate::memory::SemanticMemory;
use serde::{Deserialize, Serialize};

/// Question type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuestionType {
    /// Clarification: "What do you mean by X?"
    Clarification,
    /// Comprehension: "Can you explain Y?"
    Comprehension,
    /// Application: "How would you use Z?"
    Application,
    /// Analysis: "Why did X happen?"
    Analysis,
    /// Synthesis: "How does X relate to Y?"
    Synthesis,
    /// Evaluation: "What do you think about X?"
    Evaluation,
    /// Follow-up: "Tell me more about X"
    FollowUp,
}

/// Question difficulty level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyLevel {
    /// Simple recall questions
    Easy,
    /// Understanding questions
    Medium,
    /// Analysis and synthesis
    Hard,
    /// Expert-level reasoning
    Expert,
}

impl DifficultyLevel {
    /// Convert to numeric score (0-1)
    pub fn to_score(&self) -> f64 {
        match self {
            Self::Easy => 0.2,
            Self::Medium => 0.5,
            Self::Hard => 0.8,
            Self::Expert => 1.0,
        }
    }
    
    /// From user proficiency
    pub fn from_proficiency(proficiency: f64) -> Self {
        if proficiency < 0.3 {
            Self::Easy
        } else if proficiency < 0.6 {
            Self::Medium
        } else if proficiency < 0.9 {
            Self::Hard
        } else {
            Self::Expert
        }
    }
}

/// Generated question
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedQuestion {
    /// Question text
    pub question: String,
    /// Question type
    pub question_type: QuestionType,
    /// Difficulty level
    pub difficulty: DifficultyLevel,
    /// Expected answer (for validation)
    pub expected_answer: Option<String>,
    /// Confidence in question quality
    pub confidence: f64,
}

/// Question generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionGeneratorConfig {
    /// Temperature for generation
    pub temperature: f64,
    /// Maximum questions to generate
    pub max_questions: usize,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Adapt difficulty to user
    pub adaptive_difficulty: bool,
}

impl Default for QuestionGeneratorConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            max_questions: 3,
            min_confidence: 0.6,
            adaptive_difficulty: true,
        }
    }
}

/// Question Generator
pub struct QuestionGenerator {
    config: QuestionGeneratorConfig,
    dimension: usize,
}

impl QuestionGenerator {
    pub fn new(dimension: usize, config: QuestionGeneratorConfig) -> Self {
        Self { config, dimension }
    }
    
    pub fn default(dimension: usize) -> Self {
        Self::new(dimension, QuestionGeneratorConfig::default())
    }
    
    /// Generate questions: Q' ~ P_θ(Q' | S, A, u)
    pub fn generate_questions(
        &self,
        story_state: &ThoughtState,
        answer_state: Option<&ThoughtState>,
        user_embedding: Option<&[f64]>,
        memory: &SemanticMemory,
    ) -> Result<Vec<GeneratedQuestion>, Box<dyn std::error::Error>> {
        let mut questions = Vec::new();
        
        // Determine user proficiency from embedding
        let proficiency = self.estimate_proficiency(user_embedding);
        let difficulty = if self.config.adaptive_difficulty {
            DifficultyLevel::from_proficiency(proficiency)
        } else {
            DifficultyLevel::Medium
        };
        
        // Generate different types of questions
        let question_types = vec![
            QuestionType::Comprehension,
            QuestionType::Analysis,
            QuestionType::Application,
            QuestionType::Synthesis,
        ];
        
        for q_type in question_types.iter().take(self.config.max_questions) {
            if let Some(question) = self.generate_question_of_type(
                story_state,
                answer_state,
                q_type,
                &difficulty,
                memory,
            )? {
                if question.confidence >= self.config.min_confidence {
                    questions.push(question);
                }
            }
        }
        
        Ok(questions)
    }
    
    /// Generate specific type of question
    fn generate_question_of_type(
        &self,
        story_state: &ThoughtState,
        answer_state: Option<&ThoughtState>,
        question_type: &QuestionType,
        difficulty: &DifficultyLevel,
        memory: &SemanticMemory,
    ) -> Result<Option<GeneratedQuestion>, Box<dyn std::error::Error>> {
        // Find relevant concepts from story
        let concepts = memory.find_similar(&story_state.vector, 5)?;
        
        if concepts.is_empty() {
            return Ok(None);
        }
        
        // Extract key entities and concepts
        let key_concepts: Vec<String> = concepts
            .iter()
            .map(|(fact, _)| fact.concept.clone())
            .collect();
        
        // Generate question based on type
        let question_text = match question_type {
            QuestionType::Clarification => {
                self.generate_clarification(&key_concepts, difficulty)
            }
            QuestionType::Comprehension => {
                self.generate_comprehension(&key_concepts, difficulty)
            }
            QuestionType::Application => {
                self.generate_application(&key_concepts, difficulty)
            }
            QuestionType::Analysis => {
                self.generate_analysis(&key_concepts, difficulty)
            }
            QuestionType::Synthesis => {
                self.generate_synthesis(&key_concepts, difficulty)
            }
            QuestionType::Evaluation => {
                self.generate_evaluation(&key_concepts, difficulty)
            }
            QuestionType::FollowUp => {
                self.generate_followup(&key_concepts, difficulty)
            }
        };
        
        // Compute confidence based on concept relevance
        let confidence = concepts.iter()
            .map(|(_, sim)| sim)
            .sum::<f64>() / concepts.len() as f64;
        
        Ok(Some(GeneratedQuestion {
            question: question_text,
            question_type: question_type.clone(),
            difficulty: difficulty.clone(),
            expected_answer: None, // Could be computed from answer_state
            confidence,
        }))
    }
    
    /// Generate clarification question
    fn generate_clarification(&self, concepts: &[String], difficulty: &DifficultyLevel) -> String {
        if concepts.is_empty() {
            return "Can you clarify what you mean?".to_string();
        }
        
        let concept = &concepts[0];
        match difficulty {
            DifficultyLevel::Easy => format!("What is {}?", concept),
            DifficultyLevel::Medium => format!("Can you explain what you mean by {}?", concept),
            DifficultyLevel::Hard => format!("How would you define {} in this context?", concept),
            DifficultyLevel::Expert => format!("What are the key characteristics of {} that distinguish it from related concepts?", concept),
        }
    }
    
    /// Generate comprehension question
    fn generate_comprehension(&self, concepts: &[String], difficulty: &DifficultyLevel) -> String {
        if concepts.is_empty() {
            return "Can you explain what happened?".to_string();
        }
        
        let concept = &concepts[0];
        match difficulty {
            DifficultyLevel::Easy => format!("What happened with {}?", concept),
            DifficultyLevel::Medium => format!("Can you explain the role of {} in this situation?", concept),
            DifficultyLevel::Hard => format!("How does {} contribute to the overall outcome?", concept),
            DifficultyLevel::Expert => format!("What are the underlying mechanisms by which {} influences the system?", concept),
        }
    }
    
    /// Generate application question
    fn generate_application(&self, concepts: &[String], difficulty: &DifficultyLevel) -> String {
        if concepts.is_empty() {
            return "How would you apply this?".to_string();
        }
        
        let concept = &concepts[0];
        match difficulty {
            DifficultyLevel::Easy => format!("How would you use {}?", concept),
            DifficultyLevel::Medium => format!("Can you give an example of {} in a different context?", concept),
            DifficultyLevel::Hard => format!("How would you apply the principles of {} to solve a new problem?", concept),
            DifficultyLevel::Expert => format!("What are the implications of {} for related domains?", concept),
        }
    }
    
    /// Generate analysis question
    fn generate_analysis(&self, concepts: &[String], difficulty: &DifficultyLevel) -> String {
        if concepts.is_empty() {
            return "Why did this happen?".to_string();
        }
        
        let concept = &concepts[0];
        match difficulty {
            DifficultyLevel::Easy => format!("Why is {} important?", concept),
            DifficultyLevel::Medium => format!("What caused {} to happen?", concept),
            DifficultyLevel::Hard => format!("What are the underlying reasons for {}?", concept),
            DifficultyLevel::Expert => format!("How do the causal mechanisms of {} interact with other factors?", concept),
        }
    }
    
    /// Generate synthesis question
    fn generate_synthesis(&self, concepts: &[String], difficulty: &DifficultyLevel) -> String {
        if concepts.len() < 2 {
            return "How do these ideas connect?".to_string();
        }
        
        let concept1 = &concepts[0];
        let concept2 = &concepts[1];
        match difficulty {
            DifficultyLevel::Easy => format!("How are {} and {} related?", concept1, concept2),
            DifficultyLevel::Medium => format!("What is the connection between {} and {}?", concept1, concept2),
            DifficultyLevel::Hard => format!("How do {} and {} interact to produce the observed outcome?", concept1, concept2),
            DifficultyLevel::Expert => format!("What emergent properties arise from the interaction of {} and {}?", concept1, concept2),
        }
    }
    
    /// Generate evaluation question
    fn generate_evaluation(&self, concepts: &[String], difficulty: &DifficultyLevel) -> String {
        if concepts.is_empty() {
            return "What do you think about this?".to_string();
        }
        
        let concept = &concepts[0];
        match difficulty {
            DifficultyLevel::Easy => format!("Do you think {} is good?", concept),
            DifficultyLevel::Medium => format!("What is your opinion on {}?", concept),
            DifficultyLevel::Hard => format!("How would you evaluate the effectiveness of {}?", concept),
            DifficultyLevel::Expert => format!("What are the strengths and limitations of {} from multiple perspectives?", concept),
        }
    }
    
    /// Generate follow-up question
    fn generate_followup(&self, concepts: &[String], difficulty: &DifficultyLevel) -> String {
        if concepts.is_empty() {
            return "Can you tell me more?".to_string();
        }
        
        let concept = &concepts[0];
        match difficulty {
            DifficultyLevel::Easy => format!("Tell me more about {}", concept),
            DifficultyLevel::Medium => format!("What else can you share about {}?", concept),
            DifficultyLevel::Hard => format!("Can you elaborate on the implications of {}?", concept),
            DifficultyLevel::Expert => format!("What are the deeper theoretical foundations of {}?", concept),
        }
    }
    
    /// Estimate user proficiency from embedding
    fn estimate_proficiency(&self, user_embedding: Option<&[f64]>) -> f64 {
        if let Some(u) = user_embedding {
            // Simple heuristic: average of positive components
            let sum: f64 = u.iter().filter(|&&x| x > 0.0).sum();
            let count = u.iter().filter(|&&x| x > 0.0).count();
            if count > 0 {
                (sum / count as f64).min(1.0).max(0.0)
            } else {
                0.5 // Default to medium
            }
        } else {
            0.5 // Default to medium
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_question_generator() {
        let generator = QuestionGenerator::default(128);
        let story_state = ThoughtState::new(128);
        let memory = SemanticMemory::in_memory(128).unwrap();
        
        let questions = generator.generate_questions(
            &story_state,
            None,
            None,
            &memory,
        );
        
        assert!(questions.is_ok());
    }
    
    #[test]
    fn test_difficulty_levels() {
        assert_eq!(DifficultyLevel::Easy.to_score(), 0.2);
        assert_eq!(DifficultyLevel::Expert.to_score(), 1.0);
        
        let easy = DifficultyLevel::from_proficiency(0.1);
        assert!(matches!(easy, DifficultyLevel::Easy));
        
        let expert = DifficultyLevel::from_proficiency(0.95);
        assert!(matches!(expert, DifficultyLevel::Expert));
    }
    
    #[test]
    fn test_question_types() {
        let generator = QuestionGenerator::default(128);
        let concepts = vec!["gravity".to_string(), "mass".to_string()];
        
        let q1 = generator.generate_clarification(&concepts, &DifficultyLevel::Easy);
        assert!(q1.contains("gravity"));
        
        let q2 = generator.generate_synthesis(&concepts, &DifficultyLevel::Medium);
        assert!(q2.contains("gravity") && q2.contains("mass"));
    }
}
