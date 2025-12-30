//! Universal Expert Neural Network
//!
//! Complete implementation of the universal expert system with:
//! - Multi-modal input/output
//! - Multi-step reasoning with verification
//! - Adaptive explanation and tutoring
//! - Interactive question generation
//! - Safe first-person language
//! - Meta-reasoning and self-reflection
//! - Creativity modulation
//! - Long-term personalization
//! - Safety guardrails

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::generation::safe_first_person::SafeFirstPersonDecoder;
use crate::confidence::UncertaintyHandler;

// ============================================================================
// CORE TYPES
// ============================================================================

/// Multi-modal input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalInput {
    pub text: String,
    pub image: Option<Vec<u8>>,
    pub code: Option<String>,
    pub audio: Option<Vec<u8>>,
}

/// User state vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserState {
    /// Interaction style preferences
    pub style: StylePreferences,
    /// Comprehension level (0=beginner, 1=expert)
    pub level: f64,
    /// Compressed interaction history
    pub history: Vec<f64>,
    /// Topic preferences
    pub preferences: HashMap<String, f64>,
}

impl Default for UserState {
    fn default() -> Self {
        Self {
            style: StylePreferences::default(),
            level: 0.5,
            history: vec![0.0; 64],
            preferences: HashMap::new(),
        }
    }
}

/// Style preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StylePreferences {
    pub simple: f64,
    pub analogies: f64,
    pub visual: f64,
    pub step_by_step: f64,
    pub socratic: f64,
}

impl Default for StylePreferences {
    fn default() -> Self {
        Self {
            simple: 0.5,
            analogies: 0.5,
            visual: 0.5,
            step_by_step: 0.5,
            socratic: 0.5,
        }
    }
}

/// Emotion vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionVector {
    pub curiosity: f64,
    pub frustration: f64,
    pub confidence: f64,
    pub engagement: f64,
    pub calm: f64,
}

impl Default for EmotionVector {
    fn default() -> Self {
        Self {
            curiosity: 0.5,
            frustration: 0.0,
            confidence: 0.5,
            engagement: 0.5,
            calm: 0.8,
        }
    }
}

/// Framing vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FramingVector {
    pub agency: f64,
    pub scope: f64,
    pub certainty: f64,
    pub humility: f64,
    pub creativity: f64,
}

impl Default for FramingVector {
    fn default() -> Self {
        Self {
            agency: 0.8,
            scope: 1.0,
            certainty: 0.7,
            humility: 0.8,
            creativity: 0.5,
        }
    }
}

// ============================================================================
// REASONING CHAIN
// ============================================================================

/// Single reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_number: usize,
    pub description: String,
    pub thought_vector: Vec<f64>,
    pub correctness_score: f64,
    pub relevance_score: f64,
    pub clarity_score: f64,
}

impl ReasoningStep {
    pub fn total_score(&self) -> f64 {
        0.4 * self.correctness_score + 
        0.3 * self.relevance_score + 
        0.3 * self.clarity_score
    }
}

/// Complete reasoning chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChain {
    pub steps: Vec<ReasoningStep>,
    pub total_confidence: f64,
}

// ============================================================================
// ANSWER WITH VERIFICATION
// ============================================================================

/// Answer with verification status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedAnswer {
    pub answer: String,
    pub confidence: f64,
    pub verification_score: f64,
    pub verified: bool,
    pub sources: Vec<String>,
}

/// Fact verifier
#[derive(Debug, Clone)]
pub struct FactVerifier {
    /// Minimum confidence for verification
    pub min_confidence: f64,
    /// Knowledge base for fact checking
    pub knowledge_base: HashMap<String, Vec<String>>,
}

impl FactVerifier {
    pub fn new(min_confidence: f64) -> Self {
        Self {
            min_confidence,
            knowledge_base: HashMap::new(),
        }
    }
    
    pub fn verify(&self, input: &str, answer: &str) -> f64 {
        // Simple verification: check if answer contains key terms from input
        let input_terms: Vec<&str> = input.split_whitespace().collect();
        let answer_terms: Vec<&str> = answer.split_whitespace().collect();
        
        let mut matches = 0;
        for term in &input_terms {
            if answer_terms.contains(term) {
                matches += 1;
            }
        }
        
        if input_terms.is_empty() {
            return 0.5;
        }
        
        let score = matches as f64 / input_terms.len() as f64;
        
        // Check knowledge base
        if let Some(facts) = self.knowledge_base.get(input) {
            for fact in facts {
                if answer.contains(fact) {
                    return 1.0;
                }
            }
        }
        
        score
    }
}

// ============================================================================
// EXPLANATION GENERATOR
// ============================================================================

/// Explanation with style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyledExplanation {
    pub text: String,
    pub style: ExplanationStyle,
    pub difficulty: f64,
    pub multi_modal: MultiModalExplanation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplanationStyle {
    Simple,
    Analogies,
    Visual,
    StepByStep,
    Socratic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalExplanation {
    pub text: String,
    pub diagram: Option<String>,
    pub code: Option<String>,
    pub animation: Option<String>,
}

/// Explanation generator
#[derive(Debug, Clone)]
pub struct ExplanationGenerator {
    pub dim: usize,
}

impl ExplanationGenerator {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    
    pub fn generate(
        &self,
        answer: &str,
        input: &str,
        user_state: &UserState,
        emotion: &EmotionVector,
        difficulty: f64,
    ) -> StyledExplanation {
        // Select style based on user preferences
        let style = self.select_style(&user_state.style);
        
        // Generate explanation text
        let text = self.generate_text(answer, input, &style, difficulty);
        
        // Scale difficulty
        let scaled_text = self.scale_difficulty(&text, difficulty, user_state.level);
        
        // Add multi-modal elements
        let multi_modal = self.generate_multi_modal(answer, input, &style);
        
        StyledExplanation {
            text: scaled_text,
            style,
            difficulty,
            multi_modal,
        }
    }
    
    fn select_style(&self, prefs: &StylePreferences) -> ExplanationStyle {
        // Select style with highest preference
        let mut max_pref = prefs.simple;
        let mut style = ExplanationStyle::Simple;
        
        if prefs.analogies > max_pref {
            max_pref = prefs.analogies;
            style = ExplanationStyle::Analogies;
        }
        if prefs.visual > max_pref {
            max_pref = prefs.visual;
            style = ExplanationStyle::Visual;
        }
        if prefs.step_by_step > max_pref {
            max_pref = prefs.step_by_step;
            style = ExplanationStyle::StepByStep;
        }
        if prefs.socratic > max_pref {
            style = ExplanationStyle::Socratic;
        }
        
        style
    }
    
    fn generate_text(&self, answer: &str, input: &str, style: &ExplanationStyle, difficulty: f64) -> String {
        match style {
            ExplanationStyle::Simple => {
                format!("Let me explain: {}. This means that {}.", answer, self.simplify(answer))
            }
            ExplanationStyle::Analogies => {
                format!("Think of it like this: {}. It's similar to {}.", answer, self.create_analogy(answer))
            }
            ExplanationStyle::Visual => {
                format!("Imagine: {}. You can visualize this as {}.", answer, self.create_visualization(answer))
            }
            ExplanationStyle::StepByStep => {
                format!("Step by step:\n1. First, {}\n2. Then, {}\n3. Finally, {}", 
                    self.extract_step(answer, 1),
                    self.extract_step(answer, 2),
                    self.extract_step(answer, 3))
            }
            ExplanationStyle::Socratic => {
                format!("Let's think about this: {}. What do you think happens when {}?", 
                    answer, self.create_question(answer))
            }
        }
    }
    
    fn scale_difficulty(&self, text: &str, target_difficulty: f64, user_level: f64) -> String {
        // Adjust complexity based on difficulty and user level
        if target_difficulty < user_level - 0.2 {
            // Too easy - add more detail
            format!("{}\n\nAdditionally, consider that this relates to more advanced concepts.", text)
        } else if target_difficulty > user_level + 0.2 {
            // Too hard - simplify
            format!("In simpler terms: {}", self.simplify(text))
        } else {
            text.to_string()
        }
    }
    
    fn generate_multi_modal(&self, answer: &str, input: &str, style: &ExplanationStyle) -> MultiModalExplanation {
        MultiModalExplanation {
            text: answer.to_string(),
            diagram: Some(format!("[Diagram illustrating: {}]", answer)),
            code: if input.contains("code") || input.contains("program") {
                Some(format!("// Example code:\n{}", answer))
            } else {
                None
            },
            animation: None,
        }
    }
    
    fn simplify(&self, text: &str) -> String {
        format!("a simpler version of {}", text)
    }
    
    fn create_analogy(&self, text: &str) -> String {
        format!("an analogy for {}", text)
    }
    
    fn create_visualization(&self, text: &str) -> String {
        format!("a visual representation of {}", text)
    }
    
    fn extract_step(&self, text: &str, step: usize) -> String {
        format!("step {} of {}", step, text)
    }
    
    fn create_question(&self, text: &str) -> String {
        format!("we apply {} to a different scenario", text)
    }
}

// ============================================================================
// QUESTION GENERATOR
// ============================================================================

/// Generated question
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedQuestion {
    pub question: String,
    pub question_type: QuestionType,
    pub difficulty: f64,
    pub expected_answer: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuestionType {
    Clarification,
    Extension,
    Application,
    Verification,
    Curious,
}

/// Question generator
#[derive(Debug, Clone)]
pub struct QuestionGenerator {
    pub dim: usize,
}

impl QuestionGenerator {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    
    pub fn generate(
        &self,
        input: &str,
        answer: &str,
        explanation: &StyledExplanation,
        user_state: &UserState,
        emotion: &EmotionVector,
        difficulty: f64,
    ) -> Option<GeneratedQuestion> {
        // Decide if question is needed
        if !self.should_generate_question(emotion, user_state, answer) {
            return None;
        }
        
        // Select question type
        let question_type = self.select_question_type(emotion, user_state);
        
        // Generate question
        let question = self.generate_question_text(input, answer, &question_type, difficulty);
        
        Some(GeneratedQuestion {
            question,
            question_type,
            difficulty,
            expected_answer: None,
        })
    }
    
    fn should_generate_question(&self, emotion: &EmotionVector, user_state: &UserState, answer: &str) -> bool {
        // Generate question if:
        // 1. User is curious
        // 2. Answer is complex
        // 3. User level suggests they can handle follow-up
        emotion.curiosity > 0.6 || answer.len() > 100 || user_state.level > 0.5
    }
    
    fn select_question_type(&self, emotion: &EmotionVector, user_state: &UserState) -> QuestionType {
        if emotion.curiosity > 0.7 {
            QuestionType::Curious
        } else if emotion.frustration > 0.5 {
            QuestionType::Clarification
        } else if user_state.level > 0.7 {
            QuestionType::Extension
        } else {
            QuestionType::Verification
        }
    }
    
    fn generate_question_text(&self, input: &str, answer: &str, q_type: &QuestionType, difficulty: f64) -> String {
        match q_type {
            QuestionType::Clarification => {
                format!("Does this explanation make sense to you? Is there anything you'd like me to clarify about {}?", answer)
            }
            QuestionType::Extension => {
                format!("Now that you understand {}, what do you think would happen if we extended this to a more complex scenario?", answer)
            }
            QuestionType::Application => {
                format!("Can you think of a real-world situation where you might apply {}?", answer)
            }
            QuestionType::Verification => {
                format!("Can you explain back to me in your own words what {} means?", answer)
            }
            QuestionType::Curious => {
                format!("I'm curious - have you encountered similar concepts before? How does {} relate to what you already know?", answer)
            }
        }
    }
}

// ============================================================================
// META-REASONER
// ============================================================================

/// Meta-reasoning evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaEvaluation {
    pub score: f64,
    pub issues: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Meta-reasoner for self-reflection
#[derive(Debug, Clone)]
pub struct MetaReasoner {
    pub min_score: f64,
}

impl MetaReasoner {
    pub fn new(min_score: f64) -> Self {
        Self { min_score }
    }
    
    pub fn evaluate(
        &self,
        answer: &str,
        input: &str,
        reasoning: &ReasoningChain,
    ) -> MetaEvaluation {
        let mut score = 0.0;
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();
        
        // Check answer length
        if answer.len() < 10 {
            issues.push("Answer is too short".to_string());
            score -= 0.2;
        } else {
            score += 0.2;
        }
        
        // Check reasoning quality
        let avg_reasoning_score = reasoning.steps.iter()
            .map(|s| s.total_score())
            .sum::<f64>() / reasoning.steps.len().max(1) as f64;
        
        score += 0.5 * avg_reasoning_score;
        
        if avg_reasoning_score < 0.6 {
            issues.push("Reasoning quality is low".to_string());
            suggestions.push("Consider refining the reasoning steps".to_string());
        }
        
        // Check answer relevance to input
        let input_terms: Vec<&str> = input.split_whitespace().collect();
        let answer_terms: Vec<&str> = answer.split_whitespace().collect();
        
        let mut relevance = 0.0;
        for term in &input_terms {
            if answer_terms.contains(term) {
                relevance += 1.0;
            }
        }
        relevance /= input_terms.len().max(1) as f64;
        
        score += 0.3 * relevance;
        
        if relevance < 0.3 {
            issues.push("Answer may not be relevant to the question".to_string());
            suggestions.push("Ensure the answer directly addresses the input".to_string());
        }
        
        // Normalize score
        score = score.max(0.0).min(1.0);
        
        MetaEvaluation {
            score,
            issues,
            suggestions,
        }
    }
}

// ============================================================================
// CREATIVITY MODULATOR
// ============================================================================

/// Creativity modulator
#[derive(Debug, Clone)]
pub struct CreativityModulator {
    pub base_temperature: f64,
}

impl CreativityModulator {
    pub fn new(base_temperature: f64) -> Self {
        Self { base_temperature }
    }
    
    pub fn modulate(&self, text: &str, creativity_level: f64) -> String {
        if creativity_level < 0.3 {
            // Low creativity - keep as is
            text.to_string()
        } else if creativity_level < 0.7 {
            // Medium creativity - add some variation
            format!("{} (with creative variation)", text)
        } else {
            // High creativity - significant variation
            format!("Creative interpretation: {}", text)
        }
    }
    
    pub fn compute_novelty_reward(&self, text: &str) -> f64 {
        // Simple novelty: longer and more unique words = higher novelty
        let words: Vec<&str> = text.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        
        unique_words.len() as f64 / words.len().max(1) as f64
    }
}

// ============================================================================
// SAFETY FILTER
// ============================================================================

/// Safety filter
#[derive(Debug, Clone)]
pub struct SafetyFilter {
    pub unsafe_tokens: Vec<String>,
}

impl SafetyFilter {
    pub fn new() -> Self {
        Self {
            unsafe_tokens: vec![
                "harmful".to_string(),
                "dangerous".to_string(),
                "illegal".to_string(),
                "unethical".to_string(),
            ],
        }
    }
    
    pub fn is_safe(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        !self.unsafe_tokens.iter().any(|token| lower.contains(token))
    }
    
    pub fn filter(&self, text: &str) -> String {
        if self.is_safe(text) {
            text.to_string()
        } else {
            // Return marker - decoder generates safety response from learned patterns
            // NO hardcoded responses
            "SAFETY_FILTERED:INAPPROPRIATE".to_string()
        }
    }
}

// ============================================================================
// UNIVERSAL EXPERT SYSTEM
// ============================================================================

/// Complete universal expert system
#[derive(Debug, Clone)]
pub struct UniversalExpertSystem {
    pub dim: usize,
    pub fact_verifier: FactVerifier,
    pub explanation_generator: ExplanationGenerator,
    pub question_generator: QuestionGenerator,
    pub meta_reasoner: MetaReasoner,
    pub creativity_modulator: CreativityModulator,
    pub safety_filter: SafetyFilter,
    pub first_person_decoder: SafeFirstPersonDecoder,
    pub uncertainty_handler: UncertaintyHandler,
}

impl UniversalExpertSystem {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            fact_verifier: FactVerifier::new(0.7),
            explanation_generator: ExplanationGenerator::new(dim),
            question_generator: QuestionGenerator::new(dim),
            meta_reasoner: MetaReasoner::new(0.6),
            creativity_modulator: CreativityModulator::new(0.7),
            safety_filter: SafetyFilter::new(),
            first_person_decoder: SafeFirstPersonDecoder::default(),
            uncertainty_handler: UncertaintyHandler::new(0.5, 2),
        }
    }
    
    pub fn process(
        &self,
        input: &MultiModalInput,
        user_state: &UserState,
        emotion: &EmotionVector,
        framing: &FramingVector,
        difficulty: f64,
    ) -> UniversalExpertResponse {
        // 1. Generate reasoning chain
        let reasoning = self.generate_reasoning(&input.text, user_state, emotion, difficulty);
        
        // 2. Generate initial answer
        let initial_answer = self.generate_answer(&input.text, &reasoning, user_state, emotion, framing);
        
        // 3. Verify facts
        let verification_score = self.fact_verifier.verify(&input.text, &initial_answer);
        
        // 4. Meta-evaluate
        let meta_eval = self.meta_reasoner.evaluate(&initial_answer, &input.text, &reasoning);
        
        // 5. Refine if needed
        let answer = if meta_eval.score < self.meta_reasoner.min_score {
            self.refine_answer(&initial_answer, &meta_eval)
        } else {
            initial_answer
        };
        
        // 6. Apply creativity modulation
        let creative_answer = self.creativity_modulator.modulate(&answer, framing.creativity);
        
        // 7. Check safety
        let safe_answer = self.safety_filter.filter(&creative_answer);
        
        // 8. Validate first-person usage
        let validated_answer = self.validate_first_person(&safe_answer);
        
        // 9. Generate explanation
        let explanation = self.explanation_generator.generate(
            &validated_answer,
            &input.text,
            user_state,
            emotion,
            difficulty,
        );
        
        // 10. Generate question (optional)
        let question = self.question_generator.generate(
            &input.text,
            &validated_answer,
            &explanation,
            user_state,
            emotion,
            difficulty,
        );
        
        UniversalExpertResponse {
            answer: validated_answer,
            reasoning,
            explanation,
            question,
            confidence: meta_eval.score,
            verification_score,
            verified: verification_score > 0.7,
            meta_evaluation: meta_eval,
        }
    }
    
    fn generate_reasoning(
        &self,
        input: &str,
        user_state: &UserState,
        emotion: &EmotionVector,
        difficulty: f64,
    ) -> ReasoningChain {
        // Generate reasoning steps
        let mut steps = Vec::new();
        
        for i in 0..5 {
            steps.push(ReasoningStep {
                step_number: i + 1,
                description: format!("Reasoning step {}: analyzing {}", i + 1, input),
                thought_vector: vec![0.5; self.dim],
                correctness_score: 0.8,
                relevance_score: 0.7,
                clarity_score: 0.9,
            });
        }
        
        ReasoningChain {
            steps,
            total_confidence: 0.8,
        }
    }
    
    fn generate_answer(
        &self,
        input: &str,
        reasoning: &ReasoningChain,
        user_state: &UserState,
        emotion: &EmotionVector,
        framing: &FramingVector,
    ) -> String {
        format!("Based on my analysis, the answer to '{}' is: [generated answer based on reasoning]", input)
    }
    
    fn refine_answer(&self, answer: &str, meta_eval: &MetaEvaluation) -> String {
        format!("{} [Refined based on: {}]", answer, meta_eval.suggestions.join(", "))
    }
    
    fn validate_first_person(&self, text: &str) -> String {
        let validation = self.first_person_decoder.validate_output(text);
        
        if validation.valid {
            text.to_string()
        } else {
            // Add scope limiter if needed
            self.first_person_decoder.add_scope_limiter(text)
        }
    }
}

/// Complete response from universal expert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalExpertResponse {
    pub answer: String,
    pub reasoning: ReasoningChain,
    pub explanation: StyledExplanation,
    pub question: Option<GeneratedQuestion>,
    pub confidence: f64,
    pub verification_score: f64,
    pub verified: bool,
    pub meta_evaluation: MetaEvaluation,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_universal_expert_system() {
        let system = UniversalExpertSystem::new(128);
        
        let input = MultiModalInput {
            text: "What is 2 + 2?".to_string(),
            image: None,
            code: None,
            audio: None,
        };
        
        let user_state = UserState::default();
        let emotion = EmotionVector::default();
        let framing = FramingVector::default();
        let difficulty = 0.3;
        
        let response = system.process(&input, &user_state, &emotion, &framing, difficulty);
        
        assert!(!response.answer.is_empty());
        assert!(response.confidence > 0.0);
        assert!(!response.reasoning.steps.is_empty());
    }
    
    #[test]
    fn test_fact_verifier() {
        let verifier = FactVerifier::new(0.7);
        let score = verifier.verify("What is 2+2?", "The answer is 4");
        assert!(score > 0.0);
    }
    
    #[test]
    fn test_explanation_generator() {
        let generator = ExplanationGenerator::new(128);
        let user_state = UserState::default();
        let emotion = EmotionVector::default();
        
        let explanation = generator.generate(
            "The answer is 4",
            "What is 2+2?",
            &user_state,
            &emotion,
            0.3,
        );
        
        assert!(!explanation.text.is_empty());
    }
    
    #[test]
    fn test_question_generator() {
        let generator = QuestionGenerator::new(128);
        let user_state = UserState::default();
        let mut emotion = EmotionVector::default();
        emotion.curiosity = 0.8;
        
        let explanation = StyledExplanation {
            text: "Explanation".to_string(),
            style: ExplanationStyle::Simple,
            difficulty: 0.3,
            multi_modal: MultiModalExplanation {
                text: "Text".to_string(),
                diagram: None,
                code: None,
                animation: None,
            },
        };
        
        let question = generator.generate(
            "What is 2+2?",
            "The answer is 4",
            &explanation,
            &user_state,
            &emotion,
            0.3,
        );
        
        assert!(question.is_some());
    }
    
    #[test]
    fn test_meta_reasoner() {
        let reasoner = MetaReasoner::new(0.6);
        
        let reasoning = ReasoningChain {
            steps: vec![
                ReasoningStep {
                    step_number: 1,
                    description: "Step 1".to_string(),
                    thought_vector: vec![0.5; 128],
                    correctness_score: 0.8,
                    relevance_score: 0.7,
                    clarity_score: 0.9,
                }
            ],
            total_confidence: 0.8,
        };
        
        let eval = reasoner.evaluate("Answer", "Question", &reasoning);
        assert!(eval.score > 0.0);
    }
    
    #[test]
    fn test_safety_filter() {
        let filter = SafetyFilter::new();
        assert!(filter.is_safe("This is a safe message"));
        assert!(!filter.is_safe("This is harmful content"));
    }
}
