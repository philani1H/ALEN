//! Multi-Modal Explanation Generator
//!
//! Generates explanations in multiple formats:
//! - Text (adapted to audience)
//! - Visual (diagrams, charts, ASCII art)
//! - Analogies (domain mapping)
//! - Stepwise (logical breakdown)
//! - Examples (concrete instances)

use crate::api::user_modeling::UserState;
use crate::core::ThoughtState;
use crate::memory::SemanticMemory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// PART 1: COMPLETE EXPLANATION STRUCTURE
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteExplanation {
    /// Text explanation
    pub text: String,
    
    /// Visual representation (ASCII art, diagram description)
    pub visual: Option<VisualExplanation>,
    
    /// Analogies
    pub analogies: Vec<Analogy>,
    
    /// Stepwise breakdown
    pub steps: Vec<ReasoningStep>,
    
    /// Concrete examples
    pub examples: Vec<Example>,
    
    /// Was adapted for audience
    pub audience_adapted: bool,
    
    /// Explanation ID for tracking
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualExplanation {
    /// Type of visual
    pub visual_type: VisualType,
    
    /// ASCII art or diagram description
    pub content: String,
    
    /// Caption
    pub caption: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualType {
    Diagram,
    Chart,
    Graph,
    Tree,
    Table,
    AsciiArt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Analogy {
    /// Abstract concept
    pub abstract_concept: String,
    
    /// Concrete analogy
    pub concrete_analogy: String,
    
    /// Mapping explanation
    pub mapping: String,
    
    /// Appropriateness score
    pub appropriateness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Step number
    pub number: usize,
    
    /// Step description
    pub description: String,
    
    /// Why this step is needed
    pub justification: String,
    
    /// Result of this step
    pub result: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Example description
    pub description: String,
    
    /// Input
    pub input: String,
    
    /// Output
    pub output: String,
    
    /// Explanation of why
    pub explanation: String,
    
    /// Difficulty level (0-1)
    pub difficulty: f64,
}

// ============================================================================
// PART 2: VISUAL EXPLANATION GENERATOR
// ============================================================================

pub struct VisualExplanationGenerator;

impl VisualExplanationGenerator {
    pub fn new() -> Self {
        Self
    }
    
    /// Generate visual explanation for a concept
    pub fn generate(&self, concept: &str, solution: &str) -> Option<VisualExplanation> {
        // Detect if visual would be helpful
        if self.should_generate_visual(concept) {
            Some(self.create_visual(concept, solution))
        } else {
            None
        }
    }
    
    fn should_generate_visual(&self, concept: &str) -> bool {
        let lower = concept.to_lowercase();
        
        // Math operations benefit from visual
        lower.contains("add") || lower.contains("subtract") ||
        lower.contains("multiply") || lower.contains("divide") ||
        lower.contains("graph") || lower.contains("tree") ||
        lower.contains("diagram") || lower.contains("flow")
    }
    
    fn create_visual(&self, concept: &str, solution: &str) -> VisualExplanation {
        let lower = concept.to_lowercase();
        
        if lower.contains("add") || lower.contains("+") {
            self.create_addition_visual(concept, solution)
        } else if lower.contains("tree") {
            self.create_tree_visual(concept, solution)
        } else {
            self.create_generic_visual(concept, solution)
        }
    }
    
    fn create_addition_visual(&self, _concept: &str, solution: &str) -> VisualExplanation {
        let visual = format!(
            r#"
    ●●        ●●
    ●●   +    ●●   =   ●●●●
                        ●●●●
    
    2    +    2    =    4
"#
        );
        
        VisualExplanation {
            visual_type: VisualType::Diagram,
            content: visual,
            caption: format!("Visual representation: {}", solution),
        }
    }
    
    fn create_tree_visual(&self, _concept: &str, _solution: &str) -> VisualExplanation {
        let visual = r#"
        Root
       /    \
      A      B
     / \    / \
    C   D  E   F
"#.to_string();
        
        VisualExplanation {
            visual_type: VisualType::Tree,
            content: visual,
            caption: "Tree structure visualization".to_string(),
        }
    }
    
    fn create_generic_visual(&self, concept: &str, _solution: &str) -> VisualExplanation {
        VisualExplanation {
            visual_type: VisualType::Diagram,
            content: format!("[Diagram for: {}]", concept),
            caption: "Conceptual diagram".to_string(),
        }
    }
}

impl Default for VisualExplanationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PART 3: ANALOGY GENERATOR
// ============================================================================

pub struct AnalogyGenerator {
    /// Analogy database
    analogies: HashMap<String, Vec<Analogy>>,
}

impl AnalogyGenerator {
    pub fn new() -> Self {
        let mut analogies = HashMap::new();
        
        // Math analogies
        analogies.insert("addition".to_string(), vec![
            Analogy {
                abstract_concept: "Addition combines numbers".to_string(),
                concrete_analogy: "Like putting apples in a basket".to_string(),
                mapping: "Numbers are like apples, adding is like putting them together".to_string(),
                appropriateness: 0.9,
            },
            Analogy {
                abstract_concept: "Addition is commutative".to_string(),
                concrete_analogy: "Like mixing colors - red + blue = blue + red".to_string(),
                mapping: "Order doesn't matter in both cases".to_string(),
                appropriateness: 0.8,
            },
        ]);
        
        // Programming analogies
        analogies.insert("function".to_string(), vec![
            Analogy {
                abstract_concept: "Function takes input and produces output".to_string(),
                concrete_analogy: "Like a vending machine - you put money in, get snack out".to_string(),
                mapping: "Input is money, output is snack, function is the machine".to_string(),
                appropriateness: 0.9,
            },
        ]);
        
        // Logic analogies
        analogies.insert("if-then".to_string(), vec![
            Analogy {
                abstract_concept: "If condition then consequence".to_string(),
                concrete_analogy: "If it rains, then the ground gets wet".to_string(),
                mapping: "Condition is rain, consequence is wet ground".to_string(),
                appropriateness: 0.95,
            },
        ]);
        
        Self { analogies }
    }
    
    /// Generate analogies for a concept
    pub fn generate(&self, concept: &str, user: &UserState, max_count: usize) -> Vec<Analogy> {
        let lower = concept.to_lowercase();
        
        // Find matching analogies
        let mut matches = Vec::new();
        
        for (key, analogy_list) in &self.analogies {
            if lower.contains(key) {
                matches.extend(analogy_list.clone());
            }
        }
        
        // Filter by user's technical level
        let technical_level = user.preferences.technical_level.mean();
        matches.retain(|a| {
            // Higher technical level = can handle more abstract analogies
            a.appropriateness >= (1.0 - technical_level) * 0.5
        });
        
        // Sort by appropriateness
        matches.sort_by(|a, b| b.appropriateness.partial_cmp(&a.appropriateness).unwrap());
        
        // Return top N
        matches.into_iter().take(max_count).collect()
    }
}

impl Default for AnalogyGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PART 4: STEPWISE GENERATOR
// ============================================================================

pub struct StepwiseGenerator;

impl StepwiseGenerator {
    pub fn new() -> Self {
        Self
    }
    
    /// Break down solution into logical steps
    pub fn break_down(&self, solution: &str) -> Vec<ReasoningStep> {
        // Parse solution into steps
        let sentences: Vec<&str> = solution.split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .collect();
        
        sentences.iter().enumerate().map(|(i, sentence)| {
            ReasoningStep {
                number: i + 1,
                description: sentence.trim().to_string(),
                justification: self.generate_justification(sentence),
                result: self.extract_result(sentence),
            }
        }).collect()
    }
    
    fn generate_justification(&self, sentence: &str) -> String {
        // Simple heuristic justifications
        if sentence.contains("add") || sentence.contains("+") {
            "We add these numbers to combine them".to_string()
        } else if sentence.contains("subtract") || sentence.contains("-") {
            "We subtract to find the difference".to_string()
        } else if sentence.contains("multiply") || sentence.contains("×") {
            "We multiply to find the total".to_string()
        } else if sentence.contains("divide") || sentence.contains("÷") {
            "We divide to split into equal parts".to_string()
        } else {
            "This step follows from the previous reasoning".to_string()
        }
    }
    
    fn extract_result(&self, sentence: &str) -> Option<String> {
        // Try to extract numeric result
        let words: Vec<&str> = sentence.split_whitespace().collect();
        
        for word in words {
            if let Ok(_num) = word.parse::<f64>() {
                return Some(word.to_string());
            }
        }
        
        None
    }
}

impl Default for StepwiseGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PART 5: EXAMPLE GENERATOR
// ============================================================================

pub struct ExampleGenerator {
    /// Example database
    examples: HashMap<String, Vec<Example>>,
}

impl ExampleGenerator {
    pub fn new() -> Self {
        let mut examples = HashMap::new();
        
        // Addition examples
        examples.insert("addition".to_string(), vec![
            Example {
                description: "Simple addition".to_string(),
                input: "1 + 1".to_string(),
                output: "2".to_string(),
                explanation: "One plus one equals two".to_string(),
                difficulty: 0.1,
            },
            Example {
                description: "Adding larger numbers".to_string(),
                input: "5 + 7".to_string(),
                output: "12".to_string(),
                explanation: "Five plus seven equals twelve".to_string(),
                difficulty: 0.3,
            },
            Example {
                description: "Adding with carrying".to_string(),
                input: "28 + 47".to_string(),
                output: "75".to_string(),
                explanation: "Twenty-eight plus forty-seven equals seventy-five".to_string(),
                difficulty: 0.6,
            },
        ]);
        
        Self { examples }
    }
    
    /// Generate examples for a concept
    pub fn generate(&self, concept: &str, technical_level: f64, max_count: usize) -> Vec<Example> {
        let lower = concept.to_lowercase();
        
        // Find matching examples
        let mut matches = Vec::new();
        
        for (key, example_list) in &self.examples {
            if lower.contains(key) {
                matches.extend(example_list.clone());
            }
        }
        
        // Filter by difficulty
        matches.retain(|e| e.difficulty <= technical_level + 0.2);
        
        // Sort by difficulty
        matches.sort_by(|a, b| a.difficulty.partial_cmp(&b.difficulty).unwrap());
        
        // Return top N
        matches.into_iter().take(max_count).collect()
    }
}

impl Default for ExampleGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PART 6: MULTI-MODAL EXPLANATION GENERATOR
// ============================================================================

pub struct MultiModalExplanationGenerator {
    /// Visual generator
    visual_generator: VisualExplanationGenerator,
    
    /// Analogy generator
    analogy_generator: AnalogyGenerator,
    
    /// Stepwise generator
    stepwise_generator: StepwiseGenerator,
    
    /// Example generator
    example_generator: ExampleGenerator,
}

impl MultiModalExplanationGenerator {
    pub fn new() -> Self {
        Self {
            visual_generator: VisualExplanationGenerator::new(),
            analogy_generator: AnalogyGenerator::new(),
            stepwise_generator: StepwiseGenerator::new(),
            example_generator: ExampleGenerator::new(),
        }
    }
    
    /// Generate complete multi-modal explanation
    pub fn generate_complete_explanation(
        &self,
        concept: &str,
        solution: &str,
        user: &UserState,
    ) -> CompleteExplanation {
        // Generate visual if helpful
        let visual = self.visual_generator.generate(concept, solution);
        
        // Generate analogies for complex concepts
        let analogies = if self.is_complex(concept) {
            self.analogy_generator.generate(concept, user, 3)
        } else {
            vec![]
        };
        
        // Generate stepwise breakdown
        let steps = self.stepwise_generator.break_down(solution);
        
        // Generate examples
        let technical_level = user.preferences.technical_level.mean();
        let examples = self.example_generator.generate(concept, technical_level, 2);
        
        CompleteExplanation {
            text: solution.to_string(),
            visual,
            analogies,
            steps,
            examples,
            audience_adapted: true,
            id: Self::generate_id(),
        }
    }
    
    fn is_complex(&self, concept: &str) -> bool {
        // Heuristic: long concepts or technical terms are complex
        concept.len() > 50 || 
        concept.to_lowercase().contains("theorem") ||
        concept.to_lowercase().contains("algorithm") ||
        concept.to_lowercase().contains("function")
    }
    
    fn generate_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        format!("exp_{}", timestamp)
    }
}

impl Default for MultiModalExplanationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_visual_generator() {
        let generator = VisualExplanationGenerator::new();
        let visual = generator.generate("add 2+2", "4");
        assert!(visual.is_some());
    }
    
    #[test]
    fn test_analogy_generator() {
        let generator = AnalogyGenerator::new();
        let user = UserState::new("test_user".to_string(), 128);
        let analogies = generator.generate("addition", &user, 2);
        assert!(!analogies.is_empty());
    }
    
    #[test]
    fn test_stepwise_generator() {
        let generator = StepwiseGenerator::new();
        let steps = generator.break_down("First add 2 and 2. Then we get 4.");
        assert_eq!(steps.len(), 2);
    }
}
