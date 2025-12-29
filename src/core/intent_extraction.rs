//! Intent Extraction Module
//!
//! Implements prompt understanding as a mathematical object:
//! I = (τ, θ, C) where:
//! - τ = task vector (what to do)
//! - θ = target variables (what it's about)  
//! - C = constraint set (how to do it)
//!
//! Core insight: Answer intent, not text.
//! P(y|x) → P(y|I) where I = f_intent(x)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// The Intent State: I = (τ, θ, C)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentState {
    pub task: TaskVector,
    pub targets: Vec<TargetVariable>,
    pub constraints: ConstraintSet,
    pub confidence: f64,
    pub original_prompt: String,
}

/// Task types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum TaskType {
    Explain, Summarize, Solve, Decide, Design, Debug,
    Compare, List, Define, Translate, Generate, Analyze, Verify,
    #[default] Unknown,
}

impl TaskType {
    pub fn all() -> Vec<TaskType> {
        vec![TaskType::Explain, TaskType::Summarize, TaskType::Solve,
             TaskType::Decide, TaskType::Design, TaskType::Debug,
             TaskType::Compare, TaskType::List, TaskType::Define,
             TaskType::Translate, TaskType::Generate, TaskType::Analyze,
             TaskType::Verify, TaskType::Unknown]
    }

    pub fn keywords(&self) -> Vec<&'static str> {
        match self {
            TaskType::Explain => vec!["explain", "describe", "what is", "how does", "why"],
            TaskType::Summarize => vec!["summarize", "summary", "brief", "tldr"],
            TaskType::Solve => vec!["solve", "calculate", "compute", "find"],
            TaskType::Decide => vec!["should i", "which is better", "decide", "choose"],
            TaskType::Design => vec!["design", "create", "build", "architect"],
            TaskType::Debug => vec!["debug", "fix", "error", "bug", "not working"],
            TaskType::Compare => vec!["compare", "difference", "vs", "versus"],
            TaskType::List => vec!["list", "enumerate", "give me", "what are"],
            TaskType::Define => vec!["define", "definition", "meaning of"],
            TaskType::Translate => vec!["translate", "convert", "transform"],
            TaskType::Generate => vec!["generate", "write", "create", "make"],
            TaskType::Analyze => vec!["analyze", "analysis", "examine", "evaluate"],
            TaskType::Verify => vec!["verify", "check", "is it true", "correct"],
            TaskType::Unknown => vec![],
        }
    }
}

/// Task vector: τ = argmax P(τ|z)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskVector {
    pub primary_task: TaskType,
    pub primary_confidence: f64,
    pub secondary_task: Option<TaskType>,
    pub task_distribution: HashMap<TaskType, f64>,
}

impl Default for TaskVector {
    fn default() -> Self {
        Self {
            primary_task: TaskType::Unknown,
            primary_confidence: 0.0,
            secondary_task: None,
            task_distribution: HashMap::new(),
        }
    }
}

/// Target variable θ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetVariable {
    pub name: String,
    pub target_type: TargetType,
    pub importance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TargetType {
    #[default] Concept, Quantity, Entity, Process, Relationship, Property, Code, Decision,
}

/// Constraint set C
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintSet {
    pub format: FormatConstraints,
    pub length: LengthConstraints,
    pub style: StyleConstraints,
    pub content: ContentConstraints,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FormatConstraints {
    pub required_format: Option<OutputFormat>,
    pub include_code: Option<bool>,
    pub include_math: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OutputFormat {
    #[default] Prose, List, Table, Code, Math, StepByStep, Json,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LengthConstraints {
    pub max_words: Option<usize>,
    pub min_words: Option<usize>,
    pub target_length: Option<LengthCategory>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LengthCategory {
    OneWord, OneSentence, Brief, Short, Medium, Long, Comprehensive,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StyleConstraints {
    pub formality: Option<f64>,
    pub technical_level: Option<f64>,
    pub use_examples: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContentConstraints {
    pub must_include: Vec<String>,
    pub must_exclude: Vec<String>,
    pub focus_on: Vec<String>,
}

/// Intent Extractor: I = f_intent(x)
pub struct IntentExtractor {
    dim: usize,
}

impl IntentExtractor {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Extract intent from prompt
    pub fn extract(&self, prompt: &str) -> IntentState {
        let task = self.extract_task(prompt);
        let targets = self.extract_targets(prompt);
        let constraints = self.extract_constraints(prompt);
        let confidence = self.compute_confidence(&task, &targets);

        IntentState {
            task,
            targets,
            constraints,
            confidence,
            original_prompt: prompt.to_string(),
        }
    }

    fn extract_task(&self, prompt: &str) -> TaskVector {
        let lower = prompt.to_lowercase();
        let mut distribution = HashMap::new();

        for task_type in TaskType::all() {
            let mut score: f64 = 0.0;
            for keyword in task_type.keywords() {
                if lower.contains(keyword) {
                    score += 0.3;
                }
            }
            distribution.insert(task_type, score.min(1.0));
        }

        let total: f64 = distribution.values().sum();
        if total > 0.0 {
            for v in distribution.values_mut() {
                *v /= total;
            }
        }

        let (primary, confidence) = distribution.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(t, c)| (*t, *c))
            .unwrap_or((TaskType::Unknown, 0.0));

        TaskVector {
            primary_task: primary,
            primary_confidence: confidence,
            secondary_task: None,
            task_distribution: distribution,
        }
    }

    fn extract_targets(&self, prompt: &str) -> Vec<TargetVariable> {
        let words: Vec<&str> = prompt.split_whitespace().collect();
        let skip = ["the", "a", "an", "is", "are", "to", "of", "in", "for", "with"];
        
        words.iter()
            .filter(|w| w.len() > 2 && !skip.contains(&w.to_lowercase().as_str()))
            .take(5)
            .map(|w| TargetVariable {
                name: w.to_string(),
                target_type: TargetType::Concept,
                importance: 0.5,
            })
            .collect()
    }

    fn extract_constraints(&self, prompt: &str) -> ConstraintSet {
        let lower = prompt.to_lowercase();
        let mut constraints = ConstraintSet::default();

        if lower.contains("brief") || lower.contains("short") {
            constraints.length.target_length = Some(LengthCategory::Brief);
            constraints.length.max_words = Some(100);
        }
        if lower.contains("step by step") {
            constraints.format.required_format = Some(OutputFormat::StepByStep);
        }
        if lower.contains("math") || lower.contains("equation") {
            constraints.format.include_math = Some(true);
        }
        if lower.contains("code") {
            constraints.format.include_code = Some(true);
        }
        if lower.contains("without theory") {
            constraints.content.must_exclude.push("theory".to_string());
        }

        constraints
    }

    fn compute_confidence(&self, task: &TaskVector, targets: &[TargetVariable]) -> f64 {
        let task_conf = task.primary_confidence * 0.5;
        let target_conf = if targets.is_empty() { 0.3 } else { 0.5 };
        task_conf + target_conf
    }
}

/// Response energy: E(y) = α·irrelevance + β·verbosity + γ·constraint_violation
#[derive(Debug, Clone)]
pub struct ResponseEnergy {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl Default for ResponseEnergy {
    fn default() -> Self {
        Self { alpha: 0.4, beta: 0.3, gamma: 0.3 }
    }
}

impl ResponseEnergy {
    pub fn compute(&self, intent: &IntentState, response: &str) -> f64 {
        let word_count = response.split_whitespace().count();
        
        // Verbosity penalty
        let verbosity = if let Some(max) = intent.constraints.length.max_words {
            if word_count > max {
                (word_count - max) as f64 / max as f64
            } else { 0.0 }
        } else { 0.0 };

        // Simple energy
        self.beta * verbosity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_extraction() {
        let extractor = IntentExtractor::new(64);
        let intent = extractor.extract("Explain neural networks briefly using math.");
        
        assert_eq!(intent.task.primary_task, TaskType::Explain);
        assert!(intent.constraints.format.include_math == Some(true));
    }

    #[test]
    fn test_response_energy() {
        let extractor = IntentExtractor::new(64);
        let energy = ResponseEnergy::default();
        
        let intent = extractor.extract("What is 2+2? Answer briefly.");
        let short = "4";
        let long = "Well, let me explain. The answer is 4 because...";
        
        assert!(energy.compute(&intent, short) < energy.compute(&intent, long));
    }
}
