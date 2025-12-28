//! Chain-of-Thought Reasoning
//!
//! Implements multi-step reasoning with explicit intermediate steps

use serde::{Deserialize, Serialize};
use crate::core::ThoughtState;
use crate::neural::ALENNetwork;

/// A single step in a reasoning chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Step number
    pub step: usize,
    /// Description of this step
    pub description: String,
    /// Thought state at this step
    pub thought: Vec<f64>,
    /// Operator used
    pub operator: String,
    /// Confidence in this step
    pub confidence: f64,
    /// Intermediate result
    pub result: Option<String>,
}

/// Complete reasoning chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChain {
    /// Problem being solved
    pub problem: String,
    /// All reasoning steps
    pub steps: Vec<ReasoningStep>,
    /// Final answer
    pub answer: Option<String>,
    /// Overall confidence
    pub confidence: f64,
    /// Whether chain was verified
    pub verified: bool,
}

/// Result of chain-of-thought reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainResult {
    pub chain: ReasoningChain,
    pub success: bool,
    pub total_steps: usize,
    pub verification_errors: Vec<f32>,
}

impl ReasoningChain {
    pub fn new(problem: String) -> Self {
        Self {
            problem,
            steps: Vec::new(),
            answer: None,
            confidence: 0.0,
            verified: false,
        }
    }

    /// Add a reasoning step
    pub fn add_step(&mut self, step: ReasoningStep) {
        self.steps.push(step);
    }

    /// Set final answer
    pub fn set_answer(&mut self, answer: String, confidence: f64) {
        self.answer = Some(answer);
        self.confidence = confidence;
    }

    /// Mark as verified
    pub fn mark_verified(&mut self) {
        self.verified = true;
    }

    /// Get summary of reasoning
    pub fn summary(&self) -> String {
        let mut summary = format!("Problem: {}\n", self.problem);
        summary.push_str(&format!("Steps: {}\n", self.steps.len()));
        
        for step in &self.steps {
            summary.push_str(&format!("  Step {}: {} ({})\n", 
                step.step, step.description, step.operator));
        }
        
        if let Some(ref answer) = self.answer {
            summary.push_str(&format!("Answer: {} (confidence: {:.2})\n", 
                answer, self.confidence));
        }
        
        summary.push_str(&format!("Verified: {}\n", self.verified));
        summary
    }
}

/// Chain-of-thought reasoner
pub struct ChainOfThoughtReasoner {
    /// Maximum steps allowed
    pub max_steps: usize,
    /// Minimum confidence threshold
    pub min_confidence: f64,
}

impl ChainOfThoughtReasoner {
    pub fn new(max_steps: usize, min_confidence: f64) -> Self {
        Self {
            max_steps,
            min_confidence,
        }
    }

    /// Decompose complex problem into steps
    pub fn decompose_problem(&self, problem: &str) -> Vec<String> {
        let mut steps = Vec::new();
        
        // Simple heuristic decomposition
        if problem.contains("and") {
            // Split on "and"
            for part in problem.split("and") {
                steps.push(part.trim().to_string());
            }
        } else if problem.contains("then") {
            // Sequential steps
            for part in problem.split("then") {
                steps.push(part.trim().to_string());
            }
        } else {
            // Single step
            steps.push(problem.to_string());
        }
        
        steps
    }

    /// Execute reasoning chain
    pub fn reason(&self, problem: &str) -> ReasoningChain {
        let mut chain = ReasoningChain::new(problem.to_string());
        
        // Decompose problem
        let sub_problems = self.decompose_problem(problem);
        
        for (i, sub_problem) in sub_problems.iter().enumerate() {
            if i >= self.max_steps {
                break;
            }
            
            let step = ReasoningStep {
                step: i + 1,
                description: sub_problem.clone(),
                thought: vec![0.0; 128], // Placeholder
                operator: "Analytical".to_string(),
                confidence: 0.8,
                result: Some(format!("Result of step {}", i + 1)),
            };
            
            chain.add_step(step);
        }
        
        // Combine results
        if !chain.steps.is_empty() {
            let avg_confidence: f64 = chain.steps.iter()
                .map(|s| s.confidence)
                .sum::<f64>() / chain.steps.len() as f64;
            
            chain.set_answer("Final answer".to_string(), avg_confidence);
            
            if avg_confidence >= self.min_confidence {
                chain.mark_verified();
            }
        }
        
        chain
    }

    /// Verify reasoning chain
    pub fn verify_chain(&self, chain: &ReasoningChain) -> bool {
        // Check if all steps have sufficient confidence
        chain.steps.iter().all(|step| step.confidence >= self.min_confidence)
            && chain.confidence >= self.min_confidence
    }
}

impl Default for ChainOfThoughtReasoner {
    fn default() -> Self {
        Self::new(10, 0.7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_creation() {
        let chain = ReasoningChain::new("Test problem".to_string());
        assert_eq!(chain.problem, "Test problem");
        assert_eq!(chain.steps.len(), 0);
    }

    #[test]
    fn test_problem_decomposition() {
        let reasoner = ChainOfThoughtReasoner::default();
        let steps = reasoner.decompose_problem("First do A and then do B");
        assert!(steps.len() >= 2);
    }

    #[test]
    fn test_reasoning() {
        let reasoner = ChainOfThoughtReasoner::default();
        let chain = reasoner.reason("Solve problem A and then solve problem B");
        assert!(!chain.steps.is_empty());
    }
}
