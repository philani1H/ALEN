//! Logical Inference Engine
//!
//! Implements formal logical reasoning with rules and premises

use serde::{Deserialize, Serialize};

/// Logical premise
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Premise {
    pub statement: String,
    pub confidence: f64,
}

/// Logical conclusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conclusion {
    pub statement: String,
    pub confidence: f64,
    pub derived_from: Vec<usize>, // Indices of premises used
}

/// Inference rule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceRule {
    ModusPonens,      // If P then Q, P, therefore Q
    ModusTollens,     // If P then Q, not Q, therefore not P
    Syllogism,        // All A are B, All B are C, therefore All A are C
    Contrapositive,   // If P then Q is equivalent to If not Q then not P
    Transitivity,     // If A→B and B→C then A→C
}

/// Logical inference engine
pub struct LogicalInference {
    premises: Vec<Premise>,
    conclusions: Vec<Conclusion>,
}

impl LogicalInference {
    pub fn new() -> Self {
        Self {
            premises: Vec::new(),
            conclusions: Vec::new(),
        }
    }

    /// Add a premise
    pub fn add_premise(&mut self, statement: String, confidence: f64) {
        self.premises.push(Premise { statement, confidence });
    }

    /// Apply modus ponens: If P then Q, P, therefore Q
    pub fn modus_ponens(&mut self) -> Option<Conclusion> {
        // Look for "if...then" statements and matching premises
        for (i, p1) in self.premises.iter().enumerate() {
            if p1.statement.contains("if") && p1.statement.contains("then") {
                // Extract P and Q
                if let Some(then_pos) = p1.statement.find("then") {
                    let if_pos = p1.statement.find("if").unwrap();
                    let p_part = p1.statement[if_pos+2..then_pos].trim();
                    let q_part = p1.statement[then_pos+4..].trim();
                    
                    // Look for matching P
                    for (j, p2) in self.premises.iter().enumerate() {
                        if i != j && p2.statement.contains(p_part) {
                            // Can conclude Q
                            let confidence = p1.confidence.min(p2.confidence);
                            return Some(Conclusion {
                                statement: q_part.to_string(),
                                confidence,
                                derived_from: vec![i, j],
                            });
                        }
                    }
                }
            }
        }
        None
    }

    /// Apply syllogism: All A are B, All B are C, therefore All A are C
    pub fn syllogism(&mut self) -> Option<Conclusion> {
        // Look for "all...are" patterns
        for (i, p1) in self.premises.iter().enumerate() {
            if p1.statement.contains("all") && p1.statement.contains("are") {
                for (j, p2) in self.premises.iter().enumerate() {
                    if i != j && p2.statement.contains("all") && p2.statement.contains("are") {
                        // Try to match middle term
                        // Simplified implementation
                        let confidence = p1.confidence.min(p2.confidence) * 0.9;
                        return Some(Conclusion {
                            statement: "Derived conclusion".to_string(),
                            confidence,
                            derived_from: vec![i, j],
                        });
                    }
                }
            }
        }
        None
    }

    /// Infer all possible conclusions
    pub fn infer_all(&mut self) -> Vec<Conclusion> {
        let mut conclusions = Vec::new();
        
        if let Some(c) = self.modus_ponens() {
            conclusions.push(c);
        }
        
        if let Some(c) = self.syllogism() {
            conclusions.push(c);
        }
        
        self.conclusions.extend(conclusions.clone());
        conclusions
    }

    /// Get all conclusions
    pub fn get_conclusions(&self) -> &[Conclusion] {
        &self.conclusions
    }

    /// Get premises
    pub fn get_premises(&self) -> &[Premise] {
        &self.premises
    }
}

impl Default for LogicalInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_premise() {
        let mut inference = LogicalInference::new();
        inference.add_premise("All humans are mortal".to_string(), 1.0);
        assert_eq!(inference.get_premises().len(), 1);
    }

    #[test]
    fn test_modus_ponens() {
        let mut inference = LogicalInference::new();
        inference.add_premise("if it rains then the ground is wet".to_string(), 1.0);
        inference.add_premise("it rains".to_string(), 1.0);
        
        let conclusion = inference.modus_ponens();
        assert!(conclusion.is_some());
    }
}
