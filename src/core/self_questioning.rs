//! Self-Questioning Operator Module
//!
//! Implements the mathematical framework for explicit self-questioning:
//!
//! Q : ψ → {ψ₁, ψ₂, ..., ψₙ}
//!
//! Where:
//! - ψ = current belief/solution state
//! - ψᵢ = alternative interpretations, derivations, or assumptions
//!
//! This operator deliberately INCREASES uncertainty before reducing it,
//! which is fundamentally different from transformers.
//!
//! Energy-based pruning:
//! E(ψᵢ) = α·inconsistency + β·unprovability + γ·complexity
//! ψ* = argmin E(ψᵢ)

use crate::core::{ThoughtState, Problem, EnergyResult, Evaluator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// PART 1: EPISTEMIC UNCERTAINTY (not just aleatoric)
// ============================================================================

/// Epistemic uncertainty - "Do I actually know enough to justify this?"
/// U_e = P(proof failure | K)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicUncertainty {
    /// Probability of proof failure given knowledge state
    pub proof_failure_prob: f64,
    /// Knowledge completeness (0 = no knowledge, 1 = complete)
    pub knowledge_completeness: f64,
    /// Number of verification paths available
    pub verification_paths: usize,
    /// Consensus among paths (0 = total disagreement, 1 = perfect agreement)
    pub path_consensus: f64,
}

impl EpistemicUncertainty {
    /// Calculate epistemic uncertainty for a thought state
    pub fn calculate(
        thought: &ThoughtState,
        problem: &Problem,
        verification_results: &[SelfQuestioningVerificationResult],
    ) -> Self {
        let verification_paths = verification_results.len();
        
        // Calculate path consensus
        let path_consensus = if verification_paths > 1 {
            Self::calculate_consensus(verification_results)
        } else {
            0.0 // No consensus possible with single path
        };
        
        // Knowledge completeness based on problem coverage
        let knowledge_completeness = Self::estimate_knowledge_completeness(thought, problem);
        
        // Proof failure probability
        let proof_failure_prob = 1.0 - (path_consensus * knowledge_completeness);
        
        Self {
            proof_failure_prob,
            knowledge_completeness,
            verification_paths,
            path_consensus,
        }
    }
    
    fn calculate_consensus(results: &[SelfQuestioningVerificationResult]) -> f64 {
        if results.len() < 2 {
            return 0.0;
        }
        
        let mut agreement_count = 0;
        let mut total_comparisons = 0;
        
        for i in 0..results.len() {
            for j in (i+1)..results.len() {
                total_comparisons += 1;
                if results[i].agrees_with(&results[j]) {
                    agreement_count += 1;
                }
            }
        }
        
        if total_comparisons > 0 {
            agreement_count as f64 / total_comparisons as f64
        } else {
            0.0
        }
    }
    
    fn estimate_knowledge_completeness(thought: &ThoughtState, problem: &Problem) -> f64 {
        // Estimate based on vector similarity and constraint satisfaction
        let similarity = thought.cosine_similarity(&problem.state);
        let constraint_satisfaction = if problem.constraints.is_empty() {
            1.0
        } else {
            // Simplified: assume some constraints are satisfied
            0.7
        };
        
        (similarity + constraint_satisfaction) / 2.0
    }
}

// ============================================================================
// PART 2: CONFIDENCE AS FUNCTIONAL (not probability)
// ============================================================================

/// Confidence functional: C(y) = E_π∈Π[𝟙(π(y) = valid)]
/// An answer is confident only if it survives many checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceFunctional {
    /// Set of verification procedures applied
    pub verification_procedures: Vec<VerificationProcedure>,
    /// Consensus confidence (0-1)
    pub consensus_confidence: f64,
    /// Individual procedure results
    pub procedure_results: HashMap<String, bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationProcedure {
    pub id: String,
    pub name: String,
    pub procedure_type: VerificationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationType {
    ForwardBackward,    // g(f(x)) ≈ x
    MultiPath,          // Multiple reasoning paths agree
    ConsistencyCheck,   // No internal contradictions
    ConstraintCheck,    // Satisfies all constraints
    EnergyMinimization, // Minimal energy state
}

impl ConfidenceFunctional {
    /// Calculate confidence as consensus under proof
    pub fn calculate(verification_results: &[SelfQuestioningVerificationResult]) -> Self {
        let mut procedure_results = HashMap::new();
        let mut valid_count = 0;
        
        for result in verification_results {
            procedure_results.insert(result.procedure_id.clone(), result.is_valid);
            if result.is_valid {
                valid_count += 1;
            }
        }
        
        let consensus_confidence = if !verification_results.is_empty() {
            valid_count as f64 / verification_results.len() as f64
        } else {
            0.0
        };
        
        Self {
            verification_procedures: verification_results.iter()
                .map(|r| r.procedure.clone())
                .collect(),
            consensus_confidence,
            procedure_results,
        }
    }
}

// ============================================================================
// PART 3: SELF-QUESTIONING OPERATOR
// ============================================================================

/// Self-questioning operator: Q : ψ → {ψ₁, ψ₂, ..., ψₙ}
/// Deliberately increases uncertainty before reducing it
#[derive(Debug, Clone)]
pub struct SelfQuestioningOperator {
    /// Maximum alternative interpretations to generate
    pub max_alternatives: usize,
    /// Energy weights for pruning
    pub energy_weights: QuestioningEnergyWeights,
    /// Minimum epistemic uncertainty to trigger questioning
    pub uncertainty_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestioningEnergyWeights {
    /// Weight for inconsistency
    pub alpha: f64,
    /// Weight for unprovability
    pub beta: f64,
    /// Weight for complexity
    pub gamma: f64,
}

impl Default for QuestioningEnergyWeights {
    fn default() -> Self {
        Self {
            alpha: 0.4,
            beta: 0.4,
            gamma: 0.2,
        }
    }
}

/// Alternative interpretation generated by self-questioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeInterpretation {
    /// The alternative thought state
    pub thought: ThoughtState,
    /// Type of alternative
    pub alternative_type: AlternativeType,
    /// Energy of this alternative
    pub energy: f64,
    /// Inconsistency score
    pub inconsistency: f64,
    /// Unprovability score
    pub unprovability: f64,
    /// Complexity score
    pub complexity: f64,
    /// Why this alternative was generated
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlternativeType {
    /// Different assumption about the problem
    DifferentAssumption,
    /// Alternative reasoning path
    AlternativePath,
    /// Opposite conclusion
    Negation,
    /// Weaker claim
    Weakening,
    /// Stronger claim
    Strengthening,
    /// Different interpretation of terms
    Reinterpretation,
}

impl SelfQuestioningOperator {
    pub fn new() -> Self {
        Self {
            max_alternatives: 5,
            energy_weights: QuestioningEnergyWeights::default(),
            uncertainty_threshold: 0.3,
        }
    }
    
    /// Apply self-questioning operator
    /// Returns alternative interpretations, deliberately increasing uncertainty
    pub fn question(
        &self,
        current_thought: &ThoughtState,
        problem: &Problem,
        evaluator: &Evaluator,
    ) -> Vec<AlternativeInterpretation> {
        let mut alternatives = Vec::new();
        
        // Generate different types of alternatives
        alternatives.extend(self.generate_assumption_alternatives(current_thought, problem));
        alternatives.extend(self.generate_path_alternatives(current_thought, problem));
        alternatives.extend(self.generate_negation(current_thought, problem));
        alternatives.extend(self.generate_weakenings(current_thought, problem));
        
        // Calculate energy for each alternative
        let mut scored_alternatives: Vec<_> = alternatives.into_iter()
            .map(|mut alt| {
                alt.energy = self.calculate_energy(&alt);
                alt
            })
            .collect();
        
        // Sort by energy (lower is better)
        scored_alternatives.sort_by(|a, b| {
            a.energy.partial_cmp(&b.energy).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Return top alternatives
        scored_alternatives.into_iter()
            .take(self.max_alternatives)
            .collect()
    }
    
    /// Calculate energy: E(ψᵢ) = α·inconsistency + β·unprovability + γ·complexity
    fn calculate_energy(&self, alternative: &AlternativeInterpretation) -> f64 {
        self.energy_weights.alpha * alternative.inconsistency
            + self.energy_weights.beta * alternative.unprovability
            + self.energy_weights.gamma * alternative.complexity
    }
    
    fn generate_assumption_alternatives(
        &self,
        thought: &ThoughtState,
        problem: &Problem,
    ) -> Vec<AlternativeInterpretation> {
        let mut alternatives = Vec::new();
        
        // Generate alternative by questioning implicit assumptions
        let mut alt_vector = thought.vector.clone();
        // Perturb the vector to represent different assumption
        for i in 0..alt_vector.len() {
            alt_vector[i] *= 0.9; // Weaken assumptions
        }
        
        let alt_thought = ThoughtState {
            dimension: thought.dimension,
            confidence: thought.confidence * 0.9,
            vector: alt_vector,
            metadata: thought.metadata.clone(),
        };
        
        alternatives.push(AlternativeInterpretation {
            thought: alt_thought,
            alternative_type: AlternativeType::DifferentAssumption,
            energy: 0.0, // Will be calculated later
            inconsistency: 0.3,
            unprovability: 0.4,
            complexity: 0.2,
            rationale: "What if the implicit assumptions are wrong?".to_string(),
        });
        
        alternatives
    }
    
    fn generate_path_alternatives(
        &self,
        thought: &ThoughtState,
        problem: &Problem,
    ) -> Vec<AlternativeInterpretation> {
        let mut alternatives = Vec::new();
        
        // Generate alternative reasoning path
        let mut alt_vector = thought.vector.clone();
        // Rotate the vector to represent different path
        if alt_vector.len() > 1 {
            alt_vector.rotate_left(1);
        }
        
        let alt_thought = ThoughtState {
            dimension: thought.dimension,
            confidence: thought.confidence * 0.9,
            vector: alt_vector,
            metadata: thought.metadata.clone(),
        };
        
        alternatives.push(AlternativeInterpretation {
            thought: alt_thought,
            alternative_type: AlternativeType::AlternativePath,
            energy: 0.0,
            inconsistency: 0.2,
            unprovability: 0.3,
            complexity: 0.3,
            rationale: "Could we reach the same conclusion differently?".to_string(),
        });
        
        alternatives
    }
    
    fn generate_negation(
        &self,
        thought: &ThoughtState,
        problem: &Problem,
    ) -> Vec<AlternativeInterpretation> {
        let mut alternatives = Vec::new();
        
        // Generate negation
        let alt_vector: Vec<f64> = thought.vector.iter().map(|x| -x).collect();
        
        let alt_thought = ThoughtState {
            dimension: thought.dimension,
            confidence: thought.confidence * 0.9,
            vector: alt_vector,
            metadata: thought.metadata.clone(),
        };
        
        alternatives.push(AlternativeInterpretation {
            thought: alt_thought,
            alternative_type: AlternativeType::Negation,
            energy: 0.0,
            inconsistency: 0.5,
            unprovability: 0.5,
            complexity: 0.1,
            rationale: "What if the opposite is true?".to_string(),
        });
        
        alternatives
    }
    
    fn generate_weakenings(
        &self,
        thought: &ThoughtState,
        problem: &Problem,
    ) -> Vec<AlternativeInterpretation> {
        let mut alternatives = Vec::new();
        
        // Generate weaker claim
        let alt_vector: Vec<f64> = thought.vector.iter().map(|x| x * 0.7).collect();
        
        let alt_thought = ThoughtState {
            dimension: thought.dimension,
            confidence: thought.confidence * 0.9,
            vector: alt_vector,
            metadata: thought.metadata.clone(),
        };
        
        alternatives.push(AlternativeInterpretation {
            thought: alt_thought,
            alternative_type: AlternativeType::Weakening,
            energy: 0.0,
            inconsistency: 0.1,
            unprovability: 0.2,
            complexity: 0.2,
            rationale: "What if we claim less?".to_string(),
        });
        
        alternatives
    }
}

// ============================================================================
// PART 4: VERIFICATION RESULTS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfQuestioningVerificationResult {
    pub procedure_id: String,
    pub procedure: VerificationProcedure,
    pub is_valid: bool,
    pub confidence: f64,
    pub details: String,
}

impl SelfQuestioningVerificationResult {
    /// Check if two verification results agree
    pub fn agrees_with(&self, other: &SelfQuestioningVerificationResult) -> bool {
        self.is_valid == other.is_valid && 
        (self.confidence - other.confidence).abs() < 0.2
    }
}

// ============================================================================
// PART 5: MULTI-PATH PROOF AGREEMENT
// ============================================================================

/// Multi-path proof agreement: A = (2/k(k-1)) Σᵢ<ⱼ 𝟙(sᵢ ≡ sⱼ)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPathAgreement {
    /// Solutions from different reasoning paths
    pub solutions: Vec<PathSolution>,
    /// Agreement metric (0-1)
    pub agreement: f64,
    /// Acceptance threshold
    pub threshold: f64,
    /// Whether answer is accepted
    pub accepted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathSolution {
    pub path_type: ReasoningPathType,
    pub solution: ThoughtState,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningPathType {
    Algebraic,
    Geometric,
    Numerical,
    Symbolic,
    Heuristic,
}

impl MultiPathAgreement {
    /// Calculate agreement across multiple paths
    pub fn calculate(solutions: Vec<PathSolution>, threshold: f64) -> Self {
        let k = solutions.len();
        
        let agreement = if k < 2 {
            0.0
        } else {
            let mut agreement_count = 0;
            let mut total_comparisons = 0;
            
            for i in 0..k {
                for j in (i+1)..k {
                    total_comparisons += 1;
                    if Self::solutions_equivalent(&solutions[i], &solutions[j]) {
                        agreement_count += 1;
                    }
                }
            }
            
            if total_comparisons > 0 {
                (2.0 * agreement_count as f64) / (k * (k - 1)) as f64
            } else {
                0.0
            }
        };
        
        let accepted = agreement >= threshold;
        
        Self {
            solutions,
            agreement,
            threshold,
            accepted,
        }
    }
    
    fn solutions_equivalent(s1: &PathSolution, s2: &PathSolution) -> bool {
        let similarity = s1.solution.cosine_similarity(&s2.solution);
        similarity > 0.9 // High similarity threshold
    }
}

// ============================================================================
// PART 6: SELF-CORRECTION TRIGGER
// ============================================================================

/// Self-correction is triggered by: ∃i,j : sᵢ ≠ sⱼ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfCorrectionTrigger {
    /// Whether self-correction is needed
    pub correction_needed: bool,
    /// Disagreeing solution indices
    pub disagreements: Vec<(usize, usize)>,
    /// Reason for disagreement
    pub reasons: Vec<String>,
    /// Suggested clarifying questions
    pub clarifying_questions: Vec<String>,
}

impl SelfCorrectionTrigger {
    /// Check if self-correction is needed
    pub fn check(agreement: &MultiPathAgreement) -> Self {
        let mut disagreements = Vec::new();
        let mut reasons = Vec::new();
        
        for i in 0..agreement.solutions.len() {
            for j in (i+1)..agreement.solutions.len() {
                if !MultiPathAgreement::solutions_equivalent(
                    &agreement.solutions[i],
                    &agreement.solutions[j]
                ) {
                    disagreements.push((i, j));
                    reasons.push(format!(
                        "{:?} path disagrees with {:?} path",
                        agreement.solutions[i].path_type,
                        agreement.solutions[j].path_type
                    ));
                }
            }
        }
        
        let correction_needed = !disagreements.is_empty();
        
        let clarifying_questions = if correction_needed {
            vec![
                "Which reasoning path is more reliable for this problem?".to_string(),
                "What assumptions differ between the paths?".to_string(),
                "Can we identify the source of disagreement?".to_string(),
            ]
        } else {
            vec![]
        };
        
        Self {
            correction_needed,
            disagreements,
            reasons,
            clarifying_questions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_epistemic_uncertainty() {
        let thought = ThoughtState::random(128);
        let problem = Problem::new("test", 128);
        let results = vec![];
        
        let uncertainty = EpistemicUncertainty::calculate(&thought, &problem, &results);
        assert!(uncertainty.proof_failure_prob >= 0.0 && uncertainty.proof_failure_prob <= 1.0);
    }
    
    #[test]
    fn test_self_questioning() {
        let operator = SelfQuestioningOperator::new();
        let thought = ThoughtState::random(128);
        let problem = Problem::new("test", 128);
        let evaluator = Evaluator::default();
        
        let alternatives = operator.question(&thought, &problem, &evaluator);
        assert!(!alternatives.is_empty());
        assert!(alternatives.len() <= operator.max_alternatives);
    }
    
    #[test]
    fn test_multi_path_agreement() {
        let solutions = vec![
            PathSolution {
                path_type: ReasoningPathType::Algebraic,
                solution: ThoughtState::random(128),
                confidence: 0.8,
            },
            PathSolution {
                path_type: ReasoningPathType::Numerical,
                solution: ThoughtState::random(128),
                confidence: 0.7,
            },
        ];
        
        let agreement = MultiPathAgreement::calculate(solutions, 0.7);
        assert!(agreement.agreement >= 0.0 && agreement.agreement <= 1.0);
    }
}
