//! Verification-Driven Learning Loop
//!
//! Implements human-like mastery through:
//! 1. Generate candidate solution
//! 2. Verify correctness (forward + backward + confidence + energy + coherence)
//! 3. Store only verified solutions
//! 4. Reconstruct solutions iteratively to internalize reasoning
//!
//! Mathematical Foundation:
//! - V(Ŝ, S_true) = 1 ⟺ all checks pass
//! - θ_{t+1} = θ_t + η ∇_θ V(Ŝ, S_true)
//! - M = M ∪ {(P, Ŝ) | V(Ŝ, S_true) = 1}
//! - θ* = arg max_θ ∑_{(P,S) ∈ M} V(f_θ(P), S)

use crate::core::{ThoughtState, Problem, EnergyResult, Evaluator};
use crate::memory::{Episode, EpisodicMemory};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// PART 1: VERIFICATION RESULT
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Overall verification status
    pub verified: bool,
    
    /// Forward consistency (does solution solve problem?)
    pub forward_check: bool,
    pub forward_score: f64,
    
    /// Backward consistency (does solution reconstruct problem?)
    pub backward_check: bool,
    pub backward_score: f64,
    
    /// Confidence check
    pub confidence_check: bool,
    pub confidence: f64,
    
    /// Energy check
    pub energy_check: bool,
    pub energy: f64,
    
    /// Coherence with memory
    pub coherence_check: bool,
    pub coherence: f64,
    
    /// Detailed reasoning
    pub reasoning: String,
}

impl VerificationResult {
    pub fn all_passed(&self) -> bool {
        self.forward_check && self.backward_check && self.confidence_check 
            && self.energy_check && self.coherence_check
    }
    
    pub fn count_passed(&self) -> usize {
        [self.forward_check, self.backward_check, self.confidence_check,
         self.energy_check, self.coherence_check]
            .iter()
            .filter(|&&x| x)
            .count()
    }
}

// ============================================================================
// PART 2: RECONSTRUCTION STATISTICS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReconstructionStats {
    /// Total reconstruction attempts
    pub total: usize,
    
    /// Successful reconstructions
    pub successful: usize,
    
    /// Failed reconstructions
    pub failed: usize,
    
    /// Average similarity to original
    pub avg_similarity: f64,
    
    /// Reconstruction success rate
    pub success_rate: f64,
    
    /// Per-domain statistics
    pub domain_stats: HashMap<String, DomainReconstructionStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DomainReconstructionStats {
    pub total: usize,
    pub successful: usize,
    pub avg_similarity: f64,
}

impl ReconstructionStats {
    pub fn update(&mut self, domain: &str, success: bool, similarity: f64) {
        self.total += 1;
        if success {
            self.successful += 1;
        } else {
            self.failed += 1;
        }
        
        // Update running average
        let n = self.total as f64;
        self.avg_similarity = (self.avg_similarity * (n - 1.0) + similarity) / n;
        self.success_rate = self.successful as f64 / self.total as f64;
        
        // Update domain stats
        let domain_stat = self.domain_stats.entry(domain.to_string()).or_default();
        domain_stat.total += 1;
        if success {
            domain_stat.successful += 1;
        }
        let dn = domain_stat.total as f64;
        domain_stat.avg_similarity = (domain_stat.avg_similarity * (dn - 1.0) + similarity) / dn;
    }
}

// ============================================================================
// PART 3: VERIFICATION THRESHOLDS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationThresholds {
    /// Forward consistency threshold
    pub forward_threshold: f64,
    
    /// Backward consistency threshold
    pub backward_threshold: f64,
    
    /// Minimum confidence
    pub min_confidence: f64,
    
    /// Maximum energy
    pub max_energy: f64,
    
    /// Minimum coherence
    pub min_coherence: f64,
}

impl Default for VerificationThresholds {
    fn default() -> Self {
        Self {
            forward_threshold: 0.8,
            backward_threshold: 0.7,
            min_confidence: 0.6,
            max_energy: 0.5,
            min_coherence: 0.5,
        }
    }
}

// ============================================================================
// PART 4: VERIFICATION LOOP
// ============================================================================

pub struct VerificationLoop {
    /// Verification thresholds
    pub thresholds: VerificationThresholds,
    
    /// Verified solutions memory
    pub verified_memory: Vec<VerifiedSolution>,
    
    /// Reconstruction statistics
    pub reconstruction_stats: ReconstructionStats,
    
    /// Evaluator for energy computation
    pub evaluator: Evaluator,
    
    /// Maximum memory size before compression
    pub max_memory_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedSolution {
    pub problem: String,
    pub solution: String,
    pub thought_vector: Vec<f64>,
    pub confidence: f64,
    pub energy: f64,
    pub verification_time: u64,
    pub reconstruction_count: u32,
    pub domain: String,
}

impl VerificationLoop {
    pub fn new(evaluator: Evaluator) -> Self {
        Self {
            thresholds: VerificationThresholds::default(),
            verified_memory: Vec::new(),
            reconstruction_stats: ReconstructionStats::default(),
            evaluator,
            max_memory_size: 10000,
        }
    }
    
    /// Verify a solution and store if valid
    pub fn verify_and_store(
        &mut self,
        problem: &Problem,
        thought: &ThoughtState,
        energy: &EnergyResult,
        solution: &str,
        domain: &str,
    ) -> VerificationResult {
        // 1. Forward check: Does solution solve problem?
        let forward_score = self.check_forward(problem, solution);
        let forward_check = forward_score >= self.thresholds.forward_threshold;
        
        // 2. Backward check: Can we reconstruct problem from solution?
        let backward_score = self.check_backward(problem, thought);
        let backward_check = backward_score >= self.thresholds.backward_threshold;
        
        // 3. Confidence check
        let confidence = energy.confidence_score;
        let confidence_check = confidence >= self.thresholds.min_confidence;
        
        // 4. Energy check
        let energy_val = energy.total;
        let energy_check = energy_val <= self.thresholds.max_energy;
        
        // 5. Coherence check: Does this align with existing knowledge?
        let coherence = self.check_coherence(thought);
        let coherence_check = coherence >= self.thresholds.min_coherence;
        
        let verified = forward_check && backward_check && confidence_check 
            && energy_check && coherence_check;
        
        // Store if verified
        if verified {
            self.verified_memory.push(VerifiedSolution {
                problem: problem.input.clone(),
                solution: solution.to_string(),
                thought_vector: thought.vector.clone(),
                confidence,
                energy: energy_val,
                verification_time: Self::current_timestamp(),
                reconstruction_count: 0,
                domain: domain.to_string(),
            });
        }
        
        let reasoning = format!(
            "Forward: {:.3} ({}), Backward: {:.3} ({}), Confidence: {:.3} ({}), Energy: {:.3} ({}), Coherence: {:.3} ({})",
            forward_score, if forward_check { "✓" } else { "✗" },
            backward_score, if backward_check { "✓" } else { "✗" },
            confidence, if confidence_check { "✓" } else { "✗" },
            energy_val, if energy_check { "✓" } else { "✗" },
            coherence, if coherence_check { "✓" } else { "✗" }
        );
        
        VerificationResult {
            verified,
            forward_check,
            forward_score,
            backward_check,
            backward_score,
            confidence_check,
            confidence,
            energy_check,
            energy: energy_val,
            coherence_check,
            coherence,
            reasoning,
        }
    }
    
    /// Reconstruct a single solution to reinforce reasoning
    pub fn reconstruct_single(
        &mut self,
        idx: usize,
        solver: &dyn Fn(&str) -> (String, ThoughtState, f64),
    ) -> ReconstructionResult {
        if idx >= self.verified_memory.len() {
            return ReconstructionResult {
                success: false,
                similarity: 0.0,
                matches_original: false,
                reasoning: "Index out of bounds".to_string(),
            };
        }
        
        // Clone the data we need to avoid borrow issues
        let problem = self.verified_memory[idx].problem.clone();
        let original_solution = self.verified_memory[idx].solution.clone();
        let domain = self.verified_memory[idx].domain.clone();
        
        // Re-solve from scratch
        let (new_solution, _new_thought, _new_confidence) = solver(&problem);
        
        // Compare solutions
        let similarity = self.solution_similarity(&new_solution, &original_solution);
        let matches = similarity >= 0.9;
        
        // Update reconstruction count
        self.verified_memory[idx].reconstruction_count += 1;
        
        // Update statistics
        self.reconstruction_stats.update(&domain, matches, similarity);
        
        ReconstructionResult {
            success: matches,
            similarity,
            matches_original: matches,
            reasoning: format!(
                "Reconstructed solution similarity: {:.3}, Original: '{}', New: '{}'",
                similarity, original_solution, new_solution
            ),
        }
    }
    
    /// Reconstruct all verified solutions
    pub fn reconstruct_all(
        &mut self,
        solver: &dyn Fn(&str) -> (String, ThoughtState, f64),
    ) -> ReconstructionStats {
        let total = self.verified_memory.len();
        
        for i in 0..total {
            self.reconstruct_single(i, solver);
        }
        
        self.reconstruction_stats.clone()
    }
    
    /// Check forward consistency
    fn check_forward(&self, problem: &Problem, solution: &str) -> f64 {
        // Simple text similarity for now
        // TODO: Use semantic similarity or symbolic verification
        if let Some(expected) = &problem.target_answer {
            self.text_similarity(solution, expected)
        } else {
            0.5 // Unknown, assume moderate
        }
    }
    
    /// Check backward consistency
    fn check_backward(&self, problem: &Problem, thought: &ThoughtState) -> f64 {
        // Check if thought vector can reconstruct problem
        // Use cosine similarity between problem embedding and thought
        let problem_vec = &problem.state.vector;
        let thought_vec = &thought.vector;
        
        self.cosine_similarity(problem_vec, thought_vec)
    }
    
    /// Check coherence with existing memory
    fn check_coherence(&self, thought: &ThoughtState) -> f64 {
        if self.verified_memory.is_empty() {
            return 1.0; // No memory yet, assume coherent
        }
        
        // Average similarity to verified solutions
        let similarities: Vec<f64> = self.verified_memory
            .iter()
            .map(|v| self.cosine_similarity(&thought.vector, &v.thought_vector))
            .collect();
        
        similarities.iter().sum::<f64>() / similarities.len() as f64
    }
    
    /// Text similarity (simple)
    fn text_similarity(&self, a: &str, b: &str) -> f64 {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();
        
        if a_lower == b_lower {
            1.0
        } else if a_lower.contains(&b_lower) || b_lower.contains(&a_lower) {
            0.8
        } else {
            // Levenshtein-like simple metric
            let common_chars = a_lower.chars()
                .filter(|c| b_lower.contains(*c))
                .count();
            let max_len = a_lower.len().max(b_lower.len());
            if max_len == 0 {
                0.0
            } else {
                common_chars as f64 / max_len as f64
            }
        }
    }
    
    /// Solution similarity
    fn solution_similarity(&self, a: &str, b: &str) -> f64 {
        self.text_similarity(a, b)
    }
    
    /// Cosine similarity
    fn cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm_a < 1e-10 || norm_b < 1e-10 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
    
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> VerificationLoopStats {
        VerificationLoopStats {
            total_verified: self.verified_memory.len(),
            reconstruction_stats: self.reconstruction_stats.clone(),
            avg_confidence: self.verified_memory.iter()
                .map(|v| v.confidence)
                .sum::<f64>() / self.verified_memory.len().max(1) as f64,
            avg_energy: self.verified_memory.iter()
                .map(|v| v.energy)
                .sum::<f64>() / self.verified_memory.len().max(1) as f64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionResult {
    pub success: bool,
    pub similarity: f64,
    pub matches_original: bool,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationLoopStats {
    pub total_verified: usize,
    pub reconstruction_stats: ReconstructionStats,
    pub avg_confidence: f64,
    pub avg_energy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{EnergyWeights, Problem};
    
    #[test]
    fn test_verification_loop() {
        let evaluator = Evaluator::new(EnergyWeights::default(), 0.6);
        let mut loop_sys = VerificationLoop::new(evaluator);
        
        let problem = Problem::new("What is 2+2?", 128);
        let thought = ThoughtState::random(128);
        let energy = EnergyResult {
            total: 0.3,
            constraint_energy: 0.1,
            risk_energy: 0.1,
            uncertainty_energy: 0.1,
            verified: true,
            confidence_score: 0.8,
        };
        
        let result = loop_sys.verify_and_store(&problem, &thought, &energy, "4", "math");
        
        assert!(result.confidence_check);
        assert!(result.energy_check);
    }
}
