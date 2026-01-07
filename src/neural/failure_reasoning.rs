//! Failure Reasoning Module
//!
//! Implements human-style "Why did I fail → Correct → Learn" loop
//!
//! Mathematical Framework:
//! 1. Failure Detection: Error(Y) = ℓ(Y, Y*) > τ_err
//! 2. Failure Attribution: z = g_φ(x, Y, u, M_t)
//! 3. Cause Classification: Cause = argmax_k P(k | z)
//! 4. Strategy Adjustment: Controller_t = Controller_{t-1} + ΔController(k)
//! 5. Memory Update: M_{t+1} = Compress(M_t ⊕ {x, Y, z, k})
//! 6. Retry: Y' = f_θ(x, u, M_{t+1}, Controller_t)
//! 7. Explanation: E = h_ψ(z, k)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// FAILURE DETECTION
// ============================================================================

/// Failure detector
#[derive(Debug, Clone)]
pub struct FailureDetector {
    /// Error threshold τ_err
    pub error_threshold: f64,
    /// Confidence threshold τ_conf
    pub confidence_threshold: f64,
}

impl FailureDetector {
    pub fn new(error_threshold: f64, confidence_threshold: f64) -> Self {
        Self {
            error_threshold,
            confidence_threshold,
        }
    }
    
    /// Detect failure: Error(Y) = ℓ(Y, Y*) > τ_err
    pub fn detect_failure(
        &self,
        output: &str,
        expected: Option<&str>,
        confidence: f64,
    ) -> FailureDetection {
        let mut is_failure = false;
        let mut error_score = 0.0;
        let mut failure_type = FailureType::None;
        
        // Check confidence first
        if confidence < self.confidence_threshold {
            is_failure = true;
            error_score = 1.0 - confidence;
            failure_type = FailureType::LowConfidence;
        }
        
        // Check against expected output if available
        if let Some(expected_output) = expected {
            let loss = self.compute_loss(output, expected_output);
            if loss > self.error_threshold {
                is_failure = true;
                error_score = error_score.max(loss);
                failure_type = FailureType::IncorrectOutput;
            }
        }
        
        // Check for hallucination markers
        if self.detect_hallucination(output) {
            is_failure = true;
            error_score = error_score.max(0.8);
            failure_type = FailureType::Hallucination;
        }
        
        FailureDetection {
            is_failure,
            error_score,
            failure_type,
            confidence,
        }
    }
    
    /// Compute loss ℓ(Y, Y*)
    fn compute_loss(&self, output: &str, expected: &str) -> f64 {
        // Simple word-level similarity
        let output_words: Vec<&str> = output.split_whitespace().collect();
        let expected_words: Vec<&str> = expected.split_whitespace().collect();
        
        let mut matches = 0;
        for word in &output_words {
            if expected_words.contains(word) {
                matches += 1;
            }
        }
        
        let max_len = output_words.len().max(expected_words.len());
        if max_len == 0 {
            return 0.0;
        }
        
        1.0 - (matches as f64 / max_len as f64)
    }
    
    /// Detect hallucination patterns
    fn detect_hallucination(&self, output: &str) -> bool {
        // Check for common hallucination markers
        let markers = [
            "I'm not sure but",
            "I think maybe",
            "possibly",
            "I don't actually know",
        ];
        
        markers.iter().any(|marker| output.contains(marker))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureDetection {
    pub is_failure: bool,
    pub error_score: f64,
    pub failure_type: FailureType,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FailureType {
    None,
    LowConfidence,
    IncorrectOutput,
    Hallucination,
    KnowledgeGap,
    ReasoningError,
    RetrievalMismatch,
    StyleMismatch,
}

// ============================================================================
// FAILURE ATTRIBUTION
// ============================================================================

/// Failure attribution encoder: z = g_φ(x, Y, u, M_t)
#[derive(Debug, Clone)]
pub struct FailureAttributor {
    pub dim: usize,
}

impl FailureAttributor {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    
    /// Encode failure context into latent vector z
    pub fn encode_failure(
        &self,
        input: &str,
        output: &str,
        user_embedding: &[f64],
        memory_context: &[f64],
    ) -> Vec<f64> {
        let mut z = vec![0.0; self.dim];
        
        // Encode input
        let input_bytes = input.as_bytes();
        for (i, &byte) in input_bytes.iter().take(self.dim / 4).enumerate() {
            z[i] = byte as f64 / 255.0;
        }
        
        // Encode output
        let output_bytes = output.as_bytes();
        for (i, &byte) in output_bytes.iter().take(self.dim / 4).enumerate() {
            z[self.dim / 4 + i] = byte as f64 / 255.0;
        }
        
        // Add user embedding
        for (i, &val) in user_embedding.iter().take(self.dim / 4).enumerate() {
            z[self.dim / 2 + i] = val;
        }
        
        // Add memory context
        for (i, &val) in memory_context.iter().take(self.dim / 4).enumerate() {
            z[3 * self.dim / 4 + i] = val;
        }
        
        z
    }
    
    /// Classify failure cause: Cause = argmax_k P(k | z)
    pub fn classify_cause(&self, z: &[f64], failure_type: &FailureType) -> FailureCause {
        // Simple heuristic classification based on latent vector
        // In production, use a trained classifier
        
        match failure_type {
            FailureType::LowConfidence => {
                // Analyze why confidence is low
                let avg_activation = z.iter().sum::<f64>() / z.len() as f64;
                
                if avg_activation < 0.3 {
                    FailureCause::KnowledgeGap
                } else if avg_activation > 0.7 {
                    FailureCause::ReasoningError
                } else {
                    FailureCause::RetrievalMismatch
                }
            }
            FailureType::Hallucination => FailureCause::Hallucination,
            FailureType::IncorrectOutput => FailureCause::ReasoningError,
            FailureType::StyleMismatch => FailureCause::StyleMismatch,
            _ => FailureCause::Unknown,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FailureCause {
    KnowledgeGap,
    ReasoningError,
    RetrievalMismatch,
    Hallucination,
    StyleMismatch,
    Unknown,
}

impl FailureCause {
    pub fn description(&self) -> &str {
        match self {
            FailureCause::KnowledgeGap => "Missing required knowledge or facts",
            FailureCause::ReasoningError => "Logical error in reasoning steps",
            FailureCause::RetrievalMismatch => "Retrieved wrong information from memory",
            FailureCause::Hallucination => "Generated unsupported or false information",
            FailureCause::StyleMismatch => "Output style doesn't match requirements",
            FailureCause::Unknown => "Cause could not be determined",
        }
    }
}

// ============================================================================
// STRATEGY ADJUSTMENT
// ============================================================================

/// Controller adjustment: Controller_t = Controller_{t-1} + ΔController(k)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyController {
    /// Verbosity adjustment
    pub verbosity_delta: f64,
    /// Confidence threshold adjustment
    pub confidence_delta: f64,
    /// Reasoning depth adjustment
    pub reasoning_depth_delta: f64,
    /// Retrieval count adjustment
    pub retrieval_count_delta: usize,
    /// Verification strictness adjustment
    pub verification_strictness_delta: f64,
}

impl Default for StrategyController {
    fn default() -> Self {
        Self {
            verbosity_delta: 0.0,
            confidence_delta: 0.0,
            reasoning_depth_delta: 0.0,
            retrieval_count_delta: 0,
            verification_strictness_delta: 0.0,
        }
    }
}

impl StrategyController {
    /// Compute adjustment based on failure cause
    pub fn compute_adjustment(cause: &FailureCause) -> Self {
        match cause {
            FailureCause::KnowledgeGap => Self {
                retrieval_count_delta: 2,  // Retrieve more context
                verification_strictness_delta: 0.1,  // Be more careful
                ..Default::default()
            },
            FailureCause::ReasoningError => Self {
                reasoning_depth_delta: 2.0,  // Add more reasoning steps
                verification_strictness_delta: 0.2,  // Much more careful
                confidence_delta: 0.1,  // Require higher confidence
                ..Default::default()
            },
            FailureCause::RetrievalMismatch => Self {
                retrieval_count_delta: 3,  // Try different retrievals
                reasoning_depth_delta: 1.0,  // More careful reasoning
                ..Default::default()
            },
            FailureCause::Hallucination => Self {
                verification_strictness_delta: 0.3,  // Much stricter
                confidence_delta: 0.2,  // Much higher confidence required
                verbosity_delta: -0.2,  // Be more concise
                ..Default::default()
            },
            FailureCause::StyleMismatch => Self {
                verbosity_delta: 0.1,  // Adjust verbosity
                ..Default::default()
            },
            FailureCause::Unknown => Self::default(),
        }
    }
    
    /// Apply adjustment to current parameters
    pub fn apply(&self, params: &mut SystemParameters) {
        params.verbosity = (params.verbosity + self.verbosity_delta).max(0.0).min(1.0);
        params.confidence_threshold = (params.confidence_threshold + self.confidence_delta).max(0.0).min(1.0);
        params.reasoning_depth = (params.reasoning_depth as f64 + self.reasoning_depth_delta).max(1.0) as usize;
        params.retrieval_count = params.retrieval_count.saturating_add(self.retrieval_count_delta);
        params.verification_strictness = (params.verification_strictness + self.verification_strictness_delta).max(0.0).min(1.0);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemParameters {
    pub verbosity: f64,
    pub confidence_threshold: f64,
    pub reasoning_depth: usize,
    pub retrieval_count: usize,
    pub verification_strictness: f64,
}

impl Default for SystemParameters {
    fn default() -> Self {
        Self {
            verbosity: 0.5,
            confidence_threshold: 0.6,
            reasoning_depth: 5,
            retrieval_count: 3,
            verification_strictness: 0.7,
        }
    }
}

// ============================================================================
// FAILURE MEMORY
// ============================================================================

/// Failure memory: M_{t+1} = Compress(M_t ⊕ {x, Y, z, k})
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureMemory {
    pub entries: Vec<FailureEntry>,
    pub max_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureEntry {
    pub input: String,
    pub output: String,
    pub latent_failure: Vec<f64>,
    pub cause: FailureCause,
    pub timestamp: u64,
    pub resolved: bool,
}

impl FailureMemory {
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_size,
        }
    }
    
    /// Add failure to memory
    pub fn add_failure(
        &mut self,
        input: String,
        output: String,
        latent_failure: Vec<f64>,
        cause: FailureCause,
    ) {
        let entry = FailureEntry {
            input,
            output,
            latent_failure,
            cause,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            resolved: false,
        };
        
        self.entries.push(entry);
        
        // Compress if needed
        if self.entries.len() > self.max_size {
            self.compress();
        }
    }
    
    /// Mark failure as resolved
    pub fn mark_resolved(&mut self, index: usize) {
        if index < self.entries.len() {
            self.entries[index].resolved = true;
        }
    }
    
    /// Compress memory by removing old resolved failures
    fn compress(&mut self) {
        // Keep unresolved and recent failures
        let cutoff_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() - 3600;  // Keep last hour
        
        self.entries.retain(|entry| {
            !entry.resolved || entry.timestamp > cutoff_time
        });
        
        // If still too large, keep only most recent
        if self.entries.len() > self.max_size {
            self.entries.drain(0..self.entries.len() - self.max_size);
        }
    }
    
    /// Get similar past failures
    pub fn get_similar_failures(&self, latent: &[f64], k: usize) -> Vec<&FailureEntry> {
        let mut scored: Vec<(f64, &FailureEntry)> = self.entries
            .iter()
            .map(|entry| {
                let similarity = self.cosine_similarity(latent, &entry.latent_failure);
                (similarity, entry)
            })
            .collect();
        
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(_, entry)| entry).collect()
    }
    
    fn cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

// ============================================================================
// EXPLANATION GENERATOR
// ============================================================================

/// Explanation generator: E = h_ψ(z, k)
#[derive(Debug, Clone)]
pub struct FailureExplainer {
    pub verbose: bool,
}

impl FailureExplainer {
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }
    
    /// Generate human-readable explanation
    pub fn explain(
        &self,
        cause: &FailureCause,
        adjustment: &StrategyController,
        similar_failures: &[&FailureEntry],
    ) -> String {
        let mut explanation = String::new();
        
        // Explain cause
        explanation.push_str(&format!(
            "I failed because: {}\n\n",
            cause.description()
        ));
        
        // Explain what will be adjusted
        explanation.push_str("To improve, I will:\n");
        
        if adjustment.retrieval_count_delta > 0 {
            explanation.push_str(&format!(
                "- Retrieve {} more pieces of context\n",
                adjustment.retrieval_count_delta
            ));
        }
        
        if adjustment.reasoning_depth_delta > 0.0 {
            explanation.push_str(&format!(
                "- Add {} more reasoning steps\n",
                adjustment.reasoning_depth_delta as usize
            ));
        }
        
        if adjustment.verification_strictness_delta > 0.0 {
            explanation.push_str("- Be more careful with verification\n");
        }
        
        if adjustment.confidence_delta > 0.0 {
            explanation.push_str("- Require higher confidence before answering\n");
        }
        
        // Mention similar past failures if verbose
        if self.verbose && !similar_failures.is_empty() {
            explanation.push_str(&format!(
                "\nI've encountered {} similar failure(s) before.\n",
                similar_failures.len()
            ));
        }
        
        explanation.push_str("\nLet me try again with these adjustments.\n");
        
        explanation
    }
}

// ============================================================================
// COMPLETE FAILURE REASONING MODULE
// ============================================================================

/// Complete failure reasoning module
#[derive(Debug, Clone)]
pub struct FailureReasoningModule {
    pub detector: FailureDetector,
    pub attributor: FailureAttributor,
    pub memory: FailureMemory,
    pub explainer: FailureExplainer,
    pub parameters: SystemParameters,
}

impl FailureReasoningModule {
    pub fn new(dim: usize) -> Self {
        Self {
            detector: FailureDetector::new(0.5, 0.6),
            attributor: FailureAttributor::new(dim),
            memory: FailureMemory::new(100),
            explainer: FailureExplainer::new(true),
            parameters: SystemParameters::default(),
        }
    }
    
    /// Complete failure reasoning loop
    pub fn process_failure(
        &mut self,
        input: &str,
        output: &str,
        expected: Option<&str>,
        confidence: f64,
        user_embedding: &[f64],
        memory_context: &[f64],
    ) -> FailureReasoningResult {
        // 1. Detect failure
        let detection = self.detector.detect_failure(output, expected, confidence);
        
        if !detection.is_failure {
            return FailureReasoningResult {
                is_failure: false,
                cause: None,
                adjustment: None,
                explanation: None,
                should_retry: false,
                adjusted_parameters: self.parameters.clone(),
            };
        }
        
        // 2. Encode failure context
        let latent_failure = self.attributor.encode_failure(
            input,
            output,
            user_embedding,
            memory_context,
        );
        
        // 3. Classify cause
        let cause = self.attributor.classify_cause(&latent_failure, &detection.failure_type);
        
        // 4. Compute strategy adjustment
        let adjustment = StrategyController::compute_adjustment(&cause);
        
        // 5. Apply adjustment to parameters
        let mut adjusted_params = self.parameters.clone();
        adjustment.apply(&mut adjusted_params);
        
        // 6. Store in failure memory
        self.memory.add_failure(
            input.to_string(),
            output.to_string(),
            latent_failure.clone(),
            cause.clone(),
        );
        
        // 7. Get similar past failures
        let similar = self.memory.get_similar_failures(&latent_failure, 3);
        
        // 8. Generate explanation
        let explanation = self.explainer.explain(&cause, &adjustment, &similar);
        
        FailureReasoningResult {
            is_failure: true,
            cause: Some(cause),
            adjustment: Some(adjustment),
            explanation: Some(explanation),
            should_retry: true,
            adjusted_parameters: adjusted_params,
        }
    }
    
    /// Mark last failure as resolved
    pub fn mark_last_resolved(&mut self) {
        if !self.memory.entries.is_empty() {
            let last_idx = self.memory.entries.len() - 1;
            self.memory.mark_resolved(last_idx);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureReasoningResult {
    pub is_failure: bool,
    pub cause: Option<FailureCause>,
    pub adjustment: Option<StrategyController>,
    pub explanation: Option<String>,
    pub should_retry: bool,
    pub adjusted_parameters: SystemParameters,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_failure_detection() {
        let detector = FailureDetector::new(0.5, 0.6);
        
        // Low confidence → failure
        let detection = detector.detect_failure("answer", None, 0.4);
        assert!(detection.is_failure);
        assert_eq!(detection.failure_type, FailureType::LowConfidence);
        
        // High confidence → success
        let detection = detector.detect_failure("answer", None, 0.9);
        assert!(!detection.is_failure);
    }
    
    #[test]
    fn test_failure_attribution() {
        let attributor = FailureAttributor::new(128);
        
        let z = attributor.encode_failure(
            "input",
            "output",
            &vec![0.5; 32],
            &vec![0.3; 32],
        );
        
        assert_eq!(z.len(), 128);
        
        let cause = attributor.classify_cause(&z, &FailureType::LowConfidence);
        assert!(matches!(
            cause,
            FailureCause::KnowledgeGap | FailureCause::ReasoningError | FailureCause::RetrievalMismatch
        ));
    }
    
    #[test]
    fn test_strategy_adjustment() {
        let adjustment = StrategyController::compute_adjustment(&FailureCause::ReasoningError);
        
        assert!(adjustment.reasoning_depth_delta > 0.0);
        assert!(adjustment.verification_strictness_delta > 0.0);
        
        let mut params = SystemParameters::default();
        let original_depth = params.reasoning_depth;
        
        adjustment.apply(&mut params);
        
        assert!(params.reasoning_depth > original_depth);
    }
    
    #[test]
    fn test_failure_memory() {
        let mut memory = FailureMemory::new(10);
        
        memory.add_failure(
            "input1".to_string(),
            "output1".to_string(),
            vec![0.5; 128],
            FailureCause::KnowledgeGap,
        );
        
        assert_eq!(memory.entries.len(), 1);
        
        memory.mark_resolved(0);
        assert!(memory.entries[0].resolved);
    }
    
    #[test]
    fn test_failure_explainer() {
        let explainer = FailureExplainer::new(true);
        let adjustment = StrategyController::compute_adjustment(&FailureCause::ReasoningError);
        
        let explanation = explainer.explain(&FailureCause::ReasoningError, &adjustment, &[]);
        
        assert!(explanation.contains("failed because"));
        assert!(explanation.contains("To improve"));
    }
    
    #[test]
    fn test_complete_module() {
        let mut module = FailureReasoningModule::new(128);
        
        let result = module.process_failure(
            "What is 2+2?",
            "5",  // Wrong answer
            Some("4"),
            0.5,
            &vec![0.5; 32],
            &vec![0.3; 32],
        );
        
        assert!(result.is_failure);
        assert!(result.cause.is_some());
        assert!(result.should_retry);
        assert!(result.explanation.is_some());
    }
}
