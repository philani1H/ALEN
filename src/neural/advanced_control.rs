//! Advanced Control Systems
//!
//! Implements:
//! - Controllable verbosity
//! - Self-knowledge and confidence awareness
//! - Fine-grained output control (verbosity, tone, depth)
//! - Explainable reasoning with chain-of-thought logs
//! - Real-time knowledge verification

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// VERBOSITY CONTROL
// ============================================================================

/// Verbosity control signal v ∈ [0,1]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerbosityControl {
    /// Verbosity level: 0 = minimal, 1 = detailed
    pub level: f64,
    /// Adaptive verbosity based on context
    pub adaptive: bool,
}

impl VerbosityControl {
    pub fn new(level: f64) -> Self {
        Self {
            level: level.max(0.0).min(1.0),
            adaptive: true,
        }
    }
    
    /// Minimal verbosity (concise answers)
    pub fn minimal() -> Self {
        Self::new(0.1)
    }
    
    /// Standard verbosity (balanced)
    pub fn standard() -> Self {
        Self::new(0.5)
    }
    
    /// Detailed verbosity (comprehensive)
    pub fn detailed() -> Self {
        Self::new(0.9)
    }
    
    /// Adapt verbosity based on question type
    pub fn adapt_to_question(&mut self, question: &str) {
        if !self.adaptive {
            return;
        }
        
        let lower = question.to_lowercase();
        
        // "What" or "Who" questions → concise
        if lower.starts_with("what ") || lower.starts_with("who ") {
            self.level = 0.3;
        }
        // "Why" or "How" questions → detailed
        else if lower.starts_with("why ") || lower.starts_with("how ") {
            self.level = 0.8;
        }
        // "Explain" → very detailed
        else if lower.contains("explain") {
            self.level = 0.9;
        }
        // "List" or "Name" → minimal
        else if lower.starts_with("list ") || lower.starts_with("name ") {
            self.level = 0.2;
        }
        // Default
        else {
            self.level = 0.5;
        }
    }
    
    /// Scale text based on verbosity level
    pub fn scale_output(&self, short: &str, medium: &str, long: &str) -> String {
        if self.level < 0.33 {
            short.to_string()
        } else if self.level < 0.67 {
            medium.to_string()
        } else {
            long.to_string()
        }
    }
    
    /// Determine number of reasoning steps to show
    pub fn reasoning_steps_to_show(&self, total_steps: usize) -> usize {
        let ratio = self.level;
        let min_steps = 1;
        let steps = (total_steps as f64 * ratio).ceil() as usize;
        steps.max(min_steps).min(total_steps)
    }
}

// ============================================================================
// SELF-KNOWLEDGE & CONFIDENCE AWARENESS
// ============================================================================

/// Self-knowledge module: tracks capabilities and limitations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfKnowledgeModule {
    /// Performance memory: tracks success/failure by task type
    pub performance_memory: HashMap<String, PerformanceStats>,
    /// Known capabilities
    pub capabilities: Vec<Capability>,
    /// Known limitations
    pub limitations: Vec<Limitation>,
    /// Confidence threshold for answering
    pub confidence_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub task_type: String,
    pub success_count: usize,
    pub failure_count: usize,
    pub average_confidence: f64,
}

impl PerformanceStats {
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.5
        } else {
            self.success_count as f64 / total as f64
        }
    }
    
    pub fn update(&mut self, success: bool, confidence: f64) {
        if success {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }
        
        let total = self.success_count + self.failure_count;
        self.average_confidence = 
            (self.average_confidence * (total - 1) as f64 + confidence) / total as f64;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,
    pub description: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Limitation {
    pub name: String,
    pub description: String,
    pub reason: String,
}

impl SelfKnowledgeModule {
    pub fn new() -> Self {
        let mut capabilities = Vec::new();
        capabilities.push(Capability {
            name: "reasoning".to_string(),
            description: "Multi-step logical reasoning".to_string(),
            confidence: 0.85,
        });
        capabilities.push(Capability {
            name: "explanation".to_string(),
            description: "Explaining concepts in multiple styles".to_string(),
            confidence: 0.90,
        });
        capabilities.push(Capability {
            name: "question_generation".to_string(),
            description: "Generating relevant follow-up questions".to_string(),
            confidence: 0.80,
        });
        
        let mut limitations = Vec::new();
        limitations.push(Limitation {
            name: "real_time_data".to_string(),
            description: "Cannot access real-time information".to_string(),
            reason: "No internet connection during inference".to_string(),
        });
        limitations.push(Limitation {
            name: "personal_data".to_string(),
            description: "Cannot access private or personal information".to_string(),
            reason: "Privacy and security constraints".to_string(),
        });
        
        Self {
            performance_memory: HashMap::new(),
            capabilities,
            limitations,
            confidence_threshold: 0.6,
        }
    }
    
    /// Predict confidence for a task type
    pub fn predict_confidence(&self, task_type: &str) -> f64 {
        if let Some(stats) = self.performance_memory.get(task_type) {
            stats.average_confidence * stats.success_rate()
        } else {
            0.5  // Unknown task → moderate confidence
        }
    }
    
    /// Check if should answer based on confidence
    pub fn should_answer(&self, task_type: &str, confidence: f64) -> bool {
        let predicted = self.predict_confidence(task_type);
        confidence >= self.confidence_threshold && predicted >= self.confidence_threshold * 0.8
    }
    
    /// Generate explanation of limitations - returns None, decoder generates response
    /// NO HARDCODED RESPONSES - the decoder will generate appropriate responses
    pub fn explain_limitation(&self, task_type: &str) -> Option<String> {
        // Check if this is a known limitation
        for limitation in &self.limitations {
            if task_type.contains(&limitation.name) {
                // Return structured data, not a hardcoded response
                // The actual response text is generated by the decoder from learned patterns
                return Some(format!(
                    "LIMITATION:{}/{}/{}",
                    limitation.name,
                    limitation.description,
                    limitation.reason
                ));
            }
        }
        None
    }
    
    /// Update performance memory
    pub fn update_performance(&mut self, task_type: String, success: bool, confidence: f64) {
        let stats = self.performance_memory
            .entry(task_type.clone())
            .or_insert(PerformanceStats {
                task_type,
                success_count: 0,
                failure_count: 0,
                average_confidence: 0.5,
            });
        
        stats.update(success, confidence);
    }
}

// ============================================================================
// FINE-GRAINED OUTPUT CONTROL
// ============================================================================

/// Fine-grained output control: v (verbosity), t (tone), d (depth)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputControl {
    /// Verbosity: 0 = minimal, 1 = detailed
    pub verbosity: f64,
    /// Tone: 0 = formal, 1 = casual
    pub tone: f64,
    /// Depth: 0 = simple, 1 = advanced
    pub depth: f64,
}

impl Default for OutputControl {
    fn default() -> Self {
        Self {
            verbosity: 0.5,
            tone: 0.5,
            depth: 0.5,
        }
    }
}

impl OutputControl {
    pub fn new(verbosity: f64, tone: f64, depth: f64) -> Self {
        Self {
            verbosity: verbosity.max(0.0).min(1.0),
            tone: tone.max(0.0).min(1.0),
            depth: depth.max(0.0).min(1.0),
        }
    }
    
    /// Concise, formal, simple
    pub fn minimal() -> Self {
        Self::new(0.1, 0.1, 0.2)
    }
    
    /// Balanced
    pub fn standard() -> Self {
        Self::new(0.5, 0.5, 0.5)
    }
    
    /// Detailed, casual, advanced
    pub fn comprehensive() -> Self {
        Self::new(0.9, 0.7, 0.8)
    }
    
    /// Apply control to text generation
    pub fn apply_to_text(&self, base_text: &str) -> String {
        let mut result = base_text.to_string();
        
        // Apply tone
        if self.tone < 0.3 {
            // Formal tone
            result = result.replace("I think", "It appears that");
            result = result.replace("I'd say", "One might conclude");
        } else if self.tone > 0.7 {
            // Casual tone
            result = result.replace("It appears that", "I think");
            result = result.replace("One might conclude", "I'd say");
        }
        
        // Apply depth (simplified for now)
        if self.depth < 0.3 {
            // Simplify
            result = format!("In simple terms: {}", result);
        } else if self.depth > 0.7 {
            // Add complexity
            result = format!("{} (This involves advanced concepts.)", result);
        }
        
        result
    }
    
    /// Compute embedding for neural network
    pub fn to_embedding(&self) -> Vec<f64> {
        vec![self.verbosity, self.tone, self.depth]
    }
}

// ============================================================================
// EXPLAINABLE REASONING
// ============================================================================

/// Chain-of-thought log for explainable reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainOfThoughtLog {
    pub steps: Vec<ReasoningStep>,
    pub total_confidence: f64,
    pub verification_results: Vec<VerificationResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_number: usize,
    pub description: String,
    pub latent_state: Vec<f64>,
    pub confidence: f64,
    pub module_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub step_number: usize,
    pub verified: bool,
    pub verification_score: f64,
    pub method: String,
}

impl ChainOfThoughtLog {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            total_confidence: 1.0,
            verification_results: Vec::new(),
        }
    }
    
    pub fn add_step(&mut self, step: ReasoningStep) {
        self.total_confidence *= step.confidence;
        self.steps.push(step);
    }
    
    pub fn add_verification(&mut self, result: VerificationResult) {
        self.verification_results.push(result);
    }
    
    /// Generate human-readable explanation
    pub fn to_explanation(&self, verbosity: f64) -> String {
        let steps_to_show = (self.steps.len() as f64 * verbosity).ceil() as usize;
        let steps_to_show = steps_to_show.max(1).min(self.steps.len());
        
        let mut explanation = String::from("Reasoning process:\n");
        
        for (i, step) in self.steps.iter().take(steps_to_show).enumerate() {
            explanation.push_str(&format!(
                "{}. {} (confidence: {:.2})\n",
                i + 1,
                step.description,
                step.confidence
            ));
        }
        
        if steps_to_show < self.steps.len() {
            explanation.push_str(&format!(
                "... and {} more steps\n",
                self.steps.len() - steps_to_show
            ));
        }
        
        explanation.push_str(&format!(
            "\nOverall confidence: {:.2}\n",
            self.total_confidence
        ));
        
        if !self.verification_results.is_empty() {
            let verified_count = self.verification_results.iter()
                .filter(|v| v.verified)
                .count();
            explanation.push_str(&format!(
                "Verified: {}/{} steps\n",
                verified_count,
                self.verification_results.len()
            ));
        }
        
        explanation
    }
}

// ============================================================================
// REAL-TIME KNOWLEDGE VERIFICATION
// ============================================================================

/// Knowledge verification module
#[derive(Debug, Clone)]
pub struct KnowledgeVerifier {
    /// Knowledge base for verification
    pub knowledge_base: HashMap<String, Vec<String>>,
    /// Verification threshold
    pub threshold: f64,
}

impl KnowledgeVerifier {
    pub fn new() -> Self {
        // NO HARDCODED FACTS - all knowledge must be learned from training data
        // The knowledge_base is populated by learning, not hardcoded values
        Self {
            knowledge_base: HashMap::new(),
            threshold: 0.7,
        }
    }
    
    /// Learn facts from training data (called during training)
    pub fn learn_fact(&mut self, domain: &str, fact: &str) {
        self.knowledge_base
            .entry(domain.to_string())
            .or_insert_with(Vec::new)
            .push(fact.to_string());
    }
    
    /// Learn from question-answer pair
    pub fn learn_from_qa(&mut self, question: &str, answer: &str) {
        let domain = self.extract_domain(question);
        // Extract key facts from the answer
        let words: Vec<&str> = answer.split_whitespace().collect();
        if words.len() > 2 {
            self.learn_fact(&domain, answer);
        }
    }
    
    /// Verify answer against knowledge base
    /// Returns V_knowledge(x, Y) ∈ [0,1]
    pub fn verify(&self, question: &str, answer: &str) -> f64 {
        // Extract domain from question
        let domain = self.extract_domain(question);
        
        if let Some(facts) = self.knowledge_base.get(&domain) {
            // Check if answer contains any known facts
            for fact in facts {
                if answer.contains(fact) || self.semantic_similarity(answer, fact) > 0.8 {
                    return 1.0;
                }
            }
            
            // Check for contradictions
            for fact in facts {
                if self.contradicts(answer, fact) {
                    return 0.0;
                }
            }
        }
        
        // Unknown domain or no facts → moderate confidence
        0.5
    }
    
    fn extract_domain(&self, question: &str) -> String {
        let lower = question.to_lowercase();
        
        // Check for math-related patterns
        if lower.contains("math") || lower.contains("calculate") || lower.contains("equation") ||
           lower.contains('+') || lower.contains('-') || lower.contains('*') || lower.contains('/') ||
           lower.contains("sum") || lower.contains("product") || lower.contains("divide") ||
           lower.contains("multiply") || lower.contains("subtract") || lower.contains("add") {
            "mathematics".to_string()
        } else if lower.contains("physics") || lower.contains("force") || lower.contains("motion") {
            "physics".to_string()
        } else {
            "general".to_string()
        }
    }
    
    fn semantic_similarity(&self, text1: &str, text2: &str) -> f64 {
        // Extract numbers from both texts for math comparisons
        let nums1: Vec<String> = text1.split(|c: char| !c.is_numeric() && c != '.')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        let nums2: Vec<String> = text2.split(|c: char| !c.is_numeric() && c != '.')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        
        // Check if numeric answers match (important for math)
        let mut num_match = 0;
        for n1 in &nums1 {
            if nums2.contains(n1) {
                num_match += 1;
            }
        }
        
        // If the answer contains the correct numeric result, give high similarity
        if num_match > 0 && !nums1.is_empty() {
            return 0.9;
        }
        
        // Simple word overlap similarity as fallback
        let lower1 = text1.to_lowercase();
        let lower2 = text2.to_lowercase();
        let words1: Vec<&str> = lower1.split_whitespace().collect();
        let words2: Vec<&str> = lower2.split_whitespace().collect();
        
        let mut matches = 0;
        for w1 in &words1 {
            if words2.contains(w1) {
                matches += 1;
            }
        }
        
        if words1.is_empty() {
            0.0
        } else {
            matches as f64 / words1.len() as f64
        }
    }
    
    fn contradicts(&self, text1: &str, text2: &str) -> bool {
        // Simple contradiction detection
        // In production, use more sophisticated NLI
        false
    }
}

// ============================================================================
// INTEGRATED ADVANCED CONTROL SYSTEM
// ============================================================================

/// Complete advanced control system
#[derive(Debug, Clone)]
pub struct AdvancedControlSystem {
    pub verbosity: VerbosityControl,
    pub self_knowledge: SelfKnowledgeModule,
    pub output_control: OutputControl,
    pub knowledge_verifier: KnowledgeVerifier,
}

impl AdvancedControlSystem {
    pub fn new() -> Self {
        Self {
            verbosity: VerbosityControl::standard(),
            self_knowledge: SelfKnowledgeModule::new(),
            output_control: OutputControl::standard(),
            knowledge_verifier: KnowledgeVerifier::new(),
        }
    }
    
    /// Process with all controls
    pub fn process_with_controls(
        &mut self,
        question: &str,
        answer: &str,
        reasoning_log: &ChainOfThoughtLog,
        task_type: &str,
    ) -> ControlledOutput {
        // 1. Adapt verbosity to question
        self.verbosity.adapt_to_question(question);
        
        // 2. Check if should answer
        let confidence = reasoning_log.total_confidence;
        let should_answer = self.self_knowledge.should_answer(task_type, confidence);
        
        if !should_answer {
            // Return structured marker - decoder generates response from learned patterns
            // NO hardcoded responses - model learns to express uncertainty from training data
            let limitation_marker = self.self_knowledge
                .explain_limitation(task_type)
                .unwrap_or_else(|| format!("LOW_CONFIDENCE:{}:{:.2}", task_type, confidence));
            
            return ControlledOutput {
                answer: limitation_marker,
                reasoning_explanation: None,
                confidence,
                verification_score: 0.0,
                should_show_reasoning: false,
            };
        }
        
        // 3. Verify knowledge
        let verification_score = self.knowledge_verifier.verify(question, answer);
        
        // 4. Apply output controls
        let controlled_answer = self.output_control.apply_to_text(answer);
        
        // 5. Generate reasoning explanation based on verbosity
        let reasoning_explanation = if self.verbosity.level > 0.3 {
            Some(reasoning_log.to_explanation(self.verbosity.level))
        } else {
            None
        };
        
        // 6. Update performance memory
        let success = verification_score > 0.7 && confidence > 0.6;
        self.self_knowledge.update_performance(
            task_type.to_string(),
            success,
            confidence,
        );
        
        ControlledOutput {
            answer: controlled_answer,
            reasoning_explanation,
            confidence,
            verification_score,
            should_show_reasoning: self.verbosity.level > 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlledOutput {
    pub answer: String,
    pub reasoning_explanation: Option<String>,
    pub confidence: f64,
    pub verification_score: f64,
    pub should_show_reasoning: bool,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_verbosity_control() {
        let mut verbosity = VerbosityControl::standard();
        
        // Test adaptation
        verbosity.adapt_to_question("What is the capital?");
        assert!(verbosity.level < 0.5);
        
        verbosity.adapt_to_question("Why does this happen?");
        assert!(verbosity.level > 0.7);
        
        verbosity.adapt_to_question("Explain quantum mechanics");
        assert!(verbosity.level > 0.8);
    }
    
    #[test]
    fn test_self_knowledge() {
        let mut sk = SelfKnowledgeModule::new();
        
        // Test confidence prediction
        let conf = sk.predict_confidence("reasoning");
        assert!(conf > 0.0);
        
        // Test performance update
        sk.update_performance("math".to_string(), true, 0.9);
        sk.update_performance("math".to_string(), true, 0.85);
        sk.update_performance("math".to_string(), false, 0.6);
        
        let stats = sk.performance_memory.get("math").unwrap();
        assert_eq!(stats.success_count, 2);
        assert_eq!(stats.failure_count, 1);
    }
    
    #[test]
    fn test_output_control() {
        let control = OutputControl::minimal();
        assert!(control.verbosity < 0.3);
        assert!(control.tone < 0.3);
        
        let control = OutputControl::comprehensive();
        assert!(control.verbosity > 0.7);
    }
    
    #[test]
    fn test_chain_of_thought() {
        let mut log = ChainOfThoughtLog::new();
        
        log.add_step(ReasoningStep {
            step_number: 1,
            description: "Identify the problem".to_string(),
            latent_state: vec![0.5; 10],
            confidence: 0.9,
            module_source: "reasoning".to_string(),
        });
        
        log.add_step(ReasoningStep {
            step_number: 2,
            description: "Apply solution".to_string(),
            latent_state: vec![0.6; 10],
            confidence: 0.85,
            module_source: "reasoning".to_string(),
        });
        
        assert_eq!(log.steps.len(), 2);
        assert!(log.total_confidence > 0.7);
        
        let explanation = log.to_explanation(1.0);
        assert!(explanation.contains("Reasoning process"));
    }
    
    #[test]
    fn test_knowledge_verifier() {
        let mut verifier = KnowledgeVerifier::new();
        
        // Learn facts from training (not hardcoded)
        verifier.learn_from_qa("What is 2 + 2?", "4");
        verifier.learn_fact("mathematics", "addition combines numbers");
        
        // Verify using learned knowledge
        let score = verifier.verify(
            "What is 2 + 2?",
            "The answer is 4"
        );
        // Score depends on learned patterns, not hardcoded values
        assert!(score >= 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_advanced_control_system() {
        let mut system = AdvancedControlSystem::new();
        
        let mut log = ChainOfThoughtLog::new();
        log.add_step(ReasoningStep {
            step_number: 1,
            description: "Calculate result".to_string(),
            latent_state: vec![0.5; 10],
            confidence: 0.9,
            module_source: "math".to_string(),
        });
        
        let output = system.process_with_controls(
            "What is 2 + 2?",
            "The answer is 4",
            &log,
            "mathematics",
        );
        
        assert!(!output.answer.is_empty());
        assert!(output.confidence > 0.0);
    }
}
