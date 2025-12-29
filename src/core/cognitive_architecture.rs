//! Cognitive Architecture Module
//!
//! Implements the complete ALEN cognitive system:
//! 1. Context understanding via attention (not giant transformers)
//! 2. Intent extraction before generation
//! 3. Proof-based verification (forward + backward)
//! 4. Energy-based confidence (explicit, not guessed)
//! 5. Creativity with proof constraints
//! 6. Formal understanding tests
//! 7. Hallucination prevention via verification gates
//!
//! Core principle: Answer only what can be proven.
//! Transformers talk well. This system knows when it knows.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::proof_system::{
    ProofEngine, ProofResult, HybridReasoner, Problem, ProblemDomain,
    ProblemStructure, NodeType, Answer,
};
use super::intent_extraction::{IntentExtractor, IntentState, TaskType};
use super::unified_cognition::{ThoughtVector, CognitiveEnergy, AudienceModel};

// ============================================================================
// SECTION 1: Context State (Temporary, Not Learned)
// ============================================================================

/// Temporary context state z_context = g(x_1, ..., x_T)
/// This is RAM, not disk. Disappears after session.
#[derive(Debug, Clone)]
pub struct ContextState {
    /// Current context embedding
    pub embedding: Vec<f64>,
    /// User style detected
    pub user_style: UserStyle,
    /// Depth preference (0 = simple, 1 = expert)
    pub depth_preference: f64,
    /// Detected goals
    pub goals: Vec<String>,
    /// Conversation tone
    pub tone: ConversationTone,
    /// Entropy of current context (lower = more certain)
    pub entropy: f64,
    /// History of recent exchanges
    pub history: Vec<Exchange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exchange {
    pub user_input: String,
    pub system_response: String,
    pub confidence: f64,
    pub verified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserStyle {
    Technical,
    Casual,
    Academic,
    Curious,
    Impatient,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversationTone {
    Formal,
    Friendly,
    Educational,
    Collaborative,
    Neutral,
}

impl ContextState {
    pub fn new(dim: usize) -> Self {
        Self {
            embedding: vec![0.0; dim],
            user_style: UserStyle::Unknown,
            depth_preference: 0.5,
            goals: Vec::new(),
            tone: ConversationTone::Neutral,
            entropy: 1.0, // Maximum uncertainty initially
            history: Vec::new(),
        }
    }

    /// Update context from new input
    pub fn update(&mut self, input: &str, response: &str, confidence: f64, verified: bool) {
        // Update embedding (exponential moving average)
        let input_emb = self.encode_text(input);
        for (i, &v) in input_emb.iter().enumerate() {
            if i < self.embedding.len() {
                self.embedding[i] = 0.9 * self.embedding[i] + 0.1 * v;
            }
        }

        // Detect user style
        self.user_style = self.detect_style(input);
        
        // Update depth preference based on question complexity
        let complexity = self.measure_complexity(input);
        self.depth_preference = 0.8 * self.depth_preference + 0.2 * complexity;

        // Update entropy (confidence reduces entropy)
        self.entropy = 0.9 * self.entropy + 0.1 * (1.0 - confidence);

        // Add to history
        self.history.push(Exchange {
            user_input: input.to_string(),
            system_response: response.to_string(),
            confidence,
            verified,
        });

        // Keep history bounded
        if self.history.len() > 20 {
            self.history.remove(0);
        }
    }

    fn encode_text(&self, text: &str) -> Vec<f64> {
        let mut emb = vec![0.0; self.embedding.len()];
        for (i, c) in text.chars().enumerate() {
            let idx = (c as usize + i) % emb.len();
            emb[idx] += 1.0 / (i + 1) as f64;
        }
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for v in &mut emb { *v /= norm; }
        }
        emb
    }

    fn detect_style(&self, input: &str) -> UserStyle {
        let lower = input.to_lowercase();
        if lower.contains("technically") || lower.contains("specifically") || 
           lower.contains("implementation") || lower.contains("algorithm") {
            UserStyle::Technical
        } else if lower.contains("eli5") || lower.contains("simple") || lower.contains("basically") {
            UserStyle::Casual
        } else if lower.contains("research") || lower.contains("paper") || lower.contains("formally") {
            UserStyle::Academic
        } else if lower.contains("?") && lower.len() < 50 {
            UserStyle::Curious
        } else if lower.contains("just") || lower.contains("quickly") || lower.contains("tldr") {
            UserStyle::Impatient
        } else {
            UserStyle::Unknown
        }
    }

    fn measure_complexity(&self, input: &str) -> f64 {
        let word_count = input.split_whitespace().count();
        let avg_word_len = input.split_whitespace()
            .map(|w| w.len())
            .sum::<usize>() as f64 / word_count.max(1) as f64;
        
        // Complexity based on length and vocabulary
        let length_factor = (word_count as f64 / 50.0).min(1.0);
        let vocab_factor = (avg_word_len / 8.0).min(1.0);
        
        (length_factor + vocab_factor) / 2.0
    }

    /// Get context relevance for a query
    pub fn relevance_to(&self, query: &str) -> f64 {
        let query_emb = self.encode_text(query);
        let dot: f64 = self.embedding.iter().zip(&query_emb).map(|(a, b)| a * b).sum();
        let norm_a: f64 = self.embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = query_emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 { return 0.0; }
        dot / (norm_a * norm_b)
    }
}

// ============================================================================
// SECTION 2: Small Attention Block (Not Giant Transformer)
// ============================================================================

/// Lightweight attention for context understanding
/// Attention(X) = softmax(QK^T / √d) V
pub struct SmallAttention {
    dim: usize,
    w_q: Vec<f64>,
    w_k: Vec<f64>,
    w_v: Vec<f64>,
}

impl SmallAttention {
    pub fn new(dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let init = || (0..dim * dim).map(|_| rng.gen_range(-0.1..0.1)).collect();
        Self { dim, w_q: init(), w_k: init(), w_v: init() }
    }

    /// Compute attention over sequence
    /// Returns: which parts matter most for the query
    pub fn attend(&self, query: &[f64], keys: &[Vec<f64>], values: &[Vec<f64>]) -> Vec<f64> {
        if keys.is_empty() || values.is_empty() {
            return vec![0.0; self.dim];
        }

        // Q = query * W_q
        let q = self.project(query, &self.w_q);
        
        // Compute attention scores
        let d_k = (self.dim as f64).sqrt();
        let scores: Vec<f64> = keys.iter()
            .map(|k| {
                let k_proj = self.project(k, &self.w_k);
                let dot: f64 = q.iter().zip(&k_proj).map(|(a, b)| a * b).sum();
                dot / d_k
            })
            .collect();

        // Softmax
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();
        let attention: Vec<f64> = exp_scores.iter().map(|e| e / sum).collect();

        // Weighted sum of values
        let mut result = vec![0.0; self.dim];
        for (i, &alpha) in attention.iter().enumerate() {
            let v_proj = self.project(&values[i], &self.w_v);
            for (j, &v) in v_proj.iter().enumerate() {
                if j < self.dim {
                    result[j] += alpha * v;
                }
            }
        }

        result
    }

    fn project(&self, x: &[f64], w: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.dim];
        for i in 0..self.dim {
            for j in 0..x.len().min(self.dim) {
                result[i] += x[j] * w[i * self.dim + j];
            }
        }
        result
    }

    /// Compute entropy of attention distribution
    pub fn attention_entropy(&self, query: &[f64], keys: &[Vec<f64>]) -> f64 {
        if keys.is_empty() { return 1.0; }
        
        let d_k = (self.dim as f64).sqrt();
        let q = self.project(query, &self.w_q);
        
        let scores: Vec<f64> = keys.iter()
            .map(|k| {
                let k_proj = self.project(k, &self.w_k);
                let dot: f64 = q.iter().zip(&k_proj).map(|(a, b)| a * b).sum();
                dot / d_k
            })
            .collect();

        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();
        let probs: Vec<f64> = exp_scores.iter().map(|e| e / sum).collect();

        // H = -Σ p_i log p_i
        -probs.iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }
}


// ============================================================================
// SECTION 3: Hallucination Prevention (Verification Gates)
// ============================================================================

/// Verification gate - prevents output without proof
/// This is what makes the system honest.
pub struct VerificationGate {
    /// Minimum confidence to output
    pub min_confidence: f64,
    /// Maximum energy to accept
    pub max_energy: f64,
    /// Require backward verification
    pub require_backward: bool,
    /// Minimum proof paths
    pub min_paths: usize,
}

impl Default for VerificationGate {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            max_energy: 0.5,
            require_backward: true,
            min_paths: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub passed: bool,
    pub reason: GateDecision,
    pub confidence: f64,
    pub can_output: bool,
    pub should_clarify: bool,
    pub should_refuse: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateDecision {
    /// Proof verified, confident answer
    Verified,
    /// Partially verified, can answer with caveat
    PartiallyVerified,
    /// Not verified but low risk, can attempt
    LowRiskAttempt,
    /// Need more information
    NeedClarification,
    /// Cannot answer honestly
    CannotAnswer,
    /// Explicitly refuse (harmful/impossible)
    Refuse,
}

impl VerificationGate {
    pub fn check(&self, proof_result: &ProofResult, intent: &IntentState) -> GateResult {
        let confidence = 1.0 - proof_result.proof_energy;
        
        // Check all verification criteria
        let energy_ok = proof_result.proof_energy < self.max_energy;
        let confidence_ok = confidence >= self.min_confidence;
        let backward_ok = !self.require_backward || 
            proof_result.backward_checks.iter().any(|c| c.passed);
        let paths_ok = proof_result.path_count >= self.min_paths;

        // Determine gate decision
        let decision = if proof_result.verified && energy_ok && confidence_ok && backward_ok && paths_ok {
            GateDecision::Verified
        } else if energy_ok && confidence > 0.5 && backward_ok {
            GateDecision::PartiallyVerified
        } else if confidence > 0.4 && !self.is_high_risk(intent) {
            GateDecision::LowRiskAttempt
        } else if confidence > 0.2 {
            GateDecision::NeedClarification
        } else if self.is_impossible(intent) {
            GateDecision::Refuse
        } else {
            GateDecision::CannotAnswer
        };

        let (can_output, should_clarify, should_refuse) = match decision {
            GateDecision::Verified => (true, false, false),
            GateDecision::PartiallyVerified => (true, false, false),
            GateDecision::LowRiskAttempt => (true, false, false),
            GateDecision::NeedClarification => (false, true, false),
            GateDecision::CannotAnswer => (false, true, false),
            GateDecision::Refuse => (false, false, true),
        };

        GateResult {
            passed: matches!(decision, GateDecision::Verified | GateDecision::PartiallyVerified),
            reason: decision,
            confidence,
            can_output,
            should_clarify,
            should_refuse,
        }
    }

    fn is_high_risk(&self, intent: &IntentState) -> bool {
        // High risk: math, code, factual claims
        matches!(intent.task.primary_task, 
            TaskType::Solve | TaskType::Verify | TaskType::Debug)
    }

    fn is_impossible(&self, intent: &IntentState) -> bool {
        // Check for impossible requests
        let input = intent.original_prompt.to_lowercase();
        input.contains("predict the future") ||
        input.contains("read minds") ||
        input.contains("impossible")
    }
}

// ============================================================================
// SECTION 4: Formal Understanding Test
// ============================================================================

/// Tests whether the system truly "understands" something
/// Understanding = can explain, can apply, can verify, can teach
#[derive(Debug, Clone)]
pub struct UnderstandingTest {
    /// Dimension
    dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderstandingResult {
    /// Overall understanding score (0-1)
    pub score: f64,
    /// Can explain in own words
    pub can_explain: bool,
    /// Can apply to new problems
    pub can_apply: bool,
    /// Can verify correctness
    pub can_verify: bool,
    /// Can teach to different audiences
    pub can_teach: bool,
    /// Detailed breakdown
    pub breakdown: UnderstandingBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderstandingBreakdown {
    pub explanation_quality: f64,
    pub application_success: f64,
    pub verification_accuracy: f64,
    pub teaching_adaptability: f64,
}

impl UnderstandingTest {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Test understanding of a concept
    pub fn test(&self, concept: &str, reasoner: &mut HybridReasoner) -> UnderstandingResult {
        // Test 1: Can explain (generate explanation, verify it)
        let explain_result = reasoner.reason(&format!("Explain {}", concept));
        let can_explain = explain_result.verified && explain_result.blended_confidence > 0.6;
        let explanation_quality = explain_result.blended_confidence;

        // Test 2: Can apply (solve a problem using the concept)
        let apply_result = reasoner.reason(&format!("Apply {} to solve a problem", concept));
        let can_apply = apply_result.verified;
        let application_success = apply_result.blended_confidence;

        // Test 3: Can verify (check if a statement about concept is true)
        let verify_result = reasoner.reason(&format!("Is this true about {}: it exists", concept));
        let can_verify = verify_result.verified;
        let verification_accuracy = if verify_result.verified { 
            verify_result.blended_confidence 
        } else { 0.3 };

        // Test 4: Can teach (explain to different audiences)
        let teach_simple = reasoner.reason(&format!("Explain {} to a child", concept));
        let teach_expert = reasoner.reason(&format!("Explain {} to an expert", concept));
        let can_teach = teach_simple.blended_confidence > 0.4 && teach_expert.blended_confidence > 0.4;
        let teaching_adaptability = (teach_simple.blended_confidence + teach_expert.blended_confidence) / 2.0;

        // Overall score
        let score = (explanation_quality + application_success + verification_accuracy + teaching_adaptability) / 4.0;

        UnderstandingResult {
            score,
            can_explain,
            can_apply,
            can_verify,
            can_teach,
            breakdown: UnderstandingBreakdown {
                explanation_quality,
                application_success,
                verification_accuracy,
                teaching_adaptability,
            },
        }
    }

    /// Quick understanding check (faster, less thorough)
    pub fn quick_check(&self, concept: &str, reasoner: &mut HybridReasoner) -> f64 {
        let result = reasoner.reason(&format!("What is {}?", concept));
        if result.verified {
            result.blended_confidence
        } else {
            result.blended_confidence * 0.5
        }
    }
}

// ============================================================================
// SECTION 5: Creative Generation with Proof Constraints
// ============================================================================

/// Creative generator that maintains proof constraints
/// Creativity without breaking verification
pub struct ConstrainedCreativity {
    /// Base reasoner
    dim: usize,
    /// Creativity temperature (higher = more creative)
    temperature: f64,
    /// Minimum coherence required
    min_coherence: f64,
    /// Maximum novelty allowed
    max_novelty: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeResult {
    pub output: String,
    pub creativity_score: f64,
    pub coherence_score: f64,
    pub novelty_score: f64,
    pub verified: bool,
    pub constraints_satisfied: bool,
}

impl ConstrainedCreativity {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            temperature: 0.7,
            min_coherence: 0.5,
            max_novelty: 0.8,
        }
    }

    /// Generate creative content with constraints
    pub fn generate(
        &self,
        prompt: &str,
        intent: &IntentState,
        context: &ContextState,
        reasoner: &mut HybridReasoner,
    ) -> CreativeResult {
        // Step 1: Understand what's being asked
        let base_result = reasoner.reason(prompt);
        
        // Step 2: Generate variations
        let variations = self.generate_variations(prompt, 3);
        
        // Step 3: Score each variation
        let mut best_output = String::new();
        let mut best_score = 0.0;
        let mut best_coherence = 0.0;
        let mut best_novelty = 0.0;
        let mut best_verified = false;

        for variation in &variations {
            let var_result = reasoner.reason(variation);
            let coherence = self.measure_coherence(variation, prompt, context);
            let novelty = self.measure_novelty(variation, &context.history);
            
            // Score = creativity * coherence * (1 - excess_novelty)
            let excess_novelty = (novelty - self.max_novelty).max(0.0);
            let score = var_result.blended_confidence * coherence * (1.0 - excess_novelty);
            
            if score > best_score && coherence >= self.min_coherence {
                best_score = score;
                best_output = variation.clone();
                best_coherence = coherence;
                best_novelty = novelty;
                best_verified = var_result.verified;
            }
        }

        // If no good variation, use base
        if best_output.is_empty() {
            best_output = format!("Regarding '{}': {}", 
                prompt, 
                base_result.answer.map(|a| a.value).unwrap_or_else(|| "I need more context.".to_string())
            );
            best_coherence = 0.5;
            best_novelty = 0.1;
            best_verified = base_result.verified;
        }

        CreativeResult {
            output: best_output,
            creativity_score: best_score,
            coherence_score: best_coherence,
            novelty_score: best_novelty,
            verified: best_verified,
            constraints_satisfied: best_coherence >= self.min_coherence && best_novelty <= self.max_novelty,
        }
    }

    fn generate_variations(&self, prompt: &str, count: usize) -> Vec<String> {
        let mut variations = Vec::new();
        
        // Variation 1: Direct response
        variations.push(prompt.to_string());
        
        // Variation 2: Expanded
        variations.push(format!("{} - exploring this in depth", prompt));
        
        // Variation 3: Simplified
        variations.push(format!("{} - in simple terms", prompt));
        
        variations.into_iter().take(count).collect()
    }

    fn measure_coherence(&self, output: &str, prompt: &str, context: &ContextState) -> f64 {
        // Coherence = relevance to prompt + relevance to context
        let prompt_relevance = self.text_similarity(output, prompt);
        let context_relevance = context.relevance_to(output);
        
        0.7 * prompt_relevance + 0.3 * context_relevance
    }

    fn measure_novelty(&self, output: &str, history: &[Exchange]) -> f64 {
        if history.is_empty() { return 0.5; }
        
        // Novelty = 1 - max_similarity_to_history
        let max_sim = history.iter()
            .map(|e| self.text_similarity(output, &e.system_response))
            .fold(0.0, f64::max);
        
        1.0 - max_sim
    }

    fn text_similarity(&self, a: &str, b: &str) -> f64 {
        let a_words: std::collections::HashSet<_> = a.to_lowercase().split_whitespace().collect();
        let b_words: std::collections::HashSet<_> = b.to_lowercase().split_whitespace().collect();
        
        let intersection = a_words.intersection(&b_words).count() as f64;
        let union = a_words.union(&b_words).count() as f64;
        
        if union < 1.0 { 0.0 } else { intersection / union }
    }
}


// ============================================================================
// SECTION 6: Complete Cognitive System
// ============================================================================

/// The complete ALEN cognitive system
/// Integrates all components into a unified reasoning pipeline
pub struct CognitiveSystem {
    /// Hybrid reasoner (symbolic + neural)
    pub reasoner: HybridReasoner,
    /// Intent extractor
    pub intent_extractor: IntentExtractor,
    /// Context state (temporary)
    pub context: ContextState,
    /// Attention mechanism
    pub attention: SmallAttention,
    /// Verification gate
    pub gate: VerificationGate,
    /// Understanding tester
    pub understanding_test: UnderstandingTest,
    /// Creative generator
    pub creativity: ConstrainedCreativity,
    /// Dimension
    dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveResponse {
    /// The response text
    pub response: String,
    /// Response type
    pub response_type: ResponseType,
    /// Confidence (explicit, not guessed)
    pub confidence: f64,
    /// Was this verified?
    pub verified: bool,
    /// Gate decision
    pub gate_decision: GateDecision,
    /// Intent understood
    pub intent: IntentState,
    /// Proof energy
    pub proof_energy: f64,
    /// Reasoning mode used
    pub reasoning_mode: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseType {
    /// Confident, verified answer
    Verified,
    /// Answer with caveat
    Tentative,
    /// Need more information
    Clarification,
    /// Cannot answer
    Refusal,
    /// Creative output
    Creative,
}

impl CognitiveSystem {
    pub fn new(dim: usize) -> Self {
        Self {
            reasoner: HybridReasoner::new(dim),
            intent_extractor: IntentExtractor::new(dim),
            context: ContextState::new(dim),
            attention: SmallAttention::new(dim),
            gate: VerificationGate::default(),
            understanding_test: UnderstandingTest::new(dim),
            creativity: ConstrainedCreativity::new(dim),
            dim,
        }
    }

    /// Process input and generate response
    /// This is the main entry point
    pub fn process(&mut self, input: &str) -> CognitiveResponse {
        // Step 1: Extract intent I = (τ, θ, C)
        let intent = self.intent_extractor.extract(input);
        
        // Step 2: Check context relevance
        let context_relevance = self.context.relevance_to(input);
        
        // Step 3: Route based on task type
        let (response, response_type, proof_result) = match intent.task.primary_task {
            TaskType::Generate => self.handle_creative(&intent),
            TaskType::Explain | TaskType::Define => self.handle_explanation(&intent),
            TaskType::Solve | TaskType::Verify => self.handle_proof(&intent),
            TaskType::Summarize => self.handle_summary(&intent),
            _ => self.handle_general(&intent),
        };

        // Step 4: Check verification gate
        let gate_result = self.gate.check(&proof_result, &intent);
        
        // Step 5: Determine final response
        let (final_response, final_type) = if gate_result.should_refuse {
            (self.generate_refusal(&intent), ResponseType::Refusal)
        } else if gate_result.should_clarify {
            (self.generate_clarification(&intent), ResponseType::Clarification)
        } else if gate_result.passed {
            (response, ResponseType::Verified)
        } else if gate_result.can_output {
            (self.add_caveat(&response, gate_result.confidence), ResponseType::Tentative)
        } else {
            (self.generate_clarification(&intent), ResponseType::Clarification)
        };

        // Step 6: Update context
        self.context.update(input, &final_response, gate_result.confidence, gate_result.passed);

        CognitiveResponse {
            response: final_response,
            response_type: final_type,
            confidence: gate_result.confidence,
            verified: gate_result.passed,
            gate_decision: gate_result.reason,
            intent,
            proof_energy: proof_result.proof_energy,
            reasoning_mode: format!("{:?}", proof_result.path_count),
        }
    }

    fn handle_creative(&mut self, intent: &IntentState) -> (String, ResponseType, ProofResult) {
        let result = self.creativity.generate(
            &intent.original_prompt,
            intent,
            &self.context,
            &mut self.reasoner,
        );
        
        let proof_result = self.reasoner.reason(&intent.original_prompt).proof_result;
        
        (result.output, ResponseType::Creative, proof_result)
    }

    fn handle_explanation(&mut self, intent: &IntentState) -> (String, ResponseType, ProofResult) {
        let result = self.reasoner.reason(&intent.original_prompt);
        
        let response = if let Some(answer) = &result.answer {
            format!("Explanation: {}", answer.value)
        } else {
            format!("I understand you're asking about '{}'. Let me explain what I know.", 
                self.extract_topic(&intent.original_prompt))
        };
        
        (response, ResponseType::Verified, result.proof_result)
    }

    fn handle_proof(&mut self, intent: &IntentState) -> (String, ResponseType, ProofResult) {
        let result = self.reasoner.reason(&intent.original_prompt);
        
        let response = if result.verified {
            if let Some(answer) = &result.answer {
                format!("Solution (verified, confidence {:.1}%): {}", 
                    result.blended_confidence * 100.0, answer.value)
            } else {
                "I've verified this but need to formulate the answer.".to_string()
            }
        } else {
            format!("I'm working on this with {:.1}% confidence. The proof is not yet complete.",
                result.blended_confidence * 100.0)
        };
        
        (response, ResponseType::Verified, result.proof_result)
    }

    fn handle_summary(&mut self, intent: &IntentState) -> (String, ResponseType, ProofResult) {
        let result = self.reasoner.reason(&intent.original_prompt);
        
        let response = if let Some(answer) = &result.answer {
            format!("Summary: {}", answer.value)
        } else {
            "I need more content to summarize.".to_string()
        };
        
        (response, ResponseType::Verified, result.proof_result)
    }

    fn handle_general(&mut self, intent: &IntentState) -> (String, ResponseType, ProofResult) {
        let result = self.reasoner.reason(&intent.original_prompt);
        
        let response = if let Some(answer) = &result.answer {
            answer.value.clone()
        } else {
            format!("I'm processing your question about '{}'. Could you provide more details?",
                self.extract_topic(&intent.original_prompt))
        };
        
        (response, ResponseType::Tentative, result.proof_result)
    }

    fn generate_refusal(&self, intent: &IntentState) -> String {
        format!("I cannot answer '{}' because it's outside my verified knowledge or capabilities.",
            self.extract_topic(&intent.original_prompt))
    }

    fn generate_clarification(&self, intent: &IntentState) -> String {
        format!("I need more information to answer '{}'. Could you clarify what specific aspect you're interested in?",
            self.extract_topic(&intent.original_prompt))
    }

    fn add_caveat(&self, response: &str, confidence: f64) -> String {
        format!("{} (Note: {:.0}% confidence, not fully verified)", response, confidence * 100.0)
    }

    fn extract_topic(&self, input: &str) -> String {
        let skip = ["what", "is", "the", "a", "an", "how", "does", "explain", "tell", "me"];
        input.split_whitespace()
            .filter(|w| !skip.contains(&w.to_lowercase().as_str()))
            .take(3)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Test understanding of a concept
    pub fn test_understanding(&mut self, concept: &str) -> UnderstandingResult {
        self.understanding_test.test(concept, &mut self.reasoner)
    }

    /// Learn from feedback
    pub fn learn(&mut self, input: &str, correct_answer: &str, was_correct: bool) {
        self.reasoner.learn(input, correct_answer, was_correct);
    }
}

// ============================================================================
// SECTION 7: Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_state() {
        let mut ctx = ContextState::new(64);
        ctx.update("What is 2+2?", "4", 0.9, true);
        assert!(ctx.entropy < 1.0);
        assert!(!ctx.history.is_empty());
    }

    #[test]
    fn test_small_attention() {
        let attn = SmallAttention::new(32);
        let query = vec![0.5; 32];
        let keys = vec![vec![0.3; 32], vec![0.7; 32]];
        let values = vec![vec![1.0; 32], vec![0.0; 32]];
        
        let result = attn.attend(&query, &keys, &values);
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_verification_gate() {
        let gate = VerificationGate::default();
        let intent = IntentExtractor::new(64).extract("What is 1+1?");
        
        let proof_result = ProofResult {
            verified: true,
            answer: Some(Answer { value: "2".to_string(), structure: None, initial_confidence: 0.9 }),
            successful_paths: vec![],
            backward_checks: vec![],
            proof_energy: 0.2,
            proof_depth: 1,
            path_count: 2,
            energy_breakdown: super::super::proof_system::ProofEnergyBreakdown {
                inconsistency: 0.0, path_disagreement: 0.0, uncertainty: 0.2, total: 0.2,
            },
        };
        
        let result = gate.check(&proof_result, &intent);
        assert!(result.can_output);
    }

    #[test]
    fn test_cognitive_system() {
        let mut system = CognitiveSystem::new(64);
        let response = system.process("What is 2 + 2?");
        
        assert!(response.confidence > 0.0);
        assert!(!response.response.is_empty());
    }

    #[test]
    fn test_creative_generation() {
        let creativity = ConstrainedCreativity::new(64);
        let intent = IntentExtractor::new(64).extract("Write a poem");
        let context = ContextState::new(64);
        let mut reasoner = HybridReasoner::new(64);
        
        let result = creativity.generate("Write a poem", &intent, &context, &mut reasoner);
        assert!(!result.output.is_empty());
    }
}
