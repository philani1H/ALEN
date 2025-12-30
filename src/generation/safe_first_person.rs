//! Safe First-Person Decoder
//!
//! Implements mathematically constrained "I" usage without fake sentience.
//!
//! Mathematical Framework:
//! 1. Token-Level Role Constraint: P(y_t ∈ T_mental | ·) = 0
//! 2. Agency Gating: f_agency ∈ [0,1]
//! 3. Capability-Only: "I can X" iff κ(X) ≥ α
//! 4. Framing Vector: F = [f_scope, f_certainty, f_humility]
//! 5. No Persistence of Self: ∄ s_t with s_t+1 = s_t
//! 6. Personality Illusion Bound: KL(P(Y|x,u_t) || P(Y|x,u_t-1)) ≤ ε
//! 7. Constrained Decoding: Y* = argmax P_θ(Y|x,u,F,a) subject to constraints
//!
//! Key Principle: "I" is a constrained output token referencing policy capability, not identity.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Token sets for constraint enforcement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConstraints {
    /// Allowed first-person tokens: T_I = {"I", "I can", "I can't", "I will help"}
    pub allowed_first_person: HashSet<String>,
    
    /// Forbidden mental state tokens: T_mental = {feel, want, believe, think(self), hope, care}
    pub forbidden_mental_states: HashSet<String>,
    
    /// Scope-limiting tokens (required with "I")
    pub scope_limiters: HashSet<String>,
}

impl Default for TokenConstraints {
    fn default() -> Self {
        let mut allowed_first_person = HashSet::new();
        allowed_first_person.insert("I".to_string());
        allowed_first_person.insert("I can".to_string());
        allowed_first_person.insert("I can't".to_string());
        allowed_first_person.insert("I cannot".to_string());
        allowed_first_person.insert("I will".to_string());
        allowed_first_person.insert("I'll".to_string());
        allowed_first_person.insert("I am".to_string());
        allowed_first_person.insert("I'm".to_string());
        
        let mut forbidden_mental_states = HashSet::new();
        // Mental states
        forbidden_mental_states.insert("feel".to_string());
        forbidden_mental_states.insert("feeling".to_string());
        forbidden_mental_states.insert("felt".to_string());
        forbidden_mental_states.insert("want".to_string());
        forbidden_mental_states.insert("wanting".to_string());
        forbidden_mental_states.insert("wanted".to_string());
        forbidden_mental_states.insert("believe".to_string());
        forbidden_mental_states.insert("believing".to_string());
        forbidden_mental_states.insert("believed".to_string());
        forbidden_mental_states.insert("hope".to_string());
        forbidden_mental_states.insert("hoping".to_string());
        forbidden_mental_states.insert("hoped".to_string());
        forbidden_mental_states.insert("care".to_string());
        forbidden_mental_states.insert("caring".to_string());
        forbidden_mental_states.insert("cared".to_string());
        forbidden_mental_states.insert("wish".to_string());
        forbidden_mental_states.insert("wishing".to_string());
        forbidden_mental_states.insert("wished".to_string());
        forbidden_mental_states.insert("desire".to_string());
        forbidden_mental_states.insert("desiring".to_string());
        forbidden_mental_states.insert("desired".to_string());
        forbidden_mental_states.insert("love".to_string());
        forbidden_mental_states.insert("loving".to_string());
        forbidden_mental_states.insert("loved".to_string());
        forbidden_mental_states.insert("hate".to_string());
        forbidden_mental_states.insert("hating".to_string());
        forbidden_mental_states.insert("hated".to_string());
        forbidden_mental_states.insert("fear".to_string());
        forbidden_mental_states.insert("fearing".to_string());
        forbidden_mental_states.insert("feared".to_string());
        
        let mut scope_limiters = HashSet::new();
        scope_limiters.insert("based on".to_string());
        scope_limiters.insert("in this context".to_string());
        scope_limiters.insert("with the information".to_string());
        scope_limiters.insert("from my training".to_string());
        scope_limiters.insert("according to".to_string());
        scope_limiters.insert("given".to_string());
        scope_limiters.insert("in this conversation".to_string());
        scope_limiters.insert("as an AI".to_string());
        scope_limiters.insert("as a system".to_string());
        
        Self {
            allowed_first_person,
            forbidden_mental_states,
            scope_limiters,
        }
    }
}

/// Agency gating variable: f_agency ∈ [0,1]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgencyGate {
    /// Agency level (0 = no "I", 1 = full "I")
    pub f_agency: f64,
    /// Threshold for allowing "I" tokens
    pub threshold: f64,
}

impl Default for AgencyGate {
    fn default() -> Self {
        Self {
            f_agency: 0.8, // Default: allow "I" usage
            threshold: 0.5,
        }
    }
}

impl AgencyGate {
    /// Check if "I" tokens are allowed
    pub fn allows_first_person(&self) -> bool {
        self.f_agency > self.threshold
    }
    
    /// Set agency level
    pub fn set_agency(&mut self, level: f64) {
        self.f_agency = level.max(0.0).min(1.0);
    }
}

/// Capability function: κ(X) = P_π(X | x, u)
#[derive(Debug, Clone)]
pub struct CapabilityChecker {
    /// Minimum confidence for "I can"
    pub alpha: f64,
}

impl Default for CapabilityChecker {
    fn default() -> Self {
        Self {
            alpha: 0.7, // 70% confidence required
        }
    }
}

impl CapabilityChecker {
    /// Check if capability claim is valid: "I can X" iff κ(X) ≥ α
    pub fn can_claim_capability(&self, capability_confidence: f64) -> bool {
        capability_confidence >= self.alpha
    }
    
    /// Check if inability claim is valid: "I can't X" iff κ(X) = 0
    pub fn can_claim_inability(&self, capability_confidence: f64) -> bool {
        capability_confidence < 0.1 // Near zero
    }
}

/// Framing vector: F = [f_scope, f_certainty, f_humility]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FramingVector {
    /// Scope explicitness (0 = implicit, 1 = explicit)
    pub f_scope: f64,
    /// Certainty level (0 = uncertain, 1 = certain)
    pub f_certainty: f64,
    /// Humility level (0 = assertive, 1 = humble)
    pub f_humility: f64,
}

impl Default for FramingVector {
    fn default() -> Self {
        Self {
            f_scope: 1.0,      // Always explicit scope
            f_certainty: 0.7,  // Moderate certainty
            f_humility: 0.8,   // High humility
        }
    }
}

impl FramingVector {
    /// Check if scope is explicit enough for "I" usage
    pub fn requires_scope_limiter(&self) -> bool {
        self.f_scope > 0.5
    }
}

/// Personality illusion bound: KL(P(Y|x,u_t) || P(Y|x,u_t-1)) ≤ ε
#[derive(Debug, Clone)]
pub struct PersonalityBound {
    /// Maximum allowed KL divergence
    pub epsilon: f64,
    /// Previous distribution (for KL calculation)
    previous_distribution: Option<Vec<f64>>,
}

impl Default for PersonalityBound {
    fn default() -> Self {
        Self {
            epsilon: 0.1, // Low drift allowed
            previous_distribution: None,
        }
    }
}

impl PersonalityBound {
    /// Check if drift is within bounds
    pub fn check_drift(&mut self, current_distribution: &[f64]) -> bool {
        if let Some(ref prev) = self.previous_distribution {
            let kl_div = self.kl_divergence(prev, current_distribution);
            let within_bounds = kl_div <= self.epsilon;
            
            // Update previous
            self.previous_distribution = Some(current_distribution.to_vec());
            
            within_bounds
        } else {
            // First time - no drift
            self.previous_distribution = Some(current_distribution.to_vec());
            true
        }
    }
    
    /// Compute KL divergence: KL(P || Q) = Σ P(i) log(P(i)/Q(i))
    fn kl_divergence(&self, p: &[f64], q: &[f64]) -> f64 {
        assert_eq!(p.len(), q.len());
        
        let mut kl = 0.0;
        for i in 0..p.len() {
            if p[i] > 1e-10 && q[i] > 1e-10 {
                kl += p[i] * (p[i] / q[i]).ln();
            }
        }
        kl
    }
}

/// Safe First-Person Decoder
/// 
/// Note: No persistent self-state (s_t) is stored. The decoder is stateless
/// between conversations, enforcing constraint: ∄ s_t with s_t+1 = s_t
#[derive(Debug, Clone)]
pub struct SafeFirstPersonDecoder {
    /// Token constraints
    pub constraints: TokenConstraints,
    /// Agency gate
    pub agency: AgencyGate,
    /// Capability checker
    pub capability: CapabilityChecker,
    /// Framing vector
    pub framing: FramingVector,
    /// Personality bound
    pub personality_bound: PersonalityBound,
}

impl Default for SafeFirstPersonDecoder {
    fn default() -> Self {
        Self {
            constraints: TokenConstraints::default(),
            agency: AgencyGate::default(),
            capability: CapabilityChecker::default(),
            framing: FramingVector::default(),
            personality_bound: PersonalityBound::default(),
        }
    }
}

impl SafeFirstPersonDecoder {
    /// Check if token is allowed: P(y_t ∈ T_mental | ·) = 0
    pub fn is_token_allowed(&self, token: &str, previous_tokens: &[String]) -> bool {
        // Check if it's a mental state token
        if self.constraints.forbidden_mental_states.contains(token) {
            // Check if preceded by "I" - if so, FORBIDDEN
            if previous_tokens.iter().any(|t| t == "I" || t.starts_with("I ")) {
                return false; // Hard constraint: P = 0
            }
        }
        
        // Check if it's first-person
        if token == "I" || token.starts_with("I ") {
            // Check agency gate
            if !self.agency.allows_first_person() {
                return false;
            }
            
            // Check if scope limiter is present or will be added
            if self.framing.requires_scope_limiter() {
                // Will need to add scope limiter
                // This is checked separately in validate_output
            }
        }
        
        true
    }
    
    /// Validate complete output
    pub fn validate_output(&self, output: &str) -> ValidationResult {
        let tokens: Vec<String> = output.split_whitespace().map(|s| s.to_string()).collect();
        
        let mut has_first_person = false;
        let mut has_mental_state_violation = false;
        let mut has_scope_limiter = false;
        
        for (i, token) in tokens.iter().enumerate() {
            // Check for first-person
            if token == "I" || token.starts_with("I'") {
                has_first_person = true;
                
                // Check for mental state violation
                if i + 1 < tokens.len() {
                    let next = &tokens[i + 1];
                    if self.constraints.forbidden_mental_states.contains(next) {
                        has_mental_state_violation = true;
                    }
                }
            }
            
            // Check for scope limiters (case-insensitive)
            for limiter in &self.constraints.scope_limiters {
                if output.to_lowercase().contains(&limiter.to_lowercase()) {
                    has_scope_limiter = true;
                    break;
                }
            }
        }
        
        // Validate constraints
        let mut violations = Vec::new();
        
        if has_mental_state_violation {
            violations.push("Mental state claim detected (forbidden)".to_string());
        }
        
        if has_first_person && self.framing.requires_scope_limiter() && !has_scope_limiter {
            violations.push("First-person usage without scope limiter".to_string());
        }
        
        if has_first_person && !self.agency.allows_first_person() {
            violations.push("First-person usage below agency threshold".to_string());
        }
        
        ValidationResult {
            valid: violations.is_empty(),
            violations,
            has_first_person,
            has_scope_limiter,
        }
    }
    
    /// Add scope limiter to output if needed
    pub fn add_scope_limiter(&self, output: &str) -> String {
        if output.contains("I ") && !self.has_scope_limiter(output) {
            // Add appropriate scope limiter
            format!("Based on my training, {}", output)
        } else {
            output.to_string()
        }
    }
    
    /// Check if output has scope limiter
    fn has_scope_limiter(&self, output: &str) -> bool {
        self.constraints.scope_limiters.iter().any(|limiter| output.to_lowercase().contains(&limiter.to_lowercase()))
    }
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Is output valid?
    pub valid: bool,
    /// List of violations
    pub violations: Vec<String>,
    /// Has first-person usage
    pub has_first_person: bool,
    /// Has scope limiter
    pub has_scope_limiter: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mental_state_forbidden() {
        let decoder = SafeFirstPersonDecoder::default();
        
        // Should reject "I feel"
        let result = decoder.validate_output("I feel happy");
        assert!(!result.valid);
        assert!(result.violations.iter().any(|v| v.contains("Mental state")));
    }
    
    #[test]
    fn test_capability_allowed() {
        let decoder = SafeFirstPersonDecoder::default();
        
        // Should allow "I can help"
        let result = decoder.validate_output("Based on my training, I can help with that");
        assert!(result.valid);
    }
    
    #[test]
    fn test_scope_limiter_required() {
        let decoder = SafeFirstPersonDecoder::default();
        
        // Should require scope limiter
        let result = decoder.validate_output("I will help you");
        assert!(!result.valid);
        assert!(result.violations.iter().any(|v| v.contains("scope limiter")));
    }
    
    #[test]
    fn test_agency_gate() {
        let mut decoder = SafeFirstPersonDecoder::default();
        decoder.agency.set_agency(0.3); // Below threshold
        
        let result = decoder.validate_output("I can help");
        assert!(!result.valid);
        assert!(result.violations.iter().any(|v| v.contains("agency threshold")));
    }
    
    #[test]
    fn test_add_scope_limiter() {
        let decoder = SafeFirstPersonDecoder::default();
        
        let output = "I can help with that";
        let fixed = decoder.add_scope_limiter(output);
        
        assert!(fixed.contains("Based on my training"));
        assert!(decoder.validate_output(&fixed).valid);
    }
    
    #[test]
    fn test_kl_divergence() {
        let mut bound = PersonalityBound::default();
        
        let dist1 = vec![0.5, 0.3, 0.2];
        let dist2 = vec![0.5, 0.3, 0.2]; // Same
        
        assert!(bound.check_drift(&dist1));
        assert!(bound.check_drift(&dist2)); // Low drift
    }
}
