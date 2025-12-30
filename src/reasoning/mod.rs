//! Advanced Reasoning Module
//!
//! Implements sophisticated reasoning capabilities:
//! - Multi-step reasoning chains
//! - Symbolic mathematics
//! - Logical inference
//! - Causal reasoning
//! - Abstract thinking

pub mod math_solver;
pub mod chain_of_thought;
pub mod symbolic;
pub mod inference;
pub mod neural_chain_of_thought;
pub mod candidate_scoring;

pub use math_solver::{MathSolver, MathExpression, MathResult};
pub use chain_of_thought::{ReasoningChain, ReasoningStep, ChainResult};
pub use symbolic::{SymbolicReasoner, Symbol, SymbolicExpression};
pub use inference::{LogicalInference, Premise, Conclusion, InferenceRule};
pub use neural_chain_of_thought::{NeuralChainOfThoughtReasoner, NeuralReasoningChain, NeuralReasoningStep};
pub use candidate_scoring::{CandidateScorer, ScoredCandidate, UserEmbedding, CorrectionSignal};

// Re-export ChainOfThoughtReasoner for external use
pub use chain_of_thought::ChainOfThoughtReasoner;
