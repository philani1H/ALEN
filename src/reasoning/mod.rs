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
// TODO: Fix neural_chain_of_thought API compatibility
// pub mod neural_chain_of_thought;

pub use math_solver::{MathSolver, MathExpression, MathResult};
pub use chain_of_thought::{ReasoningChain, ReasoningStep, ChainResult};
pub use symbolic::{SymbolicReasoner, Symbol, SymbolicExpression};
pub use inference::{LogicalInference, Premise, Conclusion, InferenceRule};
// pub use neural_chain_of_thought::{NeuralChainOfThoughtReasoner, NeuralReasoningChain, NeuralReasoningStep};

// Re-export ChainOfThoughtReasoner for external use
pub use chain_of_thought::ChainOfThoughtReasoner;
