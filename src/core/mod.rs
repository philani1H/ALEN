//! Core Module - The heart of the Deliberative Reasoning System
//!
//! This module contains:
//! - state.rs: Thought State Vectors (ψ)
//! - operators.rs: Reasoning Operators (T_i)
//! - evaluator.rs: Energy Function E(ψ)
//! - selector.rs: Selection logic (argmin)
//! - advanced_math.rs: Attention, Transformers, Neural Networks

pub mod state;
pub mod operators;
pub mod evaluator;
pub mod selector;
pub mod advanced_math;

// Re-export main types for convenience
pub use state::{ThoughtState, BiasVector, Problem, ThoughtMetadata};
pub use operators::{ReasoningOperator, OperatorType, OperatorManager, OperatorStats};
pub use evaluator::{Evaluator, EnergyWeights, EnergyResult, TrainingEvaluation, RankedCandidate, BackwardCheck};
pub use selector::{Selector, SelectionStrategy, SelectionResult, SelectorBuilder};

// Re-export advanced math types
pub use advanced_math::{
    Activation, DenseLayer, LayerNorm, AttentionHead, MultiHeadAttention,
    FeedForward, TransformerLayer, PositionalEncoding, InfoTheory, MatrixOps,
    AdamOptimizer, LRScheduler,
};
