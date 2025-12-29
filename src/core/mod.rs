//! Core Module - The heart of the Deliberative Reasoning System
//!
//! This module contains:
//! - state.rs: Thought State Vectors (ψ)
//! - operators.rs: Reasoning Operators (T_i)
//! - evaluator.rs: Energy Function E(ψ)
//! - selector.rs: Selection logic (argmin)
//! - advanced_math.rs: Attention, Transformers, Neural Networks
//! - unified_cognition.rs: Unified HHP + Transformer + ALEN framework
//! - intent_extraction.rs: Prompt understanding I = (τ, θ, C)
//! - proof_system.rs: Bidirectional verification P ↔ S ↔ A
//! - cognitive_architecture.rs: Complete cognitive system

pub mod state;
pub mod operators;
pub mod evaluator;
pub mod selector;
pub mod advanced_math;
pub mod unified_cognition;
pub mod intent_extraction;
pub mod proof_system;
pub mod cognitive_architecture;

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

// Re-export unified cognition types
pub use unified_cognition::{
    InputSequence, SequenceMetadata, ContextType,
    ContextEncoder, EncoderType, ThoughtVector,
    MultiPerspectiveEncoder, ConsensusResult, PerspectiveResult,
    SummarizationObjective, SummarizationLoss,
    BidirectionalVerifier, VerificationResult,
    CognitiveEnergy, EnergyBreakdown, Constraint, ConstraintType,
    AudienceModel,
    UnifiedObjective, ObjectiveResult, OptimizationResult,
    AttentionEncoder, CompressionEncoder,
};

// Re-export intent extraction types
pub use intent_extraction::{
    IntentState, TaskType, TaskVector,
    TargetVariable, TargetType,
    ConstraintSet, FormatConstraints, OutputFormat,
    LengthConstraints, LengthCategory,
    StyleConstraints, ContentConstraints,
    IntentExtractor, ResponseEnergy,
};

// Re-export proof system types
pub use proof_system::{
    Problem as ProofProblem, Answer, SolutionPath, ReasoningStep, Transformation,
    KnowledgeBase, Axiom, InferenceRule, VerifiedFact, CachedProof, BackwardCheck,
    ProofGraph, ProofNode, ProofNodeType, ProofEdge, EdgeDirection,
    ProofEngine, ProofEnergyWeights, ProofResult, ProofEnergyBreakdown,
    HybridReasoner, NeuralConfidence, HybridResult, ReasoningMode,
    ProofBenchmark, BenchmarkResult,
    ProblemDomain, ProblemStructure, NodeType,
};

// Re-export cognitive architecture types
pub use cognitive_architecture::{
    ContextState, Exchange, UserStyle, ConversationTone,
    SmallAttention,
    VerificationGate, GateResult, GateDecision,
    UnderstandingTest, UnderstandingResult, UnderstandingBreakdown,
    ConstrainedCreativity, CreativeResult,
    CognitiveSystem, CognitiveResponse, ResponseType,
};
