//! Neural Network Module for ALEN
//!
//! Implements real neural network components:
//! - Transformer-based embeddings
//! - Learned reasoning operators
//! - Backpropagation training
//! - GPU-ready tensor operations
//! - Complete ALEN neural architecture with verification

mod tensor;
mod layers;
mod transformer;
mod learned_operators;
mod trainer;
mod alen_network;
mod integration;

pub use tensor::{Tensor, TensorShape, Device};
pub use layers::{Linear, LayerNorm, Dropout, Embedding, Conv1D};
pub use transformer::{
    TransformerEncoder, TransformerConfig, AttentionBlock,
    PositionalEncoding, MultiHeadSelfAttention, FeedForwardNetwork,
};
pub use learned_operators::{
    NeuralOperator, NeuralOperatorBank, OperatorConfig,
    GatedOperator, AttentionOperator, ResidualOperator,
};
pub use trainer::{
    Trainer, TrainerConfig, Optimizer, Adam, SGD,
    LossFunction, CrossEntropyLoss, MSELoss, ContrastiveLoss,
    TrainingBatch, TrainingMetrics, LRScheduler,
};
pub use alen_network::{
    ALENNetwork, ALENConfig, ThoughtEncoder, NeuralReasoningOperator,
    ThoughtDecoder, ThoughtVerifier, ALENForwardResult, CandidateEvaluation,
};
pub use integration::{
    NeuralReasoningEngine, VerifiedTrainingResult, NeuralInferenceResult,
    OperatorStatistics,
};
