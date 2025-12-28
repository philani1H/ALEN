//! Neural Network Module for ALEN
//!
//! Implements real neural network components:
//! - Transformer-based embeddings
//! - Learned reasoning operators
//! - Backpropagation training
//! - GPU-ready tensor operations

mod tensor;
mod layers;
mod transformer;
mod learned_operators;
mod trainer;

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
    TrainingBatch, TrainingMetrics,
};
