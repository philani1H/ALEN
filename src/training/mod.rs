//! Training Pipeline for ALEN
//!
//! Comprehensive training system that:
//! - Loads training data from JSON files
//! - Tokenizes using learned BPE
//! - Trains the transformer encoder
//! - Trains the neural operators
//! - Implements verified learning
//! - Stores learned knowledge in semantic memory

mod data_loader;
mod tokenizer;
mod pipeline;

pub use data_loader::{
    TrainingData, TrainingExample, DataLoader, DataCategory,
    CategoryData, SubcategoryExample,
};
pub use tokenizer::{Tokenizer, TokenizerConfig, TokenizedBatch};
pub use pipeline::{TrainingPipeline, PipelineConfig, TrainingStats};
