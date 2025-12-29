//! ALEN - Deliberative Reasoning AI System
//!
//! A complete AI system based on mathematical deliberative reasoning with:
//! - **Multimodal Learning**: Images, Video, Audio understanding
//! - **Advanced Math**: Attention mechanisms, Transformers, Neural networks
//! - **Verified Learning**: Only commits verified solutions to memory
//!
//! # Architecture
//!
//! ## Mathematical Foundation
//!
//! - **Thought State Vectors (|ψ⟩)**: Thoughts represented as normalized vectors in ℝⁿ
//! - **Reasoning Operators (Tᵢ)**: Transform thoughts via matrix operations |ψᵢ⟩ = Tᵢ|ψ⟩
//! - **Energy Function**: E(ψ) = αC(ψ) + βR(ψ) + γU(ψ) where C=constraints, R=risk, U=uncertainty
//! - **Selection**: ψ* = argminᵢ E(ψᵢ) - minimum energy principle
//! - **Backward Inference**: T⁻¹ψ* ≈ ψ - verifies reasoning path consistency
//! - **Learning Rule**: wᵢ ← wᵢ + η(reward - E(ψᵢ)) - operator weight updates
//!
//! # Example
//!
//! ```rust,ignore
//! use deliberative_ai::{ReasoningEngine, EngineConfig, Problem};
//!
//! let config = EngineConfig::default();
//! let mut engine = ReasoningEngine::new(config).unwrap();
//!
//! // Train on a problem
//! let problem = Problem::training("What is 2+2?", "4", 128);
//! let result = engine.train(&problem);
//! println!("Training success: {}", result.success);
//!
//! // Perform inference
//! let problem = Problem::new("What is 3+3?", 128);
//! let result = engine.infer(&problem);
//! println!("Confidence: {}", result.confidence);
//! ```

pub mod core;
pub mod memory;
pub mod learning;
pub mod control;
pub mod api;
pub mod multimodal;
pub mod generation;
pub mod knowledge;
pub mod neural;
pub mod reasoning;
pub mod storage;
pub mod math;
pub mod training;
pub mod confidence;
pub mod verification;
pub mod explanation;

// Re-export commonly used types at the crate level
pub use core::{
    // State and operators
    ThoughtState, BiasVector, Problem, ThoughtMetadata,
    ReasoningOperator, OperatorType, OperatorManager, OperatorStats,
    Evaluator, EnergyWeights, EnergyResult, TrainingEvaluation, RankedCandidate, BackwardCheck,
    Selector, SelectionStrategy, SelectionResult, SelectorBuilder,
    
    // Advanced math
    Activation, DenseLayer, AttentionHead, MultiHeadAttention,
    FeedForward, TransformerLayer, InfoTheory, MatrixOps,
    AdamOptimizer,
};

// Re-export multimodal types
pub use multimodal::{
    Modality, ImageData, ImageEncoder, VideoData, VideoEncoder,
    AudioData, AudioEncoder, MultimodalInput, MultimodalEncoder,
    CrossModalAttention,
};

// Re-export generation types
pub use generation::{
    TextGenerator, ImageGenerator, ContentGenerator,
    GenerationConfig, GeneratedContent,
    VideoGenerator, VideoGenConfig, GeneratedVideo, MotionType,
    // Dynamic vocabulary (learns from data, no hardcoded words)
    DynamicVocabulary, DynamicTextGenerator, VocabularyBuilder,
    SpecialTokens, TokenInfo,
    // BPE Tokenizer (production-grade subword tokenization)
    BPETokenizer, BPETrainer, BPEWithEmbeddings, BPESpecialTokens,
    // Semantic decoder (uses learned memory)
    SemanticDecoder,
};

pub use memory::{
    EpisodicMemory, Episode, EpisodeStatistics,
    SemanticMemory, SemanticFact, SemanticStatistics,
    EmbeddingEngine, EmbeddingConfig,
};

pub use learning::{
    FeedbackLoop, LearningConfig, TrainingResult, InferenceResult,
    BatchTrainingResult, TrainingSession, SessionStats,
    VerifiedLearner, VerificationResult, VerificationThresholds,
    VerificationChecks, VerifiedLearningSession,
    TrainingCurriculum, CurriculumPhase,
};

pub use control::{BiasController, ControlStateSummary, MetaState};

pub use knowledge::{KnowledgeBase, KnowledgeItem, KnowledgeCategory};

// Re-export reasoning types
pub use reasoning::{
    MathSolver, MathExpression, MathResult,
    ReasoningChain, ReasoningStep, ChainResult,
    SymbolicReasoner, Symbol, SymbolicExpression,
    LogicalInference, Premise, Conclusion, InferenceRule,
};

// Re-export neural network types
pub use neural::{
    Tensor, TensorShape, Device,
    Linear, LayerNorm, Dropout, Embedding, Conv1D,
    TransformerEncoder, TransformerConfig, AttentionBlock,
    MultiHeadSelfAttention, FeedForwardNetwork, PositionalEncoding,
    NeuralOperator, NeuralOperatorBank, OperatorConfig,
    GatedOperator, AttentionOperator, ResidualOperator,
    Trainer, TrainerConfig, Adam, SGD,
    MSELoss, CrossEntropyLoss, ContrastiveLoss,
    TrainingBatch, TrainingMetrics, LRScheduler,
};

pub use api::{
    ReasoningEngine, EngineConfig, EngineStatistics,
    AppState, create_router,
    TrainRequest, TrainResponse,
    InferRequest, InferResponse,
    BatchTrainRequest, BatchTrainResponse,
};

pub use storage::{StorageConfig, StorageStats};

// Re-export training types
pub use training::{
    TrainingData, TrainingExample, DataLoader, DataCategory,
    Tokenizer, TokenizerConfig, TokenizedBatch,
    TrainingPipeline, PipelineConfig, TrainingStats,
};

/// Version of the library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Name of the system
pub const SYSTEM_NAME: &str = "ALEN";

/// Default dimension for thought vectors
pub const DEFAULT_DIMENSION: usize = 128;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_training_cycle() {
        let config = EngineConfig::default();
        let mut engine = ReasoningEngine::new(config).expect("Failed to create engine");

        let problem = Problem::training(
            "The capital of France",
            "Paris",
            128
        );

        let result = engine.train(&problem);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_inference_after_training() {
        let config = EngineConfig::default();
        let mut engine = ReasoningEngine::new(config).expect("Failed to create engine");

        let problem = Problem::new("What is machine learning?", 128);
        let result = engine.infer(&problem);
        assert!(result.confidence >= 0.0);
    }

    #[test]
    fn test_multimodal_image() {
        let encoder = ImageEncoder::new(64);
        let img = ImageData::random(32, 32, 3);
        let embedding = encoder.encode(&img);
        assert_eq!(embedding.len(), 64);
    }

    #[test]
    fn test_semantic_memory_integration() {
        let memory = SemanticMemory::new(":memory:", 64).expect("Failed to create memory");
        let fact = SemanticFact {
            id: uuid::Uuid::new_v4().to_string(),
            concept: "test".to_string(),
            content: "This is a test fact".to_string(),
            embedding: vec![0.1; 64],
            category: None,
            source: None,
            confidence: 1.0,
            reinforcement_count: 0,
            last_accessed: chrono::Utc::now(),
            related_concepts: vec![],
        };
        memory.store(&fact).expect("Failed to store fact");
        
        let results = memory.search_by_concept("test", 10).expect("Failed to search");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_statistics_collection() {
        let config = EngineConfig::default();
        let engine = ReasoningEngine::new(config).expect("Failed to create engine");
        let stats = engine.get_statistics();
        assert!(stats.operator_stats.len() >= 8);
    }

    #[test]
    fn test_text_generator() {
        let config = GenerationConfig::default();
        let generator = TextGenerator::new(config, 128);
        let thought = ThoughtState::random(128);
        let text = generator.generate(&thought, 50);
        assert!(!text.is_empty());
    }
}