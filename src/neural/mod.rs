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
mod universal_network;
mod memory_augmented;
mod policy_gradient;
mod creative_latent;
mod meta_learning;
mod advanced_integration;
mod self_discovery;
mod neural_reasoning_engine;

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
pub use universal_network::{
    UniversalExpertNetwork, UniversalNetworkConfig, UniversalNetworkOutput,
    UniversalNetworkLoss, SolveBranch, VerificationBranch, ExplanationBranch,
};
pub use memory_augmented::{
    MemoryAugmentedNetwork, MemoryBank, MemoryEntry, MemoryStats,
};
pub use policy_gradient::{
    PolicyNetwork, ActorCritic, PolicyGradientTrainer, RewardFunction,
    TrainingMetrics as PolicyTrainingMetrics,
};
pub use creative_latent::{
    CreativeExplorationController, NoiseInjector, TemperatureSampler,
    DiversityPromoter, NoveltySearch, BehaviorDescriptor,
    ExplorationMode, SamplingMode, NoiseSchedule, TemperatureSchedule,
};
pub use meta_learning::{
    MetaLearningController, MAML, LearnedOptimizer, AdaptiveLearningRate,
    CurriculumLearning, Task, DataSet, MetaTrainMetrics, OptimizationMode,
};
pub use advanced_integration::{
    AdvancedALENSystem, AdvancedALENConfig, AdvancedForwardResult,
    AdvancedTrainingMetrics, SystemStats, MathProblemSolver, MathSolution,
    CodeGenerationSystem, CodeSolution, AudienceLevel, ProgrammingLanguage,
};
pub use self_discovery::{
    SelfDiscoveryLoop, KnowledgeEncoder, TransformationBank, TransformationType,
    TransformationOperator, ConsistencyVerifier, KnowledgeIntegrator,
    ExplanationGenerator, ExplanationLevel, UncertaintyEstimator,
    DiscoveryResult, DiscoveryStats,
};
pub use neural_reasoning_engine::{
    NeuralReasoningEngine, NeuralReasoningStep, NeuralReasoningTrace,
    VerificationResult as NeuralVerificationResult, EngineStats,
};
