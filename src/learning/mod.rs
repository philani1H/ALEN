//! Learning Module
//!
//! Implements the training and learning mechanisms:
//! - feedback_loop.rs: Verification-first training loop
//! - verified.rs: Comprehensive verified learning system
//! - epistemic_reward.rs: Anti-hallucination reward function
//! - active_learning.rs: Human-like learning through active recall and reasoning

pub mod feedback_loop;
pub mod verified;
pub mod epistemic_reward;
pub mod active_learning;
pub mod self_learning;
pub mod verification_loop;
pub mod meta_optimizer;

// Re-export main types
pub use feedback_loop::{
    FeedbackLoop, LearningConfig, TrainingResult, InferenceResult,
    BatchTrainingResult, TrainingSession, SessionStats,
};

pub use verified::{
    VerifiedLearner, VerificationResult, VerificationThresholds,
    VerificationChecks, VerifiedLearningSession,
    TrainingCurriculum, CurriculumPhase,
};

pub use epistemic_reward::{
    EpistemicRewardCalculator, EpistemicReward, RewardWeights,
    EpistemicOperatorStats, VerificationResult as EpistemicVerificationResult,
};

pub use active_learning::{
    ActiveLearningSystem,
    RecallChallenge, ChallengeType,
    ConversationContext, EmotionalState, UnderlyingNeed,
    DerivedKnowledge, ReasoningType,
    ActiveRecallEngine, ContextInferenceEngine, ReasoningEngine as ActiveReasoningEngine,
};

pub use self_learning::{
    SelfLearningSystem, Pattern, PatternType, Domain,
    ConversationSegment, PatternCandidate, AggregatedPattern,
    PatternExtractor, CrossUserAggregator, PrivacyFilter,
    ConfidenceUpdater, KnowledgeBase, KnowledgeStats,
};
