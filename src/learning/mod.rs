//! Learning Module
//!
//! Implements the training and learning mechanisms:
//! - feedback_loop.rs: Verification-first training loop
//! - verified.rs: Comprehensive verified learning system

pub mod feedback_loop;
pub mod verified;

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
