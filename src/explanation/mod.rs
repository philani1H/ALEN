pub mod cognitive_distance;
pub mod multimodal_generator;
pub mod effectiveness_tracker;
pub mod universal_expert;

pub use cognitive_distance::{
    CognitiveDistanceCalculator, CognitiveDistance, DistanceWeights,
    ComplexityAnalyzer, RelevanceScorer, ClarityAssessor,
};

pub use multimodal_generator::{
    MultiModalExplanationGenerator, CompleteExplanation,
    VisualExplanation, Analogy, ReasoningStep, Example,
};

pub use effectiveness_tracker::{
    TeachingEffectivenessTracker, UserFeedback, EffectivenessStats,
    ConceptEffectiveness, AudienceEffectiveness,
};

pub use universal_expert::{
    UniversalExpertSystem, UniversalExpertResponse, Solution,
};
