//! Memory Module
//!
//! This module contains:
//! - episodic.rs: Stores past experiences and training attempts
//! - semantic.rs: Stores general knowledge and facts  
//! - semantic_store.rs: Persistent verified thought storage
//! - embeddings.rs: Converts text to vector space

pub mod episodic;
pub mod semantic;
pub mod semantic_store;
pub mod embeddings;
pub mod concept_compression;
pub mod input_embeddings;

// Re-export main types
pub use episodic::{EpisodicMemory, Episode, EpisodeStatistics};
pub use semantic::{SemanticMemory, SemanticFact, SemanticStatistics};
pub use semantic_store::{SemanticStore, SemanticEntry, SemanticStats};
pub use embeddings::{EmbeddingEngine, EmbeddingConfig, EmbeddingBatch};
pub use concept_compression::{
    Concept, ConceptType, ProofSkeleton,
    MemoryDecay, ConceptExtractor, CompressionManager,
    EpisodeData, CompressionResult, CompressionStats,
    ProtectedKnowledge,
};
