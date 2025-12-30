//! Memory Module - Three-Layer Adaptive Memory Architecture
//!
//! This module implements the complete memory architecture:
//! 1. **Semantic Memory (Lance)** - "What does this mean?"
//! 2. **Learning Memory (SQLite)** - "What went wrong, and what did I learn?"
//! 3. **Control & State (SQLite)** - "How should I respond?"
//!
//! Core modules:
//! - episodic.rs: Stores past experiences and training attempts
//! - semantic.rs: Stores general knowledge and facts  
//! - semantic_store.rs: Persistent verified thought storage
//! - embeddings.rs: Converts text to vector space
//! - adaptive_memory.rs: Three-layer adaptive memory system

pub mod episodic;
pub mod semantic;
pub mod semantic_store;
pub mod embeddings;
pub mod concept_compression;
pub mod input_embeddings;
pub mod adaptive_memory;

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

// Re-export adaptive memory types (Three-Layer Architecture)
pub use adaptive_memory::{
    // Semantic Memory Layer
    SemanticUnit, SemanticUnitType, SemanticDomain, SemanticSource,
    // Learning Memory Layer
    ErrorLogEntry, ErrorType, PatternConfidence, PatternType,
    // Control & State Layer
    UserState, UserStyle, ExpertiseLevel, DecoderControl,
    // Unified Store
    AdaptiveMemoryStore, AdaptiveMemoryStats,
};
