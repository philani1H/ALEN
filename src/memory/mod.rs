//! Memory Module
//!
//! This module contains:
//! - episodic.rs: Stores past experiences and training attempts
//! - semantic.rs: Stores general knowledge and facts  
//! - embeddings.rs: Converts text to vector space

pub mod episodic;
pub mod semantic;
pub mod embeddings;

// Re-export main types
pub use episodic::{EpisodicMemory, Episode, EpisodeStatistics};
pub use semantic::{SemanticMemory, SemanticFact, SemanticStatistics};
pub use embeddings::{EmbeddingEngine, EmbeddingConfig, EmbeddingBatch};
