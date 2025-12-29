//! Input Embeddings - Semantic Space for Similarity
//!
//! Separates input space from thought space:
//! - Input space: Used for similarity, retrieval, matching
//! - Thought space: Used for reasoning, verification, solving
//!
//! Key Principle: Similarity happens BEFORE thinking, not after.
//!
//! Mathematical Foundation:
//! - Input embedding: e_x = E_input(x) ∈ ℝ^d
//! - Similarity: sim(x_i, x_j) = (e_i · e_j) / (|e_i||e_j|)
//! - Thought vector: ψ = T(e_x) (separate space)

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use serde::{Deserialize, Serialize};

// ============================================================================
// PART 1: INPUT EMBEDDING ENGINE
// ============================================================================

/// Generates embeddings for input text (semantic space)
/// This is SEPARATE from thought vectors (reasoning space)
#[derive(Debug, Clone)]
pub struct InputEmbedder {
    pub dimension: usize,
    /// Vocabulary for token-based embedding
    pub vocab: Vec<String>,
}

impl InputEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vocab: Self::build_vocab(),
        }
    }

    /// Embed input text into semantic space
    /// This is used for similarity comparison ONLY
    pub fn embed(&self, text: &str) -> Vec<f64> {
        // Normalize text
        let normalized = text.to_lowercase();
        let tokens: Vec<&str> = normalized.split_whitespace().collect();

        // Create embedding vector
        let mut embedding = vec![0.0; self.dimension];

        // Token-based embedding with position weighting
        for (pos, token) in tokens.iter().enumerate() {
            let token_vec = self.token_embedding(token);
            let position_weight = 1.0 / (1.0 + pos as f64 * 0.1); // Decay with position

            for i in 0..self.dimension {
                embedding[i] += token_vec[i] * position_weight;
            }
        }

        // Normalize to unit vector
        self.normalize(&mut embedding);

        embedding
    }

    /// Generate embedding for a single token
    fn token_embedding(&self, token: &str) -> Vec<f64> {
        let mut embedding = vec![0.0; self.dimension];

        // Hash-based embedding (deterministic)
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        let hash = hasher.finish();

        // Distribute hash across dimensions
        for i in 0..self.dimension {
            let mut dim_hasher = DefaultHasher::new();
            (hash ^ (i as u64)).hash(&mut dim_hasher);
            let dim_hash = dim_hasher.finish();
            
            // Map to [-1, 1]
            embedding[i] = ((dim_hash % 1000) as f64 / 500.0) - 1.0;
        }

        embedding
    }

    /// Normalize vector to unit length
    fn normalize(&self, vec: &mut [f64]) {
        let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for val in vec.iter_mut() {
                *val /= norm;
            }
        }
    }

    /// Calculate cosine similarity between two embeddings
    pub fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        (dot / (norm_a * norm_b)).max(-1.0).min(1.0)
    }

    fn build_vocab() -> Vec<String> {
        // Basic vocabulary - in production, load from file
        vec![
            "what", "how", "why", "when", "where", "who",
            "is", "are", "was", "were", "be", "been",
            "do", "does", "did", "done",
            "can", "could", "should", "would", "will",
            "the", "a", "an", "this", "that", "these", "those",
        ].iter().map(|s| s.to_string()).collect()
    }
}

// ============================================================================
// PART 2: ENHANCED EPISODE WITH INPUT EMBEDDING
// ============================================================================

/// Episode with separate input embedding and thought vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedEpisode {
    pub id: String,
    pub problem_input: String,
    pub answer_output: String,
    
    /// INPUT EMBEDDING - for similarity search
    pub input_embedding: Vec<f64>,
    
    /// THOUGHT VECTOR - for reasoning/verification
    pub thought_vector: Vec<f64>,
    
    pub verified: bool,
    pub confidence_score: f64,
    pub energy: f64,
    pub operator_id: String,
    pub created_at: u64,
    pub usage_count: u32,
}

impl EnhancedEpisode {
    /// Create episode with proper separation of spaces
    pub fn new(
        problem_input: String,
        answer_output: String,
        input_embedding: Vec<f64>,
        thought_vector: Vec<f64>,
        verified: bool,
        confidence_score: f64,
        energy: f64,
        operator_id: String,
    ) -> Self {
        Self {
            id: format!("ep_{}", Self::current_timestamp()),
            problem_input,
            answer_output,
            input_embedding,
            thought_vector,
            verified,
            confidence_score,
            energy,
            operator_id,
            created_at: Self::current_timestamp(),
            usage_count: 0,
        }
    }

    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

// ============================================================================
// PART 3: SIMILARITY-BASED RETRIEVAL
// ============================================================================

/// Retrieves episodes based on INPUT similarity (not thought similarity)
pub struct SimilarityRetriever {
    embedder: InputEmbedder,
    min_similarity: f64,
}

impl SimilarityRetriever {
    pub fn new(dimension: usize) -> Self {
        Self {
            embedder: InputEmbedder::new(dimension),
            min_similarity: 0.3, // Configurable threshold
        }
    }

    /// Retrieve similar episodes based on INPUT embedding
    pub fn retrieve(
        &self,
        query: &str,
        episodes: &[EnhancedEpisode],
        limit: usize,
    ) -> Vec<(EnhancedEpisode, f64)> {
        // Embed query in INPUT space
        let query_embedding = self.embedder.embed(query);

        // Calculate similarities using INPUT embeddings
        let mut scored: Vec<(EnhancedEpisode, f64)> = episodes
            .iter()
            .map(|ep| {
                let sim = self.embedder.similarity(&query_embedding, &ep.input_embedding);
                (ep.clone(), sim)
            })
            .filter(|(_, sim)| *sim >= self.min_similarity)
            .collect();

        // Sort by similarity (highest first)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        scored.into_iter().take(limit).collect()
    }

    /// Set minimum similarity threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.min_similarity = threshold.max(0.0).min(1.0);
    }
}

// ============================================================================
// PART 4: SPACE SEPARATION VALIDATOR
// ============================================================================

/// Validates that input and thought spaces are properly separated
pub struct SpaceSeparationValidator;

impl SpaceSeparationValidator {
    /// Check if similarity is being computed in the correct space
    pub fn validate_similarity_usage(
        using_input_embedding: bool,
        purpose: &str,
    ) -> Result<(), String> {
        match purpose {
            "retrieval" | "similarity" | "matching" => {
                if !using_input_embedding {
                    return Err(
                        "ERROR: Similarity for retrieval must use INPUT embeddings, not thought vectors"
                            .to_string(),
                    );
                }
            }
            "verification" | "proof" | "reasoning" => {
                if using_input_embedding {
                    return Err(
                        "ERROR: Verification must use THOUGHT vectors, not input embeddings"
                            .to_string(),
                    );
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Diagnostic: Check if spaces are properly separated
    pub fn diagnose(
        input_embedding: &[f64],
        thought_vector: &[f64],
    ) -> SpaceDiagnostic {
        let embedder = InputEmbedder::new(input_embedding.len());
        let similarity = embedder.similarity(input_embedding, thought_vector);

        SpaceDiagnostic {
            input_dim: input_embedding.len(),
            thought_dim: thought_vector.len(),
            cross_space_similarity: similarity,
            properly_separated: similarity < 0.5, // Should be low if spaces are different
        }
    }
}

#[derive(Debug)]
pub struct SpaceDiagnostic {
    pub input_dim: usize,
    pub thought_dim: usize,
    pub cross_space_similarity: f64,
    pub properly_separated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_embedder() {
        let embedder = InputEmbedder::new(128);

        let text1 = "What is machine learning?";
        let text2 = "What is machine learning?";
        let text3 = "How does quantum computing work?";

        let emb1 = embedder.embed(text1);
        let emb2 = embedder.embed(text2);
        let emb3 = embedder.embed(text3);

        // Same text should have high similarity
        let sim_same = embedder.similarity(&emb1, &emb2);
        assert!(sim_same > 0.99);

        // Different text should have lower similarity
        let sim_diff = embedder.similarity(&emb1, &emb3);
        assert!(sim_diff < sim_same);
    }

    #[test]
    fn test_similarity_retrieval() {
        let retriever = SimilarityRetriever::new(128);
        let embedder = InputEmbedder::new(128);

        let episodes = vec![
            EnhancedEpisode::new(
                "What is AI?".to_string(),
                "AI is artificial intelligence".to_string(),
                embedder.embed("What is AI?"),
                vec![0.0; 128],
                true,
                0.9,
                0.1,
                "op1".to_string(),
            ),
            EnhancedEpisode::new(
                "How does ML work?".to_string(),
                "ML learns from data".to_string(),
                embedder.embed("How does ML work?"),
                vec![0.0; 128],
                true,
                0.85,
                0.15,
                "op2".to_string(),
            ),
        ];

        let results = retriever.retrieve("What is AI?", &episodes, 5);
        
        assert!(!results.is_empty());
        assert!(results[0].1 > 0.9); // High similarity for exact match
    }

    #[test]
    fn test_space_separation() {
        let input_emb = vec![1.0, 0.0, 0.0];
        let thought_vec = vec![0.0, 1.0, 0.0];

        let diagnostic = SpaceSeparationValidator::diagnose(&input_emb, &thought_vec);
        
        // Different spaces should have low cross-similarity
        assert!(diagnostic.cross_space_similarity < 0.5);
        assert!(diagnostic.properly_separated);
    }
}
