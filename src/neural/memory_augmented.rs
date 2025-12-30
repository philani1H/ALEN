//! Memory-Augmented Neural Network
//!
//! Integrates episodic memory retrieval into neural computation:
//!
//! Mathematical Foundation:
//! Memory: M = {(x_i, S_i, L_i)}, i = 1,...,N
//! Retrieval: m = ∑_i w_i · Embed(x_i, S_i, L_i)
//! Weights: w_i = softmax(Similarity(x, x_i))
//! Augmented Input: x̃ = concat(x, a, m)
//!
//! This enables transfer learning and pattern generalization.

use super::tensor::Tensor;
use super::layers::Linear;
use std::collections::VecDeque;

// ============================================================================
// PART 1: MEMORY ENTRY
// ============================================================================

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Problem embedding
    pub problem_embedding: Vec<f64>,
    
    /// Solution embedding
    pub solution_embedding: Vec<f64>,
    
    /// Explanation embedding
    pub explanation_embedding: Vec<f64>,
    
    /// Verification score
    pub verification_score: f64,
    
    /// Usage count
    pub usage_count: usize,
    
    /// Timestamp
    pub timestamp: u64,
}

impl MemoryEntry {
    pub fn new(
        problem_embedding: Vec<f64>,
        solution_embedding: Vec<f64>,
        explanation_embedding: Vec<f64>,
        verification_score: f64,
    ) -> Self {
        Self {
            problem_embedding,
            solution_embedding,
            explanation_embedding,
            verification_score,
            usage_count: 0,
            timestamp: Self::current_timestamp(),
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
// PART 2: MEMORY BANK
// ============================================================================

#[derive(Debug, Clone)]
pub struct MemoryBank {
    /// Stored memories
    memories: VecDeque<MemoryEntry>,
    
    /// Maximum memory size
    max_size: usize,
    
    /// Embedding dimension
    embedding_dim: usize,
    
    /// Memory projection layer
    memory_proj: Linear,
}

impl MemoryBank {
    pub fn new(max_size: usize, embedding_dim: usize, memory_dim: usize) -> Self {
        Self {
            memories: VecDeque::new(),
            max_size,
            embedding_dim,
            memory_proj: Linear::new(embedding_dim * 3, memory_dim, true),
        }
    }
    
    /// Store a new memory
    pub fn store(&mut self, entry: MemoryEntry) {
        if self.memories.len() >= self.max_size {
            // Remove oldest memory
            self.memories.pop_front();
        }
        self.memories.push_back(entry);
    }
    
    /// Retrieve memory for a given problem
    /// Returns weighted combination of similar memories
    pub fn retrieve(&mut self, problem_embedding: &[f64], top_k: usize) -> Vec<f64> {
        if self.memories.is_empty() {
            return vec![0.0; self.embedding_dim * 3];
        }
        
        // Compute similarities
        let mut similarities: Vec<(usize, f64)> = self.memories
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let sim = self.cosine_similarity(problem_embedding, &entry.problem_embedding);
                (i, sim)
            })
            .collect();
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top-k
        let top_k = top_k.min(similarities.len());
        let top_indices: Vec<usize> = similarities.iter().take(top_k).map(|(i, _)| *i).collect();
        let top_sims: Vec<f64> = similarities.iter().take(top_k).map(|(_, s)| *s).collect();
        
        // Compute softmax weights
        let weights = self.softmax(&top_sims);
        
        // Weighted combination
        let mut combined = vec![0.0; self.embedding_dim * 3];
        for (idx, weight) in top_indices.iter().zip(weights.iter()) {
            let entry = &mut self.memories[*idx];
            entry.usage_count += 1;
            
            // Concatenate embeddings
            let full_embedding: Vec<f64> = entry.problem_embedding.iter()
                .chain(entry.solution_embedding.iter())
                .chain(entry.explanation_embedding.iter())
                .copied()
                .collect();
            
            for (i, val) in full_embedding.iter().enumerate() {
                combined[i] += weight * val;
            }
        }
        
        combined
    }
    
    /// Cosine similarity
    fn cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm_a < 1e-10 || norm_b < 1e-10 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
    
    /// Softmax
    fn softmax(&self, values: &[f64]) -> Vec<f64> {
        let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = values.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f64 = exp_values.iter().sum();
        
        if sum < 1e-10 {
            vec![1.0 / values.len() as f64; values.len()]
        } else {
            exp_values.iter().map(|v| v / sum).collect()
        }
    }
    
    /// Project memory to memory dimension
    pub fn project_memory(&self, memory_retrieval: &[f64]) -> Tensor {
        let vec_f32: Vec<f32> = memory_retrieval.iter().map(|&x| x as f32).collect();
        let tensor = Tensor::from_vec(vec_f32, &[1, memory_retrieval.len()]);
        self.memory_proj.forward(&tensor)
    }
    
    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let total_memories = self.memories.len();
        let avg_usage = if total_memories > 0 {
            self.memories.iter().map(|m| m.usage_count).sum::<usize>() as f64 / total_memories as f64
        } else {
            0.0
        };
        
        let avg_verification = if total_memories > 0 {
            self.memories.iter().map(|m| m.verification_score).sum::<f64>() / total_memories as f64
        } else {
            0.0
        };
        
        MemoryStats {
            total_memories,
            avg_usage,
            avg_verification,
            capacity_used: total_memories as f64 / self.max_size as f64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub avg_usage: f64,
    pub avg_verification: f64,
    pub capacity_used: f64,
}

// ============================================================================
// PART 3: MEMORY-AUGMENTED NETWORK
// ============================================================================

#[derive(Debug, Clone)]
pub struct MemoryAugmentedNetwork {
    /// Memory bank
    memory_bank: MemoryBank,
    
    /// Input embedding layer
    input_embedder: Linear,
}

impl MemoryAugmentedNetwork {
    pub fn new(
        input_dim: usize,
        embedding_dim: usize,
        memory_dim: usize,
        max_memories: usize,
    ) -> Self {
        Self {
            memory_bank: MemoryBank::new(max_memories, embedding_dim, memory_dim),
            input_embedder: Linear::new(input_dim, embedding_dim, true),
        }
    }
    
    /// Forward pass with memory retrieval
    pub fn forward_with_memory(&mut self, input: &Tensor, top_k: usize) -> (Tensor, Tensor) {
        // Embed input
        let input_embedding = self.input_embedder.forward(input);
        let input_vec = input_embedding.to_vec();
        
        // Retrieve from memory
        let input_vec_f64: Vec<f64> = input_vec.iter().map(|&x| x as f64).collect();
        let memory_retrieval = self.memory_bank.retrieve(&input_vec_f64, top_k);
        let memory_tensor = self.memory_bank.project_memory(&memory_retrieval);
        
        (input_embedding, memory_tensor)
    }
    
    /// Store a verified solution in memory
    pub fn store_verified_solution(
        &mut self,
        problem_embedding: Vec<f64>,
        solution_embedding: Vec<f64>,
        explanation_embedding: Vec<f64>,
        verification_score: f64,
    ) {
        let entry = MemoryEntry::new(
            problem_embedding,
            solution_embedding,
            explanation_embedding,
            verification_score,
        );
        self.memory_bank.store(entry);
    }
    
    /// Get memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        self.memory_bank.get_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_bank() {
        let mut bank = MemoryBank::new(100, 128, 64);
        
        let entry = MemoryEntry::new(
            vec![1.0; 128],
            vec![2.0; 128],
            vec![3.0; 128],
            0.9,
        );
        
        bank.store(entry);
        
        let stats = bank.get_stats();
        assert_eq!(stats.total_memories, 1);
    }
    
    #[test]
    fn test_memory_retrieval() {
        let mut bank = MemoryBank::new(100, 128, 64);
        
        // Store some memories
        for i in 0..5 {
            let entry = MemoryEntry::new(
                vec![i as f64; 128],
                vec![(i + 1) as f64; 128],
                vec![(i + 2) as f64; 128],
                0.8 + i as f64 * 0.02,
            );
            bank.store(entry);
        }
        
        // Retrieve
        let query = vec![2.5; 128];
        let retrieved = bank.retrieve(&query, 3);
        
        assert_eq!(retrieved.len(), 128 * 3);
    }
}
