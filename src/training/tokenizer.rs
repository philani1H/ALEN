//! Training Tokenizer
//!
//! Wraps the BPE tokenizer for training pipeline integration.
//! Handles batch tokenization and embedding preparation.

use crate::generation::bpe_tokenizer::{BPETokenizer, BPETrainer, BPEWithEmbeddings};
use serde::{Deserialize, Serialize};

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Maximum vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Minimum token frequency for BPE training
    pub min_frequency: usize,
    /// Whether to add special tokens
    pub add_special_tokens: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            embedding_dim: 512,
            max_seq_length: 512,
            min_frequency: 2,
            add_special_tokens: true,
        }
    }
}

/// Tokenizer for training pipeline
pub struct Tokenizer {
    /// Configuration
    pub config: TokenizerConfig,
    /// BPE tokenizer with embeddings
    pub bpe: Option<BPEWithEmbeddings>,
    /// Whether tokenizer is trained
    pub is_trained: bool,
}

impl Tokenizer {
    /// Create new tokenizer
    pub fn new(config: TokenizerConfig) -> Self {
        Self {
            config,
            bpe: None,
            is_trained: false,
        }
    }

    /// Train tokenizer on corpus
    pub fn train(&mut self, texts: &[String]) {
        let trainer = BPETrainer::new(self.config.vocab_size)
            .with_min_frequency(self.config.min_frequency);
        
        let tokenizer = trainer.train(texts);
        self.bpe = Some(BPEWithEmbeddings::new(tokenizer, self.config.embedding_dim));
        self.is_trained = true;
    }

    /// Tokenize single text
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        if let Some(ref bpe) = self.bpe {
            let mut ids = bpe.tokenizer.encode(text);
            
            // Truncate if needed
            if ids.len() > self.config.max_seq_length {
                ids.truncate(self.config.max_seq_length);
            }
            
            ids
        } else {
            vec![]
        }
    }

    /// Tokenize with special tokens
    pub fn tokenize_with_special(&self, text: &str) -> Vec<usize> {
        if let Some(ref bpe) = self.bpe {
            let mut ids = bpe.tokenizer.encode_with_special(text);
            
            if ids.len() > self.config.max_seq_length {
                ids.truncate(self.config.max_seq_length - 1);
                ids.push(bpe.tokenizer.special_tokens.eos_id);
            }
            
            ids
        } else {
            vec![]
        }
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[usize]) -> String {
        if let Some(ref bpe) = self.bpe {
            bpe.tokenizer.decode(ids)
        } else {
            String::new()
        }
    }

    /// Get embeddings for token IDs
    pub fn get_embeddings(&self, ids: &[usize]) -> Vec<Vec<f64>> {
        if let Some(ref bpe) = self.bpe {
            ids.iter()
                .map(|&id| bpe.get_embedding(id).to_vec())
                .collect()
        } else {
            vec![]
        }
    }

    /// Get mean embedding for text
    pub fn embed_text(&self, text: &str) -> Vec<f64> {
        if let Some(ref bpe) = self.bpe {
            bpe.mean_embedding(text)
        } else {
            vec![0.0; self.config.embedding_dim]
        }
    }

    /// Batch tokenize
    pub fn tokenize_batch(&self, texts: &[String]) -> TokenizedBatch {
        let token_ids: Vec<Vec<usize>> = texts.iter()
            .map(|t| self.tokenize(t))
            .collect();
        
        let max_len = token_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);
        
        // Pad sequences
        let pad_id = self.bpe.as_ref()
            .map(|b| b.tokenizer.special_tokens.pad_id)
            .unwrap_or(0);
        
        let padded: Vec<Vec<usize>> = token_ids.iter()
            .map(|ids| {
                let mut padded = ids.clone();
                padded.resize(max_len, pad_id);
                padded
            })
            .collect();
        
        // Create attention mask
        let attention_mask: Vec<Vec<f64>> = token_ids.iter()
            .map(|ids| {
                let mut mask = vec![1.0; ids.len()];
                mask.resize(max_len, 0.0);
                mask
            })
            .collect();
        
        TokenizedBatch {
            input_ids: padded,
            attention_mask,
            sequence_lengths: token_ids.iter().map(|ids| ids.len()).collect(),
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.bpe.as_ref()
            .map(|b| b.tokenizer.vocab_size())
            .unwrap_or(0)
    }
}

/// Batch of tokenized sequences
#[derive(Debug, Clone)]
pub struct TokenizedBatch {
    /// Token IDs [batch_size, seq_len]
    pub input_ids: Vec<Vec<usize>>,
    /// Attention mask [batch_size, seq_len]
    pub attention_mask: Vec<Vec<f64>>,
    /// Original sequence lengths
    pub sequence_lengths: Vec<usize>,
}

impl TokenizedBatch {
    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.input_ids.len()
    }

    /// Get max sequence length
    pub fn max_seq_len(&self) -> usize {
        self.input_ids.first().map(|ids| ids.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_training() {
        let config = TokenizerConfig {
            vocab_size: 100,
            embedding_dim: 64,
            max_seq_length: 128,
            min_frequency: 1,
            add_special_tokens: true,
        };

        let mut tokenizer = Tokenizer::new(config);
        
        let texts = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "world peace".to_string(),
        ];

        tokenizer.train(&texts);
        assert!(tokenizer.is_trained);
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_tokenize_decode() {
        let config = TokenizerConfig {
            vocab_size: 100,
            embedding_dim: 64,
            max_seq_length: 128,
            min_frequency: 1,
            add_special_tokens: true,
        };

        let mut tokenizer = Tokenizer::new(config);
        tokenizer.train(&["test text".to_string()]);

        let ids = tokenizer.tokenize("test");
        let decoded = tokenizer.decode(&ids);

        assert!(!ids.is_empty());
        assert!(decoded.to_lowercase().contains("test"));
    }

    #[test]
    fn test_batch_tokenization() {
        let config = TokenizerConfig {
            vocab_size: 100,
            embedding_dim: 64,
            max_seq_length: 128,
            min_frequency: 1,
            add_special_tokens: true,
        };

        let mut tokenizer = Tokenizer::new(config);
        tokenizer.train(&["hello world".to_string(), "test".to_string()]);

        let batch = tokenizer.tokenize_batch(&[
            "hello".to_string(),
            "world".to_string(),
        ]);

        assert_eq!(batch.batch_size(), 2);
        assert!(batch.max_seq_len() > 0);
    }
}
