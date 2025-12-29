//! Byte Pair Encoding (BPE) Tokenizer
//!
//! Production-grade subword tokenization that learns from data.
//! No hardcoded vocabulary - everything is learned from training corpus.
//!
//! Algorithm:
//! 1. Start with character-level vocabulary
//! 2. Count all adjacent pairs in corpus
//! 3. Merge most frequent pair into new token
//! 4. Repeat until desired vocabulary size

use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;
use serde::{Deserialize, Serialize};

/// A merge operation in BPE
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MergeRule {
    pub pair: (String, String),
    pub result: String,
    pub priority: usize,
}

/// BPE Tokenizer - learns subword units from data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    /// Token to ID mapping
    pub vocab: HashMap<String, usize>,
    /// ID to token mapping
    pub id_to_token: Vec<String>,
    /// Merge rules in order of priority
    pub merges: Vec<MergeRule>,
    /// Special tokens
    pub special_tokens: BPESpecialTokens,
    /// Maximum vocabulary size
    pub max_vocab_size: usize,
    /// Minimum frequency for a pair to be merged
    pub min_frequency: usize,
}

/// Special tokens for BPE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPESpecialTokens {
    pub pad_token: String,
    pub unk_token: String,
    pub bos_token: String,
    pub eos_token: String,
    pub sep_token: String,
    pub mask_token: String,
    pub pad_id: usize,
    pub unk_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    pub sep_id: usize,
    pub mask_id: usize,
}

impl Default for BPESpecialTokens {
    fn default() -> Self {
        Self {
            pad_token: "<PAD>".to_string(),
            unk_token: "<UNK>".to_string(),
            bos_token: "<BOS>".to_string(),
            eos_token: "<EOS>".to_string(),
            sep_token: "<SEP>".to_string(),
            mask_token: "<MASK>".to_string(),
            pad_id: 0,
            unk_id: 1,
            bos_id: 2,
            eos_id: 3,
            sep_id: 4,
            mask_id: 5,
        }
    }
}

impl BPETokenizer {
    /// Create a new BPE tokenizer (empty, ready to train)
    pub fn new(max_vocab_size: usize) -> Self {
        let special_tokens = BPESpecialTokens::default();
        
        // Initialize with special tokens
        let mut vocab = HashMap::new();
        let mut id_to_token = Vec::new();
        
        // Add special tokens first
        for (id, token) in [
            (0, &special_tokens.pad_token),
            (1, &special_tokens.unk_token),
            (2, &special_tokens.bos_token),
            (3, &special_tokens.eos_token),
            (4, &special_tokens.sep_token),
            (5, &special_tokens.mask_token),
        ] {
            vocab.insert(token.clone(), id);
            id_to_token.push(token.clone());
        }
        
        Self {
            vocab,
            id_to_token,
            merges: Vec::new(),
            special_tokens,
            max_vocab_size,
            min_frequency: 2,
        }
    }

    /// Train BPE on a corpus of texts
    pub fn train(&mut self, texts: &[String], num_merges: usize) {
        // Step 1: Build initial character vocabulary
        let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();
        
        for text in texts {
            for word in self.pre_tokenize(text) {
                // Split word into characters with word boundary marker
                let chars: Vec<String> = word.chars()
                    .map(|c| c.to_string())
                    .collect();
                
                if !chars.is_empty() {
                    // Add end-of-word marker
                    let mut word_chars = chars;
                    word_chars.push("</w>".to_string());
                    *word_freqs.entry(word_chars).or_insert(0) += 1;
                }
            }
        }
        
        // Add all unique characters to vocabulary
        let mut all_chars: HashSet<String> = HashSet::new();
        for word in word_freqs.keys() {
            for ch in word {
                all_chars.insert(ch.clone());
            }
        }
        
        for ch in all_chars {
            if !self.vocab.contains_key(&ch) {
                let id = self.id_to_token.len();
                self.vocab.insert(ch.clone(), id);
                self.id_to_token.push(ch);
            }
        }
        
        // Step 2: Iteratively merge most frequent pairs
        for merge_idx in 0..num_merges {
            if self.vocab.len() >= self.max_vocab_size {
                break;
            }
            
            // Count all pairs
            let pair_counts = self.count_pairs(&word_freqs);
            
            if pair_counts.is_empty() {
                break;
            }
            
            // Find most frequent pair
            let best_pair = pair_counts.iter()
                .max_by_key(|(_, count)| *count)
                .map(|(pair, count)| (pair.clone(), *count));
            
            if let Some(((p1, p2), count)) = best_pair {
                if count < self.min_frequency {
                    break;
                }
                
                // Create merged token
                let merged = format!("{}{}", p1, p2);
                
                // Add to vocabulary
                if !self.vocab.contains_key(&merged) {
                    let id = self.id_to_token.len();
                    self.vocab.insert(merged.clone(), id);
                    self.id_to_token.push(merged.clone());
                }
                
                // Record merge rule
                self.merges.push(MergeRule {
                    pair: (p1.clone(), p2.clone()),
                    result: merged.clone(),
                    priority: merge_idx,
                });
                
                // Apply merge to all words
                word_freqs = self.apply_merge(&word_freqs, &p1, &p2, &merged);
            } else {
                break;
            }
        }
    }

    /// Pre-tokenize text into words
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let text = text.to_lowercase();
        let mut words = Vec::new();
        let mut current_word = String::new();
        
        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '\'' {
                current_word.push(ch);
            } else {
                if !current_word.is_empty() {
                    words.push(current_word.clone());
                    current_word.clear();
                }
                // Keep punctuation as separate tokens
                if !ch.is_whitespace() {
                    words.push(ch.to_string());
                }
            }
        }
        
        if !current_word.is_empty() {
            words.push(current_word);
        }
        
        words
    }

    /// Count all adjacent pairs in the vocabulary
    fn count_pairs(&self, word_freqs: &HashMap<Vec<String>, usize>) -> HashMap<(String, String), usize> {
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        
        for (word, freq) in word_freqs {
            if word.len() < 2 {
                continue;
            }
            
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                *pair_counts.entry(pair).or_insert(0) += freq;
            }
        }
        
        pair_counts
    }

    /// Apply a merge operation to all words
    fn apply_merge(
        &self,
        word_freqs: &HashMap<Vec<String>, usize>,
        p1: &str,
        p2: &str,
        merged: &str,
    ) -> HashMap<Vec<String>, usize> {
        let mut new_word_freqs: HashMap<Vec<String>, usize> = HashMap::new();
        
        for (word, freq) in word_freqs {
            let mut new_word = Vec::new();
            let mut i = 0;
            
            while i < word.len() {
                if i < word.len() - 1 && word[i] == p1 && word[i + 1] == p2 {
                    new_word.push(merged.to_string());
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            
            *new_word_freqs.entry(new_word).or_insert(0) += freq;
        }
        
        new_word_freqs
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let words = self.pre_tokenize(text);
        let mut token_ids = Vec::new();
        
        for word in words {
            let word_tokens = self.encode_word(&word);
            token_ids.extend(word_tokens);
        }
        
        token_ids
    }

    /// Encode a single word using learned BPE merges
    fn encode_word(&self, word: &str) -> Vec<usize> {
        if word.is_empty() {
            return vec![];
        }
        
        // Start with characters
        let mut tokens: Vec<String> = word.chars()
            .map(|c| c.to_string())
            .collect();
        tokens.push("</w>".to_string());
        
        // Apply merges in order
        for merge in &self.merges {
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if tokens[i] == merge.pair.0 && tokens[i + 1] == merge.pair.1 {
                    tokens[i] = merge.result.clone();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
        
        // Convert to IDs
        tokens.iter()
            .map(|t| self.vocab.get(t).copied().unwrap_or(self.special_tokens.unk_id))
            .collect()
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[usize]) -> String {
        let tokens: Vec<&str> = ids.iter()
            .filter_map(|&id| self.id_to_token.get(id).map(|s| s.as_str()))
            .filter(|t| !t.starts_with('<') || !t.ends_with('>'))
            .collect();
        
        let mut text = tokens.join("");
        
        // Remove end-of-word markers and clean up
        text = text.replace("</w>", " ");
        text = text.trim().to_string();
        
        // Capitalize first letter
        let mut chars: Vec<char> = text.chars().collect();
        if !chars.is_empty() {
            chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
        }
        
        chars.into_iter().collect()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get token ID
    pub fn token_to_id(&self, token: &str) -> usize {
        self.vocab.get(token).copied().unwrap_or(self.special_tokens.unk_id)
    }

    /// Get token from ID
    pub fn id_to_token(&self, id: usize) -> &str {
        self.id_to_token.get(id)
            .map(|s| s.as_str())
            .unwrap_or(&self.special_tokens.unk_token)
    }

    /// Check if token exists
    pub fn contains(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }

    /// Add a single token to vocabulary
    pub fn add_token(&mut self, token: &str) -> usize {
        if let Some(&id) = self.vocab.get(token) {
            return id;
        }
        
        let id = self.id_to_token.len();
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.push(token.to_string());
        id
    }

    /// Encode with special tokens (BOS/EOS)
    pub fn encode_with_special(&self, text: &str) -> Vec<usize> {
        let mut ids = vec![self.special_tokens.bos_id];
        ids.extend(self.encode(text));
        ids.push(self.special_tokens.eos_id);
        ids
    }

    /// Batch encode multiple texts
    pub fn encode_batch(&self, texts: &[String]) -> Vec<Vec<usize>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Batch decode multiple sequences
    pub fn decode_batch(&self, batch: &[Vec<usize>]) -> Vec<String> {
        batch.iter().map(|ids| self.decode(ids)).collect()
    }
}

/// BPE Trainer for building tokenizer from corpus
pub struct BPETrainer {
    /// Target vocabulary size
    pub vocab_size: usize,
    /// Minimum frequency for merges
    pub min_frequency: usize,
    /// Show progress during training
    pub show_progress: bool,
}

impl BPETrainer {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            min_frequency: 2,
            show_progress: false,
        }
    }

    pub fn with_min_frequency(mut self, min_freq: usize) -> Self {
        self.min_frequency = min_freq;
        self
    }

    pub fn with_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }

    /// Train a new tokenizer from texts
    pub fn train(&self, texts: &[String]) -> BPETokenizer {
        let mut tokenizer = BPETokenizer::new(self.vocab_size);
        tokenizer.min_frequency = self.min_frequency;
        
        // Calculate number of merges needed
        let num_merges = self.vocab_size.saturating_sub(tokenizer.vocab.len());
        
        tokenizer.train(texts, num_merges);
        tokenizer
    }

    /// Train from file (one text per line)
    pub fn train_from_file(&self, path: &str) -> Result<BPETokenizer, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        let texts: Vec<String> = content.lines().map(|s| s.to_string()).collect();
        Ok(self.train(&texts))
    }
}

/// Tokenizer with learned embeddings
#[derive(Debug, Clone)]
pub struct BPEWithEmbeddings {
    /// The BPE tokenizer
    pub tokenizer: BPETokenizer,
    /// Token embeddings [vocab_size, dim]
    pub embeddings: Vec<Vec<f64>>,
    /// Embedding dimension
    pub dim: usize,
}

impl BPEWithEmbeddings {
    /// Create tokenizer with random embeddings
    pub fn new(tokenizer: BPETokenizer, dim: usize) -> Self {
        use rand::Rng;
        use rand_distr::{Normal, Distribution};
        
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0 / (dim as f64).sqrt()).unwrap();
        
        let embeddings: Vec<Vec<f64>> = (0..tokenizer.vocab_size())
            .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        
        Self {
            tokenizer,
            embeddings,
            dim,
        }
    }

    /// Get embedding for token ID
    pub fn get_embedding(&self, id: usize) -> &[f64] {
        if id < self.embeddings.len() {
            &self.embeddings[id]
        } else {
            &self.embeddings[self.tokenizer.special_tokens.unk_id]
        }
    }

    /// Encode text to embeddings
    pub fn encode_to_embeddings(&self, text: &str) -> Vec<Vec<f64>> {
        let ids = self.tokenizer.encode(text);
        ids.iter().map(|&id| self.get_embedding(id).to_vec()).collect()
    }

    /// Get mean embedding for text
    pub fn mean_embedding(&self, text: &str) -> Vec<f64> {
        let embeddings = self.encode_to_embeddings(text);
        
        if embeddings.is_empty() {
            return vec![0.0; self.dim];
        }
        
        let mut mean = vec![0.0; self.dim];
        for emb in &embeddings {
            for (i, &v) in emb.iter().enumerate() {
                mean[i] += v;
            }
        }
        
        let n = embeddings.len() as f64;
        for v in &mut mean {
            *v /= n;
        }
        
        // Normalize
        let norm: f64 = mean.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for v in &mut mean {
                *v /= norm;
            }
        }
        
        mean
    }

    /// Update embedding during training
    pub fn update_embedding(&mut self, id: usize, gradient: &[f64], lr: f64) {
        if id >= self.embeddings.len() {
            return;
        }
        
        for (i, &g) in gradient.iter().enumerate() {
            if i < self.dim {
                self.embeddings[id][i] -= lr * g;
            }
        }
    }

    /// Find nearest token to embedding
    pub fn nearest_token(&self, embedding: &[f64]) -> usize {
        let emb_norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if emb_norm < 1e-10 {
            return self.tokenizer.special_tokens.unk_id;
        }
        
        let mut best_id = 0;
        let mut best_sim = f64::NEG_INFINITY;
        
        for (id, token_emb) in self.embeddings.iter().enumerate() {
            // Skip special tokens
            if id < 6 {
                continue;
            }
            
            let token_norm: f64 = token_emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            if token_norm < 1e-10 {
                continue;
            }
            
            let dot: f64 = embedding.iter().zip(token_emb.iter()).map(|(a, b)| a * b).sum();
            let sim = dot / (emb_norm * token_norm);
            
            if sim > best_sim {
                best_sim = sim;
                best_id = id;
            }
        }
        
        best_id
    }

    /// Find top-k nearest tokens
    pub fn top_k_tokens(&self, embedding: &[f64], k: usize) -> Vec<(usize, f64)> {
        let emb_norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if emb_norm < 1e-10 {
            return vec![(self.tokenizer.special_tokens.unk_id, 0.0)];
        }
        
        let mut scores: Vec<(usize, f64)> = self.embeddings.iter()
            .enumerate()
            .skip(6) // Skip special tokens
            .filter_map(|(id, token_emb)| {
                let token_norm: f64 = token_emb.iter().map(|x| x * x).sum::<f64>().sqrt();
                if token_norm < 1e-10 {
                    return None;
                }
                let dot: f64 = embedding.iter().zip(token_emb.iter()).map(|(a, b)| a * b).sum();
                Some((id, dot / (emb_norm * token_norm)))
            })
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scores.into_iter().take(k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_training() {
        let texts = vec![
            "the quick brown fox".to_string(),
            "the lazy dog".to_string(),
            "the quick dog jumps".to_string(),
            "brown fox runs quickly".to_string(),
        ];
        
        let trainer = BPETrainer::new(100).with_min_frequency(1);
        let tokenizer = trainer.train(&texts);
        
        assert!(tokenizer.vocab_size() > 6); // More than just special tokens
        println!("Vocab size: {}", tokenizer.vocab_size());
    }

    #[test]
    fn test_encode_decode() {
        let texts = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "world peace".to_string(),
        ];
        
        let trainer = BPETrainer::new(50).with_min_frequency(1);
        let tokenizer = trainer.train(&texts);
        
        let encoded = tokenizer.encode("hello world");
        let decoded = tokenizer.decode(&encoded);
        
        assert!(decoded.to_lowercase().contains("hello"));
        assert!(decoded.to_lowercase().contains("world"));
    }

    #[test]
    fn test_bpe_with_embeddings() {
        let texts = vec!["test text".to_string()];
        let trainer = BPETrainer::new(50).with_min_frequency(1);
        let tokenizer = trainer.train(&texts);
        
        let bpe_emb = BPEWithEmbeddings::new(tokenizer, 64);
        let mean = bpe_emb.mean_embedding("test");
        
        assert_eq!(mean.len(), 64);
    }

    #[test]
    fn test_nearest_token() {
        let texts = vec![
            "cat dog bird".to_string(),
            "cat meows".to_string(),
            "dog barks".to_string(),
        ];
        
        let trainer = BPETrainer::new(100).with_min_frequency(1);
        let tokenizer = trainer.train(&texts);
        let bpe_emb = BPEWithEmbeddings::new(tokenizer, 64);
        
        // Get embedding for "cat" and find nearest
        let cat_emb = bpe_emb.mean_embedding("cat");
        let nearest = bpe_emb.nearest_token(&cat_emb);
        
        assert!(nearest > 0); // Should find something
    }
}
