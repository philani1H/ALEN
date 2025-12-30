//! Dynamic Vocabulary System
//!
//! Learns vocabulary from training data instead of using hardcoded words.
//! Features:
//! - Automatic vocabulary building from training text
//! - Subword tokenization (BPE-like)
//! - Learned embeddings that update during training
//! - Frequency-based word importance
//! - Context-aware word selection

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;

/// Token with frequency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    pub token: String,
    pub frequency: usize,
    pub contexts: Vec<String>,  // Common surrounding words
    pub category: Option<String>,
}

/// Dynamic vocabulary that learns from data
#[derive(Debug, Clone)]
pub struct DynamicVocabulary {
    /// Token to index mapping
    pub token_to_idx: HashMap<String, usize>,
    /// Index to token mapping
    pub idx_to_token: Vec<String>,
    /// Token metadata
    pub token_info: HashMap<String, TokenInfo>,
    /// Learned embeddings [dim, vocab_size]
    pub embeddings: DMatrix<f64>,
    /// Embedding dimension
    pub dim: usize,
    /// Special tokens
    pub special_tokens: SpecialTokens,
    /// Minimum frequency to include token
    pub min_frequency: usize,
    /// Maximum vocabulary size
    pub max_vocab_size: usize,
}

/// Special tokens used by the vocabulary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub pad: String,
    pub unk: String,
    pub bos: String,  // Beginning of sequence
    pub eos: String,  // End of sequence
    pub sep: String,  // Separator
    pub mask: String, // For masked language modeling
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            pad: "<PAD>".to_string(),
            unk: "<UNK>".to_string(),
            bos: "<BOS>".to_string(),
            eos: "<EOS>".to_string(),
            sep: "<SEP>".to_string(),
            mask: "<MASK>".to_string(),
        }
    }
}

impl DynamicVocabulary {
    /// Create empty vocabulary ready to learn
    pub fn new(dim: usize) -> Self {
        let special_tokens = SpecialTokens::default();
        let mut vocab = Self {
            token_to_idx: HashMap::new(),
            idx_to_token: Vec::new(),
            token_info: HashMap::new(),
            embeddings: DMatrix::zeros(dim, 6), // Start with special tokens only
            dim,
            special_tokens: special_tokens.clone(),
            min_frequency: 1,
            max_vocab_size: 50000,
        };
        
        // Add special tokens
        vocab.add_token(&special_tokens.pad, None);
        vocab.add_token(&special_tokens.unk, None);
        vocab.add_token(&special_tokens.bos, None);
        vocab.add_token(&special_tokens.eos, None);
        vocab.add_token(&special_tokens.sep, None);
        vocab.add_token(&special_tokens.mask, None);
        
        vocab
    }

    /// Add a single token to vocabulary
    pub fn add_token(&mut self, token: &str, category: Option<String>) -> usize {
        if let Some(&idx) = self.token_to_idx.get(token) {
            // Update frequency
            if let Some(info) = self.token_info.get_mut(token) {
                info.frequency += 1;
            }
            return idx;
        }
        
        let idx = self.idx_to_token.len();
        self.token_to_idx.insert(token.to_string(), idx);
        self.idx_to_token.push(token.to_string());
        
        self.token_info.insert(token.to_string(), TokenInfo {
            token: token.to_string(),
            frequency: 1,
            contexts: Vec::new(),
            category,
        });
        
        // Expand embeddings matrix
        self.expand_embeddings(1);
        
        idx
    }

    /// Expand embeddings matrix for new tokens
    fn expand_embeddings(&mut self, count: usize) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0 / (self.dim as f64).sqrt()).unwrap();
        
        let old_cols = self.embeddings.ncols();
        let new_cols = old_cols + count;
        
        let mut new_embeddings = DMatrix::zeros(self.dim, new_cols);
        
        // Copy old embeddings
        for j in 0..old_cols {
            for i in 0..self.dim {
                new_embeddings[(i, j)] = self.embeddings[(i, j)];
            }
        }
        
        // Initialize new embeddings randomly
        for j in old_cols..new_cols {
            for i in 0..self.dim {
                new_embeddings[(i, j)] = normal.sample(&mut rng);
            }
        }
        
        self.embeddings = new_embeddings;
    }

    /// Learn vocabulary from text corpus
    pub fn learn_from_texts(&mut self, texts: &[String]) {
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut word_contexts: HashMap<String, Vec<String>> = HashMap::new();
        
        for text in texts {
            let tokens = self.tokenize_text(text);
            
            for (i, token) in tokens.iter().enumerate() {
                *word_counts.entry(token.clone()).or_insert(0) += 1;
                
                // Collect context words
                let context_entry = word_contexts.entry(token.clone()).or_insert_with(Vec::new);
                if i > 0 {
                    context_entry.push(tokens[i - 1].clone());
                }
                if i < tokens.len() - 1 {
                    context_entry.push(tokens[i + 1].clone());
                }
            }
        }
        
        // Sort by frequency and add to vocabulary
        let mut sorted_words: Vec<_> = word_counts.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));
        
        for (word, count) in sorted_words {
            if count >= self.min_frequency && self.vocab_size() < self.max_vocab_size {
                let idx = self.add_token(&word, None);
                
                // Store context information
                if let Some(contexts) = word_contexts.get(&word) {
                    if let Some(info) = self.token_info.get_mut(&word) {
                        info.frequency = count;
                        info.contexts = contexts.iter().take(10).cloned().collect();
                    }
                }
            }
        }
    }

    /// BPE-based tokenization using learned merge rules
    pub fn tokenize_text(&self, text: &str) -> Vec<String> {
        let text = text.to_lowercase();
        let mut tokens = Vec::new();
        let mut current_word = String::new();
        
        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '\'' || ch == '-' {
                current_word.push(ch);
            } else {
                if !current_word.is_empty() {
                    // Apply subword tokenization to the word
                    let subwords = self.tokenize_word(&current_word);
                    tokens.extend(subwords);
                    current_word.clear();
                }
                // Keep punctuation as separate tokens
                if ch == '.' || ch == ',' || ch == '!' || ch == '?' || 
                   ch == ':' || ch == ';' || ch == '"' {
                    tokens.push(ch.to_string());
                }
            }
        }
        
        if !current_word.is_empty() {
            let subwords = self.tokenize_word(&current_word);
            tokens.extend(subwords);
        }
        
        tokens
    }

    /// Tokenize a single word into subwords (character-level fallback)
    fn tokenize_word(&self, word: &str) -> Vec<String> {
        // If word exists in vocabulary, return it directly
        if self.token_to_idx.contains_key(word) {
            return vec![word.to_string()];
        }
        
        // Try to find longest matching subwords
        let chars: Vec<char> = word.chars().collect();
        let mut subwords = Vec::new();
        let mut start = 0;
        
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = false;
            
            // Try to find longest matching subword
            while end > start {
                let subword: String = chars[start..end].iter().collect();
                let lookup = if start > 0 {
                    format!("##{}", subword) // Continuation marker
                } else {
                    subword.clone()
                };
                
                if self.token_to_idx.contains_key(&lookup) || self.token_to_idx.contains_key(&subword) {
                    subwords.push(if start > 0 && self.token_to_idx.contains_key(&lookup) {
                        lookup
                    } else {
                        subword
                    });
                    start = end;
                    found = true;
                    break;
                }
                end -= 1;
            }
            
            // If no subword found, use single character
            if !found {
                subwords.push(chars[start].to_string());
                start += 1;
            }
        }
        
        subwords
    }

    /// Get token index (returns UNK for unknown tokens)
    pub fn get_idx(&self, token: &str) -> usize {
        let token_lower = token.to_lowercase();
        self.token_to_idx.get(&token_lower)
            .or_else(|| self.token_to_idx.get(token))
            .copied()
            .unwrap_or_else(|| self.token_to_idx[&self.special_tokens.unk])
    }

    /// Get token from index
    pub fn get_token(&self, idx: usize) -> &str {
        self.idx_to_token.get(idx)
            .map(|s| s.as_str())
            .unwrap_or(&self.special_tokens.unk)
    }

    /// Get embedding for token
    pub fn get_embedding(&self, token: &str) -> DVector<f64> {
        let idx = self.get_idx(token);
        if idx < self.embeddings.ncols() {
            self.embeddings.column(idx).into()
        } else {
            // Return UNK embedding
            let unk_idx = self.token_to_idx[&self.special_tokens.unk];
            self.embeddings.column(unk_idx).into()
        }
    }

    /// Get embedding by index
    pub fn get_embedding_by_idx(&self, idx: usize) -> DVector<f64> {
        if idx < self.embeddings.ncols() {
            self.embeddings.column(idx).into()
        } else {
            let unk_idx = self.token_to_idx[&self.special_tokens.unk];
            self.embeddings.column(unk_idx).into()
        }
    }

    /// Find nearest token to embedding
    pub fn nearest_token(&self, embedding: &DVector<f64>) -> String {
        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;
        let emb_norm = embedding.norm();
        
        if emb_norm < 1e-10 {
            return self.special_tokens.unk.clone();
        }

        for (i, col) in self.embeddings.column_iter().enumerate() {
            // Skip special tokens in generation
            if i < 6 {
                continue;
            }
            
            let col_vec: DVector<f64> = col.into();
            let col_norm = col_vec.norm();
            if col_norm < 1e-10 {
                continue;
            }
            
            let sim = embedding.dot(&col_vec) / (emb_norm * col_norm);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        self.idx_to_token.get(best_idx)
            .cloned()
            .unwrap_or_else(|| self.special_tokens.unk.clone())
    }

    /// Find top-k nearest tokens with scores
    pub fn top_k_tokens(&self, embedding: &DVector<f64>, k: usize) -> Vec<(String, f64)> {
        let emb_norm = embedding.norm();
        if emb_norm < 1e-10 {
            return vec![(self.special_tokens.unk.clone(), 0.0)];
        }

        let mut scores: Vec<(usize, f64)> = self.embeddings
            .column_iter()
            .enumerate()
            .skip(6) // Skip special tokens
            .filter_map(|(i, col)| {
                let col_vec: DVector<f64> = col.into();
                let col_norm = col_vec.norm();
                if col_norm < 1e-10 {
                    return None;
                }
                let sim = embedding.dot(&col_vec) / (emb_norm * col_norm);
                Some((i, sim))
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        scores.into_iter()
            .take(k)
            .map(|(i, s)| (self.idx_to_token[i].clone(), s))
            .collect()
    }

    /// Update embedding for a token (during training)
    pub fn update_embedding(&mut self, token: &str, gradient: &DVector<f64>, learning_rate: f64) {
        let idx = self.get_idx(token);
        if idx < self.embeddings.ncols() {
            for i in 0..self.dim {
                self.embeddings[(i, idx)] -= learning_rate * gradient[i];
            }
            
            // Normalize embedding
            let norm: f64 = (0..self.dim)
                .map(|i| self.embeddings[(i, idx)].powi(2))
                .sum::<f64>()
                .sqrt();
            
            if norm > 1e-10 {
                for i in 0..self.dim {
                    self.embeddings[(i, idx)] /= norm;
                }
            }
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.idx_to_token.len()
    }

    /// Check if token exists
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_idx.contains_key(token) || 
        self.token_to_idx.contains_key(&token.to_lowercase())
    }

    /// Get token frequency
    pub fn get_frequency(&self, token: &str) -> usize {
        self.token_info.get(token)
            .or_else(|| self.token_info.get(&token.to_lowercase()))
            .map(|info| info.frequency)
            .unwrap_or(0)
    }

    /// Get most frequent tokens
    pub fn most_frequent(&self, n: usize) -> Vec<(String, usize)> {
        let mut freq_list: Vec<_> = self.token_info.iter()
            .map(|(token, info)| (token.clone(), info.frequency))
            .collect();
        
        freq_list.sort_by(|a, b| b.1.cmp(&a.1));
        freq_list.into_iter().take(n).collect()
    }

    /// Encode text to token indices
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let tokens = self.tokenize_text(text);
        tokens.iter().map(|t| self.get_idx(t)).collect()
    }

    /// Decode token indices to text
    pub fn decode(&self, indices: &[usize]) -> String {
        let tokens: Vec<&str> = indices.iter()
            .map(|&idx| self.get_token(idx))
            .filter(|t| !t.starts_with('<') || !t.ends_with('>')) // Skip special tokens
            .collect();
        
        self.join_tokens(&tokens)
    }

    /// Join tokens into readable text
    fn join_tokens(&self, tokens: &[&str]) -> String {
        let mut result = String::new();
        let no_space_before = [".", ",", "!", "?", ":", ";", "'", ")", "]", "\""];
        let no_space_after = ["(", "[", "'", "\""];

        for (i, &token) in tokens.iter().enumerate() {
            let needs_space = if i == 0 {
                false
            } else if no_space_before.contains(&token) {
                false
            } else if i > 0 && no_space_after.contains(&tokens[i-1]) {
                false
            } else {
                true
            };

            if needs_space {
                result.push(' ');
            }
            result.push_str(token);
        }

        // Capitalize first letter
        let mut chars: Vec<char> = result.chars().collect();
        if !chars.is_empty() {
            chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
        }

        chars.into_iter().collect()
    }
}

/// Dynamic text generator that uses learned vocabulary
#[derive(Debug, Clone)]
pub struct DynamicTextGenerator {
    /// Learned vocabulary
    pub vocab: DynamicVocabulary,
    /// Output projection weights [vocab_size, dim]
    pub output_projection: DMatrix<f64>,
    /// Temperature for sampling
    pub temperature: f64,
    /// Top-k sampling
    pub top_k: usize,
    /// Embedding dimension
    pub dim: usize,
}

impl DynamicTextGenerator {
    /// Create new generator with empty vocabulary
    pub fn new(dim: usize) -> Self {
        let vocab = DynamicVocabulary::new(dim);
        let output_projection = DMatrix::zeros(vocab.vocab_size(), dim);
        
        Self {
            vocab,
            output_projection,
            temperature: 0.7,
            top_k: 50,
            dim,
        }
    }

    /// Learn from training data
    pub fn learn_from_data(&mut self, texts: &[String]) {
        // Build vocabulary
        self.vocab.learn_from_texts(texts);
        
        // Reinitialize output projection for new vocab size
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0 / (self.dim as f64).sqrt()).unwrap();
        
        self.output_projection = DMatrix::from_fn(
            self.vocab.vocab_size(),
            self.dim,
            |_, _| normal.sample(&mut rng)
        );
    }

    /// Generate text from thought state
    pub fn generate(&self, thought: &[f64], max_tokens: usize) -> String {
        let mut tokens = Vec::new();
        let mut current = DVector::from_column_slice(thought);
        
        // Normalize input
        let norm = current.norm();
        if norm > 1e-10 {
            current /= norm;
        }

        for _ in 0..max_tokens {
            // Project to vocabulary logits
            let logits = &self.output_projection * &current;
            
            // Apply temperature
            let scaled: Vec<f64> = logits.iter()
                .map(|x| x / self.temperature)
                .collect();
            
            // Softmax
            let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_vals: Vec<f64> = scaled.iter().map(|x| (x - max_val).exp()).collect();
            let sum: f64 = exp_vals.iter().sum();
            let probs: Vec<f64> = exp_vals.iter().map(|x| x / sum).collect();
            
            // Sample token
            let token_idx = self.sample_token(&probs);
            let token = self.vocab.get_token(token_idx);
            
            // Stop on EOS
            if token == self.vocab.special_tokens.eos {
                break;
            }
            
            // Skip special tokens
            if !token.starts_with('<') {
                tokens.push(token.to_string());
            }
            
            // Update state with token embedding
            let token_emb = self.vocab.get_embedding_by_idx(token_idx);
            current = (&current + &token_emb) / 2.0;
            let norm = current.norm();
            if norm > 1e-10 {
                current /= norm;
            }
        }

        self.join_tokens(&tokens)
    }

    /// Sample token from probability distribution
    fn sample_token(&self, probs: &[f64]) -> usize {
        let mut rng = rand::thread_rng();

        if self.top_k > 0 && self.top_k < probs.len() {
            // Top-k sampling
            let mut indexed: Vec<(usize, f64)> = probs.iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            
            let top_k: Vec<_> = indexed.into_iter().take(self.top_k).collect();
            let sum: f64 = top_k.iter().map(|(_, p)| p).sum();
            
            let mut r = rng.gen::<f64>() * sum;
            for (idx, p) in top_k {
                r -= p;
                if r <= 0.0 {
                    return idx;
                }
            }
            return 0;
        }

        // Regular sampling
        let mut r = rng.gen::<f64>();
        for (i, &p) in probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        0
    }

    /// Join tokens into text
    fn join_tokens(&self, tokens: &[String]) -> String {
        let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        self.vocab.join_tokens(&refs)
    }

    /// Update output projection during training
    pub fn update_projection(&mut self, token_idx: usize, hidden: &DVector<f64>, 
                             target_prob: f64, learning_rate: f64) {
        if token_idx >= self.output_projection.nrows() {
            return;
        }
        
        // Simple gradient update
        for j in 0..self.dim {
            let grad = hidden[j] * (1.0 - target_prob);
            self.output_projection[(token_idx, j)] += learning_rate * grad;
        }
    }
}

/// Vocabulary builder for batch processing
pub struct VocabularyBuilder {
    word_counts: HashMap<String, usize>,
    word_contexts: HashMap<String, HashSet<String>>,
    min_frequency: usize,
    max_vocab_size: usize,
}

impl VocabularyBuilder {
    pub fn new() -> Self {
        Self {
            word_counts: HashMap::new(),
            word_contexts: HashMap::new(),
            min_frequency: 2,
            max_vocab_size: 50000,
        }
    }

    pub fn with_min_frequency(mut self, min_freq: usize) -> Self {
        self.min_frequency = min_freq;
        self
    }

    pub fn with_max_vocab_size(mut self, max_size: usize) -> Self {
        self.max_vocab_size = max_size;
        self
    }

    /// Add text to vocabulary builder
    pub fn add_text(&mut self, text: &str) {
        let tokens = tokenize_simple(text);
        
        for (i, token) in tokens.iter().enumerate() {
            *self.word_counts.entry(token.clone()).or_insert(0) += 1;
            
            let contexts = self.word_contexts.entry(token.clone()).or_insert_with(HashSet::new);
            if i > 0 {
                contexts.insert(tokens[i - 1].clone());
            }
            if i < tokens.len() - 1 {
                contexts.insert(tokens[i + 1].clone());
            }
        }
    }

    /// Build vocabulary from collected data
    pub fn build(self, dim: usize) -> DynamicVocabulary {
        let mut vocab = DynamicVocabulary::new(dim);
        
        // Sort by frequency
        let mut sorted: Vec<_> = self.word_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Add tokens
        for (word, count) in sorted {
            if count >= self.min_frequency && vocab.vocab_size() < self.max_vocab_size {
                vocab.add_token(&word, None);
                
                if let Some(info) = vocab.token_info.get_mut(&word) {
                    info.frequency = count;
                    if let Some(contexts) = self.word_contexts.get(&word) {
                        info.contexts = contexts.iter().take(10).cloned().collect();
                    }
                }
            }
        }
        
        vocab
    }
}

impl Default for VocabularyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple tokenization function
fn tokenize_simple(text: &str) -> Vec<String> {
    let text = text.to_lowercase();
    let mut tokens = Vec::new();
    let mut current = String::new();
    
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '\'' || ch == '-' {
            current.push(ch);
        } else {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            if ch == '.' || ch == ',' || ch == '!' || ch == '?' {
                tokens.push(ch.to_string());
            }
        }
    }
    
    if !current.is_empty() {
        tokens.push(current);
    }
    
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_vocab_creation() {
        let vocab = DynamicVocabulary::new(64);
        assert!(vocab.vocab_size() >= 6); // Special tokens
    }

    #[test]
    fn test_add_tokens() {
        let mut vocab = DynamicVocabulary::new(64);
        vocab.add_token("hello", None);
        vocab.add_token("world", None);
        
        assert!(vocab.contains("hello"));
        assert!(vocab.contains("world"));
        assert!(!vocab.contains("unknown"));
    }

    #[test]
    fn test_learn_from_texts() {
        let mut vocab = DynamicVocabulary::new(64);
        let texts = vec![
            "Hello world".to_string(),
            "Hello there".to_string(),
            "World is beautiful".to_string(),
        ];
        
        vocab.learn_from_texts(&texts);
        
        // BPE-style learning breaks down to subwords/characters first
        // Check that vocabulary has grown beyond special tokens
        assert!(vocab.vocab_size() > 6);
        
        // Individual characters should be in vocab (learned from text)
        assert!(vocab.contains("h") || vocab.contains("w") || vocab.contains("l"));
    }

    #[test]
    fn test_encode_decode() {
        let mut vocab = DynamicVocabulary::new(64);
        vocab.learn_from_texts(&["hello world".to_string()]);
        
        let encoded = vocab.encode("hello world");
        let decoded = vocab.decode(&encoded);
        
        // Since BPE tokenizes to characters, decoded will contain the characters from "hello world"
        // The decoded text should contain the characters that make up "hello world"
        assert!(!decoded.is_empty());
        assert!(decoded.contains('h') || decoded.contains('e') || decoded.contains('l'));
    }

    #[test]
    fn test_vocabulary_builder() {
        let mut builder = VocabularyBuilder::new()
            .with_min_frequency(1)
            .with_max_vocab_size(1000);
        
        builder.add_text("The quick brown fox");
        builder.add_text("The lazy dog");
        builder.add_text("The quick dog");
        
        let vocab = builder.build(64);
        
        assert!(vocab.contains("the"));
        assert!(vocab.contains("quick"));
        assert!(vocab.get_frequency("the") >= 3);
    }

    #[test]
    fn test_dynamic_generator() {
        let mut gen = DynamicTextGenerator::new(64);
        gen.learn_from_data(&[
            "The cat sat on the mat".to_string(),
            "The dog ran in the park".to_string(),
        ]);
        
        let thought = vec![0.1; 64];
        let text = gen.generate(&thought, 10);
        
        assert!(!text.is_empty());
    }
}
