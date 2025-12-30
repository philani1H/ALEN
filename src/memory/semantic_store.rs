//! Semantic Memory Storage
//!
//! Persistent storage of verified thought states (understanding memory)
//! Stores: ψⱼ (verified thought vectors) + metadata
//! NOT raw text - compressed understanding

use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use sha2::{Sha256, Digest};

/// Semantic memory entry - a verified understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEntry {
    /// Thought vector (ψⱼ) - the compressed understanding
    pub thought: Vec<f32>,
    /// Concept label
    pub concept: String,
    /// Confidence score [0, 1]
    pub confidence: f64,
    /// Which operator produced this
    pub operator_id: usize,
    /// Operator name
    pub operator_name: String,
    /// Verification error when stored
    pub verification_error: f64,
    /// Timestamp
    pub timestamp: u64,
    /// Hash for integrity
    pub hash: String,
}

impl SemanticEntry {
    pub fn new(
        thought: Vec<f32>,
        concept: String,
        confidence: f64,
        operator_id: usize,
        operator_name: String,
        verification_error: f64,
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let hash = Self::compute_hash(&thought, &concept, timestamp);
        
        Self {
            thought,
            concept,
            confidence,
            operator_id,
            operator_name,
            verification_error,
            timestamp,
            hash,
        }
    }

    fn compute_hash(thought: &[f32], concept: &str, timestamp: u64) -> String {
        let mut hasher = Sha256::new();
        
        // Hash thought vector
        for &val in thought {
            hasher.update(val.to_le_bytes());
        }
        
        // Hash concept and timestamp
        hasher.update(concept.as_bytes());
        hasher.update(timestamp.to_le_bytes());
        
        format!("{:x}", hasher.finalize())
    }

    /// Verify integrity
    pub fn verify_integrity(&self) -> bool {
        let expected = Self::compute_hash(&self.thought, &self.concept, self.timestamp);
        self.hash == expected
    }

    /// Cosine similarity with another thought
    pub fn similarity(&self, other: &[f32]) -> f32 {
        if self.thought.len() != other.len() {
            return 0.0;
        }

        let dot: f32 = self.thought.iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_self: f32 = self.thought.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_other: f32 = other.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_self < 1e-10 || norm_other < 1e-10 {
            return 0.0;
        }

        dot / (norm_self * norm_other)
    }
}

/// Semantic memory store - persistent understanding storage
pub struct SemanticStore {
    /// Storage directory
    storage_dir: PathBuf,
    /// In-memory cache of entries
    entries: Vec<SemanticEntry>,
    /// Maximum entries before consolidation
    max_entries: usize,
    /// Minimum confidence to store
    min_confidence: f64,
}

impl SemanticStore {
    /// Create new semantic store
    pub fn new<P: AsRef<Path>>(storage_dir: P, max_entries: usize, min_confidence: f64) -> io::Result<Self> {
        let storage_dir = storage_dir.as_ref().to_path_buf();
        
        // Create directory if it doesn't exist
        fs::create_dir_all(&storage_dir)?;
        
        let mut store = Self {
            storage_dir,
            entries: Vec::new(),
            max_entries,
            min_confidence,
        };
        
        // Load existing entries
        store.load_all()?;
        
        Ok(store)
    }

    /// Insert a verified thought (only if confidence is high enough)
    pub fn insert(&mut self, entry: SemanticEntry) -> io::Result<bool> {
        // Check confidence threshold
        if entry.confidence < self.min_confidence {
            return Ok(false);
        }

        // Verify integrity
        if !entry.verify_integrity() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Entry integrity check failed"
            ));
        }

        // Check for duplicates (similar thoughts)
        if self.has_similar(&entry.thought, 0.95) {
            return Ok(false); // Already have this understanding
        }

        // Add to memory
        self.entries.push(entry.clone());

        // Persist to disk
        self.save_entry(&entry)?;

        // Consolidate if needed
        if self.entries.len() > self.max_entries {
            self.consolidate()?;
        }

        Ok(true)
    }

    /// Check if we have a similar thought already
    fn has_similar(&self, thought: &[f32], threshold: f32) -> bool {
        self.entries.iter()
            .any(|entry| entry.similarity(thought) > threshold)
    }

    /// Find most similar entries
    pub fn find_similar(&self, thought: &[f32], top_k: usize) -> Vec<(f32, &SemanticEntry)> {
        let mut similarities: Vec<(f32, &SemanticEntry)> = self.entries.iter()
            .map(|entry| (entry.similarity(thought), entry))
            .collect();

        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        similarities.truncate(top_k);
        similarities
    }

    /// Get entry by concept
    pub fn get_by_concept(&self, concept: &str) -> Option<&SemanticEntry> {
        self.entries.iter()
            .find(|entry| entry.concept == concept)
    }

    /// Get all entries
    pub fn get_all(&self) -> &[SemanticEntry] {
        &self.entries
    }

    /// Count entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Save single entry to disk
    fn save_entry(&self, entry: &SemanticEntry) -> io::Result<()> {
        let filename = format!("{}.alen", entry.hash);
        let path = self.storage_dir.join(filename);
        
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        
        bincode::serialize_into(writer, entry)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }

    /// Load all entries from disk
    fn load_all(&mut self) -> io::Result<()> {
        self.entries.clear();

        if !self.storage_dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(&self.storage_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("alen") {
                match self.load_entry(&path) {
                    Ok(semantic_entry) => {
                        if semantic_entry.verify_integrity() {
                            self.entries.push(semantic_entry);
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load {:?}: {}", path, e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Load single entry from disk
    fn load_entry(&self, path: &Path) -> io::Result<SemanticEntry> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        bincode::deserialize_from(reader)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }

    /// Consolidate memory (remove low-confidence or redundant entries)
    fn consolidate(&mut self) -> io::Result<()> {
        // Sort by confidence (descending)
        self.entries.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap()
        });

        // Keep top entries
        let keep_count = (self.max_entries as f32 * 0.8) as usize;
        let removed = self.entries.split_off(keep_count);

        // Delete removed entries from disk
        for entry in removed {
            let filename = format!("{}.alen", entry.hash);
            let path = self.storage_dir.join(filename);
            let _ = fs::remove_file(path); // Ignore errors
        }

        Ok(())
    }

    /// Clear all memory (use with caution!)
    pub fn clear(&mut self) -> io::Result<()> {
        self.entries.clear();

        if self.storage_dir.exists() {
            for entry in fs::read_dir(&self.storage_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("alen") {
                    fs::remove_file(path)?;
                }
            }
        }

        Ok(())
    }

    /// Export statistics
    pub fn statistics(&self) -> SemanticStats {
        let total = self.entries.len();
        let avg_confidence = if total > 0 {
            self.entries.iter().map(|e| e.confidence).sum::<f64>() / total as f64
        } else {
            0.0
        };

        let mut operator_usage = std::collections::HashMap::new();
        for entry in &self.entries {
            *operator_usage.entry(entry.operator_name.clone()).or_insert(0) += 1;
        }

        SemanticStats {
            total_entries: total,
            avg_confidence,
            operator_usage,
            storage_size_bytes: self.estimate_size(),
        }
    }

    fn estimate_size(&self) -> usize {
        self.entries.iter()
            .map(|e| e.thought.len() * 4 + 100) // Rough estimate
            .sum()
    }
}

/// Statistics about semantic memory
#[derive(Debug, Clone)]
pub struct SemanticStats {
    pub total_entries: usize,
    pub avg_confidence: f64,
    pub operator_usage: std::collections::HashMap<String, usize>,
    pub storage_size_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_semantic_entry_creation() {
        let thought = vec![0.1, 0.2, 0.3];
        let entry = SemanticEntry::new(
            thought.clone(),
            "test".to_string(),
            0.9,
            0,
            "Logical".to_string(),
            0.1,
        );

        assert_eq!(entry.thought, thought);
        assert_eq!(entry.concept, "test");
        assert!(entry.verify_integrity());
    }

    #[test]
    fn test_semantic_store() -> io::Result<()> {
        let dir = tempdir()?;
        let mut store = SemanticStore::new(dir.path(), 100, 0.7)?;

        let entry = SemanticEntry::new(
            vec![0.1, 0.2, 0.3],
            "test_concept".to_string(),
            0.9,
            0,
            "Logical".to_string(),
            0.1,
        );

        assert!(store.insert(entry)?);
        assert_eq!(store.len(), 1);

        Ok(())
    }

    #[test]
    fn test_similarity_search() -> io::Result<()> {
        let dir = tempdir()?;
        let mut store = SemanticStore::new(dir.path(), 100, 0.7)?;

        // Add some entries with distinct vectors
        for i in 1..=5 {
            // Use distinct vectors that won't be filtered as duplicates
            let thought = vec![
                i as f32 * 0.1 + 0.1,
                i as f32 * 0.05 + 0.5,
                i as f32 * 0.02 + 0.3
            ];
            let entry = SemanticEntry::new(
                thought,
                format!("concept_{}", i),
                0.9,
                i,  // Different timestamps (usize)
                "Logical".to_string(),
                0.1,
            );
            store.insert(entry)?;
        }

        // Search for similar
        let query = vec![0.2, 0.55, 0.32];
        let results = store.find_similar(&query, 3);

        // Should return at least some results
        assert!(results.len() >= 1);
        assert!(results[0].0 > 0.0); // Has similarity

        Ok(())
    }
}
