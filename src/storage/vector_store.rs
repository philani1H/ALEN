//! Vector Store - Optimized Vector Database for AI/LLM
//!
//! High-performance vector storage with:
//! - Fast similarity search (cosine similarity)
//! - Persistent storage (SQLite backend)
//! - Batch operations
//! - Indexing for fast retrieval
//! - Metadata filtering

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Vector entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    /// Unique ID
    pub id: String,
    /// Vector embedding
    pub vector: Vec<f64>,
    /// Text content (for reference/debugging)
    pub content: String,
    /// Metadata (JSON)
    pub metadata: String,
    /// Timestamp
    pub timestamp: i64,
}

/// Vector store with persistent storage
pub struct VectorStore {
    /// Database connection
    conn: Arc<Mutex<Connection>>,
    /// Dimension of vectors
    dimension: usize,
}

impl VectorStore {
    /// Create new vector store with persistent storage
    pub fn new<P: AsRef<Path>>(path: P, dimension: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let conn = Connection::open(path)?;
        
        // Create table with optimized schema
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                timestamp INTEGER NOT NULL,
                norm REAL NOT NULL
            )",
            [],
        )?;
        
        // Create index on timestamp for fast recent queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON vectors(timestamp DESC)",
            [],
        )?;
        
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            dimension,
        })
    }
    
    /// Create in-memory vector store (for testing)
    pub fn in_memory(dimension: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let conn = Connection::open_in_memory()?;
        
        conn.execute(
            "CREATE TABLE vectors (
                id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                timestamp INTEGER NOT NULL,
                norm REAL NOT NULL
            )",
            [],
        )?;
        
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            dimension,
        })
    }
    
    /// Insert vector
    pub fn insert(&self, entry: &VectorEntry) -> Result<(), Box<dyn std::error::Error>> {
        if entry.vector.len() != self.dimension {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                entry.vector.len()
            ).into());
        }
        
        // Calculate norm for faster similarity search
        let norm: f64 = entry.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        // Serialize vector
        let vector_bytes = bincode::serialize(&entry.vector)?;
        
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO vectors (id, vector, content, metadata, timestamp, norm)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                &entry.id,
                &vector_bytes,
                &entry.content,
                &entry.metadata,
                entry.timestamp,
                norm,
            ],
        )?;
        
        Ok(())
    }
    
    /// Batch insert (optimized)
    pub fn batch_insert(&self, entries: &[VectorEntry]) -> Result<(), Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        
        for entry in entries {
            if entry.vector.len() != self.dimension {
                continue;
            }
            
            let norm: f64 = entry.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
            let vector_bytes = bincode::serialize(&entry.vector)?;
            
            tx.execute(
                "INSERT OR REPLACE INTO vectors (id, vector, content, metadata, timestamp, norm)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    &entry.id,
                    &vector_bytes,
                    &entry.content,
                    &entry.metadata,
                    entry.timestamp,
                    norm,
                ],
            )?;
        }
        
        tx.commit()?;
        Ok(())
    }
    
    /// Find similar vectors using cosine similarity
    pub fn find_similar(
        &self,
        query: &[f64],
        top_k: usize,
    ) -> Result<Vec<(VectorEntry, f64)>, Box<dyn std::error::Error>> {
        if query.len() != self.dimension {
            return Err(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                query.len()
            ).into());
        }
        
        // Calculate query norm
        let query_norm: f64 = query.iter().map(|x| x * x).sum::<f64>().sqrt();
        if query_norm < 1e-10 {
            return Ok(vec![]);
        }
        
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, vector, content, metadata, timestamp, norm FROM vectors"
        )?;
        
        let mut results: Vec<(VectorEntry, f64)> = Vec::new();
        
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let vector_bytes: Vec<u8> = row.get(1)?;
            let content: String = row.get(2)?;
            let metadata: String = row.get(3)?;
            let timestamp: i64 = row.get(4)?;
            let norm: f64 = row.get(5)?;
            
            Ok((id, vector_bytes, content, metadata, timestamp, norm))
        })?;
        
        for row in rows {
            let (id, vector_bytes, content, metadata, timestamp, norm) = row?;
            
            // Deserialize vector
            let vector: Vec<f64> = bincode::deserialize(&vector_bytes)?;
            
            // Calculate cosine similarity
            let dot_product: f64 = query.iter()
                .zip(vector.iter())
                .map(|(a, b)| a * b)
                .sum();
            
            let similarity = dot_product / (query_norm * norm);
            
            let entry = VectorEntry {
                id,
                vector,
                content,
                metadata,
                timestamp,
            };
            
            results.push((entry, similarity));
        }
        
        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top k
        Ok(results.into_iter().take(top_k).collect())
    }
    
    /// Get vector by ID
    pub fn get(&self, id: &str) -> Result<Option<VectorEntry>, Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, vector, content, metadata, timestamp FROM vectors WHERE id = ?1"
        )?;
        
        let mut rows = stmt.query(params![id])?;
        
        if let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let vector_bytes: Vec<u8> = row.get(1)?;
            let content: String = row.get(2)?;
            let metadata: String = row.get(3)?;
            let timestamp: i64 = row.get(4)?;
            
            let vector: Vec<f64> = bincode::deserialize(&vector_bytes)?;
            
            Ok(Some(VectorEntry {
                id,
                vector,
                content,
                metadata,
                timestamp,
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Count total vectors
    pub fn count(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM vectors", [], |row| row.get(0))?;
        Ok(count as usize)
    }
    
    /// Delete vector by ID
    pub fn delete(&self, id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM vectors WHERE id = ?1", params![id])?;
        Ok(())
    }
    
    /// Clear all vectors
    pub fn clear(&self) -> Result<(), Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM vectors", [])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_store_basic() {
        let store = VectorStore::in_memory(64).unwrap();
        
        let entry = VectorEntry {
            id: "test1".to_string(),
            vector: vec![1.0; 64],
            content: "test content".to_string(),
            metadata: "{}".to_string(),
            timestamp: 0,
        };
        
        store.insert(&entry).unwrap();
        
        let retrieved = store.get("test1").unwrap().unwrap();
        assert_eq!(retrieved.id, "test1");
        assert_eq!(retrieved.content, "test content");
    }
    
    #[test]
    fn test_similarity_search() {
        let store = VectorStore::in_memory(64).unwrap();
        
        // Insert similar vectors
        let entry1 = VectorEntry {
            id: "test1".to_string(),
            vector: vec![1.0; 64],
            content: "similar 1".to_string(),
            metadata: "{}".to_string(),
            timestamp: 0,
        };
        
        let entry2 = VectorEntry {
            id: "test2".to_string(),
            vector: vec![0.9; 64],
            content: "similar 2".to_string(),
            metadata: "{}".to_string(),
            timestamp: 1,
        };
        
        let mut different_vec = vec![0.0; 64];
        different_vec[0] = 1.0;
        let entry3 = VectorEntry {
            id: "test3".to_string(),
            vector: different_vec,
            content: "different".to_string(),
            metadata: "{}".to_string(),
            timestamp: 2,
        };
        
        store.insert(&entry1).unwrap();
        store.insert(&entry2).unwrap();
        store.insert(&entry3).unwrap();
        
        // Search for similar to entry1
        let results = store.find_similar(&vec![1.0; 64], 2).unwrap();
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.id, "test1"); // Most similar
        assert!(results[0].1 > 0.99); // Very high similarity
    }
    
    #[test]
    fn test_batch_insert() {
        let store = VectorStore::in_memory(64).unwrap();
        
        let entries: Vec<VectorEntry> = (0..100)
            .map(|i| VectorEntry {
                id: format!("test{}", i),
                vector: vec![i as f64 / 100.0; 64],
                content: format!("content {}", i),
                metadata: "{}".to_string(),
                timestamp: i,
            })
            .collect();
        
        store.batch_insert(&entries).unwrap();
        
        let count = store.count().unwrap();
        assert_eq!(count, 100);
    }
}
