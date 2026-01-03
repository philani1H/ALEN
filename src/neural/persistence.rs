//! Neural Network Persistence Layer
//!
//! Provides database-backed persistence for:
//! - Controller (φ) patterns and weights
//! - Core Model (θ) weights
//! - Episodic memory (vector database)
//! - Training statistics and checkpoints
//!
//! Uses SQLite for relational data and vector storage
//! Ensures training continuity across sessions and deployments

use rusqlite::{Connection, params, Result as SqlResult};
use serde::{Serialize, Deserialize};
use std::path::Path;
use chrono::{DateTime, Utc};
use anyhow::{Result, Context};

// ============================================================================
// DATABASE SCHEMA
// ============================================================================

/// Initialize database schema
pub fn init_database(db_path: &Path) -> Result<Connection> {
    let conn = Connection::open(db_path)
        .context("Failed to open database")?;

    // Create tables
    conn.execute_batch(
        r#"
        -- Controller patterns (φ parameters)
        CREATE TABLE IF NOT EXISTS controller_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id INTEGER NOT NULL,
            weights BLOB NOT NULL,
            bias BLOB NOT NULL,
            active BOOLEAN NOT NULL DEFAULT 1,
            usage_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        -- Core model weights (θ parameters)
        CREATE TABLE IF NOT EXISTS core_model_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            layer_name TEXT NOT NULL UNIQUE,
            weights BLOB NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        -- Episodic memory (vector database)
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context_vector BLOB NOT NULL,
            response TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        );

        -- Create index for fast vector retrieval
        CREATE INDEX IF NOT EXISTS idx_episodic_confidence
        ON episodic_memory(confidence DESC);

        CREATE INDEX IF NOT EXISTS idx_episodic_created
        ON episodic_memory(created_at DESC);

        -- Training checkpoints
        CREATE TABLE IF NOT EXISTS training_checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_name TEXT NOT NULL UNIQUE,
            total_training_steps INTEGER NOT NULL,
            controller_updates INTEGER NOT NULL,
            core_model_updates INTEGER NOT NULL,
            avg_confidence REAL NOT NULL,
            avg_perplexity REAL NOT NULL,
            controller_lr REAL NOT NULL,
            core_lr REAL NOT NULL,
            created_at TEXT NOT NULL
        );

        -- Training history (for analytics)
        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            target_text TEXT NOT NULL,
            generation_loss REAL NOT NULL,
            controller_loss REAL NOT NULL,
            total_loss REAL NOT NULL,
            confidence REAL NOT NULL,
            perplexity REAL NOT NULL,
            created_at TEXT NOT NULL
        );

        -- System metadata
        CREATE TABLE IF NOT EXISTS system_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        "#
    ).context("Failed to create database schema")?;

    Ok(conn)
}

// ============================================================================
// PERSISTENCE MANAGER
// ============================================================================

pub struct NeuralPersistence {
    conn: Connection,
}

impl NeuralPersistence {
    /// Create new persistence manager
    pub fn new(db_path: &Path) -> Result<Self> {
        let conn = init_database(db_path)?;
        Ok(Self { conn })
    }

    // ========================================================================
    // CONTROLLER PATTERNS (φ)
    // ========================================================================

    /// Save controller pattern
    pub fn save_controller_pattern(
        &self,
        pattern_id: usize,
        weights: &[f64],
        bias: &[f64],
        active: bool,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let weights_bytes = bincode::serialize(weights)?;
        let bias_bytes = bincode::serialize(bias)?;

        self.conn.execute(
            "INSERT OR REPLACE INTO controller_patterns
             (pattern_id, weights, bias, active, usage_count, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4,
                     COALESCE((SELECT usage_count FROM controller_patterns WHERE pattern_id = ?1), 0),
                     COALESCE((SELECT created_at FROM controller_patterns WHERE pattern_id = ?1), ?5),
                     ?5)",
            params![pattern_id as i64, weights_bytes, bias_bytes, active, now],
        )?;

        Ok(())
    }

    /// Load controller pattern
    pub fn load_controller_pattern(&self, pattern_id: usize) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
        let result = self.conn.query_row(
            "SELECT weights, bias FROM controller_patterns WHERE pattern_id = ?1 AND active = 1",
            params![pattern_id as i64],
            |row| {
                let weights_bytes: Vec<u8> = row.get(0)?;
                let bias_bytes: Vec<u8> = row.get(1)?;
                Ok((weights_bytes, bias_bytes))
            },
        );

        match result {
            Ok((w, b)) => {
                let weights: Vec<f64> = bincode::deserialize(&w)?;
                let bias: Vec<f64> = bincode::deserialize(&b)?;
                Ok(Some((weights, bias)))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Increment pattern usage count
    pub fn increment_pattern_usage(&self, pattern_id: usize) -> Result<()> {
        self.conn.execute(
            "UPDATE controller_patterns SET usage_count = usage_count + 1 WHERE pattern_id = ?1",
            params![pattern_id as i64],
        )?;
        Ok(())
    }

    // ========================================================================
    // CORE MODEL WEIGHTS (θ)
    // ========================================================================

    /// Save core model layer weights
    pub fn save_model_weights(&self, layer_name: &str, weights: &[f64]) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let weights_bytes = bincode::serialize(weights)?;

        self.conn.execute(
            "INSERT OR REPLACE INTO core_model_weights (layer_name, weights, created_at, updated_at)
             VALUES (?1, ?2,
                     COALESCE((SELECT created_at FROM core_model_weights WHERE layer_name = ?1), ?3),
                     ?3)",
            params![layer_name, weights_bytes, now],
        )?;

        Ok(())
    }

    /// Load core model layer weights
    pub fn load_model_weights(&self, layer_name: &str) -> Result<Option<Vec<f64>>> {
        let result = self.conn.query_row(
            "SELECT weights FROM core_model_weights WHERE layer_name = ?1",
            params![layer_name],
            |row| {
                let weights_bytes: Vec<u8> = row.get(0)?;
                Ok(weights_bytes)
            },
        );

        match result {
            Ok(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // ========================================================================
    // EPISODIC MEMORY (Vector Database)
    // ========================================================================

    /// Store episodic memory entry
    pub fn store_memory(
        &self,
        context_vector: &[f64],
        response: &str,
        confidence: f64,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        let vector_bytes = bincode::serialize(context_vector)?;

        self.conn.execute(
            "INSERT INTO episodic_memory (context_vector, response, confidence, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![vector_bytes, response, confidence, now],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Load recent N memories from database (for initialization)
    pub fn load_recent_memories(&self, limit: usize) -> Result<Vec<(Vec<f64>, String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT context_vector, response, confidence
             FROM episodic_memory
             ORDER BY created_at DESC
             LIMIT ?"
        )?;

        let entries = stmt.query_map([limit as i64], |row| {
            let vector_bytes: Vec<u8> = row.get(0)?;
            let response: String = row.get(1)?;
            let confidence: f64 = row.get(2)?;
            Ok((vector_bytes, response, confidence))
        })?;

        let mut memories = Vec::new();
        for entry_result in entries {
            let (vector_bytes, response, confidence) = entry_result?;
            let context_vector: Vec<f64> = bincode::deserialize(&vector_bytes)?;
            memories.push((context_vector, response, confidence));
        }

        Ok(memories)
    }

    /// Retrieve top-k similar memories using cosine similarity
    pub fn retrieve_memories(
        &self,
        query_vector: &[f64],
        top_k: usize,
        min_similarity: f64,
    ) -> Result<Vec<MemoryEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, context_vector, response, confidence, created_at
             FROM episodic_memory
             ORDER BY created_at DESC
             LIMIT 1000"  // Get recent memories for similarity search
        )?;

        let entries = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let vector_bytes: Vec<u8> = row.get(1)?;
            let response: String = row.get(2)?;
            let confidence: f64 = row.get(3)?;
            let created_at: String = row.get(4)?;

            Ok((id, vector_bytes, response, confidence, created_at))
        })?;

        // Compute similarities and rank
        let mut scored_entries = Vec::new();

        for entry_result in entries {
            let (id, vector_bytes, response, confidence, created_at) = entry_result?;
            let context_vector: Vec<f64> = bincode::deserialize(&vector_bytes)?;

            let similarity = cosine_similarity(query_vector, &context_vector);

            if similarity >= min_similarity {
                scored_entries.push((similarity, MemoryEntry {
                    id,
                    context_vector,
                    response,
                    confidence,
                    created_at,
                }));
            }
        }

        // Sort by similarity descending
        scored_entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Return top-k
        Ok(scored_entries.into_iter()
            .take(top_k)
            .map(|(_, entry)| entry)
            .collect())
    }

    /// Get total memory count
    pub fn get_memory_count(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM episodic_memory",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Prune old memories (keep only most recent N)
    pub fn prune_memories(&self, keep_count: usize) -> Result<usize> {
        let deleted = self.conn.execute(
            "DELETE FROM episodic_memory
             WHERE id NOT IN (
                 SELECT id FROM episodic_memory
                 ORDER BY created_at DESC
                 LIMIT ?1
             )",
            params![keep_count as i64],
        )?;

        Ok(deleted)
    }

    // ========================================================================
    // TRAINING CHECKPOINTS
    // ========================================================================

    /// Save training checkpoint
    pub fn save_checkpoint(&self, checkpoint: &TrainingCheckpoint) -> Result<()> {
        let now = Utc::now().to_rfc3339();

        self.conn.execute(
            "INSERT OR REPLACE INTO training_checkpoints
             (checkpoint_name, total_training_steps, controller_updates, core_model_updates,
              avg_confidence, avg_perplexity, controller_lr, core_lr, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                checkpoint.name,
                checkpoint.total_training_steps as i64,
                checkpoint.controller_updates as i64,
                checkpoint.core_model_updates as i64,
                checkpoint.avg_confidence,
                checkpoint.avg_perplexity,
                checkpoint.controller_lr,
                checkpoint.core_lr,
                now,
            ],
        )?;

        Ok(())
    }

    /// Load latest checkpoint
    pub fn load_latest_checkpoint(&self) -> Result<Option<TrainingCheckpoint>> {
        let result = self.conn.query_row(
            "SELECT checkpoint_name, total_training_steps, controller_updates, core_model_updates,
                    avg_confidence, avg_perplexity, controller_lr, core_lr
             FROM training_checkpoints
             ORDER BY created_at DESC
             LIMIT 1",
            [],
            |row| {
                Ok(TrainingCheckpoint {
                    name: row.get(0)?,
                    total_training_steps: row.get::<_, i64>(1)? as u64,
                    controller_updates: row.get::<_, i64>(2)? as u64,
                    core_model_updates: row.get::<_, i64>(3)? as u64,
                    avg_confidence: row.get(4)?,
                    avg_perplexity: row.get(5)?,
                    controller_lr: row.get(6)?,
                    core_lr: row.get(7)?,
                })
            },
        );

        match result {
            Ok(checkpoint) => Ok(Some(checkpoint)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // ========================================================================
    // TRAINING HISTORY
    // ========================================================================

    /// Log training step
    pub fn log_training_step(
        &self,
        input: &str,
        target: &str,
        generation_loss: f64,
        controller_loss: f64,
        total_loss: f64,
        confidence: f64,
        perplexity: f64,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();

        self.conn.execute(
            "INSERT INTO training_history
             (input_text, target_text, generation_loss, controller_loss, total_loss,
              confidence, perplexity, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![input, target, generation_loss, controller_loss, total_loss,
                    confidence, perplexity, now],
        )?;

        Ok(())
    }

    /// Get training statistics
    pub fn get_training_stats(&self, last_n_steps: usize) -> Result<TrainingStats> {
        let stats = self.conn.query_row(
            "SELECT
                COUNT(*) as count,
                AVG(total_loss) as avg_loss,
                AVG(confidence) as avg_confidence,
                AVG(perplexity) as avg_perplexity
             FROM (
                 SELECT * FROM training_history
                 ORDER BY created_at DESC
                 LIMIT ?1
             )",
            params![last_n_steps as i64],
            |row| {
                Ok(TrainingStats {
                    total_steps: row.get::<_, i64>(0)? as usize,
                    avg_loss: row.get(1)?,
                    avg_confidence: row.get(2)?,
                    avg_perplexity: row.get(3)?,
                })
            },
        )?;

        Ok(stats)
    }

    // ========================================================================
    // SYSTEM METADATA
    // ========================================================================

    /// Set metadata value
    pub fn set_metadata(&self, key: &str, value: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();

        self.conn.execute(
            "INSERT OR REPLACE INTO system_metadata (key, value, updated_at)
             VALUES (?1, ?2, ?3)",
            params![key, value, now],
        )?;

        Ok(())
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Result<Option<String>> {
        let result = self.conn.query_row(
            "SELECT value FROM system_metadata WHERE key = ?1",
            params![key],
            |row| row.get(0),
        );

        match result {
            Ok(value) => Ok(Some(value)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub id: i64,
    pub context_vector: Vec<f64>,
    pub response: String,
    pub confidence: f64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub name: String,
    pub total_training_steps: u64,
    pub controller_updates: u64,
    pub core_model_updates: u64,
    pub avg_confidence: f64,
    pub avg_perplexity: f64,
    pub controller_lr: f64,
    pub core_lr: f64,
}

#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub total_steps: usize,
    pub avg_loss: f64,
    pub avg_confidence: f64,
    pub avg_perplexity: f64,
}

// ============================================================================
// UTILITIES
// ============================================================================

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_persistence() -> Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("test.db");
        let persistence = NeuralPersistence::new(&db_path)?;

        // Test controller pattern
        let weights = vec![1.0, 2.0, 3.0];
        let bias = vec![0.1, 0.2];
        persistence.save_controller_pattern(0, &weights, &bias, true)?;

        let loaded = persistence.load_controller_pattern(0)?;
        assert!(loaded.is_some());
        let (w, b) = loaded.unwrap();
        assert_eq!(w, weights);
        assert_eq!(b, bias);

        // Test memory storage
        let context = vec![0.5, 0.5, 0.5];
        persistence.store_memory(&context, "test response", 0.9)?;

        let count = persistence.get_memory_count()?;
        assert_eq!(count, 1);

        Ok(())
    }
}
