//! Episodic Memory Module
//!
//! Stores specific past experiences and training attempts.
//! Only accepts verified (learned) paths.

use crate::core::{ThoughtState, Problem, EnergyResult};
use rusqlite::{Connection, params, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// A single episode (training attempt or interaction)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique identifier
    pub id: String,
    /// The problem that was solved
    pub problem_input: String,
    /// The candidate answer (as text representation)
    pub answer_output: String,
    /// The thought vector (serialized)
    pub thought_vector: Vec<f64>,
    /// Whether this was verified
    pub verified: bool,
    /// Confidence score
    pub confidence_score: f64,
    /// Energy at time of storage
    pub energy: f64,
    /// ID of operator that produced this
    pub operator_id: String,
    /// Timestamp of creation
    pub created_at: DateTime<Utc>,
    /// Optional tags for categorization
    pub tags: Vec<String>,
}

impl Episode {
    /// Create a new episode from a training attempt
    pub fn from_training(
        problem: &Problem,
        thought: &ThoughtState,
        energy: &EnergyResult,
        operator_id: &str,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            problem_input: problem.input.clone(),
            answer_output: problem.target_answer.clone().unwrap_or_default(),
            thought_vector: thought.vector.clone(),
            verified: energy.verified,
            confidence_score: energy.confidence_score,
            energy: energy.total,
            operator_id: operator_id.to_string(),
            created_at: Utc::now(),
            tags: Vec::new(),
        }
    }

    /// Create from a successful inference
    pub fn from_inference(
        input: &str,
        output: &str,
        thought: &ThoughtState,
        energy: &EnergyResult,
        operator_id: &str,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            problem_input: input.to_string(),
            answer_output: output.to_string(),
            thought_vector: thought.vector.clone(),
            verified: energy.verified,
            confidence_score: energy.confidence_score,
            energy: energy.total,
            operator_id: operator_id.to_string(),
            created_at: Utc::now(),
            tags: Vec::new(),
        }
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }
}

/// Episodic Memory Storage
pub struct EpisodicMemory {
    /// SQLite connection
    conn: Connection,
    /// Minimum confidence required to store
    min_confidence: f64,
    /// Only store verified episodes
    require_verified: bool,
}

impl EpisodicMemory {
    /// Create or open episodic memory database
    pub fn new<P: AsRef<Path>>(path: P) -> SqlResult<Self> {
        let conn = Connection::open(path)?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                problem_input TEXT NOT NULL,
                answer_output TEXT NOT NULL,
                thought_vector BLOB NOT NULL,
                verified INTEGER NOT NULL,
                confidence_score REAL NOT NULL,
                energy REAL NOT NULL,
                operator_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                tags TEXT
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_confidence 
             ON episodes(confidence_score DESC)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_operator 
             ON episodes(operator_id)",
            [],
        )?;

        Ok(Self {
            conn,
            min_confidence: 0.5,
            require_verified: true,
        })
    }

    /// Create in-memory episodic storage (for testing)
    pub fn in_memory() -> SqlResult<Self> {
        Self::new(":memory:")
    }

    /// Set minimum confidence threshold
    pub fn set_min_confidence(&mut self, threshold: f64) {
        self.min_confidence = threshold;
    }

    /// Set whether to require verified episodes
    pub fn set_require_verified(&mut self, require: bool) {
        self.require_verified = require;
    }

    /// Store an episode (only if it meets criteria)
    pub fn store(&self, episode: &Episode) -> SqlResult<bool> {
        // Check if episode meets storage criteria
        if self.require_verified && !episode.verified {
            return Ok(false);
        }
        if episode.confidence_score < self.min_confidence {
            return Ok(false);
        }

        let vector_bytes = bincode::serialize(&episode.thought_vector)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        
        let tags_json = serde_json::to_string(&episode.tags)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        self.conn.execute(
            "INSERT OR REPLACE INTO episodes 
             (id, problem_input, answer_output, thought_vector, verified, 
              confidence_score, energy, operator_id, created_at, tags)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                episode.id,
                episode.problem_input,
                episode.answer_output,
                vector_bytes,
                episode.verified as i32,
                episode.confidence_score,
                episode.energy,
                episode.operator_id,
                episode.created_at.to_rfc3339(),
                tags_json,
            ],
        )?;

        Ok(true)
    }

    /// Retrieve an episode by ID
    pub fn get(&self, id: &str) -> SqlResult<Option<Episode>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, problem_input, answer_output, thought_vector, verified,
                    confidence_score, energy, operator_id, created_at, tags
             FROM episodes WHERE id = ?1"
        )?;

        let mut rows = stmt.query(params![id])?;
        
        if let Some(row) = rows.next()? {
            Ok(Some(self.row_to_episode(row)?))
        } else {
            Ok(None)
        }
    }

    /// Find similar episodes by problem input
    pub fn find_similar(&self, input: &str, limit: usize) -> SqlResult<Vec<Episode>> {
        // Simple text-based similarity (in production, would use vector similarity)
        let mut stmt = self.conn.prepare(
            "SELECT id, problem_input, answer_output, thought_vector, verified,
                    confidence_score, energy, operator_id, created_at, tags
             FROM episodes 
             WHERE problem_input LIKE ?1
             ORDER BY confidence_score DESC
             LIMIT ?2"
        )?;

        let pattern = format!("%{}%", input.split_whitespace().take(3).collect::<Vec<_>>().join("%"));
        let mut rows = stmt.query(params![pattern, limit as i64])?;
        
        let mut episodes = Vec::new();
        while let Some(row) = rows.next()? {
            episodes.push(self.row_to_episode(row)?);
        }
        
        Ok(episodes)
    }

    /// Get top episodes by confidence
    pub fn get_top_episodes(&self, limit: usize) -> SqlResult<Vec<Episode>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, problem_input, answer_output, thought_vector, verified,
                    confidence_score, energy, operator_id, created_at, tags
             FROM episodes 
             WHERE verified = 1
             ORDER BY confidence_score DESC
             LIMIT ?1"
        )?;

        let mut rows = stmt.query(params![limit as i64])?;
        
        let mut episodes = Vec::new();
        while let Some(row) = rows.next()? {
            episodes.push(self.row_to_episode(row)?);
        }
        
        Ok(episodes)
    }

    /// Get episodes by operator
    pub fn get_by_operator(&self, operator_id: &str, limit: usize) -> SqlResult<Vec<Episode>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, problem_input, answer_output, thought_vector, verified,
                    confidence_score, energy, operator_id, created_at, tags
             FROM episodes 
             WHERE operator_id = ?1
             ORDER BY created_at DESC
             LIMIT ?2"
        )?;

        let mut rows = stmt.query(params![operator_id, limit as i64])?;
        
        let mut episodes = Vec::new();
        while let Some(row) = rows.next()? {
            episodes.push(self.row_to_episode(row)?);
        }
        
        Ok(episodes)
    }

    /// Get statistics about stored episodes
    pub fn get_statistics(&self) -> SqlResult<EpisodeStatistics> {
        let total: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM episodes",
            [],
            |row| row.get(0),
        )?;

        let verified: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM episodes WHERE verified = 1",
            [],
            |row| row.get(0),
        )?;

        let avg_confidence: f64 = self.conn.query_row(
            "SELECT AVG(confidence_score) FROM episodes WHERE verified = 1",
            [],
            |row| row.get(0),
        ).unwrap_or(0.0);

        let avg_energy: f64 = self.conn.query_row(
            "SELECT AVG(energy) FROM episodes WHERE verified = 1",
            [],
            |row| row.get(0),
        ).unwrap_or(0.0);

        Ok(EpisodeStatistics {
            total_episodes: total as usize,
            verified_episodes: verified as usize,
            average_confidence: avg_confidence,
            average_energy: avg_energy,
        })
    }

    /// Clear all episodes
    pub fn clear(&self) -> SqlResult<()> {
        self.conn.execute("DELETE FROM episodes", [])?;
        Ok(())
    }

    /// Helper to convert a row to Episode
    fn row_to_episode(&self, row: &rusqlite::Row) -> SqlResult<Episode> {
        let vector_bytes: Vec<u8> = row.get(3)?;
        let thought_vector: Vec<f64> = bincode::deserialize(&vector_bytes)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                3, rusqlite::types::Type::Blob, Box::new(e)
            ))?;
        
        let tags_json: String = row.get(9)?;
        let tags: Vec<String> = serde_json::from_str(&tags_json)
            .unwrap_or_default();

        let created_str: String = row.get(8)?;
        let created_at = DateTime::parse_from_rfc3339(&created_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        Ok(Episode {
            id: row.get(0)?,
            problem_input: row.get(1)?,
            answer_output: row.get(2)?,
            thought_vector,
            verified: row.get::<_, i32>(4)? != 0,
            confidence_score: row.get(5)?,
            energy: row.get(6)?,
            operator_id: row.get(7)?,
            created_at,
            tags,
        })
    }
}

/// Statistics about episodic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeStatistics {
    pub total_episodes: usize,
    pub verified_episodes: usize,
    pub average_confidence: f64,
    pub average_energy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_episode_creation() {
        let problem = Problem::training("2+2", "4", 64);
        let thought = ThoughtState::from_input("4", 64);
        let energy = EnergyResult {
            total: 0.2,
            constraint_energy: 0.1,
            risk_energy: 0.05,
            uncertainty_energy: 0.05,
            verified: true,
            confidence_score: 0.8,
        };

        let episode = Episode::from_training(&problem, &thought, &energy, "op1");
        assert!(episode.verified);
        assert!((episode.confidence_score - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_episodic_memory_storage() {
        let memory = EpisodicMemory::in_memory().unwrap();
        
        let episode = Episode {
            id: "test-1".to_string(),
            problem_input: "2+2".to_string(),
            answer_output: "4".to_string(),
            thought_vector: vec![0.1; 64],
            verified: true,
            confidence_score: 0.8,
            energy: 0.2,
            operator_id: "op1".to_string(),
            created_at: Utc::now(),
            tags: vec!["math".to_string()],
        };

        let stored = memory.store(&episode).unwrap();
        assert!(stored);

        let retrieved = memory.get("test-1").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().problem_input, "2+2");
    }

    #[test]
    fn test_verified_only_storage() {
        let memory = EpisodicMemory::in_memory().unwrap();
        
        let unverified = Episode {
            id: "test-2".to_string(),
            problem_input: "test".to_string(),
            answer_output: "answer".to_string(),
            thought_vector: vec![0.1; 64],
            verified: false,
            confidence_score: 0.3,
            energy: 0.7,
            operator_id: "op1".to_string(),
            created_at: Utc::now(),
            tags: vec![],
        };

        let stored = memory.store(&unverified).unwrap();
        assert!(!stored); // Should not be stored
    }
}
