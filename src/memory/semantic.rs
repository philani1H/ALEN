//! Semantic Memory Module
//!
//! Stores general knowledge and verified facts.
//! This is the long-term knowledge base.

use crate::core::ThoughtState;
use rusqlite::{Connection, params, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// A semantic fact or concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFact {
    /// Unique identifier
    pub id: String,
    /// The concept or topic
    pub concept: String,
    /// The fact or knowledge content
    pub content: String,
    /// Embedding vector for similarity search
    pub embedding: Vec<f64>,
    /// Confidence in this fact (0.0 - 1.0)
    pub confidence: f64,
    /// Number of times this fact was reinforced
    pub reinforcement_count: u32,
    /// Last time this fact was accessed
    pub last_accessed: DateTime<Utc>,
    /// Source of this fact
    pub source: Option<String>,
    /// Category/domain
    pub category: Option<String>,
    /// Related concept IDs
    pub related_concepts: Vec<String>,
}

impl SemanticFact {
    /// Create a new semantic fact
    pub fn new(concept: &str, content: &str, dimension: usize) -> Self {
        let embedding = ThoughtState::from_input(
            &format!("{} {}", concept, content),
            dimension,
        ).vector;

        Self {
            id: Uuid::new_v4().to_string(),
            concept: concept.to_string(),
            content: content.to_string(),
            embedding,
            confidence: 0.5, // Default confidence
            reinforcement_count: 1,
            last_accessed: Utc::now(),
            source: None,
            category: None,
            related_concepts: Vec::new(),
        }
    }

    /// Create with source
    pub fn with_source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }

    /// Create with category
    pub fn with_category(mut self, category: &str) -> Self {
        self.category = Some(category.to_string());
        self
    }

    /// Add a related concept
    pub fn with_related(mut self, concept_id: &str) -> Self {
        self.related_concepts.push(concept_id.to_string());
        self
    }

    /// Reinforce this fact (increase confidence)
    pub fn reinforce(&mut self) {
        self.reinforcement_count += 1;
        // Asymptotic increase towards 1.0
        self.confidence = 1.0 - (1.0 - self.confidence) * 0.9;
        self.last_accessed = Utc::now();
    }

    /// Calculate similarity to a query vector
    pub fn similarity_to(&self, query: &[f64]) -> f64 {
        if self.embedding.len() != query.len() {
            return 0.0;
        }

        let dot: f64 = self.embedding.iter()
            .zip(query.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_self: f64 = self.embedding.iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        let norm_query: f64 = query.iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        if norm_self > 0.0 && norm_query > 0.0 {
            dot / (norm_self * norm_query)
        } else {
            0.0
        }
    }
}

/// Semantic Memory Storage
pub struct SemanticMemory {
    /// SQLite connection
    conn: Connection,
    /// Vector dimension
    dimension: usize,
}

impl SemanticMemory {
    /// Create or open semantic memory database
    pub fn new<P: AsRef<Path>>(path: P, dimension: usize) -> SqlResult<Self> {
        let conn = Connection::open(path)?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                concept TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                confidence REAL NOT NULL,
                reinforcement_count INTEGER NOT NULL,
                last_accessed TEXT NOT NULL,
                source TEXT,
                category TEXT,
                related_concepts TEXT
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_concept ON facts(concept)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence DESC)",
            [],
        )?;

        Ok(Self { conn, dimension })
    }

    /// Create in-memory semantic storage
    pub fn in_memory(dimension: usize) -> SqlResult<Self> {
        Self::new(":memory:", dimension)
    }

    /// Store a semantic fact
    pub fn store(&self, fact: &SemanticFact) -> SqlResult<()> {
        let embedding_bytes = bincode::serialize(&fact.embedding)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        let related_json = serde_json::to_string(&fact.related_concepts)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        self.conn.execute(
            "INSERT OR REPLACE INTO facts 
             (id, concept, content, embedding, confidence, reinforcement_count,
              last_accessed, source, category, related_concepts)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                fact.id,
                fact.concept,
                fact.content,
                embedding_bytes,
                fact.confidence,
                fact.reinforcement_count,
                fact.last_accessed.to_rfc3339(),
                fact.source,
                fact.category,
                related_json,
            ],
        )?;

        Ok(())
    }

    /// Get a fact by ID
    pub fn get(&self, id: &str) -> SqlResult<Option<SemanticFact>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, concept, content, embedding, confidence, reinforcement_count,
                    last_accessed, source, category, related_concepts
             FROM facts WHERE id = ?1"
        )?;

        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            Ok(Some(self.row_to_fact(row)?))
        } else {
            Ok(None)
        }
    }

    /// Search facts by concept
    pub fn search_by_concept(&self, concept: &str, limit: usize) -> SqlResult<Vec<SemanticFact>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, concept, content, embedding, confidence, reinforcement_count,
                    last_accessed, source, category, related_concepts
             FROM facts 
             WHERE concept LIKE ?1
             ORDER BY confidence DESC
             LIMIT ?2"
        )?;

        let pattern = format!("%{}%", concept);
        let mut rows = stmt.query(params![pattern, limit as i64])?;

        let mut facts = Vec::new();
        while let Some(row) = rows.next()? {
            facts.push(self.row_to_fact(row)?);
        }

        Ok(facts)
    }

    /// Search facts by category
    pub fn search_by_category(&self, category: &str, limit: usize) -> SqlResult<Vec<SemanticFact>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, concept, content, embedding, confidence, reinforcement_count,
                    last_accessed, source, category, related_concepts
             FROM facts 
             WHERE category = ?1
             ORDER BY confidence DESC
             LIMIT ?2"
        )?;

        let mut rows = stmt.query(params![category, limit as i64])?;

        let mut facts = Vec::new();
        while let Some(row) = rows.next()? {
            facts.push(self.row_to_fact(row)?);
        }

        Ok(facts)
    }

    /// Find similar facts using vector similarity
    pub fn find_similar(&self, query_vector: &[f64], limit: usize) -> SqlResult<Vec<(SemanticFact, f64)>> {
        // Load all facts and compute similarity
        // In production, would use vector database or approximate nearest neighbors
        let mut stmt = self.conn.prepare(
            "SELECT id, concept, content, embedding, confidence, reinforcement_count,
                    last_accessed, source, category, related_concepts
             FROM facts"
        )?;

        let mut rows = stmt.query([])?;
        let mut facts_with_similarity = Vec::new();

        while let Some(row) = rows.next()? {
            let fact = self.row_to_fact(row)?;
            let similarity = fact.similarity_to(query_vector);
            facts_with_similarity.push((fact, similarity));
        }

        // Sort by similarity descending
        facts_with_similarity.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top limit
        facts_with_similarity.truncate(limit);

        Ok(facts_with_similarity)
    }

    /// Find similar facts by text query
    pub fn find_similar_by_text(&self, query: &str, limit: usize) -> SqlResult<Vec<(SemanticFact, f64)>> {
        let query_state = ThoughtState::from_input(query, self.dimension);
        self.find_similar(&query_state.vector, limit)
    }

    /// Get all facts (for export)
    pub fn get_all_facts(&self, limit: usize) -> SqlResult<Vec<SemanticFact>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, concept, content, embedding, confidence, reinforcement_count,
                    last_accessed, source, category, related_concepts
             FROM facts
             ORDER BY created_at DESC
             LIMIT ?1"
        )?;

        let mut rows = stmt.query(params![limit as i64])?;

        let mut facts = Vec::new();
        while let Some(row) = rows.next()? {
            facts.push(self.row_to_fact(row)?);
        }

        Ok(facts)
    }

    /// Reinforce a fact (increase confidence)
    pub fn reinforce(&self, id: &str) -> SqlResult<bool> {
        let result = self.conn.execute(
            "UPDATE facts SET 
                reinforcement_count = reinforcement_count + 1,
                confidence = 1.0 - (1.0 - confidence) * 0.9,
                last_accessed = ?1
             WHERE id = ?2",
            params![Utc::now().to_rfc3339(), id],
        )?;

        Ok(result > 0)
    }

    /// Get facts related to a given fact
    pub fn get_related(&self, id: &str) -> SqlResult<Vec<SemanticFact>> {
        let fact = self.get(id)?;
        
        if let Some(f) = fact {
            let mut related = Vec::new();
            for related_id in &f.related_concepts {
                if let Some(rel_fact) = self.get(related_id)? {
                    related.push(rel_fact);
                }
            }
            Ok(related)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get statistics
    pub fn get_statistics(&self) -> SqlResult<SemanticStatistics> {
        let total: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM facts",
            [],
            |row| row.get(0),
        )?;

        let avg_confidence: f64 = self.conn.query_row(
            "SELECT AVG(confidence) FROM facts",
            [],
            |row| row.get(0),
        ).unwrap_or(0.0);

        let categories: Vec<String> = {
            let mut stmt = self.conn.prepare(
                "SELECT DISTINCT category FROM facts WHERE category IS NOT NULL"
            )?;
            let mut rows = stmt.query([])?;
            let mut cats = Vec::new();
            while let Some(row) = rows.next()? {
                cats.push(row.get(0)?);
            }
            cats
        };

        Ok(SemanticStatistics {
            total_facts: total as usize,
            average_confidence: avg_confidence,
            categories,
        })
    }

    /// Clear all facts
    pub fn clear(&self) -> SqlResult<()> {
        self.conn.execute("DELETE FROM facts", [])?;
        Ok(())
    }

    /// Helper to convert row to SemanticFact
    fn row_to_fact(&self, row: &rusqlite::Row) -> SqlResult<SemanticFact> {
        let embedding_bytes: Vec<u8> = row.get(3)?;
        let embedding: Vec<f64> = bincode::deserialize(&embedding_bytes)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                3, rusqlite::types::Type::Blob, Box::new(e)
            ))?;

        let related_json: String = row.get(9)?;
        let related_concepts: Vec<String> = serde_json::from_str(&related_json)
            .unwrap_or_default();

        let last_accessed_str: String = row.get(6)?;
        let last_accessed = DateTime::parse_from_rfc3339(&last_accessed_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        Ok(SemanticFact {
            id: row.get(0)?,
            concept: row.get(1)?,
            content: row.get(2)?,
            embedding,
            confidence: row.get(4)?,
            reinforcement_count: row.get(5)?,
            last_accessed,
            source: row.get(7)?,
            category: row.get(8)?,
            related_concepts,
        })
    }
}

/// Statistics about semantic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticStatistics {
    pub total_facts: usize,
    pub average_confidence: f64,
    pub categories: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_fact_creation() {
        let fact = SemanticFact::new("math", "2 + 2 = 4", 64);
        assert_eq!(fact.concept, "math");
        assert_eq!(fact.embedding.len(), 64);
    }

    #[test]
    fn test_semantic_memory_storage() {
        let memory = SemanticMemory::in_memory(64).unwrap();
        
        let fact = SemanticFact::new("math", "2 + 2 = 4", 64)
            .with_category("arithmetic")
            .with_source("basic math");

        memory.store(&fact).unwrap();

        let retrieved = memory.get(&fact.id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().concept, "math");
    }

    #[test]
    fn test_similarity_search() {
        let memory = SemanticMemory::in_memory(64).unwrap();
        
        memory.store(&SemanticFact::new("addition", "1 + 1 = 2", 64)).unwrap();
        memory.store(&SemanticFact::new("multiplication", "2 * 3 = 6", 64)).unwrap();
        memory.store(&SemanticFact::new("history", "World War 2", 64)).unwrap();

        let results = memory.find_similar_by_text("math addition", 2).unwrap();
        assert!(!results.is_empty());
    }
}
