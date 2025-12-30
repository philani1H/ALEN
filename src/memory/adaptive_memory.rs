//! Adaptive Memory System - Three-Layer Architecture
//!
//! This implements the complete memory architecture:
//! 1. Semantic Memory (Lance) - "What does this mean?"
//! 2. Learning Memory (SQLite) - "What went wrong, and what did I learn?"
//! 3. Control & State (SQLite) - "How should I respond?"
//!
//! This is NOT a lookup table. It's a pattern recognition and learning system.

use rusqlite::{Connection, params, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

// =============================================================================
// 1. SEMANTIC MEMORY - "What does this mean?"
// =============================================================================

/// Semantic unit type - categorizes understanding
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SemanticUnitType {
    Concept,      // Abstract idea
    Fact,         // Concrete fact
    Procedure,    // How to do something
    Formula,      // Mathematical formula
    CodeBlock,    // Programming code
    StoryEvent,   // Narrative element
    Pattern,      // Learned pattern
}

impl SemanticUnitType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "concept" => Self::Concept,
            "fact" => Self::Fact,
            "procedure" => Self::Procedure,
            "formula" => Self::Formula,
            "code_block" | "code" => Self::CodeBlock,
            "story_event" | "story" => Self::StoryEvent,
            "pattern" => Self::Pattern,
            _ => Self::Concept,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Concept => "concept",
            Self::Fact => "fact",
            Self::Procedure => "procedure",
            Self::Formula => "formula",
            Self::CodeBlock => "code_block",
            Self::StoryEvent => "story_event",
            Self::Pattern => "pattern",
        }
    }
}

/// Domain classification for semantic units
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SemanticDomain {
    Math,
    Programming,
    Geography,
    Science,
    History,
    Language,
    Narrative,
    General,
}

impl SemanticDomain {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "math" | "mathematics" => Self::Math,
            "programming" | "code" | "coding" => Self::Programming,
            "geography" | "geo" => Self::Geography,
            "science" | "physics" | "chemistry" | "biology" => Self::Science,
            "history" => Self::History,
            "language" | "linguistics" | "grammar" => Self::Language,
            "narrative" | "story" | "fiction" => Self::Narrative,
            _ => Self::General,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Math => "math",
            Self::Programming => "programming",
            Self::Geography => "geography",
            Self::Science => "science",
            Self::History => "history",
            Self::Language => "language",
            Self::Narrative => "narrative",
            Self::General => "general",
        }
    }
}

/// Semantic unit - a piece of understanding in latent space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticUnit {
    pub id: String,
    pub embedding: Vec<f64>,
    pub content: String,
    pub unit_type: SemanticUnitType,
    pub domain: SemanticDomain,
    pub source_id: Option<String>,
    pub confidence: f64,
    pub abstraction_level: u8,  // 0=raw, 3=high-level
    pub created_at: DateTime<Utc>,
    pub version: u32,
}

impl SemanticUnit {
    pub fn new(
        content: &str,
        embedding: Vec<f64>,
        unit_type: SemanticUnitType,
        domain: SemanticDomain,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            embedding,
            content: content.to_string(),
            unit_type,
            domain,
            source_id: None,
            confidence: 0.5,
            abstraction_level: 0,
            created_at: Utc::now(),
            version: 1,
        }
    }

    /// Calculate cosine similarity to another embedding
    pub fn similarity_to(&self, other: &[f64]) -> f64 {
        if self.embedding.len() != other.len() {
            return 0.0;
        }

        let dot: f64 = self.embedding.iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_self: f64 = self.embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_other: f64 = other.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_self > 1e-10 && norm_other > 1e-10 {
            dot / (norm_self * norm_other)
        } else {
            0.0
        }
    }
}

/// Source tracking for semantic units
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSource {
    pub source_id: String,
    pub source_type: String,  // "pdf", "book", "web", "user"
    pub title: String,
    pub author: Option<String>,
    pub hash: String,
    pub version: u32,
}

// =============================================================================
// 2. LEARNING MEMORY - "What went wrong, and what did I learn?"
// =============================================================================

/// Error type classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ErrorType {
    RetrievalError,    // Retrieved wrong information
    ReasoningError,    // Logical reasoning failed
    ConfidenceError,   // Over/under confident
    GenerationError,   // Generated nonsense
    VerificationError, // Failed verification
}

impl ErrorType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "retrieval" | "retrieval_error" => Self::RetrievalError,
            "reasoning" | "reasoning_error" => Self::ReasoningError,
            "confidence" | "confidence_error" => Self::ConfidenceError,
            "generation" | "generation_error" => Self::GenerationError,
            "verification" | "verification_error" => Self::VerificationError,
            _ => Self::ReasoningError,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RetrievalError => "retrieval_error",
            Self::ReasoningError => "reasoning_error",
            Self::ConfidenceError => "confidence_error",
            Self::GenerationError => "generation_error",
            Self::VerificationError => "verification_error",
        }
    }
}

/// Error log entry - tracks what went wrong
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorLogEntry {
    pub error_id: String,
    pub input_pattern: String,
    pub wrong_output: String,
    pub correct_output: Option<String>,
    pub error_type: ErrorType,
    pub fix_applied: Option<String>,
    pub confidence_delta: f64,
    pub timestamp: DateTime<Utc>,
    pub input_embedding: Vec<f64>,
}

impl ErrorLogEntry {
    pub fn new(
        input_pattern: &str,
        wrong_output: &str,
        error_type: ErrorType,
        input_embedding: Vec<f64>,
    ) -> Self {
        Self {
            error_id: Uuid::new_v4().to_string(),
            input_pattern: input_pattern.to_string(),
            wrong_output: wrong_output.to_string(),
            correct_output: None,
            error_type,
            fix_applied: None,
            confidence_delta: -0.1, // Default penalty
            timestamp: Utc::now(),
            input_embedding,
        }
    }

    pub fn with_correction(mut self, correct_output: &str) -> Self {
        self.correct_output = Some(correct_output.to_string());
        self
    }
}

/// Pattern type for confidence tracking
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PatternType {
    Math,
    Logic,
    Language,
    Code,
    Factual,
    Creative,
}

impl PatternType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "math" | "mathematics" => Self::Math,
            "logic" | "reasoning" => Self::Logic,
            "language" | "linguistics" => Self::Language,
            "code" | "programming" => Self::Code,
            "factual" | "fact" => Self::Factual,
            "creative" | "creativity" => Self::Creative,
            _ => Self::Logic,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Math => "math",
            Self::Logic => "logic",
            Self::Language => "language",
            Self::Code => "code",
            Self::Factual => "factual",
            Self::Creative => "creative",
        }
    }
}

/// Pattern confidence tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfidence {
    pub pattern_id: String,
    pub pattern_type: PatternType,
    pub confidence: f64,
    pub usage_count: u32,
    pub success_count: u32,
    pub last_updated: DateTime<Utc>,
}

impl PatternConfidence {
    pub fn new(pattern_type: PatternType) -> Self {
        Self {
            pattern_id: Uuid::new_v4().to_string(),
            pattern_type,
            confidence: 0.5,
            usage_count: 0,
            success_count: 0,
            last_updated: Utc::now(),
        }
    }

    /// Update confidence based on result
    pub fn update(&mut self, success: bool) {
        self.usage_count += 1;
        if success {
            self.success_count += 1;
        }
        
        // Calculate confidence: smoothed success rate with prior
        let prior = 0.5;
        let weight = 10.0; // Prior weight
        self.confidence = (self.success_count as f64 + prior * weight) 
            / (self.usage_count as f64 + weight);
        self.last_updated = Utc::now();
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.usage_count == 0 {
            return 0.5;
        }
        self.success_count as f64 / self.usage_count as f64
    }
}

// =============================================================================
// 3. CONTROL & STATE - "How should I respond?"
// =============================================================================

/// User style preference
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum UserStyle {
    Short,
    Detailed,
    Technical,
    Simple,
}

impl UserStyle {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "short" | "brief" | "concise" => Self::Short,
            "detailed" | "verbose" | "long" => Self::Detailed,
            "technical" | "advanced" | "expert" => Self::Technical,
            "simple" | "basic" | "easy" => Self::Simple,
            _ => Self::Detailed,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Detailed => "detailed",
            Self::Technical => "technical",
            Self::Simple => "simple",
        }
    }
}

/// User expertise level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ExpertiseLevel {
    Beginner,
    Intermediate,
    Expert,
}

impl ExpertiseLevel {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "beginner" | "novice" | "basic" => Self::Beginner,
            "intermediate" | "medium" | "mid" => Self::Intermediate,
            "expert" | "advanced" | "pro" => Self::Expert,
            _ => Self::Intermediate,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Beginner => "beginner",
            Self::Intermediate => "intermediate",
            Self::Expert => "expert",
        }
    }
}

/// User state - personalization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserState {
    pub user_id: String,
    pub embedding: Vec<f64>,
    pub preferred_style: UserStyle,
    pub expertise_level: ExpertiseLevel,
    pub verbosity: f64,
    pub last_seen: DateTime<Utc>,
    pub interaction_count: u32,
    pub topics_of_interest: Vec<String>,
}

impl UserState {
    pub fn new(user_id: &str, dimension: usize) -> Self {
        Self {
            user_id: user_id.to_string(),
            embedding: vec![0.0; dimension],
            preferred_style: UserStyle::Detailed,
            expertise_level: ExpertiseLevel::Intermediate,
            verbosity: 0.5,
            last_seen: Utc::now(),
            interaction_count: 0,
            topics_of_interest: Vec::new(),
        }
    }

    /// Update user embedding based on interaction
    pub fn update_embedding(&mut self, interaction_embedding: &[f64], learning_rate: f64) {
        if self.embedding.len() != interaction_embedding.len() {
            return;
        }

        for (i, val) in interaction_embedding.iter().enumerate() {
            self.embedding[i] = (1.0 - learning_rate) * self.embedding[i] + learning_rate * val;
        }
        self.interaction_count += 1;
        self.last_seen = Utc::now();
    }
}

/// Decoder control parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderControl {
    pub context: String,
    pub verbosity: f64,       // 0.0=terse, 1.0=verbose
    pub creativity: f64,      // 0.0=factual, 1.0=creative
    pub explanation_level: f64, // 0.0=no explanation, 1.0=detailed
    pub ask_questions: bool,
    pub temperature: f64,
    pub max_tokens: usize,
}

impl Default for DecoderControl {
    fn default() -> Self {
        Self {
            context: String::new(),
            verbosity: 0.5,
            creativity: 0.3,
            explanation_level: 0.5,
            ask_questions: false,
            temperature: 0.7,
            max_tokens: 100,
        }
    }
}

impl DecoderControl {
    /// Create for factual responses
    pub fn factual() -> Self {
        Self {
            verbosity: 0.3,
            creativity: 0.0,
            explanation_level: 0.2,
            ask_questions: false,
            temperature: 0.3,
            max_tokens: 50,
            ..Default::default()
        }
    }

    /// Create for creative responses
    pub fn creative() -> Self {
        Self {
            verbosity: 0.7,
            creativity: 0.9,
            explanation_level: 0.5,
            ask_questions: true,
            temperature: 0.9,
            max_tokens: 200,
            ..Default::default()
        }
    }

    /// Create for explanatory responses
    pub fn explanatory() -> Self {
        Self {
            verbosity: 0.8,
            creativity: 0.2,
            explanation_level: 1.0,
            ask_questions: false,
            temperature: 0.5,
            max_tokens: 300,
            ..Default::default()
        }
    }
}

// =============================================================================
// ADAPTIVE MEMORY STORE - Unified Storage
// =============================================================================

/// Adaptive Memory Store - combines all three memory layers
pub struct AdaptiveMemoryStore {
    conn: Connection,
    dimension: usize,
}

impl AdaptiveMemoryStore {
    /// Create new adaptive memory store
    pub fn new<P: AsRef<Path>>(path: P, dimension: usize) -> SqlResult<Self> {
        let conn = Connection::open(path)?;
        let mut store = Self { conn, dimension };
        store.initialize_tables()?;
        Ok(store)
    }

    /// Create in-memory store
    pub fn in_memory(dimension: usize) -> SqlResult<Self> {
        Self::new(":memory:", dimension)
    }

    /// Initialize all tables
    fn initialize_tables(&mut self) -> SqlResult<()> {
        // 1. Semantic Units table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS semantic_units (
                id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                content TEXT NOT NULL,
                unit_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                source_id TEXT,
                confidence REAL NOT NULL,
                abstraction_level INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                version INTEGER NOT NULL
            )",
            [],
        )?;

        // 2. Sources table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                title TEXT NOT NULL,
                author TEXT,
                hash TEXT NOT NULL,
                version INTEGER NOT NULL
            )",
            [],
        )?;

        // 3. Error log table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS error_log (
                error_id TEXT PRIMARY KEY,
                input_pattern TEXT NOT NULL,
                wrong_output TEXT NOT NULL,
                correct_output TEXT,
                error_type TEXT NOT NULL,
                fix_applied TEXT,
                confidence_delta REAL NOT NULL,
                timestamp TEXT NOT NULL,
                input_embedding BLOB NOT NULL
            )",
            [],
        )?;

        // 4. Pattern confidence table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS pattern_confidence (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                usage_count INTEGER NOT NULL,
                success_count INTEGER NOT NULL,
                last_updated TEXT NOT NULL
            )",
            [],
        )?;

        // 5. User state table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS user_state (
                user_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                preferred_style TEXT NOT NULL,
                expertise_level TEXT NOT NULL,
                verbosity REAL NOT NULL,
                last_seen TEXT NOT NULL,
                interaction_count INTEGER NOT NULL,
                topics_of_interest TEXT NOT NULL
            )",
            [],
        )?;

        // 6. Decoder controls table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS decoder_controls (
                context TEXT PRIMARY KEY,
                verbosity REAL NOT NULL,
                creativity REAL NOT NULL,
                explanation_level REAL NOT NULL,
                ask_questions INTEGER NOT NULL,
                temperature REAL NOT NULL,
                max_tokens INTEGER NOT NULL
            )",
            [],
        )?;

        // Create indexes
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_domain ON semantic_units(domain)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_type ON semantic_units(unit_type)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_error_type ON error_log(error_type)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pattern_type ON pattern_confidence(pattern_type)",
            [],
        )?;

        Ok(())
    }

    // =========================================================================
    // SEMANTIC MEMORY OPERATIONS
    // =========================================================================

    /// Store semantic unit
    pub fn store_semantic_unit(&self, unit: &SemanticUnit) -> SqlResult<()> {
        let embedding_bytes = bincode::serialize(&unit.embedding)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        self.conn.execute(
            "INSERT OR REPLACE INTO semantic_units 
             (id, embedding, content, unit_type, domain, source_id, confidence,
              abstraction_level, created_at, version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                unit.id,
                embedding_bytes,
                unit.content,
                unit.unit_type.as_str(),
                unit.domain.as_str(),
                unit.source_id,
                unit.confidence,
                unit.abstraction_level,
                unit.created_at.to_rfc3339(),
                unit.version,
            ],
        )?;
        Ok(())
    }

    /// Find similar semantic units
    pub fn find_similar_units(&self, query: &[f64], limit: usize) -> SqlResult<Vec<(SemanticUnit, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, embedding, content, unit_type, domain, source_id, confidence,
                    abstraction_level, created_at, version
             FROM semantic_units"
        )?;

        let mut rows = stmt.query([])?;
        let mut results = Vec::new();

        while let Some(row) = rows.next()? {
            let embedding_bytes: Vec<u8> = row.get(1)?;
            let embedding: Vec<f64> = bincode::deserialize(&embedding_bytes)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                    1, rusqlite::types::Type::Blob, Box::new(e)
                ))?;

            let created_at_str: String = row.get(8)?;
            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let unit = SemanticUnit {
                id: row.get(0)?,
                embedding: embedding.clone(),
                content: row.get(2)?,
                unit_type: SemanticUnitType::from_str(&row.get::<_, String>(3)?),
                domain: SemanticDomain::from_str(&row.get::<_, String>(4)?),
                source_id: row.get(5)?,
                confidence: row.get(6)?,
                abstraction_level: row.get(7)?,
                created_at,
                version: row.get(9)?,
            };

            let similarity = unit.similarity_to(query);
            results.push((unit, similarity));
        }

        // Sort by similarity
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    /// Get semantic units by domain
    pub fn get_units_by_domain(&self, domain: SemanticDomain, limit: usize) -> SqlResult<Vec<SemanticUnit>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, embedding, content, unit_type, domain, source_id, confidence,
                    abstraction_level, created_at, version
             FROM semantic_units
             WHERE domain = ?1
             ORDER BY confidence DESC
             LIMIT ?2"
        )?;

        let mut rows = stmt.query(params![domain.as_str(), limit as i64])?;
        let mut units = Vec::new();

        while let Some(row) = rows.next()? {
            let embedding_bytes: Vec<u8> = row.get(1)?;
            let embedding: Vec<f64> = bincode::deserialize(&embedding_bytes)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                    1, rusqlite::types::Type::Blob, Box::new(e)
                ))?;

            let created_at_str: String = row.get(8)?;
            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            units.push(SemanticUnit {
                id: row.get(0)?,
                embedding,
                content: row.get(2)?,
                unit_type: SemanticUnitType::from_str(&row.get::<_, String>(3)?),
                domain: SemanticDomain::from_str(&row.get::<_, String>(4)?),
                source_id: row.get(5)?,
                confidence: row.get(6)?,
                abstraction_level: row.get(7)?,
                created_at,
                version: row.get(9)?,
            });
        }

        Ok(units)
    }

    // =========================================================================
    // LEARNING MEMORY OPERATIONS
    // =========================================================================

    /// Log an error
    pub fn log_error(&self, entry: &ErrorLogEntry) -> SqlResult<()> {
        let embedding_bytes = bincode::serialize(&entry.input_embedding)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        self.conn.execute(
            "INSERT INTO error_log 
             (error_id, input_pattern, wrong_output, correct_output, error_type,
              fix_applied, confidence_delta, timestamp, input_embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                entry.error_id,
                entry.input_pattern,
                entry.wrong_output,
                entry.correct_output,
                entry.error_type.as_str(),
                entry.fix_applied,
                entry.confidence_delta,
                entry.timestamp.to_rfc3339(),
                embedding_bytes,
            ],
        )?;
        Ok(())
    }

    /// Find similar errors (to avoid repeating mistakes)
    pub fn find_similar_errors(&self, query_embedding: &[f64], limit: usize) -> SqlResult<Vec<ErrorLogEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT error_id, input_pattern, wrong_output, correct_output, error_type,
                    fix_applied, confidence_delta, timestamp, input_embedding
             FROM error_log"
        )?;

        let mut rows = stmt.query([])?;
        let mut results: Vec<(ErrorLogEntry, f64)> = Vec::new();

        while let Some(row) = rows.next()? {
            let embedding_bytes: Vec<u8> = row.get(8)?;
            let embedding: Vec<f64> = bincode::deserialize(&embedding_bytes)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                    8, rusqlite::types::Type::Blob, Box::new(e)
                ))?;

            // Calculate similarity
            let similarity = cosine_similarity(query_embedding, &embedding);

            let timestamp_str: String = row.get(7)?;
            let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let entry = ErrorLogEntry {
                error_id: row.get(0)?,
                input_pattern: row.get(1)?,
                wrong_output: row.get(2)?,
                correct_output: row.get(3)?,
                error_type: ErrorType::from_str(&row.get::<_, String>(4)?),
                fix_applied: row.get(5)?,
                confidence_delta: row.get(6)?,
                timestamp,
                input_embedding: embedding,
            };

            results.push((entry, similarity));
        }

        // Sort by similarity
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(results.into_iter().take(limit).map(|(e, _)| e).collect())
    }

    /// Update pattern confidence
    pub fn update_pattern_confidence(&self, pattern_type: PatternType, success: bool) -> SqlResult<()> {
        // Get or create pattern confidence
        let mut pattern = self.get_pattern_confidence(pattern_type)?
            .unwrap_or_else(|| PatternConfidence::new(pattern_type));

        pattern.update(success);

        self.conn.execute(
            "INSERT OR REPLACE INTO pattern_confidence 
             (pattern_id, pattern_type, confidence, usage_count, success_count, last_updated)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                pattern.pattern_id,
                pattern.pattern_type.as_str(),
                pattern.confidence,
                pattern.usage_count,
                pattern.success_count,
                pattern.last_updated.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Get pattern confidence
    pub fn get_pattern_confidence(&self, pattern_type: PatternType) -> SqlResult<Option<PatternConfidence>> {
        let mut stmt = self.conn.prepare(
            "SELECT pattern_id, pattern_type, confidence, usage_count, success_count, last_updated
             FROM pattern_confidence
             WHERE pattern_type = ?1"
        )?;

        let mut rows = stmt.query(params![pattern_type.as_str()])?;

        if let Some(row) = rows.next()? {
            let last_updated_str: String = row.get(5)?;
            let last_updated = DateTime::parse_from_rfc3339(&last_updated_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            Ok(Some(PatternConfidence {
                pattern_id: row.get(0)?,
                pattern_type: PatternType::from_str(&row.get::<_, String>(1)?),
                confidence: row.get(2)?,
                usage_count: row.get(3)?,
                success_count: row.get(4)?,
                last_updated,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get all pattern confidences
    pub fn get_all_pattern_confidences(&self) -> SqlResult<HashMap<PatternType, PatternConfidence>> {
        let mut stmt = self.conn.prepare(
            "SELECT pattern_id, pattern_type, confidence, usage_count, success_count, last_updated
             FROM pattern_confidence"
        )?;

        let mut rows = stmt.query([])?;
        let mut results = HashMap::new();

        while let Some(row) = rows.next()? {
            let last_updated_str: String = row.get(5)?;
            let last_updated = DateTime::parse_from_rfc3339(&last_updated_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let pattern_type = PatternType::from_str(&row.get::<_, String>(1)?);
            
            results.insert(pattern_type, PatternConfidence {
                pattern_id: row.get(0)?,
                pattern_type,
                confidence: row.get(2)?,
                usage_count: row.get(3)?,
                success_count: row.get(4)?,
                last_updated,
            });
        }

        Ok(results)
    }

    // =========================================================================
    // USER STATE OPERATIONS
    // =========================================================================

    /// Get or create user state
    pub fn get_or_create_user(&self, user_id: &str) -> SqlResult<UserState> {
        if let Some(user) = self.get_user(user_id)? {
            return Ok(user);
        }

        let user = UserState::new(user_id, self.dimension);
        self.store_user(&user)?;
        Ok(user)
    }

    /// Get user state
    pub fn get_user(&self, user_id: &str) -> SqlResult<Option<UserState>> {
        let mut stmt = self.conn.prepare(
            "SELECT user_id, embedding, preferred_style, expertise_level, verbosity,
                    last_seen, interaction_count, topics_of_interest
             FROM user_state
             WHERE user_id = ?1"
        )?;

        let mut rows = stmt.query(params![user_id])?;

        if let Some(row) = rows.next()? {
            let embedding_bytes: Vec<u8> = row.get(1)?;
            let embedding: Vec<f64> = bincode::deserialize(&embedding_bytes)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                    1, rusqlite::types::Type::Blob, Box::new(e)
                ))?;

            let last_seen_str: String = row.get(5)?;
            let last_seen = DateTime::parse_from_rfc3339(&last_seen_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let topics_json: String = row.get(7)?;
            let topics: Vec<String> = serde_json::from_str(&topics_json).unwrap_or_default();

            Ok(Some(UserState {
                user_id: row.get(0)?,
                embedding,
                preferred_style: UserStyle::from_str(&row.get::<_, String>(2)?),
                expertise_level: ExpertiseLevel::from_str(&row.get::<_, String>(3)?),
                verbosity: row.get(4)?,
                last_seen,
                interaction_count: row.get(6)?,
                topics_of_interest: topics,
            }))
        } else {
            Ok(None)
        }
    }

    /// Store user state
    pub fn store_user(&self, user: &UserState) -> SqlResult<()> {
        let embedding_bytes = bincode::serialize(&user.embedding)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        let topics_json = serde_json::to_string(&user.topics_of_interest)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        self.conn.execute(
            "INSERT OR REPLACE INTO user_state 
             (user_id, embedding, preferred_style, expertise_level, verbosity,
              last_seen, interaction_count, topics_of_interest)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                user.user_id,
                embedding_bytes,
                user.preferred_style.as_str(),
                user.expertise_level.as_str(),
                user.verbosity,
                user.last_seen.to_rfc3339(),
                user.interaction_count,
                topics_json,
            ],
        )?;
        Ok(())
    }

    // =========================================================================
    // DECODER CONTROL OPERATIONS
    // =========================================================================

    /// Get decoder control for context
    pub fn get_decoder_control(&self, context: &str) -> SqlResult<DecoderControl> {
        let mut stmt = self.conn.prepare(
            "SELECT context, verbosity, creativity, explanation_level, ask_questions,
                    temperature, max_tokens
             FROM decoder_controls
             WHERE context = ?1"
        )?;

        let mut rows = stmt.query(params![context])?;

        if let Some(row) = rows.next()? {
            Ok(DecoderControl {
                context: row.get(0)?,
                verbosity: row.get(1)?,
                creativity: row.get(2)?,
                explanation_level: row.get(3)?,
                ask_questions: row.get::<_, i32>(4)? != 0,
                temperature: row.get(5)?,
                max_tokens: row.get(6)?,
            })
        } else {
            Ok(DecoderControl::default())
        }
    }

    /// Store decoder control
    pub fn store_decoder_control(&self, control: &DecoderControl) -> SqlResult<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO decoder_controls 
             (context, verbosity, creativity, explanation_level, ask_questions,
              temperature, max_tokens)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                control.context,
                control.verbosity,
                control.creativity,
                control.explanation_level,
                if control.ask_questions { 1 } else { 0 },
                control.temperature,
                control.max_tokens,
            ],
        )?;
        Ok(())
    }

    // =========================================================================
    // STATISTICS
    // =========================================================================

    /// Get memory statistics
    pub fn get_statistics(&self) -> SqlResult<AdaptiveMemoryStats> {
        let semantic_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM semantic_units",
            [],
            |row| row.get(0),
        )?;

        let error_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM error_log",
            [],
            |row| row.get(0),
        )?;

        let user_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM user_state",
            [],
            |row| row.get(0),
        )?;

        let avg_confidence: f64 = self.conn.query_row(
            "SELECT AVG(confidence) FROM semantic_units",
            [],
            |row| row.get(0),
        ).unwrap_or(0.0);

        Ok(AdaptiveMemoryStats {
            semantic_unit_count: semantic_count as usize,
            error_log_count: error_count as usize,
            user_count: user_count as usize,
            average_confidence: avg_confidence,
        })
    }
}

/// Statistics for adaptive memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveMemoryStats {
    pub semantic_unit_count: usize,
    pub error_log_count: usize,
    pub user_count: usize,
    pub average_confidence: f64,
}

/// Cosine similarity helper
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_unit_creation() {
        let unit = SemanticUnit::new(
            "2 + 2 = 4",
            vec![0.1; 64],
            SemanticUnitType::Fact,
            SemanticDomain::Math,
        );
        assert_eq!(unit.unit_type, SemanticUnitType::Fact);
        assert_eq!(unit.domain, SemanticDomain::Math);
    }

    #[test]
    fn test_adaptive_memory_store() -> SqlResult<()> {
        let store = AdaptiveMemoryStore::in_memory(64)?;

        // Store semantic unit
        let unit = SemanticUnit::new(
            "The capital of France is Paris",
            vec![0.1; 64],
            SemanticUnitType::Fact,
            SemanticDomain::Geography,
        );
        store.store_semantic_unit(&unit)?;

        // Find similar
        let results = store.find_similar_units(&vec![0.1; 64], 10)?;
        assert!(!results.is_empty());

        Ok(())
    }

    #[test]
    fn test_error_logging() -> SqlResult<()> {
        let store = AdaptiveMemoryStore::in_memory(64)?;

        let entry = ErrorLogEntry::new(
            "What is 2+2?",
            "5",
            ErrorType::ReasoningError,
            vec![0.1; 64],
        ).with_correction("4");

        store.log_error(&entry)?;

        let similar = store.find_similar_errors(&vec![0.1; 64], 10)?;
        assert!(!similar.is_empty());
        assert_eq!(similar[0].correct_output, Some("4".to_string()));

        Ok(())
    }

    #[test]
    fn test_pattern_confidence() -> SqlResult<()> {
        let store = AdaptiveMemoryStore::in_memory(64)?;

        // Update with successes and failures
        store.update_pattern_confidence(PatternType::Math, true)?;
        store.update_pattern_confidence(PatternType::Math, true)?;
        store.update_pattern_confidence(PatternType::Math, false)?;

        let conf = store.get_pattern_confidence(PatternType::Math)?;
        assert!(conf.is_some());
        let conf = conf.unwrap();
        assert_eq!(conf.usage_count, 3);
        assert_eq!(conf.success_count, 2);

        Ok(())
    }

    #[test]
    fn test_user_state() -> SqlResult<()> {
        let store = AdaptiveMemoryStore::in_memory(64)?;

        let mut user = store.get_or_create_user("test_user")?;
        user.preferred_style = UserStyle::Short;
        user.update_embedding(&vec![0.5; 64], 0.1);
        store.store_user(&user)?;

        let retrieved = store.get_user("test_user")?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().preferred_style, UserStyle::Short);

        Ok(())
    }

    #[test]
    fn test_decoder_control() -> SqlResult<()> {
        let store = AdaptiveMemoryStore::in_memory(64)?;

        let control = DecoderControl::creative();
        store.store_decoder_control(&DecoderControl {
            context: "creative_writing".to_string(),
            ..control
        })?;

        let retrieved = store.get_decoder_control("creative_writing")?;
        assert!(retrieved.creativity > 0.8);

        Ok(())
    }
}
