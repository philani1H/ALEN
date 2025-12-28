//! State Module - Manages Thought State Vectors (ψ)
//! 
//! A thought is represented as a vector |ψ⟩ ∈ ℝⁿ where each dimension
//! represents a semantic or logical feature.

use nalgebra::DVector;
use rand::SeedableRng;
use rand_distr::{Normal, Distribution};
use serde::{Deserialize, Serialize};

/// The dimensionality of our thought space
pub const DEFAULT_DIMENSION: usize = 128;

/// Thought State Vector - represents a "thought" in high-dimensional space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtState {
    /// The actual vector representation
    pub vector: Vec<f64>,
    /// Dimensionality of the space
    pub dimension: usize,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    /// Metadata about this thought
    pub metadata: ThoughtMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThoughtMetadata {
    pub source: Option<String>,
    pub timestamp: Option<i64>,
    pub operator_id: Option<String>,
    pub iteration: usize,
}

impl ThoughtState {
    /// Create a new thought state with given dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            vector: vec![0.0; dimension],
            dimension,
            confidence: 0.0,
            metadata: ThoughtMetadata::default(),
        }
    }

    /// Create a thought state from raw input
    /// Uses mathematical AST embedding for math expressions, falls back to text embedding
    pub fn from_input(input: &str, dimension: usize) -> Self {
        // Try to parse as mathematical expression first
        if let Ok(ast) = crate::math::MathParser::parse(input) {
            // Successfully parsed as math - use AST embedder
            let simplified = crate::math::simplify_expression(&ast);
            let embedder = crate::math::MathEmbedder::new(dimension);
            let vector = embedder.embed(&simplified);

            return Self {
                vector,
                dimension,
                confidence: 0.5,
                metadata: ThoughtMetadata {
                    source: Some(input.to_string()),
                    timestamp: Some(chrono::Utc::now().timestamp()),
                    ..Default::default()
                },
            };
        }

        // Fall back to text-based embedding for non-math content
        // Use word-based compositional embedding for better semantic similarity
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Split into words and create compositional embedding
        let lowercase_input = input.to_lowercase();
        let words: Vec<&str> = lowercase_input
            .split_whitespace()
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            // Empty input - return zero vector
            return Self {
                vector: vec![0.0; dimension],
                dimension,
                confidence: 0.0,
                metadata: ThoughtMetadata {
                    source: Some(input.to_string()),
                    timestamp: Some(chrono::Utc::now().timestamp()),
                    ..Default::default()
                },
            };
        }

        // Create compositional embedding by averaging word embeddings
        let mut combined_vector = vec![0.0; dimension];

        for word in words.iter() {
            // Generate deterministic embedding for each word
            let word_seed: u64 = word.bytes().map(|b| b as u64).sum();
            let mut word_rng = rand::rngs::StdRng::seed_from_u64(word_seed);

            // Generate word embedding
            let word_vector: Vec<f64> = (0..dimension)
                .map(|_| normal.sample(&mut word_rng))
                .collect();

            // Add to combined vector
            for (i, &val) in word_vector.iter().enumerate() {
                combined_vector[i] += val;
            }
        }

        // Average the combined vector
        let word_count = words.len() as f64;
        for val in combined_vector.iter_mut() {
            *val /= word_count;
        }

        // Normalize the vector
        let norm: f64 = combined_vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        let normalized: Vec<f64> = if norm > 1e-10 {
            combined_vector.iter().map(|x| x / norm).collect()
        } else {
            combined_vector
        };

        Self {
            vector: normalized,
            dimension,
            confidence: 0.5,
            metadata: ThoughtMetadata {
                source: Some(input.to_string()),
                timestamp: Some(chrono::Utc::now().timestamp()),
                ..Default::default()
            },
        }
    }

    /// Create a random thought state (for exploration)
    pub fn random(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        let vector: Vec<f64> = (0..dimension)
            .map(|_| normal.sample(&mut rng))
            .collect();
        
        let norm: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        let normalized: Vec<f64> = vector.iter().map(|x| x / norm).collect();
        
        Self {
            vector: normalized,
            dimension,
            confidence: 0.0,
            metadata: ThoughtMetadata::default(),
        }
    }

    /// Get the L2 norm of the thought vector
    pub fn norm(&self) -> f64 {
        self.vector.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Normalize the thought vector
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 1e-10 {
            for x in &mut self.vector {
                *x /= norm;
            }
        }
    }

    /// Compute dot product with another thought
    pub fn dot(&self, other: &ThoughtState) -> f64 {
        self.vector.iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Compute cosine similarity with another thought
    pub fn cosine_similarity(&self, other: &ThoughtState) -> f64 {
        let dot = self.dot(other);
        let norm_self = self.norm();
        let norm_other = other.norm();
        
        if norm_self > 1e-10 && norm_other > 1e-10 {
            dot / (norm_self * norm_other)
        } else {
            0.0
        }
    }

    /// Add two thought states (for combining perspectives)
    pub fn add(&self, other: &ThoughtState) -> Self {
        let vector: Vec<f64> = self.vector.iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        Self {
            vector,
            dimension: self.dimension,
            confidence: (self.confidence + other.confidence) / 2.0,
            metadata: self.metadata.clone(),
        }
    }

    /// Scale the thought vector
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            vector: self.vector.iter().map(|x| x * factor).collect(),
            dimension: self.dimension,
            confidence: self.confidence,
            metadata: self.metadata.clone(),
        }
    }

    /// Convert to nalgebra DVector for matrix operations
    pub fn to_dvector(&self) -> DVector<f64> {
        DVector::from_vec(self.vector.clone())
    }

    /// Create from nalgebra DVector
    pub fn from_dvector(dv: DVector<f64>, confidence: f64) -> Self {
        let dimension = dv.len();
        Self {
            vector: dv.iter().cloned().collect(),
            dimension,
            confidence,
            metadata: ThoughtMetadata::default(),
        }
    }

    /// Create from a Vec<f64> with specified dimension
    pub fn from_vector(vector: Vec<f64>, dimension: usize) -> Self {
        let mut v = vector;
        // Ensure correct dimension
        v.resize(dimension, 0.0);
        v.truncate(dimension);
        
        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        }
        
        Self {
            vector: v,
            dimension,
            confidence: 0.5,
            metadata: ThoughtMetadata::default(),
        }
    }

    /// Compute distance to another thought state
    pub fn distance(&self, other: &ThoughtState) -> f64 {
        self.vector.iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Bias Vector - modulates reasoning (the "emotion" component)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasVector {
    /// Risk tolerance (0.0 = conservative, 1.0 = aggressive)
    pub risk_tolerance: f64,
    /// Exploration vs exploitation (0.0 = exploit, 1.0 = explore)
    pub exploration: f64,
    /// Response urgency (0.0 = thorough, 1.0 = fast)
    pub urgency: f64,
    /// Creativity level (0.0 = literal, 1.0 = creative)
    pub creativity: f64,
}

impl Default for BiasVector {
    fn default() -> Self {
        Self {
            risk_tolerance: 0.5,
            exploration: 0.5,
            urgency: 0.5,
            creativity: 0.5,
        }
    }
}

impl BiasVector {
    /// Convert to a vector for mathematical operations
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.risk_tolerance,
            self.exploration,
            self.urgency,
            self.creativity,
        ]
    }

    /// Modulate a thought state with this bias
    pub fn modulate(&self, thought: &ThoughtState) -> ThoughtState {
        // Apply bias as a scaling factor to certain dimensions
        let mut new_vector = thought.vector.clone();
        
        // First quarter affected by risk tolerance
        let quarter = thought.dimension / 4;
        for i in 0..quarter {
            new_vector[i] *= 1.0 + (self.risk_tolerance - 0.5);
        }
        
        // Second quarter affected by exploration
        for i in quarter..(2 * quarter) {
            new_vector[i] *= 1.0 + (self.exploration - 0.5);
        }
        
        // Third quarter affected by urgency
        for i in (2 * quarter)..(3 * quarter) {
            new_vector[i] *= 1.0 + (self.urgency - 0.5);
        }
        
        // Fourth quarter affected by creativity
        for i in (3 * quarter)..thought.dimension {
            new_vector[i] *= 1.0 + (self.creativity - 0.5);
        }
        
        ThoughtState {
            vector: new_vector,
            dimension: thought.dimension,
            confidence: thought.confidence,
            metadata: thought.metadata.clone(),
        }
    }
}

/// Problem representation - structured input for the reasoning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Problem {
    /// The encoded problem state
    pub state: ThoughtState,
    /// The original input text
    pub input: String,
    /// Goal description
    pub goal: Option<String>,
    /// Constraints on the solution
    pub constraints: Vec<String>,
    /// Context from memory
    pub context: Vec<String>,
    /// Known answer (for training)
    pub target_answer: Option<String>,
    /// Target state vector (for training)
    pub target_state: Option<ThoughtState>,
}

impl Problem {
    /// Create a new problem from text input
    pub fn new(input: &str, dimension: usize) -> Self {
        Self {
            state: ThoughtState::from_input(input, dimension),
            input: input.to_string(),
            goal: None,
            constraints: Vec::new(),
            context: Vec::new(),
            target_answer: None,
            target_state: None,
        }
    }

    /// Create a training problem with known answer
    pub fn training(input: &str, answer: &str, dimension: usize) -> Self {
        let mut problem = Self::new(input, dimension);
        problem.target_answer = Some(answer.to_string());
        problem.target_state = Some(ThoughtState::from_input(answer, dimension));
        problem
    }

    /// Add a constraint
    pub fn with_constraint(mut self, constraint: &str) -> Self {
        self.constraints.push(constraint.to_string());
        self
    }

    /// Add context
    pub fn with_context(mut self, context: &str) -> Self {
        self.context.push(context.to_string());
        self
    }

    /// Set goal
    pub fn with_goal(mut self, goal: &str) -> Self {
        self.goal = Some(goal.to_string());
        self
    }

    /// Estimate problem difficulty (0.0 = easy, 1.0 = hard)
    pub fn difficulty(&self) -> f64 {
        let mut difficulty = 0.0;
        
        // Longer inputs tend to be more complex
        difficulty += (self.input.len() as f64 / 500.0).min(0.3);
        
        // More constraints = harder
        difficulty += (self.constraints.len() as f64 * 0.1).min(0.3);
        
        // Context-dependent problems are harder
        difficulty += (self.context.len() as f64 * 0.05).min(0.2);
        
        // If goal is specified, slightly harder
        if self.goal.is_some() {
            difficulty += 0.1;
        }
        
        // If it's a training problem with target, adjust based on target complexity
        if let Some(ref target) = self.target_answer {
            difficulty += (target.len() as f64 / 200.0).min(0.1);
        }
        
        difficulty.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thought_creation() {
        let thought = ThoughtState::from_input("hello world", 64);
        assert_eq!(thought.dimension, 64);
        assert!((thought.norm() - 1.0).abs() < 0.01); // Should be normalized
    }

    #[test]
    fn test_cosine_similarity() {
        let t1 = ThoughtState::from_input("hello", 64);
        let t2 = ThoughtState::from_input("hello", 64);
        let t3 = ThoughtState::from_input("goodbye", 64);
        
        // Same input should give same embedding
        assert!((t1.cosine_similarity(&t2) - 1.0).abs() < 0.01);
        // Different input should give different embedding
        assert!(t1.cosine_similarity(&t3) < 0.99);
    }

    #[test]
    fn test_bias_modulation() {
        let thought = ThoughtState::from_input("test", 64);
        let bias = BiasVector {
            risk_tolerance: 0.8,
            exploration: 0.2,
            urgency: 0.5,
            creativity: 0.9,
        };
        
        let modulated = bias.modulate(&thought);
        assert_eq!(modulated.dimension, thought.dimension);
    }
}
