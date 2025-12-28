//! AST-based mathematical expression embedder
//!
//! Converts mathematical ASTs into vectors such that
//! equivalent expressions have similar embeddings

use super::MathAST;
use rand::{Rng, SeedableRng};
use rand_distr::{Normal, Distribution};
use std::collections::HashMap;

/// Embeds mathematical expressions into vectors
pub struct MathEmbedder {
    dimension: usize,
    /// Pre-generated embeddings for operators
    operator_embeddings: HashMap<String, Vec<f64>>,
    /// Pre-generated embeddings for common functions
    function_embeddings: HashMap<String, Vec<f64>>,
}

impl MathEmbedder {
    /// Create a new math embedder
    pub fn new(dimension: usize) -> Self {
        let mut embedder = Self {
            dimension,
            operator_embeddings: HashMap::new(),
            function_embeddings: HashMap::new(),
        };

        // Initialize operator embeddings
        embedder.init_operator_embeddings();
        embedder.init_function_embeddings();

        embedder
    }

    /// Initialize embeddings for mathematical operators
    fn init_operator_embeddings(&mut self) {
        let operators = vec![
            ("add", 1001),
            ("sub", 1002),
            ("mul", 1003),
            ("div", 1004),
            ("pow", 1005),
            ("neg", 1006),
        ];

        for (op, seed) in operators {
            let embedding = Self::generate_stable_embedding(self.dimension, seed);
            self.operator_embeddings.insert(op.to_string(), embedding);
        }
    }

    /// Initialize embeddings for mathematical functions
    fn init_function_embeddings(&mut self) {
        let functions = vec![
            ("sin", 2001),
            ("cos", 2002),
            ("tan", 2003),
            ("sqrt", 2004),
            ("abs", 2005),
            ("ln", 2006),
            ("log", 2007),
            ("exp", 2008),
        ];

        for (func, seed) in functions {
            let embedding = Self::generate_stable_embedding(self.dimension, seed);
            self.function_embeddings.insert(func.to_string(), embedding);
        }
    }

    /// Generate a stable embedding from a seed
    fn generate_stable_embedding(dimension: usize, seed: u64) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let vector: Vec<f64> = (0..dimension)
            .map(|_| normal.sample(&mut rng))
            .collect();

        // Normalize
        let norm: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        vector.iter().map(|x| x / norm).collect()
    }

    /// Embed a mathematical AST into a vector
    pub fn embed(&self, ast: &MathAST) -> Vec<f64> {
        let embedding = self.embed_recursive(ast);

        // Normalize the final embedding
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding
        }
    }

    /// Recursive embedding of AST nodes
    fn embed_recursive(&self, ast: &MathAST) -> Vec<f64> {
        match ast {
            MathAST::Literal(value) => {
                // Encode literal numbers based on their value
                // Use a hash-like encoding but stable for same values
                let seed = (value.abs() * 1000000.0) as u64 + 5000;
                let mut base = Self::generate_stable_embedding(self.dimension, seed);

                // Scale by sign
                if *value < 0.0 {
                    base = base.iter().map(|x| -x).collect();
                }

                base
            }

            MathAST::Variable(name) => {
                // Variables get stable embeddings based on name
                let seed: u64 = name.bytes().map(|b| b as u64).sum::<u64>() + 10000;
                Self::generate_stable_embedding(self.dimension, seed)
            }

            MathAST::Add(left, right) => {
                let left_emb = self.embed_recursive(left);
                let right_emb = self.embed_recursive(right);
                let op_emb = &self.operator_embeddings["add"];

                self.combine_embeddings(&left_emb, &right_emb, op_emb)
            }

            MathAST::Sub(left, right) => {
                let left_emb = self.embed_recursive(left);
                let right_emb = self.embed_recursive(right);
                let op_emb = &self.operator_embeddings["sub"];

                self.combine_embeddings(&left_emb, &right_emb, op_emb)
            }

            MathAST::Mul(left, right) => {
                let left_emb = self.embed_recursive(left);
                let right_emb = self.embed_recursive(right);
                let op_emb = &self.operator_embeddings["mul"];

                self.combine_embeddings(&left_emb, &right_emb, op_emb)
            }

            MathAST::Div(left, right) => {
                let left_emb = self.embed_recursive(left);
                let right_emb = self.embed_recursive(right);
                let op_emb = &self.operator_embeddings["div"];

                self.combine_embeddings(&left_emb, &right_emb, op_emb)
            }

            MathAST::Pow(base, exp) => {
                let base_emb = self.embed_recursive(base);
                let exp_emb = self.embed_recursive(exp);
                let op_emb = &self.operator_embeddings["pow"];

                self.combine_embeddings(&base_emb, &exp_emb, op_emb)
            }

            MathAST::Neg(inner) => {
                let inner_emb = self.embed_recursive(inner);
                let op_emb = &self.operator_embeddings["neg"];

                // For unary operations, combine with zero vector
                let zero = vec![0.0; self.dimension];
                self.combine_embeddings(&inner_emb, &zero, op_emb)
            }

            MathAST::Function(name, arg) => {
                let arg_emb = self.embed_recursive(arg);
                let func_emb = self.function_embeddings
                    .get(name.as_str())
                    .cloned()
                    .unwrap_or_else(|| {
                        // Unknown function - generate stable embedding from name
                        let seed: u64 = name.bytes().map(|b| b as u64).sum::<u64>() + 20000;
                        Self::generate_stable_embedding(self.dimension, seed)
                    });

                // For functions, weight the argument more heavily
                let zero = vec![0.0; self.dimension];
                self.combine_embeddings(&arg_emb, &zero, &func_emb)
            }
        }
    }

    /// Combine three embeddings (left, right, operator)
    fn combine_embeddings(&self, left: &[f64], right: &[f64], op: &[f64]) -> Vec<f64> {
        // Weighted combination: emphasize structure via operator
        // Formula: 0.4 * left + 0.4 * right + 0.2 * op
        let mut result = vec![0.0; self.dimension];

        for i in 0..self.dimension {
            result[i] = 0.4 * left[i] + 0.4 * right[i] + 0.2 * op[i];
        }

        result
    }
}

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{MathParser, simplify_expression};

    #[test]
    fn test_embedding_dimension() {
        let embedder = MathEmbedder::new(128);
        let ast = MathParser::parse("2 + 2").unwrap();
        let embedding = embedder.embed(&ast);
        assert_eq!(embedding.len(), 128);
    }

    #[test]
    fn test_equivalent_expressions_similar() {
        let embedder = MathEmbedder::new(128);

        // Parse and simplify "2 + 2"
        let ast1 = MathParser::parse("2 + 2").unwrap();
        let simplified1 = simplify_expression(&ast1);

        // Parse "4"
        let ast2 = MathParser::parse("4").unwrap();

        let emb1 = embedder.embed(&simplified1);
        let emb2 = embedder.embed(&ast2);

        let similarity = cosine_similarity(&emb1, &emb2);

        // After simplification, "2 + 2" becomes "4", so they should be identical
        assert!((similarity - 1.0).abs() < 0.01, "Similarity: {}", similarity);
    }

    #[test]
    fn test_different_expressions_dissimilar() {
        let embedder = MathEmbedder::new(128);

        let ast1 = MathParser::parse("2 + 2").unwrap();
        let ast2 = MathParser::parse("10 * 20").unwrap();

        let emb1 = embedder.embed(&ast1);
        let emb2 = embedder.embed(&ast2);

        let similarity = cosine_similarity(&emb1, &emb2);

        // Different expressions should have lower similarity
        assert!(similarity < 0.9, "Similarity: {}", similarity);
    }

    #[test]
    fn test_same_expression_identical() {
        let embedder = MathEmbedder::new(128);

        let ast1 = MathParser::parse("x + y").unwrap();
        let ast2 = MathParser::parse("x + y").unwrap();

        let emb1 = embedder.embed(&ast1);
        let emb2 = embedder.embed(&ast2);

        let similarity = cosine_similarity(&emb1, &emb2);

        // Same expression should be identical
        assert!((similarity - 1.0).abs() < 0.01, "Similarity: {}", similarity);
    }
}
