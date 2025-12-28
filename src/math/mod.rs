//! Mathematical Expression Parsing and Embedding
//!
//! This module provides AST-based embeddings for mathematical expressions,
//! ensuring that mathematically equivalent expressions have similar vectors.
//!
//! Examples:
//! - "2 + 2" and "4" → similar vectors
//! - "x^2 + 2x + 1" and "(x+1)^2" → similar vectors

pub mod parser;
pub mod embedder;
pub mod simplifier;

pub use parser::MathParser;
pub use embedder::MathEmbedder;
pub use simplifier::simplify_expression;

use std::collections::HashMap;

/// Mathematical expression types
#[derive(Debug, Clone, PartialEq)]
pub enum MathAST {
    /// Literal number
    Literal(f64),
    /// Variable (e.g., "x", "y")
    Variable(String),
    /// Addition
    Add(Box<MathAST>, Box<MathAST>),
    /// Subtraction
    Sub(Box<MathAST>, Box<MathAST>),
    /// Multiplication
    Mul(Box<MathAST>, Box<MathAST>),
    /// Division
    Div(Box<MathAST>, Box<MathAST>),
    /// Power/Exponentiation
    Pow(Box<MathAST>, Box<MathAST>),
    /// Negation
    Neg(Box<MathAST>),
    /// Function call (sin, cos, sqrt, etc.)
    Function(String, Box<MathAST>),
}

impl MathAST {
    /// Evaluate the AST to a numerical value if possible
    pub fn eval(&self, vars: &HashMap<String, f64>) -> Option<f64> {
        match self {
            MathAST::Literal(v) => Some(*v),
            MathAST::Variable(name) => vars.get(name).copied(),
            MathAST::Add(l, r) => {
                let lv = l.eval(vars)?;
                let rv = r.eval(vars)?;
                Some(lv + rv)
            }
            MathAST::Sub(l, r) => {
                let lv = l.eval(vars)?;
                let rv = r.eval(vars)?;
                Some(lv - rv)
            }
            MathAST::Mul(l, r) => {
                let lv = l.eval(vars)?;
                let rv = r.eval(vars)?;
                Some(lv * rv)
            }
            MathAST::Div(l, r) => {
                let lv = l.eval(vars)?;
                let rv = r.eval(vars)?;
                if rv.abs() < 1e-10 {
                    None // Division by zero
                } else {
                    Some(lv / rv)
                }
            }
            MathAST::Pow(base, exp) => {
                let bv = base.eval(vars)?;
                let ev = exp.eval(vars)?;
                Some(bv.powf(ev))
            }
            MathAST::Neg(inner) => {
                let v = inner.eval(vars)?;
                Some(-v)
            }
            MathAST::Function(name, arg) => {
                let v = arg.eval(vars)?;
                match name.as_str() {
                    "sin" => Some(v.sin()),
                    "cos" => Some(v.cos()),
                    "tan" => Some(v.tan()),
                    "sqrt" => Some(v.sqrt()),
                    "abs" => Some(v.abs()),
                    "ln" => Some(v.ln()),
                    "log" => Some(v.log10()),
                    "exp" => Some(v.exp()),
                    _ => None,
                }
            }
        }
    }

    /// Get a canonical string representation
    pub fn to_canonical_string(&self) -> String {
        match self {
            MathAST::Literal(v) => {
                // Normalize floating point representation
                if v.fract() == 0.0 {
                    format!("{}", *v as i64)
                } else {
                    format!("{:.6}", v)
                }
            }
            MathAST::Variable(name) => name.clone(),
            MathAST::Add(l, r) => format!("({}+{})", l.to_canonical_string(), r.to_canonical_string()),
            MathAST::Sub(l, r) => format!("({}-{})", l.to_canonical_string(), r.to_canonical_string()),
            MathAST::Mul(l, r) => format!("({}*{})", l.to_canonical_string(), r.to_canonical_string()),
            MathAST::Div(l, r) => format!("({}/{})", l.to_canonical_string(), r.to_canonical_string()),
            MathAST::Pow(base, exp) => format!("({}^{})", base.to_canonical_string(), exp.to_canonical_string()),
            MathAST::Neg(inner) => format!("(-{})", inner.to_canonical_string()),
            MathAST::Function(name, arg) => format!("{}({})", name, arg.to_canonical_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_eval() {
        let ast = MathAST::Literal(42.0);
        assert_eq!(ast.eval(&HashMap::new()), Some(42.0));
    }

    #[test]
    fn test_addition_eval() {
        let ast = MathAST::Add(
            Box::new(MathAST::Literal(2.0)),
            Box::new(MathAST::Literal(2.0)),
        );
        assert_eq!(ast.eval(&HashMap::new()), Some(4.0));
    }

    #[test]
    fn test_canonical_string() {
        let ast = MathAST::Add(
            Box::new(MathAST::Literal(2.0)),
            Box::new(MathAST::Literal(2.0)),
        );
        assert_eq!(ast.to_canonical_string(), "(2+2)");
    }
}
