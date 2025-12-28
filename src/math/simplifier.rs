//! Mathematical expression simplification
//!
//! Normalizes expressions to canonical forms so that equivalent
//! expressions map to the same or similar forms

use super::MathAST;
use std::collections::HashMap;

/// Simplify a mathematical expression to canonical form
pub fn simplify_expression(ast: &MathAST) -> MathAST {
    let mut simplified = simplify_basic(ast);

    // Multiple passes for thorough simplification
    for _ in 0..3 {
        let new_simplified = simplify_basic(&simplified);
        if new_simplified == simplified {
            break;
        }
        simplified = new_simplified;
    }

    simplified
}

/// Single pass of basic simplification
fn simplify_basic(ast: &MathAST) -> MathAST {
    match ast {
        // Literal numbers stay as is
        MathAST::Literal(v) => MathAST::Literal(*v),

        // Variables stay as is
        MathAST::Variable(name) => MathAST::Variable(name.clone()),

        // Addition simplifications
        MathAST::Add(l, r) => {
            let left = simplify_basic(l);
            let right = simplify_basic(r);

            // Evaluate if both are literals
            if let (MathAST::Literal(lv), MathAST::Literal(rv)) = (&left, &right) {
                return MathAST::Literal(lv + rv);
            }

            // 0 + x = x
            if let MathAST::Literal(0.0) = left {
                return right;
            }

            // x + 0 = x
            if let MathAST::Literal(0.0) = right {
                return left;
            }

            MathAST::Add(Box::new(left), Box::new(right))
        }

        // Subtraction simplifications
        MathAST::Sub(l, r) => {
            let left = simplify_basic(l);
            let right = simplify_basic(r);

            // Evaluate if both are literals
            if let (MathAST::Literal(lv), MathAST::Literal(rv)) = (&left, &right) {
                return MathAST::Literal(lv - rv);
            }

            // x - 0 = x
            if let MathAST::Literal(0.0) = right {
                return left;
            }

            // x - x = 0
            if left == right {
                return MathAST::Literal(0.0);
            }

            MathAST::Sub(Box::new(left), Box::new(right))
        }

        // Multiplication simplifications
        MathAST::Mul(l, r) => {
            let left = simplify_basic(l);
            let right = simplify_basic(r);

            // Evaluate if both are literals
            if let (MathAST::Literal(lv), MathAST::Literal(rv)) = (&left, &right) {
                return MathAST::Literal(lv * rv);
            }

            // 0 * x = 0
            if let MathAST::Literal(0.0) = left {
                return MathAST::Literal(0.0);
            }

            // x * 0 = 0
            if let MathAST::Literal(0.0) = right {
                return MathAST::Literal(0.0);
            }

            // 1 * x = x
            if let MathAST::Literal(1.0) = left {
                return right;
            }

            // x * 1 = x
            if let MathAST::Literal(1.0) = right {
                return left;
            }

            MathAST::Mul(Box::new(left), Box::new(right))
        }

        // Division simplifications
        MathAST::Div(l, r) => {
            let left = simplify_basic(l);
            let right = simplify_basic(r);

            // Evaluate if both are literals
            if let (MathAST::Literal(lv), MathAST::Literal(rv)) = (&left, &right) {
                if rv.abs() > 1e-10 {
                    return MathAST::Literal(lv / rv);
                }
            }

            // x / 1 = x
            if let MathAST::Literal(1.0) = right {
                return left;
            }

            // 0 / x = 0 (if x != 0)
            if let MathAST::Literal(0.0) = left {
                return MathAST::Literal(0.0);
            }

            // x / x = 1
            if left == right {
                return MathAST::Literal(1.0);
            }

            MathAST::Div(Box::new(left), Box::new(right))
        }

        // Power simplifications
        MathAST::Pow(base, exp) => {
            let b = simplify_basic(base);
            let e = simplify_basic(exp);

            // Evaluate if both are literals
            if let (MathAST::Literal(bv), MathAST::Literal(ev)) = (&b, &e) {
                return MathAST::Literal(bv.powf(*ev));
            }

            // x ^ 0 = 1
            if let MathAST::Literal(0.0) = e {
                return MathAST::Literal(1.0);
            }

            // x ^ 1 = x
            if let MathAST::Literal(1.0) = e {
                return b;
            }

            // 0 ^ x = 0 (for positive x)
            if let MathAST::Literal(0.0) = b {
                return MathAST::Literal(0.0);
            }

            // 1 ^ x = 1
            if let MathAST::Literal(1.0) = b {
                return MathAST::Literal(1.0);
            }

            MathAST::Pow(Box::new(b), Box::new(e))
        }

        // Negation simplifications
        MathAST::Neg(inner) => {
            let simplified_inner = simplify_basic(inner);

            // Negate literals
            if let MathAST::Literal(v) = simplified_inner {
                return MathAST::Literal(-v);
            }

            // Double negation: -(-x) = x
            if let MathAST::Neg(inner_inner) = simplified_inner {
                return *inner_inner;
            }

            MathAST::Neg(Box::new(simplified_inner))
        }

        // Function simplifications
        MathAST::Function(name, arg) => {
            let simplified_arg = simplify_basic(arg);

            // Evaluate function if argument is literal
            if let MathAST::Literal(v) = simplified_arg {
                let result = match name.as_str() {
                    "sin" => Some(v.sin()),
                    "cos" => Some(v.cos()),
                    "tan" => Some(v.tan()),
                    "sqrt" => Some(v.sqrt()),
                    "abs" => Some(v.abs()),
                    "ln" => Some(v.ln()),
                    "log" => Some(v.log10()),
                    "exp" => Some(v.exp()),
                    _ => None,
                };

                if let Some(result) = result {
                    return MathAST::Literal(result);
                }
            }

            MathAST::Function(name.clone(), Box::new(simplified_arg))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::MathParser;

    #[test]
    fn test_simplify_addition() {
        let ast = MathParser::parse("2 + 2").unwrap();
        let simplified = simplify_expression(&ast);
        assert_eq!(simplified, MathAST::Literal(4.0));
    }

    #[test]
    fn test_simplify_zero_addition() {
        let ast = MathParser::parse("x + 0").unwrap();
        let simplified = simplify_expression(&ast);
        assert_eq!(simplified, MathAST::Variable("x".to_string()));
    }

    #[test]
    fn test_simplify_multiplication_by_one() {
        let ast = MathParser::parse("x * 1").unwrap();
        let simplified = simplify_expression(&ast);
        assert_eq!(simplified, MathAST::Variable("x".to_string()));
    }

    #[test]
    fn test_simplify_power_of_one() {
        let ast = MathParser::parse("x ^ 1").unwrap();
        let simplified = simplify_expression(&ast);
        assert_eq!(simplified, MathAST::Variable("x".to_string()));
    }

    #[test]
    fn test_simplify_complex() {
        let ast = MathParser::parse("2 + 3 * 0 + 5").unwrap();
        let simplified = simplify_expression(&ast);
        assert_eq!(simplified, MathAST::Literal(7.0));
    }
}
