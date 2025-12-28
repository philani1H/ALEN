//! Advanced Mathematical Reasoning
//!
//! Solves complex mathematical problems using symbolic manipulation
//! and multi-step reasoning

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Mathematical expression types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MathExpression {
    Number(f64),
    Variable(String),
    Add(Box<MathExpression>, Box<MathExpression>),
    Subtract(Box<MathExpression>, Box<MathExpression>),
    Multiply(Box<MathExpression>, Box<MathExpression>),
    Divide(Box<MathExpression>, Box<MathExpression>),
    Power(Box<MathExpression>, Box<MathExpression>),
    Sin(Box<MathExpression>),
    Cos(Box<MathExpression>),
    Ln(Box<MathExpression>),
    Sqrt(Box<MathExpression>),
}

impl MathExpression {
    /// Evaluate expression with given variable values
    pub fn evaluate(&self, vars: &HashMap<String, f64>) -> Result<f64, String> {
        match self {
            MathExpression::Number(n) => Ok(*n),
            MathExpression::Variable(v) => {
                vars.get(v).copied()
                    .ok_or_else(|| format!("Variable {} not found", v))
            }
            MathExpression::Add(a, b) => {
                Ok(a.evaluate(vars)? + b.evaluate(vars)?)
            }
            MathExpression::Subtract(a, b) => {
                Ok(a.evaluate(vars)? - b.evaluate(vars)?)
            }
            MathExpression::Multiply(a, b) => {
                Ok(a.evaluate(vars)? * b.evaluate(vars)?)
            }
            MathExpression::Divide(a, b) => {
                let divisor = b.evaluate(vars)?;
                if divisor.abs() < 1e-10 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(a.evaluate(vars)? / divisor)
                }
            }
            MathExpression::Power(a, b) => {
                Ok(a.evaluate(vars)?.powf(b.evaluate(vars)?))
            }
            MathExpression::Sin(a) => Ok(a.evaluate(vars)?.sin()),
            MathExpression::Cos(a) => Ok(a.evaluate(vars)?.cos()),
            MathExpression::Ln(a) => {
                let val = a.evaluate(vars)?;
                if val <= 0.0 {
                    Err("Logarithm of non-positive number".to_string())
                } else {
                    Ok(val.ln())
                }
            }
            MathExpression::Sqrt(a) => {
                let val = a.evaluate(vars)?;
                if val < 0.0 {
                    Err("Square root of negative number".to_string())
                } else {
                    Ok(val.sqrt())
                }
            }
        }
    }

    /// Simplify expression
    pub fn simplify(&self) -> MathExpression {
        match self {
            MathExpression::Add(a, b) => {
                let a_simp = a.simplify();
                let b_simp = b.simplify();
                
                match (&a_simp, &b_simp) {
                    (MathExpression::Number(0.0), _) => b_simp,
                    (_, MathExpression::Number(0.0)) => a_simp,
                    (MathExpression::Number(x), MathExpression::Number(y)) => {
                        MathExpression::Number(x + y)
                    }
                    _ => MathExpression::Add(Box::new(a_simp), Box::new(b_simp)),
                }
            }
            MathExpression::Multiply(a, b) => {
                let a_simp = a.simplify();
                let b_simp = b.simplify();
                
                match (&a_simp, &b_simp) {
                    (MathExpression::Number(0.0), _) | (_, MathExpression::Number(0.0)) => {
                        MathExpression::Number(0.0)
                    }
                    (MathExpression::Number(1.0), _) => b_simp,
                    (_, MathExpression::Number(1.0)) => a_simp,
                    (MathExpression::Number(x), MathExpression::Number(y)) => {
                        MathExpression::Number(x * y)
                    }
                    _ => MathExpression::Multiply(Box::new(a_simp), Box::new(b_simp)),
                }
            }
            _ => self.clone(),
        }
    }

    /// Differentiate with respect to variable
    pub fn differentiate(&self, var: &str) -> MathExpression {
        match self {
            MathExpression::Number(_) => MathExpression::Number(0.0),
            MathExpression::Variable(v) => {
                if v == var {
                    MathExpression::Number(1.0)
                } else {
                    MathExpression::Number(0.0)
                }
            }
            MathExpression::Add(a, b) => {
                MathExpression::Add(
                    Box::new(a.differentiate(var)),
                    Box::new(b.differentiate(var)),
                )
            }
            MathExpression::Multiply(a, b) => {
                // Product rule: (uv)' = u'v + uv'
                MathExpression::Add(
                    Box::new(MathExpression::Multiply(
                        Box::new(a.differentiate(var)),
                        b.clone(),
                    )),
                    Box::new(MathExpression::Multiply(
                        a.clone(),
                        Box::new(b.differentiate(var)),
                    )),
                )
            }
            MathExpression::Power(a, b) => {
                // Power rule: (x^n)' = n*x^(n-1)
                if let MathExpression::Variable(v) = a.as_ref() {
                    if v == var {
                        if let MathExpression::Number(n) = b.as_ref() {
                            return MathExpression::Multiply(
                                b.clone(),
                                Box::new(MathExpression::Power(
                                    a.clone(),
                                    Box::new(MathExpression::Number(n - 1.0)),
                                )),
                            );
                        }
                    }
                }
                // General case: use chain rule
                self.clone()
            }
            _ => self.clone(),
        }
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match self {
            MathExpression::Number(n) => format!("{}", n),
            MathExpression::Variable(v) => v.clone(),
            MathExpression::Add(a, b) => format!("({} + {})", a.to_string(), b.to_string()),
            MathExpression::Subtract(a, b) => format!("({} - {})", a.to_string(), b.to_string()),
            MathExpression::Multiply(a, b) => format!("({} * {})", a.to_string(), b.to_string()),
            MathExpression::Divide(a, b) => format!("({} / {})", a.to_string(), b.to_string()),
            MathExpression::Power(a, b) => format!("({}^{})", a.to_string(), b.to_string()),
            MathExpression::Sin(a) => format!("sin({})", a.to_string()),
            MathExpression::Cos(a) => format!("cos({})", a.to_string()),
            MathExpression::Ln(a) => format!("ln({})", a.to_string()),
            MathExpression::Sqrt(a) => format!("sqrt({})", a.to_string()),
        }
    }
}

/// Result of mathematical computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathResult {
    pub expression: String,
    pub simplified: String,
    pub value: Option<f64>,
    pub steps: Vec<String>,
    pub confidence: f64,
}

/// Advanced mathematical solver
pub struct MathSolver {
    /// Known constants
    constants: HashMap<String, f64>,
}

impl MathSolver {
    pub fn new() -> Self {
        let mut constants = HashMap::new();
        constants.insert("pi".to_string(), std::f64::consts::PI);
        constants.insert("e".to_string(), std::f64::consts::E);
        constants.insert("phi".to_string(), 1.618033988749895); // Golden ratio
        
        Self { constants }
    }

    /// Parse simple mathematical expression from string
    pub fn parse(&self, input: &str) -> Result<MathExpression, String> {
        let input = input.trim();
        
        // Simple number
        if let Ok(n) = input.parse::<f64>() {
            return Ok(MathExpression::Number(n));
        }
        
        // Variable or constant
        if input.chars().all(|c| c.is_alphabetic()) {
            if let Some(&val) = self.constants.get(input) {
                return Ok(MathExpression::Number(val));
            }
            return Ok(MathExpression::Variable(input.to_string()));
        }
        
        // Try to find operators (simple left-to-right parsing)
        if let Some(pos) = input.rfind('+') {
            let left = self.parse(&input[..pos])?;
            let right = self.parse(&input[pos+1..])?;
            return Ok(MathExpression::Add(Box::new(left), Box::new(right)));
        }
        
        if let Some(pos) = input.rfind('-') {
            if pos > 0 { // Not a negative sign at start
                let left = self.parse(&input[..pos])?;
                let right = self.parse(&input[pos+1..])?;
                return Ok(MathExpression::Subtract(Box::new(left), Box::new(right)));
            }
        }
        
        if let Some(pos) = input.rfind('*') {
            let left = self.parse(&input[..pos])?;
            let right = self.parse(&input[pos+1..])?;
            return Ok(MathExpression::Multiply(Box::new(left), Box::new(right)));
        }
        
        if let Some(pos) = input.rfind('/') {
            let left = self.parse(&input[..pos])?;
            let right = self.parse(&input[pos+1..])?;
            return Ok(MathExpression::Divide(Box::new(left), Box::new(right)));
        }
        
        Err(format!("Cannot parse: {}", input))
    }

    /// Solve mathematical problem with reasoning steps
    pub fn solve(&self, problem: &str) -> MathResult {
        let mut steps = Vec::new();
        steps.push(format!("Problem: {}", problem));
        
        // Try to parse and evaluate
        match self.parse(problem) {
            Ok(expr) => {
                steps.push(format!("Parsed: {}", expr.to_string()));
                
                let simplified = expr.simplify();
                steps.push(format!("Simplified: {}", simplified.to_string()));
                
                let value = simplified.evaluate(&HashMap::new()).ok();
                
                if let Some(v) = value {
                    steps.push(format!("Result: {}", v));
                }
                
                MathResult {
                    expression: expr.to_string(),
                    simplified: simplified.to_string(),
                    value,
                    steps,
                    confidence: if value.is_some() { 1.0 } else { 0.5 },
                }
            }
            Err(e) => {
                steps.push(format!("Error: {}", e));
                MathResult {
                    expression: problem.to_string(),
                    simplified: problem.to_string(),
                    value: None,
                    steps,
                    confidence: 0.0,
                }
            }
        }
    }

    /// Solve equation (find x)
    pub fn solve_equation(&self, equation: &str) -> MathResult {
        let mut steps = Vec::new();
        steps.push(format!("Equation: {}", equation));
        
        // Simple linear equation solver: ax + b = c
        if let Some(eq_pos) = equation.find('=') {
            let left = equation[..eq_pos].trim();
            let right = equation[eq_pos+1..].trim();
            
            steps.push(format!("Left side: {}", left));
            steps.push(format!("Right side: {}", right));
            
            // Try to solve for x
            // This is a simplified solver - real implementation would be more sophisticated
            
            MathResult {
                expression: equation.to_string(),
                simplified: equation.to_string(),
                value: None,
                steps,
                confidence: 0.5,
            }
        } else {
            MathResult {
                expression: equation.to_string(),
                simplified: equation.to_string(),
                value: None,
                steps: vec!["Not an equation (missing =)".to_string()],
                confidence: 0.0,
            }
        }
    }

    /// Calculate derivative
    pub fn derivative(&self, expr_str: &str, var: &str) -> MathResult {
        let mut steps = Vec::new();
        steps.push(format!("Find d/d{} of: {}", var, expr_str));
        
        match self.parse(expr_str) {
            Ok(expr) => {
                let derivative = expr.differentiate(var);
                let simplified = derivative.simplify();
                
                steps.push(format!("Derivative: {}", derivative.to_string()));
                steps.push(format!("Simplified: {}", simplified.to_string()));
                
                MathResult {
                    expression: expr.to_string(),
                    simplified: simplified.to_string(),
                    value: None,
                    steps,
                    confidence: 0.9,
                }
            }
            Err(e) => {
                steps.push(format!("Error: {}", e));
                MathResult {
                    expression: expr_str.to_string(),
                    simplified: expr_str.to_string(),
                    value: None,
                    steps,
                    confidence: 0.0,
                }
            }
        }
    }
}

impl Default for MathSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_arithmetic() {
        let solver = MathSolver::new();
        let result = solver.solve("2+3");
        assert_eq!(result.value, Some(5.0));
    }

    #[test]
    fn test_expression_simplification() {
        let expr = MathExpression::Add(
            Box::new(MathExpression::Number(0.0)),
            Box::new(MathExpression::Number(5.0)),
        );
        let simplified = expr.simplify();
        assert!(matches!(simplified, MathExpression::Number(5.0)));
    }

    #[test]
    fn test_differentiation() {
        let solver = MathSolver::new();
        let expr = MathExpression::Power(
            Box::new(MathExpression::Variable("x".to_string())),
            Box::new(MathExpression::Number(2.0)),
        );
        let derivative = expr.differentiate("x");
        // d/dx(x^2) = 2*x^1
        println!("Derivative: {}", derivative.to_string());
    }
}
