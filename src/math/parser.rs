//! Mathematical expression parser
//!
//! Converts text like "2 + 2" into an AST

use super::MathAST;

pub struct MathParser;

impl MathParser {
    /// Parse a mathematical expression string into an AST
    pub fn parse(input: &str) -> Result<MathAST, String> {
        let tokens = Self::tokenize(input)?;
        Self::parse_tokens(&tokens)
    }

    /// Tokenize the input string
    fn tokenize(input: &str) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        let mut chars = input.chars().peekable();

        while let Some(&ch) = chars.peek() {
            match ch {
                ' ' | '\t' | '\n' => {
                    chars.next();
                }
                '+' => {
                    tokens.push(Token::Plus);
                    chars.next();
                }
                '-' => {
                    chars.next();
                    // Check if negative number or subtraction
                    if tokens.is_empty() || matches!(tokens.last(), Some(Token::LParen) | Some(Token::Plus) | Some(Token::Minus) | Some(Token::Star) | Some(Token::Slash) | Some(Token::Caret)) {
                        // It's a negative sign
                        tokens.push(Token::Minus);
                    } else {
                        tokens.push(Token::Minus);
                    }
                }
                '*' => {
                    tokens.push(Token::Star);
                    chars.next();
                }
                '/' => {
                    tokens.push(Token::Slash);
                    chars.next();
                }
                '^' => {
                    tokens.push(Token::Caret);
                    chars.next();
                }
                '(' => {
                    tokens.push(Token::LParen);
                    chars.next();
                }
                ')' => {
                    tokens.push(Token::RParen);
                    chars.next();
                }
                '0'..='9' | '.' => {
                    let mut num_str = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_numeric() || c == '.' {
                            num_str.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    let num: f64 = num_str.parse().map_err(|_| format!("Invalid number: {}", num_str))?;
                    tokens.push(Token::Number(num));
                }
                'a'..='z' | 'A'..='Z' => {
                    let mut name = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_alphanumeric() || c == '_' {
                            name.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }

                    // Check if it's a function
                    if chars.peek() == Some(&'(') {
                        tokens.push(Token::Function(name));
                    } else {
                        tokens.push(Token::Variable(name));
                    }
                }
                _ => return Err(format!("Unexpected character: {}", ch)),
            }
        }

        Ok(tokens)
    }

    /// Parse tokens into AST using recursive descent
    fn parse_tokens(tokens: &[Token]) -> Result<MathAST, String> {
        let mut pos = 0;
        Self::parse_expression(tokens, &mut pos)
    }

    /// Parse an expression (lowest precedence)
    fn parse_expression(tokens: &[Token], pos: &mut usize) -> Result<MathAST, String> {
        let mut left = Self::parse_term(tokens, pos)?;

        while *pos < tokens.len() {
            match tokens[*pos] {
                Token::Plus => {
                    *pos += 1;
                    let right = Self::parse_term(tokens, pos)?;
                    left = MathAST::Add(Box::new(left), Box::new(right));
                }
                Token::Minus => {
                    *pos += 1;
                    let right = Self::parse_term(tokens, pos)?;
                    left = MathAST::Sub(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    /// Parse a term (multiplication/division)
    fn parse_term(tokens: &[Token], pos: &mut usize) -> Result<MathAST, String> {
        let mut left = Self::parse_factor(tokens, pos)?;

        while *pos < tokens.len() {
            match tokens[*pos] {
                Token::Star => {
                    *pos += 1;
                    let right = Self::parse_factor(tokens, pos)?;
                    left = MathAST::Mul(Box::new(left), Box::new(right));
                }
                Token::Slash => {
                    *pos += 1;
                    let right = Self::parse_factor(tokens, pos)?;
                    left = MathAST::Div(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    /// Parse a factor (power)
    fn parse_factor(tokens: &[Token], pos: &mut usize) -> Result<MathAST, String> {
        let mut left = Self::parse_unary(tokens, pos)?;

        while *pos < tokens.len() {
            match tokens[*pos] {
                Token::Caret => {
                    *pos += 1;
                    let right = Self::parse_unary(tokens, pos)?;
                    left = MathAST::Pow(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    /// Parse unary operations
    fn parse_unary(tokens: &[Token], pos: &mut usize) -> Result<MathAST, String> {
        if *pos >= tokens.len() {
            return Err("Unexpected end of expression".to_string());
        }

        match &tokens[*pos] {
            Token::Minus => {
                *pos += 1;
                let inner = Self::parse_unary(tokens, pos)?;
                Ok(MathAST::Neg(Box::new(inner)))
            }
            _ => Self::parse_primary(tokens, pos),
        }
    }

    /// Parse primary expressions (numbers, variables, parentheses, functions)
    fn parse_primary(tokens: &[Token], pos: &mut usize) -> Result<MathAST, String> {
        if *pos >= tokens.len() {
            return Err("Unexpected end of expression".to_string());
        }

        match &tokens[*pos] {
            Token::Number(n) => {
                *pos += 1;
                Ok(MathAST::Literal(*n))
            }
            Token::Variable(name) => {
                *pos += 1;
                Ok(MathAST::Variable(name.clone()))
            }
            Token::Function(name) => {
                let func_name = name.clone();
                *pos += 1;

                if *pos >= tokens.len() || tokens[*pos] != Token::LParen {
                    return Err(format!("Expected '(' after function {}", func_name));
                }
                *pos += 1; // Skip '('

                let arg = Self::parse_expression(tokens, pos)?;

                if *pos >= tokens.len() || tokens[*pos] != Token::RParen {
                    return Err("Expected ')' after function argument".to_string());
                }
                *pos += 1; // Skip ')'

                Ok(MathAST::Function(func_name, Box::new(arg)))
            }
            Token::LParen => {
                *pos += 1;
                let expr = Self::parse_expression(tokens, pos)?;

                if *pos >= tokens.len() || tokens[*pos] != Token::RParen {
                    return Err("Expected ')'".to_string());
                }
                *pos += 1;

                Ok(expr)
            }
            _ => Err(format!("Unexpected token: {:?}", tokens[*pos])),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Variable(String),
    Function(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_addition() {
        let ast = MathParser::parse("2 + 2").unwrap();
        assert_eq!(ast.eval(&std::collections::HashMap::new()), Some(4.0));
    }

    #[test]
    fn test_parse_multiplication() {
        let ast = MathParser::parse("3 * 4").unwrap();
        assert_eq!(ast.eval(&std::collections::HashMap::new()), Some(12.0));
    }

    #[test]
    fn test_parse_power() {
        let ast = MathParser::parse("2 ^ 3").unwrap();
        assert_eq!(ast.eval(&std::collections::HashMap::new()), Some(8.0));
    }

    #[test]
    fn test_parse_complex() {
        let ast = MathParser::parse("2 + 3 * 4").unwrap();
        assert_eq!(ast.eval(&std::collections::HashMap::new()), Some(14.0));
    }

    #[test]
    fn test_parse_parentheses() {
        let ast = MathParser::parse("(2 + 3) * 4").unwrap();
        assert_eq!(ast.eval(&std::collections::HashMap::new()), Some(20.0));
    }
}
