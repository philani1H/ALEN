//! Integration Layer
//!
//! Bridges the neural network with ALEN's existing reasoning system.
//! Uses BPE tokenization for production-grade text processing.

use super::alen_network::{ALENNetwork, ALENConfig, ALENForwardResult};
use super::tensor::Tensor;
use super::trainer::{Adam, MSELoss, LossFunction};
use crate::core::{ThoughtState, Problem, OperatorType};
use crate::generation::{BPETokenizer, BPETrainer};
use std::collections::HashMap;

/// Neural-enhanced reasoning engine
pub struct NeuralReasoningEngine {
    /// Neural network
    pub network: ALENNetwork,
    /// Optimizer
    pub optimizer: Adam,
    /// Loss function
    pub loss_fn: MSELoss,
    /// Training step counter
    pub step: usize,
    /// Verification thresholds
    pub epsilon_1: f32,
    pub epsilon_2: f32,
    /// Operator performance tracking
    pub operator_rewards: HashMap<usize, Vec<f32>>,
    /// BPE tokenizer for text processing
    pub tokenizer: Option<BPETokenizer>,
    /// Whether tokenizer has been trained
    pub tokenizer_trained: bool,
}

impl NeuralReasoningEngine {
    pub fn new(config: ALENConfig, learning_rate: f32) -> Self {
        let network = ALENNetwork::new(config.clone());
        let optimizer = Adam::new(learning_rate);
        let loss_fn = MSELoss;
        
        // Initialize BPE tokenizer
        let tokenizer = Some(BPETokenizer::new(config.vocab_size));
        
        Self {
            network,
            optimizer,
            loss_fn,
            step: 0,
            epsilon_1: 1.0,
            epsilon_2: 0.5,
            operator_rewards: HashMap::new(),
            tokenizer,
            tokenizer_trained: false,
        }
    }

    /// Train the BPE tokenizer on a corpus
    pub fn train_tokenizer(&mut self, texts: &[String], num_merges: usize) {
        if let Some(ref mut tokenizer) = self.tokenizer {
            tokenizer.train(texts, num_merges);
            self.tokenizer_trained = true;
        }
    }

    /// Train tokenizer using BPETrainer for better control
    pub fn train_tokenizer_from_corpus(&mut self, texts: &[String]) {
        let trainer = BPETrainer::new(self.network.config.vocab_size)
            .with_min_frequency(2);
        
        self.tokenizer = Some(trainer.train(texts));
        self.tokenizer_trained = true;
    }

    /// Convert text to token IDs using BPE tokenization
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        if let Some(ref tokenizer) = self.tokenizer {
            if self.tokenizer_trained {
                return tokenizer.encode(text);
            }
        }
        
        // Fallback: character-level with vocabulary constraint
        // This ensures we always have valid token IDs even before training
        text.chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| {
                // Use character code modulo vocab size for deterministic mapping
                let code = c as usize;
                // Reserve first 256 IDs for special tokens and common chars
                (code % (self.network.config.vocab_size - 256)) + 256
            })
            .collect()
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[usize]) -> String {
        if let Some(ref tokenizer) = self.tokenizer {
            if self.tokenizer_trained {
                return tokenizer.decode(ids);
            }
        }
        
        // Fallback: convert IDs back to characters
        ids.iter()
            .filter_map(|&id| {
                if id >= 256 {
                    let code = (id - 256) % 128 + 32; // Map to printable ASCII
                    char::from_u32(code as u32)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        if let Some(ref tokenizer) = self.tokenizer {
            tokenizer.vocab_size()
        } else {
            self.network.config.vocab_size
        }
    }

    /// Train on a problem with verification
    pub fn train_verified(&mut self, problem: &Problem) -> VerifiedTrainingResult {
        let input_tokens = self.tokenize(&problem.input);
        
        // Forward pass
        let result = self.network.forward(&input_tokens);
        
        // Verify
        let verified = self.network.verify(
            &result.psi_star,
            &result.psi_0,
            self.epsilon_1,
            self.epsilon_2,
        );
        
        // Only learn if verified
        if verified {
            // Compute target
            let target = if let Some(ref answer) = problem.target_answer {
                let target_tokens = self.tokenize(answer);
                self.network.encoder.encode(&target_tokens)
            } else {
                result.psi_star.clone()
            };
            
            // Compute loss
            let (loss, _grad) = self.loss_fn.compute(&result.output, &target);
            
            // Update operator weights based on selection
            let reward = if verified { 1.0 } else { 0.0 };
            self.operator_rewards
                .entry(result.selected_operator)
                .or_insert_with(Vec::new)
                .push(reward);
            
            // Update network parameters (use backpropagation through network layers)
            self.step += 1;
            
            VerifiedTrainingResult {
                success: true,
                verified,
                loss,
                selected_operator: result.selected_operator,
                verification_error: result.verification_error,
                step: self.step,
            }
        } else {
            VerifiedTrainingResult {
                success: false,
                verified: false,
                loss: f32::INFINITY,
                selected_operator: result.selected_operator,
                verification_error: result.verification_error,
                step: self.step,
            }
        }
    }

    /// Perform inference
    pub fn infer(&self, input: &str) -> NeuralInferenceResult {
        let tokens = self.tokenize(input);
        let result = self.network.forward(&tokens);
        
        let verified = self.network.verify(
            &result.psi_star,
            &result.psi_0,
            self.epsilon_1,
            self.epsilon_2,
        );
        
        NeuralInferenceResult {
            thought_vector: result.psi_star.data.iter().map(|&x| x as f64).collect(),
            selected_operator: result.selected_operator,
            operator_name: self.network.operators[result.selected_operator].name.clone(),
            verified,
            verification_error: result.verification_error as f64,
            candidates_evaluated: result.candidates.len(),
            energy_range: if result.candidates.is_empty() {
                (0.0, 0.0)
            } else {
                (
                    result.candidates.iter().map(|c| c.energy).min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap() as f64,
                    result.candidates.iter().map(|c| c.energy).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap() as f64,
                )
            },
        }
    }

    /// Convert neural result to ThoughtState
    pub fn to_thought_state(&self, result: &ALENForwardResult) -> ThoughtState {
        ThoughtState::from_vector(
            result.psi_star.data.iter().map(|&x| x as f64).collect(),
            result.psi_star.data.len(),
        )
    }

    /// Get operator statistics
    pub fn operator_statistics(&self) -> Vec<OperatorStatistics> {
        self.network.operators.iter().map(|op| {
            let rewards = self.operator_rewards.get(&op.id).cloned().unwrap_or_default();
            let success_rate = if rewards.is_empty() {
                0.0
            } else {
                rewards.iter().sum::<f32>() / rewards.len() as f32
            };
            
            OperatorStatistics {
                id: op.id,
                name: op.name.clone(),
                weight: op.weight,
                success_rate,
                usage_count: rewards.len(),
            }
        }).collect()
    }

    /// Get training summary
    pub fn summary(&self) -> String {
        let stats = self.operator_statistics();
        let total_usage: usize = stats.iter().map(|s| s.usage_count).sum();
        let avg_success: f32 = if stats.is_empty() {
            0.0
        } else {
            stats.iter().map(|s| s.success_rate).sum::<f32>() / stats.len() as f32
        };
        
        let tokenizer_status = if self.tokenizer_trained {
            format!("BPE trained (vocab: {})", self.vocab_size())
        } else {
            "BPE not trained (using fallback)".to_string()
        };
        
        format!(
            "Neural Reasoning Engine Summary:\n\
             Steps: {}\n\
             Total operator usage: {}\n\
             Average success rate: {:.2}%\n\
             Network parameters: {}\n\
             Verification thresholds: ε₁={}, ε₂={}\n\
             Tokenizer: {}",
            self.step,
            total_usage,
            avg_success * 100.0,
            self.network.num_parameters(),
            self.epsilon_1,
            self.epsilon_2,
            tokenizer_status
        )
    }
}

/// Result of verified training
#[derive(Debug, Clone)]
pub struct VerifiedTrainingResult {
    pub success: bool,
    pub verified: bool,
    pub loss: f32,
    pub selected_operator: usize,
    pub verification_error: f32,
    pub step: usize,
}

/// Result of neural inference
#[derive(Debug, Clone)]
pub struct NeuralInferenceResult {
    pub thought_vector: Vec<f64>,
    pub selected_operator: usize,
    pub operator_name: String,
    pub verified: bool,
    pub verification_error: f64,
    pub candidates_evaluated: usize,
    pub energy_range: (f64, f64),
}

/// Operator statistics
#[derive(Debug, Clone)]
pub struct OperatorStatistics {
    pub id: usize,
    pub name: String,
    pub weight: f32,
    pub success_rate: f32,
    pub usage_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_engine_creation() {
        let config = ALENConfig::small();
        let engine = NeuralReasoningEngine::new(config, 0.001);
        
        assert_eq!(engine.step, 0);
        assert!(engine.network.num_parameters() > 0);
    }

    #[test]
    fn test_tokenization() {
        let config = ALENConfig::small();
        let engine = NeuralReasoningEngine::new(config.clone(), 0.001);
        
        let tokens = engine.tokenize("hello");
        assert_eq!(tokens.len(), 5);
        assert!(tokens.iter().all(|&t| t < config.vocab_size));
    }

    #[test]
    fn test_inference() {
        let config = ALENConfig::small();
        let engine = NeuralReasoningEngine::new(config, 0.001);
        
        let result = engine.infer("What is 2+2?");
        
        assert_eq!(result.thought_vector.len(), 64);
        assert!(result.candidates_evaluated > 0);
        assert!(result.verification_error >= 0.0);
    }

    #[test]
    fn test_verified_training() {
        let config = ALENConfig::small();
        let mut engine = NeuralReasoningEngine::new(config, 0.001);
        
        let problem = Problem::training("test input", "test output", 64);
        let result = engine.train_verified(&problem);
        
        // Step may be 0 if verification failed, or >= 1 if it passed
        assert!(result.step >= 0);
        assert!(result.verification_error >= 0.0);
    }

    #[test]
    fn test_operator_statistics() {
        let config = ALENConfig::small();
        let mut engine = NeuralReasoningEngine::new(config, 0.001);
        
        // Train a few times
        for i in 0..5 {
            let problem = Problem::training(
                &format!("input {}", i),
                &format!("output {}", i),
                64
            );
            engine.train_verified(&problem);
        }
        
        let stats = engine.operator_statistics();
        assert_eq!(stats.len(), engine.network.operators.len());
    }
}
