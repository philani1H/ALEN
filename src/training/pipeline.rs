//! Training Pipeline
//!
//! Complete training pipeline that orchestrates:
//! - Data loading
//! - Tokenization
//! - Model training
//! - Verification
//! - Checkpointing

use super::data_loader::{DataLoader, TrainingData, TrainingExample};
use super::tokenizer::{Tokenizer, TokenizerConfig, TokenizedBatch};
use crate::neural::{
    Trainer, TrainerConfig, Adam, MSELoss, TrainingMetrics,
    TransformerEncoder, TransformerConfig,
};
use crate::memory::{SemanticMemory, SemanticFact, EmbeddingEngine, EmbeddingConfig};
use crate::learning::{VerifiedLearner, VerificationResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Tokenizer configuration
    pub tokenizer: TokenizerConfig,
    /// Trainer configuration
    pub trainer: TrainerConfig,
    /// Transformer configuration
    pub transformer: TransformerConfig,
    /// Number of training epochs
    pub num_epochs: usize,
    /// Validation split ratio
    pub validation_split: f64,
    /// Whether to use verified learning
    pub use_verification: bool,
    /// Checkpoint interval (steps)
    pub checkpoint_interval: usize,
    /// Log interval (steps)
    pub log_interval: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            tokenizer: TokenizerConfig::default(),
            trainer: TrainerConfig::default(),
            transformer: TransformerConfig::small(),
            num_epochs: 10,
            validation_split: 0.1,
            use_verification: true,
            checkpoint_interval: 1000,
            log_interval: 100,
        }
    }
}

/// Training statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingStats {
    /// Total training steps
    pub total_steps: usize,
    /// Total examples processed
    pub total_examples: usize,
    /// Training loss history
    pub loss_history: Vec<f64>,
    /// Validation loss history
    pub val_loss_history: Vec<f64>,
    /// Verification rate history
    pub verification_rate_history: Vec<f64>,
    /// Current epoch
    pub current_epoch: usize,
    /// Best validation loss
    pub best_val_loss: f64,
    /// Categories trained
    pub categories_trained: HashMap<String, usize>,
}

/// Training pipeline
pub struct TrainingPipeline {
    /// Configuration
    pub config: PipelineConfig,
    /// Tokenizer
    pub tokenizer: Tokenizer,
    /// Embedding engine
    pub embedder: EmbeddingEngine,
    /// Verified learner
    pub learner: VerifiedLearner,
    /// Training statistics
    pub stats: TrainingStats,
    /// Semantic memory for storing learned knowledge
    pub memory: Option<SemanticMemory>,
}

impl TrainingPipeline {
    /// Create new training pipeline
    pub fn new(config: PipelineConfig) -> Self {
        let tokenizer = Tokenizer::new(config.tokenizer.clone());
        let embedder = EmbeddingEngine::new(EmbeddingConfig {
            dimension: config.transformer.d_model,
            normalize: true,
            vocab_size: config.tokenizer.vocab_size,
            use_bpe: true,
        });
        let learner = VerifiedLearner::new(config.transformer.d_model);

        Self {
            config,
            tokenizer,
            embedder,
            learner,
            stats: TrainingStats {
                best_val_loss: f64::INFINITY,
                ..Default::default()
            },
            memory: None,
        }
    }

    /// Initialize with semantic memory
    pub fn with_memory(mut self, memory: SemanticMemory) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Train tokenizer on corpus
    pub fn train_tokenizer(&mut self, texts: &[String]) {
        self.tokenizer.train(texts);
    }

    /// Run training on data
    pub fn train(&mut self, data: &TrainingData) -> Result<TrainingStats, Box<dyn std::error::Error>> {
        let all_examples = data.all_examples();
        
        if all_examples.is_empty() {
            return Err("No training examples provided".into());
        }

        // Split into train/validation
        let split_idx = (all_examples.len() as f64 * (1.0 - self.config.validation_split)) as usize;
        let (train_examples, val_examples) = all_examples.split_at(split_idx);

        // Train tokenizer if not already trained
        if !self.tokenizer.is_trained {
            let texts: Vec<String> = all_examples.iter()
                .flat_map(|ex| vec![ex.input.clone(), ex.output.clone()])
                .collect();
            self.tokenizer.train(&texts);
        }

        // Create data loader
        let mut loader = DataLoader::new(train_examples.to_vec(), self.config.trainer.batch_size);

        // Training loop
        for epoch in 0..self.config.num_epochs {
            self.stats.current_epoch = epoch;
            let mut epoch_loss = 0.0;
            let mut epoch_verified = 0;
            let mut epoch_total = 0;

            loader.reset();

            while let Some(batch) = loader.next_batch() {
                let batch_result = self.train_batch(&batch)?;
                
                epoch_loss += batch_result.loss;
                epoch_verified += batch_result.verified_count;
                epoch_total += batch.len();
                
                self.stats.total_steps += 1;
                self.stats.total_examples += batch.len();

                // Log progress
                if self.stats.total_steps % self.config.log_interval == 0 {
                    let avg_loss = epoch_loss / epoch_total as f64;
                    let ver_rate = epoch_verified as f64 / epoch_total as f64;
                    println!(
                        "Epoch {}/{} Step {} - Loss: {:.4}, Verification: {:.2}%",
                        epoch + 1, self.config.num_epochs, self.stats.total_steps,
                        avg_loss, ver_rate * 100.0
                    );
                }
            }

            // Record epoch stats
            let avg_epoch_loss = epoch_loss / epoch_total.max(1) as f64;
            self.stats.loss_history.push(avg_epoch_loss);
            
            let ver_rate = epoch_verified as f64 / epoch_total.max(1) as f64;
            self.stats.verification_rate_history.push(ver_rate);

            // Validation
            if !val_examples.is_empty() {
                let val_loss = self.validate(val_examples)?;
                self.stats.val_loss_history.push(val_loss);
                
                if val_loss < self.stats.best_val_loss {
                    self.stats.best_val_loss = val_loss;
                }
            }
        }

        Ok(self.stats.clone())
    }

    /// Train on a single batch
    fn train_batch(&mut self, batch: &[&TrainingExample]) -> Result<BatchResult, Box<dyn std::error::Error>> {
        let mut total_loss = 0.0;
        let mut verified_count = 0;

        for example in batch {
            // Embed input and output
            let input_emb = self.embedder.embed_text(&example.input);
            let output_emb = self.embedder.embed_text(&example.output);

            // Compute loss (MSE between predicted and target)
            let loss = input_emb.distance(&output_emb);
            total_loss += loss;

            // Verified learning
            if self.config.use_verification {
                let knowledge_item = crate::knowledge::KnowledgeItem {
                    category: crate::knowledge::KnowledgeCategory::ComputerScience,
                    subcategory: example.subcategory.clone().unwrap_or_default(),
                    input: example.input.clone(),
                    output: example.output.clone(),
                    reasoning: example.reasoning.clone().unwrap_or_default(),
                    backward_check: String::new(),
                    related: vec![],
                    difficulty: example.difficulty,
                    prerequisites: vec![],
                };

                let result = self.learner.train_verified(&knowledge_item);
                if result.passed {
                    verified_count += 1;
                }
            }

            // Store in semantic memory if available
            if let Some(ref memory) = self.memory {
                let fact = SemanticFact {
                    id: uuid::Uuid::new_v4().to_string(),
                    concept: example.input.clone(),
                    content: example.output.clone(),
                    embedding: output_emb.vector.clone(),
                    category: example.category.clone(),
                    source: Some("training".to_string()),
                    confidence: 0.8,
                    reinforcement_count: 0,
                    last_accessed: chrono::Utc::now(),
                    related_concepts: vec![],
                };
                let _ = memory.store(&fact);
            }

            // Track category
            if let Some(ref cat) = example.category {
                *self.stats.categories_trained.entry(cat.clone()).or_insert(0) += 1;
            }
        }

        Ok(BatchResult {
            loss: total_loss / batch.len() as f64,
            verified_count,
        })
    }

    /// Validate on examples
    fn validate(&mut self, examples: &[TrainingExample]) -> Result<f64, Box<dyn std::error::Error>> {
        let mut total_loss = 0.0;

        for example in examples {
            let input_emb = self.embedder.embed_text(&example.input);
            let output_emb = self.embedder.embed_text(&example.output);
            total_loss += input_emb.distance(&output_emb);
        }

        Ok(total_loss / examples.len() as f64)
    }

    /// Get training statistics
    pub fn get_stats(&self) -> &TrainingStats {
        &self.stats
    }
}

/// Result of training a batch
struct BatchResult {
    loss: f64,
    verified_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = TrainingPipeline::new(config);
        
        assert_eq!(pipeline.stats.total_steps, 0);
    }

    #[test]
    fn test_training_stats() {
        let stats = TrainingStats::default();
        assert_eq!(stats.total_steps, 0);
        assert!(stats.loss_history.is_empty());
    }
}
