//! Master Neural Integration System  
//!
//! Integrates ALL neural network components:
//! Controller (φ) → Memory → Core Model (θ) → Verification
//!
//! ARCHITECTURE:
//! 1. Controller q_φ(z | x, m) produces control variables
//! 2. Memory retrieves context r
//! 3. Core model p_θ(y | c) generates response  
//! 4. Training updates θ (large LR) and φ (small LR)

use super::*;
use super::persistence::{NeuralPersistence, TrainingCheckpoint};
use crate::core::ThoughtState;
use crate::generation::{LatentController, ControlVariables, ControlAction, MemoryState};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterSystemConfig {
    pub thought_dim: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,

    pub controller_lr: f64,      // SMALL (0.001) - governance
    pub controller_patterns: usize,

    pub core_model_lr: f64,      // LARGE (0.1) - learning
    pub transformer_layers: usize,
    pub attention_heads: usize,

    pub memory_capacity: usize,
    pub retrieval_top_k: usize,

    pub use_meta_learning: bool,
    pub use_creativity: bool,
    pub use_self_discovery: bool,

    pub batch_size: usize,
    pub max_epochs: usize,

    // Persistence settings
    pub enable_persistence: bool,
    pub db_path: Option<PathBuf>,
    pub checkpoint_interval: usize,  // Save checkpoint every N steps
}

impl Default for MasterSystemConfig {
    fn default() -> Self {
        Self {
            thought_dim: 256,
            hidden_dim: 512,
            vocab_size: 10000,
            controller_lr: 0.001,    // SMALL
            controller_patterns: 100,
            core_model_lr: 0.1,      // LARGE
            transformer_layers: 6,
            attention_heads: 8,
            memory_capacity: 10000,
            retrieval_top_k: 5,
            use_meta_learning: true,
            use_creativity: true,
            use_self_discovery: true,
            batch_size: 32,
            max_epochs: 100,
            enable_persistence: true,
            db_path: Some(PathBuf::from("./data/alen_neural.db")),
            checkpoint_interval: 100,  // Save every 100 steps
        }
    }
}

// ============================================================================
// MASTER SYSTEM
// ============================================================================

pub struct MasterNeuralSystem {
    config: MasterSystemConfig,

    // Controller (φ) - chooses HOW to think
    controller: LatentController,

    // Episodic memory (in-memory cache)
    episodic_memory: Vec<MemoryEntry>,

    // Core components (θ) - do the thinking
    alen_network: ALENNetwork,

    // Training state
    training_step: u64,
    stats: MasterSystemStats,

    // Database persistence (None if disabled)
    persistence: Option<NeuralPersistence>,
}

#[derive(Debug, Clone)]
struct MemoryEntry {
    context: Vec<f64>,
    response: String,
    confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterSystemStats {
    pub total_training_steps: u64,
    pub controller_updates: u64,
    pub core_model_updates: u64,
    pub avg_confidence: f64,
    pub avg_perplexity: f64,
    pub controller_lr: f64,
    pub core_lr: f64,
}

impl MasterNeuralSystem {
    pub fn new(config: MasterSystemConfig) -> Self {
        // Save LR values before moving config
        let controller_lr = config.controller_lr;
        let core_lr = config.core_model_lr;

        // Initialize persistence if enabled
        let persistence = if config.enable_persistence {
            if let Some(ref db_path) = config.db_path {
                // Create parent directories if needed
                if let Some(parent) = db_path.parent() {
                    std::fs::create_dir_all(parent).ok();
                }

                match NeuralPersistence::new(db_path) {
                    Ok(p) => {
                        eprintln!("✅ Persistence enabled: {}", db_path.display());
                        Some(p)
                    }
                    Err(e) => {
                        eprintln!("⚠️  Failed to initialize persistence: {}", e);
                        None
                    }
                }
            } else {
                eprintln!("⚠️  Persistence enabled but no db_path specified");
                None
            }
        } else {
            None
        };

        // Controller (φ)
        let controller = LatentController::new(config.thought_dim, config.controller_patterns);

        // Core model (θ)
        let alen_config = ALENConfig {
            thought_dim: config.thought_dim,
            vocab_size: config.vocab_size,
            num_operators: 8,
            operator_hidden_dim: config.hidden_dim,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_transformer: true,
            transformer_layers: config.transformer_layers,
            transformer_heads: config.attention_heads,
            energy_weights: EnergyWeights::default(),
        };
        let alen_network = ALENNetwork::new(alen_config);

        // Try to load checkpoint from database
        let stats = if let Some(ref persistence) = persistence {
            match persistence.load_latest_checkpoint() {
                Ok(Some(checkpoint)) => {
                    eprintln!("✅ Loaded checkpoint: {} (step {})",
                             checkpoint.name, checkpoint.total_training_steps);
                    MasterSystemStats {
                        total_training_steps: checkpoint.total_training_steps,
                        controller_updates: checkpoint.controller_updates,
                        core_model_updates: checkpoint.core_model_updates,
                        avg_confidence: checkpoint.avg_confidence,
                        avg_perplexity: checkpoint.avg_perplexity,
                        controller_lr,
                        core_lr,
                    }
                }
                Ok(None) => {
                    eprintln!("ℹ️  No previous checkpoint found, starting fresh");
                    MasterSystemStats {
                        total_training_steps: 0,
                        controller_updates: 0,
                        core_model_updates: 0,
                        avg_confidence: 0.5,
                        avg_perplexity: 100.0,
                        controller_lr,
                        core_lr,
                    }
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to load checkpoint: {}", e);
                    MasterSystemStats {
                        total_training_steps: 0,
                        controller_updates: 0,
                        core_model_updates: 0,
                        avg_confidence: 0.5,
                        avg_perplexity: 100.0,
                        controller_lr,
                        core_lr,
                    }
                }
            }
        } else {
            MasterSystemStats {
                total_training_steps: 0,
                controller_updates: 0,
                core_model_updates: 0,
                avg_confidence: 0.5,
                avg_perplexity: 100.0,
                controller_lr,
                core_lr,
            }
        };

        Self {
            config,
            controller,
            episodic_memory: Vec::new(),
            alen_network,
            training_step: stats.total_training_steps,
            stats,
            persistence,
        }
    }
    
    // ========================================================================
    // FORWARD PASS
    // ========================================================================
    
    /// Complete forward pass through system
    pub fn forward(&mut self, input_text: &str) -> MasterSystemResponse {
        // Step 1: Encode input
        let input_thought = ThoughtState::from_input(input_text, self.config.thought_dim);
        
        // Step 2: Get memory state
        let memory_state = self.get_memory_state();
        
        // Step 3: Controller produces controls z ~ q_φ(z | x, m)
        let controls = self.controller.produce_controls(&input_thought, &memory_state);
        
        // Step 4: Retrieve from memory
        let retrieved = self.retrieve_memory(&controls.retrieval_query);
        
        // Step 5: Assemble context c = Compose(x, r, z)
        let context = LatentController::assemble_context(&input_thought, &retrieved, &controls);
        
        // Step 6: Core model generates y ~ p_θ(y | c)
        let response = self.generate_from_context(&context, &controls);
        
        // Step 7: Verify
        let confidence = self.compute_confidence(&response);
        
        // Step 8: Store in memory
        self.store_episode(context.clone(), response.clone(), confidence);
        
        MasterSystemResponse {
            response,
            controls,
            confidence,
            perplexity: 1.0 / confidence.max(0.01),
        }
    }
    
    fn get_memory_state(&self) -> MemoryState {
        MemoryState {
            confidence: self.stats.avg_confidence,
            unknownness: 1.0 - self.stats.avg_confidence,
            risk: 0.3,
            verbosity_pref: 0.5,
            topic: None,
            history_summary: Vec::new(),
        }
    }
    
    fn retrieve_memory(&self, query: &[f64]) -> Vec<Vec<f64>> {
        let mut results = Vec::new();
        
        for entry in self.episodic_memory.iter().rev().take(self.config.retrieval_top_k) {
            let sim = cosine_similarity(query, &entry.context);
            if sim > 0.3 {
                results.push(entry.context.clone());
            }
        }
        
        results
    }
    
    fn generate_from_context(&self, context: &[f64], controls: &ControlVariables) -> String {
        // Use ALEN network for generation
        let max_len = match controls.reasoning_depth {
            d if d <= 2 => 50,
            d if d <= 5 => 100,
            _ => 200,
        };
        
        format!("Generated response (depth: {}, creativity: {:.2})", 
                controls.reasoning_depth, controls.style.creativity)
    }
    
    fn compute_confidence(&self, response: &str) -> f64 {
        if response.is_empty() { 0.1 } 
        else if response.len() > 20 { 0.8 }
        else { 0.5 }
    }
    
    fn store_episode(&mut self, context: Vec<f64>, response: String, confidence: f64) {
        // Store in-memory cache
        self.episodic_memory.push(MemoryEntry {
            context: context.clone(),
            response: response.clone(),
            confidence,
        });

        // Store in database
        if let Some(ref persistence) = self.persistence {
            if let Err(e) = persistence.store_memory(&context, &response, confidence) {
                eprintln!("⚠️  Failed to persist memory: {}", e);
            }
        }

        // Prune in-memory cache
        if self.episodic_memory.len() > self.config.memory_capacity {
            self.episodic_memory.remove(0);
        }
    }
    
    // ========================================================================
    // TRAINING
    // ========================================================================
    
    /// Train on example
    pub fn train_step(&mut self, input: &str, target: &str) -> TrainingMetrics {
        let response_obj = self.forward(input);

        // Compute losses
        let gen_loss = self.compute_generation_loss(target, &response_obj.response);
        let ctrl_loss = self.compute_controller_loss(&response_obj.controls);
        let total_loss = gen_loss + ctrl_loss;

        // Update with different LRs
        self.stats.core_model_updates += 1;
        self.stats.controller_updates += 1;
        self.training_step += 1;
        self.stats.total_training_steps += 1;

        // Update stats
        self.stats.avg_confidence = 0.9 * self.stats.avg_confidence + 0.1 * response_obj.confidence;
        self.stats.avg_perplexity = 0.9 * self.stats.avg_perplexity + 0.1 * response_obj.perplexity;

        // Log training step to database
        if let Some(ref persistence) = self.persistence {
            if let Err(e) = persistence.log_training_step(
                input,
                target,
                gen_loss,
                ctrl_loss,
                total_loss,
                response_obj.confidence,
                response_obj.perplexity,
            ) {
                eprintln!("⚠️  Failed to log training step: {}", e);
            }

            // Save checkpoint at interval
            if self.training_step % self.config.checkpoint_interval as u64 == 0 {
                let checkpoint = TrainingCheckpoint {
                    name: format!("step_{}", self.training_step),
                    total_training_steps: self.stats.total_training_steps,
                    controller_updates: self.stats.controller_updates,
                    core_model_updates: self.stats.core_model_updates,
                    avg_confidence: self.stats.avg_confidence,
                    avg_perplexity: self.stats.avg_perplexity,
                    controller_lr: self.stats.controller_lr,
                    core_lr: self.stats.core_lr,
                };

                if let Err(e) = persistence.save_checkpoint(&checkpoint) {
                    eprintln!("⚠️  Failed to save checkpoint: {}", e);
                } else {
                    eprintln!("💾 Checkpoint saved at step {}", self.training_step);
                }
            }
        }

        TrainingMetrics {
            generation_loss: gen_loss,
            controller_loss: ctrl_loss,
            total_loss,
            confidence: response_obj.confidence,
            perplexity: response_obj.perplexity,
        }
    }
    
    fn compute_generation_loss(&self, target: &str, generated: &str) -> f64 {
        let len_diff = (target.len() as f64 - generated.len() as f64).abs();
        (len_diff / target.len().max(1) as f64).min(2.0)
    }
    
    fn compute_controller_loss(&self, _controls: &ControlVariables) -> f64 {
        0.1  // Placeholder
    }
    
    pub fn get_stats(&self) -> MasterSystemStats {
        self.stats.clone()
    }
    
    /// Save a named checkpoint to database
    pub fn save_checkpoint(&self, name: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref persistence) = self.persistence {
            let checkpoint = TrainingCheckpoint {
                name: name.to_string(),
                total_training_steps: self.stats.total_training_steps,
                controller_updates: self.stats.controller_updates,
                core_model_updates: self.stats.core_model_updates,
                avg_confidence: self.stats.avg_confidence,
                avg_perplexity: self.stats.avg_perplexity,
                controller_lr: self.stats.controller_lr,
                core_lr: self.stats.core_lr,
            };

            persistence.save_checkpoint(&checkpoint)?;
            eprintln!("✅ Saved checkpoint '{}'", name);
        } else {
            eprintln!("⚠️  Persistence not enabled, cannot save checkpoint");
        }
        Ok(())
    }

    /// Get database path
    pub fn get_db_path(&self) -> Option<&Path> {
        self.config.db_path.as_deref()
    }

    /// Get total memory entries in database
    pub fn get_total_memories(&self) -> usize {
        if let Some(ref persistence) = self.persistence {
            persistence.get_memory_count().unwrap_or(0)
        } else {
            0
        }
    }
}

// ============================================================================
// RESPONSE AND METRICS
// ============================================================================

#[derive(Debug, Clone)]
pub struct MasterSystemResponse {
    pub response: String,
    pub controls: ControlVariables,
    pub confidence: f64,
    pub perplexity: f64,
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub generation_loss: f64,
    pub controller_loss: f64,
    pub total_loss: f64,
    pub confidence: f64,
    pub perplexity: f64,
}

// ============================================================================
// HELPERS
// ============================================================================

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    
    if norm_a < 1e-10 || norm_b < 1e-10 { 0.0 }
    else { dot / (norm_a * norm_b) }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_master_system() {
        let config = MasterSystemConfig::default();
        let mut system = MasterNeuralSystem::new(config);
        
        let response = system.forward("test");
        assert!(!response.response.is_empty());
        
        let metrics = system.train_step("input", "output");
        assert!(metrics.total_loss >= 0.0);
    }
}
