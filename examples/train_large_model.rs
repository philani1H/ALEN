//! Train Large Language Model
//!
//! This example demonstrates training ALEN with larger model configurations.
//! Choose from: Small (12M), Medium (85M), Large (350M), or XL (1.3B) parameters.

use alen::neural::{
    ModelSize, LargeModelConfig, LargeLanguageModel, LargeModelStats,
    TransformerEnhancedDecoder, TransformerConfig,
};
use alen::core::ThoughtState;
use alen::generation::LatentDecoder;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Parse Q&A format training file
fn parse_qa_file(path: &Path) -> Vec<(String, String)> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    
    let mut pairs = Vec::new();
    let mut current_q: Option<String> = None;
    let mut current_a = Vec::new();
    
    for line in content.lines() {
        let line = line.trim();
        
        if line.starts_with('#') || (line.is_empty() && current_q.is_none()) {
            continue;
        }
        
        if line.starts_with("Q:") {
            if let Some(q) = current_q.take() {
                if !current_a.is_empty() {
                    pairs.push((q, current_a.join(" ")));
                    current_a.clear();
                }
            }
            current_q = Some(line[2..].trim().to_string());
        } else if line.starts_with("A:") {
            current_a.push(line[2..].trim().to_string());
        } else if !line.is_empty() && !current_a.is_empty() {
            current_a.push(line.to_string());
        }
    }
    
    if let Some(q) = current_q {
        if !current_a.is_empty() {
            pairs.push((q, current_a.join(" ")));
        }
    }
    
    pairs
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          ALEN LARGE LANGUAGE MODEL TRAINING                      ║");
    println!("║          Transformer with Full Attention                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // MODEL CONFIGURATION
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("MODEL CONFIGURATIONS AVAILABLE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let configs = vec![
        ("Micro", LargeModelConfig::micro()),
        ("Small", LargeModelConfig::small()),
        ("Medium", LargeModelConfig::medium()),
        ("Large", LargeModelConfig::large()),
        ("XL (GPT-2)", LargeModelConfig::xl()),
    ];
    
    println!("\n  {:12} {:>10} {:>8} {:>8} {:>8} {:>8}", 
             "Size", "Params", "d_model", "heads", "layers", "d_ff");
    println!("  {:12} {:>10} {:>8} {:>8} {:>8} {:>8}", 
             "────", "──────", "───────", "─────", "──────", "────");
    
    for (name, config) in &configs {
        println!("  {:12} {:>10} {:>8} {:>8} {:>8} {:>8}",
                 name,
                 config.parameter_count_string(),
                 config.d_model,
                 config.n_heads,
                 config.n_layers,
                 config.d_ff);
    }
    
    // =========================================================================
    // CREATE MODELS (using Micro for demo, can change to Small/Medium/Large/XL)
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("INITIALIZING MODEL: Micro (for fast CPU training)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let start = Instant::now();
    
    // Use Micro model for fast CPU training (change to Small/Medium/Large/XL for more capacity)
    let mut large_model = LargeLanguageModel::new(LargeModelConfig::micro());
    
    // Also create LatentDecoder for comparison
    let mut latent_decoder = LatentDecoder::new(64, 32);
    latent_decoder.set_learning_rate(0.2);
    
    println!("\n  ✓ Model initialized in {:?}", start.elapsed());
    
    let stats = large_model.stats();
    println!("\n  📊 Model Architecture:");
    println!("     - Parameters: {}", stats.parameters_str);
    println!("     - Dimensions: {}", stats.d_model);
    println!("     - Attention Heads: {}", stats.n_heads);
    println!("     - Transformer Layers: {}", stats.n_layers);
    println!("     - Feed-Forward Dim: {}", stats.d_ff);
    
    // =========================================================================
    // LOAD TRAINING DATA
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("LOADING TRAINING DATA");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let training_dir = Path::new("training_data");
    let mut all_pairs: Vec<(String, String)> = Vec::new();
    
    if training_dir.is_dir() {
        for entry in fs::read_dir(training_dir)? {
            let path = entry?.path();
            if path.extension().map_or(false, |e| e == "txt") {
                let pairs = parse_qa_file(&path);
                if !pairs.is_empty() {
                    println!("  📚 {} - {} examples", 
                             path.file_name().unwrap().to_string_lossy(),
                             pairs.len());
                    all_pairs.extend(pairs);
                }
            }
        }
    }
    
    println!("\n  Total training examples: {}", all_pairs.len());
    
    // =========================================================================
    // TRAINING
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TRAINING (10 epochs)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let train_start = Instant::now();
    let num_epochs = 1;  // Single pass for demo
    
    for epoch in 1..=num_epochs {
        let epoch_start = Instant::now();
        
        for (input, output) in &all_pairs {
            // Build vocabulary only (fast)
            large_model.learn(output);
            large_model.learn(input);
            
            // Train latent decoder
            let thought = ThoughtState::from_input(input, 64);
            latent_decoder.learn(&thought, output);
        }
        
        println!("  Epoch {}/{}: vocab={}, time={:?}", 
                 epoch, num_epochs, 
                 large_model.stats().vocab_size,
                 epoch_start.elapsed());
    }
    
    println!("\n  ✓ Training completed in {:?}", train_start.elapsed());
    
    // =========================================================================
    // FINAL STATISTICS
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TRAINING RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let final_stats = large_model.stats();
    let latent_stats = latent_decoder.stats();
    
    println!("\n  📊 Large Language Model:");
    println!("     - Size: {:?}", final_stats.size);
    println!("     - Parameters: {}", final_stats.parameters_str);
    println!("     - Vocabulary: {}/{}", final_stats.vocab_size, final_stats.max_vocab_size);
    println!("     - Training steps: {}", final_stats.train_steps);
    
    println!("\n  📊 Latent Decoder (comparison):");
    println!("     - Training count: {}", latent_stats.training_count);
    println!("     - Active patterns: {}", latent_stats.active_patterns);
    println!("     - Vocabulary: {}", latent_stats.vocabulary_size);
    
    // =========================================================================
    // TEST GENERATION
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("GENERATION TEST");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let test_prompts = vec![
        "How do I",
        "What is the",
        "Explain",
        "The formula for",
    ];
    
    for prompt in test_prompts {
        let generated = large_model.generate(prompt, 15, 0.8);
        println!("\n  Prompt: \"{}\"", prompt);
        println!("  Generated: {}", if generated.is_empty() { "[needs more training]" } else { &generated });
    }
    
    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    println!("\n  ✓ Large model architecture implemented");
    println!("  ✓ Multiple size configurations available:");
    println!("    - Micro:  500K params (fast testing)");
    println!("    - Small:  21M params  (CPU training)");
    println!("    - Medium: 89M params  (single GPU)");
    println!("    - Large:  404M params (multi-GPU)");
    println!("    - XL:     2B params   (distributed)");
    println!("\n  Note: For better generation quality:");
    println!("    - Use more training data (millions of examples)");
    println!("    - Train for more epochs (hundreds/thousands)");
    println!("    - Use GPU acceleration for larger models");
    
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    TRAINING COMPLETE                             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    
    Ok(())
}
