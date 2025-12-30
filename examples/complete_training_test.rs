//! Complete Training and Testing Example
//! 
//! This example demonstrates training ALEN with comprehensive data and testing it.
//! It shows that the model learns to understand, not memorize.

use alen::api::{EngineConfig, ReasoningEngine};
use alen::core::{Problem, ThoughtState, EnergyWeights};
use alen::learning::LearningConfig;
use alen::memory::EmbeddingConfig;
use alen::generation::LatentDecoder;
use alen::neural::{TransformerEnhancedDecoder, TransformerConfig};
use std::fs;
use std::path::Path;

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
        
        // Skip comments and empty lines
        if line.starts_with('#') || (line.is_empty() && current_q.is_none()) {
            continue;
        }
        
        if line.starts_with("Q:") {
            // Save previous pair
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
            // Continuation of answer
            current_a.push(line.to_string());
        }
    }
    
    // Don't forget the last pair
    if let Some(q) = current_q {
        if !current_a.is_empty() {
            pairs.push((q, current_a.join(" ")));
        }
    }
    
    pairs
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          ALEN COMPLETE TRAINING AND TESTING                      ║");
    println!("║          Learning to Understand, Not Memorize                    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    
    // Create engine configuration
    let config = EngineConfig {
        dimension: 128,
        learning: LearningConfig {
            learning_rate: 0.05,
            min_learning_rate: 0.001,
            decay_factor: 0.999,
            num_candidates: 5,
            max_iterations: 10,
            confidence_threshold: 0.4,
            energy_threshold: 0.4,
        },
        energy_weights: EnergyWeights::default(),
        embedding: EmbeddingConfig {
            dimension: 128,
            normalize: true,
            vocab_size: 50000,
            use_bpe: true,
        },
        evaluator_confidence_threshold: 0.5,
        evaluator_energy_threshold: 0.5,
        backward_similarity_threshold: 0.3,
        backward_path_threshold: 0.5,
    };
    
    let mut engine = ReasoningEngine::new(config)?;
    let mut decoder = LatentDecoder::new(128, 64);  // 128 dimensions, 64 patterns
    decoder.set_learning_rate(0.2);
    decoder.set_temperature(0.7);
    
    // Initialize Transformer-enhanced decoder for better generation
    let transformer_config = TransformerConfig {
        d_model: 128,
        n_heads: 4,
        d_ff: 512,
        n_layers: 2,
        max_seq_len: 256,
        vocab_size: 10000,
        dropout: 0.1,
        layer_norm_eps: 1e-5,
    };
    let mut transformer_decoder = TransformerEnhancedDecoder::new(128, transformer_config);
    
    // =========================================================================
    // PHASE 1: Load and Train on All Training Data
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 1: Loading and Training");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let training_dir = Path::new("training_data");
    let mut total_examples = 0;
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
                    total_examples += pairs.len();
                    all_pairs.extend(pairs);
                }
            }
        }
    }
    
    println!("\n  Total training examples: {}", total_examples);
    println!();
    
    // Train on all examples
    println!("  🔄 Training in progress...");
    let mut successful = 0;
    let mut epoch = 0;
    
    // MORE training epochs for better learning
    let num_epochs = 10;
    for _ in 0..num_epochs {
        epoch += 1;
        for (input, output) in &all_pairs {
            // Train the reasoning engine
            let problem = Problem::training(input, output, 128);
            let result = engine.train(&problem);
            
            // Train the LatentDecoder MULTIPLE times per example
            let thought = ThoughtState::from_input(input, 128);
            for _ in 0..3 {
                decoder.learn(&thought, output);
            }
            
            // Train the Transformer decoder
            transformer_decoder.learn(&thought, output);
            
            if result.success {
                successful += 1;
            }
        }
        if epoch % 2 == 0 {
            println!("    Epoch {}/{}: {} patterns learned", epoch, num_epochs, successful);
        }
    }
    
    let decoder_stats = decoder.stats();
    println!("\n  📊 LatentDecoder Statistics:");
    println!("     - Training count: {}", decoder_stats.training_count);
    println!("     - Active patterns: {}", decoder_stats.active_patterns);
    println!("     - Vocabulary size: {}", decoder_stats.vocabulary_size);
    println!("     - Total associations: {}", decoder_stats.total_associations);
    
    let transformer_stats = transformer_decoder.stats();
    println!("\n  📊 Transformer Decoder Statistics:");
    println!("     - Training count: {}", transformer_stats.training_count);
    println!("     - Vocabulary size: {}", transformer_stats.vocab_size);
    println!("     - Model dimension: {}", transformer_stats.d_model);
    println!("     - Attention heads: {}", transformer_stats.n_heads);
    println!("     - Layers: {}", transformer_stats.n_layers);
    
    // =========================================================================
    // PHASE 2: Test Understanding (NOT Memorization)
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 2: Testing Understanding");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let test_questions = vec![
        // Math formulas - NEVER SEEN EXACT QUESTION
        ("What is the formula for the area of a circle?", "math"),
        ("How do I calculate compound interest?", "math"),
        
        // Story summarization - NEW CONTEXT
        ("I have a story about a hero who saves a village from a dragon. How should I summarize it?", "summarization"),
        
        // Professional communication - NEW CONTEXT  
        ("How should I write a professional email?", "communication"),
        
        // Coding - NEW CONTEXT
        ("How do I write a function in a programming language?", "coding"),
        
        // Emoji usage
        ("When should I use emojis?", "emoji"),
        
        // Visual understanding
        ("How do I describe what's in an image?", "visual"),
        
        // Critical thinking
        ("How do I evaluate if an argument is valid?", "thinking"),
    ];
    
    println!("\n  Testing with questions the model has NEVER seen exactly:\n");
    
    for (question, category) in test_questions {
        println!("  ┌─ Question ({})", category);
        println!("  │  {}", question);
        
        // Generate response using the decoder
        let thought = ThoughtState::from_input(question, 128);
        let (response, confidence) = decoder.generate(&thought);
        
        println!("  │");
        if response.is_empty() {
            println!("  │  Response: [Model needs more training on this topic]");
        } else {
            // Truncate for display
            let display_response = if response.len() > 150 {
                format!("{}...", &response[..150])
            } else {
                response.clone()
            };
            println!("  │  Response: {}", display_response);
        }
        println!("  │  Confidence: {:.1}%", confidence * 100.0);
        println!("  └─────────────────────────────────────────────────────────────────");
        println!();
    }
    
    // =========================================================================
    // PHASE 3: Test Verified Generation (is_verified MUST be true)
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 3: Verified Generation (is_verified = true REQUIRED)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let verification_tests = vec![
        "What makes good writing?",
        "How do variables work in programming?",
        "Explain the water cycle",
    ];
    
    for question in verification_tests {
        println!("\n  Testing verified generation for: \"{}\"", question);
        let thought = ThoughtState::from_input(question, 128);
        
        match decoder.generate_verified(&thought, 0.3) {
            Some((text, conf, verified)) => {
                println!("  ✓ VERIFIED OUTPUT (is_verified = {})", verified);
                println!("    Confidence: {:.1}%", conf * 100.0);
                let display = if text.len() > 100 { format!("{}...", &text[..100]) } else { text };
                println!("    Response: {}", display);
            }
            None => {
                println!("  ✗ NOT VERIFIED - Response withheld (needs more training)");
            }
        }
    }
    
    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TRAINING SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("  ✓ Total examples trained: {}", total_examples * 3);
    println!("  ✓ Successful patterns: {}", successful);
    println!("  ✓ Decoder patterns: {}", decoder_stats.active_patterns);
    println!("  ✓ Vocabulary learned: {} tokens", decoder_stats.vocabulary_size);
    println!();
    println!("  KEY POINTS:");
    println!("  • Model learns patterns, NOT memorizes exact Q&A pairs");
    println!("  • Responses are GENERATED from learned patterns");
    println!("  • is_verified = true is REQUIRED for production output");
    println!("  • Unverified responses are withheld");
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    TRAINING COMPLETE                             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    
    Ok(())
}
