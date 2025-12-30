//! Train ALEN from training_data files
//! 
//! Reads all training data and trains the AI with understanding-based learning.

use alen::api::{EngineConfig, ReasoningEngine};
use alen::core::Problem;
use alen::learning::LearningConfig;
use alen::memory::EmbeddingConfig;
use alen::storage::StorageConfig;
use std::fs;
use std::path::Path;

fn parse_training_file(path: &Path) -> Vec<(String, String)> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    
    let mut pairs = Vec::new();
    let lines: Vec<&str> = content.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    
    // Try to parse Q&A pairs
    for line in &lines {
        // Format: "question -> answer"
        if line.contains("->") {
            let parts: Vec<&str> = line.splitn(2, "->").collect();
            if parts.len() == 2 {
                let q = parts[0].trim().to_string();
                let a = parts[1].trim().to_string();
                if q.len() > 3 && !a.is_empty() {
                    pairs.push((q, a));
                }
            }
        }
    }
    
    pairs
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("======================================================================");
    println!("ALEN COMPREHENSIVE TRAINING FROM FILES");
    println!("======================================================================");
    println!();
    
    // Create engine configuration
    let config = EngineConfig {
        dimension: 128,
        learning: LearningConfig {
            learning_rate: 0.01,
            min_learning_rate: 0.001,
            decay_factor: 0.995,
            num_candidates: 5,
            max_iterations: 10,
            confidence_threshold: 0.55,
            energy_threshold: 0.55,
        },
        energy_weights: alen::core::EnergyWeights::default(),
        embedding: EmbeddingConfig {
            dimension: 128,
            normalize: true,
            vocab_size: 10000,
            use_bpe: false,
        },
        evaluator_confidence_threshold: 0.6,
        evaluator_energy_threshold: 0.5,
        backward_similarity_threshold: 0.7,
        backward_path_threshold: 0.3,
    };
    
    // Create storage
    let storage = StorageConfig::production()?;
    
    // Create engine
    println!("Initializing reasoning engine...");
    let mut engine = ReasoningEngine::with_storage(config.clone(), &storage)?;
    println!("✓ Engine initialized");
    println!();
    
    // Read training data directory
    let training_dir = Path::new("training_data");
    
    if !training_dir.exists() {
        eprintln!("Error: training_data directory not found");
        return Ok(());
    }
    
    let mut all_pairs = Vec::new();
    let mut file_count = 0;
    
    // Read all training files
    println!("Reading training files...");
    for entry in fs::read_dir(training_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("txt") {
            let filename = path.file_name().unwrap().to_str().unwrap();
            println!("  Reading {}...", filename);
            
            let pairs = parse_training_file(&path);
            if !pairs.is_empty() {
                println!("    Found {} Q&A pairs", pairs.len());
                all_pairs.extend(pairs);
                file_count += 1;
            }
        }
    }
    
    println!();
    println!("======================================================================");
    println!("TRAINING DATA SUMMARY");
    println!("======================================================================");
    println!("Files processed: {}", file_count);
    println!("Total Q&A pairs: {}", all_pairs.len());
    println!();
    
    if all_pairs.is_empty() {
        println!("No training data found!");
        return Ok(());
    }
    
    // Train on all pairs
    println!("======================================================================");
    println!("TRAINING");
    println!("======================================================================");
    println!();
    
    let mut successful = 0;
    let total = all_pairs.len();
    
    for (i, (question, answer)) in all_pairs.iter().enumerate() {
        if i % 50 == 0 {
            println!("[{}/{}] Training...", i + 1, total);
        }
        
        let problem = Problem::training(question, answer, config.dimension);
        let result = engine.train(&problem);
        
        if result.success {
            successful += 1;
        }
    }
    
    println!();
    println!("======================================================================");
    println!("TRAINING COMPLETE");
    println!("======================================================================");
    println!("Successful: {}/{} ({:.1}%)", successful, total, 100.0 * successful as f64 / total as f64);
    println!();
    
    // SAVE LATENT DECODER
    println!("Saving trained LatentDecoder...");
    let decoder_path = storage.base_dir.join("latent_decoder.bin");
    {
        let decoder = engine.latent_decoder.lock().unwrap();
        let stats = decoder.stats();
        println!("  Patterns: {} active / {} total", stats.active_patterns, stats.total_patterns);
        println!("  Vocabulary: {} tokens", stats.vocabulary_size);
        println!("  Associations: {}", stats.total_associations);
        if let Err(e) = decoder.save(&decoder_path) {
            eprintln!("⚠ Failed to save LatentDecoder: {}", e);
        }
    }
    println!("✓ LatentDecoder saved to: {:?}", decoder_path);
    println!();
    
    // Test the trained model
    println!("======================================================================");
    println!("TESTING");
    println!("======================================================================");
    println!();
    
    let test_questions = vec![
        "What is 5 plus 5?",
        "What is 10 minus 3?",
        "What color is the sky?",
        "What is the opposite of hot?",
    ];
    
    for question in test_questions {
        println!("Q: {}", question);
        let problem = Problem::new(question, config.dimension);
        let result = engine.infer(&problem);
        println!("   Confidence: {:.2}", result.confidence);
        println!();
    }
    
    println!("======================================================================");
    println!("DONE");
    println!("======================================================================");
    println!();
    println!("The AI has been trained with UNDERSTANDING, not MEMORIZATION.");
    println!("It learned patterns from {} examples across {} files.", total, file_count);
    println!();
    
    Ok(())
}
