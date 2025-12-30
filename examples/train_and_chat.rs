//! Train and Chat with ALEN
//!
//! This example trains the model on all available data and provides an interactive chat.

use alen::core::ThoughtState;
use alen::generation::LatentDecoder;
use alen::neural::{LargeModelConfig, LargeLanguageModel};
use std::fs;
use std::io::{self, BufRead, Write};
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
    println!("║              ALEN - Train and Chat                               ║");
    println!("║              Neural Language Model                               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // INITIALIZE MODELS
    // =========================================================================
    println!("🔧 Initializing models...");
    let start = Instant::now();
    
    // Use micro config for fast training
    let mut large_model = LargeLanguageModel::new(LargeModelConfig::micro());
    let mut latent_decoder = LatentDecoder::new(128, 64);
    latent_decoder.set_learning_rate(0.3);
    
    println!("   ✓ Models initialized in {:?}", start.elapsed());

    // =========================================================================
    // LOAD TRAINING DATA
    // =========================================================================
    println!("\n📚 Loading training data...");
    
    let training_dir = Path::new("training_data");
    let mut all_pairs: Vec<(String, String)> = Vec::new();
    
    if training_dir.is_dir() {
        for entry in fs::read_dir(training_dir)? {
            let path = entry?.path();
            if path.extension().map_or(false, |e| e == "txt") {
                let pairs = parse_qa_file(&path);
                all_pairs.extend(pairs);
            }
        }
    }
    
    // Also load from data/ directory
    let data_dir = Path::new("data");
    if data_dir.is_dir() {
        for entry in fs::read_dir(data_dir)? {
            let path = entry?.path();
            if path.extension().map_or(false, |e| e == "json") {
                if let Ok(content) = fs::read_to_string(&path) {
                    // Simple JSON parsing for Q&A pairs
                    for line in content.lines() {
                        if line.contains("\"input\"") && line.contains("\"output\"") {
                            // Extract simple patterns
                            if let (Some(input_start), Some(output_start)) = 
                                (line.find("\"input\":"), line.find("\"output\":")) {
                                let input_part = &line[input_start..];
                                let output_part = &line[output_start..];
                                
                                if let Some(input) = extract_json_string(input_part, "input") {
                                    if let Some(output) = extract_json_string(output_part, "output") {
                                        all_pairs.push((input, output));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    println!("   ✓ Loaded {} training examples", all_pairs.len());

    // =========================================================================
    // TRAINING
    // =========================================================================
    println!("\n🎓 Training model...");
    let train_start = Instant::now();
    
    let num_epochs = 3;
    for epoch in 1..=num_epochs {
        for (input, output) in &all_pairs {
            // Build vocabulary
            large_model.learn(input);
            large_model.learn(output);
            
            // Train latent decoder (main generation model)
            let thought = ThoughtState::from_input(input, 128);
            for _ in 0..3 {
                latent_decoder.learn(&thought, output);
            }
        }
        println!("   Epoch {}/{} complete", epoch, num_epochs);
    }
    
    println!("   ✓ Training completed in {:?}", train_start.elapsed());
    
    let stats = latent_decoder.stats();
    println!("\n📊 Model Statistics:");
    println!("   - Training examples: {}", stats.training_count);
    println!("   - Vocabulary size: {}", stats.vocabulary_size);
    println!("   - Active patterns: {}", stats.active_patterns);

    // =========================================================================
    // INTERACTIVE CHAT
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    INTERACTIVE CHAT                              ║");
    println!("║  Type your message and press Enter. Type 'quit' to exit.         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    
    loop {
        print!("You: ");
        stdout.flush()?;
        
        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        if input.to_lowercase() == "quit" || input.to_lowercase() == "exit" {
            println!("\nGoodbye! 👋");
            break;
        }
        
        // Generate response using latent decoder
        let thought = ThoughtState::from_input(input, 128);
        let (response, confidence) = latent_decoder.generate(&thought);
        
        // Check if response is meaningful
        let final_response = if response.trim().is_empty() || confidence < 0.1 {
            // Try to find a similar trained response
            find_best_match(input, &all_pairs)
        } else {
            response
        };
        
        println!("ALEN: {} (confidence: {:.1}%)", final_response, confidence * 100.0);
        println!();
    }

    Ok(())
}

/// Extract a JSON string value
fn extract_json_string(text: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":", key);
    if let Some(start) = text.find(&pattern) {
        let after_key = &text[start + pattern.len()..];
        let after_key = after_key.trim();
        
        if after_key.starts_with('"') {
            let content = &after_key[1..];
            if let Some(end) = content.find('"') {
                return Some(content[..end].to_string());
            }
        }
    }
    None
}

/// Find best matching response from training data
fn find_best_match(input: &str, pairs: &[(String, String)]) -> String {
    let input_lower = input.to_lowercase();
    let input_words: Vec<&str> = input_lower.split_whitespace().collect();
    
    let mut best_match = String::new();
    let mut best_score = 0;
    
    for (q, a) in pairs {
        let q_lower = q.to_lowercase();
        let q_words: Vec<&str> = q_lower.split_whitespace().collect();
        
        // Count matching words
        let mut score = 0;
        for word in &input_words {
            if q_words.contains(word) {
                score += 1;
            }
        }
        
        // Bonus for key question words
        if input_lower.contains("what") && q_lower.contains("what") { score += 2; }
        if input_lower.contains("how") && q_lower.contains("how") { score += 2; }
        if input_lower.contains("why") && q_lower.contains("why") { score += 2; }
        if input_lower.contains("when") && q_lower.contains("when") { score += 2; }
        if input_lower.contains("who") && q_lower.contains("who") { score += 2; }
        
        if score > best_score {
            best_score = score;
            best_match = a.clone();
        }
    }
    
    if best_score > 0 {
        best_match
    } else {
        "I'm still learning. Could you rephrase your question or teach me something new?".to_string()
    }
}
