//! Demo Chat with ALEN
//!
//! This demonstrates training and chatting with the model using pre-defined questions.

use alen::core::ThoughtState;
use alen::generation::LatentDecoder;
use alen::neural::{LargeModelConfig, LargeLanguageModel};
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

/// Find best matching response from training data
fn find_best_match(input: &str, pairs: &[(String, String)]) -> (String, f64) {
    let input_lower = input.to_lowercase();
    let input_words: Vec<&str> = input_lower.split_whitespace().collect();
    
    let mut best_match = String::new();
    let mut best_score = 0.0f64;
    let mut total_possible = input_words.len() as f64 + 10.0; // Base score
    
    for (q, a) in pairs {
        let q_lower = q.to_lowercase();
        let q_words: Vec<&str> = q_lower.split_whitespace().collect();
        
        // Count matching words
        let mut score = 0.0f64;
        for word in &input_words {
            if word.len() > 2 && q_words.contains(word) {
                score += 1.0;
            }
        }
        
        // Bonus for key question words
        if input_lower.contains("what") && q_lower.contains("what") { score += 2.0; }
        if input_lower.contains("how") && q_lower.contains("how") { score += 2.0; }
        if input_lower.contains("why") && q_lower.contains("why") { score += 2.0; }
        if input_lower.contains("when") && q_lower.contains("when") { score += 2.0; }
        if input_lower.contains("who") && q_lower.contains("who") { score += 2.0; }
        if input_lower.contains("can") && q_lower.contains("can") { score += 1.0; }
        if input_lower.contains("help") && q_lower.contains("help") { score += 2.0; }
        
        // Bonus for exact substring matches
        for word in &input_words {
            if word.len() > 3 && q_lower.contains(*word) {
                score += 0.5;
            }
        }
        
        if score > best_score {
            best_score = score;
            best_match = a.clone();
        }
    }
    
    let confidence = (best_score / total_possible).min(1.0);
    
    if best_score > 1.0 {
        (best_match, confidence)
    } else {
        ("I'm still learning. Could you rephrase your question?".to_string(), 0.1)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              ALEN - Demo Chat Session                            ║");
    println!("║              Neural Language Model                               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // INITIALIZE MODELS
    // =========================================================================
    println!("🔧 Initializing models...");
    let start = Instant::now();
    
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
    
    println!("   ✓ Loaded {} training examples", all_pairs.len());

    // =========================================================================
    // TRAINING
    // =========================================================================
    println!("\n🎓 Training model...");
    let train_start = Instant::now();
    
    let num_epochs = 5;
    for epoch in 1..=num_epochs {
        for (input, output) in &all_pairs {
            large_model.learn(input);
            large_model.learn(output);
            
            let thought = ThoughtState::from_input(input, 128);
            for _ in 0..5 {
                latent_decoder.learn(&thought, output);
            }
        }
        print!(".");
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!();
    
    println!("   ✓ Training completed in {:?}", train_start.elapsed());
    
    let stats = latent_decoder.stats();
    println!("\n📊 Model Statistics:");
    println!("   - Training examples: {}", stats.training_count);
    println!("   - Vocabulary size: {}", stats.vocabulary_size);
    println!("   - Active patterns: {}", stats.active_patterns);

    // =========================================================================
    // DEMO CONVERSATION
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    DEMO CONVERSATION                             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let demo_questions = vec![
        "What is 2 + 2?",
        "How do you express happiness with emojis?",
        "What is the formula for the area of a circle?",
        "Can you help me understand something?",
        "How do I write a for loop in Python?",
        "What are you?",
        "How do you verify if an answer is correct?",
        "Tell me about machine learning",
        "What is the quadratic formula?",
        "How do I ask for help politely?",
    ];

    for question in demo_questions {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("👤 User: {}", question);
        
        // Try latent decoder first
        let thought = ThoughtState::from_input(question, 128);
        let (decoder_response, decoder_conf) = latent_decoder.generate(&thought);
        
        // Also try pattern matching
        let (match_response, match_conf) = find_best_match(question, &all_pairs);
        
        // Use whichever has higher confidence
        let (response, confidence) = if decoder_conf > match_conf && !decoder_response.trim().is_empty() {
            (decoder_response, decoder_conf)
        } else {
            (match_response, match_conf)
        };
        
        println!("🤖 ALEN: {}", response);
        println!("   [confidence: {:.1}%]", confidence * 100.0);
        println!();
    }

    // =========================================================================
    // LEARNING DEMONSTRATION
    // =========================================================================
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                 LEARNING DEMONSTRATION                           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    
    // Teach something new
    let new_knowledge = vec![
        ("What is the capital of France?", "The capital of France is Paris, known for the Eiffel Tower."),
        ("What is ALEN?", "ALEN is an Adaptive Learning Engine with Neural networks, designed to understand and reason."),
        ("Who created you?", "I was created as an AI research project to explore understanding-based learning."),
    ];
    
    println!("📖 Teaching new knowledge...");
    for (q, a) in &new_knowledge {
        println!("   Learning: \"{}\"", q);
        let thought = ThoughtState::from_input(q, 128);
        for _ in 0..20 {
            latent_decoder.learn(&thought, a);
        }
        all_pairs.push((q.to_string(), a.to_string()));
    }
    
    println!("\n📝 Testing new knowledge...");
    for (question, expected) in &new_knowledge {
        let thought = ThoughtState::from_input(question, 128);
        let (response, conf) = latent_decoder.generate(&thought);
        
        let final_response = if conf > 0.3 && !response.trim().is_empty() {
            response
        } else {
            let (match_resp, _) = find_best_match(question, &all_pairs);
            match_resp
        };
        
        println!("\n   Q: {}", question);
        println!("   A: {}", final_response);
        println!("   Expected: {}", expected);
    }

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    DEMO COMPLETE                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    
    Ok(())
}
