//! Full Training Script
//!
//! Trains ALEN on all available datasets:
//! - Basic training data (100 questions)
//! - Advanced questions (40 questions)
//! - Comprehensive test data (80 questions)
//! Total: 220 questions across multiple domains

use alen::neural::{NeuralReasoningEngine, ALENConfig};
use alen::memory::{SemanticStore, SemanticEntry};
use alen::control::{EmotionSystem, EmotionalStimulus, StimulusType, MoodEngine, EmotionalResponse, Emotion, Neurotransmitters};
use alen::core::Problem;
use serde::{Deserialize, Serialize};
use std::fs;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct QuestionAnswer {
    q: String,
    a: String,
    #[serde(default)]
    difficulty: String,
    #[serde(default)]
    requires: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Category {
    category: String,
    questions: Vec<QuestionAnswer>,
}

fn load_dataset(path: &str) -> Result<Vec<Category>, Box<dyn std::error::Error>> {
    let data = fs::read_to_string(path)?;
    let categories: Vec<Category> = serde_json::from_str(&data)?;
    Ok(categories)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              ALEN FULL TRAINING SYSTEM                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load all datasets
    println!("Loading datasets...");
    
    let mut all_data = Vec::new();
    let datasets = vec![
        ("data/training_data.json", "Basic Training"),
        ("data/advanced_questions.json", "Advanced Questions"),
        ("data/comprehensive_test_data.json", "Comprehensive Tests"),
    ];

    let mut total_questions = 0;
    for (path, name) in &datasets {
        match load_dataset(path) {
            Ok(data) => {
                let count: usize = data.iter().map(|c| c.questions.len()).sum();
                println!("  ✓ Loaded {}: {} questions", name, count);
                total_questions += count;
                all_data.extend(data);
            }
            Err(e) => {
                println!("  ⚠ Could not load {}: {}", name, e);
            }
        }
    }
    
    println!("\nTotal questions loaded: {}\n", total_questions);

    // Configuration
    let config = ALENConfig {
        thought_dim: 128,
        vocab_size: 10000,
        num_operators: 8,
        operator_hidden_dim: 256,
        dropout: 0.1,
        layer_norm_eps: 1e-5,
        use_transformer: false,
        transformer_layers: 4,
        transformer_heads: 4,
    };

    println!("Configuration:");
    println!("  Thought dimension: {}", config.thought_dim);
    println!("  Number of operators: {}", config.num_operators);
    println!("  Operator hidden dim: {}", config.operator_hidden_dim);
    println!("  Use transformer: {}\n", config.use_transformer);

    // Initialize systems
    println!("Initializing systems...");
    let mut neural_engine = NeuralReasoningEngine::new(config.clone(), 0.001);
    let mut semantic_store = SemanticStore::new("storage/semantic", 2000, 0.7)
        .expect("Failed to create semantic store");
    let mut emotion_system = EmotionSystem::new();
    let mut mood_engine = MoodEngine::new();
    
    println!("✓ Neural engine initialized ({} parameters)", neural_engine.network.num_parameters());
    println!("✓ Semantic store initialized ({} existing entries)", semantic_store.len());
    println!("✓ Emotion system initialized");
    println!("✓ Mood engine initialized\n");

    println!("═══════════════════════════════════════════════════════════════");
    println!("                    TRAINING PHASE");
    println!("═══════════════════════════════════════════════════════════════\n");

    let epochs = 3;
    let mut total_verified = 0;
    let mut total_trained = 0;
    let mut category_stats: HashMap<String, (usize, usize)> = HashMap::new();
    let mut difficulty_stats: HashMap<String, (usize, usize)> = HashMap::new();

    for epoch in 1..=epochs {
        println!("Epoch {}/{}", epoch, epochs);
        println!("─────────────────────────────────────────────────────────────");

        let mut epoch_verified = 0;
        let mut epoch_loss = 0.0;
        let mut epoch_count = 0;

        for category in &all_data {
            for (i, qa) in category.questions.iter().enumerate() {
                let problem = Problem::training(&qa.q, &qa.a, config.thought_dim);
                let result = neural_engine.train_verified(&problem);

                total_trained += 1;
                epoch_count += 1;
                
                // Update category stats
                let cat_stats = category_stats.entry(category.category.clone()).or_insert((0, 0));
                cat_stats.1 += 1;
                
                // Update difficulty stats
                if !qa.difficulty.is_empty() {
                    let diff_stats = difficulty_stats.entry(qa.difficulty.clone()).or_insert((0, 0));
                    diff_stats.1 += 1;
                }
                
                if result.verified {
                    epoch_verified += 1;
                    total_verified += 1;
                    cat_stats.0 += 1;
                    
                    if !qa.difficulty.is_empty() {
                        let diff_stats = difficulty_stats.entry(qa.difficulty.clone()).or_insert((0, 0));
                        diff_stats.0 += 1;
                    }
                    
                    // Store in semantic memory
                    let inference_result = neural_engine.infer(&qa.q);
                    let entry = SemanticEntry::new(
                        inference_result.thought_vector.iter().map(|&x| x as f32).collect(),
                        qa.q.clone(),
                        1.0 - inference_result.verification_error,
                        inference_result.selected_operator,
                        inference_result.operator_name.clone(),
                        inference_result.verification_error,
                    );
                    
                    let _ = semantic_store.insert(entry);
                    
                    // Emotional response to success
                    let stimulus = EmotionalStimulus {
                        stimulus_type: StimulusType::Success,
                        intensity: (1.0 - result.loss) as f64,
                        valence: 0.8,
                        context: format!("Verified: {}", qa.q),
                    };
                    let _regulated = emotion_system.process(stimulus);
                    
                    // Create emotional response for mood engine
                    let emotion_response = EmotionalResponse {
                        emotion: Emotion::Joy,
                        valence: 0.8,
                        arousal: 0.6,
                        intensity: (1.0 - result.loss) as f64,
                        neurotransmitters: Neurotransmitters::default(),
                    };
                    mood_engine.update_from_emotion(&emotion_response);
                } else {
                    // Emotional response to failure
                    let stimulus = EmotionalStimulus {
                        stimulus_type: StimulusType::Failure,
                        intensity: 0.6,
                        valence: -0.5,
                        context: format!("Failed: {}", qa.q),
                    };
                    let _regulated = emotion_system.process(stimulus);
                    
                    // Create emotional response for mood engine
                    let emotion_response = EmotionalResponse {
                        emotion: Emotion::Sadness,
                        valence: -0.5,
                        arousal: 0.5,
                        intensity: 0.6,
                        neurotransmitters: Neurotransmitters::default(),
                    };
                    mood_engine.update_from_emotion(&emotion_response);
                }

                if result.success {
                    epoch_loss += result.loss;
                }

                // Progress update every 50 items
                if epoch_count % 50 == 0 {
                    let current_mood = mood_engine.current_mood();
                    println!("  [{}/{}] Verified: {}/{} ({:.1}%) | Mood: {}",
                        epoch_count,
                        total_questions,
                        epoch_verified,
                        epoch_count,
                        (epoch_verified as f32 / epoch_count as f32) * 100.0,
                        current_mood.as_str()
                    );
                }
            }
        }

        let avg_loss = if epoch_count > 0 {
            epoch_loss / epoch_count as f32
        } else {
            0.0
        };
        let verification_rate = (epoch_verified as f32 / epoch_count as f32) * 100.0;

        println!("\nEpoch {} Summary:", epoch);
        println!("  Verified: {}/{} ({:.1}%)", epoch_verified, epoch_count, verification_rate);
        println!("  Average loss: {:.6}", avg_loss);
        println!("  Current mood: {}", mood_engine.current_mood().as_str());
        println!();
        
        // Decay mood between epochs
        mood_engine.decay();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("                  TRAINING COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Total samples: {}", total_trained);
    println!("Total verified: {} ({:.1}%)\n", 
        total_verified, 
        (total_verified as f32 / total_trained as f32) * 100.0
    );

    println!("═══════════════════════════════════════════════════════════════");
    println!("                CATEGORY PERFORMANCE");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut sorted_categories: Vec<_> = category_stats.iter().collect();
    sorted_categories.sort_by(|a, b| b.1.0.cmp(&a.1.0));

    for (category, (verified, total)) in sorted_categories {
        let rate = (*verified as f32 / *total as f32) * 100.0;
        println!("  {:35} {:3}/{:3} ({:5.1}%)", 
            category, verified, total, rate);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                DIFFICULTY BREAKDOWN");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut sorted_difficulty: Vec<_> = difficulty_stats.iter().collect();
    sorted_difficulty.sort_by_key(|(k, _)| match k.as_str() {
        "easy" => 0,
        "medium" => 1,
        "hard" => 2,
        _ => 3,
    });

    for (difficulty, (verified, total)) in sorted_difficulty {
        let rate = (*verified as f32 / *total as f32) * 100.0;
        println!("  {:10} {:3}/{:3} ({:5.1}%)", 
            difficulty, verified, total, rate);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                OPERATOR STATISTICS");
    println!("═══════════════════════════════════════════════════════════════\n");

    let stats = neural_engine.operator_statistics();
    let mut sorted_stats = stats.clone();
    sorted_stats.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));

    for stat in &sorted_stats {
        if stat.usage_count > 0 {
            println!("  {} (ID: {})", stat.name, stat.id);
            println!("    Usage: {} times ({:.1}%)", 
                stat.usage_count,
                (stat.usage_count as f32 / total_trained as f32) * 100.0
            );
            println!("    Success rate: {:.1}%", stat.success_rate * 100.0);
            println!();
        }
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("                EMOTIONAL & MOOD ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════\n");

    let current_emotion = emotion_system.current_emotion();
    let mood_stats = mood_engine.statistics();
    
    println!("Current Emotional State:");
    println!("  Emotion: {}", current_emotion.as_str());
    println!("  Mood: {}", mood_stats.current_mood.as_str());
    println!("\nMood Levels:");
    println!("  Reward: {:.2}", mood_stats.reward_level);
    println!("  Stress: {:.2}", mood_stats.stress_level);
    println!("  Trust: {:.2}", mood_stats.trust_level);
    println!("  Curiosity: {:.2}", mood_stats.curiosity_level);
    println!("  Energy: {:.2}", mood_stats.energy_level);
    println!("\nPerception:");
    println!("  Bias: {:.2}", mood_stats.perception_bias);
    println!("  Reaction threshold: {:.2}", mood_stats.reaction_threshold);
    println!("  Stable: {}", mood_stats.is_stable);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                SEMANTIC MEMORY");
    println!("═══════════════════════════════════════════════════════════════\n");

    let semantic_stats = semantic_store.statistics();
    println!("Total entries: {}", semantic_stats.total_entries);
    println!("Average confidence: {:.4}", semantic_stats.avg_confidence);
    println!("Storage size: {} bytes", semantic_stats.storage_size_bytes);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                TRAINING COMPLETE                             ║");
    println!("║                                                              ║");
    println!("║  ALEN has been trained on {} questions                    ║", total_questions);
    println!("║  Verification rate: {:.1}%                                    ║", 
        (total_verified as f32 / total_trained as f32) * 100.0);
    println!("║  Semantic memory: {} entries                              ║", semantic_stats.total_entries);
    println!("║  Current mood: {:20}                          ║", mood_stats.current_mood.as_str());
    println!("║                                                              ║");
    println!("║  Model is ready for testing and deployment.                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}
