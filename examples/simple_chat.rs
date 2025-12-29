use alen::core::{CognitiveArchitecture, Problem, ProblemDomain};
use alen::memory::{EpisodicMemory, SemanticMemory};
use alen::learning::VerifiedLearningSystem;
use std::io::{self, Write};

fn main() {
    println!("\n{}", "=".repeat(70));
    println!("🤖 ALEN - Adaptive Learning Expert Network");
    println!("{}", "=".repeat(70));
    println!("\nWelcome! I'm ALEN, your AI assistant.");
    println!("Type 'quit' to exit, 'help' for commands.\n");
    
    // Initialize ALEN system
    println!("🔧 Initializing ALEN...");
    
    let episodic_memory = EpisodicMemory::new();
    let semantic_memory = SemanticMemory::new("alen_semantic.db").expect("Failed to create semantic memory");
    let mut learning_system = VerifiedLearningSystem::new(episodic_memory, semantic_memory);
    
    println!("✓ ALEN initialized!\n");
    
    // Chat loop
    loop {
        print!("You: ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        match input.to_lowercase().as_str() {
            "quit" | "exit" | "bye" => {
                println!("\n👋 Goodbye! Thanks for chatting with ALEN.");
                break;
            }
            "help" => {
                print_help();
                continue;
            }
            _ => {}
        }
        
        // Process input
        println!("\nALEN: ");
        
        // Create a problem from the input
        let problem = Problem {
            id: uuid::Uuid::new_v4().to_string(),
            domain: ProblemDomain::General,
            description: input.to_string(),
            constraints: vec![],
            context: std::collections::HashMap::new(),
        };
        
        // Process with learning system
        match learning_system.process_and_learn(&problem) {
            Ok(result) => {
                if let Some(answer) = result.answer {
                    println!("{}", answer.content);
                    println!("\n📊 Confidence: {:.1}%", result.confidence * 100.0);
                } else {
                    println!("I'm not sure how to answer that yet. I'm still learning!");
                }
            }
            Err(e) => {
                println!("I encountered an error: {}", e);
            }
        }
        
        println!();
    }
}

fn print_help() {
    println!("\n{}", "=".repeat(70));
    println!("📚 ALEN Help");
    println!("{}", "=".repeat(70));
    println!("\nCommands:");
    println!("  help             - Show this help message");
    println!("  quit/exit/bye    - Exit the program");
    println!("\nJust type your question or statement, and I'll do my best to help!");
    println!("{}", "=".repeat(70));
}
