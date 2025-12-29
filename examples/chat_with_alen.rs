use alen::neural::{
    MathProblemSolver, CodeGenerationSystem, AdvancedALENConfig,
    AudienceLevel, ProgrammingLanguage, TransformerConfig,
    NoiseSchedule, TemperatureSchedule,
};
use std::io::{self, Write};

fn main() {
    println!("\n{}", "=".repeat(70));
    println!("🤖 ALEN - Advanced Learning Expert Network");
    println!("{}", "=".repeat(70));
    println!("\nWelcome! I'm ALEN, your AI assistant with advanced neural capabilities.");
    println!("I can help you with:");
    println!("  • Mathematical problems (type 'math: your problem')");
    println!("  • Code generation (type 'code: your specification')");
    println!("  • General questions (just type your question)");
    println!("\nType 'help' for more options, 'quit' to exit.\n");
    
    // Initialize systems
    println!("🔧 Initializing neural systems...");
    let config = create_config();
    let mut math_solver = MathProblemSolver::new(config.clone());
    let mut code_generator = CodeGenerationSystem::new(config);
    println!("✓ Systems ready!\n");
    
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
                println!("\n👋 Goodbye! Thanks for using ALEN.");
                break;
            }
            "help" => {
                print_help();
                continue;
            }
            "stats" => {
                print_stats();
                continue;
            }
            _ => {}
        }
        
        // Process input
        println!("\nALEN: ");
        
        if input.to_lowercase().starts_with("math:") {
            let problem = input[5..].trim();
            handle_math_problem(&mut math_solver, problem);
        } else if input.to_lowercase().starts_with("code:") {
            let spec = input[5..].trim();
            handle_code_generation(&mut code_generator, spec);
        } else {
            handle_general_question(input);
        }
        
        println!();
    }
}

fn create_config() -> AdvancedALENConfig {
    AdvancedALENConfig {
        problem_input_dim: 128,
        audience_profile_dim: 32,
        memory_retrieval_dim: 64,
        solution_embedding_dim: 128,
        explanation_embedding_dim: 128,
        solve_hidden_dims: vec![256, 128],
        verify_hidden_dims: vec![128, 64],
        explain_hidden_dims: vec![256, 128],
        transformer_config: TransformerConfig {
            d_model: 128,
            n_heads: 4,
            n_layers: 2,
            d_ff: 512,
            dropout: 0.1,
            max_seq_len: 256,
        },
        dropout_rate: 0.1,
        loss_weights: (0.5, 0.3, 0.2),
        max_memories: 100,
        action_space_size: 1000,
        temperature: 1.0,
        gamma: 0.99,
        policy_learning_rate: 0.001,
        max_trajectory_length: 50,
        noise_sigma: 0.1,
        noise_schedule: NoiseSchedule::CosineAnneal { total_steps: 1000 },
        temperature_schedule: TemperatureSchedule::ExponentialCooling { decay_rate: 0.001 },
        diversity_weight: 0.1,
        novelty_k: 10,
        novelty_threshold: 0.5,
        inner_lr: 0.01,
        outer_lr: 0.001,
        inner_steps: 3,
        meta_hidden_dim: 128,
        base_lr: 0.001,
    }
}

fn handle_math_problem(solver: &mut MathProblemSolver, problem: &str) {
    println!("🔢 Solving mathematical problem...\n");
    
    // Determine audience level based on problem complexity
    let audience_level = if problem.contains("derivative") || problem.contains("integral") {
        AudienceLevel::Undergraduate
    } else if problem.contains("prove") || problem.contains("theorem") {
        AudienceLevel::Graduate
    } else {
        AudienceLevel::HighSchool
    };
    
    let solution = solver.solve(problem, audience_level);
    
    println!("📝 Solution: {}", solution.solution);
    println!("\n💡 Explanation:");
    println!("{}", solution.explanation);
    println!("\n📊 Confidence: {:.1}%", solution.confidence * 100.0);
}

fn handle_code_generation(generator: &mut CodeGenerationSystem, spec: &str) {
    println!("💻 Generating code...\n");
    
    // Determine language from specification
    let language = if spec.to_lowercase().contains("rust") {
        ProgrammingLanguage::Rust
    } else if spec.to_lowercase().contains("python") {
        ProgrammingLanguage::Python
    } else if spec.to_lowercase().contains("javascript") || spec.to_lowercase().contains("js") {
        ProgrammingLanguage::JavaScript
    } else if spec.to_lowercase().contains("java") {
        ProgrammingLanguage::Java
    } else {
        ProgrammingLanguage::Python // Default
    };
    
    let code_solution = generator.generate(spec, language);
    
    println!("📄 Generated Code ({:?}):", code_solution.language);
    println!("{}", "-".repeat(60));
    println!("{}", code_solution.code);
    println!("{}", "-".repeat(60));
    println!("\n💡 Explanation:");
    println!("{}", code_solution.explanation);
    println!("\n📊 Confidence: {:.1}%", code_solution.confidence * 100.0);
}

fn handle_general_question(question: &str) {
    println!("🤔 Processing your question...\n");
    
    // For now, provide a helpful response
    println!("I understand you're asking: \"{}\"", question);
    println!("\nI'm currently optimized for:");
    println!("  • Mathematical problem solving (prefix with 'math:')");
    println!("  • Code generation (prefix with 'code:')");
    println!("\nFor best results, please specify the type of help you need!");
}

fn print_help() {
    println!("\n{}", "=".repeat(70));
    println!("📚 ALEN Help");
    println!("{}", "=".repeat(70));
    println!("\nCommands:");
    println!("  math: <problem>  - Solve a mathematical problem");
    println!("                     Example: math: solve x^2 + 2x + 1 = 0");
    println!("\n  code: <spec>     - Generate code from specification");
    println!("                     Example: code: write a Python function for fibonacci");
    println!("\n  help             - Show this help message");
    println!("  stats            - Show system statistics");
    println!("  quit/exit/bye    - Exit the program");
    println!("\nAudience Levels (auto-detected):");
    println!("  • Elementary     - Simple explanations");
    println!("  • High School    - Moderate detail");
    println!("  • Undergraduate  - Technical detail");
    println!("  • Graduate       - Advanced concepts");
    println!("  • Expert         - Full technical depth");
    println!("\nSupported Languages:");
    println!("  • Python, Rust, JavaScript, Java");
    println!("{}", "=".repeat(70));
}

fn print_stats() {
    println!("\n{}", "=".repeat(70));
    println!("📊 System Statistics");
    println!("{}", "=".repeat(70));
    println!("\nNeural Architecture:");
    println!("  • Multi-branch (solve, verify, explain)");
    println!("  • Memory-augmented learning");
    println!("  • Policy gradient training");
    println!("  • Creative exploration");
    println!("  • Meta-learning optimization");
    println!("\nCapabilities:");
    println!("  • Universal problem solving");
    println!("  • Audience adaptation");
    println!("  • Confidence estimation");
    println!("  • Explanation generation");
    println!("{}", "=".repeat(70));
}
