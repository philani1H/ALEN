//! ALEN - Advanced Learning Engine
//!
//! A Deliberative Reasoning AI System with Multimodal Understanding
//! and Verified Learning capabilities.
//!
//! # Usage
//!
//! ```bash
//! # Run with default settings (port 3000)
//! cargo run
//!
//! # Run with custom port
//! ALEN_PORT=8080 cargo run
//!
//! # Run with custom dimension
//! ALEN_DIMENSION=256 cargo run
//! ```
//!
//! # API Endpoints
//!
//! ## Training
//! - `POST /train` - Train on a single problem
//! - `POST /train/batch` - Train on multiple problems
//!
//! ## Inference
//! - `POST /infer` - Perform reasoning on an input
//!
//! ## Multimodal
//! - `POST /image` - Process image input
//! - `POST /audio` - Process audio input
//! - `POST /video` - Process video input
//! - `POST /fusion` - Fuse multiple modalities
//!
//! ## Generation
//! - `POST /generate/text` - Generate text from thought
//! - `POST /generate/image` - Generate image from thought
//!
//! ## Memory Management
//! - `POST /facts` - Add a semantic fact
//! - `POST /facts/search` - Search facts by similarity
//! - `GET /memory/episodic/stats` - Get episodic memory statistics
//! - `GET /memory/episodic/top/:limit` - Get top verified episodes
//! - `DELETE /memory/episodic/clear` - Clear episodic memory
//! - `DELETE /memory/semantic/clear` - Clear semantic memory
//!
//! ## System
//! - `GET /health` - Health check
//! - `GET /stats` - System statistics
//! - `GET /operators` - Operator performance
//!
//! ## Control
//! - `POST /bias` - Set bias parameters
//! - `POST /bias/reset` - Reset bias to defaults
//! - `POST /learning/reset` - Reset learning rate

use alen::api::{AppState, EngineConfig, ReasoningEngine, create_router, ConversationManager};
use alen::learning::LearningConfig;
use alen::core::EnergyWeights;
use alen::memory::EmbeddingConfig;
use alen::storage::StorageConfig;

use std::env;
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;

/// Application configuration from environment variables
struct AppConfig {
    port: u16,
    host: String,
    dimension: usize,
    learning_rate: f64,
    max_iterations: usize,
    confidence_threshold: f64,
}

impl AppConfig {
    fn from_env() -> Self {
        Self {
            port: env::var("ALEN_PORT")
                .or_else(|_| env::var("DELIBERATIVE_AI_PORT"))
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3000),
            host: env::var("ALEN_HOST")
                .or_else(|_| env::var("DELIBERATIVE_AI_HOST"))
                .unwrap_or_else(|_| "0.0.0.0".to_string()),
            dimension: env::var("ALEN_DIMENSION")
                .or_else(|_| env::var("DELIBERATIVE_AI_DIMENSION"))
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(128),
            learning_rate: env::var("DELIBERATIVE_AI_LEARNING_RATE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.01),
            max_iterations: env::var("DELIBERATIVE_AI_MAX_ITERATIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
            confidence_threshold: env::var("DELIBERATIVE_AI_CONFIDENCE_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.7),
        }
    }
}

fn print_banner() {
    println!(r#"
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       █████╗ ██╗     ███████╗███╗   ██╗                              ║
║      ██╔══██╗██║     ██╔════╝████╗  ██║                              ║
║      ███████║██║     █████╗  ██╔██╗ ██║                              ║
║      ██╔══██║██║     ██╔══╝  ██║╚██╗██║                              ║
║      ██║  ██║███████╗███████╗██║ ╚████║                              ║
║      ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝                              ║
║                                                                      ║
║       Advanced Learning Engine with Neural Understanding             ║
║                        Version 0.2.0                                 ║
║                                                                      ║
║  "Thoughts as vectors, reasoning as operators, learning as energy"   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"#);
}

fn print_config(config: &AppConfig) {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│                         Configuration                               │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  Host:                  {:<42} │", format!("{}:{}", config.host, config.port));
    println!("│  Vector Dimension:      {:<42} │", config.dimension);
    println!("│  Learning Rate:         {:<42} │", config.learning_rate);
    println!("│  Max Iterations:        {:<42} │", config.max_iterations);
    println!("│  Confidence Threshold:  {:<42} │", config.confidence_threshold);
    println!("└─────────────────────────────────────────────────────────────────────┘");
}

fn print_endpoints(port: u16) {
    println!("\n┌─────────────────────────────────────────────────────────────────────┐");
    println!("│                         API Endpoints                               │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│                                                                     │");
    println!("│  Training:                                                          │");
    println!("│    POST /train              Train on single problem                 │");
    println!("│    POST /train/batch        Train on multiple problems              │");
    println!("│    POST /train/comprehensive  Full training with epochs             │");
    println!("│                                                                     │");
    println!("│  Learning:                                                          │");
    println!("│    POST /learn              Learn knowledge facts                   │");
    println!("│    POST /query              Query learned knowledge                 │");
    println!("│                                                                     │");
    println!("│  Inference:                                                         │");
    println!("│    POST /infer              Perform reasoning                       │");
    println!("│                                                                     │");
    println!("│  Generation:                                                        │");
    println!("│    POST /generate/text      Generate text from thought              │");
    println!("│    POST /generate/image     Generate image from thought             │");
    println!("│                                                                     │");
    println!("│  Multimodal:                                                        │");
    println!("│    POST /multimodal/image   Process image input                     │");
    println!("│    POST /multimodal/audio   Process audio input                     │");
    println!("│    POST /multimodal/video   Process video input                     │");
    println!("│    POST /multimodal/fuse    Fuse multiple modalities                │");
    println!("│                                                                     │");
    println!("│  Memory:                                                            │");
    println!("│    POST /facts              Add semantic fact                       │");
    println!("│    POST /facts/search       Search facts                            │");
    println!("│    GET  /memory/episodic/stats     Memory statistics               │");
    println!("│    GET  /memory/episodic/top/:n    Top episodes                    │");
    println!("│    DEL  /memory/episodic/clear     Clear episodic                  │");
    println!("│    DEL  /memory/semantic/clear     Clear semantic                  │");
    println!("│                                                                     │");
    println!("│  System:                                                            │");
    println!("│    GET  /health             Health check                            │");
    println!("│    GET  /stats              System statistics                       │");
    println!("│    GET  /operators          Operator stats                          │");
    println!("│    GET  /capabilities       System capabilities                     │");
    println!("│                                                                     │");
    println!("│  Control:                                                           │");
    println!("│    POST /bias               Set bias parameters                     │");
    println!("│    POST /bias/reset         Reset bias                              │");
    println!("│    POST /learning/reset     Reset learning rate                     │");
    println!("│                                                                     │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!("\n  Server running at: http://localhost:{}", port);
    println!("  Try: curl http://localhost:{}/health\n", port);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .compact()
        .init();

    // Print startup banner
    print_banner();

    // Load configuration
    let app_config = AppConfig::from_env();
    print_config(&app_config);

    // Create engine configuration
    let engine_config = EngineConfig {
        dimension: app_config.dimension,
        learning: LearningConfig {
            learning_rate: app_config.learning_rate,
            min_learning_rate: 0.001,
            decay_factor: 0.995,
            num_candidates: 5,
            max_iterations: app_config.max_iterations,
            confidence_threshold: app_config.confidence_threshold,
            energy_threshold: 0.5,
        },
        energy_weights: EnergyWeights::default(),
        embedding: EmbeddingConfig {
            dimension: app_config.dimension,
            normalize: true,
            vocab_size: 10000,
        },
        evaluator_confidence_threshold: 0.6,  // Stricter for testing
        evaluator_energy_threshold: 0.5,
        backward_similarity_threshold: 0.7,  // Enabled for math AST verification
        backward_path_threshold: 0.3,  // Enabled for structure consistency
    };

    // Initialize storage configuration
    info!("Initializing storage...");
    let storage_config = match StorageConfig::production() {
        Ok(s) => {
            info!("✓ Storage configured at: {:?}", s.base_dir);
            s
        }
        Err(e) => {
            error!("Failed to configure storage: {}", e);
            return Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>);
        }
    };

    // Create the reasoning engine with persistent storage
    info!("Initializing reasoning engine...");
    let engine = match ReasoningEngine::with_storage(engine_config.clone(), &storage_config) {
        Ok(e) => {
            info!("✓ Reasoning engine initialized successfully");
            e
        }
        Err(e) => {
            error!("Failed to create reasoning engine: {}", e);
            return Err(e);
        }
    };

    // Create conversation manager
    info!("Initializing conversation manager...");
    let conversation_manager = ConversationManager::new();
    info!("✓ Conversation manager initialized");

    // Create shared state
    let state = Arc::new(AppState {
        engine: tokio::sync::Mutex::new(engine),
        config: engine_config,
        conversation_manager,
        storage: storage_config,
    });

    // Create router with middleware
    let app = create_router(state)
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    // Create address
    let addr: SocketAddr = format!("{}:{}", app_config.host, app_config.port)
        .parse()
        .expect("Invalid address");

    // Print endpoints
    print_endpoints(app_config.port);

    // Start server
    info!("Starting server on {}", addr);
    let listener = TcpListener::bind(addr).await?;
    
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│                    Server Started Successfully                      │");
    println!("│                       Press Ctrl+C to stop                          │");
    println!("└─────────────────────────────────────────────────────────────────────┘\n");

    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_config_defaults() {
        // Clear any environment variables that might affect the test
        env::remove_var("DELIBERATIVE_AI_PORT");
        env::remove_var("DELIBERATIVE_AI_HOST");
        env::remove_var("DELIBERATIVE_AI_DIMENSION");
        
        let config = AppConfig::from_env();
        
        assert_eq!(config.port, 3000);
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.dimension, 128);
    }
}
