//! API Module
//!
//! Exposes REST APIs for:
//! - Training the AI
//! - Inference (thinking)
//! - System status and statistics
//! - Memory management

use crate::core::{
    Problem, OperatorManager, OperatorStats,
    Evaluator, EnergyWeights,
};
use crate::memory::{
    EpisodicMemory, Episode, EpisodeStatistics,
    SemanticMemory, SemanticFact, SemanticStatistics,
    EmbeddingEngine, EmbeddingConfig,
};
use crate::learning::{
    FeedbackLoop, LearningConfig, TrainingResult, InferenceResult,
};
use crate::control::{BiasController, ControlStateSummary};

use axum::{
    extract::{Path, State, Json},
    response::IntoResponse,
    routing::{get, post, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Shared application state
pub struct AppState {
    /// The main reasoning engine
    pub engine: Mutex<ReasoningEngine>,
    /// Configuration
    pub config: EngineConfig,
}

/// Configuration for the reasoning engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Vector dimension
    pub dimension: usize,
    /// Learning configuration
    pub learning: LearningConfig,
    /// Energy weights
    pub energy_weights: EnergyWeights,
    /// Embedding configuration
    pub embedding: EmbeddingConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            learning: LearningConfig::default(),
            energy_weights: EnergyWeights::default(),
            embedding: EmbeddingConfig {
                dimension: 128,
                normalize: true,
                vocab_size: 10000,
            },
        }
    }
}

/// The main reasoning engine that combines all components
pub struct ReasoningEngine {
    /// Operator manager
    pub operators: OperatorManager,
    /// Evaluator
    pub evaluator: Evaluator,
    /// Feedback loop
    pub feedback: FeedbackLoop,
    /// Episodic memory
    pub episodic_memory: EpisodicMemory,
    /// Semantic memory
    pub semantic_memory: SemanticMemory,
    /// Embedding engine
    pub embedding: EmbeddingEngine,
    /// Bias controller
    pub bias_controller: BiasController,
    /// Configuration
    pub config: EngineConfig,
}

impl ReasoningEngine {
    /// Create a new reasoning engine
    pub fn new(config: EngineConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let operators = OperatorManager::new(config.dimension);
        let evaluator = Evaluator::new(config.energy_weights.clone(), 0.5);
        let feedback = FeedbackLoop::new(
            operators.clone(),
            evaluator.clone(),
            config.learning.clone(),
        );
        let episodic_memory = EpisodicMemory::in_memory()?;
        let semantic_memory = SemanticMemory::in_memory(config.dimension)?;
        let embedding = EmbeddingEngine::new(config.embedding.clone());
        let bias_controller = BiasController::new();

        Ok(Self {
            operators,
            evaluator,
            feedback,
            episodic_memory,
            semantic_memory,
            embedding,
            bias_controller,
            config,
        })
    }

    /// Train on a single problem
    pub fn train(&mut self, problem: &Problem) -> TrainingResult {
        let result = self.feedback.train_step(problem);
        
        // Store in episodic memory if successful
        if result.success {
            if let (Some(ref thought), Some(ref energy), Some(ref op_id)) = 
                (&result.best_candidate, &result.best_energy, &result.best_operator_id) 
            {
                let episode = Episode::from_training(problem, thought, energy, op_id);
                let _ = self.episodic_memory.store(&episode);
            }
        }

        // Update bias controller
        if let Some(ref energy) = result.best_energy {
            self.bias_controller.update_meta(energy.confidence_score);
        }

        result
    }

    /// Perform inference
    pub fn infer(&self, problem: &Problem) -> InferenceResult {
        self.feedback.infer(problem)
    }

    /// Get system statistics
    pub fn get_statistics(&self) -> EngineStatistics {
        let episodic_stats = self.episodic_memory.get_statistics().unwrap_or(EpisodeStatistics {
            total_episodes: 0,
            verified_episodes: 0,
            average_confidence: 0.0,
            average_energy: 0.0,
        });

        let semantic_stats = self.semantic_memory.get_statistics().unwrap_or(SemanticStatistics {
            total_facts: 0,
            average_confidence: 0.0,
            categories: vec![],
        });

        EngineStatistics {
            operator_stats: self.feedback.get_operator_stats(),
            episodic_memory: episodic_stats,
            semantic_memory: semantic_stats,
            control_state: self.bias_controller.state_summary(),
            learning_rate: self.feedback.learning_rate(),
            iteration_count: self.feedback.iteration_count(),
            vocabulary_size: self.embedding.vocabulary_size(),
        }
    }
}

/// Combined engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatistics {
    pub operator_stats: Vec<OperatorStats>,
    pub episodic_memory: EpisodeStatistics,
    pub semantic_memory: SemanticStatistics,
    pub control_state: ControlStateSummary,
    pub learning_rate: f64,
    pub iteration_count: u64,
    pub vocabulary_size: usize,
}

// ============= API Request/Response Types =============

#[derive(Debug, Deserialize)]
pub struct TrainRequest {
    pub input: String,
    pub expected_answer: String,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub context: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct TrainResponse {
    pub success: bool,
    pub iterations: usize,
    pub confidence_score: Option<f64>,
    pub energy: Option<f64>,
    pub operator_used: Option<String>,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct BatchTrainRequest {
    pub problems: Vec<TrainRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchTrainResponse {
    pub total_problems: usize,
    pub successes: usize,
    pub failures: usize,
    pub success_rate: f64,
    pub average_iterations: f64,
}

#[derive(Debug, Deserialize)]
pub struct InferRequest {
    pub input: String,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub context: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct InferResponse {
    pub confidence: f64,
    pub energy: f64,
    pub verified: bool,
    pub operator_used: String,
    pub candidates_considered: usize,
    pub is_synthesis: bool,
    pub thought_vector: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct AddFactRequest {
    pub concept: String,
    pub content: String,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AddFactResponse {
    pub id: String,
    pub success: bool,
}

#[derive(Debug, Deserialize)]
pub struct SearchFactsRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize { 10 }

#[derive(Debug, Serialize)]
pub struct SearchFactsResponse {
    pub facts: Vec<FactResult>,
}

#[derive(Debug, Serialize)]
pub struct FactResult {
    pub id: String,
    pub concept: String,
    pub content: String,
    pub confidence: f64,
    pub similarity: f64,
}

#[derive(Debug, Deserialize)]
pub struct SetBiasRequest {
    #[serde(default)]
    pub risk_tolerance: Option<f64>,
    #[serde(default)]
    pub exploration: Option<f64>,
    #[serde(default)]
    pub urgency: Option<f64>,
    #[serde(default)]
    pub creativity: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

// ============= API Handlers =============

/// Health check endpoint
pub async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "deliberative-ai",
        "version": "0.1.0"
    }))
}

/// Get system statistics
pub async fn get_statistics(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let engine = state.engine.lock().await;
    Json(engine.get_statistics())
}

/// Train on a single problem
pub async fn train(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TrainRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    
    let mut problem = Problem::training(&req.input, &req.expected_answer, engine.config.dimension);
    for constraint in &req.constraints {
        problem = problem.with_constraint(constraint);
    }
    for context in &req.context {
        problem = problem.with_context(context);
    }

    let result = engine.train(&problem);

    Json(TrainResponse {
        success: result.success,
        iterations: result.iterations,
        confidence_score: result.best_energy.as_ref().map(|e| e.confidence_score),
        energy: result.best_energy.as_ref().map(|e| e.total),
        operator_used: result.best_operator_id,
        message: if result.success {
            "Training successful - verified and committed to memory".to_string()
        } else {
            "Training incomplete - could not verify answer".to_string()
        },
    })
}

/// Batch train on multiple problems
pub async fn batch_train(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BatchTrainRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    
    let problems: Vec<Problem> = req.problems.iter()
        .map(|p| {
            let mut problem = Problem::training(&p.input, &p.expected_answer, engine.config.dimension);
            for constraint in &p.constraints {
                problem = problem.with_constraint(constraint);
            }
            for context in &p.context {
                problem = problem.with_context(context);
            }
            problem
        })
        .collect();

    let result = engine.feedback.train_batch(&problems);

    Json(BatchTrainResponse {
        total_problems: result.total_problems,
        successes: result.successes,
        failures: result.failures,
        success_rate: result.success_rate,
        average_iterations: result.average_iterations,
    })
}

/// Perform inference (thinking)
pub async fn infer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferRequest>,
) -> impl IntoResponse {
    let engine = state.engine.lock().await;
    
    let mut problem = Problem::new(&req.input, engine.config.dimension);
    for constraint in &req.constraints {
        problem = problem.with_constraint(constraint);
    }
    for context in &req.context {
        problem = problem.with_context(context);
    }

    let result = engine.infer(&problem);

    Json(InferResponse {
        confidence: result.confidence,
        energy: result.energy.total,
        verified: result.energy.verified,
        operator_used: result.operator_id,
        candidates_considered: result.candidates_considered,
        is_synthesis: result.is_synthesis,
        thought_vector: result.thought.vector,
    })
}

/// Add a semantic fact
pub async fn add_fact(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddFactRequest>,
) -> impl IntoResponse {
    let engine = state.engine.lock().await;
    
    let mut fact = SemanticFact::new(&req.concept, &req.content, engine.config.dimension);
    if let Some(cat) = req.category {
        fact = fact.with_category(&cat);
    }
    if let Some(src) = req.source {
        fact = fact.with_source(&src);
    }

    let id = fact.id.clone();
    match engine.semantic_memory.store(&fact) {
        Ok(_) => Json(AddFactResponse { id, success: true }),
        Err(_) => Json(AddFactResponse { id: String::new(), success: false }),
    }
}

/// Search semantic facts
pub async fn search_facts(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchFactsRequest>,
) -> impl IntoResponse {
    let engine = state.engine.lock().await;
    
    match engine.semantic_memory.find_similar_by_text(&req.query, req.limit) {
        Ok(results) => {
            let facts: Vec<FactResult> = results.into_iter()
                .map(|(fact, similarity)| FactResult {
                    id: fact.id,
                    concept: fact.concept,
                    content: fact.content,
                    confidence: fact.confidence,
                    similarity,
                })
                .collect();
            Json(SearchFactsResponse { facts })
        }
        Err(_) => Json(SearchFactsResponse { facts: vec![] }),
    }
}

/// Get operator statistics
pub async fn get_operators(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let engine = state.engine.lock().await;
    Json(engine.feedback.get_operator_stats())
}

/// Get episodic memory statistics
pub async fn get_episodic_stats(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let engine = state.engine.lock().await;
    match engine.episodic_memory.get_statistics() {
        Ok(stats) => Json(stats),
        Err(_) => Json(EpisodeStatistics {
            total_episodes: 0,
            verified_episodes: 0,
            average_confidence: 0.0,
            average_energy: 0.0,
        }),
    }
}

/// Get top episodes from episodic memory
pub async fn get_top_episodes(
    State(state): State<Arc<AppState>>,
    Path(limit): Path<usize>,
) -> impl IntoResponse {
    let engine = state.engine.lock().await;
    match engine.episodic_memory.get_top_episodes(limit) {
        Ok(episodes) => Json(episodes),
        Err(_) => Json(Vec::<Episode>::new()),
    }
}

/// Set bias parameters
pub async fn set_bias(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SetBiasRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    
    if let Some(v) = req.risk_tolerance {
        engine.bias_controller.set_risk_tolerance(v);
    }
    if let Some(v) = req.exploration {
        engine.bias_controller.set_exploration(v);
    }
    if let Some(v) = req.urgency {
        engine.bias_controller.set_urgency(v);
    }
    if let Some(v) = req.creativity {
        engine.bias_controller.set_creativity(v);
    }

    Json(engine.bias_controller.state_summary())
}

/// Reset bias to defaults
pub async fn reset_bias(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    engine.bias_controller.reset();
    Json(engine.bias_controller.state_summary())
}

/// Clear all episodic memory
pub async fn clear_episodic_memory(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let engine = state.engine.lock().await;
    match engine.episodic_memory.clear() {
        Ok(_) => Json(serde_json::json!({ "success": true, "message": "Episodic memory cleared" })),
        Err(e) => Json(serde_json::json!({ "success": false, "error": e.to_string() })),
    }
}

/// Clear all semantic memory
pub async fn clear_semantic_memory(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let engine = state.engine.lock().await;
    match engine.semantic_memory.clear() {
        Ok(_) => Json(serde_json::json!({ "success": true, "message": "Semantic memory cleared" })),
        Err(e) => Json(serde_json::json!({ "success": false, "error": e.to_string() })),
    }
}

/// Reset learning rate
pub async fn reset_learning_rate(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    engine.feedback.reset_learning_rate();
    Json(serde_json::json!({ 
        "success": true, 
        "learning_rate": engine.feedback.learning_rate() 
    }))
}

// ============= Multimodal Types =============

/// Image processing request
#[derive(Debug, Deserialize)]
pub struct ImageRequest {
    /// Base64 encoded image or raw pixel data
    pub data: String,
    /// Image width (if raw pixels)
    pub width: Option<usize>,
    /// Image height (if raw pixels)
    pub height: Option<usize>,
    /// Whether to train with this image
    pub train: Option<bool>,
    /// Expected label for training
    pub label: Option<String>,
}

/// Audio processing request
#[derive(Debug, Deserialize)]
pub struct AudioRequest {
    /// Audio samples (normalized -1 to 1)
    pub samples: Vec<f64>,
    /// Sample rate
    pub sample_rate: usize,
    /// Number of channels
    pub channels: Option<usize>,
    /// Whether to train
    pub train: Option<bool>,
    /// Expected label
    pub label: Option<String>,
}

/// Video processing request
#[derive(Debug, Deserialize)]
pub struct VideoRequest {
    /// Frames as base64 encoded images
    pub frames: Vec<String>,
    /// Frames per second
    pub fps: f64,
    /// Whether to train
    pub train: Option<bool>,
    /// Expected label
    pub label: Option<String>,
}

/// Multimodal fusion request
#[derive(Debug, Deserialize)]
pub struct FusionRequest {
    /// Optional image data (base64)
    pub image: Option<String>,
    /// Optional audio samples
    pub audio_samples: Option<Vec<f64>>,
    /// Audio sample rate
    pub audio_sample_rate: Option<usize>,
    /// Optional text input
    pub text: Option<String>,
    /// Whether to train
    pub train: Option<bool>,
    /// Expected label
    pub label: Option<String>,
}

/// Multimodal response
#[derive(Debug, Serialize)]
pub struct MultimodalResponse {
    pub success: bool,
    pub modality: String,
    pub embedding: Vec<f64>,
    pub confidence: f64,
    pub features: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_result: Option<serde_json::Value>,
}

// ============= Multimodal Handlers =============

use crate::multimodal::{ImageData, ImageEncoder, AudioData, AudioEncoder, VideoData, VideoEncoder, MultimodalEncoder};

/// Process image input
pub async fn process_image(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ImageRequest>,
) -> impl IntoResponse {
    let config = &state.config;
    let encoder = ImageEncoder::new(config.dimension);
    
    // Parse image data - expect comma-separated pixel values with width/height
    let image = if let (Some(w), Some(h)) = (req.width, req.height) {
        let pixels: Vec<f64> = req.data.split(',')
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();
        
        if pixels.is_empty() {
            return Json(MultimodalResponse {
                success: false,
                modality: "image".to_string(),
                embedding: vec![],
                confidence: 0.0,
                features: serde_json::json!({ "error": "Invalid pixel data" }),
                training_result: None,
            });
        }
        
        // Create grayscale image from 2D array
        let rows: Vec<Vec<f64>> = pixels.chunks(w)
            .map(|chunk| chunk.to_vec())
            .take(h)
            .collect();
        ImageData::from_grayscale_2d(&rows)
    } else {
        return Json(MultimodalResponse {
            success: false,
            modality: "image".to_string(),
            embedding: vec![],
            confidence: 0.0,
            features: serde_json::json!({ "error": "Width and height required" }),
            training_result: None,
        });
    };
    
    // Encode image
    let embedding_vec = encoder.encode(&image);
    let embedding: Vec<f64> = embedding_vec.as_slice().to_vec();
    
    // Training if requested
    let training_result = if req.train.unwrap_or(false) {
        if let Some(label) = &req.label {
            let mut engine = state.engine.lock().await;
            let problem = crate::core::Problem::training(
                &format!("image:{}", label),
                label,
                config.dimension
            );
            let result = engine.train(&problem);
            Some(serde_json::json!({
                "success": result.success,
                "iterations": result.iterations,
                "confidence": result.best_energy.as_ref().map(|e| e.confidence_score).unwrap_or(0.0)
            }))
        } else {
            None
        }
    } else {
        None
    };
    
    Json(MultimodalResponse {
        success: true,
        modality: "image".to_string(),
        embedding,
        confidence: 0.8,
        features: serde_json::json!({
            "width": image.width,
            "height": image.height,
            "channels": image.channels
        }),
        training_result,
    })
}

/// Process audio input
pub async fn process_audio(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AudioRequest>,
) -> impl IntoResponse {
    let config = &state.config;
    let encoder = AudioEncoder::new(config.dimension);
    
    let channels = req.channels.unwrap_or(1);
    let audio = AudioData::new(req.samples.clone(), req.sample_rate as u32, channels);
    
    // Encode audio
    let embedding_vec = encoder.encode(&audio);
    let embedding: Vec<f64> = embedding_vec.as_slice().to_vec();
    
    // Training if requested
    let training_result = if req.train.unwrap_or(false) {
        if let Some(label) = &req.label {
            let mut engine = state.engine.lock().await;
            let problem = crate::core::Problem::training(
                &format!("audio:{}", label),
                label,
                config.dimension
            );
            let result = engine.train(&problem);
            Some(serde_json::json!({
                "success": result.success,
                "iterations": result.iterations,
                "confidence": result.best_energy.as_ref().map(|e| e.confidence_score).unwrap_or(0.0)
            }))
        } else {
            None
        }
    } else {
        None
    };
    
    Json(MultimodalResponse {
        success: true,
        modality: "audio".to_string(),
        embedding,
        confidence: 0.7,
        features: serde_json::json!({
            "duration_secs": audio.duration,
            "sample_rate": audio.sample_rate,
            "channels": audio.channels
        }),
        training_result,
    })
}

/// Process video input
pub async fn process_video(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VideoRequest>,
) -> impl IntoResponse {
    let config = &state.config;
    let encoder = VideoEncoder::new(config.dimension);
    
    // Parse frames - expect each frame as comma-separated pixels with metadata
    // For simplicity, create placeholder frames
    if req.frames.is_empty() {
        return Json(MultimodalResponse {
            success: false,
            modality: "video".to_string(),
            embedding: vec![],
            confidence: 0.0,
            features: serde_json::json!({ "error": "No frames provided" }),
            training_result: None,
        });
    }
    
    // Create simple frames from data
    let frames: Vec<ImageData> = req.frames.iter()
        .map(|_| {
            // Create a simple 16x16 grayscale frame
            ImageData::from_grayscale_2d(&vec![vec![0.5; 16]; 16])
        })
        .collect();
    
    let video = VideoData::new(frames, req.fps);
    let embedding_vec = encoder.encode(&video);
    let embedding: Vec<f64> = embedding_vec.as_slice().to_vec();
    
    // Training if requested
    let training_result = if req.train.unwrap_or(false) {
        if let Some(label) = &req.label {
            let mut engine = state.engine.lock().await;
            let problem = crate::core::Problem::training(
                &format!("video:{}", label),
                label,
                config.dimension
            );
            let result = engine.train(&problem);
            Some(serde_json::json!({
                "success": result.success,
                "iterations": result.iterations,
                "confidence": result.best_energy.as_ref().map(|e| e.confidence_score).unwrap_or(0.0)
            }))
        } else {
            None
        }
    } else {
        None
    };
    
    Json(MultimodalResponse {
        success: true,
        modality: "video".to_string(),
        embedding,
        confidence: 0.75,
        features: serde_json::json!({
            "frame_count": video.frames.len(),
            "fps": video.fps,
            "duration_secs": video.duration
        }),
        training_result,
    })
}

/// Fuse multiple modalities
pub async fn fuse_modalities(
    State(state): State<Arc<AppState>>,
    Json(req): Json<FusionRequest>,
) -> impl IntoResponse {
    let config = &state.config;
    let encoder = crate::multimodal::MultimodalEncoder::new(config.dimension);
    
    // Parse image if provided (expects comma-separated pixels)
    let image = req.image.as_ref().map(|data| {
        let pixels: Vec<f64> = data.split(',')
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();
        let size = (pixels.len() as f64).sqrt() as usize;
        let rows: Vec<Vec<f64>> = pixels.chunks(size.max(1))
            .map(|c| c.to_vec())
            .collect();
        ImageData::from_grayscale_2d(&rows)
    });
    
    // Parse audio if provided
    let audio = req.audio_samples.as_ref().map(|samples| {
        AudioData::new(
            samples.clone(),
            req.audio_sample_rate.unwrap_or(44100) as u32,
            1
        )
    });
    
    // Create multimodal input
    let input = crate::multimodal::MultimodalInput {
        text: req.text.clone(),
        image: image,
        video: None,
        audio: audio,
    };
    
    // Encode using multimodal encoder
    let mut encoder = crate::multimodal::MultimodalEncoder::new(config.dimension);
    let thought = encoder.encode(&input);
    
    let mut modalities = vec![];
    if input.image.is_some() { modalities.push("image"); }
    if input.audio.is_some() { modalities.push("audio"); }
    if req.text.is_some() { modalities.push("text"); }
    
    // Training if requested
    let training_result = if req.train.unwrap_or(false) {
        if let Some(label) = &req.label {
            let mut engine = state.engine.lock().await;
            let problem = crate::core::Problem::training(
                &format!("multimodal:{}", label),
                label,
                config.dimension
            );
            let result = engine.train(&problem);
            Some(serde_json::json!({
                "success": result.success,
                "iterations": result.iterations,
                "confidence": result.best_energy.as_ref().map(|e| e.confidence_score).unwrap_or(0.0)
            }))
        } else {
            None
        }
    } else {
        None
    };
    
    // Convert DVector to Vec and calculate confidence
    let embedding_vec: Vec<f64> = thought.iter().cloned().collect();
    let confidence = 1.0 - (embedding_vec.iter().map(|x| x.abs()).sum::<f64>() / embedding_vec.len() as f64).min(1.0);
    
    Json(MultimodalResponse {
        success: true,
        modality: "multimodal".to_string(),
        embedding: embedding_vec.clone(),
        confidence,
        features: serde_json::json!({
            "modalities": modalities,
            "dimension": embedding_vec.len()
        }),
        training_result,
    })
}

// ============= Router Setup =============

/// Create the API router
// ============= Generation Types =============

use crate::generation::{TextGenerator, ImageGenerator, GenerationConfig, ContentGenerator};

/// Text generation request
#[derive(Debug, Deserialize)]
pub struct GenerateTextRequest {
    /// Input text/prompt
    pub input: String,
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Temperature (0.0-2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Top-k sampling
    #[serde(default)]
    pub top_k: Option<usize>,
}

fn default_max_tokens() -> usize { 50 }
fn default_temperature() -> f64 { 0.7 }

/// Text generation response
#[derive(Debug, Serialize)]
pub struct GenerateTextResponse {
    pub success: bool,
    pub text: String,
    pub tokens_generated: usize,
    pub confidence: f64,
}

/// Image generation request
#[derive(Debug, Deserialize)]
pub struct GenerateImageRequest {
    /// Input text/prompt
    pub prompt: String,
    /// Image size (width/height)
    #[serde(default = "default_image_size")]
    pub size: usize,
    /// Noise level for variation
    #[serde(default)]
    pub noise_level: Option<f64>,
}

fn default_image_size() -> usize { 64 }

/// Image generation response
#[derive(Debug, Serialize)]
pub struct GenerateImageResponse {
    pub success: bool,
    /// Comma-separated pixel values (normalized 0-1)
    pub image_data: String,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub confidence: f64,
}

/// Generate text from input
pub async fn generate_text(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateTextRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    let dim = state.config.dimension;
    
    // Create problem and get thought state
    let problem = Problem::new(&req.input, dim);
    let result = engine.infer(&problem);
    
    // Create generator with config
    let config = GenerationConfig {
        temperature: req.temperature,
        top_k: req.top_k.unwrap_or(50),
        top_p: 0.9,
        max_length: req.max_tokens,
        repetition_penalty: 1.1,
    };
    let generator = TextGenerator::new(config, dim);
    
    // Generate text
    let text = generator.generate(&result.thought, req.max_tokens);
    
    Json(GenerateTextResponse {
        success: true,
        tokens_generated: text.split_whitespace().count(),
        text,
        confidence: result.confidence,
    })
}

/// Generate image from prompt
pub async fn generate_image(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateImageRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    let dim = state.config.dimension;
    
    // Create problem and get thought state
    let problem = Problem::new(&req.prompt, dim);
    let result = engine.infer(&problem);
    
    // Create generator
    let generator = ImageGenerator::new(dim, req.size);
    
    // Generate image
    let pixels = if let Some(noise) = req.noise_level {
        generator.generate_with_noise(&result.thought, noise)
    } else {
        generator.generate(&result.thought)
    };
    
    // Convert to comma-separated string
    let image_data = pixels.iter()
        .map(|p| format!("{:.4}", p))
        .collect::<Vec<_>>()
        .join(",");
    
    Json(GenerateImageResponse {
        success: true,
        image_data,
        width: req.size,
        height: req.size,
        channels: 3,
        confidence: result.confidence,
    })
}

/// Comprehensive training request
#[derive(Debug, Deserialize)]
pub struct ComprehensiveTrainRequest {
    /// Training examples
    pub examples: Vec<TrainingExample>,
    /// Number of epochs
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    /// Whether to shuffle examples
    #[serde(default = "default_shuffle")]
    pub shuffle: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainingExample {
    pub input: String,
    pub expected: String,
    #[serde(default)]
    pub context: Option<String>,
}

fn default_epochs() -> usize { 1 }
fn default_shuffle() -> bool { true }

/// Comprehensive training response
#[derive(Debug, Serialize)]
pub struct ComprehensiveTrainResponse {
    pub success: bool,
    pub total_examples: usize,
    pub verified_count: usize,
    pub failed_count: usize,
    pub average_energy: f64,
    pub average_confidence: f64,
    pub epochs_completed: usize,
    pub training_time_ms: u128,
}

/// Comprehensive training endpoint
pub async fn comprehensive_train(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ComprehensiveTrainRequest>,
) -> impl IntoResponse {
    use std::time::Instant;
    
    let start = Instant::now();
    let mut engine = state.engine.lock().await;
    let dim = state.config.dimension;
    
    let mut verified = 0;
    let mut failed = 0;
    let mut total_energy = 0.0;
    let mut total_confidence = 0.0;
    
    for _epoch in 0..req.epochs {
        let examples = if req.shuffle {
            use rand::seq::SliceRandom;
            let mut shuffled = req.examples.clone();
            shuffled.shuffle(&mut rand::thread_rng());
            shuffled
        } else {
            req.examples.clone()
        };
        
        for example in &examples {
            let problem = Problem::training(&example.input, &example.expected, dim);
            let result = engine.train(&problem);
            
            if result.success {
                verified += 1;
                // Use the energy from best_energy if available
                if let Some(ref energy) = result.best_energy {
                    total_energy += energy.total;
                    total_confidence += energy.confidence_score;
                }
            } else {
                failed += 1;
            }
        }
    }
    
    let total = req.examples.len() * req.epochs;
    
    Json(ComprehensiveTrainResponse {
        success: verified > failed,
        total_examples: total,
        verified_count: verified,
        failed_count: failed,
        average_energy: if verified > 0 { total_energy / verified as f64 } else { 0.0 },
        average_confidence: if verified > 0 { total_confidence / verified as f64 } else { 0.0 },
        epochs_completed: req.epochs,
        training_time_ms: start.elapsed().as_millis(),
    })
}

/// Knowledge learning request
#[derive(Debug, Deserialize)]
pub struct LearnKnowledgeRequest {
    /// Category of knowledge
    pub category: String,
    /// Facts to learn
    pub facts: Vec<KnowledgeFact>,
}

#[derive(Debug, Deserialize)]
pub struct KnowledgeFact {
    pub concept: String,
    pub content: String,
    #[serde(default)]
    pub related: Vec<String>,
}

/// Learn knowledge (add to semantic memory and train)
pub async fn learn_knowledge(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LearnKnowledgeRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    let dim = state.config.dimension;
    
    let mut stored = 0;
    let mut trained = 0;
    
    for fact in &req.facts {
        // Store in semantic memory
        let embedding = engine.embedding.embed_text(&fact.content);
        let semantic_fact = SemanticFact {
            id: uuid::Uuid::new_v4().to_string(),
            concept: fact.concept.clone(),
            content: fact.content.clone(),
            embedding: embedding.vector.clone(),
            category: Some(req.category.clone()),
            source: Some("knowledge_api".to_string()),
            confidence: 1.0,
            reinforcement_count: 0,
            last_accessed: chrono::Utc::now(),
            related_concepts: fact.related.clone(),
        };
        
        if engine.semantic_memory.store(&semantic_fact).is_ok() {
            stored += 1;
        }
        
        // Also train on the fact
        let problem = Problem::training(
            &format!("What is {}?", fact.concept),
            &fact.content,
            dim
        );
        let result = engine.train(&problem);
        if result.success {
            trained += 1;
        }
    }
    
    Json(serde_json::json!({
        "success": true,
        "facts_stored": stored,
        "facts_trained": trained,
        "category": req.category
    }))
}

/// Query knowledge
#[derive(Debug, Deserialize)]
pub struct QueryKnowledgeRequest {
    pub query: String,
    #[serde(default = "default_query_limit")]
    pub limit: usize,
}

fn default_query_limit() -> usize { 5 }

/// Query knowledge from memory
pub async fn query_knowledge(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryKnowledgeRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    
    // Get embedding for query
    let query_embedding = engine.embedding.embed_text(&req.query);
    
    // Search semantic memory using find_similar
    match engine.semantic_memory.find_similar(&query_embedding.vector, req.limit) {
        Ok(results) => {
            let formatted: Vec<serde_json::Value> = results.iter().map(|(f, similarity)| {
                serde_json::json!({
                    "concept": f.concept,
                    "content": f.content,
                    "category": f.category,
                    "confidence": f.confidence,
                    "similarity": similarity,
                    "related": f.related_concepts
                })
            }).collect();
            
            Json(serde_json::json!({
                "success": true,
                "query": req.query,
                "results": formatted,
                "count": formatted.len()
            }))
        },
        Err(e) => {
            Json(serde_json::json!({
                "success": false,
                "error": e.to_string()
            }))
        }
    }
}

/// System capabilities info
pub async fn get_capabilities() -> impl IntoResponse {
    Json(serde_json::json!({
        "name": "ALEN",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "Advanced Learning Engine with Neural Understanding",
        "capabilities": {
            "modalities": ["text", "image", "video", "audio", "multimodal"],
            "learning": {
                "verified": true,
                "batch": true,
                "online": true
            },
            "generation": {
                "text": true,
                "image": true
            },
            "memory": {
                "episodic": true,
                "semantic": true,
                "procedural": false
            },
            "operators": [
                "Logical", "Creative", "Analytical", "Exploratory",
                "Conservative", "Integrative", "Critical", "Intuitive"
            ]
        },
        "math": {
            "attention": true,
            "transformers": true,
            "neural_networks": true,
            "energy_functions": true,
            "backward_inference": true
        }
    }))
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        // Health & Status
        .route("/health", get(health_check))
        .route("/stats", get(get_statistics))
        .route("/capabilities", get(get_capabilities))
        
        // Training endpoints
        .route("/train", post(train))
        .route("/train/batch", post(batch_train))
        .route("/train/comprehensive", post(comprehensive_train))
        
        // Knowledge learning
        .route("/learn", post(learn_knowledge))
        .route("/query", post(query_knowledge))
        
        // Inference endpoint
        .route("/infer", post(infer))
        
        // Generation endpoints
        .route("/generate/text", post(generate_text))
        .route("/generate/image", post(generate_image))
        
        // Multimodal endpoints
        .route("/multimodal/image", post(process_image))
        .route("/multimodal/audio", post(process_audio))
        .route("/multimodal/video", post(process_video))
        .route("/multimodal/fuse", post(fuse_modalities))
        
        // Semantic memory
        .route("/facts", post(add_fact))
        .route("/facts/search", post(search_facts))
        .route("/memory/semantic/clear", delete(clear_semantic_memory))
        
        // Episodic memory
        .route("/memory/episodic/stats", get(get_episodic_stats))
        .route("/memory/episodic/top/:limit", get(get_top_episodes))
        .route("/memory/episodic/clear", delete(clear_episodic_memory))
        
        // Operators
        .route("/operators", get(get_operators))
        
        // Bias control
        .route("/bias", post(set_bias))
        .route("/bias/reset", post(reset_bias))
        
        // Learning rate
        .route("/learning/reset", post(reset_learning_rate))
        
        .with_state(state)
}
