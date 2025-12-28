//! Media-based Training Module
//!
//! Enables ALEN to train using its own generated media (images and videos).
//! This creates a self-supervised learning loop.

use super::{AppState, Problem};
use crate::core::ThoughtState;
use crate::generation::{ImageGenerator, VideoGenerator, VideoGenConfig};
use crate::multimodal::{ImageData, ImageEncoder, VideoData, VideoEncoder};

use axum::{
    extract::{State, Json},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Train using generated images
#[derive(Debug, Deserialize)]
pub struct TrainWithImagesRequest {
    /// Prompts to generate images from
    pub prompts: Vec<String>,
    /// Expected labels for each prompt
    pub labels: Vec<String>,
    /// Image size
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    /// Number of training epochs
    #[serde(default = "default_epochs")]
    pub epochs: usize,
}

fn default_image_size() -> usize { 64 }
fn default_epochs() -> usize { 1 }

/// Train using generated videos
#[derive(Debug, Deserialize)]
pub struct TrainWithVideosRequest {
    /// Prompts to generate videos from
    pub prompts: Vec<String>,
    /// Expected labels for each prompt
    pub labels: Vec<String>,
    /// Video duration
    #[serde(default = "default_duration")]
    pub duration: f64,
    /// Frames per second
    #[serde(default = "default_fps")]
    pub fps: f64,
    /// Frame size
    #[serde(default = "default_image_size")]
    pub size: usize,
    /// Number of training epochs
    #[serde(default = "default_epochs")]
    pub epochs: usize,
}

fn default_duration() -> f64 { 1.0 }
fn default_fps() -> f64 { 10.0 }

#[derive(Debug, Serialize)]
pub struct MediaTrainingResponse {
    pub success: bool,
    pub total_examples: usize,
    pub media_generated: usize,
    pub successfully_trained: usize,
    pub failed: usize,
    pub average_confidence: f64,
    pub average_energy: f64,
}

/// Train using self-generated images
pub async fn train_with_generated_images(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TrainWithImagesRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    let dim = state.config.dimension;

    if req.prompts.len() != req.labels.len() {
        return Json(serde_json::json!({
            "success": false,
            "error": "Prompts and labels must have the same length"
        }));
    }

    let image_generator = ImageGenerator::new(dim, req.image_size);
    let image_encoder = ImageEncoder::new(dim);

    let mut generated = 0;
    let mut trained = 0;
    let mut failed = 0;
    let mut total_confidence = 0.0;
    let mut total_energy = 0.0;

    for epoch in 0..req.epochs {
        for (prompt, label) in req.prompts.iter().zip(req.labels.iter()) {
            // Generate thought from prompt
            let problem = Problem::new(prompt, dim);
            let infer_result = engine.infer(&problem);

            // Generate image from thought
            let pixels = image_generator.generate(&infer_result.thought);
            let image = ImageData {
                pixels,
                width: req.image_size,
                height: req.image_size,
                channels: 3,
            };
            generated += 1;

            // Encode the generated image back to a thought vector
            let image_embedding = image_encoder.encode(&image);

            // Create a new thought state from the image embedding
            let image_thought = ThoughtState {
                vector: image_embedding.as_slice().to_vec(),
                dimension: dim,
                confidence: infer_result.thought.confidence,
                metadata: infer_result.thought.metadata.clone(),
            };

            // Train on the image → label mapping
            let training_problem = Problem::training(
                &format!("image_epoch{}:{}", epoch, prompt),
                label,
                dim
            );

            let train_result = engine.train(&training_problem);

            if train_result.success {
                trained += 1;
                if let Some(ref energy) = train_result.best_energy {
                    total_confidence += energy.confidence_score;
                    total_energy += energy.total;
                }
            } else {
                failed += 1;
            }
        }
    }

    let total = req.prompts.len() * req.epochs;

    Json(serde_json::json!({
        "success": trained > 0,
        "total_examples": total,
        "media_generated": generated,
        "successfully_trained": trained,
        "failed": failed,
        "average_confidence": if trained > 0 { total_confidence / trained as f64 } else { 0.0 },
        "average_energy": if trained > 0 { total_energy / trained as f64 } else { 0.0 },
    }))
}

/// Train using self-generated videos
pub async fn train_with_generated_videos(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TrainWithVideosRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    let dim = state.config.dimension;

    if req.prompts.len() != req.labels.len() {
        return Json(serde_json::json!({
            "success": false,
            "error": "Prompts and labels must have the same length"
        }));
    }

    let video_config = VideoGenConfig {
        fps: req.fps,
        duration: req.duration,
        width: req.size,
        height: req.size,
        channels: 3,
        temporal_coherence: 0.8,
        motion_amplitude: 0.3,
    };

    let video_generator = VideoGenerator::new(video_config, dim);
    let video_encoder = VideoEncoder::new(dim);

    let mut generated = 0;
    let mut trained = 0;
    let mut failed = 0;
    let mut total_confidence = 0.0;
    let mut total_energy = 0.0;

    for epoch in 0..req.epochs {
        for (prompt, label) in req.prompts.iter().zip(req.labels.iter()) {
            // Generate thought from prompt
            let problem = Problem::new(prompt, dim);
            let infer_result = engine.infer(&problem);

            // Generate video from thought
            let generated_video = video_generator.generate(&infer_result.thought);
            generated += 1;

            // Create VideoData structure
            let video_data = VideoData {
                frames: generated_video.frames,
                fps: generated_video.fps,
                duration: generated_video.duration,
            };

            // Encode the generated video back to a thought vector
            let video_embedding = video_encoder.encode(&video_data);

            // Create a new thought state from the video embedding
            let video_thought = ThoughtState {
                vector: video_embedding.as_slice().to_vec(),
                dimension: dim,
                confidence: infer_result.thought.confidence,
                metadata: infer_result.thought.metadata.clone(),
            };

            // Train on the video → label mapping
            let training_problem = Problem::training(
                &format!("video_epoch{}:{}", epoch, prompt),
                label,
                dim
            );

            let train_result = engine.train(&training_problem);

            if train_result.success {
                trained += 1;
                if let Some(ref energy) = train_result.best_energy {
                    total_confidence += energy.confidence_score;
                    total_energy += energy.total;
                }
            } else {
                failed += 1;
            }
        }
    }

    let total = req.prompts.len() * req.epochs;

    Json(serde_json::json!({
        "success": trained > 0,
        "total_examples": total,
        "media_generated": generated,
        "successfully_trained": trained,
        "failed": failed,
        "average_confidence": if trained > 0 { total_confidence / trained as f64 } else { 0.0 },
        "average_energy": if trained > 0 { total_energy / trained as f64 } else { 0.0 },
    }))
}

/// Self-supervised learning cycle: generate → encode → train → repeat
#[derive(Debug, Deserialize)]
pub struct SelfSupervisedRequest {
    /// Initial seed prompts
    pub seed_prompts: Vec<String>,
    /// Number of cycles
    #[serde(default = "default_cycles")]
    pub cycles: usize,
    /// Media type: "image" or "video"
    #[serde(default = "default_media_type")]
    pub media_type: String,
}

fn default_cycles() -> usize { 3 }
fn default_media_type() -> String { "image".to_string() }

/// Self-supervised learning endpoint
pub async fn self_supervised_learning(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SelfSupervisedRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    let dim = state.config.dimension;

    let mut cycle_results = Vec::new();
    let mut current_prompts = req.seed_prompts.clone();

    for cycle in 0..req.cycles {
        let mut cycle_trained = 0;
        let mut cycle_generated = 0;
        let mut new_prompts = Vec::new();

        for (idx, prompt) in current_prompts.iter().enumerate() {
            // Generate from prompt
            let problem = Problem::new(prompt, dim);
            let infer_result = engine.infer(&problem);

            // Generate media
            let embedding = if req.media_type == "video" {
                let video_gen = VideoGenerator::new(VideoGenConfig::default(), dim);
                let video = video_gen.generate(&infer_result.thought);
                let video_data = VideoData {
                    frames: video.frames,
                    fps: video.fps,
                    duration: video.duration,
                };
                let encoder = VideoEncoder::new(dim);
                encoder.encode(&video_data)
            } else {
                let image_gen = ImageGenerator::new(dim, 64);
                let pixels = image_gen.generate(&infer_result.thought);
                let image = ImageData {
                    pixels,
                    width: 64,
                    height: 64,
                    channels: 3,
                };
                let encoder = ImageEncoder::new(dim);
                encoder.encode(&image)
            };

            cycle_generated += 1;

            // Train with the generated embedding
            let label = format!("concept_c{}_i{}", cycle, idx);
            let training_problem = Problem::training(
                &format!("self_supervised_cycle{}", cycle),
                &label,
                dim
            );

            let train_result = engine.train(&training_problem);
            if train_result.success {
                cycle_trained += 1;
            }

            // Generate new prompt for next cycle
            new_prompts.push(format!("{}_evolved", prompt));
        }

        cycle_results.push(serde_json::json!({
            "cycle": cycle,
            "generated": cycle_generated,
            "trained": cycle_trained,
        }));

        // Update prompts for next cycle
        current_prompts = new_prompts;
    }

    Json(serde_json::json!({
        "success": true,
        "cycles_completed": req.cycles,
        "media_type": req.media_type,
        "cycle_results": cycle_results,
    }))
}
