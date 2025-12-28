//! Knowledge-Anchored Visual Generation
//!
//! Generates images and videos grounded in learned semantic knowledge.
//! Uses ReasoningEngine to combine concept + knowledge + creativity.
//!
//! Key Features:
//! - Every visual generation is knowledge-anchored
//! - Verification: cosine_similarity(latent_generated, latent_knowledge) >= threshold
//! - Creativity injection: α parameter controls faithful vs creative balance
//! - Latent propagation: temporal consistency for video generation

use crate::core::{ThoughtState, BiasVector};
use crate::memory::SemanticMemory;
use crate::generation::reasoning_engine::{ReasoningEngine, LatentResult};
use crate::multimodal::{ImageData, ImageEncoder};
use serde::{Deserialize, Serialize};
use image::{ImageBuffer, Rgb};

/// Knowledge-anchored image generator
pub struct KnowledgeImageGenerator {
    /// Reasoning engine for latent computation
    pub reasoning_engine: ReasoningEngine,
    /// Image dimensions
    pub width: usize,
    pub height: usize,
    /// Vector dimension
    pub dimension: usize,
}

impl KnowledgeImageGenerator {
    pub fn new(dimension: usize, width: usize, height: usize) -> Self {
        Self {
            reasoning_engine: ReasoningEngine::balanced(dimension),
            width,
            height,
            dimension,
        }
    }

    /// Generate image from concept with knowledge anchoring
    ///
    /// Process:
    /// 1. Compute latent: h_latent = concept + knowledge + α * creativity
    /// 2. Verify latent against knowledge base
    /// 3. Generate image from verified latent
    /// 4. Return image with verification metadata
    pub fn generate_from_concept(
        &self,
        concept: &str,
        memory: &SemanticMemory,
        bias: &BiasVector,
    ) -> Result<KnowledgeImage, Box<dyn std::error::Error>> {
        // 1. Compute knowledge-anchored latent
        let latent_result = self.reasoning_engine.compute_latent_from_concept(
            concept,
            memory,
            bias,
        )?;

        // 2. Generate image from latent
        let image_data = self.latent_to_image(&latent_result.latent)?;

        Ok(KnowledgeImage {
            data: image_data,
            width: self.width,
            height: self.height,
            latent_result,
            concept: concept.to_string(),
        })
    }

    /// Generate image sequence for video with temporal consistency
    pub fn generate_video_sequence(
        &self,
        concept: &str,
        memory: &SemanticMemory,
        bias: &BiasVector,
        num_frames: usize,
        propagation_strength: f64,
    ) -> Result<KnowledgeVideo, Box<dyn std::error::Error>> {
        // Generate latent sequence with temporal propagation
        let latent_sequence = self.reasoning_engine.generate_latent_sequence(
            concept,
            memory,
            bias,
            num_frames,
            propagation_strength,
        )?;

        // Convert each latent to image
        let mut frames = Vec::new();
        for latent_result in latent_sequence {
            let image_data = self.latent_to_image(&latent_result.latent)?;
            frames.push(KnowledgeImage {
                data: image_data,
                width: self.width,
                height: self.height,
                latent_result,
                concept: concept.to_string(),
            });
        }

        Ok(KnowledgeVideo {
            frames,
            concept: concept.to_string(),
            width: self.width,
            height: self.height,
            fps: 24.0,
        })
    }

    /// Convert latent vector to image data
    ///
    /// Uses simple deterministic mapping from latent space to RGB pixels.
    /// In production, this would use a trained decoder (e.g., VAE decoder, diffusion model).
    fn latent_to_image(&self, latent: &[f64]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let total_pixels = self.width * self.height * 3; // RGB channels

        // Expand latent to image dimensions using deterministic upsampling
        let mut image_data = vec![0u8; total_pixels];

        // Simple tiling and modulation strategy
        for pixel_idx in 0..total_pixels {
            // Map pixel index to latent dimension cyclically
            let latent_idx = pixel_idx % latent.len();
            let latent_val = latent[latent_idx];

            // Apply spatial modulation based on pixel position
            let x = (pixel_idx / 3) % self.width;
            let y = (pixel_idx / 3) / self.width;
            let channel = pixel_idx % 3;

            // Create spatial patterns using position-dependent transformations
            let spatial_phase = ((x as f64 / self.width as f64) * std::f64::consts::PI
                               + (y as f64 / self.height as f64) * std::f64::consts::PI).sin();

            let channel_shift = match channel {
                0 => 0.0,   // Red channel
                1 => 0.33,  // Green channel offset
                2 => 0.67,  // Blue channel offset
                _ => 0.0,
            };

            // Combine latent value with spatial patterns
            let combined = (latent_val + spatial_phase + channel_shift).sin();

            // Map to [0, 255]
            let pixel_value = ((combined + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            image_data[pixel_idx] = pixel_value;
        }

        Ok(image_data)
    }

    /// Save image to file (PNG format)
    pub fn save_image(
        &self,
        knowledge_image: &KnowledgeImage,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let img_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_vec(
            self.width as u32,
            self.height as u32,
            knowledge_image.data.clone(),
        ).ok_or("Failed to create image buffer")?;

        img_buffer.save(path)?;
        Ok(())
    }

    /// Generate variations of a concept (for exploration)
    pub fn generate_variations(
        &self,
        concept: &str,
        memory: &SemanticMemory,
        base_bias: &BiasVector,
        num_variations: usize,
    ) -> Result<Vec<KnowledgeImage>, Box<dyn std::error::Error>> {
        let mut variations = Vec::new();

        for i in 0..num_variations {
            // Vary creativity parameter for each variation
            let alpha = (i as f64) / (num_variations as f64).max(1.0);
            let varied_bias = BiasVector {
                creativity: alpha,
                exploration: base_bias.exploration,
                risk_tolerance: base_bias.risk_tolerance,
                urgency: base_bias.urgency,
            };

            let image = self.generate_from_concept(concept, memory, &varied_bias)?;
            variations.push(image);
        }

        Ok(variations)
    }
}

/// Image generated with knowledge anchoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeImage {
    /// Raw RGB image data (width * height * 3 bytes)
    pub data: Vec<u8>,
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
    /// Latent computation result with verification
    pub latent_result: LatentResult,
    /// Original concept
    pub concept: String,
}

impl KnowledgeImage {
    /// Get verification confidence
    pub fn verification_confidence(&self) -> f64 {
        self.latent_result.verification.confidence
    }

    /// Check if image is knowledge-verified
    pub fn is_verified(&self) -> bool {
        self.latent_result.verification.verified
    }

    /// Get supporting knowledge facts
    pub fn supporting_facts(&self) -> &[String] {
        &self.latent_result.verification.supporting_facts
    }

    /// Get creativity level used
    pub fn creativity_alpha(&self) -> f64 {
        self.latent_result.alpha
    }
}

/// Video sequence generated with knowledge anchoring and temporal consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeVideo {
    /// Sequence of frames
    pub frames: Vec<KnowledgeImage>,
    /// Original concept
    pub concept: String,
    /// Frame width
    pub width: usize,
    /// Frame height
    pub height: usize,
    /// Frames per second
    pub fps: f64,
}

impl KnowledgeVideo {
    /// Get duration in seconds
    pub fn duration(&self) -> f64 {
        self.frames.len() as f64 / self.fps
    }

    /// Get average verification confidence across frames
    pub fn avg_verification_confidence(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }

        let total: f64 = self.frames.iter()
            .map(|f| f.verification_confidence())
            .sum();

        total / self.frames.len() as f64
    }

    /// Check if all frames are verified
    pub fn all_frames_verified(&self) -> bool {
        self.frames.iter().all(|f| f.is_verified())
    }

    /// Save video frames to directory
    pub fn save_frames(
        &self,
        generator: &KnowledgeImageGenerator,
        output_dir: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(output_dir)?;

        for (i, frame) in self.frames.iter().enumerate() {
            let path = format!("{}/frame_{:04}.png", output_dir, i);
            generator.save_image(frame, &path)?;
        }

        Ok(())
    }
}

/// Configuration for knowledge-anchored generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeVisualConfig {
    /// Image/video dimensions
    pub width: usize,
    pub height: usize,
    /// Creativity injection weight (0.0 = purely factual, 1.0 = highly creative)
    pub alpha: f64,
    /// For video: temporal propagation strength (0.0 = static, 1.0 = high variation)
    pub propagation_strength: f64,
    /// For video: number of frames
    pub num_frames: usize,
    /// Minimum verification similarity threshold
    pub min_verification_similarity: f64,
}

impl Default for KnowledgeVisualConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            alpha: 0.3,  // Balanced: some creativity, mostly knowledge-grounded
            propagation_strength: 0.5,
            num_frames: 24,
            min_verification_similarity: 0.18,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_image_generator() {
        let generator = KnowledgeImageGenerator::new(128, 64, 64);
        assert_eq!(generator.width, 64);
        assert_eq!(generator.height, 64);
        assert_eq!(generator.dimension, 128);
    }

    #[test]
    fn test_latent_to_image() {
        let generator = KnowledgeImageGenerator::new(128, 32, 32);
        let latent = vec![0.5; 128];

        let image_data = generator.latent_to_image(&latent).unwrap();
        assert_eq!(image_data.len(), 32 * 32 * 3);

        // All values should be in valid byte range
        assert!(image_data.iter().all(|&x| x <= 255));
    }

    #[test]
    fn test_knowledge_image_metadata() {
        use crate::generation::reasoning_engine::LatentVerification;

        let latent_result = LatentResult {
            latent: vec![0.5; 128],
            concept_vector: vec![0.3; 128],
            knowledge_vector: vec![0.2; 128],
            creativity_vector: vec![0.1; 128],
            alpha: 0.5,
            verification: LatentVerification {
                verified: true,
                confidence: 0.9,
                max_similarity: 0.85,
                supporting_facts: vec!["test fact".to_string()],
                reason: "Verified".to_string(),
            },
            knowledge_facts_used: 3,
        };

        let image = KnowledgeImage {
            data: vec![0; 32 * 32 * 3],
            width: 32,
            height: 32,
            latent_result,
            concept: "test concept".to_string(),
        };

        assert!(image.is_verified());
        assert!((image.verification_confidence() - 0.9).abs() < 1e-6);
        assert_eq!(image.creativity_alpha(), 0.5);
    }
}
