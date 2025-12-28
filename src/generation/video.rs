//! Video Generation Module
//!
//! Generate videos from thought states using temporal coherence and motion modeling.

use crate::core::ThoughtState;
use crate::multimodal::ImageData;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use serde::{Deserialize, Serialize};

/// Video generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoGenConfig {
    /// Frames per second
    pub fps: f64,
    /// Total duration in seconds
    pub duration: f64,
    /// Frame width
    pub width: usize,
    /// Frame height
    pub height: usize,
    /// Number of channels (3 for RGB)
    pub channels: usize,
    /// Temporal coherence strength (0-1, higher = smoother motion)
    pub temporal_coherence: f64,
    /// Motion amplitude (0-1)
    pub motion_amplitude: f64,
}

impl Default for VideoGenConfig {
    fn default() -> Self {
        Self {
            fps: 30.0,
            duration: 2.0,
            width: 64,
            height: 64,
            channels: 3,
            temporal_coherence: 0.8,
            motion_amplitude: 0.3,
        }
    }
}

/// Generated video data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedVideo {
    /// Video frames as image data
    pub frames: Vec<ImageData>,
    /// Frames per second
    pub fps: f64,
    /// Total duration
    pub duration: f64,
    /// Generation confidence
    pub confidence: f64,
}

/// Video generator using thought states
pub struct VideoGenerator {
    pub config: VideoGenConfig,
    pub dim: usize,
    /// Frame generation network layers
    pub frame_generators: Vec<DMatrix<f64>>,
    /// Temporal transition matrix
    pub temporal_transition: DMatrix<f64>,
    /// Motion field generator
    pub motion_generator: DMatrix<f64>,
}

impl VideoGenerator {
    pub fn new(config: VideoGenConfig, dim: usize) -> Self {
        use rand_distr::{Normal, Distribution};
        let mut rng = rand::thread_rng();

        // Create progressive upsampling layers for frame generation
        let frame_size = config.width * config.height * config.channels;
        let mut frame_generators = Vec::new();

        let layer_sizes = vec![dim, dim * 2, dim * 4, frame_size / 4, frame_size];

        for i in 0..layer_sizes.len() - 1 {
            let std = (2.0 / (layer_sizes[i] + layer_sizes[i + 1]) as f64).sqrt();
            let normal = Normal::new(0.0, std).unwrap();

            let matrix = DMatrix::from_fn(layer_sizes[i + 1], layer_sizes[i], |_, _| {
                normal.sample(&mut rng)
            });

            frame_generators.push(matrix);
        }

        // Temporal transition matrix for smooth frame progression
        let std = (1.0 / dim as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let temporal_transition = DMatrix::from_fn(dim, dim, |_, _| {
            normal.sample(&mut rng)
        });

        // Motion field generator
        let motion_dim = 32; // Compressed motion representation
        let motion_generator = DMatrix::from_fn(motion_dim, dim, |_, _| {
            normal.sample(&mut rng)
        });

        Self {
            config,
            dim,
            frame_generators,
            temporal_transition,
            motion_generator,
        }
    }

    /// Generate a complete video from a thought state
    pub fn generate(&self, thought: &ThoughtState) -> GeneratedVideo {
        let num_frames = (self.config.fps * self.config.duration) as usize;
        let mut frames = Vec::with_capacity(num_frames);

        // Generate motion trajectory from thought
        let motion_field = self.generate_motion_field(thought);

        // Initial frame state
        let mut current_state = DVector::from_column_slice(&thought.vector);

        for frame_idx in 0..num_frames {
            // Apply temporal evolution
            let t = frame_idx as f64 / num_frames as f64;
            current_state = self.evolve_state(&current_state, t, &motion_field);

            // Generate frame from current state
            let frame = self.generate_frame(&current_state);
            frames.push(frame);
        }

        GeneratedVideo {
            frames,
            fps: self.config.fps,
            duration: self.config.duration,
            confidence: thought.confidence,
        }
    }

    /// Generate motion field from thought
    fn generate_motion_field(&self, thought: &ThoughtState) -> DVector<f64> {
        let thought_vec = DVector::from_column_slice(&thought.vector);
        let motion = &self.motion_generator * &thought_vec;

        // Normalize
        let norm = motion.norm();
        if norm > 1e-10 {
            motion / norm
        } else {
            motion
        }
    }

    /// Evolve state over time with motion
    fn evolve_state(&self, state: &DVector<f64>, t: f64, motion: &DVector<f64>) -> DVector<f64> {
        // Apply temporal transition
        let mut next_state = &self.temporal_transition * state;

        // Add motion influence (using sinusoidal motion for smooth animation)
        let motion_phase = 2.0 * std::f64::consts::PI * t;
        let motion_scale = self.config.motion_amplitude * motion_phase.sin();

        // Blend motion into state (resize motion to match state dimension)
        for i in 0..next_state.len() {
            let motion_idx = i % motion.len();
            next_state[i] += motion_scale * motion[motion_idx];
        }

        // Apply temporal coherence (blend with previous state)
        let coherence = self.config.temporal_coherence;
        next_state = state.clone() * coherence + next_state * (1.0 - coherence);

        // Normalize to prevent drift
        let norm = next_state.norm();
        if norm > 1e-10 {
            next_state / norm
        } else {
            next_state
        }
    }

    /// Generate single frame from state
    fn generate_frame(&self, state: &DVector<f64>) -> ImageData {
        let mut current = state.clone();

        // Pass through frame generation layers with activations
        for (idx, generator) in self.frame_generators.iter().enumerate() {
            current = generator * &current;

            // Apply ReLU activation except on final layer
            if idx < self.frame_generators.len() - 1 {
                current = current.map(|x| x.max(0.0));
            }
        }

        // Normalize to [0, 1] range
        let min_val = current.min();
        let max_val = current.max();
        let range = (max_val - min_val).max(1e-10);

        let pixels: Vec<f64> = current.iter()
            .map(|&x| ((x - min_val) / range).clamp(0.0, 1.0))
            .collect();

        ImageData {
            pixels,
            width: self.config.width,
            height: self.config.height,
            channels: self.config.channels,
        }
    }

    /// Generate video with specific motion pattern
    pub fn generate_with_motion(&self, thought: &ThoughtState, motion_type: MotionType) -> GeneratedVideo {
        let num_frames = (self.config.fps * self.config.duration) as usize;
        let mut frames = Vec::with_capacity(num_frames);

        let mut current_state = DVector::from_column_slice(&thought.vector);

        for frame_idx in 0..num_frames {
            let t = frame_idx as f64 / num_frames as f64;

            // Apply motion pattern
            current_state = match motion_type {
                MotionType::Linear => self.apply_linear_motion(&current_state, t),
                MotionType::Circular => self.apply_circular_motion(&current_state, t),
                MotionType::Oscillating => self.apply_oscillating_motion(&current_state, t),
                MotionType::Expanding => self.apply_expanding_motion(&current_state, t),
                MotionType::Random => self.apply_random_motion(&current_state, t),
            };

            let frame = self.generate_frame(&current_state);
            frames.push(frame);
        }

        GeneratedVideo {
            frames,
            fps: self.config.fps,
            duration: self.config.duration,
            confidence: thought.confidence,
        }
    }

    fn apply_linear_motion(&self, state: &DVector<f64>, t: f64) -> DVector<f64> {
        let transition = &self.temporal_transition * state;
        state.clone() * (1.0 - t * 0.5) + transition * (t * 0.5)
    }

    fn apply_circular_motion(&self, state: &DVector<f64>, t: f64) -> DVector<f64> {
        let angle = 2.0 * std::f64::consts::PI * t;
        let cos_component = state.clone() * angle.cos();
        let sin_component = (&self.temporal_transition * state) * angle.sin();
        (cos_component + sin_component) * 0.7 + state.clone() * 0.3
    }

    fn apply_oscillating_motion(&self, state: &DVector<f64>, t: f64) -> DVector<f64> {
        let phase = (2.0 * std::f64::consts::PI * t * 2.0).sin();
        let oscillation = &self.temporal_transition * state * phase * self.config.motion_amplitude;
        state.clone() + oscillation
    }

    fn apply_expanding_motion(&self, state: &DVector<f64>, t: f64) -> DVector<f64> {
        let scale = 1.0 + t * self.config.motion_amplitude;
        state.clone() * scale
    }

    fn apply_random_motion(&self, state: &DVector<f64>, t: f64) -> DVector<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1 * self.config.motion_amplitude).unwrap();

        let noise: DVector<f64> = DVector::from_fn(state.len(), |_, _| {
            normal.sample(&mut rng)
        });

        state.clone() * 0.9 + noise * 0.1
    }

    /// Generate interpolation video between two thoughts
    pub fn generate_interpolation(&self, thought_a: &ThoughtState, thought_b: &ThoughtState) -> GeneratedVideo {
        let num_frames = (self.config.fps * self.config.duration) as usize;
        let mut frames = Vec::with_capacity(num_frames);

        let state_a = DVector::from_column_slice(&thought_a.vector);
        let state_b = DVector::from_column_slice(&thought_b.vector);

        for frame_idx in 0..num_frames {
            let t = frame_idx as f64 / (num_frames - 1).max(1) as f64;

            // Spherical linear interpolation (slerp) for smooth transition
            let current_state = self.slerp(&state_a, &state_b, t);
            let frame = self.generate_frame(&current_state);
            frames.push(frame);
        }

        let avg_confidence = (thought_a.confidence + thought_b.confidence) / 2.0;

        GeneratedVideo {
            frames,
            fps: self.config.fps,
            duration: self.config.duration,
            confidence: avg_confidence,
        }
    }

    /// Spherical linear interpolation between two vectors
    fn slerp(&self, a: &DVector<f64>, b: &DVector<f64>, t: f64) -> DVector<f64> {
        let dot = a.dot(b);
        let theta = dot.acos();

        if theta.abs() < 1e-6 {
            // Vectors are nearly parallel, use linear interpolation
            return a.clone() * (1.0 - t) + b.clone() * t;
        }

        let sin_theta = theta.sin();
        let weight_a = ((1.0 - t) * theta).sin() / sin_theta;
        let weight_b = (t * theta).sin() / sin_theta;

        a.clone() * weight_a + b.clone() * weight_b
    }
}

/// Motion pattern types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MotionType {
    Linear,
    Circular,
    Oscillating,
    Expanding,
    Random,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_generation() {
        let config = VideoGenConfig::default();
        let generator = VideoGenerator::new(config, 64);
        let thought = ThoughtState::random(64);

        let video = generator.generate(&thought);

        assert_eq!(video.frames.len(), 60); // 2 seconds at 30 fps
        assert_eq!(video.fps, 30.0);
        assert!((video.duration - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_motion_patterns() {
        let config = VideoGenConfig {
            duration: 1.0,
            ..Default::default()
        };
        let generator = VideoGenerator::new(config, 64);
        let thought = ThoughtState::random(64);

        for motion in &[
            MotionType::Linear,
            MotionType::Circular,
            MotionType::Oscillating,
            MotionType::Expanding,
            MotionType::Random,
        ] {
            let video = generator.generate_with_motion(&thought, *motion);
            assert_eq!(video.frames.len(), 30);
        }
    }

    #[test]
    fn test_interpolation() {
        let config = VideoGenConfig::default();
        let generator = VideoGenerator::new(config, 64);
        let thought_a = ThoughtState::random(64);
        let thought_b = ThoughtState::random(64);

        let video = generator.generate_interpolation(&thought_a, &thought_b);

        assert_eq!(video.frames.len(), 60);
        // First and last frames should be different
        assert_ne!(video.frames[0].pixels, video.frames[59].pixels);
    }

    #[test]
    fn test_frame_dimensions() {
        let config = VideoGenConfig {
            width: 32,
            height: 32,
            channels: 3,
            ..Default::default()
        };
        let generator = VideoGenerator::new(config, 64);
        let thought = ThoughtState::random(64);

        let video = generator.generate(&thought);

        for frame in &video.frames {
            assert_eq!(frame.width, 32);
            assert_eq!(frame.height, 32);
            assert_eq!(frame.channels, 3);
            assert_eq!(frame.pixels.len(), 32 * 32 * 3);
        }
    }
}
