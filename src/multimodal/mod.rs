//! ALEN Multimodal Learning Module
//!
//! Enables ALEN to learn from multiple modalities:
//! - Images: Feature extraction, visual embeddings, patch processing
//! - Video: Temporal analysis, frame sequences, motion understanding  
//! - Audio: Waveform analysis, spectrograms, frequency features
//! - Fusion: Cross-modal attention, unified representations

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::f64::consts::PI;

/// Modality types supported by ALEN
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
    Video,
    Audio,
    Multimodal,
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Modality::Text => write!(f, "text"),
            Modality::Image => write!(f, "image"),
            Modality::Video => write!(f, "video"),
            Modality::Audio => write!(f, "audio"),
            Modality::Multimodal => write!(f, "multimodal"),
        }
    }
}

// ============================================================================
// IMAGE PROCESSING
// ============================================================================

/// Image data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    /// Pixel values (normalized 0-1)
    pub pixels: Vec<f64>,
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// Channels (1=grayscale, 3=RGB, 4=RGBA)
    pub channels: usize,
}

impl ImageData {
    /// Create from raw bytes
    pub fn from_bytes(bytes: &[u8], width: usize, height: usize, channels: usize) -> Self {
        let pixels: Vec<f64> = bytes.iter().map(|&b| b as f64 / 255.0).collect();
        Self { pixels, width, height, channels }
    }

    /// Create from base64 encoded image
    pub fn from_base64(data: &str, width: usize, height: usize, channels: usize) -> Result<Self, String> {
        let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data)
            .map_err(|e| format!("Base64 decode error: {}", e))?;
        Ok(Self::from_bytes(&bytes, width, height, channels))
    }

    /// Create grayscale from 2D data
    pub fn from_grayscale_2d(data: &[Vec<f64>]) -> Self {
        let height = data.len();
        let width = if height > 0 { data[0].len() } else { 0 };
        let pixels: Vec<f64> = data.iter().flatten().cloned().collect();
        Self { pixels, width, height, channels: 1 }
    }

    /// Create random image for testing
    pub fn random(width: usize, height: usize, channels: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let pixels: Vec<f64> = (0..width * height * channels)
            .map(|_| rng.gen::<f64>())
            .collect();
        Self { pixels, width, height, channels }
    }

    /// Get pixel value at (x, y, channel)
    pub fn get_pixel(&self, x: usize, y: usize, c: usize) -> f64 {
        if x < self.width && y < self.height && c < self.channels {
            let idx = (y * self.width + x) * self.channels + c;
            self.pixels.get(idx).copied().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Set pixel value
    pub fn set_pixel(&mut self, x: usize, y: usize, c: usize, value: f64) {
        if x < self.width && y < self.height && c < self.channels {
            let idx = (y * self.width + x) * self.channels + c;
            if idx < self.pixels.len() {
                self.pixels[idx] = value.clamp(0.0, 1.0);
            }
        }
    }

    /// Convert to grayscale using luminance formula
    pub fn to_grayscale(&self) -> Self {
        if self.channels == 1 {
            return self.clone();
        }

        let mut gray_pixels = Vec::with_capacity(self.width * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let r = self.get_pixel(x, y, 0);
                let g = self.get_pixel(x, y, 1.min(self.channels - 1));
                let b = self.get_pixel(x, y, 2.min(self.channels - 1));
                // ITU-R BT.601 luma coefficients
                gray_pixels.push(0.299 * r + 0.587 * g + 0.114 * b);
            }
        }

        Self {
            pixels: gray_pixels,
            width: self.width,
            height: self.height,
            channels: 1,
        }
    }

    /// Resize using bilinear interpolation
    pub fn resize(&self, new_width: usize, new_height: usize) -> Self {
        if new_width == 0 || new_height == 0 {
            return Self { pixels: vec![], width: 0, height: 0, channels: self.channels };
        }

        let mut new_pixels = Vec::with_capacity(new_width * new_height * self.channels);
        
        for y in 0..new_height {
            for x in 0..new_width {
                for c in 0..self.channels {
                    let src_x = if new_width > 1 {
                        x as f64 * (self.width - 1) as f64 / (new_width - 1) as f64
                    } else {
                        0.0
                    };
                    let src_y = if new_height > 1 {
                        y as f64 * (self.height - 1) as f64 / (new_height - 1) as f64
                    } else {
                        0.0
                    };
                    
                    // Bilinear interpolation
                    let x0 = src_x.floor() as usize;
                    let y0 = src_y.floor() as usize;
                    let x1 = (x0 + 1).min(self.width - 1);
                    let y1 = (y0 + 1).min(self.height - 1);
                    
                    let dx = src_x - x0 as f64;
                    let dy = src_y - y0 as f64;
                    
                    let p00 = self.get_pixel(x0, y0, c);
                    let p10 = self.get_pixel(x1, y0, c);
                    let p01 = self.get_pixel(x0, y1, c);
                    let p11 = self.get_pixel(x1, y1, c);
                    
                    let val = (1.0 - dx) * (1.0 - dy) * p00
                            + dx * (1.0 - dy) * p10
                            + (1.0 - dx) * dy * p01
                            + dx * dy * p11;
                    
                    new_pixels.push(val);
                }
            }
        }

        Self {
            pixels: new_pixels,
            width: new_width,
            height: new_height,
            channels: self.channels,
        }
    }

    /// Extract patches for Vision Transformer style processing
    pub fn extract_patches(&self, patch_size: usize) -> Vec<Vec<f64>> {
        let mut patches = Vec::new();
        let gray = self.to_grayscale();
        
        for py in (0..gray.height).step_by(patch_size) {
            for px in (0..gray.width).step_by(patch_size) {
                let mut patch = Vec::with_capacity(patch_size * patch_size);
                for y in 0..patch_size {
                    for x in 0..patch_size {
                        patch.push(gray.get_pixel(px + x, py + y, 0));
                    }
                }
                patches.push(patch);
            }
        }
        patches
    }

    /// Compute histogram (grayscale, 256 bins)
    pub fn histogram(&self) -> Vec<f64> {
        let gray = self.to_grayscale();
        let mut hist = vec![0.0; 256];
        let total = gray.pixels.len() as f64;
        
        for &p in &gray.pixels {
            let bin = ((p * 255.0) as usize).min(255);
            hist[bin] += 1.0;
        }
        
        // Normalize
        for h in &mut hist {
            *h /= total;
        }
        hist
    }

    /// Apply convolution with a kernel
    pub fn convolve(&self, kernel: &DMatrix<f64>) -> Self {
        let gray = self.to_grayscale();
        let kh = kernel.nrows();
        let kw = kernel.ncols();
        let pad_h = kh / 2;
        let pad_w = kw / 2;
        
        let mut result = vec![0.0; gray.width * gray.height];
        
        for y in 0..gray.height {
            for x in 0..gray.width {
                let mut sum = 0.0;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let ix = (x + kx).saturating_sub(pad_w);
                        let iy = (y + ky).saturating_sub(pad_h);
                        let ix = ix.min(gray.width - 1);
                        let iy = iy.min(gray.height - 1);
                        sum += gray.get_pixel(ix, iy, 0) * kernel[(ky, kx)];
                    }
                }
                result[y * gray.width + x] = sum.clamp(0.0, 1.0);
            }
        }
        
        Self {
            pixels: result,
            width: gray.width,
            height: gray.height,
            channels: 1,
        }
    }
}

/// Image encoder - extracts embeddings from images
#[derive(Debug, Clone)]
pub struct ImageEncoder {
    /// Output embedding dimension
    pub embedding_dim: usize,
    /// Patch size for Vision Transformer style
    pub patch_size: usize,
    /// Standard processing size
    pub image_size: usize,
    /// Convolution filters for feature extraction
    pub filters: Vec<DMatrix<f64>>,
    /// Projection weights
    pub projection: DMatrix<f64>,
}

impl ImageEncoder {
    pub fn new(embedding_dim: usize) -> Self {
        let image_size = 64;
        let patch_size = 8;
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let patch_dim = patch_size * patch_size;
        
        // Initialize filters for edge detection and textures
        let filters = Self::create_filters();
        
        // Initialize projection matrix with Xavier initialization
        use rand::Rng;
        use rand_distr::{Normal, Distribution};
        let mut rng = rand::thread_rng();
        let std = (2.0 / (patch_dim + embedding_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        let projection = DMatrix::from_fn(embedding_dim, patch_dim * filters.len() + num_patches, |_, _| {
            normal.sample(&mut rng)
        });
        
        Self {
            embedding_dim,
            patch_size,
            image_size,
            filters,
            projection,
        }
    }

    fn create_filters() -> Vec<DMatrix<f64>> {
        vec![
            // Sobel horizontal
            DMatrix::from_row_slice(3, 3, &[
                -1.0, -2.0, -1.0,
                 0.0,  0.0,  0.0,
                 1.0,  2.0,  1.0,
            ]),
            // Sobel vertical
            DMatrix::from_row_slice(3, 3, &[
                -1.0, 0.0, 1.0,
                -2.0, 0.0, 2.0,
                -1.0, 0.0, 1.0,
            ]),
            // Laplacian
            DMatrix::from_row_slice(3, 3, &[
                 0.0, -1.0,  0.0,
                -1.0,  4.0, -1.0,
                 0.0, -1.0,  0.0,
            ]),
            // Gaussian blur
            DMatrix::from_row_slice(3, 3, &[
                1.0/16.0, 2.0/16.0, 1.0/16.0,
                2.0/16.0, 4.0/16.0, 2.0/16.0,
                1.0/16.0, 2.0/16.0, 1.0/16.0,
            ]),
        ]
    }

    /// Encode an image to embedding vector
    pub fn encode(&self, image: &ImageData) -> DVector<f64> {
        // Resize to standard size
        let resized = image.resize(self.image_size, self.image_size);
        
        // Extract features using convolution filters
        let mut features = Vec::new();
        for filter in &self.filters {
            let filtered = resized.convolve(filter);
            // Global average pooling
            let avg: f64 = filtered.pixels.iter().sum::<f64>() / filtered.pixels.len() as f64;
            let max = filtered.pixels.iter().cloned().fold(0.0_f64, f64::max);
            let variance = filtered.pixels.iter()
                .map(|x| (x - avg).powi(2))
                .sum::<f64>() / filtered.pixels.len() as f64;
            features.push(avg);
            features.push(max);
            features.push(variance.sqrt());
        }
        
        // Extract patches and flatten
        let patches = resized.extract_patches(self.patch_size);
        for patch in &patches {
            let patch_mean: f64 = patch.iter().sum::<f64>() / patch.len() as f64;
            features.push(patch_mean);
        }
        
        // Pad or truncate to projection input size
        let input_size = self.projection.ncols();
        features.resize(input_size, 0.0);
        
        // Project to embedding dimension
        let input = DVector::from_vec(features);
        let embedding = &self.projection * &input;
        
        // Normalize
        let norm = embedding.norm();
        if norm > 1e-10 {
            embedding / norm
        } else {
            embedding
        }
    }

    /// Encode batch of images
    pub fn encode_batch(&self, images: &[ImageData]) -> Vec<DVector<f64>> {
        images.iter().map(|img| self.encode(img)).collect()
    }
}

// ============================================================================
// VIDEO PROCESSING
// ============================================================================

/// Video data as sequence of frames
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoData {
    /// Frames of the video
    pub frames: Vec<ImageData>,
    /// Frames per second
    pub fps: f64,
    /// Duration in seconds
    pub duration: f64,
}

impl VideoData {
    pub fn new(frames: Vec<ImageData>, fps: f64) -> Self {
        let duration = frames.len() as f64 / fps;
        Self { frames, fps, duration }
    }

    /// Create from frame data
    pub fn from_frames(frames: Vec<ImageData>) -> Self {
        Self::new(frames, 30.0) // Default 30 fps
    }

    /// Get frame at specific time
    pub fn get_frame_at(&self, time: f64) -> Option<&ImageData> {
        let frame_idx = (time * self.fps).floor() as usize;
        self.frames.get(frame_idx)
    }

    /// Sample frames uniformly
    pub fn sample_frames(&self, num_frames: usize) -> Vec<&ImageData> {
        if self.frames.is_empty() || num_frames == 0 {
            return vec![];
        }
        
        let step = self.frames.len() as f64 / num_frames as f64;
        (0..num_frames)
            .map(|i| {
                let idx = ((i as f64 * step).floor() as usize).min(self.frames.len() - 1);
                &self.frames[idx]
            })
            .collect()
    }

    /// Compute optical flow approximation between consecutive frames
    pub fn compute_motion(&self) -> Vec<f64> {
        if self.frames.len() < 2 {
            return vec![];
        }
        
        let mut motion = Vec::new();
        for i in 1..self.frames.len() {
            let prev = &self.frames[i - 1].to_grayscale();
            let curr = &self.frames[i].to_grayscale();
            
            // Simple motion: sum of absolute differences
            let diff: f64 = prev.pixels.iter()
                .zip(curr.pixels.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>() / prev.pixels.len() as f64;
            
            motion.push(diff);
        }
        motion
    }
}

/// Video encoder with temporal understanding
#[derive(Debug, Clone)]
pub struct VideoEncoder {
    /// Image encoder for individual frames
    pub image_encoder: ImageEncoder,
    /// Number of frames to sample
    pub num_sample_frames: usize,
    /// Temporal attention weights
    pub temporal_weights: DMatrix<f64>,
    /// Output embedding dimension
    pub embedding_dim: usize,
}

impl VideoEncoder {
    pub fn new(embedding_dim: usize) -> Self {
        let num_sample_frames = 8;
        let image_encoder = ImageEncoder::new(embedding_dim);
        
        // Initialize temporal attention matrix
        use rand_distr::{Normal, Distribution};
        let mut rng = rand::thread_rng();
        let std = (1.0 / num_sample_frames as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        let temporal_weights = DMatrix::from_fn(embedding_dim, embedding_dim * num_sample_frames, |_, _| {
            normal.sample(&mut rng)
        });
        
        Self {
            image_encoder,
            num_sample_frames,
            temporal_weights,
            embedding_dim,
        }
    }

    /// Encode video to single embedding
    pub fn encode(&self, video: &VideoData) -> DVector<f64> {
        // Sample frames
        let sampled = video.sample_frames(self.num_sample_frames);
        
        // Encode each frame
        let frame_embeddings: Vec<DVector<f64>> = sampled
            .iter()
            .map(|frame| self.image_encoder.encode(frame))
            .collect();
        
        // Handle empty case
        if frame_embeddings.is_empty() {
            return DVector::zeros(self.embedding_dim);
        }
        
        // Compute motion features
        let motion = video.compute_motion();
        let motion_avg = if motion.is_empty() { 0.0 } else {
            motion.iter().sum::<f64>() / motion.len() as f64
        };
        
        // Concatenate frame embeddings with motion
        let mut concat = Vec::new();
        for emb in &frame_embeddings {
            concat.extend(emb.iter());
        }
        
        // Pad to expected size
        let expected_size = self.temporal_weights.ncols();
        concat.resize(expected_size, motion_avg);
        
        // Apply temporal attention
        let input = DVector::from_vec(concat);
        let embedding = &self.temporal_weights * &input;
        
        // Normalize
        let norm = embedding.norm();
        if norm > 1e-10 {
            embedding / norm
        } else {
            embedding
        }
    }

    /// Encode video and return per-frame embeddings + aggregated
    pub fn encode_with_frames(&self, video: &VideoData) -> (DVector<f64>, Vec<DVector<f64>>) {
        let sampled = video.sample_frames(self.num_sample_frames);
        let frame_embeddings: Vec<DVector<f64>> = sampled
            .iter()
            .map(|frame| self.image_encoder.encode(frame))
            .collect();
        
        let aggregated = self.encode(video);
        (aggregated, frame_embeddings)
    }
}

// ============================================================================
// AUDIO PROCESSING
// ============================================================================

/// Audio data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioData {
    /// Samples (normalized -1 to 1)
    pub samples: Vec<f64>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: usize,
    /// Duration in seconds
    pub duration: f64,
}

impl AudioData {
    pub fn new(samples: Vec<f64>, sample_rate: u32, channels: usize) -> Self {
        let duration = samples.len() as f64 / (sample_rate as f64 * channels as f64);
        Self { samples, sample_rate, channels, duration }
    }

    /// Create from mono samples
    pub fn from_mono(samples: Vec<f64>, sample_rate: u32) -> Self {
        Self::new(samples, sample_rate, 1)
    }

    /// Convert to mono by averaging channels
    pub fn to_mono(&self) -> Vec<f64> {
        if self.channels == 1 {
            return self.samples.clone();
        }
        
        let num_samples = self.samples.len() / self.channels;
        let mut mono = Vec::with_capacity(num_samples);
        
        for i in 0..num_samples {
            let mut sum = 0.0;
            for c in 0..self.channels {
                sum += self.samples[i * self.channels + c];
            }
            mono.push(sum / self.channels as f64);
        }
        mono
    }

    /// Compute spectrogram using Short-Time Fourier Transform
    pub fn spectrogram(&self, window_size: usize, hop_size: usize) -> Vec<Vec<f64>> {
        let mono = self.to_mono();
        let mut spec = Vec::new();
        
        let mut i = 0;
        while i + window_size <= mono.len() {
            let window: Vec<f64> = mono[i..i + window_size]
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    // Apply Hann window
                    let hann = 0.5 * (1.0 - (2.0 * PI * j as f64 / (window_size - 1) as f64).cos());
                    x * hann
                })
                .collect();
            
            // Compute magnitude spectrum using DFT
            let spectrum = Self::compute_magnitude_spectrum(&window);
            spec.push(spectrum);
            
            i += hop_size;
        }
        spec
    }

    /// Compute magnitude spectrum using DFT
    fn compute_magnitude_spectrum(window: &[f64]) -> Vec<f64> {
        let n = window.len();
        let mut magnitudes = Vec::with_capacity(n / 2 + 1);
        
        for k in 0..=n/2 {
            let mut real = 0.0;
            let mut imag = 0.0;
            
            for (t, &x) in window.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * t as f64 / n as f64;
                real += x * angle.cos();
                imag += x * angle.sin();
            }
            
            magnitudes.push((real * real + imag * imag).sqrt());
        }
        magnitudes
    }

    /// Compute Mel-frequency cepstral coefficients
    pub fn mfcc(&self, num_coeffs: usize) -> Vec<f64> {
        let window_size = 1024;
        let hop_size = 512;
        let spec = self.spectrogram(window_size, hop_size);
        
        if spec.is_empty() {
            return vec![0.0; num_coeffs];
        }
        
        // Average spectrogram over time
        let avg_spec: Vec<f64> = (0..spec[0].len())
            .map(|i| {
                spec.iter().map(|s| s.get(i).copied().unwrap_or(0.0)).sum::<f64>() / spec.len() as f64
            })
            .collect();
        
        // Apply log and DCT to get MFCCs
        let log_spec: Vec<f64> = avg_spec.iter()
            .map(|&x| (x + 1e-10).ln())
            .collect();
        
        // Simplified DCT
        let mut mfccs = Vec::with_capacity(num_coeffs);
        for k in 0..num_coeffs {
            let mut coeff = 0.0;
            for (n, &x) in log_spec.iter().enumerate() {
                coeff += x * (PI * k as f64 * (n as f64 + 0.5) / log_spec.len() as f64).cos();
            }
            mfccs.push(coeff);
        }
        mfccs
    }

    /// Compute zero-crossing rate
    pub fn zero_crossing_rate(&self) -> f64 {
        let mono = self.to_mono();
        if mono.len() < 2 {
            return 0.0;
        }
        
        let crossings: usize = mono.windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        
        crossings as f64 / (mono.len() - 1) as f64
    }

    /// Compute RMS energy
    pub fn rms_energy(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        
        let sum_sq: f64 = self.samples.iter().map(|x| x * x).sum();
        (sum_sq / self.samples.len() as f64).sqrt()
    }
}

/// Audio encoder
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    /// Output embedding dimension
    pub embedding_dim: usize,
    /// Number of MFCC coefficients
    pub num_mfcc: usize,
    /// Projection matrix
    pub projection: DMatrix<f64>,
}

impl AudioEncoder {
    pub fn new(embedding_dim: usize) -> Self {
        let num_mfcc = 13;
        let feature_dim = num_mfcc + 3; // MFCCs + ZCR + RMS + duration
        
        use rand_distr::{Normal, Distribution};
        let mut rng = rand::thread_rng();
        let std = (2.0 / (feature_dim + embedding_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        let projection = DMatrix::from_fn(embedding_dim, feature_dim, |_, _| {
            normal.sample(&mut rng)
        });
        
        Self {
            embedding_dim,
            num_mfcc,
            projection,
        }
    }

    /// Encode audio to embedding vector
    pub fn encode(&self, audio: &AudioData) -> DVector<f64> {
        // Extract features
        let mut features = audio.mfcc(self.num_mfcc);
        features.push(audio.zero_crossing_rate());
        features.push(audio.rms_energy());
        features.push(audio.duration.min(60.0) / 60.0); // Normalized duration
        
        // Project to embedding dimension
        let input = DVector::from_vec(features);
        let embedding = &self.projection * &input;
        
        // Normalize
        let norm = embedding.norm();
        if norm > 1e-10 {
            embedding / norm
        } else {
            embedding
        }
    }
}

// ============================================================================
// MULTIMODAL FUSION
// ============================================================================

/// Multimodal input container
#[derive(Debug, Clone)]
pub struct MultimodalInput {
    pub text: Option<String>,
    pub image: Option<ImageData>,
    pub video: Option<VideoData>,
    pub audio: Option<AudioData>,
}

impl MultimodalInput {
    pub fn text_only(text: &str) -> Self {
        Self {
            text: Some(text.to_string()),
            image: None,
            video: None,
            audio: None,
        }
    }

    pub fn image_only(image: ImageData) -> Self {
        Self {
            text: None,
            image: Some(image),
            video: None,
            audio: None,
        }
    }

    pub fn text_and_image(text: &str, image: ImageData) -> Self {
        Self {
            text: Some(text.to_string()),
            image: Some(image),
            video: None,
            audio: None,
        }
    }

    pub fn modalities(&self) -> Vec<Modality> {
        let mut m = Vec::new();
        if self.text.is_some() { m.push(Modality::Text); }
        if self.image.is_some() { m.push(Modality::Image); }
        if self.video.is_some() { m.push(Modality::Video); }
        if self.audio.is_some() { m.push(Modality::Audio); }
        m
    }
}

/// Cross-modal attention for fusing embeddings
#[derive(Debug, Clone)]
pub struct CrossModalAttention {
    /// Query projection
    pub w_q: DMatrix<f64>,
    /// Key projection  
    pub w_k: DMatrix<f64>,
    /// Value projection
    pub w_v: DMatrix<f64>,
    /// Output projection
    pub w_o: DMatrix<f64>,
    /// Dimension
    pub d_model: usize,
}

impl CrossModalAttention {
    pub fn new(d_model: usize) -> Self {
        use rand_distr::{Normal, Distribution};
        let mut rng = rand::thread_rng();
        let std = (1.0 / d_model as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        let mut init_matrix = || {
            DMatrix::from_fn(d_model, d_model, |_, _| normal.sample(&mut rng))
        };
        
        Self {
            w_q: init_matrix(),
            w_k: init_matrix(),
            w_v: init_matrix(),
            w_o: init_matrix(),
            d_model,
        }
    }

    /// Attend from query modality to key/value modality
    pub fn attend(&self, query: &DVector<f64>, key: &DVector<f64>, value: &DVector<f64>) -> DVector<f64> {
        let q = &self.w_q * query;
        let k = &self.w_k * key;
        let v = &self.w_v * value;
        
        // Scaled dot-product attention (single query-key pair)
        let score = q.dot(&k) / (self.d_model as f64).sqrt();
        let attention = (score.exp()).min(1e10); // Softmax degenerates to scalar
        
        let attended = attention * &v;
        &self.w_o * &attended
    }

    /// Fuse multiple embeddings using cross-attention
    pub fn fuse(&self, embeddings: &[DVector<f64>]) -> DVector<f64> {
        if embeddings.is_empty() {
            return DVector::zeros(self.d_model);
        }
        
        if embeddings.len() == 1 {
            return embeddings[0].clone();
        }
        
        // Use first as query, attend to all others
        let query = &embeddings[0];
        let mut fused = query.clone();
        
        for other in &embeddings[1..] {
            let attended = self.attend(query, other, other);
            fused += &attended;
        }
        
        // Normalize
        let norm = fused.norm();
        if norm > 1e-10 {
            fused / norm
        } else {
            fused
        }
    }
}

/// Complete multimodal encoder
#[derive(Debug, Clone)]
pub struct MultimodalEncoder {
    pub embedding_dim: usize,
    pub text_encoder: crate::memory::EmbeddingEngine,
    pub image_encoder: ImageEncoder,
    pub video_encoder: VideoEncoder,
    pub audio_encoder: AudioEncoder,
    pub fusion: CrossModalAttention,
}

impl MultimodalEncoder {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            text_encoder: crate::memory::EmbeddingEngine::new(crate::memory::EmbeddingConfig {
                dimension: embedding_dim,
                normalize: true,
                vocab_size: 10000,
            }),
            image_encoder: ImageEncoder::new(embedding_dim),
            video_encoder: VideoEncoder::new(embedding_dim),
            audio_encoder: AudioEncoder::new(embedding_dim),
            fusion: CrossModalAttention::new(embedding_dim),
        }
    }

    /// Encode multimodal input to unified embedding
    pub fn encode(&mut self, input: &MultimodalInput) -> DVector<f64> {
        let mut embeddings = Vec::new();
        
        if let Some(ref text) = input.text {
            let text_state = self.text_encoder.embed_text(text);
            embeddings.push(DVector::from_vec(text_state.vector));
        }
        
        if let Some(ref image) = input.image {
            embeddings.push(self.image_encoder.encode(image));
        }
        
        if let Some(ref video) = input.video {
            embeddings.push(self.video_encoder.encode(video));
        }
        
        if let Some(ref audio) = input.audio {
            embeddings.push(self.audio_encoder.encode(audio));
        }
        
        if embeddings.is_empty() {
            return DVector::zeros(self.embedding_dim);
        }
        
        self.fusion.fuse(&embeddings)
    }

    /// Get individual modality embeddings
    pub fn encode_separate(&mut self, input: &MultimodalInput) -> std::collections::HashMap<Modality, DVector<f64>> {
        let mut result = std::collections::HashMap::new();
        
        if let Some(ref text) = input.text {
            let text_state = self.text_encoder.embed_text(text);
            result.insert(Modality::Text, DVector::from_vec(text_state.vector));
        }
        
        if let Some(ref image) = input.image {
            result.insert(Modality::Image, self.image_encoder.encode(image));
        }
        
        if let Some(ref video) = input.video {
            result.insert(Modality::Video, self.video_encoder.encode(video));
        }
        
        if let Some(ref audio) = input.audio {
            result.insert(Modality::Audio, self.audio_encoder.encode(audio));
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_data() {
        let img = ImageData::random(64, 64, 3);
        assert_eq!(img.width, 64);
        assert_eq!(img.height, 64);
        assert_eq!(img.channels, 3);
    }

    #[test]
    fn test_image_resize() {
        let img = ImageData::random(100, 100, 1);
        let resized = img.resize(50, 50);
        assert_eq!(resized.width, 50);
        assert_eq!(resized.height, 50);
    }

    #[test]
    fn test_image_encoder() {
        let encoder = ImageEncoder::new(128);
        let img = ImageData::random(64, 64, 3);
        let embedding = encoder.encode(&img);
        assert_eq!(embedding.len(), 128);
    }

    #[test]
    fn test_video_encoder() {
        let encoder = VideoEncoder::new(128);
        let frames: Vec<ImageData> = (0..10)
            .map(|_| ImageData::random(32, 32, 3))
            .collect();
        let video = VideoData::from_frames(frames);
        let embedding = encoder.encode(&video);
        assert_eq!(embedding.len(), 128);
    }

    #[test]
    fn test_audio_encoder() {
        let encoder = AudioEncoder::new(128);
        let samples: Vec<f64> = (0..44100)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let audio = AudioData::from_mono(samples, 44100);
        let embedding = encoder.encode(&audio);
        assert_eq!(embedding.len(), 128);
    }

    #[test]
    fn test_multimodal_fusion() {
        let mut encoder = MultimodalEncoder::new(128);
        let input = MultimodalInput {
            text: Some("Hello world".to_string()),
            image: Some(ImageData::random(32, 32, 3)),
            video: None,
            audio: None,
        };
        let embedding = encoder.encode(&input);
        assert_eq!(embedding.len(), 128);
    }

    #[test]
    fn test_cross_modal_attention() {
        let attention = CrossModalAttention::new(64);
        let e1 = DVector::from_element(64, 1.0);
        let e2 = DVector::from_element(64, 0.5);
        let fused = attention.fuse(&[e1, e2]);
        assert_eq!(fused.len(), 64);
    }
}
