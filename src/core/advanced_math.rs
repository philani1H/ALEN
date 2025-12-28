//! ALEN Advanced Mathematics Module
//!
//! Sophisticated mathematical operations for AI reasoning:
//! - Neural Network Layers (Dense, Residual)
//! - Attention Mechanisms (Self, Multi-Head)
//! - Transformer Components (Encoder layers)
//! - Activation Functions (ReLU, GELU, Swish, etc.)
//! - Optimization (Adam, Learning Rate Schedules)
//! - Information Theory (Entropy, KL Divergence)

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Activation {
    ReLU,
    LeakyReLU(f64),
    Sigmoid,
    Tanh,
    GELU,
    Swish,
    Softmax,
    Softplus,
}

impl Default for Activation {
    fn default() -> Self { Activation::ReLU }
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU(a) => if x > 0.0 { x } else { a * x },
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::GELU => 0.5 * x * (1.0 + ((2.0/PI).sqrt() * (x + 0.044715*x.powi(3))).tanh()),
            Activation::Swish => x / (1.0 + (-x).exp()),
            Activation::Softmax => x,
            Activation::Softplus => (1.0 + x.exp()).ln(),
        }
    }

    pub fn apply_vector(&self, v: &[f64]) -> Vec<f64> {
        match self {
            Activation::Softmax => {
                let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();
                let sum: f64 = exp.iter().sum();
                exp.iter().map(|x| x / sum.max(1e-10)).collect()
            },
            _ => v.iter().map(|x| self.apply(*x)).collect(),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::LeakyReLU(a) => if x > 0.0 { 1.0 } else { *a },
            Activation::Sigmoid => { let s = self.apply(x); s * (1.0 - s) },
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::GELU => {
                let c = (2.0/PI).sqrt();
                let inner = c * (x + 0.044715 * x.powi(3));
                0.5 * (1.0 + inner.tanh()) + 0.5 * x * (1.0 - inner.tanh().powi(2)) * c * (1.0 + 3.0*0.044715*x.powi(2))
            },
            Activation::Swish => { let s = 1.0/(1.0+(-x).exp()); s + x*s*(1.0-s) },
            _ => 1.0,
        }
    }
}

// ============================================================================
// NEURAL NETWORK LAYERS
// ============================================================================

#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: DMatrix<f64>,
    pub biases: DVector<f64>,
    pub activation: Activation,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl DenseLayer {
    pub fn new(input_dim: usize, output_dim: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        Self {
            weights: DMatrix::from_fn(output_dim, input_dim, |_,_| normal.sample(&mut rng)),
            biases: DVector::zeros(output_dim),
            activation, input_dim, output_dim,
        }
    }

    pub fn forward(&self, input: &DVector<f64>) -> DVector<f64> {
        let linear = &self.weights * input + &self.biases;
        DVector::from_iterator(self.output_dim, linear.iter().map(|x| self.activation.apply(*x)))
    }
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub gamma: DVector<f64>,
    pub beta: DVector<f64>,
    pub eps: f64,
    pub dim: usize,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        Self { gamma: DVector::from_element(dim, 1.0), beta: DVector::zeros(dim), eps: 1e-5, dim }
    }

    pub fn forward(&self, x: &DVector<f64>) -> DVector<f64> {
        let mean = x.mean();
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
        let std = (var + self.eps).sqrt();
        DVector::from_iterator(self.dim, x.iter().enumerate().map(|(i,v)| self.gamma[i]*(v-mean)/std + self.beta[i]))
    }
}

// ============================================================================
// ATTENTION MECHANISMS
// ============================================================================

#[derive(Debug, Clone)]
pub struct AttentionHead {
    pub w_q: DMatrix<f64>,
    pub w_k: DMatrix<f64>,
    pub w_v: DMatrix<f64>,
    pub d_k: usize,
}

impl AttentionHead {
    pub fn new(d_model: usize, d_k: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (1.0 / d_model as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let mut init = |r,c| DMatrix::from_fn(r, c, |_,_| normal.sample(&mut rng));
        Self { w_q: init(d_k, d_model), w_k: init(d_k, d_model), w_v: init(d_k, d_model), d_k }
    }

    pub fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let q = &self.w_q * x;
        let k = &self.w_k * x;
        let v = &self.w_v * x;
        let scores = q.transpose() * &k / (self.d_k as f64).sqrt();
        let attn = Self::softmax_rows(&scores);
        v * &attn.transpose()
    }

    fn softmax_rows(m: &DMatrix<f64>) -> DMatrix<f64> {
        let mut r = m.clone();
        for mut row in r.row_iter_mut() {
            let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = row.iter().map(|x| if x.is_finite() {(x-max).exp()} else {0.0}).collect();
            let sum = exp.iter().sum::<f64>().max(1e-10);
            for (i, val) in row.iter_mut().enumerate() { *val = exp[i] / sum; }
        }
        r
    }
}

#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub heads: Vec<AttentionHead>,
    pub w_o: DMatrix<f64>,
    pub num_heads: usize,
    pub d_model: usize,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_k = d_model / num_heads;
        let heads: Vec<_> = (0..num_heads).map(|_| AttentionHead::new(d_model, d_k)).collect();
        let mut rng = rand::thread_rng();
        let std = (1.0 / d_model as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let w_o = DMatrix::from_fn(d_model, d_model, |_,_| normal.sample(&mut rng));
        Self { heads, w_o, num_heads, d_model }
    }

    pub fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let outs: Vec<_> = self.heads.iter().map(|h| h.forward(x)).collect();
        let concat = Self::vstack(&outs);
        &self.w_o * &concat
    }

    fn vstack(ms: &[DMatrix<f64>]) -> DMatrix<f64> {
        if ms.is_empty() { return DMatrix::zeros(0,0); }
        let rows: usize = ms.iter().map(|m| m.nrows()).sum();
        let cols = ms[0].ncols();
        let mut r = DMatrix::zeros(rows, cols);
        let mut off = 0;
        for m in ms {
            for i in 0..m.nrows() { for j in 0..m.ncols() { r[(off+i,j)] = m[(i,j)]; } }
            off += m.nrows();
        }
        r
    }
}

// ============================================================================
// TRANSFORMER COMPONENTS
// ============================================================================

#[derive(Debug, Clone)]
pub struct FeedForward {
    pub w1: DMatrix<f64>,
    pub b1: DVector<f64>,
    pub w2: DMatrix<f64>,
    pub b2: DVector<f64>,
    pub d_model: usize,
    pub d_ff: usize,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = rand::thread_rng();
        let s1 = (2.0/d_model as f64).sqrt();
        let s2 = (2.0/d_ff as f64).sqrt();
        let n1 = Normal::new(0.0, s1).unwrap();
        let n2 = Normal::new(0.0, s2).unwrap();
        Self {
            w1: DMatrix::from_fn(d_ff, d_model, |_,_| n1.sample(&mut rng)),
            b1: DVector::zeros(d_ff),
            w2: DMatrix::from_fn(d_model, d_ff, |_,_| n2.sample(&mut rng)),
            b2: DVector::zeros(d_model),
            d_model, d_ff,
        }
    }

    pub fn forward(&self, x: &DVector<f64>) -> DVector<f64> {
        let h = &self.w1 * x + &self.b1;
        let h = DVector::from_iterator(h.len(), h.iter().map(|v| Activation::GELU.apply(*v)));
        &self.w2 * &h + &self.b2
    }
}

#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    pub encodings: DMatrix<f64>,
    pub d_model: usize,
    pub max_len: usize,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let mut enc = DMatrix::zeros(d_model, max_len);
        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f64 / 10000_f64.powf(2.0*(i/2) as f64 / d_model as f64);
                enc[(i, pos)] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }
        Self { encodings: enc, d_model, max_len }
    }

    pub fn encode(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let len = x.ncols().min(self.max_len);
        x + self.encodings.columns(0, len)
    }
}

#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub attention: MultiHeadAttention,
    pub ff: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub d_model: usize,
}

impl TransformerLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, num_heads),
            ff: FeedForward::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            d_model,
        }
    }

    pub fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let n1 = self.norm_cols(&self.norm1, x);
        let attn = self.attention.forward(&n1);
        let x = x + &attn;
        let n2 = self.norm_cols(&self.norm2, &x);
        let mut ff = DMatrix::zeros(self.d_model, x.ncols());
        for (i, col) in n2.column_iter().enumerate() {
            let v = DVector::from_column_slice(col.as_slice());
            ff.set_column(i, &self.ff.forward(&v));
        }
        x + ff
    }

    fn norm_cols(&self, norm: &LayerNorm, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut r = DMatrix::zeros(x.nrows(), x.ncols());
        for (i, col) in x.column_iter().enumerate() {
            let v = DVector::from_column_slice(col.as_slice());
            r.set_column(i, &norm.forward(&v));
        }
        r
    }
}

// ============================================================================
// INFORMATION THEORY
// ============================================================================

pub struct InfoTheory;

impl InfoTheory {
    pub fn entropy(probs: &[f64]) -> f64 {
        probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum()
    }

    pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
        p.iter().zip(q.iter())
            .filter(|(&pi, &qi)| pi > 0.0 && qi > 0.0)
            .map(|(&pi, &qi)| pi * (pi / qi).ln())
            .sum()
    }

    pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
        let m: Vec<f64> = p.iter().zip(q.iter()).map(|(a,b)| (a+b)/2.0).collect();
        0.5 * Self::kl_divergence(p, &m) + 0.5 * Self::kl_divergence(q, &m)
    }

    pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
        p.iter().zip(q.iter()).filter(|(_,&qi)| qi > 0.0).map(|(&pi,&qi)| -pi*qi.ln()).sum()
    }

    pub fn mutual_information(joint: &DMatrix<f64>) -> f64 {
        let p_x: Vec<f64> = (0..joint.nrows()).map(|i| joint.row(i).sum()).collect();
        let p_y: Vec<f64> = (0..joint.ncols()).map(|j| joint.column(j).sum()).collect();
        let mut mi = 0.0;
        for i in 0..joint.nrows() {
            for j in 0..joint.ncols() {
                let p = joint[(i,j)];
                if p > 0.0 && p_x[i] > 0.0 && p_y[j] > 0.0 {
                    mi += p * (p / (p_x[i] * p_y[j])).ln();
                }
            }
        }
        mi
    }
}

// ============================================================================
// OPTIMIZATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    pub m: DVector<f64>,
    pub v: DVector<f64>,
    pub t: usize,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl AdamOptimizer {
    pub fn new(dim: usize) -> Self {
        Self { m: DVector::zeros(dim), v: DVector::zeros(dim), t: 0, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
    }

    pub fn step(&mut self, params: &DVector<f64>, grad: &DVector<f64>, lr: f64) -> DVector<f64> {
        self.t += 1;
        self.m = self.beta1 * &self.m + (1.0 - self.beta1) * grad;
        self.v = self.beta2 * &self.v + (1.0 - self.beta2) * grad.component_mul(grad);
        let m_hat = &self.m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.t as i32));
        params - lr * m_hat.component_div(&(v_hat.map(|x| x.sqrt() + self.eps)))
    }
}

#[derive(Debug, Clone)]
pub struct LRScheduler {
    pub base_lr: f64,
    pub warmup: usize,
    pub total: usize,
    pub step: usize,
}

impl LRScheduler {
    pub fn new(base_lr: f64, warmup: usize, total: usize) -> Self {
        Self { base_lr, warmup, total, step: 0 }
    }

    pub fn get_lr(&mut self) -> f64 {
        self.step += 1;
        if self.step < self.warmup {
            self.base_lr * self.step as f64 / self.warmup as f64
        } else {
            let prog = (self.step - self.warmup) as f64 / (self.total - self.warmup).max(1) as f64;
            self.base_lr * 0.5 * (1.0 + (PI * prog).cos())
        }
    }
}

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

pub struct MatrixOps;

impl MatrixOps {
    pub fn dominant_eigenvalue(m: &DMatrix<f64>, iters: usize) -> (f64, DVector<f64>) {
        let n = m.nrows();
        let mut v = DVector::from_element(n, 1.0/(n as f64).sqrt());
        let mut eigenval = 0.0;
        for _ in 0..iters {
            let mv = m * &v;
            eigenval = v.dot(&mv);
            let norm = mv.norm();
            if norm > 1e-10 { v = mv / norm; }
        }
        (eigenval, v)
    }

    pub fn frobenius_norm(m: &DMatrix<f64>) -> f64 {
        m.iter().map(|x| x*x).sum::<f64>().sqrt()
    }

    pub fn cosine_similarity(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
        let dot = a.dot(b);
        let na = a.norm();
        let nb = b.norm();
        if na > 1e-10 && nb > 1e-10 { dot / (na * nb) } else { 0.0 }
    }

    pub fn outer(a: &DVector<f64>, b: &DVector<f64>) -> DMatrix<f64> {
        let mut r = DMatrix::zeros(a.len(), b.len());
        for i in 0..a.len() { for j in 0..b.len() { r[(i,j)] = a[i] * b[j]; } }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activations() {
        assert!((Activation::ReLU.apply(-1.0)).abs() < 1e-10);
        assert!((Activation::ReLU.apply(1.0) - 1.0).abs() < 1e-10);
        assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_softmax() {
        let p = Activation::Softmax.apply_vector(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dense() {
        let l = DenseLayer::new(10, 5, Activation::ReLU);
        let o = l.forward(&DVector::from_element(10, 1.0));
        assert_eq!(o.len(), 5);
    }

    #[test]
    fn test_mha() {
        let mha = MultiHeadAttention::new(64, 8);
        let x = DMatrix::from_fn(64, 10, |_,_| rand::random::<f64>());
        let o = mha.forward(&x);
        assert_eq!(o.nrows(), 64);
    }

    #[test]
    fn test_transformer() {
        let t = TransformerLayer::new(64, 4, 256);
        let x = DMatrix::from_fn(64, 8, |_,_| rand::random::<f64>());
        let o = t.forward(&x);
        assert_eq!(o.nrows(), 64);
        assert_eq!(o.ncols(), 8);
    }

    #[test]
    fn test_entropy() {
        let u = vec![0.25, 0.25, 0.25, 0.25];
        assert!((InfoTheory::entropy(&u) - 4.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_adam() {
        let mut o = AdamOptimizer::new(5);
        let p = DVector::from_element(5, 1.0);
        let g = DVector::from_element(5, 0.1);
        let np = o.step(&p, &g, 0.01);
        assert!(np.norm() < p.norm());
    }
}
