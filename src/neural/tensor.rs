//! Tensor Operations for Neural Networks
//!
//! A lightweight tensor implementation supporting:
//! - N-dimensional arrays
//! - Basic operations (matmul, add, etc.)
//! - Automatic differentiation (autograd)
//! - CPU execution (GPU-ready interface)

use std::ops::{Add, Mul, Sub, Neg};
use std::sync::{Arc, Mutex};
use rand::Rng;
use rand_distr::{Normal, Distribution};

/// Device for tensor computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CPU,
    CUDA(usize), // GPU index
}

impl Default for Device {
    fn default() -> Self {
        Device::CPU
    }
}

/// Shape of a tensor
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape(pub Vec<usize>);

impl TensorShape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self(dims)
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    pub fn dim(&self, i: usize) -> usize {
        self.0[i]
    }
}

impl From<Vec<usize>> for TensorShape {
    fn from(v: Vec<usize>) -> Self {
        Self(v)
    }
}

impl From<&[usize]> for TensorShape {
    fn from(s: &[usize]) -> Self {
        Self(s.to_vec())
    }
}

/// Gradient tracking for autograd
#[derive(Clone)]
pub struct GradFn {
    pub name: String,
    pub inputs: Vec<Tensor>,
    pub backward: Option<Arc<dyn Fn(&Tensor, &Tensor) -> Vec<Tensor> + Send + Sync>>,
}

impl std::fmt::Debug for GradFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradFn")
            .field("name", &self.name)
            .field("inputs", &self.inputs.len())
            .finish()
    }
}

/// N-dimensional Tensor with autograd support
#[derive(Clone)]
pub struct Tensor {
    /// Raw data storage
    pub data: Vec<f32>,
    /// Shape of the tensor
    pub shape: TensorShape,
    /// Device (CPU/GPU)
    pub device: Device,
    /// Whether to track gradients
    pub requires_grad: bool,
    /// Accumulated gradient
    pub grad: Arc<Mutex<Option<Vec<f32>>>>,
    /// Gradient function for backprop
    grad_fn: Option<Arc<GradFn>>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("device", &self.device)
            .field("requires_grad", &self.requires_grad)
            .field("data_len", &self.data.len())
            .finish()
    }
}

impl Tensor {
    /// Create a new tensor from data
    pub fn new(data: Vec<f32>, shape: impl Into<TensorShape>) -> Self {
        let shape = shape.into();
        assert_eq!(data.len(), shape.numel(), "Data size must match shape");
        Self {
            data: data,
            shape,
            device: Device::CPU,
            requires_grad: false,
            grad: Arc::new(Mutex::new(None)),
            grad_fn: None,
        }
    }

    /// Create tensor with gradient tracking
    pub fn with_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Create a zeros tensor
    pub fn zeros(shape: impl Into<TensorShape>) -> Self {
        let shape = shape.into();
        Self::new(vec![0.0; shape.numel()], shape)
    }

    /// Create a ones tensor
    pub fn ones(shape: impl Into<TensorShape>) -> Self {
        let shape = shape.into();
        Self::new(vec![1.0; shape.numel()], shape)
    }

    /// Create tensor with random normal values
    pub fn randn(shape: impl Into<TensorShape>) -> Self {
        let shape = shape.into();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f32> = (0..shape.numel())
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();
        Self::new(data, shape)
    }

    /// Create tensor with Xavier/Glorot initialization
    pub fn xavier(shape: impl Into<TensorShape>) -> Self {
        let shape = shape.into();
        let fan_in = if shape.ndim() >= 2 { shape.dim(1) } else { shape.dim(0) };
        let fan_out = shape.dim(0);
        let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
        
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, std).unwrap();
        let data: Vec<f32> = (0..shape.numel())
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();
        Self::new(data, shape)
    }

    /// Create tensor with Kaiming/He initialization
    pub fn kaiming(shape: impl Into<TensorShape>) -> Self {
        let shape = shape.into();
        let fan_in = if shape.ndim() >= 2 { shape.dim(1) } else { shape.dim(0) };
        let std = (2.0 / fan_in as f64).sqrt();
        
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, std).unwrap();
        let data: Vec<f32> = (0..shape.numel())
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();
        Self::new(data, shape)
    }

    /// Get element at index
    pub fn get(&self, indices: &[usize]) -> f32 {
        let idx = self.flat_index(indices);
        self.data[idx]
    }

    /// Set element at index (creates new tensor - immutable design)
    pub fn set(&self, indices: &[usize], value: f32) -> Self {
        let idx = self.flat_index(indices);
        let mut new_data = self.data.clone();
        new_data[idx] = value;
        Self::new(new_data, self.shape.clone())
    }

    /// Convert multi-dimensional index to flat index
    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.ndim());
        let mut idx = 0;
        let mut stride = 1;
        for i in (0..indices.len()).rev() {
            idx += indices[i] * stride;
            stride *= self.shape.dim(i);
        }
        idx
    }

    /// Reshape tensor
    pub fn reshape(&self, new_shape: impl Into<TensorShape>) -> Self {
        let new_shape = new_shape.into();
        assert_eq!(self.shape.numel(), new_shape.numel(), "Total elements must match");
        Self {
            data: self.data.clone(),
            shape: new_shape,
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(Mutex::new(None)),
            grad_fn: None,
        }
    }

    /// Transpose 2D tensor
    pub fn transpose(&self) -> Self {
        assert_eq!(self.shape.ndim(), 2, "Transpose requires 2D tensor");
        let (rows, cols) = (self.shape.dim(0), self.shape.dim(1));
        let mut new_data = vec![0.0; self.shape.numel()];
        
        for i in 0..rows {
            for j in 0..cols {
                new_data[j * rows + i] = self.data[i * cols + j];
            }
        }
        
        Self::new(new_data, vec![cols, rows])
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape.ndim(), 2, "matmul requires 2D tensors");
        assert_eq!(other.shape.ndim(), 2, "matmul requires 2D tensors");
        assert_eq!(self.shape.dim(1), other.shape.dim(0), "Inner dimensions must match");

        let m = self.shape.dim(0);
        let k = self.shape.dim(1);
        let n = other.shape.dim(1);

        let mut result = vec![0.0; m * n];

        // Basic matmul - can be optimized with BLAS
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        let mut output = Self::new(result, vec![m, n]);
        
        // Set up autograd if needed
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            let self_clone = self.clone();
            let other_clone = other.clone();
            output.grad_fn = Some(Arc::new(GradFn {
                name: "matmul".to_string(),
                inputs: vec![self_clone.clone(), other_clone.clone()],
                backward: Some(Arc::new(move |_output, grad_output| {
                    // d/dA (A @ B) = grad @ B^T
                    // d/dB (A @ B) = A^T @ grad
                    let grad_a = Tensor::new(grad_output.data.to_vec(), grad_output.shape.clone())
                        .matmul(&other_clone.transpose());
                    let grad_b = self_clone.transpose()
                        .matmul(&Tensor::new(grad_output.data.to_vec(), grad_output.shape.clone()));
                    vec![grad_a, grad_b]
                })),
            }));
        }

        output
    }

    /// Batched matrix multiplication [B, M, K] @ [B, K, N] -> [B, M, N]
    pub fn bmm(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape.ndim(), 3, "bmm requires 3D tensors");
        assert_eq!(other.shape.ndim(), 3, "bmm requires 3D tensors");
        assert_eq!(self.shape.dim(0), other.shape.dim(0), "Batch sizes must match");
        assert_eq!(self.shape.dim(2), other.shape.dim(1), "Inner dimensions must match");

        let batch = self.shape.dim(0);
        let m = self.shape.dim(1);
        let k = self.shape.dim(2);
        let n = other.shape.dim(2);

        let mut result = vec![0.0; batch * m * n];

        for b in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        let self_idx = b * m * k + i * k + l;
                        let other_idx = b * k * n + l * n + j;
                        sum += self.data[self_idx] * other.data[other_idx];
                    }
                    result[b * m * n + i * n + j] = sum;
                }
            }
        }

        Self::new(result, vec![batch, m, n])
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape, "Shapes must match for add");
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        let mut output = Self::new(data, self.shape.clone());
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
        }
        output
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn mul(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape, "Shapes must match for mul");
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Self::new(data, self.shape.clone())
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f32) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| x * scalar).collect();
        Self::new(data, self.shape.clone())
    }

    /// ReLU activation
    pub fn relu(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x.max(0.0)).collect();
        Self::new(data, self.shape.clone())
    }

    /// GELU activation (Gaussian Error Linear Unit)
    pub fn gelu(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| {
            let x64 = x as f64;
            (0.5 * x64 * (1.0 + (0.7978845608 * (x64 + 0.044715 * x64.powi(3))).tanh())) as f32
        }).collect();
        Self::new(data, self.shape.clone())
    }

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| {
            1.0 / (1.0 + (-x).exp())
        }).collect();
        Self::new(data, self.shape.clone())
    }

    /// Tanh activation
    pub fn tanh(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x.tanh()).collect();
        Self::new(data, self.shape.clone())
    }

    /// Softmax along last dimension
    pub fn softmax(&self) -> Self {
        let last_dim = self.shape.dim(self.shape.ndim() - 1);
        let outer_size = self.shape.numel() / last_dim;
        
        let mut data = vec![0.0; self.shape.numel()];
        
        for i in 0..outer_size {
            let start = i * last_dim;
            let end = start + last_dim;
            
            // Find max for numerical stability
            let max_val = self.data[start..end].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            
            // Compute exp and sum
            let mut sum = 0.0;
            for j in start..end {
                let exp_val = (self.data[j] - max_val).exp();
                data[j] = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for j in start..end {
                data[j] /= sum;
            }
        }
        
        Self::new(data, self.shape.clone())
    }

    /// Layer normalization
    pub fn layer_norm(&self, eps: f32) -> Self {
        let last_dim = self.shape.dim(self.shape.ndim() - 1);
        let outer_size = self.shape.numel() / last_dim;
        
        let mut data = vec![0.0; self.shape.numel()];
        
        for i in 0..outer_size {
            let start = i * last_dim;
            let end = start + last_dim;
            
            // Compute mean
            let mean: f32 = self.data[start..end].iter().sum::<f32>() / last_dim as f32;
            
            // Compute variance
            let var: f32 = self.data[start..end].iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / last_dim as f32;
            
            // Normalize
            let std = (var + eps).sqrt();
            for j in start..end {
                data[j] = (self.data[j] - mean) / std;
            }
        }
        
        Self::new(data, self.shape.clone())
    }

    /// Sum all elements
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f32 {
        self.sum() / self.shape.numel() as f32
    }

    /// Normalize to unit L2 norm
    pub fn normalize(&self) -> Self {
        let norm: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            return self.clone();
        }
        self.scale(1.0 / norm)
    }

    /// Sum along dimension
    pub fn sum_dim(&self, dim: usize) -> Self {
        assert!(dim < self.shape.ndim(), "Dimension out of bounds");
        
        let mut new_shape = self.shape.0.clone();
        new_shape.remove(dim);
        if new_shape.is_empty() {
            new_shape.push(1);
        }
        
        let new_numel: usize = new_shape.iter().product();
        let mut result = vec![0.0; new_numel];
        
        let dim_size = self.shape.dim(dim);
        let outer_size: usize = self.shape.0[..dim].iter().product();
        let inner_size: usize = self.shape.0[dim + 1..].iter().product();
        
        for outer in 0..outer_size.max(1) {
            for inner in 0..inner_size.max(1) {
                let mut sum = 0.0;
                for d in 0..dim_size {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    sum += self.data[idx];
                }
                let out_idx = outer * inner_size + inner;
                result[out_idx] = sum;
            }
        }
        
        Self::new(result, new_shape)
    }

    /// Dropout (training mode)
    pub fn dropout(&self, p: f32, training: bool) -> Self {
        if !training || p == 0.0 {
            return self.clone();
        }
        
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - p);
        let data: Vec<f32> = self.data.iter().map(|&x| {
            if rng.gen::<f32>() < p {
                0.0
            } else {
                x * scale
            }
        }).collect();
        
        Self::new(data, self.shape.clone())
    }

    /// Concatenate tensors along dimension
    pub fn cat(tensors: &[Tensor], dim: usize) -> Self {
        assert!(!tensors.is_empty(), "Cannot concatenate empty list");
        
        let first = &tensors[0];
        let ndim = first.shape.ndim();
        assert!(dim < ndim, "Dimension out of bounds");
        
        // Verify shapes match except for concat dimension
        for t in tensors.iter().skip(1) {
            assert_eq!(t.shape.ndim(), ndim);
            for i in 0..ndim {
                if i != dim {
                    assert_eq!(t.shape.dim(i), first.shape.dim(i));
                }
            }
        }
        
        // Calculate new shape
        let mut new_shape = first.shape.0.clone();
        new_shape[dim] = tensors.iter().map(|t| t.shape.dim(dim)).sum();
        
        // Simple case: concatenate along last dimension
        if dim == ndim - 1 {
            let outer_size: usize = first.shape.0[..dim].iter().product();
            let mut result = Vec::with_capacity(new_shape.iter().product());
            
            for outer in 0..outer_size.max(1) {
                for t in tensors {
                    let dim_size = t.shape.dim(dim);
                    let start = outer * dim_size;
                    result.extend_from_slice(&t.data[start..start + dim_size]);
                }
            }
            
            return Self::new(result, new_shape);
        }
        
        // General case - more complex indexing
        let new_numel: usize = new_shape.iter().product();
        let mut result = vec![0.0; new_numel];
        
        // TODO: Implement general case
        // For now, fall back to simple copy
        let mut offset = 0;
        for t in tensors {
            for &val in t.data.iter() {
                if offset < result.len() {
                    result[offset] = val;
                    offset += 1;
                }
            }
        }
        
        Self::new(result, new_shape)
    }

    /// Backward pass (simple version)
    pub fn backward(&self) {
        if !self.requires_grad {
            return;
        }
        
        // Initialize gradient as ones (for scalar loss)
        let grad = vec![1.0; self.shape.numel()];
        *self.grad.lock().unwrap() = Some(grad);
        
        // TODO: Implement full autograd with topological sort
        // For now, gradients are computed manually in the trainer
    }

    /// Zero gradients
    pub fn zero_grad(&self) {
        *self.grad.lock().unwrap() = None;
    }

    /// Get gradient
    pub fn get_grad(&self) -> Option<Vec<f32>> {
        self.grad.lock().unwrap().as_ref().map(|g| g.to_vec())
    }

    /// Convert to Vec
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.to_vec()
    }

    /// Convert to 2D Vec
    pub fn to_vec2d(&self) -> Vec<Vec<f32>> {
        assert_eq!(self.shape.ndim(), 2);
        let rows = self.shape.dim(0);
        let cols = self.shape.dim(1);
        
        (0..rows)
            .map(|i| self.data[i * cols..(i + 1) * cols].to_vec())
            .collect()
    }
}

// Operator overloads
impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        self.add(other)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Tensor {
        self.mul(other)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        self.scale(-1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.shape.0, vec![2, 3]);
        assert_eq!(t.data.len(), 6);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.matmul(&b);
        
        assert_eq!(c.shape.0, vec![2, 2]);
        // [1,2] @ [5,6] = 1*5 + 2*7 = 19, 1*6 + 2*8 = 22
        // [3,4]   [7,8]   3*5 + 4*7 = 43, 3*6 + 4*8 = 50
        assert!((c.data[0] - 19.0).abs() < 1e-5);
        assert!((c.data[1] - 22.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let s = t.softmax();
        
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let n = t.layer_norm(1e-5);
        
        // Each row should have mean ~0 and std ~1
        let mean1 = (n.data[0] + n.data[1]) / 2.0;
        assert!(mean1.abs() < 1e-5);
    }
}
