//! # TurboQuant: High-Performance Vector Quantization
//!
//! TurboQuant is an online, data-oblivious vector quantization algorithm optimized for
//! high-dimensional Euclidean spaces. It provides efficient compression of vectors with
//! two variants:
//!
//! - **MSE-optimized**: Minimizes reconstruction error using random rotation and scalar quantization
//! - **Product-optimized**: Provides unbiased inner product estimation for similarity search
//!
//! ## Architecture
//!
//! The library is organized into several modules:
//!
//! - [`mse`]: MSE-optimized quantization using random rotation matrices
//! - [`prod`]: Product quantization combining MSE + QJL for inner product estimation
//! - [`centroids`]: K-means centroid generation with hardcoded fallbacks
//! - [`matrix`]: Random rotation matrix generation via QR decomposition
//! - [`qjl`]: Quantized Johnson-Lindenstrauss transform for dimensionality reduction
//! - [`bitpack`]: Bit-packing utilities for compact 1-bit storage
//!
//! ## Quick Example
//!
//! ```rust
//! use turboquant::mse::TurboQuantMse;
//! use turboquant::VectorQuantizer;
//! use ndarray::Array1;
//!
//! // Create a quantizer for 4-dimensional vectors with 2 bits per coordinate
//! let quantizer = TurboQuantMse::new(4, 2);
//!
//! // Quantize a vector
//! let vector = Array1::from_vec(vec![0.1, -0.5, 0.3, 0.2]);
//! let quantized = quantizer.quantize(&vector.view());
//!
//! // Reconstruct the vector
//! let mut reconstructed = Array1::zeros(4);
//! quantizer.dequantize(&quantized, &mut reconstructed);
//! ```

pub mod bitpack;
pub mod centroids;
pub mod matrix;
pub mod mse;
pub mod prod;
pub mod qjl;

use ndarray::{Array1, ArrayView1};

/// Trait for single-vector quantization operations.
///
/// This trait defines the core interface for quantizing and dequantizing vectors.
/// Implementations should support both single-vector operations and zero-allocation
/// patterns using pre-allocated buffers.
///
/// # Type Parameters
///
/// * `QuantizedType` - The compressed representation type (e.g., `MseQuantizedVector`)
///
/// # Example
///
/// ```rust
/// use turboquant::mse::TurboQuantMse;
/// use turboquant::VectorQuantizer;
/// use ndarray::Array1;
///
/// let quantizer = TurboQuantMse::new(64, 2);
/// let vector = Array1::zeros(64);
///
/// let quantized = quantizer.quantize(&vector.view());
/// let mut output = Array1::zeros(64);
/// quantizer.dequantize(&quantized, &mut output);
/// ```
pub trait VectorQuantizer {
    /// The compressed representation type for this quantizer.
    type QuantizedType;

    /// Quantize a single vector into compressed form.
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector to quantize
    ///
    /// # Returns
    ///
    /// The quantized representation of the vector.
    fn quantize(&self, x: &ArrayView1<f32>) -> Self::QuantizedType;

    /// Quantize a vector into an existing buffer (zero-allocation pattern).
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector to quantize
    /// * `out` - Pre-allocated output buffer to write quantized data
    fn quantize_into(&self, x: &ArrayView1<f32>, out: &mut Self::QuantizedType);

    /// Reconstruct a vector from its quantized representation.
    ///
    /// # Arguments
    ///
    /// * `quantized` - The compressed vector representation
    /// * `out` - Pre-allocated output buffer for the reconstructed vector
    fn dequantize(&self, quantized: &Self::QuantizedType, out: &mut Array1<f32>);
}

/// Trait for batch quantization operations with parallel processing.
///
/// Extends `VectorQuantizer` with parallel batch processing capabilities using Rayon.
/// This is particularly useful when quantizing large numbers of vectors (e.g., token matrices).
///
/// # Example
///
/// ```rust
/// use turboquant::mse::TurboQuantMse;
/// use turboquant::{BatchQuantizer, VectorQuantizer};
/// use ndarray::Array1;
///
/// let quantizer = TurboQuantMse::new(64, 2);
/// let vectors: Vec<Array1<f32>> = vec![Array1::zeros(64), Array1::zeros(64)];
/// let views: Vec<_> = vectors.iter().map(|v| v.view()).collect();
///
/// // Quantize all vectors in parallel
/// let quantized_batch = quantizer.quantize_batch(&views);
///
/// // Dequantize all vectors in parallel
/// let mut outputs = vec![Array1::zeros(64); 2];
/// quantizer.dequantize_batch(&quantized_batch, &mut outputs);
/// ```
pub trait BatchQuantizer: VectorQuantizer {
    /// Quantize multiple vectors in parallel.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Slice of vector views to quantize
    ///
    /// # Returns
    ///
    /// Vector of quantized representations.
    fn quantize_batch(&self, vectors: &[ArrayView1<f32>]) -> Vec<Self::QuantizedType>;

    /// Dequantize multiple vectors in parallel.
    ///
    /// # Arguments
    ///
    /// * `quantized` - Slice of quantized vectors
    /// * `outputs` - Mutable slice of pre-allocated output arrays
    fn dequantize_batch(&self, quantized: &[Self::QuantizedType], outputs: &mut [Array1<f32>]);
}
