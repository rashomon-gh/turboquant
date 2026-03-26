//! # TurboQuantMse: MSE-Optimized Vector Quantization
//!
//! This module implements the MSE-optimized variant of TurboQuant, which minimizes
//! reconstruction error using random rotation and independent scalar quantization.
//!
//! ## Algorithm
//!
//! The quantization process works as follows:
//!
//! 1. **Rotation**: Apply a random orthogonal rotation matrix Π to the input vector
//!    - This induces a uniform (Beta) distribution on rotated coordinates
//!    - For high dimensions, this approaches N(0, 1/d)
//!
//! 2. **Quantization**: For each rotated coordinate, find the nearest centroid
//!    - Centroids are precomputed using k-means on the theoretical distribution
//!    - Store only the centroid index (b bits per coordinate)
//!
//! 3. **Dequantization**: Map indices back to centroids, then apply inverse rotation
//!    - Reconstructed vector: x̃ = Π^T · c[indices]
//!
//! ## Properties
//!
//! - **MSE-optimal**: Minimizes E[||x - x̃||²] for the given bit budget
//! - **Data-oblivious**: No training required, works for any distribution
//! - **Online**: Can quantize vectors one at a time without batch processing
//!
//! ## Example
//!
//! ```rust
//! use turboquant::mse::TurboQuantMse;
//! use turboquant::VectorQuantizer;
//! use ndarray::Array1;
//!
//! // Create quantizer for 4-dimensional vectors with 2 bits per coordinate
//! let quantizer = TurboQuantMse::new(4, 2);
//!
//! // Quantize a vector
//! let vector = Array1::from_vec(vec![0.1, -0.5, 0.3, 0.2]);
//! let quantized = quantizer.quantize(&vector.view());
//!
//! // Reconstruct the vector
//! let mut reconstructed = Array1::zeros(4);
//! quantizer.dequantize(&quantized, &mut reconstructed);
//!
//! // Memory usage: 4 bytes for indices (vs 16 bytes for original f32 vector)
//! ```

use crate::centroids::{self, generate_centroids};
use crate::matrix::{self, generate_rotation_matrix_seeded};
use crate::{BatchQuantizer, VectorQuantizer};
use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;

/// MSE-optimized vector quantizer using random rotation and scalar quantization.
///
/// This quantizer provides optimal reconstruction quality (minimum MSE) for a given
/// bit budget by leveraging random rotation and precomputed k-means centroids.
///
/// # Type Parameters
///
/// The quantizer supports 1-8 bits per coordinate, trading off accuracy for compression.
///
/// # Fields
///
/// * `d` - Dimension of the input vectors
/// * `b` - Bits per coordinate (1-8)
/// * `rotation_matrix` - Random orthogonal matrix Π
/// * `rotation_matrix_t` - Precomputed transpose Π^T for efficient dequantization
/// * `centroids` - Precomputed k-means centroids for the rotated distribution
///
/// # Example
///
/// ```rust
/// use turboquant::mse::TurboQuantMse;
///
/// let quantizer = TurboQuantMse::new(1024, 2);
/// println!("Compression ratio: {}x", 4.0 / 2.0); // 32-bit -> 2-bit
/// ```
pub struct TurboQuantMse {
    /// Dimension of the input vectors
    pub d: usize,
    /// Bits per coordinate (1-8)
    pub b: usize,
    /// Random rotation matrix Π (d×d orthogonal)
    pub rotation_matrix: Array2<f32>,
    /// Precomputed transpose Π^T for dequantization
    pub rotation_matrix_t: Array2<f32>,
    /// K-means centroids for scalar quantization (2^b values)
    pub centroids: Vec<f32>,
}

impl TurboQuantMse {
    /// Create a new MSE quantizer with random seed.
    ///
    /// # Arguments
    ///
    /// * `d` - Dimension of vectors to quantize
    /// * `b` - Bits per coordinate (1-8)
    ///
    /// # Panics
    ///
    /// Panics if `b < 1` or `d == 0`.
    pub fn new(d: usize, b: usize) -> Self {
        Self::new_seeded(d, b, rand::random())
    }

    /// Create a new MSE quantizer with a specific random seed.
    ///
    /// Useful for reproducibility in testing and benchmarking.
    ///
    /// # Arguments
    ///
    /// * `d` - Dimension of vectors to quantize
    /// * `b` - Bits per coordinate (1-8)
    /// * `seed` - Random seed for rotation matrix generation
    ///
    /// # Example
    ///
    /// ```rust
    /// use turboquant::mse::TurboQuantMse;
    ///
    /// let q1 = TurboQuantMse::new_seeded(128, 2, 42);
    /// let q2 = TurboQuantMse::new_seeded(128, 2, 42);
    /// // q1 and q2 will produce identical quantization results
    /// ```
    pub fn new_seeded(d: usize, b: usize, seed: u64) -> Self {
        assert!(b >= 1, "TurboQuantMse requires at least 1 bit");
        assert!(d > 0, "Dimension must be positive");

        let rotation_matrix = generate_rotation_matrix_seeded(d, seed);
        let rotation_matrix_t = rotation_matrix.t().to_owned();
        let centroids = generate_centroids(d, b);

        Self {
            d,
            b,
            rotation_matrix,
            rotation_matrix_t,
            centroids,
        }
    }

    /// Quantize a vector into pre-allocated buffers (zero-allocation pattern).
    ///
    /// This is the hot path for performance-critical applications.
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector to quantize
    /// * `indices` - Output buffer for centroid indices (length = d)
    /// * `rotated` - Temporary buffer for rotated vector (length = d)
    ///
    /// # Algorithm
    ///
    /// 1. rotated = Π · x
    /// 2. indices[i] = argmin_j |rotated[i] - centroid[j]|
    pub fn quantize_into_buffer(
        &self,
        x: &ArrayView1<f32>,
        indices: &mut [u8],
        rotated: &mut [f32],
    ) {
        debug_assert_eq!(x.len(), self.d);
        debug_assert_eq!(indices.len(), self.d);
        debug_assert_eq!(rotated.len(), self.d);

        // Step 1: Apply random rotation
        matrix::matrix_vector_multiply(
            &self.rotation_matrix.view(),
            x.as_slice().unwrap(),
            rotated,
        );

        // Step 2: Find nearest centroid for each rotated coordinate
        centroids::find_nearest_centroids_batch(rotated, &self.centroids, indices);
    }

    /// Dequantize indices into a vector using pre-allocated buffers.
    ///
    /// # Arguments
    ///
    /// * `indices` - Centroid indices from quantization
    /// * `out` - Output buffer for reconstructed vector (length = d)
    /// * `temp` - Temporary buffer for centroid values (length = d)
    ///
    /// # Algorithm
    ///
    /// 1. temp[i] = centroids[indices[i]]
    /// 2. out = Π^T · temp
    pub fn dequantize_into_buffer(&self, indices: &[u8], out: &mut [f32], temp: &mut [f32]) {
        debug_assert_eq!(indices.len(), self.d);
        debug_assert_eq!(out.len(), self.d);
        debug_assert_eq!(temp.len(), self.d);

        // Step 1: Map indices to centroid values
        for (i, &idx) in indices.iter().enumerate() {
            temp[i] = self.centroids[idx as usize];
        }

        // Step 2: Apply inverse rotation
        matrix::matrix_transpose_vector_multiply(&self.rotation_matrix.view(), temp, out);
    }
}

/// Quantized representation of a vector using MSE-optimized quantization.
///
/// Stores only the centroid indices, achieving b bits per coordinate.
///
/// # Memory Layout
///
/// For dimension d and b bits:
/// - Storage: d bytes (using u8 for indices)
/// - Original: 4d bytes (f32)
/// - Compression: 4/b × reduction
#[derive(Clone, Debug)]
pub struct MseQuantizedVector {
    /// Centroid indices for each coordinate (0 to 2^b - 1)
    pub indices: Vec<u8>,
}

impl VectorQuantizer for TurboQuantMse {
    type QuantizedType = MseQuantizedVector;

    fn quantize(&self, x: &ArrayView1<f32>) -> Self::QuantizedType {
        let mut indices = vec![0u8; self.d];
        let mut rotated = vec![0.0f32; self.d];
        self.quantize_into_buffer(x, &mut indices, &mut rotated);
        MseQuantizedVector { indices }
    }

    fn quantize_into(&self, x: &ArrayView1<f32>, out: &mut Self::QuantizedType) {
        let mut rotated = vec![0.0f32; self.d];
        self.quantize_into_buffer(x, &mut out.indices, &mut rotated);
    }

    fn dequantize(&self, quantized: &Self::QuantizedType, out: &mut Array1<f32>) {
        let mut temp = vec![0.0f32; self.d];
        self.dequantize_into_buffer(&quantized.indices, out.as_slice_mut().unwrap(), &mut temp);
    }
}

impl BatchQuantizer for TurboQuantMse {
    /// Quantize multiple vectors in parallel using Rayon.
    ///
    /// Automatically parallelizes across available CPU cores for
    /// efficient batch processing of large vector collections.
    fn quantize_batch(&self, vectors: &[ArrayView1<f32>]) -> Vec<Self::QuantizedType> {
        vectors.par_iter().map(|x| self.quantize(x)).collect()
    }

    /// Dequantize multiple vectors in parallel using Rayon.
    fn dequantize_batch(&self, quantized: &[Self::QuantizedType], outputs: &mut [Array1<f32>]) {
        outputs
            .par_iter_mut()
            .zip(quantized.par_iter())
            .for_each(|(out, q)| {
                self.dequantize(q, out);
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_quantize_dequantize_roundtrip() {
        let d = 64;
        let b = 2;
        let quantizer = TurboQuantMse::new_seeded(d, b, 42);

        let original = Array1::from_vec((0..d).map(|i| (i as f32) * 0.01 - 0.3).collect());

        let quantized = quantizer.quantize(&original.view());

        let mut reconstructed = Array1::zeros(d);
        quantizer.dequantize(&quantized, &mut reconstructed);

        for i in 0..d {
            assert!(
                reconstructed[i].is_finite(),
                "Reconstructed value should be finite"
            );
        }
    }

    #[test]
    fn test_mse_batch_quantization() {
        let d = 32;
        let b = 1;
        let quantizer = TurboQuantMse::new_seeded(d, b, 123);

        let vectors: Vec<Array1<f32>> = (0..10)
            .map(|i| Array1::from_vec((0..d).map(|j| (j as f32 + i as f32) * 0.05).collect()))
            .collect();

        let views: Vec<ArrayView1<f32>> = vectors.iter().map(|v| v.view()).collect();

        let quantized = quantizer.quantize_batch(&views);

        assert_eq!(quantized.len(), 10);
        for q in &quantized {
            assert_eq!(q.indices.len(), d);
        }
    }

    #[test]
    fn test_quantize_into_buffer() {
        let d = 16;
        let b = 1;
        let quantizer = TurboQuantMse::new_seeded(d, b, 42);

        let x = Array1::from_vec((0..d).map(|i| (i as f32) * 0.1).collect());

        let mut indices = vec![0u8; d];
        let mut rotated = vec![0.0f32; d];
        quantizer.quantize_into_buffer(&x.view(), &mut indices, &mut rotated);

        for &idx in &indices {
            assert!(idx < (1 << b) as u8);
        }
    }
}
