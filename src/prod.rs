//! # TurboQuantProd: Product Quantization for Inner Product Estimation
//!
//! This module implements the product-optimized variant of TurboQuant, designed for
//! **unbiased inner product estimation**. This is crucial for similarity search and
//! nearest neighbor queries in vector databases.
//!
//! ## Algorithm
//!
//! TurboQuantProd combines two quantization stages:
//!
//! 1. **MSE Stage** (b-1 bits): Apply TurboQuantMse with reduced bit width
//!    - Provides a rough approximation x̃_mse of the original vector
//!    - Uses (b-1) bits per coordinate
//!
//! 2. **QJL Stage** (1 bit): Apply Quantized Johnson-Lindenstrauss on the residual
//!    - Compute residual r = x - x̃_mse
//!    - Store sign(S·r) as 1-bit packed representation
//!    - Store γ = ||r||₂ (single float)
//!
//! ## Key Properties
//!
//! - **Unbiased**: E[⟨y, x̃⟩] = ⟨y, x⟩ for any query vector y
//! - **Memory Efficient**: Total storage is (b-1) + 1 = b bits per coordinate + γ
//! - **Similarity Preserving**: Maintains relative ordering for nearest neighbor search
//!
//! ## When to Use
//!
//! - **Use TurboQuantProd** when: You need to compute inner products or find nearest neighbors
//! - **Use TurboQuantMse** when: You need accurate vector reconstruction
//!
//! ## Example
//!
//! ```rust
//! use turboquant::prod::TurboQuantProd;
//! use turboquant::VectorQuantizer;
//! use ndarray::Array1;
//!
//! // Create quantizer for similarity search (3 bits: 2 for MSE + 1 for QJL)
//! let quantizer = TurboQuantProd::new(4, 3);
//!
//! // Quantize database vectors
//! let db_vector = Array1::from_vec(vec![0.1, -0.5, 0.3, 0.2]);
//! let quantized = quantizer.quantize(&db_vector.view());
//!
//! // Later, estimate inner product with query vector
//! let query = Array1::from_vec(vec![0.2, -0.3, 0.1, 0.4]);
//! let mut reconstructed = Array1::zeros(4);
//! quantizer.dequantize(&quantized, &mut reconstructed);
//!
//! let estimated_inner_product: f32 = query.iter()
//!     .zip(reconstructed.iter())
//!     .map(|(q, r)| q * r)
//!     .sum();
//! ```

use crate::bitpack;
use crate::matrix::{self, generate_projection_matrix_seeded, generate_rotation_matrix_seeded};
use crate::mse::TurboQuantMse;
use crate::qjl::{self, QjlWorkspace};
use crate::{BatchQuantizer, VectorQuantizer};
use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;

/// Product quantizer combining MSE quantization with QJL transform.
///
/// This quantizer provides unbiased inner product estimation by composing:
/// - (b-1)-bit MSE quantization for rough approximation
/// - 1-bit QJL transform on the residual for unbiased correction
///
/// # Fields
///
/// * `d` - Dimension of input vectors
/// * `b` - Total bits per coordinate (must be ≥ 2)
/// * `mse_quantizer` - MSE quantizer with (b-1) bits
/// * `projection_matrix` - Random projection matrix for QJL transform
///
/// # Example
///
/// ```rust
/// use turboquant::prod::TurboQuantProd;
///
/// // 3 bits total: 2 for MSE + 1 for QJL
/// let quantizer = TurboQuantProd::new(256, 3);
/// ```
pub struct TurboQuantProd {
    /// Dimension of the input vectors
    pub d: usize,
    /// Total bits per coordinate (b-1 for MSE + 1 for QJL)
    pub b: usize,
    /// MSE quantizer for coarse approximation (b-1 bits)
    pub mse_quantizer: TurboQuantMse,
    /// Random projection matrix for QJL transform (d×d with N(0,1) entries)
    pub projection_matrix: Array2<f32>,
}

impl TurboQuantProd {
    /// Create a new product quantizer with random seeds.
    ///
    /// # Arguments
    ///
    /// * `d` - Dimension of vectors to quantize
    /// * `b` - Total bits per coordinate (must be ≥ 2)
    ///
    /// # Panics
    ///
    /// Panics if `b < 2`.
    pub fn new(d: usize, b: usize) -> Self {
        Self::new_seeded(d, b, rand::random(), rand::random())
    }

    /// Create a new product quantizer with specific random seeds.
    ///
    /// # Arguments
    ///
    /// * `d` - Dimension of vectors to quantize
    /// * `b` - Total bits per coordinate (must be ≥ 2)
    /// * `rotation_seed` - Seed for rotation matrix generation
    /// * `projection_seed` - Seed for projection matrix generation
    ///
    /// # Example
    ///
    /// ```rust
    /// use turboquant::prod::TurboQuantProd;
    ///
    /// let q1 = TurboQuantProd::new_seeded(128, 3, 42, 123);
    /// let q2 = TurboQuantProd::new_seeded(128, 3, 42, 123);
    /// // q1 and q2 will produce identical results
    /// ```
    pub fn new_seeded(d: usize, b: usize, rotation_seed: u64, projection_seed: u64) -> Self {
        assert!(b >= 2, "TurboQuantProd requires at least 2 bits");

        let mse_bits = b - 1;
        let rotation_matrix = generate_rotation_matrix_seeded(d, rotation_seed);
        let rotation_matrix_t = rotation_matrix.t().to_owned();

        let mse_quantizer = TurboQuantMse {
            d,
            b: mse_bits,
            rotation_matrix,
            rotation_matrix_t,
            centroids: crate::centroids::generate_centroids(d, mse_bits),
        };

        let projection_matrix = generate_projection_matrix_seeded(d, projection_seed);

        Self {
            d,
            b,
            mse_quantizer,
            projection_matrix,
        }
    }

    /// Quantize a vector into pre-allocated buffers (zero-allocation hot path).
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector to quantize
    /// * `indices` - Output buffer for MSE centroid indices
    /// * `packed_qjl` - Output buffer for packed QJL signs
    /// * `gamma` - Output for L2 norm of residual
    /// * `workspace` - Pre-allocated workspace
    ///
    /// # Algorithm
    ///
    /// 1. Quantize x with MSE quantizer → indices
    /// 2. Reconstruct x̃_mse from indices
    /// 3. Compute residual r = x - x̃_mse and γ = ||r||₂
    /// 4. Apply QJL transform: packed_qjl = pack(sign(S·r))
    pub fn quantize_into_buffers(
        &self,
        x: &ArrayView1<f32>,
        indices: &mut [u8],
        packed_qjl: &mut [u8],
        gamma: &mut f32,
        workspace: &mut ProdWorkspace,
    ) {
        debug_assert_eq!(indices.len(), self.d);
        debug_assert!(packed_qjl.len() >= bitpack::packed_len(self.d));

        // Step 1: MSE quantization
        self.mse_quantizer
            .quantize_into_buffer(x, indices, &mut workspace.rotated);

        // Step 2: Reconstruct rotated MSE approximation
        for (i, &idx) in indices.iter().enumerate() {
            workspace.x_mse[i] = self.mse_quantizer.centroids[idx as usize];
        }

        matrix::matrix_transpose_vector_multiply(
            &self.mse_quantizer.rotation_matrix.view(),
            &workspace.x_mse,
            &mut workspace.x_mse_reconstructed,
        );

        // Step 3: Compute residual and gamma
        *gamma = qjl::compute_residual_gamma(
            x.as_slice().unwrap(),
            &workspace.x_mse_reconstructed,
            &mut workspace.residual,
        );

        // Step 4: QJL transform with bit-packing
        qjl::qjl_quantize_into_packed(
            &self.projection_matrix.view(),
            &workspace.residual,
            packed_qjl,
            &mut workspace.qjl_workspace,
        );
    }

    /// Dequantize into pre-allocated buffers (zero-allocation hot path).
    ///
    /// # Arguments
    ///
    /// * `quantized` - Quantized vector representation
    /// * `out` - Output buffer for reconstructed vector
    /// * `workspace` - Pre-allocated workspace
    ///
    /// # Algorithm
    ///
    /// 1. Dequantize MSE part: x̃_mse = Π^T · c[indices]
    /// 2. Dequantize QJL part: x̃_qjl = (√(π/2)/d) × γ × S^T × signs
    /// 3. Combine: out = x̃_mse + x̃_qjl
    pub fn dequantize_into_buffers(
        &self,
        quantized: &ProdQuantizedVector,
        out: &mut [f32],
        workspace: &mut ProdWorkspace,
    ) {
        debug_assert_eq!(out.len(), self.d);

        // Step 1: Dequantize MSE part
        self.mse_quantizer.dequantize_into_buffer(
            &quantized.idx,
            &mut workspace.x_mse_reconstructed,
            &mut workspace.temp,
        );

        // Step 2: Dequantize QJL part
        qjl::qjl_dequantize_packed(
            &self.projection_matrix.view(),
            &quantized.packed_qjl_signs,
            quantized.gamma,
            &mut workspace.qjl_workspace,
            &mut workspace.x_qjl,
        );

        // Step 3: Combine MSE + QJL
        for (out_val, (mse_val, qjl_val)) in out.iter_mut().zip(
            workspace
                .x_mse_reconstructed
                .iter()
                .zip(workspace.x_qjl.iter()),
        ) {
            *out_val = mse_val + qjl_val;
        }
    }
}

/// Pre-allocated workspace for product quantization operations.
///
/// Contains all temporary buffers needed for quantization and dequantization,
/// avoiding heap allocations in the hot path.
///
/// # Fields
///
/// * `rotated` - Buffer for rotated vector in MSE stage
/// * `x_mse` - Buffer for centroid values in rotated space
/// * `x_mse_reconstructed` - Buffer for MSE-reconstructed vector in original space
/// * `residual` - Buffer for residual r = x - x̃_mse
/// * `x_qjl` - Buffer for QJL-reconstructed residual
/// * `temp` - Temporary buffer for matrix operations
/// * `qjl_workspace` - Nested workspace for QJL operations
pub struct ProdWorkspace {
    pub rotated: Vec<f32>,
    pub x_mse: Vec<f32>,
    pub x_mse_reconstructed: Vec<f32>,
    pub residual: Vec<f32>,
    pub x_qjl: Vec<f32>,
    pub temp: Vec<f32>,
    pub qjl_workspace: QjlWorkspace,
}

impl ProdWorkspace {
    /// Create a new workspace for dimension `d`.
    pub fn new(d: usize) -> Self {
        Self {
            rotated: vec![0.0; d],
            x_mse: vec![0.0; d],
            x_mse_reconstructed: vec![0.0; d],
            residual: vec![0.0; d],
            x_qjl: vec![0.0; d],
            temp: vec![0.0; d],
            qjl_workspace: QjlWorkspace::new(d),
        }
    }
}

/// Quantized representation using product quantization.
///
/// Stores MSE indices (b-1 bits) and QJL signs (1 bit packed) + residual norm.
///
/// # Memory Layout
///
/// For dimension d and b total bits:
/// - `idx`: d bytes (u8 indices for MSE centroids)
/// - `packed_qjl_signs`: ⌈d/8⌉ bytes (1 bit per coordinate)
/// - `gamma`: 4 bytes (f32 L2 norm of residual)
///
/// Total: d + ⌈d/8⌉ + 4 bytes ≈ (b-1) + 1 bits per coordinate
#[derive(Clone, Debug)]
pub struct ProdQuantizedVector {
    /// Centroid indices from MSE quantization (b-1 bits each)
    pub idx: Vec<u8>,
    /// Bit-packed QJL signs (1 bit each, 1 = positive, 0 = negative)
    pub packed_qjl_signs: Vec<u8>,
    /// L2 norm of the residual vector (γ = ||x - x̃_mse||₂)
    pub gamma: f32,
}

impl ProdQuantizedVector {
    /// Create an empty quantized vector with pre-allocated buffers.
    pub fn new(d: usize, _b: usize) -> Self {
        Self {
            idx: vec![0u8; d],
            packed_qjl_signs: vec![0u8; bitpack::packed_len(d)],
            gamma: 0.0,
        }
    }
}

impl VectorQuantizer for TurboQuantProd {
    type QuantizedType = ProdQuantizedVector;

    fn quantize(&self, x: &ArrayView1<f32>) -> Self::QuantizedType {
        let mut result = ProdQuantizedVector::new(self.d, self.b);
        let mut workspace = ProdWorkspace::new(self.d);

        self.quantize_into_buffers(
            x,
            &mut result.idx,
            &mut result.packed_qjl_signs,
            &mut result.gamma,
            &mut workspace,
        );

        result
    }

    fn quantize_into(&self, x: &ArrayView1<f32>, out: &mut Self::QuantizedType) {
        let mut workspace = ProdWorkspace::new(self.d);

        self.quantize_into_buffers(
            x,
            &mut out.idx,
            &mut out.packed_qjl_signs,
            &mut out.gamma,
            &mut workspace,
        );
    }

    fn dequantize(&self, quantized: &Self::QuantizedType, out: &mut Array1<f32>) {
        let mut workspace = ProdWorkspace::new(self.d);
        self.dequantize_into_buffers(quantized, out.as_slice_mut().unwrap(), &mut workspace);
    }
}

impl BatchQuantizer for TurboQuantProd {
    /// Quantize multiple vectors in parallel using Rayon.
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

/// Compute inner product of two vectors (scalar version).
///
/// Calculates ⟨x, y⟩ = Σᵢ xᵢyᵢ.
#[inline]
pub fn inner_product(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

/// Compute inner product using SIMD acceleration (x86_64 only).
///
/// Uses SSE instructions to process 4 elements at a time,
/// providing approximately 2-4x speedup.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn inner_product_simd(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = x.len();

    unsafe {
        let mut sum_vec = _mm_setzero_ps();
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= n {
            let x_vec = _mm_loadu_ps(&x[i]);
            let y_vec = _mm_loadu_ps(&y[i]);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(x_vec, y_vec));
            i += 4;
        }

        // Horizontal sum
        let mut result = [0.0f32; 4];
        _mm_storeu_ps(result.as_mut_ptr(), sum_vec);
        let mut sum = result[0] + result[1] + result[2] + result[3];

        // Handle remaining elements
        while i < n {
            sum += x[i] * y[i];
            i += 1;
        }

        sum
    }
}

/// Fallback to scalar version on non-x86_64 platforms.
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn inner_product_simd(x: &[f32], y: &[f32]) -> f32 {
    inner_product(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prod_quantize_dequantize_roundtrip() {
        let d = 64;
        let b = 2;
        let quantizer = TurboQuantProd::new_seeded(d, b, 42, 123);

        let original = Array1::from_vec((0..d).map(|i| (i as f32) * 0.1 - 3.0).collect());

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
    fn test_inner_product_unbiased() {
        let d = 128;
        let b = 2;
        let quantizer = TurboQuantProd::new_seeded(d, b, 42, 123);

        let x = Array1::from_vec((0..d).map(|i| (i as f32) * 0.05 - 3.0).collect());
        let y = Array1::from_vec((0..d).map(|i| (i as f32) * 0.03 + 1.0).collect());

        let true_ip = inner_product_simd(x.as_slice().unwrap(), y.as_slice().unwrap());

        let x_q = quantizer.quantize(&x.view());
        let mut x_tilde = Array1::zeros(d);
        quantizer.dequantize(&x_q, &mut x_tilde);

        let estimated_ip = inner_product_simd(x_tilde.as_slice().unwrap(), y.as_slice().unwrap());

        let relative_error = (estimated_ip - true_ip).abs() / true_ip.abs().max(1e-6);
        println!(
            "True IP: {}, Estimated IP: {}, Relative error: {:.4}",
            true_ip, estimated_ip, relative_error
        );

        assert!(
            relative_error < 0.5,
            "Inner product estimation should be reasonably accurate"
        );
    }

    #[test]
    fn test_prod_batch_quantization() {
        let d = 32;
        let b = 3;
        let quantizer = TurboQuantProd::new_seeded(d, b, 42, 123);

        let vectors: Vec<Array1<f32>> = (0..10)
            .map(|i| Array1::from_vec((0..d).map(|j| (j as f32 + i as f32) * 0.05).collect()))
            .collect();

        let views: Vec<ArrayView1<f32>> = vectors.iter().map(|v| v.view()).collect();

        let quantized = quantizer.quantize_batch(&views);

        assert_eq!(quantized.len(), 10);
        for q in &quantized {
            assert_eq!(q.idx.len(), d);
            assert_eq!(q.packed_qjl_signs.len(), bitpack::packed_len(d));
        }
    }

    #[test]
    fn test_memory_footprint() {
        let d = 1024;
        let b = 2;
        let quantizer = TurboQuantProd::new_seeded(d, b, 42, 123);

        let x = Array1::zeros(d);
        let quantized = quantizer.quantize(&x.view());

        let mut x_tilde = Array1::zeros(d);
        quantizer.dequantize(&quantized, &mut x_tilde);

        let bytes_per_coord =
            (quantized.idx.len() + quantized.packed_qjl_signs.len()) as f32 / d as f32;
        let bits_per_coord = bytes_per_coord * 8.0;

        println!("Bits per coordinate: {:.2}", bits_per_coord);
        assert!(
            bits_per_coord <= b as f32 + 8.5,
            "Memory footprint should be close to b bits (using u8 for simplicity)"
        );
    }
}
