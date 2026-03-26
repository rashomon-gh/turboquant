//! # Matrix Generation and Operations
//!
//! This module provides utilities for generating random matrices and performing
//! matrix-vector operations with SIMD optimization.
//!
//! ## Matrix Types
//!
//! ### Rotation Matrices (Π)
//! Random orthogonal matrices generated via QR decomposition of Gaussian random matrices.
//! These are used in TurboQuantMse to induce a uniform distribution on rotated coordinates.
//!
//! ### Projection Matrices (S)
//! Random matrices with i.i.d. N(0,1) entries, used in the QJL transform for
//! dimensionality reduction with unbiased inner product estimation.
//!
//! ## SIMD Optimization
//!
//! Matrix-vector multiplication is SIMD-accelerated on x86_64 platforms using SSE instructions,
//! providing approximately 2-4x speedup over scalar code.
//!
//! ## Example
//!
//! ```rust
//! use turboquant::matrix::{generate_rotation_matrix_seeded, matrix_vector_multiply};
//! use ndarray::Array2;
//!
//! // Generate a reproducible rotation matrix
//! let rotation = generate_rotation_matrix_seeded(128, 42);
//!
//! // Multiply matrix by vector
//! let vec = vec![0.1; 128];
//! let mut out = vec![0.0; 128];
//! matrix_vector_multiply(&rotation.view(), &vec, &mut out);
//! ```

use faer::Mat;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// Generate a random orthogonal rotation matrix using QR decomposition.
///
/// Creates a d×d matrix by:
/// 1. Sampling a d×d matrix with i.i.d. N(0,1) entries
/// 2. Performing QR decomposition
/// 3. Extracting the orthogonal Q matrix
///
/// # Arguments
///
/// * `d` - Dimension of the square rotation matrix
/// * `rng` - Random number generator
///
/// # Returns
///
/// An orthogonal matrix Q such that Q^T Q = I.
///
/// # Example
///
/// ```rust
/// use turboquant::matrix::generate_rotation_matrix_with_rng;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let rotation = generate_rotation_matrix_with_rng(64, &mut rng);
/// ```
pub fn generate_rotation_matrix_with_rng(d: usize, rng: &mut StdRng) -> Array2<f32> {
    // Step 1: Generate a d×d Gaussian random matrix
    let mut gaussian: Mat<f32> = Mat::zeros(d, d);
    for i in 0..d {
        for j in 0..d {
            gaussian[(i, j)] = StandardNormal.sample(rng);
        }
    }

    // Step 2: Perform QR decomposition
    let qr = gaussian.qr();
    let q = qr.compute_q();

    // Step 3: Convert faer matrix to ndarray
    let mut result = Array2::<f32>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            result[[i, j]] = q[(i, j)];
        }
    }
    result
}

/// Generate a random rotation matrix using OS-provided entropy.
///
/// # Arguments
///
/// * `d` - Dimension of the rotation matrix
///
/// # Returns
///
/// A random orthogonal matrix.
pub fn generate_rotation_matrix(d: usize) -> Array2<f32> {
    let mut rng = StdRng::from_os_rng();
    generate_rotation_matrix_with_rng(d, &mut rng)
}

/// Generate a reproducible rotation matrix from a seed.
///
/// # Arguments
///
/// * `d` - Dimension of the rotation matrix
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// A deterministic orthogonal matrix based on the seed.
///
/// # Example
///
/// ```rust
/// use turboquant::matrix::generate_rotation_matrix_seeded;
///
/// let rotation = generate_rotation_matrix_seeded(128, 42);
/// ```
pub fn generate_rotation_matrix_seeded(d: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    generate_rotation_matrix_with_rng(d, &mut rng)
}

/// Generate a random projection matrix with i.i.d. N(0,1) entries.
///
/// This matrix is used in the QJL (Quantized Johnson-Lindenstrauss) transform
/// for dimensionality reduction with inner product preservation.
///
/// # Arguments
///
/// * `d` - Dimension of the square projection matrix
/// * `rng` - Random number generator
///
/// # Returns
///
/// A d×d matrix with N(0,1) entries.
pub fn generate_projection_matrix_with_rng(d: usize, rng: &mut StdRng) -> Array2<f32> {
    let mut result = Array2::<f32>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            result[[i, j]] = StandardNormal.sample(rng);
        }
    }
    result
}

/// Generate a random projection matrix using OS-provided entropy.
pub fn generate_projection_matrix(d: usize) -> Array2<f32> {
    let mut rng = StdRng::from_os_rng();
    generate_projection_matrix_with_rng(d, &mut rng)
}

/// Generate a reproducible projection matrix from a seed.
pub fn generate_projection_matrix_seeded(d: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    generate_projection_matrix_with_rng(d, &mut rng)
}

/// Matrix-vector multiplication (scalar version).
///
/// Computes out = matrix × vec for a dense matrix.
///
/// # Arguments
///
/// * `matrix` - Input matrix (rows × cols)
/// * `vec` - Input vector (length = cols)
/// * `out` - Output vector (length = rows, pre-allocated)
///
/// # Safety
///
/// The caller must ensure `vec.len() == matrix.ncols()` and `out.len() == matrix.nrows()`.
pub fn matrix_vector_multiply(matrix: &ndarray::ArrayView2<f32>, vec: &[f32], out: &mut [f32]) {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    debug_assert_eq!(vec.len(), cols);
    debug_assert_eq!(out.len(), rows);

    for i in 0..rows {
        let mut sum = 0.0f32;
        for j in 0..cols {
            sum += matrix[[i, j]] * vec[j];
        }
        out[i] = sum;
    }
}

/// Matrix-transpose-vector multiplication.
///
/// Computes out = matrix^T × vec, which is equivalent to treating columns of `matrix`
/// as the rows for the multiplication.
///
/// # Arguments
///
/// * `matrix` - Input matrix (rows × cols)
/// * `vec` - Input vector (length = rows)
/// * `out` - Output vector (length = cols, pre-allocated)
///
/// # Example
///
/// ```rust
/// use turboquant::matrix::{generate_rotation_matrix_seeded, matrix_transpose_vector_multiply};
///
/// let rotation = generate_rotation_matrix_seeded(64, 42);
/// let rotated = vec![0.1; 64];
/// let mut original_space = vec![0.0; 64];
/// matrix_transpose_vector_multiply(&rotation.view(), &rotated, &mut original_space);
/// ```
pub fn matrix_transpose_vector_multiply(
    matrix: &ndarray::ArrayView2<f32>,
    vec: &[f32],
    out: &mut [f32],
) {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    debug_assert_eq!(vec.len(), rows);
    debug_assert_eq!(out.len(), cols);

    for j in 0..cols {
        let mut sum = 0.0f32;
        for i in 0..rows {
            sum += matrix[[i, j]] * vec[i];
        }
        out[j] = sum;
    }
}

/// SIMD-accelerated matrix-vector multiplication (x86_64 only).
///
/// Uses SSE instructions to process 4 elements at a time, providing
/// significant speedup for large matrices.
///
/// # Performance
///
/// Approximately 2-4x faster than the scalar version for typical dimensions.
///
/// # Arguments
///
/// * `matrix` - Input matrix
/// * `vec` - Input vector
/// * `out` - Output vector (pre-allocated)
#[cfg(target_arch = "x86_64")]
pub fn matrix_vector_multiply_simd(
    matrix: &ndarray::ArrayView2<f32>,
    vec: &[f32],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    let rows = matrix.nrows();
    let cols = matrix.ncols();
    debug_assert_eq!(vec.len(), cols);
    debug_assert_eq!(out.len(), rows);

    unsafe {
        #[allow(clippy::needless_range_loop)]
        for i in 0..rows {
            let row = matrix.row(i);
            let mut sum = _mm_setzero_ps();
            let mut j = 0;

            // Process 4 elements at a time using SSE
            while j + 4 <= cols {
                let m_vec = _mm_loadu_ps(&row[j]);
                let v_vec = _mm_loadu_ps(&vec[j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(m_vec, v_vec));
                j += 4;
            }

            // Horizontal sum of the SIMD register
            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), sum);
            let mut total = result[0] + result[1] + result[2] + result[3];

            // Handle remaining elements
            while j < cols {
                total += row[j] * vec[j];
                j += 1;
            }

            out[i] = total;
        }
    }
}

/// Fallback to scalar version on non-x86_64 platforms.
#[cfg(not(target_arch = "x86_64"))]
pub fn matrix_vector_multiply_simd(
    matrix: &ndarray::ArrayView2<f32>,
    vec: &[f32],
    out: &mut [f32],
) {
    matrix_vector_multiply(matrix, vec, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rotation_matrix_is_orthogonal() {
        let d = 16;
        let rotation = generate_rotation_matrix(d);

        // Verify Q^T Q = I
        let product = rotation.t().dot(&rotation);
        for i in 0..d {
            for j in 0..d {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(product[[i, j]], expected, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_seeded_rotation_reproducible() {
        let d = 8;
        let r1 = generate_rotation_matrix_seeded(d, 42);
        let r2 = generate_rotation_matrix_seeded(d, 42);

        for i in 0..d {
            for j in 0..d {
                assert_relative_eq!(r1[[i, j]], r2[[i, j]]);
            }
        }
    }
}
