//! # Quantized Johnson-Lindenstrauss (QJL) Transform
//!
//! This module implements the Quantized Johnson-Lindenstrauss transform, which provides
//! unbiased inner product estimation through random projection and 1-bit quantization.
//!
//! ## Theory
//!
//! The QJL transform works by:
//! 1. Projecting the residual vector r onto a random subspace using matrix S
//! 2. Quantizing each projected coordinate to ±1 (1-bit quantization)
//! 3. Storing only the signs and the L2 norm (γ) of the residual
//!
//! For reconstruction, we use the property that E[sign(S·r)] provides an unbiased estimate
//! of the original residual, scaled by √(π/2)/d.
//!
//! ## Key Properties
//!
//! - **Unbiased Estimation**: E[⟨y, x̃⟩] = ⟨y, x⟩ for any query vector y
//! - **Memory Efficiency**: Only 1 bit per coordinate + 1 float (γ) for the residual
//! - **Fast Computation**: Simple matrix multiplication and sign extraction
//!
//! ## Example
//!
//! ```rust
//! use turboquant::qjl::{qjl_quantize, qjl_dequantize, QjlWorkspace, compute_residual_gamma};
//! use turboquant::matrix::generate_projection_matrix_seeded;
//!
//! let d = 16;
//! let s_matrix = generate_projection_matrix_seeded(d, 42);
//!
//! // Original vector and its MSE approximation
//! let x = vec![0.1, -0.5, 0.3, 0.2, 0.0, 0.4, -0.1, 0.3,
//!              0.2, -0.2, 0.1, 0.0, 0.3, -0.3, 0.2, 0.1];
//! let x_mse = vec![0.09, -0.48, 0.29, 0.21, 0.01, 0.39, -0.11, 0.28,
//!                  0.18, -0.21, 0.12, -0.01, 0.31, -0.29, 0.19, 0.11];
//!
//! // Compute residual
//! let mut residual = vec![0.0; d];
//! let gamma = compute_residual_gamma(&x, &x_mse, &mut residual);
//!
//! // Quantize with QJL
//! let mut workspace = QjlWorkspace::new(d);
//! qjl_quantize(&s_matrix.view(), &residual, &mut workspace);
//!
//! // Reconstruct
//! let mut reconstructed = vec![0.0; d];
//! qjl_dequantize(&s_matrix.view(), &workspace.signs, gamma, &mut reconstructed);
//! ```

use crate::bitpack;
use ndarray::ArrayView2;

/// √(π/2) constant used for unbiased reconstruction.
///
/// This factor corrects the bias introduced by 1-bit quantization,
/// ensuring E[reconstruction] = original residual.
const SQRT_PI_OVER_2: f32 = 1.253_314_1;

/// Workspace for QJL transform operations.
///
/// Pre-allocated buffers to avoid allocations in the hot path.
/// Should be reused across multiple quantization operations.
///
/// # Fields
///
/// * `projected` - Buffer for S·r projection result
/// * `signs` - Quantized signs (-1 or +1)
/// * `unpacked_signs` - Temporary buffer for unpacking bit-packed signs
pub struct QjlWorkspace {
    pub projected: Vec<f32>,
    pub signs: Vec<i8>,
    pub unpacked_signs: Vec<i8>,
}

impl QjlWorkspace {
    /// Create a new workspace for dimension `d`.
    pub fn new(d: usize) -> Self {
        Self {
            projected: vec![0.0; d],
            signs: vec![0i8; d],
            unpacked_signs: vec![0i8; d],
        }
    }
}

/// Quantize a residual vector using the QJL transform.
///
/// Computes sign(S·r) where S is the projection matrix and r is the residual.
///
/// # Arguments
///
/// * `s_matrix` - Random projection matrix (d×d with N(0,1) entries)
/// * `residual` - Residual vector r = x - x̃_mse
/// * `workspace` - Pre-allocated workspace buffers
///
/// # Output
///
/// The quantized signs are stored in `workspace.signs`.
#[inline]
pub fn qjl_quantize(s_matrix: &ArrayView2<f32>, residual: &[f32], workspace: &mut QjlWorkspace) {
    let d = residual.len();
    debug_assert_eq!(workspace.projected.len(), d);
    debug_assert_eq!(workspace.signs.len(), d);

    for i in 0..d {
        let mut sum = 0.0f32;
        for j in 0..d {
            sum += s_matrix[[i, j]] * residual[j];
        }
        workspace.projected[i] = sum;
        workspace.signs[i] = if sum >= 0.0 { 1 } else { -1 };
    }
}

/// Quantize a residual vector and pack the signs into bits.
///
/// Combines quantization with bit-packing for maximum memory efficiency.
///
/// # Arguments
///
/// * `s_matrix` - Random projection matrix
/// * `residual` - Residual vector to quantize
/// * `packed_out` - Output buffer for packed bits (must be at least `packed_len(d)` bytes)
/// * `workspace` - Pre-allocated workspace
#[inline]
pub fn qjl_quantize_into_packed(
    s_matrix: &ArrayView2<f32>,
    residual: &[f32],
    packed_out: &mut [u8],
    workspace: &mut QjlWorkspace,
) {
    qjl_quantize(s_matrix, residual, workspace);
    bitpack::pack_bits_into(&workspace.signs, packed_out);
}

/// Compute the residual vector and its L2 norm (γ).
///
/// Calculates r = x - x̃ and returns ||r||₂.
///
/// # Arguments
///
/// * `x` - Original vector
/// * `x_tilde` - Approximate/reconstructed vector
/// * `residual` - Output buffer for the residual (pre-allocated)
///
/// # Returns
///
/// The L2 norm of the residual: √(Σᵢ(xᵢ - x̃ᵢ)²)
#[inline]
pub fn compute_residual_gamma(x: &[f32], x_tilde: &[f32], residual: &mut [f32]) -> f32 {
    debug_assert_eq!(x.len(), x_tilde.len());
    debug_assert_eq!(x.len(), residual.len());

    let mut gamma_sq = 0.0f32;
    for i in 0..x.len() {
        residual[i] = x[i] - x_tilde[i];
        gamma_sq += residual[i] * residual[i];
    }
    gamma_sq.sqrt()
}

/// Reconstruct a residual vector from quantized QJL signs.
///
/// Computes (√(π/2)/d) × γ × S^T × signs, which provides an unbiased estimate
/// of the original residual vector.
///
/// # Arguments
///
/// * `s_matrix` - Random projection matrix (same as used in quantization)
/// * `qjl_signs` - Quantized signs (-1 or +1 for each coordinate)
/// * `gamma` - L2 norm of the original residual
/// * `out` - Output buffer for the reconstructed residual
///
/// # Mathematical Background
///
/// For a residual r, we have:
/// - Quantization: signs = sign(S·r)
/// - Dequantization: r̂ = (√(π/2)/d) × γ × S^T × signs
/// - Property: E[r̂] = r (unbiased estimator)
#[inline]
pub fn qjl_dequantize(s_matrix: &ArrayView2<f32>, qjl_signs: &[i8], gamma: f32, out: &mut [f32]) {
    let d = out.len();
    debug_assert_eq!(qjl_signs.len(), d);

    let scale_factor = (SQRT_PI_OVER_2 / (d as f32)) * gamma;

    for j in 0..d {
        let mut sum = 0.0f32;
        for i in 0..d {
            sum += s_matrix[[i, j]] * (qjl_signs[i] as f32);
        }
        out[j] = sum * scale_factor;
    }
}

/// Reconstruct a residual from bit-packed QJL signs.
///
/// Unpacks the bit-packed signs and then applies standard dequantization.
///
/// # Arguments
///
/// * `s_matrix` - Random projection matrix
/// * `packed_signs` - Bit-packed quantized signs
/// * `gamma` - L2 norm of the original residual
/// * `workspace` - Workspace for unpacking
/// * `out` - Output buffer for the reconstructed residual
#[inline]
pub fn qjl_dequantize_packed(
    s_matrix: &ArrayView2<f32>,
    packed_signs: &[u8],
    gamma: f32,
    workspace: &mut QjlWorkspace,
    out: &mut [f32],
) {
    let d = out.len();
    bitpack::unpack_bits(packed_signs, d, &mut workspace.unpacked_signs);
    qjl_dequantize(s_matrix, &workspace.unpacked_signs, gamma, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_qjl_roundtrip() {
        let d = 16;
        let mut s_matrix = Array2::<f32>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                s_matrix[[i, j]] = if i == j { 1.0 } else { 0.5 };
            }
        }

        let residual: Vec<f32> = (0..d).map(|i| (i as f32) * 0.1).collect();

        let mut workspace = QjlWorkspace::new(d);
        qjl_quantize(&s_matrix.view(), &residual, &mut workspace);

        let gamma = residual.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut reconstructed = vec![0.0; d];
        qjl_dequantize(
            &s_matrix.view(),
            &workspace.signs,
            gamma,
            &mut reconstructed,
        );

        for val in &reconstructed {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_packed_roundtrip() {
        let d = 16;
        let mut s_matrix = Array2::<f32>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                s_matrix[[i, j]] = if i == j { 1.0 } else { 0.3 };
            }
        }

        let residual: Vec<f32> = (0..d).map(|i| (i as f32) * 0.1 - 0.5).collect();

        let mut workspace = QjlWorkspace::new(d);
        let mut packed = vec![0u8; bitpack::packed_len(d)];

        qjl_quantize_into_packed(&s_matrix.view(), &residual, &mut packed, &mut workspace);

        let gamma = residual.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut reconstructed = vec![0.0; d];
        qjl_dequantize_packed(
            &s_matrix.view(),
            &packed,
            gamma,
            &mut workspace,
            &mut reconstructed,
        );

        for val in &reconstructed {
            assert!(val.is_finite());
        }
    }
}
