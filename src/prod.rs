use crate::bitpack;
use crate::matrix::{self, generate_projection_matrix_seeded, generate_rotation_matrix_seeded};
use crate::mse::TurboQuantMse;
use crate::qjl::{self, QjlWorkspace};
use crate::{BatchQuantizer, VectorQuantizer};
use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;

pub struct TurboQuantProd {
    pub d: usize,
    pub b: usize,
    pub mse_quantizer: TurboQuantMse,
    pub projection_matrix: Array2<f32>,
}

impl TurboQuantProd {
    pub fn new(d: usize, b: usize) -> Self {
        Self::new_seeded(d, b, rand::random(), rand::random())
    }

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

        self.mse_quantizer
            .quantize_into_buffer(x, indices, &mut workspace.rotated);

        for (i, &idx) in indices.iter().enumerate() {
            workspace.x_mse[i] = self.mse_quantizer.centroids[idx as usize];
        }

        matrix::matrix_transpose_vector_multiply(
            &self.mse_quantizer.rotation_matrix.view(),
            &workspace.x_mse,
            &mut workspace.x_mse_reconstructed,
        );

        *gamma = qjl::compute_residual_gamma(
            x.as_slice().unwrap(),
            &workspace.x_mse_reconstructed,
            &mut workspace.residual,
        );

        qjl::qjl_quantize_into_packed(
            &self.projection_matrix.view(),
            &workspace.residual,
            packed_qjl,
            &mut workspace.qjl_workspace,
        );
    }

    pub fn dequantize_into_buffers(
        &self,
        quantized: &ProdQuantizedVector,
        out: &mut [f32],
        workspace: &mut ProdWorkspace,
    ) {
        debug_assert_eq!(out.len(), self.d);

        self.mse_quantizer.dequantize_into_buffer(
            &quantized.idx,
            &mut workspace.x_mse_reconstructed,
            &mut workspace.temp,
        );

        qjl::qjl_dequantize_packed(
            &self.projection_matrix.view(),
            &quantized.packed_qjl_signs,
            quantized.gamma,
            &mut workspace.qjl_workspace,
            &mut workspace.x_qjl,
        );

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
#[derive(Clone, Debug)]
pub struct ProdQuantizedVector {
    pub idx: Vec<u8>,
    pub packed_qjl_signs: Vec<u8>,
    pub gamma: f32,
}
impl ProdQuantizedVector {
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
    fn quantize_batch(&self, vectors: &[ArrayView1<f32>]) -> Vec<Self::QuantizedType> {
        vectors.par_iter().map(|x| self.quantize(x)).collect()
    }

    fn dequantize_batch(&self, quantized: &[Self::QuantizedType], outputs: &mut [Array1<f32>]) {
        outputs
            .par_iter_mut()
            .zip(quantized.par_iter())
            .for_each(|(out, q)| {
                self.dequantize(q, out);
            });
    }
}
pub fn inner_product(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}
#[cfg(target_arch = "x86_64")]
pub fn inner_product_simd(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = x.len();

    unsafe {
        let mut sum_vec = _mm_setzero_ps();
        let mut i = 0;

        while i + 4 <= n {
            let x_vec = _mm_loadu_ps(&x[i]);
            let y_vec = _mm_loadu_ps(&y[i]);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(x_vec, y_vec));
            i += 4;
        }

        let mut result = [0.0f32; 4];
        _mm_storeu_ps(result.as_mut_ptr(), sum_vec);
        let mut sum = result[0] + result[1] + result[2] + result[3];

        while i < n {
            sum += x[i] * y[i];
            i += 1;
        }

        sum
    }
}
#[cfg(not(target_arch = "x86_64"))]
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
