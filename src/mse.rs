use crate::centroids::{self, generate_centroids};
use crate::matrix::{self, generate_rotation_matrix_seeded};
use crate::{BatchQuantizer, VectorQuantizer};
use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;

pub struct TurboQuantMse {
    pub d: usize,
    pub b: usize,
    pub rotation_matrix: Array2<f32>,
    pub rotation_matrix_t: Array2<f32>,
    pub centroids: Vec<f32>,
}

impl TurboQuantMse {
    pub fn new(d: usize, b: usize) -> Self {
        Self::new_seeded(d, b, rand::random())
    }

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

    pub fn quantize_into_buffer(
        &self,
        x: &ArrayView1<f32>,
        indices: &mut [u8],
        rotated: &mut [f32],
    ) {
        debug_assert_eq!(x.len(), self.d);
        debug_assert_eq!(indices.len(), self.d);
        debug_assert_eq!(rotated.len(), self.d);

        matrix::matrix_vector_multiply(
            &self.rotation_matrix.view(),
            x.as_slice().unwrap(),
            rotated,
        );

        centroids::find_nearest_centroids_batch(rotated, &self.centroids, indices);
    }

    pub fn dequantize_into_buffer(&self, indices: &[u8], out: &mut [f32], temp: &mut [f32]) {
        debug_assert_eq!(indices.len(), self.d);
        debug_assert_eq!(out.len(), self.d);
        debug_assert_eq!(temp.len(), self.d);

        for (i, &idx) in indices.iter().enumerate() {
            temp[i] = self.centroids[idx as usize];
        }

        matrix::matrix_transpose_vector_multiply(&self.rotation_matrix.view(), temp, out);
    }
}

#[derive(Clone, Debug)]
pub struct MseQuantizedVector {
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
