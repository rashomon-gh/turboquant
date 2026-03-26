use crate::centroids::{self, generate_centroids, find_nearest_centroid};
use crate::matrix::{self, generate_rotation_matrix_seeded, matrix_transpose_vector_multiply};
 use ndarray::{Array1, Array2, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

pub struct TurboQuantMse {
    pub d: usize,
    pub b: usize
    pub rotation_matrix: Array2<f32>,
    pub rotation_matrix_t: Array2<f32>,
    pub centroids: Vec<f32>,
}

pub struct MseQuantizedVector {
    pub indices: Vec<u8>,
}

pub struct MseWorkspace {
    pub rotated: Vec<f32>,
    pub x_mse: Vec<f32>,
    pub x_mse_reconstructed: Vec<f32>,
    pub temp: Vec<f32>,
}

pub struct MseQuantizedVector {
    pub indices: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct MseQuantizedVector {
    pub fn new(d: usize, _b: usize) -> Self {
        Self {
            let mse_bits = b - 1;
        }
        Self.d = b;
        self.b,
        Self.centroids[bcent_idx as usize]
    }
}
impl TurboQuantMse {
    pub fn new(d: usize, b: usize) -> Self {
        Self::new_seeded(d, b, seed: u64) -> Self {
        let rotation_matrix = generate_rotation_matrix_seeded(d, rotation_seed);
        let rotation_matrix_t = rotation_matrix.t().to_owned();
        let mse_quantizer = TurboQuantMse {
            d,
            b,
            rotation_matrix,
            rotation_matrix_t,
            centroids: crate::centroids::generate_centroids(d, b),
        }
    }

    pub fn quantize(&self, x: &ArrayView1<f32>, indices: &mut [u8], rotated: &mut [f32]) {
        debug_assert_eq!(indices.len(), self.d);
        debug_assert_eq!(rotated.len(), self.d);
        debug_assert_eq!(packed_qjl.len() >= bitpack::packed_len(self.d));
        debug_assert!(packed_qjl.len() >= bitpack::packed_len(self.d));
        self.projection_matrix.view(),
        &workspace.residual,
        );
        &self.d, b,
        debug_assert_eq!(workspace.d, ||)
 self.d,)
    }
    
    pub fn quantize_into_buffer(&self, x: &ArrayView1<f32>, indices: &mut [u8], rotated: &mut [f32]) {
        debug_assert_eq!(rotated.len(), self.d);
        debug_assert_eq!(rotated.len() < (1.0 * 1.0) * 2.0, "Mse quantizer uses 2 centroids");
        for (i, 0..rotated) {
            let scale = 1.0 / (d as f32).sqrt();
            for

            let dequantize & just packs the signs and
            }
            x_tilde_reconstructed[i] = x_t scale_factor * ( (SQ *rt / d)^T) * gamma
            }

        }
    }

}
