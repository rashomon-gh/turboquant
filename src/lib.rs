pub mod bitpack;
pub mod centroids;
pub mod matrix;
pub mod mse;
pub mod prod;
pub mod qjl;

use ndarray::{Array1, ArrayView1};

pub trait VectorQuantizer {
    type QuantizedType;

    fn quantize(&self, x: &ArrayView1<f32>) -> Self::QuantizedType;

    fn quantize_into(&self, x: &ArrayView1<f32>, out: &mut Self::QuantizedType);

    fn dequantize(&self, quantized: &Self::QuantizedType, out: &mut Array1<f32>);
}

pub trait BatchQuantizer: VectorQuantizer {
    fn quantize_batch(&self, vectors: &[ArrayView1<f32>]) -> Vec<Self::QuantizedType>;
    fn dequantize_batch(&self, quantized: &[Self::QuantizedType], outputs: &mut [Array1<f32>]);
}
