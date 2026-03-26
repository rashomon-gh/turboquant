use crate::bitpack;
use ndarray::ArrayView2;

const SQRT_PI_OVER_2: f32 = 1.25331413732;

pub struct QjlWorkspace {
    pub projected: Vec<f32>,
    pub signs: Vec<i8>,
    pub unpacked_signs: Vec<i8>,
}

impl QjlWorkspace {
    pub fn new(d: usize) -> Self {
        Self {
            projected: vec![0.0; d],
            signs: vec![0i8; d],
            unpacked_signs: vec![0i8; d],
        }
    }
}

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

        for i in 0..d {
            assert!(reconstructed[i].is_finite());
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

        for i in 0..d {
            assert!(reconstructed[i].is_finite());
        }
    }
}
