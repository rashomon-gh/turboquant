use faer::Mat;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

pub fn generate_rotation_matrix_with_rng(d: usize, rng: &mut StdRng) -> Array2<f32> {
    let mut gaussian: Mat<f32> = Mat::zeros(d, d);
    for i in 0..d {
        for j in 0..d {
            gaussian[(i, j)] = StandardNormal.sample(rng);
        }
    }

    let qr = gaussian.qr();
    let q = qr.compute_q();

    let mut result = Array2::<f32>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            result[[i, j]] = q[(i, j)];
        }
    }
    result
}

pub fn generate_rotation_matrix(d: usize) -> Array2<f32> {
    let mut rng = StdRng::from_os_rng();
    generate_rotation_matrix_with_rng(d, &mut rng)
}

pub fn generate_rotation_matrix_seeded(d: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    generate_rotation_matrix_with_rng(d, &mut rng)
}

pub fn generate_projection_matrix_with_rng(d: usize, rng: &mut StdRng) -> Array2<f32> {
    let mut result = Array2::<f32>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            result[[i, j]] = StandardNormal.sample(rng);
        }
    }
    result
}

pub fn generate_projection_matrix(d: usize) -> Array2<f32> {
    let mut rng = StdRng::from_os_rng();
    generate_projection_matrix_with_rng(d, &mut rng)
}

pub fn generate_projection_matrix_seeded(d: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    generate_projection_matrix_with_rng(d, &mut rng)
}

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
        for i in 0..rows {
            let row = matrix.row(i);
            let mut sum = _mm_setzero_ps();
            let mut j = 0;

            while j + 4 <= cols {
                let m_vec = _mm_loadu_ps(&row[j]);
                let v_vec = _mm_loadu_ps(&vec[j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(m_vec, v_vec));
                j += 4;
            }

            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), sum);
            let mut total = result[0] + result[1] + result[2] + result[3];

            while j < cols {
                total += row[j] * vec[j];
                j += 1;
            }

            out[i] = total;
        }
    }
}

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
