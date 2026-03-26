const SQRT_2_OVER_PI: f32 = 0.7978845608;

pub fn generate_centroids(d: usize, b: usize) -> Vec<f32> {
    let num_centroids = 1usize << b;
    let sqrt_d = (d as f32).sqrt();

    match b {
        1 => vec![-SQRT_2_OVER_PI / sqrt_d, SQRT_2_OVER_PI / sqrt_d],
        2 => {
            let c1 = 0.453 / sqrt_d;
            let c2 = 1.51 / sqrt_d;
            vec![-c2, -c1, c1, c2]
        }
        _ => {
            let mut centroids = Vec::with_capacity(num_centroids);
            let sigma = 1.0 / sqrt_d;
            let step = 2.0 * sigma * 4.0 / (num_centroids as f32 - 1.0);
            let start = -sigma * 4.0;

            for i in 0..num_centroids {
                centroids.push(start + step * (i as f32));
            }
            centroids
        }
    }
}

#[inline]
pub fn find_nearest_centroid(value: f32, centroids: &[f32]) -> u8 {
    let mut min_idx = 0;
    let mut min_dist = (value - centroids[0]).abs();

    for (i, &c) in centroids.iter().enumerate().skip(1) {
        let dist = (value - c).abs();
        if dist < min_dist {
            min_dist = dist;
            min_idx = i;
        }
    }

    min_idx as u8
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn find_nearest_centroid_simd(value: f32, centroids: &[f32]) -> u8 {
    use std::arch::x86_64::*;

    let n = centroids.len();
    if n < 4 {
        return find_nearest_centroid(value, centroids);
    }

    unsafe {
        let val_vec = _mm_set1_ps(value);
        let mut min_dist = f32::INFINITY;
        let mut min_idx = 0u8;
        let mut i = 0;

        while i + 4 <= n {
            let c_vec = _mm_loadu_ps(&centroids[i]);
            let diff = _mm_sub_ps(val_vec, c_vec);
            let abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.0), diff);
            let mut dists = [0.0f32; 4];
            _mm_storeu_ps(dists.as_mut_ptr(), abs_diff);

            for (j, &d) in dists.iter().enumerate() {
                if d < min_dist {
                    min_dist = d;
                    min_idx = (i + j) as u8;
                }
            }
            i += 4;
        }

        while i < n {
            let dist = (value - centroids[i]).abs();
            if dist < min_dist {
                min_dist = dist;
                min_idx = i as u8;
            }
            i += 1;
        }

        min_idx
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn find_nearest_centroid_simd(value: f32, centroids: &[f32]) -> u8 {
    find_nearest_centroid(value, centroids)
}

pub fn find_nearest_centroids_batch(values: &[f32], centroids: &[f32], indices: &mut [u8]) {
    debug_assert_eq!(values.len(), indices.len());

    #[cfg(target_arch = "x86_64")]
    {
        for (i, &v) in values.iter().enumerate() {
            indices[i] = find_nearest_centroid_simd(v, centroids);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        for (i, &v) in values.iter().enumerate() {
            indices[i] = find_nearest_centroid(v, centroids);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_centroids_1bit() {
        let centroids = generate_centroids(1024, 1);
        assert_eq!(centroids.len(), 2);

        let expected = SQRT_2_OVER_PI / (1024.0_f32).sqrt();
        assert_relative_eq!(centroids[0], -expected, epsilon = 1e-6);
        assert_relative_eq!(centroids[1], expected, epsilon = 1e-6);
    }

    #[test]
    fn test_centroids_2bit() {
        let centroids = generate_centroids(1024, 2);
        assert_eq!(centroids.len(), 4);

        let sqrt_d = (1024.0_f32).sqrt();
        assert_relative_eq!(centroids[0], -1.51 / sqrt_d, epsilon = 1e-6);
        assert_relative_eq!(centroids[1], -0.453 / sqrt_d, epsilon = 1e-6);
        assert_relative_eq!(centroids[2], 0.453 / sqrt_d, epsilon = 1e-6);
        assert_relative_eq!(centroids[3], 1.51 / sqrt_d, epsilon = 1e-6);
    }

    #[test]
    fn test_find_nearest_centroid() {
        let centroids = vec![-1.0, 0.0, 1.0];

        assert_eq!(find_nearest_centroid(-1.5, &centroids), 0);
        assert_eq!(find_nearest_centroid(-0.3, &centroids), 1);
        assert_eq!(find_nearest_centroid(0.8, &centroids), 2);
    }

    #[test]
    fn test_find_nearest_centroids_batch() {
        let centroids = vec![-1.0, 0.0, 1.0];
        let values = vec![-1.5, -0.3, 0.8, 0.1];
        let mut indices = vec![0u8; 4];

        find_nearest_centroids_batch(&values, &centroids, &mut indices);

        assert_eq!(indices, vec![0, 1, 2, 1]);
    }
}
