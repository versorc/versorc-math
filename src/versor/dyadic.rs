// ============================================================================
// VERSORC-MATH: DYADIC VERSORS & FAST WALSH-HADAMARD TRANSFORM (FWHT)
// License: Apache-2.0
//
// Fast Walsh-Hadamard Transform in O(N log N) additions/subtractions with
// Rademacher sign scramblers S in {-1, +1}^N.
// Orthonormally smears activation and weight outliers into an isotropic
// distribution with exact isometry:
// R = (1 / sqrt(N)) * H_N * S
// R^T * R = I
// ============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use super::VersorAction;

/// Dyadic Walsh-Hadamard Versor with Rademacher sign scrambling.
#[derive(Debug, Clone)]
pub struct DyadicWalshVersor {
    dim: usize,
    block_size: usize,
    rademacher_signs: Vec<f32>,
    norm_factor: f32,
}

impl DyadicWalshVersor {
    /// Constructs a dyadic versor for dimension `dim`.
    /// Operates in block-diagonal Walsh-Hadamard blocks if `dim` is not a power of 2.
    pub fn new(dim: usize, seed: u64) -> Self {
        let block_size = if dim.is_power_of_two() {
            dim
        } else {
            let mut b = 256;
            while b >= 64 {
                if dim % b == 0 {
                    break;
                }
                b /= 2;
            }
            assert!(b >= 64 && dim % b == 0, "Dimension {} must be divisible by at least 64", dim);
            b
        };

        let mut signs = Vec::with_capacity(dim);
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        for _ in 0..dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bit = (state >> 33) & 1;
            signs.push(if bit == 0 { 1.0f32 } else { -1.0f32 });
        }

        let norm_factor = 1.0 / (block_size as f32).sqrt();

        Self {
            dim,
            block_size,
            rademacher_signs: signs,
            norm_factor,
        }
    }

    #[inline(always)]
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Evaluates the Fast Walsh-Hadamard Transform (FWHT) in O(N log N) in-place.
    pub fn fwht_in_place(data: &mut [f32]) {
        let n = data.len();
        assert!(n.is_power_of_two(), "Slice length must be a power of 2");

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && n >= 8 {
                unsafe {
                    Self::fwht_avx2(data);
                    return;
                }
            }
        }

        Self::fwht_scalar(data);
    }

    /// Scalar reference FWHT.
    #[inline]
    pub fn fwht_scalar(data: &mut [f32]) {
        let n = data.len();
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in i..i + h {
                    let u = data[j];
                    let v = data[j + h];
                    data[j] = u + v;
                    data[j + h] = u - v;
                }
            }
            h *= 2;
        }
    }

    /// AVX2-accelerated in-place FWHT.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn fwht_avx2(data: &mut [f32]) {
        let n = data.len();
        let mut h = 1;
        while h < 8 && h < n {
            for i in (0..n).step_by(h * 2) {
                for j in i..i + h {
                    let u = data[j];
                    let v = data[j + h];
                    data[j] = u + v;
                    data[j + h] = u - v;
                }
            }
            h *= 2;
        }

        while h < n {
            let mut i = 0;
            while i < n {
                let mut j = 0;
                while j + 8 <= h {
                    let ptr_u = data.as_mut_ptr().add(i + j);
                    let ptr_v = data.as_mut_ptr().add(i + j + h);
                    let u = _mm256_loadu_ps(ptr_u);
                    let v = _mm256_loadu_ps(ptr_v);
                    _mm256_storeu_ps(ptr_u, _mm256_add_ps(u, v));
                    _mm256_storeu_ps(ptr_v, _mm256_sub_ps(u, v));
                    j += 8;
                }
                while j < h {
                    let u = data[i + j];
                    let v = data[i + j + h];
                    data[i + j] = u + v;
                    data[i + j + h] = u - v;
                    j += 1;
                }
                i += h * 2;
            }
            h *= 2;
        }
    }

    /// Applies the full randomized dyadic versor rotation in-place:
    /// x = (1 / sqrt(N)) * H_N * (S * x)
    pub fn apply_in_place(&self, x: &mut [f32]) {
        assert_eq!(x.len(), self.dim, "Input slice dimension mismatch");

        // 1. Pointwise Rademacher sign multiplication
        for (val, &sign) in x.iter_mut().zip(&self.rademacher_signs) {
            *val *= sign;
        }

        // 2. Block-wise Walsh-Hadamard transforms
        for chunk in x.chunks_exact_mut(self.block_size) {
            Self::fwht_in_place(chunk);
            for val in chunk.iter_mut() {
                *val *= self.norm_factor;
            }
        }
    }

    /// Inverts the randomized dyadic versor rotation in-place:
    /// Since R is an orthonormal isometry (R^T * R = I), R^-1 = S * (1 / sqrt(N)) * H_N.
    pub fn invert_in_place(&self, x: &mut [f32]) {
        assert_eq!(x.len(), self.dim, "Input slice dimension mismatch");

        // 1. Block-wise Walsh-Hadamard transforms (H_N is symmetric: H_N^T = H_N)
        for chunk in x.chunks_exact_mut(self.block_size) {
            Self::fwht_in_place(chunk);
            for val in chunk.iter_mut() {
                *val *= self.norm_factor;
            }
        }

        // 2. Pointwise Rademacher sign multiplication (S^-1 = S since s_i in {-1, +1})
        for (val, &sign) in x.iter_mut().zip(&self.rademacher_signs) {
            *val *= sign;
        }
    }
}

impl VersorAction<[f32]> for DyadicWalshVersor {
    #[inline]
    fn transform_in_place(&self, target: &mut [f32]) {
        self.apply_in_place(target);
    }
}
