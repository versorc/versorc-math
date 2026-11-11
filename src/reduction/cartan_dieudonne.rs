// ============================================================================
// VERSORC-MATH: CARTAN-DIEUDONNE DECOMPOSITION INTO 1-VERSORS
// License: Apache-2.0 OR MIT
//
// By the Cartan-Dieudonne theorem, every orthogonal transformation in O(n)
// can be decomposed into the product of at most n reflections (1-versores):
// Q = H_1 * H_2 * ... * H_k   (where k <= n)
// ============================================================================

use crate::versor::{HouseholderChain, HouseholderVersor};

/// Factorizes an orthogonal or near-orthogonal n x n matrix into a cascade of 1-versores.
pub struct CartanDieudonneDecomposer;

impl CartanDieudonneDecomposer {
    /// Decomposes an n x n orthogonal matrix Q (stored row-major: n * n) into a HouseholderChain.
    pub fn decompose_orthogonal(matrix: &[f32], n: usize) -> HouseholderChain {
        assert_eq!(matrix.len(), n * n, "Matrix size must equal n * n");
        let mut a = matrix.to_vec();
        let mut chain = HouseholderChain::new(n);

        let mut intermediate_versors = Vec::new();

        for j in 0..n {
            // Extract column vector below diagonal: x = a[j..n, j]
            let mut norm_sq = 0.0f32;
            for i in j..n {
                let val = a[i * n + j];
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt();
            if norm < 1e-12 {
                continue;
            }

            // Target is -sign(a[j, j]) * norm * e_j
            let alpha = if a[j * n + j] >= 0.0 { -norm } else { norm };
            let mut v = vec![0.0f32; n];
            for i in j..n {
                v[i] = a[i * n + j];
            }
            v[j] -= alpha;

            let mut v_norm_sq = 0.0f32;
            for i in j..n {
                v_norm_sq += v[i] * v[i];
            }

            if v_norm_sq > 1e-12 {
                let versor = HouseholderVersor::new(v);
                // Apply reflection to remaining columns of A from the left: A = H * A
                for col in j..n {
                    let mut col_vec = vec![0.0f32; n];
                    for row in 0..n {
                        col_vec[row] = a[row * n + col];
                    }
                    versor.reflect_in_place(&mut col_vec);
                    for row in 0..n {
                        a[row * n + col] = col_vec[row];
                    }
                }
                intermediate_versors.push(versor);
            }
        }

        // 1. Account for diagonal signs D = diag(sign(a_jj))
        // If a_jj < 0, it represents a coordinate reflection across plane x_j = 0
        for j in 0..n {
            if a[j * n + j] < 0.0 {
                let mut v_diag = vec![0.0f32; n];
                v_diag[j] = 1.0;
                chain.push(HouseholderVersor::new(v_diag));
            }
        }

        // 2. Q = H_1 * H_2 * ... * H_k * D
        // Since we already applied D, we apply H_k, ..., H_1 in reverse order
        for v in intermediate_versors.into_iter().rev() {
            chain.push(v);
        }

        chain
    }
}
