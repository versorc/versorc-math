// ============================================================================
// VERSORC-MATH: CARTAN-DIEUDONNE DECOMPOSITION INTO 1-VERSORS
// License: Apache-2.0
//
// By the Cartan-Dieudonne theorem, every orthogonal transformation in O(n)
// can be decomposed into the product of at most n reflections (1-versores):
// Q = H_1 * H_2 * ... * H_k   (where k <= n)
// ============================================================================

use crate::versor::{HouseholderChain, HouseholderVersor};

/// Factorizes an orthogonal or near-orthogonal n x n matrix into a cascade of 1-versores.
pub struct CartanDieudonneDecomposer;

impl CartanDieudonneDecomposer {
    /// Decomposes an n x n orthogonal matrix Q (stored row-major: n * n) into a HouseholderChain
    /// with strictly at most `n` 1-versor reflections (Cartan-Dieudonné theorem).
    pub fn decompose_orthogonal(matrix: &[f32], n: usize) -> HouseholderChain {
        assert_eq!(matrix.len(), n * n, "Matrix size must equal n * n");
        let mut a = matrix.to_vec();
        let mut chain = HouseholderChain::new(n);

        let mut intermediate_versors = Vec::with_capacity(n);

        for j in 0..n {
            let mut subdiag_norm_sq = 0.0f32;
            for i in (j + 1)..n {
                let val = a[i * n + j];
                subdiag_norm_sq += val * val;
            }
            let diag = a[j * n + j];
            let norm = (diag * diag + subdiag_norm_sq).sqrt();
            if norm < 1e-12 {
                continue;
            }

            // Column j is already aligned along +e_j -> no reflection needed
            if diag >= 0.0 && subdiag_norm_sq <= 1e-12 {
                continue;
            }

            // Reflect column j onto +norm * e_j in a single Householder step so the
            // remaining orthogonal matrix converges directly to +I_n without a trailing sign pass.
            let mut v = vec![0.0f32; n];
            v[j] = if diag > 0.0 {
                // Cancellation-free Parlett identity: diag - norm = -subdiag / (diag + norm)
                -subdiag_norm_sq / (diag + norm)
            } else {
                diag - norm
            };
            for i in (j + 1)..n {
                v[i] = a[i * n + j];
            }

            let v_norm_sq: f32 = v[j..n].iter().map(|&x| x * x).sum();
            if v_norm_sq > 1e-12 {
                let versor = HouseholderVersor::new(v);
                // Apply reflection from the left: A = H * A
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

        // Q = H_0 * H_1 * ... * H_{k-1} => apply in reverse order H_{k-1} .. H_0 on vector x
        for v in intermediate_versors.into_iter().rev() {
            chain.push(v);
        }

        chain
    }
}
