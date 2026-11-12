// ============================================================================
// VERSORC-MATH: 1-VERSORS & HOUSEHOLDER REFLECTION CASCADES
// License: Apache-2.0
// ============================================================================

use super::VersorAction;

/// 1-Versor representing a Householder hyperplane reflection in R^n:
/// H_v(x) = -v * x * v^-1 = x - 2 * (v . x / ||v||^2) * v
#[derive(Debug, Clone, PartialEq)]
pub struct HouseholderVersor {
    /// Reflection normal vector v
    pub v: Vec<f32>,
    /// Precomputed factor: 2.0 / ||v||^2
    pub tau: f32,
}

impl HouseholderVersor {
    /// Constructs a 1-versor reflection operator from normal vector v.
    pub fn new(v: Vec<f32>) -> Self {
        let mut norm_sq = 0.0f32;
        for &x in &v {
            norm_sq += x * x;
        }
        assert!(norm_sq > 1e-15, "Householder normal vector cannot have zero norm");
        let tau = 2.0 / norm_sq;
        Self { v, tau }
    }

    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.v.len()
    }

    /// Reflects an arbitrary vector x in-place in O(n) FLOPs:
    /// x = x - tau * (v . x) * v
    #[inline]
    pub fn reflect_in_place(&self, x: &mut [f32]) {
        assert_eq!(x.len(), self.v.len(), "Vector dimension mismatch");
        let mut dot = 0.0f32;
        for i in 0..x.len() {
            dot += self.v[i] * x[i];
        }
        let alpha = dot * self.tau;
        for i in 0..x.len() {
            x[i] -= alpha * self.v[i];
        }
    }
}

impl VersorAction<[f32]> for HouseholderVersor {
    #[inline]
    fn transform_in_place(&self, target: &mut [f32]) {
        self.reflect_in_place(target);
    }
}

/// Cascade of k sequential 1-versores (Householder reflections).
/// Represents an arbitrary orthogonal transformation Q = H_1 * H_2 * ... * H_k
/// in O(k * n) time and O(k * n) space, bypassing O(n^2) dense matrix limits.
#[derive(Debug, Clone, PartialEq)]
pub struct HouseholderChain {
    pub dim: usize,
    pub versors: Vec<HouseholderVersor>,
}

impl HouseholderChain {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            versors: Vec::new(),
        }
    }

    /// Appends a 1-versor reflection to the cascade.
    pub fn push(&mut self, versor: HouseholderVersor) {
        assert_eq!(versor.dim(), self.dim, "Versor dimension mismatch");
        self.versors.push(versor);
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.versors.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.versors.is_empty()
    }

    /// Evaluates the full versor cascade on vector x in O(k * n) FLOPs:
    /// y = H_k * ... * H_2 * H_1 * x
    #[inline]
    pub fn apply_in_place(&self, x: &mut [f32]) {
        assert_eq!(x.len(), self.dim, "Target vector dimension mismatch");
        for v in &self.versors {
            v.reflect_in_place(x);
        }
    }

    /// Reconstructs the equivalent dense n x n orthogonal matrix Q in O(k * n^2).
    /// Used for verification and validation against classical linear algebra libraries.
    pub fn to_dense_matrix(&self) -> Vec<f32> {
        let n = self.dim;
        let mut q = vec![0.0f32; n * n];
        // Initialize as Identity I_n
        for i in 0..n {
            q[i * n + i] = 1.0;
        }

        // Apply reflections to columns of Identity
        let mut col = vec![0.0f32; n];
        for j in 0..n {
            for i in 0..n {
                col[i] = if i == j { 1.0 } else { 0.0 };
            }
            self.apply_in_place(&mut col);
            for i in 0..n {
                q[i * n + j] = col[i];
            }
        }
        q
    }
}

impl VersorAction<[f32]> for HouseholderChain {
    #[inline]
    fn transform_in_place(&self, target: &mut [f32]) {
        self.apply_in_place(target);
    }
}
