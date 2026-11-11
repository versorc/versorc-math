// ============================================================================
// VERSORC-MATH: ORTHOGONAL PLANAR SPINOR BANKS (Cl-RoPE GENERALIZATION)
// License: Apache-2.0 OR MIT
// ============================================================================

use super::VersorAction;

/// Bank of mutually orthogonal 2-versores (Planar Spinors) acting on disjoint 2D planes:
/// (x_{2i}, x_{2i+1}) |-> R_i * (x_{2i}, x_{2i+1}) * R_i~
/// Evaluates in strictly O(N) operations and exactly preserves L2 norm.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanarSpinorBank {
    pub dim: usize,
    /// Cosine of each planar rotation
    pub cos_thetas: Vec<f32>,
    /// Sine of each planar rotation
    pub sin_thetas: Vec<f32>,
}

impl PlanarSpinorBank {
    /// Constructs a planar spinor bank from an array of angles theta_i (radians) for each 2D plane.
    /// Dimension `dim` must be even (2 * num_planes).
    pub fn from_angles(angles: &[f32]) -> Self {
        let num_planes = angles.len();
        let dim = num_planes * 2;
        let mut cos_thetas = Vec::with_capacity(num_planes);
        let mut sin_thetas = Vec::with_capacity(num_planes);
        for &th in angles {
            cos_thetas.push(th.cos());
            sin_thetas.push(th.sin());
        }
        Self {
            dim,
            cos_thetas,
            sin_thetas,
        }
    }

    /// Constructs a multi-frequency Geometric Spinor Bank (RoPE geometric formulation):
    /// theta_i = position * freq_base^(-2i / dim)
    pub fn from_rope(dim: usize, position: usize, freq_base: f32) -> Self {
        assert_eq!(dim % 2, 0, "Dimension must be even for planar spinor bank");
        let num_planes = dim / 2;
        let mut angles = Vec::with_capacity(num_planes);
        for i in 0..num_planes {
            let freq = 1.0 / freq_base.powf((2 * i) as f32 / dim as f32);
            angles.push(position as f32 * freq);
        }
        Self::from_angles(&angles)
    }

    /// Applies the 2-versor spinor bank in-place across all planes in O(N) FLOPs.
    #[inline]
    pub fn apply_in_place(&self, x: &mut [f32]) {
        assert_eq!(x.len(), self.dim, "Input slice dimension mismatch");
        let num_planes = self.cos_thetas.len();
        for i in 0..num_planes {
            let idx0 = i * 2;
            let idx1 = idx0 + 1;
            let u = x[idx0];
            let v = x[idx1];
            let cos_th = self.cos_thetas[i];
            let sin_th = self.sin_thetas[i];

            x[idx0] = u * cos_th - v * sin_th;
            x[idx1] = u * sin_th + v * cos_th;
        }
    }
}

impl VersorAction<[f32]> for PlanarSpinorBank {
    #[inline]
    fn transform_in_place(&self, target: &mut [f32]) {
        self.apply_in_place(target);
    }
}
