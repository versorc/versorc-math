// ============================================================================
// VERSORC-MATH: CONFORMAL HYPERBOLIC POINCARE BALL (B^d)
// License: Apache-2.0 OR MIT
//
// Conformal Hyperbolic Geometry, Mobius Gyrovector Addition, and Geodesics.
// Invariant: Points strictly reside within the open unit ball ||x|| < 1.
// ============================================================================

pub const EPSILON_BOUNDARY: f32 = 1e-5;
pub const MAX_NORM_SQ: f32 = (1.0 - EPSILON_BOUNDARY) * (1.0 - EPSILON_BOUNDARY);

/// Point in the Conformal Poincaré Ball B^d (Curvature c = -1).
#[derive(Debug, Clone, PartialEq)]
pub struct PoincarePoint {
    pub coords: Vec<f32>,
}

impl PoincarePoint {
    /// Creates a new Poincaré point, automatically projecting it inside the open unit ball if necessary.
    pub fn new(coords: Vec<f32>) -> Self {
        let mut pt = Self { coords };
        pt.project_to_ball();
        pt
    }

    /// Origin of the Poincaré ball (all zeros).
    pub fn origin(dim: usize) -> Self {
        Self {
            coords: vec![0.0f32; dim],
        }
    }

    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.coords.len()
    }

    /// Squared Euclidean norm ||x||^2.
    #[inline]
    pub fn norm_sq(&self) -> f32 {
        let mut sum = 0.0f32;
        for &x in &self.coords {
            sum += x * x;
        }
        sum
    }

    /// Euclidean norm ||x||.
    #[inline(always)]
    pub fn norm(&self) -> f32 {
        self.norm_sq().sqrt()
    }

    /// Projects the point strictly inside ||x|| < 1 - EPSILON_BOUNDARY.
    pub fn project_to_ball(&mut self) {
        let n_sq = self.norm_sq();
        if n_sq >= MAX_NORM_SQ {
            let n = n_sq.sqrt();
            let scale = (1.0 - EPSILON_BOUNDARY) / n.max(1e-7);
            for x in &mut self.coords {
                *x *= scale;
            }
        }
    }

    /// Conformal factor of the Riemannian metric: lambda_x = 2 / (1 - ||x||^2).
    #[inline]
    pub fn lambda(&self) -> f32 {
        let n_sq = self.norm_sq().min(MAX_NORM_SQ);
        2.0f32 / (1.0f32 - n_sq)
    }

    /// Geodesic hyperbolic distance:
    /// d_B(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2) * (1 - ||v||^2)))
    pub fn geodesic_distance(&self, other: &Self) -> f32 {
        assert_eq!(self.dim(), other.dim(), "Dimension mismatch in hyperbolic distance");
        let mut diff_sq = 0.0f32;
        for i in 0..self.dim() {
            let d = self.coords[i] - other.coords[i];
            diff_sq += d * d;
        }
        let u_sq = self.norm_sq().min(MAX_NORM_SQ);
        let v_sq = other.norm_sq().min(MAX_NORM_SQ);
        let denom = (1.0f32 - u_sq) * (1.0f32 - v_sq);
        let arg = 1.0f32 + 2.0f32 * diff_sq / denom.max(1e-12);
        // arcosh(z) = ln(z + sqrt(z^2 - 1))
        let z = arg.max(1.0f32);
        (z + (z * z - 1.0f32).sqrt()).ln()
    }

    /// Möbius gyrovector addition in the Poincaré ball:
    /// u (+) v = ((1 + 2<u, v> + ||v||^2) u + (1 - ||u||^2) v) / (1 + 2<u, v> + ||u||^2 ||v||^2)
    pub fn mobius_add(&self, other: &Self) -> Self {
        assert_eq!(self.dim(), other.dim(), "Dimension mismatch in Mobius addition");
        let u_sq = self.norm_sq().min(MAX_NORM_SQ);
        let v_sq = other.norm_sq().min(MAX_NORM_SQ);

        let mut uv_dot = 0.0f32;
        for i in 0..self.dim() {
            uv_dot += self.coords[i] * other.coords[i];
        }

        let num_u_coeff = 1.0f32 + 2.0f32 * uv_dot + v_sq;
        let num_v_coeff = 1.0f32 - u_sq;
        let denom = 1.0f32 + 2.0f32 * uv_dot + u_sq * v_sq;
        let inv_denom = 1.0f32 / denom.max(1e-7);

        let mut res = vec![0.0f32; self.dim()];
        for i in 0..self.dim() {
            res[i] = (num_u_coeff * self.coords[i] + num_v_coeff * other.coords[i]) * inv_denom;
        }

        Self::new(res)
    }
}
