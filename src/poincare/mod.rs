// ============================================================================
// VERSORC-MATH: CONFORMAL HYPERBOLIC POINCARE BALL (B^d)
// License: Apache-2.0
//
// Conformal Hyperbolic Geometry, Mobius Gyrovector Addition, and Geodesics.
// Invariant: Points strictly reside within the open unit ball ||x|| < 1.
// ============================================================================

pub mod cayley_klein;
pub use cayley_klein::CayleyKleinSpace;

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
        assert_eq!(
            self.dim(),
            other.dim(),
            "Dimension mismatch in hyperbolic distance"
        );
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
        assert_eq!(
            self.dim(),
            other.dim(),
            "Dimension mismatch in Mobius addition"
        );
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
        for ((dst, &uc), &vc) in res
            .iter_mut()
            .zip(self.coords.iter())
            .zip(other.coords.iter())
        {
            *dst = (num_u_coeff * uc + num_v_coeff * vc) * inv_denom;
        }

        Self::new(res)
    }

    /// Fast monotonic key for ranking hyperbolic distance to `other` without transcendental functions (no sqrt/ln/arcosh).
    /// Preserves strict ranking order: argmin d_B(u, v) == argmin key(u, v).
    #[inline(always)]
    pub fn hyperbolic_distance_key(&self, other: &Self) -> f32 {
        assert_eq!(
            self.dim(),
            other.dim(),
            "Dimension mismatch in hyperbolic distance key"
        );
        let mut diff_sq = 0.0f32;
        for i in 0..self.dim() {
            let d = self.coords[i] - other.coords[i];
            diff_sq += d * d;
        }
        let u_sq = self.norm_sq().min(MAX_NORM_SQ);
        let v_sq = other.norm_sq().min(MAX_NORM_SQ);
        let denom = (1.0f32 - u_sq) * (1.0f32 - v_sq);
        diff_sq / denom.max(1e-12)
    }

    /// Conversion from Poincaré Ball to Klein Model (where geodesics are Euclidean straight chords):
    /// k = 2 * u / (1 + ||u||^2)
    #[inline]
    pub fn to_klein(&self) -> Vec<f32> {
        let u_sq = self.norm_sq().min(MAX_NORM_SQ);
        let factor = 2.0f32 / (1.0f32 + u_sq);
        self.coords.iter().map(|&x| x * factor).collect()
    }

    /// Conversion from Klein Model back to Poincaré Ball:
    /// u = k / (1 + sqrt(1 - ||k||^2))
    #[inline]
    pub fn from_klein(k: &[f32]) -> Self {
        let mut k_sq = 0.0f32;
        for &x in k {
            k_sq += x * x;
        }
        let k_sq_clamped = k_sq.min(MAX_NORM_SQ);
        let denom = 1.0f32 + (1.0f32 - k_sq_clamped).sqrt();
        let inv_denom = 1.0f32 / denom.max(1e-7);
        let u = k.iter().map(|&x| x * inv_denom).collect();
        Self::new(u)
    }

    /// Exact closed-form O(K) computation of the Einstein Hyperbolic Midpoint
    /// using projective equivalence in the Klein model:
    /// \mu_K = \frac{\sum_k w_k \gamma_k k_k}{\sum_k w_k \gamma_k}, \quad \gamma_k = \frac{1}{\sqrt{1 - ||k_k||^2}}
    pub fn einstein_midpoint(points: &[Self], weights: &[f32]) -> Self {
        assert_eq!(
            points.len(),
            weights.len(),
            "Points and weights count mismatch"
        );
        if points.is_empty() {
            return Self::new(vec![]);
        }
        let dim = points[0].dim();
        if points.len() == 1 {
            return points[0].clone();
        }

        let mut sum_k = vec![0.0f32; dim];
        let mut sum_gamma = 0.0f32;

        for (pt, &w) in points.iter().zip(weights.iter()) {
            if w <= 0.0 {
                continue;
            }
            let k = pt.to_klein();
            let mut k_sq = 0.0f32;
            for &x in &k {
                k_sq += x * x;
            }
            let gamma = 1.0f32 / (1.0f32 - k_sq.min(MAX_NORM_SQ)).sqrt().max(1e-7);
            let eff_w = w * gamma;

            for i in 0..dim {
                sum_k[i] += eff_w * k[i];
            }
            sum_gamma += eff_w;
        }

        if sum_gamma <= 1e-9 {
            return points[0].clone();
        }

        let inv_sum = 1.0f32 / sum_gamma;
        let mut mean_k = vec![0.0f32; dim];
        for i in 0..dim {
            mean_k[i] = sum_k[i] * inv_sum;
        }

        Self::from_klein(&mean_k)
    }

    /// Riemannian Stochastic Gradient Descent (RSGD) step with Conformal Scaling:
    /// Updates point along Riemannian gradient:
    /// z^{(t+1)} = z^{(t)} - \eta * ((1 - ||z||^2) / 2)^2 * \nabla_E \mathcal{L}
    #[inline]
    pub fn rsgd_step(&mut self, euclid_grad: &[f32], learning_rate: f32) {
        assert_eq!(
            self.dim(),
            euclid_grad.len(),
            "Dimension mismatch in RSGD step"
        );
        let n_sq = self.norm_sq().min(MAX_NORM_SQ);
        let conf = (1.0f32 - n_sq) * 0.5f32;
        let riemann_scale = learning_rate * (conf * conf);

        for (coord, &grad) in self.coords.iter_mut().zip(euclid_grad.iter()) {
            *coord -= riemann_scale * grad;
        }
        self.project_to_ball();
    }

    /// Conformal exponential map from tangent plane at origin T_0(B^d) to ball B^d:
    /// \exp_0(v) = \tanh(||v|| / 2) * (v / ||v||)
    #[inline]
    pub fn exp_map_zero(v: &[f32]) -> Self {
        let mut norm_sq = 0.0f32;
        for &x in v {
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt();
        let scale = if norm < 1e-9 {
            0.5f32
        } else {
            (norm * 0.5f32).tanh() / norm
        };
        let coords = v.iter().map(|&x| x * scale).collect();
        Self::new(coords)
    }

    /// Conformal logarithmic map from ball B^d to tangent space at origin T_0(B^d):
    /// \log_0(u) = 2 * \operatorname{artanh}(||u||) * (u / ||u||)
    #[inline]
    pub fn log_map_zero(&self) -> Vec<f32> {
        let norm = self.norm().min(1.0 - EPSILON_BOUNDARY);
        let scale = if norm < 1e-9 {
            2.0f32
        } else {
            let artanh = 0.5f32 * ((1.0 + norm) / (1.0 - norm)).ln();
            2.0f32 * artanh / norm
        };
        self.coords.iter().map(|&x| x * scale).collect()
    }
}
