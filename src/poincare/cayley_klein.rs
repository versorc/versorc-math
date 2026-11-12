// ============================================================================
// VERSORC-MATH: CAYLEY-KLEIN UNIFIED METRIC GEOMETRY (TAU-PARAMETRIZED)
// License: Apache-2.0
//
// Unifies Hyperbolic (tau < 0), Euclidean (tau = 0), and Elliptic (tau > 0)
// metric spaces under a continuous projective metric framework.
// Curvature kappa = tau.
// ============================================================================

/// Continuous Cayley-Klein metric space parametrized by curvature constant tau.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CayleyKleinSpace {
    /// Curvature parameter:
    /// - tau < 0: Hyperbolic space (kappa = tau, Poincare ball with r_max = 1 / sqrt(|tau|))
    /// - tau = 0: Euclidean space (kappa = 0)
    /// - tau > 0: Elliptic / Spherical space (kappa = tau)
    pub tau: f32,
}

impl CayleyKleinSpace {
    /// Creates a Cayley-Klein space with curvature parameter `tau`.
    #[inline(always)]
    pub const fn new(tau: f32) -> Self {
        Self { tau }
    }

    /// Canonical Hyperbolic space with curvature kappa = -1.0.
    #[inline(always)]
    pub const fn hyperbolic() -> Self {
        Self { tau: -1.0 }
    }

    /// Canonical Euclidean flat space with curvature kappa = 0.0.
    #[inline(always)]
    pub const fn euclidean() -> Self {
        Self { tau: 0.0 }
    }

    /// Canonical Elliptic / Spherical space with curvature kappa = +1.0.
    #[inline(always)]
    pub const fn elliptic() -> Self {
        Self { tau: 1.0 }
    }

    /// Returns the maximum allowable norm ||x|| for points in this space:
    /// - For tau < 0: strictly bounded inside the open ball ||x|| < 1 / sqrt(|tau|).
    /// - For tau >= 0: unbounded (f32::INFINITY).
    #[inline]
    pub fn max_radius(&self) -> f32 {
        if self.tau < -1e-7 {
            1.0f32 / (-self.tau).sqrt()
        } else {
            f32::INFINITY
        }
    }

    /// Computes the Riemannian conformal factor:
    /// lambda_tau(x) = 2.0 / (1.0 + tau * ||x||^2)
    #[inline]
    pub fn conformal_factor(&self, x: &[f32]) -> f32 {
        let mut norm_sq = 0.0f32;
        for &xi in x {
            norm_sq += xi * xi;
        }
        let denom = 1.0f32 + self.tau * norm_sq;
        2.0f32 / denom.max(1e-7)
    }

    /// Computes the exact geodesic distance d_tau(x, y) between two points in R^n:
    /// - tau < 0 (Hyperbolic):
    ///   d_tau(x, y) = (1 / sqrt(|tau|)) * arcosh(1 + (2 |tau| ||x - y||^2) / ((1 - |tau| ||x||^2)(1 - |tau| ||y||^2)))
    /// - tau == 0 (Euclidean):
    ///   d_0(x, y) = ||x - y||
    /// - tau > 0 (Elliptic / Spherical):
    ///   d_tau(x, y) = (1 / sqrt(tau)) * arccos(1 - (2 tau ||x - y||^2) / ((1 + tau ||x||^2)(1 + tau ||y||^2)))
    pub fn geodesic_distance(&self, x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len(), "Dimension mismatch in Cayley-Klein distance");
        let mut diff_sq = 0.0f32;
        let mut x_sq = 0.0f32;
        let mut y_sq = 0.0f32;

        for i in 0..x.len() {
            let d = x[i] - y[i];
            diff_sq += d * d;
            x_sq += x[i] * x[i];
            y_sq += y[i] * y[i];
        }

        if self.tau.abs() < 1e-7 {
            // Euclidean limit: ||x - y||
            return diff_sq.sqrt();
        }

        if self.tau < 0.0 {
            // Hyperbolic branch
            let abs_tau = -self.tau;
            let inv_sqrt_tau = 1.0f32 / abs_tau.sqrt();

            let u_factor = (1.0f32 - abs_tau * x_sq).max(1e-7);
            let v_factor = (1.0f32 - abs_tau * y_sq).max(1e-7);
            let denom = u_factor * v_factor;

            let delta = 1.0f32 + (2.0f32 * abs_tau * diff_sq) / denom;
            let delta_clamped = delta.max(1.0f32);
            // arcosh(z) = ln(z + sqrt(z^2 - 1))
            let arcosh = (delta_clamped + (delta_clamped * delta_clamped - 1.0f32).sqrt()).ln();
            inv_sqrt_tau * arcosh
        } else {
            // Elliptic / Spherical branch
            let tau = self.tau;
            let inv_sqrt_tau = 1.0f32 / tau.sqrt();

            let u_factor = 1.0f32 + tau * x_sq;
            let v_factor = 1.0f32 + tau * y_sq;
            let denom = (u_factor * v_factor).max(1e-7);

            let arg = 1.0f32 - (2.0f32 * tau * diff_sq) / denom;
            let arg_clamped = arg.clamp(-1.0f32, 1.0f32);
            inv_sqrt_tau * arg_clamped.acos()
        }
    }

    /// Computes the generalized Möbius Gyrovector Addition u (+)_\tau v:
    /// u \oplus_\tau v = \frac{(1 - 2\tau\langle u,v\rangle - \tau\|v\|^2)u + (1 + \tau\|u\|^2)v}{1 - 2\tau\langle u,v\rangle + \tau^2\|u\|^2\|v\|^2}
    pub fn mobius_add(&self, u: &[f32], v: &[f32]) -> Vec<f32> {
        assert_eq!(u.len(), v.len(), "Dimension mismatch in Cayley-Klein Mobius addition");
        let n = u.len();

        let mut uv_dot = 0.0f32;
        let mut u_sq = 0.0f32;
        let mut v_sq = 0.0f32;
        for i in 0..n {
            uv_dot += u[i] * v[i];
            u_sq += u[i] * u[i];
            v_sq += v[i] * v[i];
        }

        let tau = self.tau;
        let num_u_coeff = 1.0f32 - 2.0f32 * tau * uv_dot - tau * v_sq;
        let num_v_coeff = 1.0f32 + tau * u_sq;
        let denom = 1.0f32 - 2.0f32 * tau * uv_dot + (tau * tau) * u_sq * v_sq;
        let inv_denom = 1.0f32 / denom.max(1e-7);

        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = (num_u_coeff * u[i] + num_v_coeff * v[i]) * inv_denom;
        }
        out
    }

    /// Stereographic Conformal Projection to Euclidean display coordinates:
    /// x_{euclidean} = \frac{x}{1 + \frac{\tau}{4} \|x\|^2}
    pub fn project_to_euclidean(&self, x: &[f32]) -> Vec<f32> {
        let mut norm_sq = 0.0f32;
        for &xi in x {
            norm_sq += xi * xi;
        }
        let denom = 1.0f32 + (self.tau * 0.25f32) * norm_sq;
        let scale = 1.0f32 / denom.max(1e-7);
        x.iter().map(|&xi| xi * scale).collect()
    }
}
