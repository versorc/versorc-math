// ============================================================================
// VERSORC-MATH: METRIC SIGNATURES FOR CLIFFORD ALGEBRAS Cl(p, q, r)
// License: Apache-2.0
// ============================================================================

/// Metric signature defining a Clifford Geometric Algebra Cl(p, q, r).
/// - P: Number of basis vectors squaring to +1 (Euclidean dimensions).
/// - Q: Number of basis vectors squaring to -1 (Anti-Euclidean / Minkowski time-like).
/// - R: Number of basis vectors squaring to 0 (Degenerate / Projective dimensions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MetricSignature<const P: usize, const Q: usize, const R: usize>;

impl<const P: usize, const Q: usize, const R: usize> MetricSignature<P, Q, R> {
    pub const P: usize = P;
    pub const Q: usize = Q;
    pub const R: usize = R;
    pub const DIM: usize = P + Q + R;
    pub const ALGEBRA_DIM: usize = 1 << Self::DIM;

    /// Evaluates the quadratic form of the i-th canonical basis vector e_i (0-indexed).
    /// Returns +1 for Euclidean, -1 for Anti-Euclidean, and 0 for Degenerate.
    #[inline(always)]
    pub const fn basis_square(i: usize) -> i8 {
        if i < P {
            1
        } else if i < P + Q {
            -1
        } else if i < P + Q + R {
            0
        } else {
            panic!("Basis index out of range for metric signature");
        }
    }
}

/// Standard 2D Euclidean Geometric Algebra Cl(2, 0, 0).
pub type SignatureEuclidean2D = MetricSignature<2, 0, 0>;

/// Standard 3D Euclidean Geometric Algebra Cl(3, 0, 0).
pub type SignatureEuclidean3D = MetricSignature<3, 0, 0>;

/// Spacetime / Minkowski Algebra Cl(1, 3, 0) (East Coast signature).
pub type SignatureMinkowski = MetricSignature<1, 3, 0>;

/// 3D Projective Geometric Algebra (PGA) Cl(3, 0, 1).
pub type SignaturePGA3D = MetricSignature<3, 0, 1>;

/// 3D Conformal Geometric Algebra (CGA) Cl(4, 1, 0).
pub type SignatureCGA3D = MetricSignature<4, 1, 0>;

/// Anti-Euclidean 2D Algebra Cl(0, 2, 0) isomorphic to Hamilton Quaternions H.
pub type SignatureHamilton2D = MetricSignature<0, 2, 0>;
