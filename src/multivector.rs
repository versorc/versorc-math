// ============================================================================
// VERSORC-MATH: GENERALIZED CLIFFORD MULTIVECTOR IMPLEMENTATION
// License: Apache-2.0 OR MIT
// ============================================================================

use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
use crate::blade::Blade;

/// Fixed-size stack-allocated Multivector in Clifford Algebra Cl(p, q, r).
/// - P, Q, R: Metric signature.
/// - N: Algebra dimension (2^(P + Q + R)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Multivector<const P: usize, const Q: usize, const R: usize, const N: usize> {
    pub coords: [f32; N],
}

impl<const P: usize, const Q: usize, const R: usize, const N: usize> Default
    for Multivector<P, Q, R, N>
{
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<const P: usize, const Q: usize, const R: usize, const N: usize> Multivector<P, Q, R, N> {
    /// Zero multivector.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self { coords: [0.0f32; N] }
    }

    /// Scalar multivector: alpha * 1.
    #[inline]
    pub fn from_scalar(alpha: f32) -> Self {
        let mut mv = Self::zero();
        mv.coords[0] = alpha;
        mv
    }

    /// Creates a 1-vector from coordinate slice.
    #[inline]
    pub fn from_vector(vec: &[f32]) -> Self {
        let dim = P + Q + R;
        assert!(vec.len() <= dim, "Vector dimension exceeds algebra basis");
        let mut mv = Self::zero();
        for (i, &val) in vec.iter().enumerate() {
            mv.coords[1 << i] = val;
        }
        mv
    }

    /// Basis element e_i (1-indexed: e_1, e_2, ...).
    #[inline]
    pub fn basis(i: usize) -> Self {
        assert!(i >= 1 && i <= P + Q + R, "Basis index out of range");
        let mut mv = Self::zero();
        mv.coords[1 << (i - 1)] = 1.0;
        mv
    }

    /// Scalar part <A>_0.
    #[inline(always)]
    pub fn scalar(&self) -> f32 {
        self.coords[0]
    }

    /// Extracts the k-th grade projection <A>_k.
    #[inline]
    pub fn grade_projection(&self, grade: usize) -> Self {
        let mut res = Self::zero();
        for (mask, &coeff) in self.coords.iter().enumerate() {
            if (mask as u32).count_ones() as usize == grade {
                res.coords[mask] = coeff;
            }
        }
        res
    }

    /// Clifford Reverse: A~ = sum (-1)^{k(k-1)/2} <A>_k.
    #[inline]
    pub fn reverse(&self) -> Self {
        let mut res = Self::zero();
        for (mask, &coeff) in self.coords.iter().enumerate() {
            let sign = Blade::new(mask as u32).reverse_sign();
            res.coords[mask] = coeff * (sign as f32);
        }
        res
    }

    /// Grade Involution: A^ = sum (-1)^k <A>_k.
    #[inline]
    pub fn grade_involution(&self) -> Self {
        let mut res = Self::zero();
        for (mask, &coeff) in self.coords.iter().enumerate() {
            let sign = Blade::new(mask as u32).involution_sign();
            res.coords[mask] = coeff * (sign as f32);
        }
        res
    }

    /// Clifford Conjugation: \bar{A} = sum (-1)^{k(k+1)/2} <A>_k.
    #[inline]
    pub fn conjugation(&self) -> Self {
        let mut res = Self::zero();
        for (mask, &coeff) in self.coords.iter().enumerate() {
            let sign = Blade::new(mask as u32).conjugation_sign();
            res.coords[mask] = coeff * (sign as f32);
        }
        res
    }

    /// Computes the Clifford Geometric Product: C = A * B.
    #[inline]
    pub fn geometric_product(&self, rhs: &Self) -> Self {
        let mut res = Self::zero();
        for (i, &a) in self.coords.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let blade_a = Blade::new(i as u32);
            for (j, &b) in rhs.coords.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
                let blade_b = Blade::new(j as u32);
                if let Some((blade_c, sign)) = blade_a.geometric_product::<P, Q, R>(blade_b) {
                    res.coords[blade_c.0 as usize] += a * b * (sign as f32);
                }
            }
        }
        res
    }

    /// Computes the Exterior Wedge Product: C = A ^ B.
    #[inline]
    pub fn wedge(&self, rhs: &Self) -> Self {
        let mut res = Self::zero();
        for (i, &a) in self.coords.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let blade_a = Blade::new(i as u32);
            for (j, &b) in rhs.coords.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
                let blade_b = Blade::new(j as u32);
                if let Some((blade_c, sign)) = blade_a.wedge_product(blade_b) {
                    res.coords[blade_c.0 as usize] += a * b * (sign as f32);
                }
            }
        }
        res
    }

    /// Computes the Left Contraction: C = A _| B.
    #[inline]
    pub fn left_contraction(&self, rhs: &Self) -> Self {
        let mut res = Self::zero();
        for (i, &a) in self.coords.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let blade_a = Blade::new(i as u32);
            for (j, &b) in rhs.coords.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
                let blade_b = Blade::new(j as u32);
                if let Some((blade_c, sign)) = blade_a.left_contraction::<P, Q, R>(blade_b) {
                    res.coords[blade_c.0 as usize] += a * b * (sign as f32);
                }
            }
        }
        res
    }

    /// Computes the Right Contraction: C = A |_ B.
    #[inline]
    pub fn right_contraction(&self, rhs: &Self) -> Self {
        let mut res = Self::zero();
        for (i, &a) in self.coords.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let blade_a = Blade::new(i as u32);
            for (j, &b) in rhs.coords.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
                let blade_b = Blade::new(j as u32);
                if let Some((blade_c, sign)) = blade_a.right_contraction::<P, Q, R>(blade_b) {
                    res.coords[blade_c.0 as usize] += a * b * (sign as f32);
                }
            }
        }
        res
    }

    /// Scalar product <A, B> = <A B~>_0.
    #[inline]
    pub fn scalar_product(&self, rhs: &Self) -> f32 {
        self.geometric_product(&rhs.reverse()).scalar()
    }

    /// Squared magnitude: ||A||^2 = <A A~>_0.
    #[inline]
    pub fn norm_sq(&self) -> f32 {
        self.scalar_product(self)
    }

    /// Magnitude / L2 norm.
    #[inline]
    pub fn norm(&self) -> f32 {
        self.norm_sq().max(0.0).sqrt()
    }

    /// Normalized multivector: A / ||A||.
    #[inline]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n > 1e-12 {
            *self * (1.0 / n)
        } else {
            *self
        }
    }

    /// Versor sandwich product: X' = V X V~.
    #[inline]
    pub fn sandwich(&self, x: &Self) -> Self {
        self.geometric_product(x).geometric_product(&self.reverse())
    }

    /// Returns the unit pseudoscalar I = e_1 ^ e_2 ^ ... ^ e_n.
    #[inline]
    pub fn pseudoscalar() -> Self {
        let mut mv = Self::zero();
        mv.coords[N - 1] = 1.0;
        mv
    }

    /// Computes the pseudoscalar inverse I^{-1} for non-degenerate metrics (R == 0).
    /// Under signature (P, Q, 0): I^2 = (-1)^{n(n-1)/2 + Q}, so I^{-1} = I * (-1)^{n(n-1)/2 + Q}.
    #[inline]
    pub fn pseudoscalar_inverse() -> Option<Self> {
        if R > 0 {
            return None; // Degenerate metric has I^2 == 0
        }
        let n = P + Q;
        let sign_power = (n * (n - 1) / 2 + Q) % 2;
        let sign = if sign_power == 0 { 1.0f32 } else { -1.0f32 };
        let mut mv = Self::zero();
        mv.coords[N - 1] = sign;
        Some(mv)
    }

    /// Computes the Hodge Dual: A* = A I^{-1} (Macdonald Def 6.16, Hestenes Section 1.2).
    #[inline]
    pub fn dual(&self) -> Option<Self> {
        let inv_i = Self::pseudoscalar_inverse()?;
        Some(self.geometric_product(&inv_i))
    }

    /// Computes the Undual (reverse duality): A I.
    #[inline]
    pub fn undual(&self) -> Self {
        self.geometric_product(&Self::pseudoscalar())
    }

    /// Computes the Lie Commutator Product: [A, B] = 1/2 (A B - B A) (Hestenes Eq 1.55).
    /// Anticommutative: [A, B] = -[B, A]. Satisfies the Jacobi identity.
    #[inline]
    pub fn commutator(&self, rhs: &Self) -> Self {
        (*self * *rhs - *rhs * *self) * 0.5
    }

    /// Computes the Jordan Anticommutator (symmetric product): A ⊙ B = 1/2 (A B + B A).
    #[inline]
    pub fn anticommutator(&self, rhs: &Self) -> Self {
        (*self * *rhs + *rhs * *self) * 0.5
    }

    /// Computes the Regressive (Meet) Product: A ∨ B = (A* ∧ B*).undual()
    /// In non-degenerate spaces, gives the geometric intersection of blade subspaces.
    #[inline]
    pub fn regressive_product(&self, rhs: &Self) -> Option<Self> {
        let dual_a = self.dual()?;
        let dual_b = rhs.dual()?;
        let wedge_dual = dual_a.wedge(&dual_b);
        Some(wedge_dual.undual())
    }

    /// Computes the Hestenes Inner Product (Fat Dot): A • B = sum_{r>=1, s>=1} < <A>_r <B>_s >_{|r - s|}.
    /// As defined in Hestenes & Sobczyk (1984), Section 1.1, Eq. 1.8.
    #[inline]
    pub fn hestenes_inner_product(&self, rhs: &Self) -> Self {
        let dim = P + Q + R;
        let mut res = Self::zero();
        for r in 1..=dim {
            let a_r = self.grade_projection(r);
            if a_r.norm_sq() == 0.0 {
                continue;
            }
            for s in 1..=dim {
                let b_s = rhs.grade_projection(s);
                if b_s.norm_sq() == 0.0 {
                    continue;
                }
                let target_grade = if r >= s { r - s } else { s - r };
                let prod = a_r.geometric_product(&b_s);
                res += prod.grade_projection(target_grade);
            }
        }
        res
    }

    /// Computes the exact closed-form algebraic inverse A^{-1} for multivectors:
    /// - For Cl(2, 0, 0): A^{-1} = \bar{A} / (A \bar{A})_0
    /// - For Cl(3, 0, 0): A^{-1} = \bar{A} (A \bar{A})~ / det(A)
    /// - General fallback for versors: A~ / ||A||^2
    /// Returns None if multivector is singular (det(A) == 0).
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        if N == 4 && P == 2 && Q == 0 && R == 0 {
            // Cl(2) exact inverse
            let conj = self.conjugation();
            let denom = (*self * conj).scalar();
            if denom.abs() > 1e-12 {
                Some(conj * (1.0 / denom))
            } else {
                None
            }
        } else if N == 8 && P == 3 && Q == 0 && R == 0 {
            // Cl(3) exact closed-form inverse
            let conj = self.conjugation();
            let c = *self * conj; // C in grade 0 and 3
            let c_rev = c.reverse();
            let det = (c * c_rev).scalar();
            if det.abs() > 1e-12 {
                let num = conj * c_rev;
                Some(num * (1.0 / det))
            } else {
                None
            }
        } else {
            // General versor fallback: if A A~ is a nonzero scalar
            let rev = self.reverse();
            let norm_sq = self.norm_sq();
            if norm_sq.abs() > 1e-12 {
                let candidate = rev * (1.0 / norm_sq);
                let test = *self * candidate;
                if (test.scalar() - 1.0).abs() < 1e-4 && (test - Self::from_scalar(1.0)).norm() < 1e-4 {
                    return Some(candidate);
                }
            }
            None
        }
    }
}

// Operator overloads
impl<const P: usize, const Q: usize, const R: usize, const N: usize> Add for Multivector<P, Q, R, N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut coords = [0.0f32; N];
        for i in 0..N {
            coords[i] = self.coords[i] + rhs.coords[i];
        }
        Self { coords }
    }
}

impl<const P: usize, const Q: usize, const R: usize, const N: usize> AddAssign for Multivector<P, Q, R, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coords[i] += rhs.coords[i];
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize, const N: usize> Sub for Multivector<P, Q, R, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut coords = [0.0f32; N];
        for i in 0..N {
            coords[i] = self.coords[i] - rhs.coords[i];
        }
        Self { coords }
    }
}

impl<const P: usize, const Q: usize, const R: usize, const N: usize> SubAssign for Multivector<P, Q, R, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coords[i] -= rhs.coords[i];
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize, const N: usize> Mul<f32> for Multivector<P, Q, R, N> {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        let mut coords = [0.0f32; N];
        for i in 0..N {
            coords[i] = self.coords[i] * scalar;
        }
        Self { coords }
    }
}

impl<const P: usize, const Q: usize, const R: usize, const N: usize> Mul for Multivector<P, Q, R, N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.geometric_product(&rhs)
    }
}

impl<const P: usize, const Q: usize, const R: usize, const N: usize> Neg for Multivector<P, Q, R, N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut coords = [0.0f32; N];
        for i in 0..N {
            coords[i] = -self.coords[i];
        }
        Self { coords }
    }
}

// Canonical Type Aliases
pub type Cl2 = Multivector<2, 0, 0, 4>;
pub type Cl3 = Multivector<3, 0, 0, 8>;
pub type ClMinkowski = Multivector<1, 3, 0, 16>;
pub type ClPGA3D = Multivector<3, 0, 1, 16>;
pub type ClCGA3D = Multivector<4, 1, 0, 32>;
pub type ClHamilton = Multivector<0, 2, 0, 4>;
