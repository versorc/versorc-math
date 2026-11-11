// ============================================================================
// VERSORC-MATH: BASIS BLADES & CAYLEY MULTIPLICATION SIGN CALCULATOR
// License: Apache-2.0 OR MIT
// ============================================================================

use crate::signature::MetricSignature;

/// Basis blade represented as a bitmask of generator vectors.
/// E.g., scalar = 0b0000, e_1 = 0b0001, e_2 = 0b0010, e_12 = 0b0011, e_123 = 0b0111.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Blade(pub u32);

impl Blade {
    pub const SCALAR: Self = Self(0);

    #[inline(always)]
    pub const fn new(mask: u32) -> Self {
        Self(mask)
    }

    #[inline(always)]
    pub const fn generator(i: usize) -> Self {
        Self(1 << i)
    }

    /// Grade of the blade (number of basis vectors spanning it).
    #[inline(always)]
    pub const fn grade(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Reverse sign of the blade: (-1)^{k(k-1)/2}.
    #[inline(always)]
    pub const fn reverse_sign(self) -> i8 {
        let k = self.grade();
        if k < 2 {
            1
        } else if (k * (k - 1) / 2) % 2 == 0 {
            1
        } else {
            -1
        }
    }

    /// Grade involution sign of the blade: (-1)^k.
    #[inline(always)]
    pub const fn involution_sign(self) -> i8 {
        if self.grade() % 2 == 0 {
            1
        } else {
            -1
        }
    }

    /// Clifford conjugation sign of the blade: (-1)^{k(k+1)/2}.
    #[inline(always)]
    pub const fn conjugation_sign(self) -> i8 {
        let k = self.grade();
        if (k * (k + 1) / 2) % 2 == 0 {
            1
        } else {
            -1
        }
    }

    /// Computes the geometric product of two basis blades e_I * e_J under metric Cl(p, q, r).
    /// Returns Some((resulting_blade, sign)) or None if the result is zero (due to degenerate r-generators).
    #[inline]
    pub fn geometric_product<const P: usize, const Q: usize, const R: usize>(
        self,
        rhs: Self,
    ) -> Option<(Self, i8)> {
        let mask_a = self.0;
        let mask_b = rhs.0;

        // 1. Calculate permutation transpositions (swaps) to order generators
        let mut swaps = 0u32;
        let mut temp_b = mask_b;
        while temp_b != 0 {
            let j = temp_b.trailing_zeros();
            // Count generators in A that are strictly greater than generator j
            swaps += (mask_a >> (j + 1)).count_ones();
            temp_b &= temp_b - 1; // Clear lowest set bit
        }

        let mut sign: i8 = if swaps % 2 == 0 { 1 } else { -1 };

        // 2. Evaluate metric contraction for common generators: e_k * e_k = eta_kk
        let common = mask_a & mask_b;
        if common != 0 {
            let mut temp_common = common;
            while temp_common != 0 {
                let k = temp_common.trailing_zeros() as usize;
                let metric_sq = MetricSignature::<P, Q, R>::basis_square(k);
                if metric_sq == 0 {
                    return None; // Degenerate generator squared to zero
                }
                sign *= metric_sq;
                temp_common &= temp_common - 1;
            }
        }

        let resulting_mask = mask_a ^ mask_b;
        Some((Self(resulting_mask), sign))
    }

    /// Computes the exterior wedge product e_I ^ e_J.
    /// Nonzero if and only if blades share no common generators (I & J == 0).
    #[inline]
    pub fn wedge_product(self, rhs: Self) -> Option<(Self, i8)> {
        if (self.0 & rhs.0) != 0 {
            return None; // Linearly dependent generators wedge to zero
        }

        // Swaps identical to geometric product when common == 0
        let mut swaps = 0u32;
        let mut temp_b = rhs.0;
        while temp_b != 0 {
            let j = temp_b.trailing_zeros();
            swaps += (self.0 >> (j + 1)).count_ones();
            temp_b &= temp_b - 1;
        }

        let sign: i8 = if swaps % 2 == 0 { 1 } else { -1 };
        Some((Self(self.0 | rhs.0), sign))
    }

    /// Computes the left contraction e_I _| e_J under metric Cl(p, q, r).
    /// Nonzero only if grade(result) == grade(J) - grade(I).
    #[inline]
    pub fn left_contraction<const P: usize, const Q: usize, const R: usize>(
        self,
        rhs: Self,
    ) -> Option<(Self, i8)> {
        let r = self.grade();
        let s = rhs.grade();
        if r > s {
            return None;
        }
        let (prod_blade, sign) = self.geometric_product::<P, Q, R>(rhs)?;
        if prod_blade.grade() == s - r {
            Some((prod_blade, sign))
        } else {
            None
        }
    }

    /// Computes the right contraction e_I |_ e_J under metric Cl(p, q, r).
    /// Nonzero only if grade(result) == grade(I) - grade(J).
    #[inline]
    pub fn right_contraction<const P: usize, const Q: usize, const R: usize>(
        self,
        rhs: Self,
    ) -> Option<(Self, i8)> {
        let r = self.grade();
        let s = rhs.grade();
        if r < s {
            return None;
        }
        let (prod_blade, sign) = self.geometric_product::<P, Q, R>(rhs)?;
        if prod_blade.grade() == r - s {
            Some((prod_blade, sign))
        } else {
            None
        }
    }
}
