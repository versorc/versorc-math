#![allow(clippy::assign_op_pattern)]
use crate::{simd::*, *};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy)]
struct DualNumberGroups {
    /// 1, e01
    g0: Simd32x2,
}

#[derive(Clone, Copy)]
pub union DualNumber {
    groups: DualNumberGroups,
    /// 1, e01, 0, 0
    elements: [f32; 4],
}

impl DualNumber {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(scalar: f32, e01: f32) -> Self {
        Self { elements: [scalar, e01, 0.0, 0.0] }
    }
    pub const fn from_groups(g0: Simd32x2) -> Self {
        Self { groups: DualNumberGroups { g0 } }
    }
    #[inline(always)]
    pub fn group0(&self) -> Simd32x2 {
        unsafe { self.groups.g0 }
    }
    #[inline(always)]
    pub fn group0_mut(&mut self) -> &mut Simd32x2 {
        unsafe { &mut self.groups.g0 }
    }
}

const DUALNUMBER_INDEX_REMAP: [usize; 2] = [0, 1];

impl std::ops::Index<usize> for DualNumber {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &self.elements[DUALNUMBER_INDEX_REMAP[index]] }
    }
}

impl std::ops::IndexMut<usize> for DualNumber {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut self.elements[DUALNUMBER_INDEX_REMAP[index]] }
    }
}

impl std::convert::From<DualNumber> for [f32; 2] {
    fn from(vector: DualNumber) -> Self {
        unsafe { [vector.elements[0], vector.elements[1]] }
    }
}

impl std::convert::From<[f32; 2]> for DualNumber {
    fn from(array: [f32; 2]) -> Self {
        Self { elements: [array[0], array[1], 0.0, 0.0] }
    }
}

impl std::fmt::Debug for DualNumber {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter
            .debug_struct("DualNumber")
            .field("1", &self[0])
            .field("e01", &self[1])
            .finish()
    }
}

impl Add<DualNumber> for f32 {
    type Output = DualNumber;

    fn add(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from(self) * Simd32x2::from([1.0, 0.0]) + other.group0() } }
    }
}

impl Sub<DualNumber> for f32 {
    type Output = DualNumber;

    fn sub(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from(self) * Simd32x2::from([1.0, 0.0]) - other.group0() } }
    }
}

impl GeometricProduct<DualNumber> for f32 {
    type Output = DualNumber;

    fn geometric_product(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from(self) * other.group0() } }
    }
}

impl RegressiveProduct<DualNumber> for f32 {
    type Output = f32;

    fn regressive_product(self, other: DualNumber) -> f32 {
        self * other.group0()[1]
    }
}

impl OuterProduct<DualNumber> for f32 {
    type Output = DualNumber;

    fn outer_product(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from(self) * other.group0() } }
    }
}

impl InnerProduct<DualNumber> for f32 {
    type Output = DualNumber;

    fn inner_product(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from(self) * other.group0() } }
    }
}

impl LeftContraction<DualNumber> for f32 {
    type Output = DualNumber;

    fn left_contraction(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from(self) * other.group0() } }
    }
}

impl RightContraction<DualNumber> for f32 {
    type Output = f32;

    fn right_contraction(self, other: DualNumber) -> f32 {
        self * other.group0()[0]
    }
}

impl ScalarProduct<DualNumber> for f32 {
    type Output = f32;

    fn scalar_product(self, other: DualNumber) -> f32 {
        self * other.group0()[0]
    }
}

impl Zero for DualNumber {
    fn zero() -> Self {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from(0.0) } }
    }
}

impl One for DualNumber {
    fn one() -> Self {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from([1.0, 0.0]) } }
    }
}

impl Neg for DualNumber {
    type Output = DualNumber;

    fn neg(self) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() * Simd32x2::from(-1.0) } }
    }
}

impl Automorphism for DualNumber {
    type Output = DualNumber;

    fn automorphism(self) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() } }
    }
}

impl Reversal for DualNumber {
    type Output = DualNumber;

    fn reversal(self) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() * Simd32x2::from([1.0, -1.0]) } }
    }
}

impl Conjugation for DualNumber {
    type Output = DualNumber;

    fn conjugation(self) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() * Simd32x2::from([1.0, -1.0]) } }
    }
}

impl Dual for DualNumber {
    type Output = DualNumber;

    fn dual(self) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: swizzle!(self.group0(), 1, 0) } }
    }
}

impl Into<f32> for DualNumber {
    fn into(self) -> f32 {
        self.group0()[0]
    }
}

impl Add<f32> for DualNumber {
    type Output = DualNumber;

    fn add(self, other: f32) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() + Simd32x2::from(other) * Simd32x2::from([1.0, 0.0]) } }
    }
}

impl AddAssign<f32> for DualNumber {
    fn add_assign(&mut self, other: f32) {
        *self = (*self).add(other);
    }
}

impl Sub<f32> for DualNumber {
    type Output = DualNumber;

    fn sub(self, other: f32) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() - Simd32x2::from(other) * Simd32x2::from([1.0, 0.0]) } }
    }
}

impl SubAssign<f32> for DualNumber {
    fn sub_assign(&mut self, other: f32) {
        *self = (*self).sub(other);
    }
}

impl GeometricProduct<f32> for DualNumber {
    type Output = DualNumber;

    fn geometric_product(self, other: f32) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() * Simd32x2::from(other) } }
    }
}

impl RegressiveProduct<f32> for DualNumber {
    type Output = f32;

    fn regressive_product(self, other: f32) -> f32 {
        self.group0()[1] * other
    }
}

impl OuterProduct<f32> for DualNumber {
    type Output = DualNumber;

    fn outer_product(self, other: f32) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() * Simd32x2::from(other) } }
    }
}

impl InnerProduct<f32> for DualNumber {
    type Output = DualNumber;

    fn inner_product(self, other: f32) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() * Simd32x2::from(other) } }
    }
}

impl LeftContraction<f32> for DualNumber {
    type Output = f32;

    fn left_contraction(self, other: f32) -> f32 {
        self.group0()[0] * other
    }
}

impl RightContraction<f32> for DualNumber {
    type Output = DualNumber;

    fn right_contraction(self, other: f32) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() * Simd32x2::from(other) } }
    }
}

impl ScalarProduct<f32> for DualNumber {
    type Output = f32;

    fn scalar_product(self, other: f32) -> f32 {
        self.group0()[0] * other
    }
}

impl Add<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn add(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() + other.group0() } }
    }
}

impl AddAssign<DualNumber> for DualNumber {
    fn add_assign(&mut self, other: DualNumber) {
        *self = (*self).add(other);
    }
}

impl Sub<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn sub(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() - other.group0() } }
    }
}

impl SubAssign<DualNumber> for DualNumber {
    fn sub_assign(&mut self, other: DualNumber) {
        *self = (*self).sub(other);
    }
}

impl Mul<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn mul(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() * other.group0() } }
    }
}

impl MulAssign<DualNumber> for DualNumber {
    fn mul_assign(&mut self, other: DualNumber) {
        *self = (*self).mul(other);
    }
}

impl Div<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn div(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from([self.group0()[0], self.group0()[1]]) * Simd32x2::from([1.0, 1.0]) / Simd32x2::from([other.group0()[0], other.group0()[1]]) * Simd32x2::from([1.0, 1.0]) } }
    }
}

impl DivAssign<DualNumber> for DualNumber {
    fn div_assign(&mut self, other: DualNumber) {
        *self = (*self).div(other);
    }
}

impl GeometricProduct<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn geometric_product(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from(self.group0()[0]) * other.group0() + self.group0() * Simd32x2::from(other.group0()[0]) * Simd32x2::from([0.0, 1.0]) } }
    }
}

impl RegressiveProduct<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn regressive_product(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from(self.group0()[1]) * other.group0() + Simd32x2::from(self.group0()[0]) * swizzle!(other.group0(), 1, 0) * Simd32x2::from([1.0, 0.0]) } }
    }
}

impl OuterProduct<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn outer_product(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from(self.group0()[0]) * other.group0() + self.group0() * Simd32x2::from(other.group0()[0]) * Simd32x2::from([0.0, 1.0]) } }
    }
}

impl InnerProduct<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn inner_product(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from(self.group0()[0]) * other.group0() + self.group0() * Simd32x2::from(other.group0()[0]) * Simd32x2::from([0.0, 1.0]) } }
    }
}

impl LeftContraction<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn left_contraction(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: Simd32x2::from(self.group0()[0]) * other.group0() } }
    }
}

impl RightContraction<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn right_contraction(self, other: DualNumber) -> DualNumber {
        DualNumber { groups: DualNumberGroups { g0: self.group0() * Simd32x2::from(other.group0()[0]) } }
    }
}

impl ScalarProduct<DualNumber> for DualNumber {
    type Output = f32;

    fn scalar_product(self, other: DualNumber) -> f32 {
        self.group0()[0] * other.group0()[0]
    }
}

impl SquaredMagnitude for DualNumber {
    type Output = f32;

    fn squared_magnitude(self) -> f32 {
        self.scalar_product(self.reversal())
    }
}

impl Magnitude for DualNumber {
    type Output = f32;

    fn magnitude(self) -> f32 {
        self.squared_magnitude().sqrt()
    }
}

impl Mul<f32> for DualNumber {
    type Output = DualNumber;

    fn mul(self, other: f32) -> DualNumber {
        self.geometric_product(other)
    }
}

impl MulAssign<f32> for DualNumber {
    fn mul_assign(&mut self, other: f32) {
        *self = (*self).mul(other);
    }
}

impl Signum for DualNumber {
    type Output = DualNumber;

    fn signum(self) -> DualNumber {
        self.geometric_product(1.0 / self.magnitude())
    }
}

impl Inverse for DualNumber {
    type Output = DualNumber;

    fn inverse(self) -> DualNumber {
        self.reversal().geometric_product(1.0 / self.squared_magnitude())
    }
}

impl Powi for DualNumber {
    type Output = DualNumber;

    fn powi(self, exponent: isize) -> DualNumber {
        if exponent == 0 {
            return DualNumber::one();
        }
        let mut x: DualNumber = if exponent < 0 { self.inverse() } else { self };
        let mut y: DualNumber = DualNumber::one();
        let mut n: isize = exponent.abs();
        while 1 < n {
            if n & 1 == 1 {
                y = x.geometric_product(y);
            }
            x = x.geometric_product(x);
            n = n >> 1;
        }
        x.geometric_product(y)
    }
}

impl GeometricQuotient<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn geometric_quotient(self, other: DualNumber) -> DualNumber {
        self.geometric_product(other.inverse())
    }
}

impl Transformation<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn transformation(self, other: DualNumber) -> DualNumber {
        self.geometric_product(other).geometric_product(self.reversal())
    }
}

impl GeometricQuotient<f32> for DualNumber {
    type Output = DualNumber;

    fn geometric_quotient(self, other: f32) -> DualNumber {
        self.geometric_product(other.inverse())
    }
}

impl Transformation<f32> for DualNumber {
    type Output = f32;

    fn transformation(self, other: f32) -> f32 {
        self.geometric_product(other).geometric_product(self.reversal()).into()
    }
}

impl GeometricQuotient<DualNumber> for f32 {
    type Output = DualNumber;

    fn geometric_quotient(self, other: DualNumber) -> DualNumber {
        self.geometric_product(other.inverse())
    }
}

impl Transformation<DualNumber> for f32 {
    type Output = DualNumber;

    fn transformation(self, other: DualNumber) -> DualNumber {
        self.geometric_product(other).geometric_product(self.reversal())
    }
}

