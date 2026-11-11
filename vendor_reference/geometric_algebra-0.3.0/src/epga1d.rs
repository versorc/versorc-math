#![allow(clippy::assign_op_pattern)]
use crate::{simd::*, *};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy)]
struct ComplexNumberGroups {
    /// 1, e01
    g0: Simd32x2,
}

#[derive(Clone, Copy)]
pub union ComplexNumber {
    groups: ComplexNumberGroups,
    /// 1, e01, 0, 0
    elements: [f32; 4],
}

impl ComplexNumber {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(scalar: f32, e01: f32) -> Self {
        Self { elements: [scalar, e01, 0.0, 0.0] }
    }
    pub const fn from_groups(g0: Simd32x2) -> Self {
        Self { groups: ComplexNumberGroups { g0 } }
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

const COMPLEXNUMBER_INDEX_REMAP: [usize; 2] = [0, 1];

impl std::ops::Index<usize> for ComplexNumber {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &self.elements[COMPLEXNUMBER_INDEX_REMAP[index]] }
    }
}

impl std::ops::IndexMut<usize> for ComplexNumber {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut self.elements[COMPLEXNUMBER_INDEX_REMAP[index]] }
    }
}

impl std::convert::From<ComplexNumber> for [f32; 2] {
    fn from(vector: ComplexNumber) -> Self {
        unsafe { [vector.elements[0], vector.elements[1]] }
    }
}

impl std::convert::From<[f32; 2]> for ComplexNumber {
    fn from(array: [f32; 2]) -> Self {
        Self { elements: [array[0], array[1], 0.0, 0.0] }
    }
}

impl std::fmt::Debug for ComplexNumber {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter
            .debug_struct("ComplexNumber")
            .field("1", &self[0])
            .field("e01", &self[1])
            .finish()
    }
}

impl Add<ComplexNumber> for f32 {
    type Output = ComplexNumber;

    fn add(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from(self) * Simd32x2::from([1.0, 0.0]) + other.group0() } }
    }
}

impl Sub<ComplexNumber> for f32 {
    type Output = ComplexNumber;

    fn sub(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from(self) * Simd32x2::from([1.0, 0.0]) - other.group0() } }
    }
}

impl GeometricProduct<ComplexNumber> for f32 {
    type Output = ComplexNumber;

    fn geometric_product(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from(self) * other.group0() } }
    }
}

impl RegressiveProduct<ComplexNumber> for f32 {
    type Output = f32;

    fn regressive_product(self, other: ComplexNumber) -> f32 {
        self * other.group0()[1]
    }
}

impl OuterProduct<ComplexNumber> for f32 {
    type Output = ComplexNumber;

    fn outer_product(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from(self) * other.group0() } }
    }
}

impl InnerProduct<ComplexNumber> for f32 {
    type Output = ComplexNumber;

    fn inner_product(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from(self) * other.group0() } }
    }
}

impl LeftContraction<ComplexNumber> for f32 {
    type Output = ComplexNumber;

    fn left_contraction(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from(self) * other.group0() } }
    }
}

impl RightContraction<ComplexNumber> for f32 {
    type Output = f32;

    fn right_contraction(self, other: ComplexNumber) -> f32 {
        self * other.group0()[0]
    }
}

impl ScalarProduct<ComplexNumber> for f32 {
    type Output = f32;

    fn scalar_product(self, other: ComplexNumber) -> f32 {
        self * other.group0()[0]
    }
}

impl Zero for ComplexNumber {
    fn zero() -> Self {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from(0.0) } }
    }
}

impl One for ComplexNumber {
    fn one() -> Self {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from([1.0, 0.0]) } }
    }
}

impl Neg for ComplexNumber {
    type Output = ComplexNumber;

    fn neg(self) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: self.group0() * Simd32x2::from(-1.0) } }
    }
}

impl Automorphism for ComplexNumber {
    type Output = ComplexNumber;

    fn automorphism(self) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: self.group0() } }
    }
}

impl Reversal for ComplexNumber {
    type Output = ComplexNumber;

    fn reversal(self) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: self.group0() * Simd32x2::from([1.0, -1.0]) } }
    }
}

impl Conjugation for ComplexNumber {
    type Output = ComplexNumber;

    fn conjugation(self) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: self.group0() * Simd32x2::from([1.0, -1.0]) } }
    }
}

impl Dual for ComplexNumber {
    type Output = ComplexNumber;

    fn dual(self) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: swizzle!(self.group0(), 1, 0) } }
    }
}

impl Into<f32> for ComplexNumber {
    fn into(self) -> f32 {
        self.group0()[0]
    }
}

impl Add<f32> for ComplexNumber {
    type Output = ComplexNumber;

    fn add(self, other: f32) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: self.group0() + Simd32x2::from(other) * Simd32x2::from([1.0, 0.0]) } }
    }
}

impl AddAssign<f32> for ComplexNumber {
    fn add_assign(&mut self, other: f32) {
        *self = (*self).add(other);
    }
}

impl Sub<f32> for ComplexNumber {
    type Output = ComplexNumber;

    fn sub(self, other: f32) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: self.group0() - Simd32x2::from(other) * Simd32x2::from([1.0, 0.0]) } }
    }
}

impl SubAssign<f32> for ComplexNumber {
    fn sub_assign(&mut self, other: f32) {
        *self = (*self).sub(other);
    }
}

impl GeometricProduct<f32> for ComplexNumber {
    type Output = ComplexNumber;

    fn geometric_product(self, other: f32) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: self.group0() * Simd32x2::from(other) } }
    }
}

impl RegressiveProduct<f32> for ComplexNumber {
    type Output = f32;

    fn regressive_product(self, other: f32) -> f32 {
        self.group0()[1] * other
    }
}

impl OuterProduct<f32> for ComplexNumber {
    type Output = ComplexNumber;

    fn outer_product(self, other: f32) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: self.group0() * Simd32x2::from(other) } }
    }
}

impl InnerProduct<f32> for ComplexNumber {
    type Output = ComplexNumber;

    fn inner_product(self, other: f32) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: self.group0() * Simd32x2::from(other) } }
    }
}

impl LeftContraction<f32> for ComplexNumber {
    type Output = f32;

    fn left_contraction(self, other: f32) -> f32 {
        self.group0()[0] * other
    }
}

impl RightContraction<f32> for ComplexNumber {
    type Output = ComplexNumber;

    fn right_contraction(self, other: f32) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: self.group0() * Simd32x2::from(other) } }
    }
}

impl ScalarProduct<f32> for ComplexNumber {
    type Output = f32;

    fn scalar_product(self, other: f32) -> f32 {
        self.group0()[0] * other
    }
}

impl Add<ComplexNumber> for ComplexNumber {
    type Output = ComplexNumber;

    fn add(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: self.group0() + other.group0() } }
    }
}

impl AddAssign<ComplexNumber> for ComplexNumber {
    fn add_assign(&mut self, other: ComplexNumber) {
        *self = (*self).add(other);
    }
}

impl Sub<ComplexNumber> for ComplexNumber {
    type Output = ComplexNumber;

    fn sub(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: self.group0() - other.group0() } }
    }
}

impl SubAssign<ComplexNumber> for ComplexNumber {
    fn sub_assign(&mut self, other: ComplexNumber) {
        *self = (*self).sub(other);
    }
}

impl Mul<ComplexNumber> for ComplexNumber {
    type Output = ComplexNumber;

    fn mul(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: self.group0() * other.group0() } }
    }
}

impl MulAssign<ComplexNumber> for ComplexNumber {
    fn mul_assign(&mut self, other: ComplexNumber) {
        *self = (*self).mul(other);
    }
}

impl Div<ComplexNumber> for ComplexNumber {
    type Output = ComplexNumber;

    fn div(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from([self.group0()[0], self.group0()[1]]) * Simd32x2::from([1.0, 1.0]) / Simd32x2::from([other.group0()[0], other.group0()[1]]) * Simd32x2::from([1.0, 1.0]) } }
    }
}

impl DivAssign<ComplexNumber> for ComplexNumber {
    fn div_assign(&mut self, other: ComplexNumber) {
        *self = (*self).div(other);
    }
}

impl GeometricProduct<ComplexNumber> for ComplexNumber {
    type Output = ComplexNumber;

    fn geometric_product(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from(self.group0()[0]) * other.group0() + Simd32x2::from(self.group0()[1]) * swizzle!(other.group0(), 1, 0) * Simd32x2::from([-1.0, 1.0]) } }
    }
}

impl RegressiveProduct<ComplexNumber> for ComplexNumber {
    type Output = ComplexNumber;

    fn regressive_product(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from(self.group0()[1]) * other.group0() + Simd32x2::from(self.group0()[0]) * swizzle!(other.group0(), 1, 0) * Simd32x2::from([1.0, 0.0]) } }
    }
}

impl OuterProduct<ComplexNumber> for ComplexNumber {
    type Output = ComplexNumber;

    fn outer_product(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from(self.group0()[0]) * other.group0() + self.group0() * Simd32x2::from(other.group0()[0]) * Simd32x2::from([0.0, 1.0]) } }
    }
}

impl InnerProduct<ComplexNumber> for ComplexNumber {
    type Output = ComplexNumber;

    fn inner_product(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from(self.group0()[0]) * other.group0() + Simd32x2::from(self.group0()[1]) * swizzle!(other.group0(), 1, 0) * Simd32x2::from([-1.0, 1.0]) } }
    }
}

impl LeftContraction<ComplexNumber> for ComplexNumber {
    type Output = ComplexNumber;

    fn left_contraction(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from(self.group0()[0]) * other.group0() + swizzle!(self.group0(), 1, 0) * swizzle!(other.group0(), 1, 0) * Simd32x2::from([-1.0, 0.0]) } }
    }
}

impl RightContraction<ComplexNumber> for ComplexNumber {
    type Output = ComplexNumber;

    fn right_contraction(self, other: ComplexNumber) -> ComplexNumber {
        ComplexNumber { groups: ComplexNumberGroups { g0: Simd32x2::from(self.group0()[1]) * swizzle!(other.group0(), 1, 0) * Simd32x2::from([-1.0, 1.0]) + Simd32x2::from(self.group0()[0]) * Simd32x2::from(other.group0()[0]) * Simd32x2::from([1.0, 0.0]) } }
    }
}

impl ScalarProduct<ComplexNumber> for ComplexNumber {
    type Output = f32;

    fn scalar_product(self, other: ComplexNumber) -> f32 {
        self.group0()[0] * other.group0()[0] - self.group0()[1] * other.group0()[1]
    }
}

impl SquaredMagnitude for ComplexNumber {
    type Output = f32;

    fn squared_magnitude(self) -> f32 {
        self.scalar_product(self.reversal())
    }
}

impl Magnitude for ComplexNumber {
    type Output = f32;

    fn magnitude(self) -> f32 {
        self.squared_magnitude().sqrt()
    }
}

impl Mul<f32> for ComplexNumber {
    type Output = ComplexNumber;

    fn mul(self, other: f32) -> ComplexNumber {
        self.geometric_product(other)
    }
}

impl MulAssign<f32> for ComplexNumber {
    fn mul_assign(&mut self, other: f32) {
        *self = (*self).mul(other);
    }
}

impl Signum for ComplexNumber {
    type Output = ComplexNumber;

    fn signum(self) -> ComplexNumber {
        self.geometric_product(1.0 / self.magnitude())
    }
}

impl Inverse for ComplexNumber {
    type Output = ComplexNumber;

    fn inverse(self) -> ComplexNumber {
        self.reversal().geometric_product(1.0 / self.squared_magnitude())
    }
}

impl Powi for ComplexNumber {
    type Output = ComplexNumber;

    fn powi(self, exponent: isize) -> ComplexNumber {
        if exponent == 0 {
            return ComplexNumber::one();
        }
        let mut x: ComplexNumber = if exponent < 0 { self.inverse() } else { self };
        let mut y: ComplexNumber = ComplexNumber::one();
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

impl GeometricQuotient<ComplexNumber> for ComplexNumber {
    type Output = ComplexNumber;

    fn geometric_quotient(self, other: ComplexNumber) -> ComplexNumber {
        self.geometric_product(other.inverse())
    }
}

impl Transformation<ComplexNumber> for ComplexNumber {
    type Output = ComplexNumber;

    fn transformation(self, other: ComplexNumber) -> ComplexNumber {
        self.geometric_product(other).geometric_product(self.reversal())
    }
}

impl GeometricQuotient<f32> for ComplexNumber {
    type Output = ComplexNumber;

    fn geometric_quotient(self, other: f32) -> ComplexNumber {
        self.geometric_product(other.inverse())
    }
}

impl Transformation<f32> for ComplexNumber {
    type Output = f32;

    fn transformation(self, other: f32) -> f32 {
        self.geometric_product(other).geometric_product(self.reversal()).into()
    }
}

impl GeometricQuotient<ComplexNumber> for f32 {
    type Output = ComplexNumber;

    fn geometric_quotient(self, other: ComplexNumber) -> ComplexNumber {
        self.geometric_product(other.inverse())
    }
}

impl Transformation<ComplexNumber> for f32 {
    type Output = ComplexNumber;

    fn transformation(self, other: ComplexNumber) -> ComplexNumber {
        self.geometric_product(other).geometric_product(self.reversal())
    }
}

