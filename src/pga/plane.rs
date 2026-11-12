// ============================================================================
// VERSORC-MATH: 3D PROJECTIVE GEOMETRIC ALGEBRA (PGA) - PLANE (GRADE 1)
// License: Apache-2.0 OR MIT
// ============================================================================

use std::ops::{Add, Div, Mul, Neg, Sub};
use super::point::PgaPoint;
use super::line::PgaLine;

/// Plane in 3D Plane-based Projective Geometric Algebra Cl*(3, 0, 1).
/// Grade-1 1-vector: P = a*e1 + b*e2 + c*e3 + d*e0.
/// Represents the affine plane equation: a*x + b*y + c*z + d = 0.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct PgaPlane {
    pub d: f32, // e0: Ideal component (distance scalar when normalized)
    pub a: f32, // e1: Normal x
    pub b: f32, // e2: Normal y
    pub c: f32, // e3: Normal z
}

impl Default for PgaPlane {
    #[inline(always)]
    fn default() -> Self {
        Self::xy()
    }
}

impl PgaPlane {
    /// Constructs a plane from Cartesian coefficients: a*x + b*y + c*z + d = 0.
    #[inline(always)]
    pub const fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self { d, a, b, c }
    }

    /// Plane through a point with a given normal vector.
    #[inline]
    pub fn from_point_normal(point: &[f32; 3], normal: &[f32; 3]) -> Self {
        let d = -(normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2]);
        Self::new(normal[0], normal[1], normal[2], d)
    }

    /// Coordinate plane x = 0 (e1).
    #[inline(always)]
    pub const fn yz() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }

    /// Coordinate plane y = 0 (e2).
    #[inline(always)]
    pub const fn xz() -> Self {
        Self::new(0.0, 1.0, 0.0, 0.0)
    }

    /// Coordinate plane z = 0 (e3).
    #[inline(always)]
    pub const fn xy() -> Self {
        Self::new(0.0, 0.0, 1.0, 0.0)
    }

    /// Ideal plane at infinity e0 (d = 1, a = b = c = 0).
    #[inline(always)]
    pub const fn infinity() -> Self {
        Self { d: 1.0, a: 0.0, b: 0.0, c: 0.0 }
    }

    /// Normal direction vector [a, b, c].
    #[inline(always)]
    pub const fn normal(&self) -> [f32; 3] {
        [self.a, self.b, self.c]
    }

    /// Squared Euclidean norm of the plane's normal: a^2 + b^2 + c^2.
    #[inline(always)]
    pub fn norm_sq(&self) -> f32 {
        self.a * self.a + self.b * self.b + self.c * self.c
    }

    /// Euclidean norm of the plane's normal.
    #[inline(always)]
    pub fn norm(&self) -> f32 {
        self.norm_sq().sqrt()
    }

    /// Returns a normalized plane where a^2 + b^2 + c^2 = 1.
    #[inline]
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n > 1e-12 {
            let inv = 1.0 / n;
            Self {
                d: self.d * inv,
                a: self.a * inv,
                b: self.b * inv,
                c: self.c * inv,
            }
        } else {
            *self
        }
    }

    /// Signed distance from the normalized plane to an affine point.
    #[inline]
    pub fn distance_to_point(&self, p: &PgaPoint) -> f32 {
        let norm_plane = self.normalized();
        let pt = p.to_euclidean();
        norm_plane.a * pt[0] + norm_plane.b * pt[1] + norm_plane.c * pt[2] + norm_plane.d
    }

    /// Reflects an affine point across this plane via 1-versor sandwich P * p * P.
    #[inline]
    pub fn reflect_point(&self, p: &PgaPoint) -> PgaPoint {
        let n = self.normalized();
        let [x, y, z] = p.to_euclidean();
        let signed_dist = n.a * x + n.b * y + n.c * z + n.d;
        let factor = 2.0 * signed_dist;
        PgaPoint::new(
            x - factor * n.a,
            y - factor * n.b,
            z - factor * n.c,
        )
    }

    /// Reflects another plane across this plane via 1-versor sandwich.
    #[inline]
    pub fn reflect_plane(&self, other: &Self) -> Self {
        let n = self.normalized();
        let o = other.normalized();
        let dot = n.a * o.a + n.b * o.b + n.c * o.c;
        let factor = 2.0 * dot;
        Self::new(
            factor * n.a - o.a,
            factor * n.b - o.b,
            factor * n.c - o.c,
            factor * n.d - o.d,
        )
    }

    /// Computes the intersection line (meet product \vee) between two planes.
    #[inline]
    pub fn meet(&self, other: &Self) -> PgaLine {
        // Line bivector = P1 ^ P2 in dual space:
        // Direction vector = normal1 x normal2
        let vx = self.b * other.c - self.c * other.b;
        let vy = self.c * other.a - self.a * other.c;
        let vz = self.a * other.b - self.b * other.a;

        // Moment vector = d1 * normal2 - d2 * normal1
        let mx = self.d * other.a - other.d * self.a;
        let my = self.d * other.b - other.d * self.b;
        let mz = self.d * other.c - other.d * self.c;

        PgaLine::from_direction_and_moment([vx, vy, vz], [mx, my, mz])
    }
}

impl Neg for PgaPlane {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            d: -self.d,
            a: -self.a,
            b: -self.b,
            c: -self.c,
        }
    }
}

impl Add for PgaPlane {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            d: self.d + rhs.d,
            a: self.a + rhs.a,
            b: self.b + rhs.b,
            c: self.c + rhs.c,
        }
    }
}

impl Sub for PgaPlane {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            d: self.d - rhs.d,
            a: self.a - rhs.a,
            b: self.b - rhs.b,
            c: self.c - rhs.c,
        }
    }
}

impl Mul<f32> for PgaPlane {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self {
            d: self.d * rhs,
            a: self.a * rhs,
            b: self.b * rhs,
            c: self.c * rhs,
        }
    }
}

impl Div<f32> for PgaPlane {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        let inv = 1.0 / rhs;
        Self {
            d: self.d * inv,
            a: self.a * inv,
            b: self.b * inv,
            c: self.c * inv,
        }
    }
}
