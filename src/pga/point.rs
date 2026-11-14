// ============================================================================
// VERSORC-MATH: 3D PROJECTIVE GEOMETRIC ALGEBRA (PGA) - POINT (GRADE 3)
// License: Apache-2.0
// ============================================================================

use super::line::PgaLine;
use super::plane::PgaPlane;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Point in 3D Plane-based Projective Geometric Algebra Cl*(3, 0, 1).
/// Grade-3 trivector: P = x*e032 + y*e013 + z*e021 + w*e123.
/// Standard affine points have homogeneous weight w = 1.0.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct PgaPoint {
    pub x: f32, // e032
    pub y: f32, // e013
    pub z: f32, // e021
    pub w: f32, // e123: Homogeneous weight (1.0 for Euclidean points)
}

impl Default for PgaPoint {
    #[inline(always)]
    fn default() -> Self {
        Self::origin()
    }
}

impl PgaPoint {
    /// Constructs a standard normalized Euclidean point with homogeneous weight w = 1.0.
    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, w: 1.0 }
    }

    /// Constructs a point with explicit homogeneous weight w.
    #[inline(always)]
    pub const fn from_homogeneous(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Origin point (0, 0, 0, 1) = e123.
    #[inline(always)]
    pub const fn origin() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }

    /// Ideal direction vector at infinity (w = 0.0).
    #[inline(always)]
    pub const fn ideal(dx: f32, dy: f32, dz: f32) -> Self {
        Self {
            x: dx,
            y: dy,
            z: dz,
            w: 0.0,
        }
    }

    /// Tests if this point is an ideal point at infinity (|w| < 1e-12).
    #[inline(always)]
    pub fn is_ideal(&self) -> bool {
        self.w.abs() < 1e-12
    }

    /// Returns the affine Euclidean coordinates [x/w, y/w, z/w].
    #[inline]
    pub fn to_euclidean(&self) -> [f32; 3] {
        if self.w.abs() > 1e-12 {
            let inv_w = 1.0 / self.w;
            [self.x * inv_w, self.y * inv_w, self.z * inv_w]
        } else {
            [self.x, self.y, self.z]
        }
    }

    /// Returns a normalized point with w = 1.0 (if not ideal).
    #[inline]
    pub fn normalized(&self) -> Self {
        if self.w.abs() > 1e-12 {
            let inv_w = 1.0 / self.w;
            Self {
                x: self.x * inv_w,
                y: self.y * inv_w,
                z: self.z * inv_w,
                w: 1.0,
            }
        } else {
            *self
        }
    }

    /// Euclidean distance between two affine points.
    #[inline]
    pub fn distance(&self, other: &Self) -> f32 {
        self.distance_sq(other).sqrt()
    }

    /// Squared Euclidean distance between two affine points.
    #[inline]
    pub fn distance_sq(&self, other: &Self) -> f32 {
        let p1 = self.to_euclidean();
        let p2 = other.to_euclidean();
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];
        dx * dx + dy * dy + dz * dz
    }

    /// Computes the affine midpoint between two points.
    #[inline]
    pub fn midpoint(&self, other: &Self) -> Self {
        let p1 = self.to_euclidean();
        let p2 = other.to_euclidean();
        Self::new(
            0.5 * (p1[0] + p2[0]),
            0.5 * (p1[1] + p2[1]),
            0.5 * (p1[2] + p2[2]),
        )
    }

    /// Computes the line connecting two points (join product \wedge).
    #[inline]
    pub fn join(&self, other: &Self) -> PgaLine {
        let p1 = self.to_euclidean();
        let p2 = other.to_euclidean();

        // Direction vector = P2 - P1
        let vx = p2[0] - p1[0];
        let vy = p2[1] - p1[1];
        let vz = p2[2] - p1[2];

        // Moment vector = P1 x P2
        let mx = p1[1] * p2[2] - p1[2] * p2[1];
        let my = p1[2] * p2[0] - p1[0] * p2[2];
        let mz = p1[0] * p2[1] - p1[1] * p2[0];

        PgaLine::from_direction_and_moment([vx, vy, vz], [mx, my, mz])
    }

    /// Computes the plane through this point and a line (join product).
    #[inline]
    pub fn join_line(&self, line: &PgaLine) -> PgaPlane {
        let p = self.to_euclidean();
        let dir = line.direction();
        let mom = line.moment();

        // Normal = P x dir - mom
        let nx = p[1] * dir[2] - p[2] * dir[1] - mom[0];
        let ny = p[2] * dir[0] - p[0] * dir[2] - mom[1];
        let nz = p[0] * dir[1] - p[1] * dir[0] - mom[2];

        // Distance d = -(nx * px + ny * py + nz * pz)
        let d = -(nx * p[0] + ny * p[1] + nz * p[2]);

        PgaPlane::new(nx, ny, nz, d)
    }
}

impl Neg for PgaPoint {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

impl Add for PgaPoint {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

impl Sub for PgaPoint {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}

impl Mul<f32> for PgaPoint {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            w: self.w * rhs,
        }
    }
}

impl Div<f32> for PgaPoint {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        let inv = 1.0 / rhs;
        Self {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
            w: self.w * inv,
        }
    }
}
