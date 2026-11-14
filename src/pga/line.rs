// ============================================================================
// VERSORC-MATH: 3D PROJECTIVE GEOMETRIC ALGEBRA (PGA) - LINE (GRADE 2)
// License: Apache-2.0
// ============================================================================

use super::plane::PgaPlane;
use super::point::PgaPoint;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Line in 3D Projective Geometric Algebra Cl*(3, 0, 1) (Plücker bivector).
/// Composed of Euclidean direction (vx, vy, vz) and moment (mx, my, mz) about origin.
/// Plücker condition: vx*mx + vy*my + vz*mz = 0.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct PgaLine {
    pub vx: f32, // Direction x
    pub vy: f32, // Direction y
    pub vz: f32, // Direction z
    pub mx: f32, // Moment x
    pub my: f32, // Moment y
    pub mz: f32, // Moment z
    pub _pad0: f32,
    pub _pad1: f32,
}

impl Default for PgaLine {
    #[inline(always)]
    fn default() -> Self {
        Self::z_axis()
    }
}

impl PgaLine {
    /// Constructs a Plücker line from direction and moment components.
    #[inline(always)]
    pub const fn new(vx: f32, vy: f32, vz: f32, mx: f32, my: f32, mz: f32) -> Self {
        Self {
            vx,
            vy,
            vz,
            mx,
            my,
            mz,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }

    /// Constructs a Plücker line from direction and moment arrays.
    #[inline(always)]
    pub const fn from_direction_and_moment(dir: [f32; 3], mom: [f32; 3]) -> Self {
        Self::new(dir[0], dir[1], dir[2], mom[0], mom[1], mom[2])
    }

    /// Canonical X axis line through origin.
    #[inline(always)]
    pub const fn x_axis() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Canonical Y axis line through origin.
    #[inline(always)]
    pub const fn y_axis() -> Self {
        Self::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Canonical Z axis line through origin.
    #[inline(always)]
    pub const fn z_axis() -> Self {
        Self::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    }

    /// Constructs a line through a point with a given direction.
    #[inline]
    pub fn from_point_and_direction(point: &[f32; 3], dir: &[f32; 3]) -> Self {
        let mx = point[1] * dir[2] - point[2] * dir[1];
        let my = point[2] * dir[0] - point[0] * dir[2];
        let mz = point[0] * dir[1] - point[1] * dir[0];
        Self::new(dir[0], dir[1], dir[2], mx, my, mz)
    }

    /// Direction vector [vx, vy, vz].
    #[inline(always)]
    pub const fn direction(&self) -> [f32; 3] {
        [self.vx, self.vy, self.vz]
    }

    /// Moment vector [mx, my, mz].
    #[inline(always)]
    pub const fn moment(&self) -> [f32; 3] {
        [self.mx, self.my, self.mz]
    }

    /// Direction norm ||v||.
    #[inline(always)]
    pub fn norm(&self) -> f32 {
        (self.vx * self.vx + self.vy * self.vy + self.vz * self.vz).sqrt()
    }

    /// Returns a normalized line with unit direction ||v|| = 1.
    #[inline]
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n > 1e-12 {
            let inv = 1.0 / n;
            Self::new(
                self.vx * inv,
                self.vy * inv,
                self.vz * inv,
                self.mx * inv,
                self.my * inv,
                self.mz * inv,
            )
        } else {
            *self
        }
    }

    /// Returns the point on the line closest to the coordinate origin.
    #[inline]
    pub fn closest_point_to_origin(&self) -> PgaPoint {
        let v2 = self.vx * self.vx + self.vy * self.vy + self.vz * self.vz;
        if v2 > 1e-12 {
            let inv_v2 = 1.0 / v2;
            let px = (self.vy * self.mz - self.vz * self.my) * inv_v2;
            let py = (self.vz * self.mx - self.vx * self.mz) * inv_v2;
            let pz = (self.vx * self.my - self.vy * self.mx) * inv_v2;
            PgaPoint::new(px, py, pz)
        } else {
            PgaPoint::origin()
        }
    }

    /// Orthogonal projection of an affine point onto the line.
    #[inline]
    pub fn project_point(&self, p: &PgaPoint) -> PgaPoint {
        let norm_line = self.normalized();
        let pt = p.to_euclidean();
        let p0 = norm_line.closest_point_to_origin().to_euclidean();

        let dx = pt[0] - p0[0];
        let dy = pt[1] - p0[1];
        let dz = pt[2] - p0[2];

        let dot = dx * norm_line.vx + dy * norm_line.vy + dz * norm_line.vz;
        PgaPoint::new(
            p0[0] + dot * norm_line.vx,
            p0[1] + dot * norm_line.vy,
            p0[2] + dot * norm_line.vz,
        )
    }

    /// Perpendicular distance from an affine point to the line.
    #[inline]
    pub fn distance_to_point(&self, p: &PgaPoint) -> f32 {
        let proj = self.project_point(p);
        p.distance(&proj)
    }

    /// Computes the intersection point between this line and a plane (meet \vee).
    #[inline]
    pub fn meet_plane(&self, plane: &PgaPlane) -> PgaPoint {
        let norm_p = plane.normalized();
        let [nx, ny, nz] = norm_p.normal();
        let d = norm_p.d;

        let denom = self.vx * nx + self.vy * ny + self.vz * nz;
        if denom.abs() < 1e-12 {
            // Parallel: intersection is ideal point at infinity
            PgaPoint::ideal(self.vx, self.vy, self.vz)
        } else {
            let p0 = self.closest_point_to_origin().to_euclidean();
            let t = -(nx * p0[0] + ny * p0[1] + nz * p0[2] + d) / denom;
            PgaPoint::new(
                p0[0] + t * self.vx,
                p0[1] + t * self.vy,
                p0[2] + t * self.vz,
            )
        }
    }
}

impl Neg for PgaLine {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.vx, -self.vy, -self.vz, -self.mx, -self.my, -self.mz)
    }
}

impl Add for PgaLine {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(
            self.vx + rhs.vx,
            self.vy + rhs.vy,
            self.vz + rhs.vz,
            self.mx + rhs.mx,
            self.my + rhs.my,
            self.mz + rhs.mz,
        )
    }
}

impl Sub for PgaLine {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(
            self.vx - rhs.vx,
            self.vy - rhs.vy,
            self.vz - rhs.vz,
            self.mx - rhs.mx,
            self.my - rhs.my,
            self.mz - rhs.mz,
        )
    }
}

impl Mul<f32> for PgaLine {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self::new(
            self.vx * rhs,
            self.vy * rhs,
            self.vz * rhs,
            self.mx * rhs,
            self.my * rhs,
            self.mz * rhs,
        )
    }
}

impl Div<f32> for PgaLine {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        let inv = 1.0 / rhs;
        Self::new(
            self.vx * inv,
            self.vy * inv,
            self.vz * inv,
            self.mx * inv,
            self.my * inv,
            self.mz * inv,
        )
    }
}
