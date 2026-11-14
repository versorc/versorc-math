// ============================================================================
// VERSORC-MATH: 3D PROJECTIVE GEOMETRIC ALGEBRA (PGA) - MOTOR (GRADE 0 + 2 + 4)
// License: Apache-2.0
// ============================================================================

use super::line::PgaLine;
use super::plane::PgaPlane;
use super::point::PgaPoint;
use crate::versor::Quaternion;
use std::ops::{Add, Mul, Neg, Sub};

/// Motor in 3D Projective Geometric Algebra Cl*(3, 0, 1).
/// Even multivector combining rotation and translation (SE(3) Lie group).
/// Grade 0 (scalar) + Grade 2 (bivectors) + Grade 4 (pseudoscalar).
/// M = s + (b23*e23 + b31*e31 + b12*e12) + (d01*e01 + d02*e02 + d03*e03) + e0123*e0123.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct PgaMotor {
    pub s: f32,     // 1: Scalar component
    pub b23: f32,   // e23: Rotational bivector x (quaternion -i)
    pub b31: f32,   // e31: Rotational bivector y (quaternion -j)
    pub b12: f32,   // e12: Rotational bivector z (quaternion -k)
    pub d01: f32,   // e01: Translational bivector x
    pub d02: f32,   // e02: Translational bivector y
    pub d03: f32,   // e03: Translational bivector z
    pub e0123: f32, // e0123: Pseudoscalar / dual scalar
}

impl Default for PgaMotor {
    #[inline(always)]
    fn default() -> Self {
        Self::identity()
    }
}

impl PgaMotor {
    /// Identity motor (no rotation, no translation).
    #[inline(always)]
    pub const fn identity() -> Self {
        Self {
            s: 1.0,
            b23: 0.0,
            b31: 0.0,
            b12: 0.0,
            d01: 0.0,
            d02: 0.0,
            d03: 0.0,
            e0123: 0.0,
        }
    }

    /// Construct a motor from raw components.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        s: f32,
        b23: f32,
        b31: f32,
        b12: f32,
        d01: f32,
        d02: f32,
        d03: f32,
        e0123: f32,
    ) -> Self {
        Self {
            s,
            b23,
            b31,
            b12,
            d01,
            d02,
            d03,
            e0123,
        }
    }

    /// Pure rotation rotor around origin: R = cos(theta/2) - sin(theta/2) * (ux*e23 + uy*e31 + uz*e12).
    #[inline]
    pub fn from_rotation(axis: &[f32; 3], angle_rad: f32) -> Self {
        let half = angle_rad * 0.5;
        let sin_half = half.sin();
        let cos_half = half.cos();
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let scale = if norm > 1e-12 { sin_half / norm } else { 0.0 };

        Self {
            s: cos_half,
            b23: -axis[0] * scale,
            b31: -axis[1] * scale,
            b12: -axis[2] * scale,
            d01: 0.0,
            d02: 0.0,
            d03: 0.0,
            e0123: 0.0,
        }
    }

    /// Pure translation motor: T = 1 - 0.5 * (tx*e01 + ty*e02 + tz*e03).
    #[inline]
    pub const fn from_translation(tx: f32, ty: f32, tz: f32) -> Self {
        Self {
            s: 1.0,
            b23: 0.0,
            b31: 0.0,
            b12: 0.0,
            d01: -0.5 * tx,
            d02: -0.5 * ty,
            d03: -0.5 * tz,
            e0123: 0.0,
        }
    }

    /// Rigid body transformation combining rotation followed by translation: M = T * R.
    #[inline]
    pub fn from_rotation_translation(
        axis: &[f32; 3],
        angle_rad: f32,
        translation: &[f32; 3],
    ) -> Self {
        let r = Self::from_rotation(axis, angle_rad);
        let t = Self::from_translation(translation[0], translation[1], translation[2]);
        t.geometric_product(&r)
    }

    /// Screw motion: rotation by `angle` around line `l` and translation by `pitch * angle` along it.
    #[inline]
    pub fn from_screw_axis(line: &PgaLine, angle_rad: f32, pitch: f32) -> Self {
        let l = line.normalized();
        let half_ang = angle_rad * 0.5;
        let half_d = (pitch * angle_rad) * 0.5;

        let cos_ang = half_ang.cos();
        let sin_ang = half_ang.sin();

        // Exponential map of general screw generator
        let s = cos_ang;
        let e0123 = half_d * sin_ang;

        let b23 = -l.vx * sin_ang;
        let b31 = -l.vy * sin_ang;
        let b12 = -l.vz * sin_ang;

        let d01 = -l.mx * sin_ang - half_d * cos_ang * l.vx;
        let d02 = -l.my * sin_ang - half_d * cos_ang * l.vy;
        let d03 = -l.mz * sin_ang - half_d * cos_ang * l.vz;

        Self::new(s, b23, b31, b12, d01, d02, d03, e0123)
    }

    /// Clifford reverse: M~ = s - b23*e23 - b31*e31 - b12*e12 - d01*e01 - d02*e02 - d03*e03 + e0123*e0123.
    #[inline(always)]
    pub fn reverse(&self) -> Self {
        Self {
            s: self.s,
            b23: -self.b23,
            b31: -self.b31,
            b12: -self.b12,
            d01: -self.d01,
            d02: -self.d02,
            d03: -self.d03,
            e0123: self.e0123,
        }
    }

    /// Squared rotational norm: s^2 + b23^2 + b31^2 + b12^2.
    #[inline(always)]
    pub fn rotor_norm_sq(&self) -> f32 {
        self.s * self.s + self.b23 * self.b23 + self.b31 * self.b31 + self.b12 * self.b12
    }

    /// Rotational norm.
    #[inline(always)]
    pub fn rotor_norm(&self) -> f32 {
        self.rotor_norm_sq().sqrt()
    }

    /// Normalizes the motor so that M * M~ = 1.
    #[inline]
    pub fn normalized(&self) -> Self {
        let r_sq = self.rotor_norm_sq();
        if r_sq < 1e-15 {
            return Self::identity();
        }
        let inv_r = 1.0 / r_sq.sqrt();
        let inv_r_sq = 1.0 / r_sq;

        // Ideal / dual normalization correction
        let dot_dual =
            self.s * self.e0123 - self.b23 * self.d01 - self.b31 * self.d02 - self.b12 * self.d03;
        let t = dot_dual * inv_r_sq;

        Self {
            s: self.s * inv_r,
            b23: self.b23 * inv_r,
            b31: self.b31 * inv_r,
            b12: self.b12 * inv_r,
            d01: (self.d01 + self.b23 * t) * inv_r,
            d02: (self.d02 + self.b31 * t) * inv_r,
            d03: (self.d03 + self.b12 * t) * inv_r,
            e0123: (self.e0123 - self.s * t) * inv_r,
        }
    }

    /// Motor inverse M^-1. For normalized motors, M^-1 = M~.
    #[inline]
    pub fn inverse(&self) -> Self {
        let rev = self.reverse();
        let r_sq = self.rotor_norm_sq();
        if r_sq < 1e-15 {
            return Self::identity();
        }
        let inv_r_sq = 1.0 / r_sq;
        let dot_dual = 2.0
            * (self.s * self.e0123
                - self.b23 * self.d01
                - self.b31 * self.d02
                - self.b12 * self.d03);

        Self {
            s: rev.s * inv_r_sq,
            b23: rev.b23 * inv_r_sq,
            b31: rev.b31 * inv_r_sq,
            b12: rev.b12 * inv_r_sq,
            d01: (rev.d01 * inv_r_sq) - (self.b23 * dot_dual * inv_r_sq * inv_r_sq),
            d02: (rev.d02 * inv_r_sq) - (self.b31 * dot_dual * inv_r_sq * inv_r_sq),
            d03: (rev.d03 * inv_r_sq) - (self.b12 * dot_dual * inv_r_sq * inv_r_sq),
            e0123: (rev.e0123 * inv_r_sq) - (self.s * dot_dual * inv_r_sq * inv_r_sq),
        }
    }

    /// Geometric product of two motors: M = self * rhs.
    #[inline]
    pub fn geometric_product(&self, rhs: &Self) -> Self {
        let a = self;
        let b = rhs;

        // Grade 0 & 2 (Rotor part)
        let s = a.s * b.s - a.b23 * b.b23 - a.b31 * b.b31 - a.b12 * b.b12;
        let b23 = a.s * b.b23 + a.b23 * b.s - a.b31 * b.b12 + a.b12 * b.b31;
        let b31 = a.s * b.b31 + a.b31 * b.s - a.b12 * b.b23 + a.b23 * b.b12;
        let b12 = a.s * b.b12 + a.b12 * b.s - a.b23 * b.b31 + a.b31 * b.b23;

        // Grade 2 ideal & Grade 4 (Translator / Pseudoscalar part)
        let e0123 = a.s * b.e0123
            + a.e0123 * b.s
            + a.b23 * b.d01
            + a.d01 * b.b23
            + a.b31 * b.d02
            + a.d02 * b.b31
            + a.b12 * b.d03
            + a.d03 * b.b12;

        let d01 = a.s * b.d01 + a.d01 * b.s - a.b23 * b.e0123 - a.e0123 * b.b23 + a.b12 * b.d02
            - a.b31 * b.d03
            - a.d02 * b.b12
            + a.d03 * b.b31;

        let d02 = a.s * b.d02 + a.d02 * b.s - a.b31 * b.e0123 - a.e0123 * b.b31 + a.b23 * b.d03
            - a.b12 * b.d01
            - a.d03 * b.b23
            + a.d01 * b.b12;

        let d03 = a.s * b.d03 + a.d03 * b.s - a.b12 * b.e0123 - a.e0123 * b.b12 + a.b31 * b.d01
            - a.b23 * b.d02
            - a.d01 * b.b31
            + a.d02 * b.b23;

        Self {
            s,
            b23,
            b31,
            b12,
            d01,
            d02,
            d03,
            e0123,
        }
    }

    /// Transforms a homogeneous or affine point via sandwich product: p' = M * p * M~.
    /// Normalizes the motor and preserves `p.w` so ideal points (`w = 0`) rotate without translating.
    #[inline]
    pub fn transform_point(&self, p: &PgaPoint) -> PgaPoint {
        let m = self.normalized();
        let q_r = Quaternion::new(m.s, -m.b23, -m.b31, -m.b12);
        let q_d = Quaternion::new(-m.e0123, -m.d01, -m.d02, -m.d03);

        let rot_p = q_r.rotate_vector3(&[p.x, p.y, p.z]);
        let t_quat = (q_d * q_r.conjugate()) * 2.0;

        PgaPoint::from_homogeneous(
            rot_p[0] + p.w * t_quat.x,
            rot_p[1] + p.w * t_quat.y,
            rot_p[2] + p.w * t_quat.z,
            p.w,
        )
    }

    /// Transforms a plane via sandwich product: P' = M * P * M~.
    #[inline]
    pub fn transform_plane(&self, plane: &PgaPlane) -> PgaPlane {
        let m = self.normalized();
        let norm_p = plane.normalized();
        if norm_p.norm_sq() <= 1e-12 {
            // Ideal plane d*e0 is invariant under proper rigid motions
            return *plane;
        }
        let [nx, ny, nz] = norm_p.normal();

        // Rotate normal using conjugate rotor
        let qw = m.s;
        let qx = -m.b23;
        let qy = -m.b31;
        let qz = -m.b12;

        let rx = 2.0 * (qy * nz - qz * ny);
        let ry = 2.0 * (qz * nx - qx * nz);
        let rz = 2.0 * (qx * ny - qy * nx);

        let rot_nx = nx + qw * rx + (qy * rz - qz * ry);
        let rot_ny = ny + qw * ry + (qz * rx - qx * rz);
        let rot_nz = nz + qw * rz + (qx * ry - qy * rx);

        // Distance updated by translation offset
        let p_pt = PgaPoint::new(-norm_p.d * nx, -norm_p.d * ny, -norm_p.d * nz);
        let trans_pt = m.transform_point(&p_pt);
        let [tx, ty, tz] = trans_pt.to_euclidean();

        let new_d = -(rot_nx * tx + rot_ny * ty + rot_nz * tz);
        PgaPlane::new(rot_nx, rot_ny, rot_nz, new_d)
    }

    /// Transforms a line via sandwich product: L' = M * L * M~.
    #[inline]
    pub fn transform_line(&self, line: &PgaLine) -> PgaLine {
        let p1 = line.closest_point_to_origin();
        let dir = line.direction();
        let p2 = PgaPoint::new(p1.x + dir[0], p1.y + dir[1], p1.z + dir[2]);

        let tp1 = self.transform_point(&p1);
        let tp2 = self.transform_point(&p2);

        tp1.join(&tp2)
    }

    /// Screw linear interpolation (ScLERP) between this motor and `target` with parameter `t` in [0, 1].
    #[inline]
    pub fn sclerp(&self, target: &Self, t: f32) -> Self {
        if t <= 0.0 {
            return *self;
        }
        if t >= 1.0 {
            return *target;
        }

        // Relative motor: Delta = target * self^-1; enforce shortest arc (delta.s >= 0,
        // tolerating ~1e-7 cos(pi/2) float rounding at exact 180-degree boundary)
        let mut delta = target.geometric_product(&self.inverse()).normalized();
        if delta.s < -1e-6 {
            delta = -delta;
        }

        // Extract angle and rotation axis
        let half_ang = delta.s.clamp(-1.0, 1.0).acos();
        let sin_half = half_ang.sin();

        let delta_t = if sin_half.abs() < 1e-6 {
            // Pure translation limit: t_vec = 2 * (-d01, -d02, -d03)
            let tx = -2.0 * delta.d01 * t;
            let ty = -2.0 * delta.d02 * t;
            let tz = -2.0 * delta.d03 * t;
            Self::from_translation(tx, ty, tz)
        } else {
            let inv_sin = 1.0 / sin_half;
            // Direction unit vector of screw axis
            let vx = -delta.b23 * inv_sin;
            let vy = -delta.b31 * inv_sin;
            let vz = -delta.b12 * inv_sin;

            // Total translation distance d along screw axis
            let half_d = delta.e0123 * inv_sin;
            let cos_half = delta.s;

            // Moment of screw axis
            let mx = (-delta.d01 - half_d * cos_half * vx) * inv_sin;
            let my = (-delta.d02 - half_d * cos_half * vy) * inv_sin;
            let mz = (-delta.d03 - half_d * cos_half * vz) * inv_sin;

            let line = PgaLine::from_direction_and_moment([vx, vy, vz], [mx, my, mz]);

            let ang_t = (2.0 * half_ang) * t;
            let pitch_t = if ang_t.abs() > 1e-6 {
                (2.0 * half_d * t) / ang_t
            } else {
                0.0
            };

            Self::from_screw_axis(&line, ang_t, pitch_t)
        };

        delta_t.geometric_product(self).normalized()
    }
}

impl Mul for PgaMotor {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.geometric_product(&rhs)
    }
}

impl Mul<&PgaMotor> for &PgaMotor {
    type Output = PgaMotor;
    #[inline(always)]
    fn mul(self, rhs: &PgaMotor) -> PgaMotor {
        self.geometric_product(rhs)
    }
}

impl Add for PgaMotor {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            s: self.s + rhs.s,
            b23: self.b23 + rhs.b23,
            b31: self.b31 + rhs.b31,
            b12: self.b12 + rhs.b12,
            d01: self.d01 + rhs.d01,
            d02: self.d02 + rhs.d02,
            d03: self.d03 + rhs.d03,
            e0123: self.e0123 + rhs.e0123,
        }
    }
}

impl Sub for PgaMotor {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            s: self.s - rhs.s,
            b23: self.b23 - rhs.b23,
            b31: self.b31 - rhs.b31,
            b12: self.b12 - rhs.b12,
            d01: self.d01 - rhs.d01,
            d02: self.d02 - rhs.d02,
            d03: self.d03 - rhs.d03,
            e0123: self.e0123 - rhs.e0123,
        }
    }
}

impl Neg for PgaMotor {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            s: -self.s,
            b23: -self.b23,
            b31: -self.b31,
            b12: -self.b12,
            d01: -self.d01,
            d02: -self.d02,
            d03: -self.d03,
            e0123: -self.e0123,
        }
    }
}
