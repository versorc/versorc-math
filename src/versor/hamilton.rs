// ============================================================================
// VERSORC-MATH: HAMILTON QUATERNIONS & S^3 GEODESIC SLERP INTERPOLATION
// License: Apache-2.0
// ============================================================================

use super::VersorAction;
use std::ops::{Add, Mul, Neg, Sub};

/// Classical Hamilton Quaternion q = w + x*i + y*j + z*k in H ~= Cl_0,2^+.
/// Represents an even 2-versor (rotor) on 3D Euclidean space.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct Quaternion {
    pub w: f32, // Scalar (real) component
    pub x: f32, // i component (bivector -e_23)
    pub y: f32, // j component (bivector -e_31)
    pub z: f32, // k component (bivector -e_12)
}

impl Default for Quaternion {
    #[inline(always)]
    fn default() -> Self {
        Self::identity()
    }
}

impl Quaternion {
    /// Identity versor 1 + 0i + 0j + 0k.
    #[inline(always)]
    pub const fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    #[inline(always)]
    pub const fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Creates a unit rotor versor from rotation angle (radians) and normalized 3D axis.
    pub fn from_axis_angle(axis: &[f32; 3], angle_rad: f32) -> Self {
        let half = angle_rad * 0.5;
        let s = half.sin();
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let scale = if norm > 1e-12 { s / norm } else { 0.0 };
        Self {
            w: half.cos(),
            x: axis[0] * scale,
            y: axis[1] * scale,
            z: axis[2] * scale,
        }
    }

    /// Squared quaternion norm ||q||^2 = w^2 + x^2 + y^2 + z^2.
    #[inline(always)]
    pub fn norm_sq(&self) -> f32 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Quaternion magnitude ||q||.
    #[inline(always)]
    pub fn norm(&self) -> f32 {
        self.norm_sq().sqrt()
    }

    /// Quaternion conjugate \bar{q} = w - xi - yj - zk (Clifford reverse).
    #[inline(always)]
    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Quaternion inverse q^-1 = \bar{q} / ||q||^2.
    #[inline]
    pub fn inverse(&self) -> Self {
        let n_sq = self.norm_sq();
        assert!(n_sq > 1e-15, "Cannot invert zero-norm quaternion");
        let inv = 1.0 / n_sq;
        Self {
            w: self.w * inv,
            x: -self.x * inv,
            y: -self.y * inv,
            z: -self.z * inv,
        }
    }

    /// Normalizes to unit versor on S^3.
    #[inline]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n > 1e-12 {
            let inv = 1.0 / n;
            Self {
                w: self.w * inv,
                x: self.x * inv,
                y: self.y * inv,
                z: self.z * inv,
            }
        } else {
            Self::identity()
        }
    }

    /// Hamilton product: q1 * q2.
    #[inline]
    pub fn hamilton_product(&self, rhs: &Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }

    /// Rotates a 3D vector v via the sandwich product v' = q * v * q^-1.
    /// Utilizes Rodrigues vector-bivector formulation:
    /// v' = v + 2 * q_v x (q_v x v + w * v)
    #[inline]
    pub fn rotate_vector3(&self, v: &[f32; 3]) -> [f32; 3] {
        let qv = [self.x, self.y, self.z];
        // t = 2 * cross(qv, v)
        let t = [
            2.0 * (qv[1] * v[2] - qv[2] * v[1]),
            2.0 * (qv[2] * v[0] - qv[0] * v[2]),
            2.0 * (qv[0] * v[1] - qv[1] * v[0]),
        ];
        // v' = v + w * t + cross(qv, t)
        [
            v[0] + self.w * t[0] + (qv[1] * t[2] - qv[2] * t[1]),
            v[1] + self.w * t[1] + (qv[2] * t[0] - qv[0] * t[2]),
            v[2] + self.w * t[2] + (qv[0] * t[1] - qv[1] * t[0]),
        ]
    }

    /// Geodesic Spherical Linear Interpolation (SLERP) on unit hypersphere S^3.
    /// Follows the shortest geodesic path between unit versors.
    pub fn slerp(q0: Self, mut q1: Self, t: f32) -> Self {
        let mut dot = q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z;

        // If negative, negate q1 to take the shorter arc on S^3
        if dot < 0.0 {
            q1 = -q1;
            dot = -dot;
        }

        // Clamp to avoid numerical domain error in acos
        let dot = dot.clamp(-1.0, 1.0);

        // If very close, use linear interpolation to prevent division by zero
        if dot > 0.9995 {
            let res = q0 * (1.0 - t) + q1 * t;
            return res.normalize();
        }

        let theta_0 = dot.acos();
        let theta = theta_0 * t;
        let sin_theta = theta.sin();
        let sin_theta_0 = theta_0.sin();

        let s0 = (theta_0 - theta).sin() / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;

        (q0 * s0) + (q1 * s1)
    }
}

impl VersorAction<[f32; 3]> for Quaternion {
    #[inline]
    fn transform_in_place(&self, target: &mut [f32; 3]) {
        *target = self.rotate_vector3(target);
    }
}

impl Mul for Quaternion {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.hamilton_product(&rhs)
    }
}

impl Add for Quaternion {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Quaternion {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            w: self.w - rhs.w,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f32> for Quaternion {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self {
        Self {
            w: self.w * s,
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl Neg for Quaternion {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            w: -self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}
