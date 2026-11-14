// ============================================================================
// VERSORC-MATH: DUAL QUATERNIONS (H \otimes D) & RIGID SE(3) KINEMATICS
// License: Apache-2.0
// ============================================================================

use super::hamilton::Quaternion;
use crate::pga::PgaMotor;
use std::ops::{Add, Mul, Neg, Sub};

/// Dual Quaternion Q = q_r + \epsilon * q_d in H \otimes D ~= Cl+(3, 0, 1).
/// Isomorphic to the even subalgebra of 3D PGA, uniting rotation and translation into a single 8-D element.
/// Memory layout: 32 bytes (16 bytes real quaternion + 16 bytes dual quaternion), 32-byte aligned.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct DualQuaternion {
    pub real: Quaternion, // Rotation part q_r
    pub dual: Quaternion, // Translation displacement part q_d
}

impl Default for DualQuaternion {
    #[inline(always)]
    fn default() -> Self {
        Self::identity()
    }
}

impl DualQuaternion {
    /// Identity dual quaternion (1 + 0\epsilon).
    #[inline(always)]
    pub const fn identity() -> Self {
        Self {
            real: Quaternion::identity(),
            dual: Quaternion::new(0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Construct from explicit real and dual quaternions.
    #[inline(always)]
    pub const fn new(real: Quaternion, dual: Quaternion) -> Self {
        Self { real, dual }
    }

    /// Constructs a unit dual quaternion from rotation quaternion and 3D translation vector:
    /// q_r = rotation, q_d = 0.5 * (0 + tx*i + ty*j + tz*k) * q_r.
    #[inline]
    pub fn from_rotation_translation(rotation: &Quaternion, translation: &[f32; 3]) -> Self {
        let q_r = rotation.normalize();
        let t_quat = Quaternion::new(0.0, translation[0], translation[1], translation[2]);
        let half_t_qr = t_quat * q_r;

        Self {
            real: q_r,
            dual: Quaternion::new(
                0.5 * half_t_qr.w,
                0.5 * half_t_qr.x,
                0.5 * half_t_qr.y,
                0.5 * half_t_qr.z,
            ),
        }
    }

    /// Constructs a pure translation dual quaternion (no rotation).
    #[inline]
    pub fn from_translation(tx: f32, ty: f32, tz: f32) -> Self {
        Self::from_rotation_translation(&Quaternion::identity(), &[tx, ty, tz])
    }

    /// Constructs a pure rotation dual quaternion (no translation).
    #[inline]
    pub fn from_rotation(axis: &[f32; 3], angle_rad: f32) -> Self {
        let q = Quaternion::from_axis_angle(axis, angle_rad);
        Self::new(q, Quaternion::new(0.0, 0.0, 0.0, 0.0))
    }

    /// Decomposes this dual quaternion into rotation quaternion and translation vector [tx, ty, tz]:
    /// translation = 2 * q_d * q_r^*.
    #[inline]
    pub fn to_rotation_translation(&self) -> (Quaternion, [f32; 3]) {
        let q_r = self.real.normalize();
        let q_r_conj = q_r.conjugate();
        let t_quat = (self.dual * q_r_conj) * 2.0;

        (q_r, [t_quat.x, t_quat.y, t_quat.z])
    }

    /// Transforms an affine 3D point: p' = R(p) + t.
    #[inline]
    pub fn transform_point(&self, p: &[f32; 3]) -> [f32; 3] {
        let rot_p = self.real.rotate_vector3(p);
        let (_, t) = self.to_rotation_translation();
        [rot_p[0] + t[0], rot_p[1] + t[1], rot_p[2] + t[2]]
    }

    /// Transforms a 3D direction vector (applies rotation only, ignoring translation).
    #[inline]
    pub fn transform_vector(&self, v: &[f32; 3]) -> [f32; 3] {
        self.real.rotate_vector3(v)
    }

    /// Quaternion conjugate: q_r^* + \epsilon * q_d^*.
    #[inline(always)]
    pub fn quaternion_conjugate(&self) -> Self {
        Self {
            real: self.real.conjugate(),
            dual: self.dual.conjugate(),
        }
    }

    /// Dual conjugate: q_r - \epsilon * q_d.
    #[inline(always)]
    pub fn dual_conjugate(&self) -> Self {
        Self {
            real: self.real,
            dual: -self.dual,
        }
    }

    /// Total (Clifford) conjugate: q_r^* - \epsilon * q_d^*.
    #[inline(always)]
    pub fn total_conjugate(&self) -> Self {
        Self {
            real: self.real.conjugate(),
            dual: -self.dual.conjugate(),
        }
    }

    /// Dual norm: returns (real_norm, dual_norm_projection).
    #[inline]
    pub fn norm(&self) -> (f32, f32) {
        let r_norm = self.real.norm();
        if r_norm > 1e-12 {
            let dot = self.real.w * self.dual.w
                + self.real.x * self.dual.x
                + self.real.y * self.dual.y
                + self.real.z * self.dual.z;
            (r_norm, dot / r_norm)
        } else {
            (0.0, 0.0)
        }
    }

    /// Normalizes this dual quaternion to satisfy ||q_r|| = 1 and q_r . q_d = 0.
    #[inline]
    pub fn normalize(&self) -> Self {
        let mag = self.real.norm();
        if mag > 1e-12 {
            let inv_mag = 1.0 / mag;
            let norm_real = self.real * inv_mag;
            let dot = norm_real.w * self.dual.w
                + norm_real.x * self.dual.x
                + norm_real.y * self.dual.y
                + norm_real.z * self.dual.z;

            let norm_dual = Quaternion::new(
                (self.dual.w - norm_real.w * dot) * inv_mag,
                (self.dual.x - norm_real.x * dot) * inv_mag,
                (self.dual.y - norm_real.y * dot) * inv_mag,
                (self.dual.z - norm_real.z * dot) * inv_mag,
            );

            Self {
                real: norm_real,
                dual: norm_dual,
            }
        } else {
            Self::identity()
        }
    }

    /// Inverse dual quaternion Q^-1. For unit dual quaternions, Q^-1 = Q^*.
    #[inline]
    pub fn inverse(&self) -> Self {
        self.normalize().quaternion_conjugate()
    }

    /// Dual Quaternion Linear Blending (DLB) with weights summing to 1.
    /// Used globally in GPU skeletal animation and robotics blending.
    #[inline]
    pub fn dlb(weights: &[f32], dqs: &[Self]) -> Self {
        assert_eq!(
            weights.len(),
            dqs.len(),
            "Weights and dual quaternions length mismatch"
        );
        assert!(!dqs.is_empty(), "Cannot blend empty dual quaternions");

        let mut sum_real = Quaternion::new(0.0, 0.0, 0.0, 0.0);
        let mut sum_dual = Quaternion::new(0.0, 0.0, 0.0, 0.0);

        let q0 = &dqs[0];

        for (i, (&w, dq)) in weights.iter().zip(dqs.iter()).enumerate() {
            // Hemisphere alignment: ensure inner product with first real quaternion >= 0
            let sign = if i > 0
                && (dq.real.w * q0.real.w
                    + dq.real.x * q0.real.x
                    + dq.real.y * q0.real.y
                    + dq.real.z * q0.real.z)
                    < 0.0
            {
                -1.0
            } else {
                1.0
            };
            let eff_w = w * sign;

            sum_real.w += eff_w * dq.real.w;
            sum_real.x += eff_w * dq.real.x;
            sum_real.y += eff_w * dq.real.y;
            sum_real.z += eff_w * dq.real.z;

            sum_dual.w += eff_w * dq.dual.w;
            sum_dual.x += eff_w * dq.dual.x;
            sum_dual.y += eff_w * dq.dual.y;
            sum_dual.z += eff_w * dq.dual.z;
        }

        Self::new(sum_real, sum_dual).normalize()
    }

    /// Screw Linear Interpolation (ScLERP) along the geodesic screw axis between two unit dual quaternions.
    #[inline]
    pub fn sclerp(&self, other: &Self, t: f32) -> Self {
        // Convert to PgaMotor, run PGA ScLERP, convert back
        let m1 = PgaMotor::from(*self);
        let m2 = PgaMotor::from(*other);
        let interp = m1.sclerp(&m2, t);
        DualQuaternion::from(interp)
    }
}

impl Mul for DualQuaternion {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // (A_r + \epsilon A_d)(B_r + \epsilon B_d) = (A_r B_r) + \epsilon (A_r B_d + A_d B_r)
        let real_part = self.real * rhs.real;
        let dual_part = (self.real * rhs.dual) + (self.dual * rhs.real);
        Self {
            real: real_part,
            dual: dual_part,
        }
    }
}

impl Add for DualQuaternion {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            real: self.real + rhs.real,
            dual: self.dual + rhs.dual,
        }
    }
}

impl Sub for DualQuaternion {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            real: self.real - rhs.real,
            dual: self.dual - rhs.dual,
        }
    }
}

impl Neg for DualQuaternion {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            real: -self.real,
            dual: -self.dual,
        }
    }
}

// ============================================================================
// BIJECTION: PgaMotor <---> DualQuaternion
// ============================================================================

impl From<PgaMotor> for DualQuaternion {
    #[inline]
    fn from(m: PgaMotor) -> Self {
        // PgaMotor: s, b23, b31, b12, d01, d02, d03, e0123
        // Hamilton quaternion: w = s, x = -b23, y = -b31, z = -b12
        // Dual quaternion: w_d = -e0123, x_d = -d01, y_d = -d02, z_d = -d03
        Self {
            real: Quaternion::new(m.s, -m.b23, -m.b31, -m.b12),
            dual: Quaternion::new(-m.e0123, -m.d01, -m.d02, -m.d03),
        }
    }
}

impl From<DualQuaternion> for PgaMotor {
    #[inline]
    fn from(dq: DualQuaternion) -> Self {
        PgaMotor::new(
            dq.real.w, -dq.real.x, -dq.real.y, -dq.real.z, -dq.dual.x, -dq.dual.y, -dq.dual.z,
            -dq.dual.w,
        )
    }
}
