// ============================================================================
// VERSORC-MATH: CAYLEY-KLEIN GEOMETRIC ALGEBRA ENGINE (CK-GA 3D)
// License: Apache-2.0
//
// Unifies Projective Geometric Algebra (PGA, tau = 0), Hyperbolic Clifford
// Algebra (tau < 0, Cl(3,1,0)), and Spherical/Elliptic Clifford Algebra
// (tau > 0, Cl(4,0,0)) under a single continuous Clifford multivector framework.
//
// Basis generators: {e1, e2, e3, e0}
// e1^2 = +1, e2^2 = +1, e3^2 = +1, e0^2 = tau.
//
// Memory Layout: Strictly 64-byte aligned (#[repr(C, align(64))]) to match
// exactly ONE physical L1d/L2 cache line (16 x f32 = 64 bytes).
// ============================================================================

use std::ops::{Add, Mul, Neg, Sub};

/// Bitmasks for `CkMultivector3D` basis ordering: bit 0 = e1, bit 1 = e2, bit 2 = e3, bit 3 = e0.
const CK_MASK: [u8; 16] = [
    0b0000, // 0: 1
    0b0001, // 1: e1
    0b0010, // 2: e2
    0b0100, // 3: e3
    0b1000, // 4: e0
    0b0011, // 5: e12
    0b0110, // 6: e23
    0b0101, // 7: e31
    0b1001, // 8: e01
    0b1010, // 9: e02
    0b1100, // 10: e03
    0b1011, // 11: e012
    0b1110, // 12: e023
    0b1101, // 13: e031
    0b0111, // 14: e123
    0b1111, // 15: e0123
];

/// Sign factor (+1 or -1) converting `CkMultivector3D` basis element `i` into ascending-bit order.
const CK_TO_CANON_SIGN: [f32; 16] = [
    1.0,  // 0: 1
    1.0,  // 1: e1
    1.0,  // 2: e2
    1.0,  // 3: e3
    1.0,  // 4: e0
    1.0,  // 5: e12 = +e1 e2
    1.0,  // 6: e23 = +e2 e3
    -1.0, // 7: e31 = -e1 e3
    -1.0, // 8: e01 = -e1 e0
    -1.0, // 9: e02 = -e2 e0
    -1.0, // 10: e03 = -e3 e0
    1.0,  // 11: e012 = +e1 e2 e0
    1.0,  // 12: e023 = +e2 e3 e0
    -1.0, // 13: e031 = -e1 e3 e0
    1.0,  // 14: e123 = +e1 e2 e3
    -1.0, // 15: e0123 = -e1 e2 e3 e0
];

/// Inverse lookup mapping ascending bitmask (0..16) to `CkMultivector3D` coefficient index.
const MASK_TO_CK_IDX: [usize; 16] = [
    0,  // 0b0000 -> 1
    1,  // 0b0001 -> e1
    2,  // 0b0010 -> e2
    5,  // 0b0011 -> e12
    3,  // 0b0100 -> e3
    7,  // 0b0101 -> e31
    6,  // 0b0110 -> e23
    14, // 0b0111 -> e123
    4,  // 0b1000 -> e0
    8,  // 0b1001 -> e01
    9,  // 0b1010 -> e02
    11, // 0b1011 -> e012
    10, // 0b1100 -> e03
    13, // 0b1101 -> e031
    12, // 0b1110 -> e023
    15, // 0b1111 -> e0123
];

/// Canonical ascending-bit blade product swap parity.
#[inline(always)]
fn ck_canonical_swap_sign(mask_a: u8, mask_b: u8) -> f32 {
    let mut exact_swaps = 0u32;
    for shift in 1..4u8 {
        if (mask_a >> shift) & 1 != 0 {
            exact_swaps += (mask_b & ((1u8 << shift) - 1)).count_ones();
        }
    }
    if exact_swaps.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// 64-byte cache-line aligned Multivector in 3D Cayley-Klein Clifford Algebra.
/// 16 coefficients corresponding to the 2^4 = 16 basis blades.
///
/// Blade Index Ordering:
/// - 0:  1 (Scalar)
/// - 1:  e1
/// - 2:  e2
/// - 3:  e3
/// - 4:  e0 (Projective generator, e0^2 = tau)
/// - 5:  e12 = e1 ^ e2
/// - 6:  e23 = e2 ^ e3
/// - 7:  e31 = e3 ^ e1
/// - 8:  e01 = e0 ^ e1
/// - 9:  e02 = e0 ^ e2
/// - 10: e03 = e0 ^ e3
/// - 11: e012 = e0 ^ e1 ^ e2
/// - 12: e023 = e0 ^ e2 ^ e3
/// - 13: e031 = e0 ^ e3 ^ e1
/// - 14: e123 = e1 ^ e2 ^ e3 (Spatial pseudoscalar)
/// - 15: e0123 = I (Full pseudoscalar)
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CkMultivector3D {
    pub c: [f32; 16],
}

impl Default for CkMultivector3D {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl CkMultivector3D {
    #[inline(always)]
    pub const fn zero() -> Self {
        Self { c: [0.0f32; 16] }
    }

    #[inline(always)]
    pub fn scalar(val: f32) -> Self {
        let mut m = Self::zero();
        m.c[0] = val;
        m
    }

    /// Grade 1 Basis Vector e1.
    #[inline(always)]
    pub fn e1() -> Self {
        let mut m = Self::zero();
        m.c[1] = 1.0;
        m
    }

    /// Grade 1 Basis Vector e2.
    #[inline(always)]
    pub fn e2() -> Self {
        let mut m = Self::zero();
        m.c[2] = 1.0;
        m
    }

    /// Grade 1 Basis Vector e3.
    #[inline(always)]
    pub fn e3() -> Self {
        let mut m = Self::zero();
        m.c[3] = 1.0;
        m
    }

    /// Grade 1 Projective Origin Generator e0 (where e0^2 = tau).
    #[inline(always)]
    pub fn e0() -> Self {
        let mut m = Self::zero();
        m.c[4] = 1.0;
        m
    }

    /// Clifford Reverse: A~
    /// Flips sign of blades with grade k = 2, 3: sign = (-1)^{k(k-1)/2}.
    #[inline]
    pub fn reverse(&self) -> Self {
        let mut res = *self;
        // Grade 2 blades: e12, e23, e31, e01, e02, e03
        for c in &mut res.c[5..=10] {
            *c = -*c;
        }
        // Grade 3 blades: e012, e023, e031, e123
        for c in &mut res.c[11..=14] {
            *c = -*c;
        }
        res
    }

    /// Computes the Cayley-Klein geometric product `self * rhs` under curvature metric `tau` (`e0^2 = tau`).
    #[inline]
    pub fn geometric_product(&self, rhs: &Self, tau: f32) -> Self {
        let mut res = Self::zero();
        for (i, &a) in self.c.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let ma = CK_MASK[i];
            let sa = CK_TO_CANON_SIGN[i];
            for (j, &b) in rhs.c.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
                let mb = CK_MASK[j];
                let k = MASK_TO_CK_IDX[(ma ^ mb) as usize];
                let metric = if ((ma & mb) & 0b1000) != 0 { tau } else { 1.0 };
                let sign =
                    sa * CK_TO_CANON_SIGN[j] * CK_TO_CANON_SIGN[k] * ck_canonical_swap_sign(ma, mb);
                res.c[k] += a * b * metric * sign;
            }
        }
        res
    }

    /// Exterior wedge product `self ^ rhs` (metric-independent).
    #[inline]
    pub fn wedge(&self, rhs: &Self) -> Self {
        let mut res = Self::zero();
        for (i, &a) in self.c.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let ma = CK_MASK[i];
            let sa = CK_TO_CANON_SIGN[i];
            for (j, &b) in rhs.c.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
                let mb = CK_MASK[j];
                if (ma & mb) != 0 {
                    continue;
                }
                let k = MASK_TO_CK_IDX[(ma ^ mb) as usize];
                let sign =
                    sa * CK_TO_CANON_SIGN[j] * CK_TO_CANON_SIGN[k] * ck_canonical_swap_sign(ma, mb);
                res.c[k] += a * b * sign;
            }
        }
        res
    }

    /// Scalar product `<A * B~>_0` under curvature `tau`.
    #[inline]
    pub fn scalar_product(&self, rhs: &Self, tau: f32) -> f32 {
        self.geometric_product(&rhs.reverse(), tau).c[0]
    }

    /// Cayley-Klein sandwich product `self * x * self~` under curvature `tau`.
    #[inline]
    pub fn sandwich(&self, x: &Self, tau: f32) -> Self {
        self.geometric_product(x, tau)
            .geometric_product(&self.reverse(), tau)
    }

    /// Euclidean coefficient L2 norm (positive-definite for residual verification).
    #[inline]
    pub fn euclidean_norm(&self) -> f32 {
        self.c.iter().map(|&v| v * v).sum::<f32>().sqrt()
    }
}

impl Add for CkMultivector3D {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut out = Self::zero();
        for (o, (&a, &b)) in out.c.iter_mut().zip(self.c.iter().zip(rhs.c.iter())) {
            *o = a + b;
        }
        out
    }
}

impl Sub for CkMultivector3D {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut out = Self::zero();
        for (o, (&a, &b)) in out.c.iter_mut().zip(self.c.iter().zip(rhs.c.iter())) {
            *o = a - b;
        }
        out
    }
}

impl Neg for CkMultivector3D {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut out = Self::zero();
        for (o, &a) in out.c.iter_mut().zip(self.c.iter()) {
            *o = -a;
        }
        out
    }
}

impl Mul<f32> for CkMultivector3D {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        let mut out = Self::zero();
        for (o, &a) in out.c.iter_mut().zip(self.c.iter()) {
            *o = a * rhs;
        }
        out
    }
}

/// Homogeneous Point in 3D Cayley-Klein Space: P = x e1 + y e2 + z e3 + w e0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CkPoint3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl CkPoint3D {
    /// Creates a normalized point in CK space (w = 1.0).
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, w: 1.0 }
    }

    /// Origin of CK space (0, 0, 0, 1).
    #[inline(always)]
    pub fn origin() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }

    /// Converts to multivector representation P = x e1 + y e2 + z e3 + w e0.
    #[inline]
    pub fn to_multivector(&self) -> CkMultivector3D {
        let mut m = CkMultivector3D::zero();
        m.c[1] = self.x;
        m.c[2] = self.y;
        m.c[3] = self.z;
        m.c[4] = self.w;
        m
    }

    /// Converts to dual/trivector point representation valid across all `tau` (`tau < 0, == 0, > 0`):
    /// P_tri = w * e123 - x * e023 - y * e031 - z * e012.
    #[inline]
    pub fn to_trivector(&self) -> CkMultivector3D {
        let mut m = CkMultivector3D::zero();
        m.c[14] = self.w; // e123
        m.c[12] = -self.x; // e023
        m.c[13] = -self.y; // e031
        m.c[11] = -self.z; // e012
        m
    }

    /// Extracts a homogeneous `CkPoint3D` from a multivector's grade-3 trivector part.
    #[inline]
    pub fn from_trivector(mv: &CkMultivector3D) -> Self {
        Self {
            x: -mv.c[12],
            y: -mv.c[13],
            z: -mv.c[11],
            w: mv.c[14],
        }
    }

    /// Computes the Clifford inner product norm squared P ._tau P:
    /// ||P||_tau^2 = x^2 + y^2 + z^2 + tau * w^2.
    #[inline(always)]
    pub fn norm_sq(&self, tau: f32) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + tau * (self.w * self.w)
    }

    /// Computes the Cayley-Klein Clifford inner product P ._tau Q:
    /// P ._tau Q = x1*x2 + y1*y2 + z1*z2 + tau * w1*w2.
    #[inline(always)]
    pub fn inner_product(&self, other: &Self, tau: f32) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + tau * (self.w * other.w)
    }

    /// Computes the exact projective Cayley-Klein geodesic distance between two points:
    /// Preserved identically by `CkMotor3D::transform_point` for all curvatures `tau`:
    /// - tau < 0 (Beltrami-Klein hyperbolic):
    ///   d_tau(P, Q) = (1 / sqrt(|tau|)) * arcosh((1 - |tau| (p . q)) / sqrt((1 - |tau| ||p||^2)(1 - |tau| ||q||^2)))
    /// - tau == 0 (Euclidean):
    ///   d_0(P, Q) = ||p - q||
    /// - tau > 0 (Gnomonic spherical / elliptic):
    ///   d_tau(P, Q) = (1 / sqrt(tau)) * arccos(|1 + tau (p . q)| / sqrt((1 + tau ||p||^2)(1 + tau ||q||^2)))
    pub fn geodesic_distance(&self, other: &Self, tau: f32) -> f32 {
        let px = self.x / self.w;
        let py = self.y / self.w;
        let pz = self.z / self.w;
        let qx = other.x / other.w;
        let qy = other.y / other.w;
        let qz = other.z / other.w;

        let dx = px - qx;
        let dy = py - qy;
        let dz = pz - qz;
        let diff_sq = dx * dx + dy * dy + dz * dz;

        if tau.abs() < 1e-7 {
            return diff_sq.sqrt();
        }

        let p_sq = px * px + py * py + pz * pz;
        let q_sq = qx * qx + qy * qy + qz * qz;
        let p_dot_q = px * qx + py * qy + pz * qz;

        if tau < 0.0 {
            let abs_tau = -tau;
            let inv_sqrt_tau = 1.0f32 / abs_tau.sqrt();

            let u_factor = (1.0f32 - abs_tau * p_sq).max(1e-7);
            let v_factor = (1.0f32 - abs_tau * q_sq).max(1e-7);
            let denom = (u_factor * v_factor).sqrt();

            let arg = ((1.0f32 - abs_tau * p_dot_q) / denom).max(1.0f32);
            arg.acosh() * inv_sqrt_tau
        } else {
            let inv_sqrt_tau = 1.0f32 / tau.sqrt();

            let u_factor = 1.0f32 + tau * p_sq;
            let v_factor = 1.0f32 + tau * q_sq;
            let denom = (u_factor * v_factor).sqrt().max(1e-7);

            let arg = ((1.0f32 + tau * p_dot_q).abs() / denom).clamp(-1.0f32, 1.0f32);
            inv_sqrt_tau * arg.acos()
        }
    }

    /// Evaluates the projective Exterior Wedge Join L = P ^ Q (forms a CkLine3D bivector).
    #[inline]
    pub fn join(&self, other: &Self) -> CkLine3D {
        // Spatial direction v = p_spatial * q_w - q_spatial * p_w
        let vx = self.y * other.z - self.z * other.y;
        let vy = self.z * other.x - self.x * other.z;
        let vz = self.x * other.y - self.y * other.x;

        // Projective moment m = p_spatial * q_0 - q_spatial * p_0
        let mx = self.x * other.w - other.x * self.w;
        let my = self.y * other.w - other.y * self.w;
        let mz = self.z * other.w - other.z * self.w;

        CkLine3D {
            vx,
            vy,
            vz,
            mx,
            my,
            mz,
        }
    }
}

/// Plane in 3D Cayley-Klein Space: H = nx e1 + ny e2 + nz e3 - d e0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CkPlane3D {
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub d: f32,
}

impl CkPlane3D {
    /// Creates a plane with normal (nx, ny, nz) and distance d from origin.
    #[inline(always)]
    pub fn new(nx: f32, ny: f32, nz: f32, d: f32) -> Self {
        Self { nx, ny, nz, d }
    }

    /// Norm squared H ._tau H = nx^2 + ny^2 + nz^2 + tau * d^2.
    #[inline(always)]
    pub fn norm_sq(&self, tau: f32) -> f32 {
        self.nx * self.nx + self.ny * self.ny + self.nz * self.nz + tau * (self.d * self.d)
    }

    /// Reflects a point P across this plane using the universal 1-versor sandwich:
    /// P' = - H P H^-1 = P - 2 * (P ._tau H / (H ._tau H)) * H.
    pub fn reflect_point(&self, p: &CkPoint3D, tau: f32) -> CkPoint3D {
        let h_norm_sq = self.norm_sq(tau);
        assert!(
            h_norm_sq.abs() > 1e-12,
            "Degenerate plane in Cayley-Klein reflection"
        );

        // P ._tau H = nx * px + ny * py + nz * pz - tau * d * pw
        let p_dot_h = self.nx * p.x + self.ny * p.y + self.nz * p.z - tau * self.d * p.w;
        let scale = 2.0f32 * p_dot_h / h_norm_sq;

        CkPoint3D {
            x: p.x - scale * self.nx,
            y: p.y - scale * self.ny,
            z: p.z - scale * self.nz,
            w: p.w + scale * self.d,
        }
    }
}

/// Geodesic Line in 3D Cayley-Klein Space (Plücker Bivector):
/// L = vx e23 + vy e31 + vz e12 + mx e01 + my e02 + mz e03.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CkLine3D {
    /// Direction components (spatial bivectors)
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    /// Moment components (projective bivectors)
    pub mx: f32,
    pub my: f32,
    pub mz: f32,
}

impl CkLine3D {
    /// Plücker condition: v . m == 0 (must be valid line blade).
    #[inline(always)]
    pub fn plucker_inner(&self) -> f32 {
        self.vx * self.mx + self.vy * self.my + self.vz * self.mz
    }

    /// Converts line to full multivector representation.
    #[inline]
    pub fn to_multivector(&self) -> CkMultivector3D {
        let mut m = CkMultivector3D::zero();
        m.c[6] = self.vx; // e23
        m.c[7] = self.vy; // e31
        m.c[5] = self.vz; // e12
        m.c[8] = self.mx; // e01
        m.c[9] = self.my; // e02
        m.c[10] = self.mz; // e03
        m
    }
}

/// Unified 3D Cayley-Klein Motor (2-Versor):
/// M = scalar + B_rot + B_trans + pseudoscalar
/// Represents universal rigid body motion / isometries across Euclidean,
/// Hyperbolic, and Spherical spaces under sandwich action: P |-> M P M~.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CkMotor3D {
    pub s: f32,
    pub rot: [f32; 3],   // [e23, e31, e12]
    pub trans: [f32; 3], // [e01, e02, e03]
    pub p: f32,          // I (pseudoscalar)
}

impl Default for CkMotor3D {
    #[inline(always)]
    fn default() -> Self {
        Self::identity()
    }
}

impl CkMotor3D {
    /// Identity motor M = 1.
    #[inline(always)]
    pub fn identity() -> Self {
        Self {
            s: 1.0,
            rot: [0.0; 3],
            trans: [0.0; 3],
            p: 0.0,
        }
    }

    /// Constructs a pure spatial rotation motor around unit axis (ux, uy, uz) by angle theta:
    /// R = cos(theta/2) - sin(theta/2) (ux e23 + uy e31 + uz e12)
    pub fn from_rotation(axis: [f32; 3], theta: f32) -> Self {
        let half = theta * 0.5f32;
        let c = half.cos();
        let s = half.sin();
        Self {
            s: c,
            rot: [-s * axis[0], -s * axis[1], -s * axis[2]],
            trans: [0.0; 3],
            p: 0.0,
        }
    }

    /// Constructs a pure geodesic translation motor along unit direction (dx, dy, dz)
    /// by distance delta under curvature parameter tau:
    /// - If tau == 0 (Euclidean):
    ///   T = 1 - (delta/2) * (e0 ^ d)
    /// - If tau < 0 (Hyperbolic boost):
    ///   T = cosh(delta*sqrt(|tau|)/2) - (sinh(delta*sqrt(|tau|)/2)/sqrt(|tau|)) * (e0 ^ d)
    /// - If tau > 0 (Spherical translation):
    ///   T = cos(delta*sqrt(tau)/2) - (sin(delta*sqrt(tau)/2)/sqrt(tau)) * (e0 ^ d)
    pub fn from_translation(direction: [f32; 3], delta: f32, tau: f32) -> Self {
        let half_d = delta * 0.5f32;

        if tau.abs() < 1e-7 {
            Self {
                s: 1.0,
                rot: [0.0; 3],
                trans: [
                    -half_d * direction[0],
                    -half_d * direction[1],
                    -half_d * direction[2],
                ],
                p: 0.0,
            }
        } else if tau < 0.0 {
            let sqrt_abs = (-tau).sqrt();
            let arg = half_d * sqrt_abs;
            let c = arg.cosh();
            let s = arg.sinh() / sqrt_abs;
            Self {
                s: c,
                rot: [0.0; 3],
                trans: [-s * direction[0], -s * direction[1], -s * direction[2]],
                p: 0.0,
            }
        } else {
            let sqrt_tau = tau.sqrt();
            let arg = half_d * sqrt_tau;
            let c = arg.cos();
            let s = arg.sin() / sqrt_tau;
            Self {
                s: c,
                rot: [0.0; 3],
                trans: [-s * direction[0], -s * direction[1], -s * direction[2]],
                p: 0.0,
            }
        }
    }

    /// Converts this motor into its full 16-blade `CkMultivector3D` representation.
    #[inline]
    pub fn to_multivector(&self) -> CkMultivector3D {
        let mut m = CkMultivector3D::zero();
        m.c[0] = self.s;
        m.c[6] = self.rot[0]; // e23
        m.c[7] = self.rot[1]; // e31
        m.c[5] = self.rot[2]; // e12
        m.c[8] = self.trans[0]; // e01
        m.c[9] = self.trans[1]; // e02
        m.c[10] = self.trans[2]; // e03
        m.c[15] = self.p; // e0123
        m
    }

    /// Reconstructs a `CkMotor3D` from the even subalgebra Cl+(3, 0, tau) of a multivector.
    #[inline]
    pub fn from_multivector(mv: &CkMultivector3D) -> Self {
        Self {
            s: mv.c[0],
            rot: [mv.c[6], mv.c[7], mv.c[5]],
            trans: [mv.c[8], mv.c[9], mv.c[10]],
            p: mv.c[15],
        }
    }

    /// Clifford Reverse of the motor: M~ (negates bivector grades, keeps scalar and pseudoscalar).
    #[inline]
    pub fn reverse(&self) -> Self {
        Self {
            s: self.s,
            rot: [-self.rot[0], -self.rot[1], -self.rot[2]],
            trans: [-self.trans[0], -self.trans[1], -self.trans[2]],
            p: self.p,
        }
    }

    /// Composes two Cayley-Klein motors via geometric product `self * rhs` under metric `tau`.
    #[inline]
    pub fn geometric_product(&self, rhs: &Self, tau: f32) -> Self {
        Self::from_multivector(
            &self
                .to_multivector()
                .geometric_product(&rhs.to_multivector(), tau),
        )
    }

    /// Normalizes the motor under the Study quadric in metric `tau` so `M * M~ = 1`.
    #[inline]
    pub fn normalized(&self, tau: f32) -> Self {
        let mm = self.geometric_product(&self.reverse(), tau);
        let alpha = mm.s;
        let beta = mm.p;
        let study_sq = alpha * alpha - tau * beta * beta;
        if study_sq <= 1e-12 {
            return Self::identity();
        }
        let u = study_sq.sqrt();
        let sum = alpha + u;
        if sum <= 1e-12 {
            return Self::identity();
        }
        let r = (0.5 * sum).sqrt();
        let s_part = beta / (2.0 * r);
        let inv_u = 1.0 / u;
        let inv_sqrt_study = Self {
            s: r * inv_u,
            rot: [0.0; 3],
            trans: [0.0; 3],
            p: -s_part * inv_u,
        };
        self.geometric_product(&inv_sqrt_study, tau)
    }

    /// Exact algebraic inverse `M^-1` under metric `tau` via Study conjugate.
    #[inline]
    pub fn inverse(&self, tau: f32) -> Self {
        let rev = self.reverse();
        let mm = self.geometric_product(&rev, tau);
        let denom = mm.s * mm.s - tau * mm.p * mm.p;
        if denom.abs() < 1e-12 {
            return Self::identity();
        }
        let inv_denom = 1.0 / denom;
        let mm_inv = Self {
            s: mm.s * inv_denom,
            rot: [0.0; 3],
            trans: [0.0; 3],
            p: -mm.p * inv_denom,
        };
        rev.geometric_product(&mm_inv, tau)
    }

    /// Applies the exact versor sandwich transformation P |-> M * P * M~ under metric `tau`.
    pub fn transform_point(&self, p: &CkPoint3D, tau: f32) -> CkPoint3D {
        let m = self.to_multivector();
        let p_tri = p.to_trivector();
        let transformed = CkPoint3D::from_trivector(&m.sandwich(&p_tri, tau));

        if transformed.w.abs() > 1e-7 {
            let inv_w = 1.0 / transformed.w;
            CkPoint3D {
                x: transformed.x * inv_w,
                y: transformed.y * inv_w,
                z: transformed.z * inv_w,
                w: 1.0,
            }
        } else {
            transformed
        }
    }
}
