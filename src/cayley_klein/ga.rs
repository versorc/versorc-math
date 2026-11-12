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
        for i in 5..=10 {
            res.c[i] = -res.c[i];
        }
        // Grade 3 blades: e012, e023, e031, e123
        for i in 11..=14 {
            res.c[i] = -res.c[i];
        }
        res
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
        Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
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

    /// Computes the exact Cayley-Klein geodesic distance between two points:
    /// - tau < 0 (Hyperbolic):
    ///   d_tau(P, Q) = (1 / sqrt(|tau|)) * arcosh(1 + (2 |tau| ||p - q||^2) / ((1 - |tau| ||p||^2)(1 - |tau| ||q||^2)))
    /// - tau == 0 (Euclidean):
    ///   d_0(P, Q) = ||p - q||
    /// - tau > 0 (Spherical):
    ///   d_tau(P, Q) = (1 / sqrt(tau)) * arccos(1 - (2 tau ||p - q||^2) / ((1 + tau ||p||^2)(1 + tau ||q||^2)))
    pub fn geodesic_distance(&self, other: &Self, tau: f32) -> f32 {
        let dx = self.x / self.w - other.x / other.w;
        let dy = self.y / self.w - other.y / other.w;
        let dz = self.z / self.w - other.z / other.w;
        let diff_sq = dx * dx + dy * dy + dz * dz;

        if tau.abs() < 1e-7 {
            return diff_sq.sqrt();
        }

        let p_sq = (self.x * self.x + self.y * self.y + self.z * self.z) / (self.w * self.w);
        let q_sq = (other.x * other.x + other.y * other.y + other.z * other.z) / (other.w * other.w);

        if tau < 0.0 {
            let abs_tau = -tau;
            let inv_sqrt_tau = 1.0f32 / abs_tau.sqrt();

            let u_factor = (1.0f32 - abs_tau * p_sq).max(1e-7);
            let v_factor = (1.0f32 - abs_tau * q_sq).max(1e-7);
            let denom = u_factor * v_factor;

            let delta = 1.0f32 + (2.0f32 * abs_tau * diff_sq) / denom;
            let delta_clamped = delta.max(1.0f32);
            let arcosh = (delta_clamped + (delta_clamped * delta_clamped - 1.0f32).sqrt()).ln();
            inv_sqrt_tau * arcosh
        } else {
            let inv_sqrt_tau = 1.0f32 / tau.sqrt();

            let u_factor = 1.0f32 + tau * p_sq;
            let v_factor = 1.0f32 + tau * q_sq;
            let denom = (u_factor * v_factor).max(1e-7);

            let arg = 1.0f32 - (2.0f32 * tau * diff_sq) / denom;
            let arg_clamped = arg.clamp(-1.0f32, 1.0f32);
            inv_sqrt_tau * arg_clamped.acos()
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
        assert!(h_norm_sq.abs() > 1e-12, "Degenerate plane in Cayley-Klein reflection");

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
    /// R = cos(theta/2) + sin(theta/2) (ux e23 + uy e31 + uz e12)
    pub fn from_rotation(axis: [f32; 3], theta: f32) -> Self {
        let half = theta * 0.5f32;
        let c = half.cos();
        let s = half.sin();
        Self {
            s: c,
            rot: [s * axis[0], s * axis[1], s * axis[2]],
            trans: [0.0; 3],
            p: 0.0,
        }
    }

    /// Constructs a pure geodesic translation motor along unit direction (dx, dy, dz)
    /// by distance delta under curvature parameter tau:
    /// - If tau == 0 (Euclidean):
    ///   T = 1 - (delta/2) * (d ^ e0)
    /// - If tau < 0 (Hyperbolic boost):
    ///   T = cosh(delta*sqrt(|tau|)/2) - (sinh(delta*sqrt(|tau|)/2)/sqrt(|tau|)) * (d ^ e0)
    /// - If tau > 0 (Spherical rotation):
    ///   T = cos(delta*sqrt(tau)/2) - (sin(delta*sqrt(tau)/2)/sqrt(tau)) * (d ^ e0)
    pub fn from_translation(direction: [f32; 3], delta: f32, tau: f32) -> Self {
        let half_d = delta * 0.5f32;

        if tau.abs() < 1e-7 {
            // Euclidean PGA exact translation: T = 1 + (delta/2) * (d ^ e0)
            Self {
                s: 1.0,
                rot: [0.0; 3],
                trans: [half_d * direction[0], half_d * direction[1], half_d * direction[2]],
                p: 0.0,
            }
        } else if tau < 0.0 {
            // Hyperbolic boost: T = cosh(delta*sqrt(|tau|)/2) + (sinh(delta*sqrt(|tau|)/2)/sqrt(|tau|)) * (d ^ e0)
            let sqrt_abs = (-tau).sqrt();
            let arg = half_d * sqrt_abs;
            let c = arg.cosh();
            let s = arg.sinh() / sqrt_abs;
            Self {
                s: c,
                rot: [0.0; 3],
                trans: [s * direction[0], s * direction[1], s * direction[2]],
                p: 0.0,
            }
        } else {
            // Spherical translation (rotation on S3): T = cos(delta*sqrt(tau)/2) + (sin(delta*sqrt(tau)/2)/sqrt(tau)) * (d ^ e0)
            let sqrt_tau = tau.sqrt();
            let arg = half_d * sqrt_tau;
            let c = arg.cos();
            let s = arg.sin() / sqrt_tau;
            Self {
                s: c,
                rot: [0.0; 3],
                trans: [s * direction[0], s * direction[1], s * direction[2]],
                p: 0.0,
            }
        }
    }

    /// Applies the versor sandwich transformation P |-> M * P * M~ to point P.
    pub fn transform_point(&self, p: &CkPoint3D, tau: f32) -> CkPoint3D {
        // Exact versor sandwich evaluated directly in O(1) FLOPs
        // 1. Spatial rotation
        let rx = self.rot[0];
        let ry = self.rot[1];
        let rz = self.rot[2];
        let s = self.s;

        // Rotate point: p_rot = R * p * R~
        let px = p.x;
        let py = p.y;
        let pz = p.z;

        let tx = 2.0 * (ry * pz - rz * py);
        let ty = 2.0 * (rz * px - rx * pz);
        let tz = 2.0 * (rx * py - ry * px);

        let x_rot = px + s * tx + (ry * tz - rz * ty);
        let y_rot = py + s * ty + (rz * tx - rx * tz);
        let z_rot = pz + s * tz + (rx * ty - ry * tx);

        // 2. Translational motor action
        let mx = self.trans[0];
        let my = self.trans[1];
        let mz = self.trans[2];

        // Translation shifts coordinates proportionally to w
        let x_trans = x_rot * s + (2.0 * mx * p.w);
        let y_trans = y_rot * s + (2.0 * my * p.w);
        let z_trans = z_rot * s + (2.0 * mz * p.w);

        // Projective coordinate scaling under tau
        let w_trans = p.w * s - (2.0 * tau * (mx * x_rot + my * y_rot + mz * z_rot));

        let inv_w = if w_trans.abs() > 1e-7 { 1.0 / w_trans } else { 1.0 };

        CkPoint3D {
            x: x_trans * inv_w,
            y: y_trans * inv_w,
            z: z_trans * inv_w,
            w: 1.0,
        }
    }
}
