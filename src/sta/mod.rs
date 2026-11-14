// ============================================================================
// VERSORC-MATH: SPACETIME ALGEBRA (STA) Cl(1, 3) (MINKOWSKI SPACETIME)
// License: Apache-2.0
// ============================================================================

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Minkowski 4-vector in Spacetime Algebra Cl(1, 3).
/// Basis: \gamma_0^2 = +1 (timelike), \gamma_1^2 = \gamma_2^2 = \gamma_3^2 = -1 (spacelike).
/// Metric interval: s^2 = (ct)^2 - (x^2 + y^2 + z^2).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct StaVector {
    pub ct: f32, // \gamma_0: Time component (c * t)
    pub x: f32,  // \gamma_1: Spatial x
    pub y: f32,  // \gamma_2: Spatial y
    pub z: f32,  // \gamma_3: Spatial z
}

impl Default for StaVector {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl StaVector {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(ct: f32, x: f32, y: f32, z: f32) -> Self {
        Self { ct, x, y, z }
    }

    #[inline(always)]
    pub const fn zero() -> Self {
        Self {
            ct: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Spacetime interval invariant s^2 = (ct)^2 - (x^2 + y^2 + z^2).
    #[inline(always)]
    pub fn interval_sq(&self) -> f32 {
        self.ct * self.ct - (self.x * self.x + self.y * self.y + self.z * self.z)
    }

    /// Minkowski inner product: u . v = u_0 * v_0 - (u_x*v_x + u_y*v_y + u_z*v_z).
    #[inline(always)]
    pub fn dot(&self, other: &Self) -> f32 {
        self.ct * other.ct - (self.x * other.x + self.y * other.y + self.z * other.z)
    }

    /// Checks if interval is timelike (s^2 > 0).
    #[inline(always)]
    pub fn is_timelike(&self) -> bool {
        self.interval_sq() > 0.0
    }

    /// Checks if interval is spacelike (s^2 < 0).
    #[inline(always)]
    pub fn is_spacelike(&self) -> bool {
        self.interval_sq() < 0.0
    }

    /// Checks if interval is lightlike / null (|s^2| < 1e-12).
    #[inline(always)]
    pub fn is_lightlike(&self) -> bool {
        self.interval_sq().abs() < 1e-12
    }

    /// Proper time interval \Delta \tau = sqrt(s^2) / c (with c = 1).
    #[inline]
    pub fn proper_time(&self) -> Option<f32> {
        let s2 = self.interval_sq();
        if s2 >= 0.0 {
            Some(s2.sqrt())
        } else {
            None
        }
    }
}

impl Add for StaVector {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            ct: self.ct + rhs.ct,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for StaVector {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            ct: self.ct - rhs.ct,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for StaVector {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            ct: -self.ct,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul<f32> for StaVector {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self {
            ct: self.ct * rhs,
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Div<f32> for StaVector {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        let inv = 1.0 / rhs;
        Self {
            ct: self.ct * inv,
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }
}

/// Electromagnetic Faraday Bivector in Spacetime Algebra: F = E + I*c*B.
/// Unifies electric and magnetic field vectors into a single grade-2 bivector.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct StaBivector {
    pub ex: f32, // \gamma_1 \gamma_0: Electric field x
    pub ey: f32, // \gamma_2 \gamma_0: Electric field y
    pub ez: f32, // \gamma_3 \gamma_0: Electric field z
    pub bx: f32, // \gamma_3 \gamma_2: Magnetic field x
    pub by: f32, // \gamma_1 \gamma_3: Magnetic field y
    pub bz: f32, // \gamma_2 \gamma_1: Magnetic field z
    pub _pad0: f32,
    pub _pad1: f32,
}

impl Default for StaBivector {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl StaBivector {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(ex: f32, ey: f32, ez: f32, bx: f32, by: f32, bz: f32) -> Self {
        Self {
            ex,
            ey,
            ez,
            bx,
            by,
            bz,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }

    #[inline(always)]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// First relativistic invariant: E^2 - c^2*B^2 (with c = 1).
    #[inline(always)]
    pub fn invariant_1(&self) -> f32 {
        (self.ex * self.ex + self.ey * self.ey + self.ez * self.ez)
            - (self.bx * self.bx + self.by * self.by + self.bz * self.bz)
    }

    /// Second relativistic invariant: 2 * E . B.
    #[inline(always)]
    pub fn invariant_2(&self) -> f32 {
        2.0 * (self.ex * self.bx + self.ey * self.by + self.ez * self.bz)
    }

    /// Poynting vector direction (energy flux): S = E x B.
    #[inline]
    pub fn poynting_vector(&self) -> [f32; 3] {
        [
            self.ey * self.bz - self.ez * self.by,
            self.ez * self.bx - self.ex * self.bz,
            self.ex * self.by - self.ey * self.bx,
        ]
    }
}

/// Lorentz Boost Versor (hyperbolic rotor) along velocity vector v = (vx, vy, vz).
/// B = cosh(phi/2) - sinh(phi/2) * (v_hat . \sigma).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct StaLorentzBoost {
    pub cosh_half: f32,
    pub sinh_half_x: f32,
    pub sinh_half_y: f32,
    pub sinh_half_z: f32,
}

impl StaLorentzBoost {
    /// Constructs a Lorentz boost from velocity vector v = (vx, vy, vz) with c = 1.
    /// beta = ||v|| < 1.0.
    #[inline]
    pub fn from_velocity(vx: f32, vy: f32, vz: f32) -> Self {
        let v2 = vx * vx + vy * vy + vz * vz;
        let beta = v2.sqrt();
        assert!(beta < 1.0, "Velocity must be strictly subluminal (|v| < c)");

        if beta < 1e-12 {
            return Self {
                cosh_half: 1.0,
                sinh_half_x: 0.0,
                sinh_half_y: 0.0,
                sinh_half_z: 0.0,
            };
        }

        // Rapidity phi = atanh(beta)
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
        // Half-rapidity identities:
        // cosh(phi/2) = sqrt((gamma + 1) / 2)
        // sinh(phi/2) = sqrt((gamma - 1) / 2)
        let cosh_half = ((gamma + 1.0) * 0.5).sqrt();
        let sinh_half = ((gamma - 1.0) * 0.5).sqrt();

        let inv_beta = 1.0 / beta;
        Self {
            cosh_half,
            sinh_half_x: sinh_half * (vx * inv_beta),
            sinh_half_y: sinh_half * (vy * inv_beta),
            sinh_half_z: sinh_half * (vz * inv_beta),
        }
    }

    /// Transforms a Minkowski 4-vector via 2-versor sandwich x' = B * x * B~.
    #[inline]
    pub fn transform_vector(&self, v: &StaVector) -> StaVector {
        let ch = self.cosh_half;
        let sx = self.sinh_half_x;
        let sy = self.sinh_half_y;
        let sz = self.sinh_half_z;

        let s2 = sx * sx + sy * sy + sz * sz;
        let gamma = ch * ch + s2;
        let two_ch = 2.0 * ch;

        // Boost transformation formulas:
        let s_dot_r = sx * v.x + sy * v.y + sz * v.z;
        let new_ct = gamma * v.ct - two_ch * s_dot_r;

        let factor = if s2 > 1e-12 {
            (gamma - 1.0) * s_dot_r / s2
        } else {
            0.0
        };

        let new_x = v.x - two_ch * v.ct * sx + factor * sx;
        let new_y = v.y - two_ch * v.ct * sy + factor * sy;
        let new_z = v.z - two_ch * v.ct * sz + factor * sz;

        StaVector::new(new_ct, new_x, new_y, new_z)
    }
}

/// The 6 Spinor Classes according to Pertti Lounesto's Classification Theorem:
/// Lounesto (2001), "Clifford Algebras and Spinors", Chapter 12 (Flags, Poles and Dipoles).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LounestoClass {
    /// Class 1: Regular Dirac spinor with both non-zero scalar invariants (sigma != 0, omega != 0).
    Class1DiracGeneral,
    /// Class 2: Regular Dirac electron spinor with omega == 0 (Yvon-Takabayasi angle beta in {0, pi}).
    Class2DiracElectron,
    /// Class 3: Dirac spinor with sigma == 0 and omega != 0.
    Class3DiracPseudoscalar,
    /// Class 4: Singular Flag-Dipole spinor (sigma == omega == 0, S != 0, K != 0).
    Class4FlagDipole,
    /// Class 5: Singular Flag-Pole / Majorana spinor (sigma == omega == K == 0, S != 0).
    Class5MajoranaFlagPole,
    /// Class 6: Singular Dipole / Weyl neutrino spinor (sigma == omega == S == 0, K != 0).
    Class6WeylDipole,
}

/// Bilinear Covariants of a Dirac Spinor in Cl(1, 3):
/// sigma (scalar R1), omega (pseudoscalar R2), J (current), S (spin density bivector), K (chiral current).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpinorBilinearCovariants {
    pub sigma: f32,     // Scalar invariant <psi psi~>_0
    pub omega: f32,     // Pseudoscalar invariant <psi psi~>_4
    pub j: StaVector,   // Dirac probability current 4-vector
    pub s: StaBivector, // Spin density bivector
    pub k: StaVector,   // Chiral / pseudovector current
}

/// Dirac Spinor in Spacetime Algebra Cl(1, 3) as an Even Multivector Cl+(1, 3):
/// psi = scalar + (ex \gamma_10 + ey \gamma_20 + ez \gamma_30 + bx \gamma_32 + by \gamma_13 + bz \gamma_21) + pseudoscalar \gamma_0123.
/// Alignment: 32 bytes (cache line half, 8 floats).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct StaSpinor {
    pub scalar: f32,       // Grade 0: alpha
    pub ex: f32,           // Grade 2: boost x (\gamma_1 \gamma_0)
    pub ey: f32,           // Grade 2: boost y (\gamma_2 \gamma_0)
    pub ez: f32,           // Grade 2: boost z (\gamma_3 \gamma_0)
    pub bx: f32,           // Grade 2: rotation x (\gamma_3 \gamma_2)
    pub by: f32,           // Grade 2: rotation y (\gamma_1 \gamma_3)
    pub bz: f32,           // Grade 2: rotation z (\gamma_2 \gamma_1)
    pub pseudoscalar: f32, // Grade 4: beta (\gamma_0123)
}

impl Default for StaSpinor {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl StaSpinor {
    #[inline(always)]
    pub const fn zero() -> Self {
        Self {
            scalar: 0.0,
            ex: 0.0,
            ey: 0.0,
            ez: 0.0,
            bx: 0.0,
            by: 0.0,
            bz: 0.0,
            pseudoscalar: 0.0,
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        scalar: f32,
        ex: f32,
        ey: f32,
        ez: f32,
        bx: f32,
        by: f32,
        bz: f32,
        pseudoscalar: f32,
    ) -> Self {
        Self {
            scalar,
            ex,
            ey,
            ez,
            bx,
            by,
            bz,
            pseudoscalar,
        }
    }

    /// Reversion of spinor: psi~ = scalar - B + pseudoscalar * I.
    #[inline(always)]
    pub fn reverse(&self) -> Self {
        Self {
            scalar: self.scalar,
            ex: -self.ex,
            ey: -self.ey,
            ez: -self.ez,
            bx: -self.bx,
            by: -self.by,
            bz: -self.bz,
            pseudoscalar: self.pseudoscalar,
        }
    }

    /// Computes Lounesto's 5 Bilinear Covariants (sigma, omega, J, S, K) (Lounesto 2001, Eq. 12.1).
    #[inline]
    pub fn bilinear_covariants(&self) -> SpinorBilinearCovariants {
        let s = self.scalar;
        let p = self.pseudoscalar;
        let ex = self.ex;
        let ey = self.ey;
        let ez = self.ez;
        let bx = self.bx;
        let by = self.by;
        let bz = self.bz;

        // Scalar invariant: sigma = <psi psi~>_0 = s^2 - p^2 - (ex^2 + ey^2 + ez^2) + (bx^2 + by^2 + bz^2)
        let sigma = (s * s - p * p) - (ex * ex + ey * ey + ez * ez) + (bx * bx + by * by + bz * bz);

        // Pseudoscalar invariant: omega = <psi psi~>_4 = 2*s*p - 2*(ex*bx + ey*by + ez*bz)
        let omega = 2.0 * (s * p - (ex * bx + ey * by + ez * bz));

        // Dirac probability current J^mu = <psi \gamma^mu \psi~>_1
        let j0 = s * s + p * p + (ex * ex + ey * ey + ez * ez) + (bx * bx + by * by + bz * bz);
        let jx = 2.0 * (s * ex + p * bx + ey * bz - ez * by);
        let jy = 2.0 * (s * ey + p * by + ez * bx - ex * bz);
        let jz = 2.0 * (s * ez + p * bz + ex * by - ey * bx);
        let j = StaVector::new(j0, jx, jy, jz);

        // Spin density bivector S = psi (\gamma_2 \gamma_1) \psi~
        let sex = 2.0 * (-s * ey + p * by + bz * ex + ez * bx);
        let sey = 2.0 * (s * ex - p * bx + bz * ey + ez * by);
        let sez = -2.0 * (s * p + ex * bx + ey * by - ez * bz);
        let sbx = 2.0 * (-s * by - p * ey + bz * bx - ez * ex);
        let sby = 2.0 * (s * bx + p * ex + bz * by - ez * ey);
        let sbz = (s * s - p * p) + (ex * ex + ey * ey - ez * ez) - (bx * bx + by * by - bz * bz);
        let s_biv = StaBivector::new(sex, sey, sez, sbx, sby, sbz);

        // Chiral spin 4-vector K = psi \gamma_3 \psi~
        let k0 = 2.0 * (s * ez + p * bz - ex * by + ey * bx);
        let kx = 2.0 * (-s * by + p * ey + ex * ez + bz * bx);
        let ky = 2.0 * (s * bx - p * ex + ey * ez + bz * by);
        let kz = (s * s + p * p) - (ex * ex + ey * ey - ez * ez) - (bx * bx + by * by - bz * bz);
        let k = StaVector::new(k0, kx, ky, kz);

        SpinorBilinearCovariants {
            sigma,
            omega,
            j,
            s: s_biv,
            k,
        }
    }

    /// Classifies the spinor into one of the 6 Lounesto classes (Lounesto 2001, Ch. 12).
    #[inline]
    pub fn lounesto_class(&self, tol: f32) -> LounestoClass {
        let cov = self.bilinear_covariants();
        let sig_nonzero = cov.sigma.abs() > tol;
        let ome_nonzero = cov.omega.abs() > tol;

        if sig_nonzero && ome_nonzero {
            LounestoClass::Class1DiracGeneral
        } else if sig_nonzero && !ome_nonzero {
            LounestoClass::Class2DiracElectron
        } else if !sig_nonzero && ome_nonzero {
            LounestoClass::Class3DiracPseudoscalar
        } else {
            // Singular spinors (sigma == 0 and omega == 0):
            // Both K^2 and S^2 vanish identically by Fierz identities, so we must
            // test Euclidean coefficient norms rather than Minkowski invariants.
            let s_mag_sq = cov.s.ex * cov.s.ex
                + cov.s.ey * cov.s.ey
                + cov.s.ez * cov.s.ez
                + cov.s.bx * cov.s.bx
                + cov.s.by * cov.s.by
                + cov.s.bz * cov.s.bz;
            let k_mag_sq =
                cov.k.ct * cov.k.ct + cov.k.x * cov.k.x + cov.k.y * cov.k.y + cov.k.z * cov.k.z;
            let s_nonzero = s_mag_sq > tol * tol;
            let k_nonzero = k_mag_sq > tol * tol;

            if s_nonzero && k_nonzero {
                LounestoClass::Class4FlagDipole
            } else if s_nonzero && !k_nonzero {
                LounestoClass::Class5MajoranaFlagPole
            } else {
                LounestoClass::Class6WeylDipole
            }
        }
    }
}
