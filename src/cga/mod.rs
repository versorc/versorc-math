// ============================================================================
// VERSORC-MATH: 3D CONFORMAL GEOMETRIC ALGEBRA (CGA) Cl(4, 1)
// License: Apache-2.0 OR MIT
// ============================================================================

use std::ops::{Add, Div, Mul, Neg, Sub};

/// 5D Conformal Null-Basis Vector in Cl(4, 1).
/// Basis: e1, e2, e3 (Euclidean, e_i^2 = +1), no (origin), ni (infinity).
/// Metric properties:
/// - no^2 = 0, ni^2 = 0
/// - no . ni = -1
/// Alignment: 32 bytes (8 floats: x, y, z, no, ni, plus 3 padding floats).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct CgaVector5D {
    pub x: f32,     // e1: Euclidean x
    pub y: f32,     // e2: Euclidean y
    pub z: f32,     // e3: Euclidean z
    pub no: f32,    // n_o: Origin null basis vector
    pub ni: f32,    // n_\infty: Infinity null basis vector
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

impl Default for CgaVector5D {
    #[inline(always)]
    fn default() -> Self {
        Self::origin()
    }
}

impl CgaVector5D {
    /// Zero vector.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            no: 0.0,
            ni: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }

    /// Origin point n_o (0, 0, 0, 1, 0).
    #[inline(always)]
    pub const fn origin() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            no: 1.0,
            ni: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }

    /// Infinity point n_\infty (0, 0, 0, 0, 1).
    #[inline(always)]
    pub const fn infinity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            no: 0.0,
            ni: 1.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }

    /// Embeds a 3D Euclidean point x = (x, y, z) into CGA:
    /// P = x + 0.5 * ||x||^2 * n_\infty + n_o.
    #[inline]
    pub fn point(x: f32, y: f32, z: f32) -> Self {
        let sq = x * x + y * y + z * z;
        Self {
            x,
            y,
            z,
            no: 1.0,
            ni: 0.5 * sq,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }

    /// Embeds a 3D sphere with center c = (cx, cy, cz) and radius r:
    /// S = c + 0.5 * (||c||^2 - r^2) * n_\infty + n_o.
    #[inline]
    pub fn sphere(cx: f32, cy: f32, cz: f32, radius: f32) -> Self {
        let sq = cx * cx + cy * cy + cz * cz;
        Self {
            x: cx,
            y: cy,
            z: cz,
            no: 1.0,
            ni: 0.5 * (sq - radius * radius),
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }

    /// Embeds a 3D plane with unit normal n = (nx, ny, nz) and signed distance d:
    /// \pi = n + d * n_\infty (with no = 0).
    #[inline]
    pub fn plane(nx: f32, ny: f32, nz: f32, d: f32) -> Self {
        Self {
            x: nx,
            y: ny,
            z: nz,
            no: 0.0,
            ni: d,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }

    /// Computes the Lorentzian inner product:
    /// U . V = (ux*vx + uy*vy + uz*vz) - (u_no * v_ni + u_ni * v_no).
    #[inline(always)]
    pub fn inner_product(&self, other: &Self) -> f32 {
        (self.x * other.x + self.y * other.y + self.z * other.z)
            - (self.no * other.ni + self.ni * other.no)
    }

    /// Extracts Euclidean coordinates [x/no, y/no, z/no].
    #[inline]
    pub fn to_euclidean(&self) -> [f32; 3] {
        if self.no.abs() > 1e-12 {
            let inv = 1.0 / self.no;
            [self.x * inv, self.y * inv, self.z * inv]
        } else {
            [self.x, self.y, self.z]
        }
    }

    /// Computes squared Euclidean distance between two conformal points:
    /// ||p1 - p2||^2 = -2 * (P1 . P2) / (P1_no * P2_no).
    #[inline]
    pub fn squared_distance(&self, other: &Self) -> f32 {
        let dot = self.inner_product(other);
        let weight = self.no * other.no;
        if weight.abs() > 1e-12 {
            (-2.0 * dot / weight).max(0.0)
        } else {
            0.0
        }
    }

    /// Euclidean distance between two conformal points.
    #[inline(always)]
    pub fn distance(&self, other: &Self) -> f32 {
        self.squared_distance(other).sqrt()
    }

    /// If this vector represents a sphere, extracts its radius: r = sqrt(S . S).
    #[inline]
    pub fn sphere_radius(&self) -> Option<f32> {
        if self.no.abs() > 1e-12 {
            let inv_no = 1.0 / self.no;
            let norm_s = *self * inv_no;
            let dot = norm_s.inner_product(&norm_s);
            if dot >= 0.0 {
                Some(dot.sqrt())
            } else {
                None // Imaginary sphere (point pair)
            }
        } else {
            None // Plane (infinite radius)
        }
    }

    /// Reflects this conformal point across a sphere via 1-versor sandwich: P' = S * P * S / ||S||^2.
    /// This is the classical spherical inversion!
    #[inline]
    pub fn invert_in_sphere(&self, sphere: &Self) -> Self {
        let center = sphere.to_euclidean();
        let r_sq = sphere.inner_product(sphere);
        let pt = self.to_euclidean();

        let dx = pt[0] - center[0];
        let dy = pt[1] - center[1];
        let dz = pt[2] - center[2];

        let dist_sq = dx * dx + dy * dy + dz * dz;
        if dist_sq < 1e-12 {
            // Origin of inversion maps to infinity
            Self::infinity()
        } else {
            let scale = r_sq / dist_sq;
            let new_x = center[0] + dx * scale;
            let new_y = center[1] + dy * scale;
            let new_z = center[2] + dz * scale;
            Self::point(new_x, new_y, new_z)
        }
    }

    /// Applies a conformal translation by vector d = (dx, dy, dz).
    #[inline]
    pub fn translate(&self, dx: f32, dy: f32, dz: f32) -> Self {
        let pt = self.to_euclidean();
        Self::point(pt[0] + dx, pt[1] + dy, pt[2] + dz)
    }

    /// Applies a conformal dilation / uniform scale with factor lambda > 0.
    #[inline]
    pub fn dilate(&self, lambda: f32) -> Self {
        let pt = self.to_euclidean();
        Self::point(pt[0] * lambda, pt[1] * lambda, pt[2] * lambda)
    }
}

impl Add for CgaVector5D {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            no: self.no + rhs.no,
            ni: self.ni + rhs.ni,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

impl Sub for CgaVector5D {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            no: self.no - rhs.no,
            ni: self.ni - rhs.ni,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

impl Neg for CgaVector5D {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            no: -self.no,
            ni: -self.ni,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

impl Mul<f32> for CgaVector5D {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            no: self.no * rhs,
            ni: self.ni * rhs,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

impl Div<f32> for CgaVector5D {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        let inv = 1.0 / rhs;
        Self {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
            no: self.no * inv,
            ni: self.ni * inv,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

// Convenient aliases for industry-standard Conformal entities:
pub type CgaPoint = CgaVector5D;
pub type CgaSphere = CgaVector5D;
pub type CgaPlane = CgaVector5D;

/// Conformal Point Pair (2-Blade P = p1 ^ p2) in Cl(4, 1):
/// Represents two discrete points in 3D Euclidean space (Dorst et al., 2007, Chapter 14).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CgaPointPair {
    pub p1: CgaPoint,
    pub p2: CgaPoint,
}

impl CgaPointPair {
    #[inline]
    pub fn new(p1: CgaPoint, p2: CgaPoint) -> Self {
        Self { p1, p2 }
    }

    /// Euclidean distance between the pair of points: d = ||p1 - p2||.
    #[inline]
    pub fn distance(&self) -> f32 {
        self.p1.distance(&self.p2)
    }

    /// Midpoint of the point pair in 3D Euclidean space.
    #[inline]
    pub fn midpoint(&self) -> [f32; 3] {
        let e1 = self.p1.to_euclidean();
        let e2 = self.p2.to_euclidean();
        [
            (e1[0] + e2[0]) * 0.5,
            (e1[1] + e2[1]) * 0.5,
            (e1[2] + e2[2]) * 0.5,
        ]
    }
}

/// Conformal Circle (3-Blade C = p1 ^ p2 ^ p3 or dual C* = Sphere ^ Plane) in Cl(4, 1):
/// Represents an oriented circle in 3D space with center, radius, and normal plane (Dorst et al., 2007, Chapter 14).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CgaCircle {
    pub center: [f32; 3],
    pub normal: [f32; 3],
    pub radius: f32,
}

impl CgaCircle {
    #[inline]
    pub fn new(center: [f32; 3], normal: [f32; 3], radius: f32) -> Self {
        let n_norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let inv_n = if n_norm > 1e-12 { 1.0 / n_norm } else { 1.0 };
        Self {
            center,
            normal: [normal[0] * inv_n, normal[1] * inv_n, normal[2] * inv_n],
            radius: radius.max(0.0),
        }
    }

    /// Constructs a circle from the intersection of a sphere and a plane.
    #[inline]
    pub fn from_sphere_plane_intersection(sphere: &CgaSphere, plane: &CgaPlane) -> Option<Self> {
        let sc = sphere.to_euclidean();
        let r = sphere.sphere_radius()?;
        let n = [plane.x, plane.y, plane.z];
        let n_len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if n_len < 1e-12 {
            return None;
        }
        let inv_n = 1.0 / n_len;
        let n_unit = [n[0] * inv_n, n[1] * inv_n, n[2] * inv_n];

        // Signed distance from sphere center to plane
        let dist = n_unit[0] * sc[0] + n_unit[1] * sc[1] + n_unit[2] * sc[2] + plane.ni * inv_n;
        if dist.abs() > r {
            return None; // No real intersection
        }

        let circle_r = (r * r - dist * dist).max(0.0).sqrt();
        let circle_center = [
            sc[0] - dist * n_unit[0],
            sc[1] - dist * n_unit[1],
            sc[2] - dist * n_unit[2],
        ];

        Some(Self::new(circle_center, n_unit, circle_r))
    }

    /// Area of the circle: A = \pi * r^2.
    #[inline]
    pub fn area(&self) -> f32 {
        std::f32::consts::PI * self.radius * self.radius
    }

    /// Circumference: L = 2 * \pi * r.
    #[inline]
    pub fn circumference(&self) -> f32 {
        2.0 * std::f32::consts::PI * self.radius
    }
}

/// Conformal Versor (Motor / Dilator / Rotor) in Cl(4, 1):
/// Unifies rotation, translation, and uniform scaling (dilation) into a single multivector operator.
/// (Dorst, Fontijne, Mann 2007, Chapter 13 & 15).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CgaConformalMotor {
    pub translation: [f32; 3],
    pub rotation_axis: [f32; 3],
    pub rotation_angle_rad: f32,
    pub dilation_factor: f32,
}

impl Default for CgaConformalMotor {
    #[inline(always)]
    fn default() -> Self {
        Self::identity()
    }
}

impl CgaConformalMotor {
    #[inline(always)]
    pub const fn identity() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation_axis: [0.0, 0.0, 1.0],
            rotation_angle_rad: 0.0,
            dilation_factor: 1.0,
        }
    }

    #[inline]
    pub fn from_translation(dx: f32, dy: f32, dz: f32) -> Self {
        Self {
            translation: [dx, dy, dz],
            rotation_axis: [0.0, 0.0, 1.0],
            rotation_angle_rad: 0.0,
            dilation_factor: 1.0,
        }
    }

    #[inline]
    pub fn from_rotation(axis: [f32; 3], angle_rad: f32) -> Self {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let inv = if norm > 1e-12 { 1.0 / norm } else { 1.0 };
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation_axis: [axis[0] * inv, axis[1] * inv, axis[2] * inv],
            rotation_angle_rad: angle_rad,
            dilation_factor: 1.0,
        }
    }

    #[inline]
    pub fn from_dilation(lambda: f32) -> Self {
        assert!(lambda > 0.0, "Dilation factor must be strictly positive");
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation_axis: [0.0, 0.0, 1.0],
            rotation_angle_rad: 0.0,
            dilation_factor: lambda,
        }
    }

    /// Transforms a Conformal Point: p' = Motor(p) = Translation(Rotation(Dilation(p))).
    #[inline]
    pub fn transform_point(&self, pt: &CgaPoint) -> CgaPoint {
        let [x0, y0, z0] = pt.to_euclidean();

        // 1. Dilation
        let x1 = x0 * self.dilation_factor;
        let y1 = y0 * self.dilation_factor;
        let z1 = z0 * self.dilation_factor;

        // 2. Rotation via Rodrigues rotor formula
        let theta = self.rotation_angle_rad;
        let [nx, ny, nz] = self.rotation_axis;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let dot_nr = nx * x1 + ny * y1 + nz * z1;

        let rx = x1 * cos_t + (ny * z1 - nz * y1) * sin_t + nx * dot_nr * (1.0 - cos_t);
        let ry = y1 * cos_t + (nz * x1 - nx * z1) * sin_t + ny * dot_nr * (1.0 - cos_t);
        let rz = z1 * cos_t + (nx * y1 - ny * x1) * sin_t + nz * dot_nr * (1.0 - cos_t);

        // 3. Translation
        let tx = rx + self.translation[0];
        let ty = ry + self.translation[1];
        let tz = rz + self.translation[2];

        CgaPoint::point(tx, ty, tz)
    }
}
