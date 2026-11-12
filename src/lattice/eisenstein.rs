// ============================================================================
// VERSORC-MATH: 3D EISENSTEIN A3 FCC CLOSE-PACKED LATTICE (Z[omega])
// License: Apache-2.0 OR MIT
//
// 3D Gotthold Eisenstein lattice tessellation (A3 root lattice / Face-Centered Cubic).
// Implements Kepler-optimal sphere packing in dimension 3 with Kissing Number = 12.
// ============================================================================

pub const EISENSTEIN_DIMENSION: usize = 3;
pub const EISENSTEIN_KISSING_NUMBER: usize = 12;

/// Exact irrational scale factors in rhombohedral A3 FCC geometry:
pub const SQRT3_DIV2: f32 = 0.8660254037844386; // sqrt(3) / 2
pub const SQRT3_DIV6: f32 = 0.2886751345948129; // sqrt(3) / 6
pub const SQRT2_DIV3: f32 = 0.816496580927726;  // sqrt(2/3)
pub const RATIO_2_SQRT3: f32 = 1.1547005383792515; // 2 / sqrt(3)
pub const SQRT3_DIV2_INV: f32 = 1.224744871391589; // sqrt(3/2)

/// The 3D Eisenstein A3 FCC Close-Packed Lattice Engine.
pub struct EisensteinA3Lattice;

impl EisensteinA3Lattice {
    /// Generates all 12 minimal root / kissing neighbors of the A3 lattice at unit distance:
    /// D_{A_3}^2(u, v, w) == 1.
    pub fn generate_12_neighbors() -> [[i32; 3]; EISENSTEIN_KISSING_NUMBER] {
        [
            // Permutations of (+-1, 0, 0) in basis
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            // Pairwise shared edges with D_{A_3}^2 == 1
            [1, 1, 0],
            [-1, -1, 0],
            [1, 0, 1],
            [-1, 0, -1],
            [0, 1, 1],
            [0, -1, -1],
        ]
    }

    /// Computes the exact discrete lattice squared distance D_{A_3}^2 in the rhombohedral basis:
    /// D_{A_3}^2 = \Delta u^2 + \Delta v^2 + \Delta w^2 - \Delta u \Delta v - \Delta u \Delta w - \Delta v \Delta w
    #[inline(always)]
    pub fn distance_sq(du: i32, dv: i32, dw: i32) -> i64 {
        let u = du as i64;
        let v = dv as i64;
        let w = dw as i64;
        (u * u) + (v * v) + (w * w) - (u * v) - (u * w) - (v * w)
    }

    /// Converts discrete Eisenstein (u, v, w) coordinates into continuous Euclidean 3D coordinates (x, y, z):
    /// - x = u - 0.5 * v - 0.5 * w
    /// - y = v * (sqrt(3)/2) - w * (sqrt(3)/6)
    /// - z = w * sqrt(2/3)
    #[inline]
    pub fn to_euclidean(u: i32, v: i32, w: i32) -> [f32; 3] {
        let uf = u as f32;
        let vf = v as f32;
        let wf = w as f32;

        let x = uf - 0.5 * vf - 0.5 * wf;
        let y = vf * SQRT3_DIV2 - wf * SQRT3_DIV6;
        let z = wf * SQRT2_DIV3;

        [x, y, z]
    }

    /// Converts continuous Euclidean 3D coordinates (x, y, z) to the nearest discrete
    /// Eisenstein (u, v, w) lattice point:
    /// - w = round(z * sqrt(3/2))
    /// - v = round(y * (2/sqrt(3)) + w / 3)
    /// - u = round(x + 0.5 * v + 0.5 * w)
    #[inline]
    pub fn from_euclidean(x: f32, y: f32, z: f32) -> [i32; 3] {
        let w_f = z * SQRT3_DIV2_INV;
        let w = w_f.round() as i32;

        let v_f = y * RATIO_2_SQRT3 + (w as f32) / 3.0f32;
        let v = v_f.round() as i32;

        let u_f = x + 0.5f32 * (v as f32) + 0.5f32 * (w as f32);
        let u = u_f.round() as i32;

        [u, v, w]
    }

    /// Maps discrete Eisenstein (u, v, w) coordinates directly to the conformal Poincaré Ball B^3:
    /// Projects continuous radius r via hyperbolic tangent: r_poincare = tanh(scale * r_euclid).
    pub fn to_poincare(u: i32, v: i32, w: i32, scale: f32) -> [f32; 3] {
        let [x, y, z] = Self::to_euclidean(u, v, w);
        let r3d = (x * x + y * y + z * z).sqrt();

        if r3d > 1e-7 {
            let r_scaled = scale * r3d;
            let r_poincare = r_scaled.tanh();
            let norm = r_poincare / r3d;
            [x * norm, y * norm, z * norm]
        } else {
            [0.0, 0.0, 0.0]
        }
    }
}
