// ============================================================================
// VERSORC-MATH: DATA-ORIENTED DESIGN (DOD) & 64-BYTE CACHE-LINE BATCH PROCESSING
// License: Apache-2.0
//
// Designed around Universal 64-Bit / 64-Byte Cache-Line Alignment (#[repr(C, align(64))]).
// Eliminates false sharing, cache-line straddling, and pointer chasing across
// L1/L2/L3 cache hierarchies on modern 64-bit architectures (x86_64, aarch64, riscv64).
// ============================================================================

use crate::poincare::cayley_klein::CayleyKleinSpace;
use crate::lattice::eisenstein::{
    RATIO_2_SQRT3, SQRT3_DIV2_INV,
};
use crate::pga::{PgaMotor, PgaPoint};

/// 64-byte cache-line aligned FP32 accumulator (512 bits = 8 x 64-bit words).
/// Strictly matches the physical 64-byte line width of modern L1d/L2 caches.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Universal64Accumulator {
    pub lanes: [f32; 16],
}

impl Default for Universal64Accumulator {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl Universal64Accumulator {
    #[inline(always)]
    pub const fn zero() -> Self {
        Self { lanes: [0.0f32; 16] }
    }

    /// Creates an accumulator from 16 floats.
    #[inline(always)]
    pub const fn from_lanes(lanes: [f32; 16]) -> Self {
        Self { lanes }
    }

    /// Tree-reduces all 16 lanes in balanced pairs to minimize IEEE-754 rounding drift.
    #[inline(always)]
    pub fn horizontal_sum(&self) -> f32 {
        let s0 = (self.lanes[0] + self.lanes[1]) + (self.lanes[2] + self.lanes[3]);
        let s1 = (self.lanes[4] + self.lanes[5]) + (self.lanes[6] + self.lanes[7]);
        let s2 = (self.lanes[8] + self.lanes[9]) + (self.lanes[10] + self.lanes[11]);
        let s3 = (self.lanes[12] + self.lanes[13]) + (self.lanes[14] + self.lanes[15]);
        (s0 + s1) + (s2 + s3)
    }
}

/// Universal 64-byte raw chunk for SWAR and bitwise memory streaming.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Universal64Chunk {
    pub words: [u64; 8],
}

impl Default for Universal64Chunk {
    #[inline(always)]
    fn default() -> Self {
        Self { words: [0u64; 8] }
    }
}

/// Structure of Arrays (SoA) Batch of 16 3D/4D Points.
/// Each coordinate array is exactly 64 bytes (1 L1d cache line).
/// Total size: 256 bytes (fits cleanly in 4 contiguous L1d cache lines).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoABatch16 {
    pub x: [f32; 16],
    pub y: [f32; 16],
    pub z: [f32; 16],
    pub w: [f32; 16],
}

impl Default for SoABatch16 {
    #[inline(always)]
    fn default() -> Self {
        Self {
            x: [0.0; 16],
            y: [0.0; 16],
            z: [0.0; 16],
            w: [0.0; 16],
        }
    }
}

impl SoABatch16 {
    pub const CAPACITY: usize = 16;

    /// Creates an all-zero SoA batch.
    #[inline(always)]
    pub fn zero() -> Self {
        Self::default()
    }

    /// Creates an SoA batch from an array of 16 3D points [x, y, z].
    pub fn from_aos_3d(points: &[[f32; 3]; 16]) -> Self {
        let mut batch = Self::default();
        for i in 0..16 {
            batch.x[i] = points[i][0];
            batch.y[i] = points[i][1];
            batch.z[i] = points[i][2];
            batch.w[i] = 1.0;
        }
        batch
    }

    /// Evaluates the Cayley-Klein geodesic distance from this batch of 16 points
    /// to another batch of 16 points in parallel, storing the 16 distances
    /// in a 64-byte cache-line aligned accumulator with ZERO dynamic memory allocations.
    pub fn batch_geodesic_distance(
        &self,
        other: &Self,
        space: &CayleyKleinSpace,
    ) -> Universal64Accumulator {
        let mut out = Universal64Accumulator::zero();
        let tau = space.tau;

        if tau.abs() < 1e-7 {
            // Euclidean batch
            for i in 0..16 {
                let dx = self.x[i] - other.x[i];
                let dy = self.y[i] - other.y[i];
                let dz = self.z[i] - other.z[i];
                out.lanes[i] = (dx * dx + dy * dy + dz * dz).sqrt();
            }
            return out;
        }

        if tau < 0.0 {
            // Hyperbolic batch
            let abs_tau = -tau;
            let inv_sqrt_tau = 1.0f32 / abs_tau.sqrt();

            for i in 0..16 {
                let dx = self.x[i] - other.x[i];
                let dy = self.y[i] - other.y[i];
                let dz = self.z[i] - other.z[i];
                let diff_sq = dx * dx + dy * dy + dz * dz;

                let u_sq = self.x[i] * self.x[i] + self.y[i] * self.y[i] + self.z[i] * self.z[i];
                let v_sq = other.x[i] * other.x[i] + other.y[i] * other.y[i] + other.z[i] * other.z[i];

                let u_fact = (1.0f32 - abs_tau * u_sq).max(1e-7);
                let v_fact = (1.0f32 - abs_tau * v_sq).max(1e-7);
                let denom = u_fact * v_fact;

                let delta = (1.0f32 + (2.0f32 * abs_tau * diff_sq) / denom).max(1.0f32);
                let arcosh = (delta + (delta * delta - 1.0f32).sqrt()).ln();
                out.lanes[i] = inv_sqrt_tau * arcosh;
            }
        } else {
            // Elliptic / Spherical batch
            let inv_sqrt_tau = 1.0f32 / tau.sqrt();

            for i in 0..16 {
                let dx = self.x[i] - other.x[i];
                let dy = self.y[i] - other.y[i];
                let dz = self.z[i] - other.z[i];
                let diff_sq = dx * dx + dy * dy + dz * dz;

                let u_sq = self.x[i] * self.x[i] + self.y[i] * self.y[i] + self.z[i] * self.z[i];
                let v_sq = other.x[i] * other.x[i] + other.y[i] * other.y[i] + other.z[i] * other.z[i];

                let u_fact = 1.0f32 + tau * u_sq;
                let v_fact = 1.0f32 + tau * v_sq;
                let denom = (u_fact * v_fact).max(1e-7);

                let arg = (1.0f32 - (2.0f32 * tau * diff_sq) / denom).clamp(-1.0f32, 1.0f32);
                out.lanes[i] = inv_sqrt_tau * arg.acos();
            }
        }

        out
    }

    /// Evaluates the Stereographic Conformal Projection to Euclidean display coordinates in-place
    /// across all 16 points in parallel:
    /// x_{euc} = x / (1 + (tau/4) * ||x||^2)
    pub fn batch_project_to_euclidean(&mut self, space: &CayleyKleinSpace) {
        let tau = space.tau;
        if tau.abs() < 1e-7 {
            return;
        }

        let tau_div_4 = tau * 0.25f32;
        for i in 0..16 {
            let norm_sq = self.x[i] * self.x[i] + self.y[i] * self.y[i] + self.z[i] * self.z[i];
            let denom = 1.0f32 + tau_div_4 * norm_sq;
            let scale = 1.0f32 / denom.max(1e-7);
            self.x[i] *= scale;
            self.y[i] *= scale;
            self.z[i] *= scale;
        }
    }

    /// Converts 16 continuous 3D points into discrete Eisenstein A3 FCC lattice coordinates
    /// (u, v, w) in a single pass without allocations:
    pub fn batch_to_eisenstein_a3(&self) -> ([i32; 16], [i32; 16], [i32; 16]) {
        let mut u_out = [0i32; 16];
        let mut v_out = [0i32; 16];
        let mut w_out = [0i32; 16];

        for i in 0..16 {
            let w_f = self.z[i] * SQRT3_DIV2_INV;
            let w = w_f.round() as i32;

            let v_f = self.y[i] * RATIO_2_SQRT3 + (w as f32) / 3.0f32;
            let v = v_f.round() as i32;

            let u_f = self.x[i] + 0.5f32 * (v as f32) + 0.5f32 * (w as f32);
            let u = u_f.round() as i32;

            u_out[i] = u;
            v_out[i] = v;
            w_out[i] = w;
        }

        (u_out, v_out, w_out)
    }

    /// Evaluates 2D planar spinor (Givens / RoPE) rotations over the (X, Y) lanes of all 16 elements
    /// in-place in 64-byte cache lines.
    #[inline]
    pub fn batch_planar_spinor_rotate(&mut self, cos_th: f32, sin_th: f32) {
        for i in 0..16 {
            let u = self.x[i];
            let v = self.y[i];
            self.x[i] = u * cos_th - v * sin_th;
            self.y[i] = u * sin_th + v * cos_th;
        }
    }
}

/// Structure-of-Arrays (SoA) batch of 16 PGA Motors / Dual Quaternions.
/// Total size: 512 bytes = exactly 8 contiguous 64-byte L1d cache lines.
/// Each component array (s, b23, b31, b12, d01, d02, d03, e0123) is exactly 64 bytes.
/// Guarantees optimal cache line prefetching and zero-straddle SIMD vectorization.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct SoAPgaMotorBatch16 {
    pub s: [f32; 16],     // Cache line 0: 64 bytes
    pub b23: [f32; 16],   // Cache line 1: 64 bytes
    pub b31: [f32; 16],   // Cache line 2: 64 bytes
    pub b12: [f32; 16],   // Cache line 3: 64 bytes
    pub d01: [f32; 16],   // Cache line 4: 64 bytes
    pub d02: [f32; 16],   // Cache line 5: 64 bytes
    pub d03: [f32; 16],   // Cache line 6: 64 bytes
    pub e0123: [f32; 16], // Cache line 7: 64 bytes
}

impl Default for SoAPgaMotorBatch16 {
    #[inline(always)]
    fn default() -> Self {
        Self::identity()
    }
}

impl SoAPgaMotorBatch16 {
    /// Batch of 16 identity motors.
    #[inline]
    pub fn identity() -> Self {
        Self {
            s: [1.0f32; 16],
            b23: [0.0f32; 16],
            b31: [0.0f32; 16],
            b12: [0.0f32; 16],
            d01: [0.0f32; 16],
            d02: [0.0f32; 16],
            d03: [0.0f32; 16],
            e0123: [0.0f32; 16],
        }
    }

    /// Broadcasts a single PgaMotor across all 16 SIMD lanes.
    #[inline]
    pub fn broadcast(m: &PgaMotor) -> Self {
        Self {
            s: [m.s; 16],
            b23: [m.b23; 16],
            b31: [m.b31; 16],
            b12: [m.b12; 16],
            d01: [m.d01; 16],
            d02: [m.d02; 16],
            d03: [m.d03; 16],
            e0123: [m.e0123; 16],
        }
    }

    /// Loads 16 individual PgaMotors into the SoA cache-aligned layout.
    #[inline]
    pub fn from_aos(motors: &[PgaMotor; 16]) -> Self {
        let mut batch = Self::identity();
        for i in 0..16 {
            batch.s[i] = motors[i].s;
            batch.b23[i] = motors[i].b23;
            batch.b31[i] = motors[i].b31;
            batch.b12[i] = motors[i].b12;
            batch.d01[i] = motors[i].d01;
            batch.d02[i] = motors[i].d02;
            batch.d03[i] = motors[i].d03;
            batch.e0123[i] = motors[i].e0123;
        }
        batch
    }

    /// Extracts the i-th motor from the batch.
    #[inline]
    pub fn get_motor(&self, i: usize) -> PgaMotor {
        assert!(i < 16, "Index out of range for 16-lane motor batch");
        PgaMotor::new(
            self.s[i],
            self.b23[i],
            self.b31[i],
            self.b12[i],
            self.d01[i],
            self.d02[i],
            self.d03[i],
            self.e0123[i],
        )
    }

    /// Transforms an entire batch of 16 points in-place using the corresponding 16 motors.
    #[inline]
    pub fn transform_points(&self, points: &mut SoABatch16) {
        for i in 0..16 {
            let m = self.get_motor(i);
            let pt = PgaPoint::new(points.x[i], points.y[i], points.z[i]);
            let t_pt = m.transform_point(&pt);
            let [x, y, z] = t_pt.to_euclidean();
            points.x[i] = x;
            points.y[i] = y;
            points.z[i] = z;
        }
    }
}
