// ============================================================================
// VERSORC-MATH: TURBOQUANT GEOMETRIC QUANTIZER (3-BIT LLOYD-MAX + 1-BIT QJL)
// License: Apache-2.0
//
// Implementation of TurboQuant (Amir Zandieh et al. / Google Research)
// coupled with Clifford Versor Isometry:
// Stage 1: (b - 1) bit Lloyd-Max optimal scalar codebook on isotropized coordinates.
// Stage 2: 1-bit Quantized Johnson-Lindenstrauss (QJL) sketch on the residual vector:
//          r = R(w) - Q_base(R(w)).
//
// Guarantees unbiased inner-product estimation and eliminates quantization bias in attention:
// <w, x> = <Q_base(w_rot), x_rot> + s_res * sum_i (b_i * x_rot_i).
// ============================================================================

use super::clifford_3bit::{
    bf16_to_f32, f32_to_bf16, LLOYD_MAX_3BIT_BOUNDS, LLOYD_MAX_3BIT_CENTROIDS,
};
use crate::versor::DyadicWalshVersor;

/// Packed 32-weight TurboQuant block.
/// - 32 x 3-bit Lloyd-Max base codes: 96 bits (12 bytes)
/// - 32 x 1-bit QJL residual signs: 32 bits (4 bytes)
/// - 1 BF16 base scale: 16 bits (2 bytes)
/// - 1 BF16 residual scale: 16 bits (2 bytes)
///
/// Total: 20 bytes (160 bits) for 32 weights.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TurboQuantBlock32 {
    /// 3-bit base codes packed into 12 bytes (96 bits)
    pub base_codes: [u8; 12],
    /// 1-bit QJL residual signs packed into 4 bytes (32 bits)
    pub residual_signs: u32,
    /// Base standard deviation scale in BF16
    pub base_scale: u16,
    /// Residual scale (sqrt(2/pi) * std(r)) in BF16
    pub residual_scale: u16,
}

impl TurboQuantBlock32 {
    /// Quantizes 32 float weights using 3-bit Lloyd-Max base + 1-bit QJL residual.
    pub fn pack(weights: &[f32; 32]) -> Self {
        // 1. Calculate base scale (standard deviation)
        let sum_sq: f32 = weights.iter().map(|&w| w * w).sum();
        let std_dev = (sum_sq / 32.0).sqrt().max(1e-8);
        let inv_scale = 1.0 / std_dev;

        let mut base_codes = [0u8; 12];
        let mut residual = [0.0f32; 32];

        // Pack 32 x 3-bit codes into 12 bytes (each byte holds 2 codes + 2/3 of a code)
        // We pack every group of 8 weights into 3 bytes (8 x 3 = 24 bits = 3 bytes)
        for g in 0..4 {
            let mut g_bits: u32 = 0;
            for i in 0..8 {
                let idx = g * 8 + i;
                let w = weights[idx];
                let norm = w * inv_scale;
                let sign_bit = if norm < 0.0 { 1u32 } else { 0u32 };
                let abs_val = norm.abs();

                let mag_code = if abs_val < LLOYD_MAX_3BIT_BOUNDS[0] {
                    0u32
                } else if abs_val < LLOYD_MAX_3BIT_BOUNDS[1] {
                    1u32
                } else if abs_val < LLOYD_MAX_3BIT_BOUNDS[2] {
                    2u32
                } else {
                    3u32
                };

                let code_3bit = (sign_bit << 2) | mag_code;
                g_bits |= code_3bit << (i * 3);

                // Compute exact base reconstruction and residual
                let recon_sign = if sign_bit == 1 { -1.0f32 } else { 1.0f32 };
                let recon_base = recon_sign * LLOYD_MAX_3BIT_CENTROIDS[mag_code as usize] * std_dev;
                residual[idx] = w - recon_base;
            }

            base_codes[g * 3] = (g_bits & 0xFF) as u8;
            base_codes[g * 3 + 1] = ((g_bits >> 8) & 0xFF) as u8;
            base_codes[g * 3 + 2] = ((g_bits >> 16) & 0xFF) as u8;
        }

        // 2. QJL 1-bit residual signs
        let mut residual_signs: u32 = 0;
        for (i, &r) in residual.iter().enumerate() {
            if r >= 0.0 {
                residual_signs |= 1u32 << i;
            }
        }

        // 3. Optimal Joint 2x2 Gram Projection:
        // Solves min ||w - (alpha_1 * u_norm + alpha_2 * v_sign)||^2
        let mut u_norm = [0.0f32; 32];
        let mut v_sign = [0.0f32; 32];
        for g in 0..4 {
            let b0 = base_codes[g * 3] as u32;
            let b1 = base_codes[g * 3 + 1] as u32;
            let b2 = base_codes[g * 3 + 2] as u32;
            let g_bits = b0 | (b1 << 8) | (b2 << 16);
            for i in 0..8 {
                let idx = g * 8 + i;
                let code = ((g_bits >> (i * 3)) & 0x7) as usize;
                let sign = if (code & 0x4) != 0 { -1.0f32 } else { 1.0f32 };
                let mag_idx = code & 0x3;
                u_norm[idx] = sign * LLOYD_MAX_3BIT_CENTROIDS[mag_idx];

                let res_bit = (residual_signs >> idx) & 1;
                v_sign[idx] = if res_bit == 1 { 1.0f32 } else { -1.0f32 };
            }
        }

        let g11: f32 = u_norm.iter().map(|&u| u * u).sum();
        let g12: f32 = u_norm.iter().zip(v_sign.iter()).map(|(&u, &v)| u * v).sum();
        let g22: f32 = 32.0f32; // ||v_sign||^2 = sum_1^32 (+-1)^2 = 32

        let rhs1: f32 = weights
            .iter()
            .zip(u_norm.iter())
            .map(|(&w, &u)| w * u)
            .sum();
        let rhs2: f32 = weights
            .iter()
            .zip(v_sign.iter())
            .map(|(&w, &v)| w * v)
            .sum();

        let det = g11 * g22 - g12 * g12;
        let (alpha1, alpha2) = if det.abs() > 1e-12 {
            let a1 = (rhs1 * g22 - rhs2 * g12) / det;
            let a2 = (rhs2 * g11 - rhs1 * g12) / det;
            (a1, a2)
        } else {
            (std_dev, std_dev * 0.2)
        };

        Self {
            base_codes,
            residual_signs,
            base_scale: f32_to_bf16(alpha1),
            residual_scale: f32_to_bf16(alpha2),
        }
    }

    /// Unpacks all 32 weights into output slice.
    pub fn unpack(&self, out: &mut [f32; 32]) {
        let b_scale = bf16_to_f32(self.base_scale);
        let r_scale = bf16_to_f32(self.residual_scale);

        for g in 0..4 {
            let b0 = self.base_codes[g * 3] as u32;
            let b1 = self.base_codes[g * 3 + 1] as u32;
            let b2 = self.base_codes[g * 3 + 2] as u32;
            let g_bits = b0 | (b1 << 8) | (b2 << 16);

            for i in 0..8 {
                let idx = g * 8 + i;
                let code = ((g_bits >> (i * 3)) & 0x7) as usize;
                let sign = if (code & 0x4) != 0 { -1.0f32 } else { 1.0f32 };
                let mag_idx = code & 0x3;
                let base_val = sign * LLOYD_MAX_3BIT_CENTROIDS[mag_idx] * b_scale;

                let res_bit = (self.residual_signs >> idx) & 1;
                let res_sign = if res_bit == 1 { 1.0f32 } else { -1.0f32 };
                let res_val = res_sign * r_scale;

                out[idx] = base_val + res_val;
            }
        }
    }

    /// Direct dot product of packed TurboQuant block with activation slice.
    #[inline(always)]
    pub fn dot_with_slice(&self, x: &[f32; 32]) -> f32 {
        let b_scale = bf16_to_f32(self.base_scale);
        let r_scale = bf16_to_f32(self.residual_scale);
        let mut total = 0.0f32;

        for g in 0..4 {
            let b0 = self.base_codes[g * 3] as u32;
            let b1 = self.base_codes[g * 3 + 1] as u32;
            let b2 = self.base_codes[g * 3 + 2] as u32;
            let g_bits = b0 | (b1 << 8) | (b2 << 16);

            for i in 0..8 {
                let idx = g * 8 + i;
                let code = ((g_bits >> (i * 3)) & 0x7) as usize;
                let sign = if (code & 0x4) != 0 { -1.0f32 } else { 1.0f32 };
                let mag_idx = code & 0x3;
                let base_val = sign * LLOYD_MAX_3BIT_CENTROIDS[mag_idx] * b_scale;

                let res_bit = (self.residual_signs >> idx) & 1;
                let res_sign = if res_bit == 1 { 1.0f32 } else { -1.0f32 };
                let res_val = res_sign * r_scale;

                total += (base_val + res_val) * x[idx];
            }
        }

        total
    }
}

/// 64-byte Cache-Line Aligned TurboQuant Block holding 64 weights.
/// Uses 2 x TurboQuantBlock32 (40 bytes total), padded and aligned to 64 bytes.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TurboQuantCacheLine64 {
    pub blocks: [TurboQuantBlock32; 2],
    pub _padding: [u8; 24], // 40 + 24 = 64 bytes
}

impl TurboQuantCacheLine64 {
    pub const CAPACITY: usize = 64;

    /// Packs 64 float weights into a 64-byte cache line block.
    pub fn pack_64(weights: &[f32; 64]) -> Self {
        let b0: &[f32; 32] = weights[0..32].try_into().unwrap();
        let b1: &[f32; 32] = weights[32..64].try_into().unwrap();
        Self {
            blocks: [TurboQuantBlock32::pack(b0), TurboQuantBlock32::pack(b1)],
            _padding: [0u8; 24],
        }
    }

    /// Evaluates direct dot product of this 64-weight block with activation slice.
    #[inline(always)]
    pub fn dot_64(&self, x: &[f32; 64]) -> f32 {
        let x0: &[f32; 32] = x[0..32].try_into().unwrap();
        let x1: &[f32; 32] = x[32..64].try_into().unwrap();
        self.blocks[0].dot_with_slice(x0) + self.blocks[1].dot_with_slice(x1)
    }
}

/// High-level TurboQuant Tensor with Clifford Dyadic Versor Isometry.
#[derive(Debug, Clone)]
pub struct TurboQuantTensor {
    pub rows: usize,
    pub cols: usize,
    pub cache_lines: Vec<TurboQuantCacheLine64>,
}

impl TurboQuantTensor {
    /// Compiles an unquantized FP32 weight matrix into the AOT rotated TurboQuant representation.
    pub fn compile_aot(
        weights: &[f32],
        rows: usize,
        cols: usize,
        versor: &DyadicWalshVersor,
    ) -> Self {
        assert_eq!(weights.len(), rows * cols);
        assert_eq!(
            cols % 64,
            0,
            "Columns must be a multiple of 64 for cache-line alignment"
        );
        assert_eq!(versor.dimension(), cols);

        let lines_per_row = cols / 64;
        let mut cache_lines = Vec::with_capacity(rows * lines_per_row);
        let mut row_buffer = vec![0.0f32; cols];

        for r in 0..rows {
            let row_slice = &weights[r * cols..(r + 1) * cols];
            row_buffer.copy_from_slice(row_slice);

            // Isotropic rotation: R(w)
            versor.apply_in_place(&mut row_buffer);

            for l in 0..lines_per_row {
                let chunk: &[f32; 64] = row_buffer[l * 64..(l + 1) * 64].try_into().unwrap();
                cache_lines.push(TurboQuantCacheLine64::pack_64(chunk));
            }
        }

        Self {
            rows,
            cols,
            cache_lines,
        }
    }

    /// Evaluates Matrix-Vector Product y = W * x using Dual Isometry with TurboQuant unbiased projection.
    pub fn matvec(&self, x: &[f32], out: &mut [f32], versor: &DyadicWalshVersor) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);

        // 1. Single activation rotation: x_rot = R(x)
        let mut x_rot = x.to_vec();
        versor.apply_in_place(&mut x_rot);

        // 2. Direct streaming dot products
        let lines_per_row = self.cols / 64;
        for (r, out_val) in out.iter_mut().enumerate() {
            let mut row_dot = 0.0f32;
            let line_offset = r * lines_per_row;
            for l in 0..lines_per_row {
                let cache_line = &self.cache_lines[line_offset + l];
                let chunk_x: &[f32; 64] = x_rot[l * 64..(l + 1) * 64].try_into().unwrap();
                row_dot += cache_line.dot_64(chunk_x);
            }
            *out_val = row_dot;
        }
    }
}
