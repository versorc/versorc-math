// ============================================================================
// VERSORC-MATH: CLIFFORD 3-BIT GEOMETRIC QUANTIZER & DOD CACHE-LINE PACKAGING
// License: Apache-2.0 OR MIT
//
// Combines Clifford Dyadic Versor Isometry (R in Spin(N)) with Optimal
// 3-Bit Lloyd-Max Scalar Centroids for Isotropized Normal Distributions N(0, sigma^2).
//
// 1. Dual Isometry: <V^-1 w_rot, x> == <w_rot, V x>.
//    Rotates activation x ONCE in O(N log N) additions/subtractions;
//    weights remain packed in memory and NEVER undergo inverse transforms at inference.
// 2. Hardware DOD Packaging:
//    - Clifford3BitWord64: 16 weights (48 bits) + 1 BF16 scale (16 bits) = 64 bits (8 bytes).
//    - Clifford3BitCacheLine64: 8 words = 128 weights in exactly 64 bytes (1 L1d cache line).
// ============================================================================

use crate::versor::DyadicWalshVersor;

/// Precomputed Lloyd-Max centroids for N(0, 1) standard normal distribution (3-bit symmetric, 8 levels).
pub const LLOYD_MAX_3BIT_CENTROIDS: [f32; 4] = [
    0.245_086_3, // Code 0: 0.2451 sigma
    0.756_041_4, // Code 1: 0.7560 sigma
    1.343_918_2, // Code 2: 1.3439 sigma
    2.152_014_2, // Code 3: 2.1520 sigma
];

/// Precomputed Lloyd-Max decision boundaries for N(0, 1).
pub const LLOYD_MAX_3BIT_BOUNDS: [f32; 3] = [
    0.500_563_8, // Boundary 0-1
    1.049_979_8, // Boundary 1-2
    1.747_966_2, // Boundary 2-3
];

/// Single 64-bit universal storage word.
/// Contains exactly 16 3-bit quantized weights (48 bits) + 1 BF16 scale (16 bits).
/// Total: 64 bits (8 bytes). Zero bit waste.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Clifford3BitWord64(pub u64);

impl Clifford3BitWord64 {
    /// Encodes 16 float weights into a single 64-bit word.
    /// - Low 48 bits: 16 x 3-bit quantized codes (1 sign bit + 2 magnitude bits).
    /// - High 16 bits: BF16 scale factor.
    pub fn pack(weights: &[f32; 16], scale: f32) -> Self {
        let inv_scale = if scale.abs() > 1e-12 { 1.0 / scale } else { 1.0 };
        let mut bits: u64 = 0;

        for (i, &w) in weights.iter().enumerate() {
            let normalized = w * inv_scale;
            let sign_bit = if normalized < 0.0 { 1u64 } else { 0u64 };
            let abs_val = normalized.abs();

            let mag_code = if abs_val < LLOYD_MAX_3BIT_BOUNDS[0] {
                0u64
            } else if abs_val < LLOYD_MAX_3BIT_BOUNDS[1] {
                1u64
            } else if abs_val < LLOYD_MAX_3BIT_BOUNDS[2] {
                2u64
            } else {
                3u64
            };

            let code_3bit = (sign_bit << 2) | mag_code;
            bits |= code_3bit << (i * 3);
        }

        let bf16_scale = f32_to_bf16(scale) as u64;
        bits |= bf16_scale << 48;

        Self(bits)
    }

    /// Encodes 16 float weights into a single 64-bit word using an exact orthogonal least-squares scale:
    /// alpha* = <w, c_norm> / ||c_norm||^2.
    /// Strictly minimizes reconstruction error ||w - alpha* c_norm||^2.
    pub fn pack_least_squares(weights: &[f32; 16]) -> Self {
        let sum_sq: f32 = weights.iter().map(|&w| w * w).sum();
        let std_dev = (sum_sq / 16.0).sqrt().max(1e-8);
        let inv_std = 1.0 / std_dev;

        let mut bits: u64 = 0;
        let mut dot_wc = 0.0f32;
        let mut norm_c_sq = 0.0f32;

        for (i, &w) in weights.iter().enumerate() {
            let normalized = w * inv_std;
            let sign_bit = if normalized < 0.0 { 1u64 } else { 0u64 };
            let abs_val = normalized.abs();

            let mag_code = if abs_val < LLOYD_MAX_3BIT_BOUNDS[0] {
                0usize
            } else if abs_val < LLOYD_MAX_3BIT_BOUNDS[1] {
                1usize
            } else if abs_val < LLOYD_MAX_3BIT_BOUNDS[2] {
                2usize
            } else {
                3usize
            };

            let code_3bit = (sign_bit << 2) | (mag_code as u64);
            bits |= code_3bit << (i * 3);

            let sign_f = if sign_bit != 0 { -1.0f32 } else { 1.0f32 };
            let c_norm = sign_f * LLOYD_MAX_3BIT_CENTROIDS[mag_code];
            dot_wc += w * c_norm;
            norm_c_sq += c_norm * c_norm;
        }

        let alpha_ls = if norm_c_sq > 1e-12 { dot_wc / norm_c_sq } else { std_dev };
        let bf16_scale = f32_to_bf16(alpha_ls) as u64;
        bits |= bf16_scale << 48;

        Self(bits)
    }

    /// Extracts the BF16 scale factor as FP32.
    #[inline(always)]
    pub fn scale(&self) -> f32 {
        let bf16 = (self.0 >> 48) as u16;
        bf16_to_f32(bf16)
    }

    /// Unpacks all 16 reconstructed float weights into the destination buffer.
    pub fn unpack(&self, out: &mut [f32; 16]) {
        let scale = self.scale();
        for i in 0..16 {
            let code = ((self.0 >> (i * 3)) & 0x7) as usize;
            let sign = if (code & 0x4) != 0 { -1.0f32 } else { 1.0f32 };
            let mag_idx = code & 0x3;
            out[i] = sign * LLOYD_MAX_3BIT_CENTROIDS[mag_idx] * scale;
        }
    }

    /// Direct dot product of the packed 16 weights with an unpacked FP32 activation slice.
    /// Operates without intermediate memory allocation.
    #[inline(always)]
    pub fn dot_with_slice(&self, x: &[f32; 16]) -> f32 {
        let scale = self.scale();
        let mut sum = 0.0f32;
        for i in 0..16 {
            let code = ((self.0 >> (i * 3)) & 0x7) as usize;
            let sign = if (code & 0x4) != 0 { -1.0f32 } else { 1.0f32 };
            let mag_idx = code & 0x3;
            let w = sign * LLOYD_MAX_3BIT_CENTROIDS[mag_idx] * scale;
            sum += w * x[i];
        }
        sum
    }
}

/// 64-byte Cache-Line Block of 128 weights.
/// Holds exactly 8 x Clifford3BitWord64 words = 128 weights.
/// Size is exactly 64 bytes (1 L1d cache line) and 64-byte cache-line aligned.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Clifford3BitCacheLine64 {
    pub words: [Clifford3BitWord64; 8],
}

impl Default for Clifford3BitCacheLine64 {
    fn default() -> Self {
        Self {
            words: [Clifford3BitWord64(0); 8],
        }
    }
}

impl Clifford3BitCacheLine64 {
    pub const CAPACITY: usize = 128;

    /// Packs 128 float weights into a single 64-byte cache line block.
    /// Automatically computes optimal per-16 sub-block standard deviation scales.
    pub fn pack_128(weights: &[f32; 128]) -> Self {
        let mut words = [Clifford3BitWord64(0); 8];
        for b in 0..8 {
            let chunk_weights: &[f32; 16] = weights[b * 16..(b + 1) * 16].try_into().unwrap();
            words[b] = Clifford3BitWord64::pack_least_squares(chunk_weights);
        }
        Self { words }
    }

    /// Unpacks all 128 float weights into destination slice.
    pub fn unpack_128(&self, out: &mut [f32; 128]) {
        for b in 0..8 {
            let chunk_out: &mut [f32; 16] = (&mut out[b * 16..(b + 1) * 16]).try_into().unwrap();
            self.words[b].unpack(chunk_out);
        }
    }

    /// Returns a direct 64-byte slice view of this cache line block.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 64] {
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// Converts a slice of cache lines into a contiguous byte slice without allocations.
    #[inline(always)]
    pub fn slice_as_bytes(slice: &[Self]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 64)
        }
    }

    /// Casts an aligned slice of bytes into a slice of 64-byte cache lines.
    #[inline(always)]
    pub fn slice_from_bytes(bytes: &[u8]) -> Result<&[Self], String> {
        if bytes.len() % 64 != 0 {
            return Err(format!("Byte slice length {} is not a multiple of 64", bytes.len()));
        }
        if bytes.as_ptr() as usize % 64 != 0 {
            return Err(format!(
                "Byte slice pointer {:p} is not 64-byte cache-line aligned",
                bytes.as_ptr()
            ));
        }
        let count = bytes.len() / 64;
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const Self, count) })
    }

    /// Evaluates direct dot product of this 128-weight cache line against an activation slice.
    #[inline(always)]
    pub fn dot_128(&self, x: &[f32; 128]) -> f32 {
        let mut total = 0.0f32;
        for b in 0..8 {
            let chunk_x: &[f32; 16] = x[b * 16..(b + 1) * 16].try_into().unwrap();
            total += self.words[b].dot_with_slice(chunk_x);
        }
        total
    }
}

/// High-level Geometric 3-bit Quantized Tensor with Dyadic Versor Isometry.
#[derive(Debug, Clone)]
pub struct Clifford3BitTensor {
    pub rows: usize,
    pub cols: usize,
    pub cache_lines: Vec<Clifford3BitCacheLine64>,
}

impl Clifford3BitTensor {
    /// Compiles an unquantized FP32 weight matrix into the AOT rotated Clifford 3-bit representation.
    /// Each row is first rotated by `versor` in-place, then packed into 64-byte cache lines.
    pub fn compile_aot(weights: &[f32], rows: usize, cols: usize, versor: &DyadicWalshVersor) -> Self {
        assert_eq!(weights.len(), rows * cols);
        assert_eq!(cols % 128, 0, "Columns must be a multiple of 128 for cache-line alignment");
        assert_eq!(versor.dimension(), cols);

        let lines_per_row = cols / 128;
        let mut cache_lines = Vec::with_capacity(rows * lines_per_row);
        let mut row_buffer = vec![0.0f32; cols];

        for r in 0..rows {
            let row_slice = &weights[r * cols..(r + 1) * cols];
            row_buffer.copy_from_slice(row_slice);

            // Isotropic rotation: R(w)
            versor.apply_in_place(&mut row_buffer);

            for l in 0..lines_per_row {
                let chunk: &[f32; 128] = row_buffer[l * 128..(l + 1) * 128].try_into().unwrap();
                cache_lines.push(Clifford3BitCacheLine64::pack_128(chunk));
            }
        }

        Self {
            rows,
            cols,
            cache_lines,
        }
    }

    /// Evaluates Matrix-Vector Product y = W * x using Dual Isometry:
    /// <W_row, x> == <R(W_row), R(x)>.
    /// The activation x is rotated ONCE in O(N log N); weights are operated on directly in 3-bit packed cache lines.
    pub fn matvec(&self, x: &[f32], out: &mut [f32], versor: &DyadicWalshVersor) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);

        // 1. Single activation rotation: x_rot = R(x)
        let mut x_rot = x.to_vec();
        versor.apply_in_place(&mut x_rot);

        // 2. Direct streaming dot products over 64-byte cache lines
        let lines_per_row = self.cols / 128;
        for r in 0..self.rows {
            let mut row_dot = 0.0f32;
            let line_offset = r * lines_per_row;
            for l in 0..lines_per_row {
                let cache_line = &self.cache_lines[line_offset + l];
                let chunk_x: &[f32; 128] = x_rot[l * 128..(l + 1) * 128].try_into().unwrap();
                row_dot += cache_line.dot_128(chunk_x);
            }
            out[r] = row_dot;
        }
    }
}

/// Converts an IEEE 754 f32 float to BF16 (Brain Floating Point 16-bit).
#[inline(always)]
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    // Round to nearest even
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Converts a BF16 bit pattern back to IEEE 754 f32 float.
#[inline(always)]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
