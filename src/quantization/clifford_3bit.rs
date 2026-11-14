// ============================================================================
// VERSORC-MATH: CLIFFORD 3-BIT GEOMETRIC QUANTIZER & DOD CACHE-LINE PACKAGING
// License: Apache-2.0
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

/// Dynamically computes optimal Lloyd-Max 1D scalar quantization centroids and decision boundaries
/// for an empirical distribution of sample data via iterative expectation-maximization.
pub fn compute_lloyd_max_1d(
    samples: &[f32],
    num_levels: usize,
    max_iters: usize,
    tolerance: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(num_levels >= 2, "Lloyd-Max requires at least 2 levels");
    assert!(!samples.is_empty(), "Samples cannot be empty");

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();

    // Initialize centroids evenly spaced across quantiles
    let mut centroids = Vec::with_capacity(num_levels);
    for k in 0..num_levels {
        let q_idx = ((k as f32 + 0.5) / num_levels as f32 * n as f32) as usize;
        centroids.push(sorted[q_idx.min(n - 1)]);
    }

    let mut bounds = vec![0.0f32; num_levels - 1];

    for _ in 0..max_iters {
        // 1. Update decision boundaries
        for k in 0..num_levels - 1 {
            bounds[k] = 0.5 * (centroids[k] + centroids[k + 1]);
        }

        // 2. Assign samples to clusters and compute new centroids
        let mut sums = vec![0.0f32; num_levels];
        let mut counts = vec![0usize; num_levels];

        for &x in &sorted {
            let mut assigned = num_levels - 1;
            for (k, &b) in bounds.iter().enumerate() {
                if x < b {
                    assigned = k;
                    break;
                }
            }
            sums[assigned] += x;
            counts[assigned] += 1;
        }

        let mut max_shift = 0.0f32;
        for k in 0..num_levels {
            if counts[k] > 0 {
                let new_c = sums[k] / (counts[k] as f32);
                let shift = (new_c - centroids[k]).abs();
                if shift > max_shift {
                    max_shift = shift;
                }
                centroids[k] = new_c;
            }
        }

        if max_shift < tolerance {
            break;
        }
    }

    // Final boundary update
    for k in 0..num_levels - 1 {
        bounds[k] = 0.5 * (centroids[k] + centroids[k + 1]);
    }

    (centroids, bounds)
}

/// Dynamically computes optimal 3-bit symmetric Lloyd-Max magnitude centroids (4 levels)
/// and decision boundaries (3 bounds) for normalized absolute weights `|w| / sigma`.
///
/// The resulting `(centroids, bounds)` pair can be passed directly into
/// [`Clifford3BitWord64::pack_with_codebook`], [`Clifford3BitWord64::pack_least_squares_with_codebook`],
/// and [`Clifford3BitWord64::unpack_with_codebook`] for layer-calibrated quantization.
pub fn compute_symmetric_lloyd_max_3bit(
    abs_samples: &[f32],
    max_iters: usize,
    tolerance: f32,
) -> ([f32; 4], [f32; 3]) {
    let (centroids, bounds) = compute_lloyd_max_1d(abs_samples, 4, max_iters, tolerance);
    let mut c4 = [0.0f32; 4];
    let mut b3 = [0.0f32; 3];
    c4.copy_from_slice(&centroids[..4]);
    b3.copy_from_slice(&bounds[..3]);
    (c4, b3)
}

/// Single 64-bit universal storage word.
/// Contains exactly 16 3-bit quantized weights (48 bits) + 1 BF16 scale (16 bits).
/// Total: 64 bits (8 bytes). Zero bit waste.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Clifford3BitWord64(pub u64);

impl Clifford3BitWord64 {
    /// Encodes 16 normalized weights (`w * inv_scale`) into 48 packed 3-bit codes
    /// (sign bit + 2 magnitude bits) using `bounds`, while accumulating `<w, c_norm>`
    /// and `||c_norm||^2` against `centroids` for optional least-squares scaling.
    #[inline(always)]
    fn encode_bits(
        weights: &[f32; 16],
        inv_scale: f32,
        centroids: &[f32; 4],
        bounds: &[f32; 3],
    ) -> (u64, f32, f32) {
        let mut bits: u64 = 0;
        let mut dot_wc = 0.0f32;
        let mut norm_c_sq = 0.0f32;

        for (i, &w) in weights.iter().enumerate() {
            let normalized = w * inv_scale;
            let sign_bit = if normalized < 0.0 { 1u64 } else { 0u64 };
            let abs_val = normalized.abs();

            let mag_code = if abs_val < bounds[0] {
                0usize
            } else if abs_val < bounds[1] {
                1usize
            } else if abs_val < bounds[2] {
                2usize
            } else {
                3usize
            };

            let code_3bit = (sign_bit << 2) | (mag_code as u64);
            bits |= code_3bit << (i * 3);

            let sign_f = if sign_bit != 0 { -1.0f32 } else { 1.0f32 };
            let c_norm = sign_f * centroids[mag_code];
            dot_wc += w * c_norm;
            norm_c_sq += c_norm * c_norm;
        }

        (bits, dot_wc, norm_c_sq)
    }

    /// Encodes 16 float weights into a single 64-bit word using the default
    /// N(0, 1) Lloyd-Max bounds and the provided scale.
    pub fn pack(weights: &[f32; 16], scale: f32) -> Self {
        Self::pack_with_codebook(weights, scale, &LLOYD_MAX_3BIT_BOUNDS)
    }

    /// Encodes 16 float weights into a single 64-bit word using caller-provided
    /// decision boundaries (e.g. from [`compute_symmetric_lloyd_max_3bit`]).
    /// - Low 48 bits: 16 x 3-bit quantized codes (1 sign bit + 2 magnitude bits).
    /// - High 16 bits: BF16 scale factor.
    pub fn pack_with_codebook(weights: &[f32; 16], scale: f32, bounds: &[f32; 3]) -> Self {
        let inv_scale = if scale.abs() > 1e-12 {
            1.0 / scale
        } else {
            1.0
        };
        let (mut bits, _, _) =
            Self::encode_bits(weights, inv_scale, &LLOYD_MAX_3BIT_CENTROIDS, bounds);
        let bf16_scale = f32_to_bf16(scale) as u64;
        bits |= bf16_scale << 48;
        Self(bits)
    }

    /// Encodes 16 float weights into a single 64-bit word using the default
    /// N(0, 1) Lloyd-Max codebook and exact orthogonal least-squares scaling:
    /// `alpha* = <w, c_norm> / ||c_norm||^2`.
    /// Strictly minimizes reconstruction error `||w - alpha* c_norm||^2`.
    pub fn pack_least_squares(weights: &[f32; 16]) -> Self {
        Self::pack_least_squares_with_codebook(
            weights,
            &LLOYD_MAX_3BIT_CENTROIDS,
            &LLOYD_MAX_3BIT_BOUNDS,
        )
    }

    /// Encodes 16 float weights using custom symmetric Lloyd-Max `(centroids, bounds)`
    /// and orthogonal least-squares scaling `alpha* = <w, c_norm> / ||c_norm||^2`.
    pub fn pack_least_squares_with_codebook(
        weights: &[f32; 16],
        centroids: &[f32; 4],
        bounds: &[f32; 3],
    ) -> Self {
        let sum_sq: f32 = weights.iter().map(|&w| w * w).sum();
        let std_dev = (sum_sq / 16.0).sqrt().max(1e-8);
        let inv_std = 1.0 / std_dev;

        let (mut bits, dot_wc, norm_c_sq) = Self::encode_bits(weights, inv_std, centroids, bounds);

        let alpha_ls = if norm_c_sq > 1e-12 {
            dot_wc / norm_c_sq
        } else {
            std_dev
        };
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

    /// Unpacks all 16 reconstructed float weights using the default N(0, 1) centroids.
    pub fn unpack(&self, out: &mut [f32; 16]) {
        self.unpack_with_codebook(out, &LLOYD_MAX_3BIT_CENTROIDS);
    }

    /// Unpacks all 16 reconstructed float weights using caller-provided magnitude centroids
    /// matching the codebook used at pack time.
    pub fn unpack_with_codebook(&self, out: &mut [f32; 16], centroids: &[f32; 4]) {
        let scale = self.scale();
        for (i, slot) in out.iter_mut().enumerate() {
            let code = ((self.0 >> (i * 3)) & 0x7) as usize;
            let sign = if (code & 0x4) != 0 { -1.0f32 } else { 1.0f32 };
            let mag_idx = code & 0x3;
            *slot = sign * centroids[mag_idx] * scale;
        }
    }

    /// Direct dot product of the packed 16 weights with an unpacked FP32 activation slice.
    /// Operates without intermediate memory allocation.
    #[inline(always)]
    pub fn dot_with_slice(&self, x: &[f32; 16]) -> f32 {
        let scale = self.scale();
        let mut sum = 0.0f32;
        for (i, &xv) in x.iter().enumerate() {
            let code = ((self.0 >> (i * 3)) & 0x7) as usize;
            let sign = if (code & 0x4) != 0 { -1.0f32 } else { 1.0f32 };
            let mag_idx = code & 0x3;
            let w = sign * LLOYD_MAX_3BIT_CENTROIDS[mag_idx] * scale;
            sum += w * xv;
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

    /// Encodes a single 64-byte cache line into little-endian bytes without `unsafe`.
    pub fn to_le_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        for (i, w) in self.words.iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&w.0.to_le_bytes());
        }
        out
    }

    /// Decodes a single 64-byte cache line from arbitrarily aligned little-endian bytes without `unsafe`.
    pub fn from_le_bytes(bytes: &[u8; 64]) -> Self {
        let mut words = [Clifford3BitWord64(0); 8];
        for (i, w) in words.iter_mut().enumerate() {
            let chunk: [u8; 8] = bytes[i * 8..(i + 1) * 8].try_into().unwrap();
            *w = Clifford3BitWord64(u64::from_le_bytes(chunk));
        }
        Self { words }
    }

    /// Serializes a slice of cache lines to little-endian bytes without `unsafe`.
    pub fn slice_to_le_bytes(slice: &[Self]) -> Vec<u8> {
        let mut out = Vec::with_capacity(slice.len() * 64);
        for line in slice {
            for w in &line.words {
                out.extend_from_slice(&w.0.to_le_bytes());
            }
        }
        out
    }

    /// Returns a direct 64-byte slice view of this cache line block.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 64] {
        const {
            assert!(std::mem::size_of::<Self>() == 64);
            assert!(std::mem::align_of::<Self>() == 64);
        }
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// Converts a slice of cache lines into a contiguous byte slice without allocations.
    #[inline(always)]
    pub fn slice_as_bytes(slice: &[Self]) -> &[u8] {
        const {
            assert!(std::mem::size_of::<Self>() == 64);
            assert!(std::mem::align_of::<Self>() == 64);
        }
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 64) }
    }

    /// Casts an aligned slice of bytes into a slice of 64-byte cache lines.
    #[inline(always)]
    pub fn slice_from_bytes(bytes: &[u8]) -> Result<&[Self], String> {
        const {
            assert!(std::mem::size_of::<Self>() == 64);
            assert!(std::mem::align_of::<Self>() == 64);
        }
        if !bytes.len().is_multiple_of(64) {
            return Err(format!(
                "Byte slice length {} is not a multiple of 64",
                bytes.len()
            ));
        }
        if !(bytes.as_ptr() as usize).is_multiple_of(64) {
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
    pub fn compile_aot(
        weights: &[f32],
        rows: usize,
        cols: usize,
        versor: &DyadicWalshVersor,
    ) -> Self {
        assert_eq!(weights.len(), rows * cols);
        assert_eq!(
            cols % 128,
            0,
            "Columns must be a multiple of 128 for cache-line alignment"
        );
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
    /// Automatically utilizes 8-way AVX2 fused vectorization when available on the host CPU.
    pub fn matvec(&self, x: &[f32], out: &mut [f32], versor: &DyadicWalshVersor) {
        let mut scratch = vec![0.0f32; self.cols];
        self.matvec_with_scratch(x, out, versor, &mut scratch);
    }

    /// Evaluates Matrix-Vector Product y = W * x reusing a caller-provided scratch buffer
    /// of length `self.cols` to avoid heap allocations in hot loops.
    pub fn matvec_with_scratch(
        &self,
        x: &[f32],
        out: &mut [f32],
        versor: &DyadicWalshVersor,
        scratch: &mut [f32],
    ) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        assert_eq!(scratch.len(), self.cols);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                crate::kernels::clifford_3bit_avx2::matvec_clifford_3bit_with_scratch_avx2(
                    &self.cache_lines,
                    self.rows,
                    self.cols,
                    x,
                    out,
                    versor,
                    scratch,
                );
                return;
            }
        }

        self.matvec_scalar_with_scratch(x, out, versor, scratch);
    }

    /// Explicit scalar reference evaluation of Matrix-Vector Product y = W * x.
    pub fn matvec_scalar(&self, x: &[f32], out: &mut [f32], versor: &DyadicWalshVersor) {
        let mut scratch = vec![0.0f32; self.cols];
        self.matvec_scalar_with_scratch(x, out, versor, &mut scratch);
    }

    /// Explicit scalar evaluation reusing a caller-provided scratch buffer of length `self.cols`.
    pub fn matvec_scalar_with_scratch(
        &self,
        x: &[f32],
        out: &mut [f32],
        versor: &DyadicWalshVersor,
        scratch: &mut [f32],
    ) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        assert_eq!(scratch.len(), self.cols);

        scratch.copy_from_slice(x);
        versor.apply_in_place(scratch);

        let lines_per_row = self.cols / 128;
        for (r, out_val) in out.iter_mut().enumerate().take(self.rows) {
            let mut row_dot = 0.0f32;
            let line_offset = r * lines_per_row;
            for l in 0..lines_per_row {
                let cache_line = &self.cache_lines[line_offset + l];
                let chunk_x: &[f32; 128] = scratch[l * 128..(l + 1) * 128].try_into().unwrap();
                row_dot += cache_line.dot_128(chunk_x);
            }
            *out_val = row_dot;
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
