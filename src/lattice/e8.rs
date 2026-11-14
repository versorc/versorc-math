// ============================================================================
// VERSORC-MATH: GOSSET E8 ROOT LATTICE & CONWAY-SLOANE FAST DECODER
// License: Apache-2.0
//
// E8 is the unique even unimodular lattice in dimension 8.
// Composed of the union of two D8 cosets: E8 = D8 U (D8 + 1/2 * 1).
// Possesses exactly 240 minimal root versors with norm squared ||r||^2 = 2.0.
// ============================================================================

pub const E8_DIMENSION: usize = 8;
pub const E8_NUM_ROOTS: usize = 240;
pub const E8_CODEBOOK_SIZE: usize = 1024; // Exactly 1 KiB
pub const E8_NUM_CENTROIDS: usize = E8_CODEBOOK_SIZE / E8_DIMENSION; // 128 centroids in 8D

/// E8 Lattice Codebook of 1 KiB.
/// Explicitly aligned to 64 bytes to guarantee residence in exactly 16 contiguous L1d cache lines.
#[repr(C, align(64))]
#[derive(Debug, Clone, PartialEq)]
pub struct E8LatticeCodebook {
    /// 128 centroids x 8 integer coordinates (scaled int8) = 1024 bytes
    pub table: [i8; E8_CODEBOOK_SIZE],
}

const _: () = assert!(
    std::mem::size_of::<E8LatticeCodebook>() == E8_CODEBOOK_SIZE,
    "E8LatticeCodebook must measure exactly 1024 bytes (1 KiB)"
);

impl Default for E8LatticeCodebook {
    fn default() -> Self {
        Self::new()
    }
}

impl E8LatticeCodebook {
    /// Initializes E8P codebook with canonical E8 half-integer roots scaled by 2 (coordinates in {+1, -1}).
    /// Every 8D centroid has an even number of minus signs, satisfying E8 membership.
    pub fn new() -> Self {
        let mut table = [0i8; E8_CODEBOOK_SIZE];
        let mut idx = 0;
        for c in 0..E8_NUM_CENTROIDS {
            let mut parity = 0;
            for d in 0..7 {
                let sign = if ((c >> d) & 1) == 0 { 1i8 } else { -1i8 };
                if sign == -1 {
                    parity ^= 1;
                }
                table[idx] = sign;
                idx += 1;
            }
            // 8th coordinate enforces even number of negative signs
            table[idx] = if parity == 0 { 1i8 } else { -1i8 };
            idx += 1;
        }
        Self { table }
    }

    /// Reads centroid `index` (0..128) from the 1-KiB L1d codebook and scales back to half-integer E8 roots.
    #[inline]
    pub fn decode_centroid(&self, index: usize, scale: f32) -> [f32; E8_DIMENSION] {
        assert!(index < E8_NUM_CENTROIDS, "E8 centroid index out of bounds");
        let base = index * E8_DIMENSION;
        let factor = 0.5 * scale;
        let mut out = [0.0f32; E8_DIMENSION];
        for (d, dst) in out.iter_mut().enumerate() {
            *dst = (self.table[base + d] as f32) * factor;
        }
        out
    }

    /// Searches the 1-KiB `self.table` codebook for the half-integer E8 root centroid maximizing
    /// inner product `<x, c_i>` (nearest E8P root under equal radius).
    pub fn nearest_centroid(&self, x: &[f32; E8_DIMENSION]) -> usize {
        let mut best_idx = 0usize;
        let mut best_dot = f32::NEG_INFINITY;
        for c in 0..E8_NUM_CENTROIDS {
            let base = c * E8_DIMENSION;
            let dot: f32 = x
                .iter()
                .zip(self.table[base..base + E8_DIMENSION].iter())
                .map(|(&xi, &ti)| xi * (ti as f32))
                .sum();
            if dot > best_dot {
                best_dot = dot;
                best_idx = c;
            }
        }
        best_idx
    }

    /// Decodes a 4-bit nibble [0..15] back into its 2D constituent weights (w0, w1).
    /// Invertible bijection with project_and_encode_to_nibbles. Unbiased symmetric levels in [-1.5, +1.5].
    #[inline(always)]
    pub fn decode_nibble(nibble: u8) -> (f32, f32) {
        let q0 = (nibble & 0x03) as f32;
        let q1 = ((nibble >> 2) & 0x03) as f32;
        let w0 = q0 - 1.5;
        let w1 = q1 - 1.5;
        (w0, w1)
    }

    /// Decodes 4 4-bit nibbles (2 bytes) back to an 8-dimensional E8 vector.
    #[inline(always)]
    pub fn decode_4_nibbles(nibbles: &[u8; 4], scale: f32) -> [f32; E8_DIMENSION] {
        let mut out = [0.0f32; E8_DIMENSION];
        for k in 0..4 {
            let (w0, w1) = Self::decode_nibble(nibbles[k]);
            out[2 * k] = w0 * scale;
            out[2 * k + 1] = w1 * scale;
        }
        out
    }

    /// Projects an 8D continuous vector onto the closest vector in the Conway-Sloane half-integer E8 coset (D8 + 1/2*1),
    /// then discretizes each 2D sub-vector to 1 of 16 symmetric zero-mean states (4 bits: 2 bits per coordinate).
    /// Levels per coordinate: {-1.5, -0.5, +0.5, +1.5} * scale. Strictly unbiased (mean = 0.0).
    pub fn project_and_encode_to_nibbles(x: &[f32; E8_DIMENSION], scale: f32) -> [u8; 4] {
        let inv_scale = if scale > 1e-7 { 1.0 / scale } else { 1.0 };
        let mut shifted_x = [0.0f32; E8_DIMENSION];
        for d in 0..E8_DIMENSION {
            shifted_x[d] = x[d] * inv_scale - 0.5;
        }

        let p_int = E8Lattice::project_to_d8(&shifted_x);
        let mut nibbles = [0u8; 4];
        for k in 0..4 {
            let w0 = p_int[2 * k] as f32 + 0.5;
            let w1 = p_int[2 * k + 1] as f32 + 0.5;
            let q0 = ((w0 + 1.5).round() as i32).clamp(0, 3) as u8;
            let q1 = ((w1 + 1.5).round() as i32).clamp(0, 3) as u8;
            nibbles[k] = (q0 & 0x03) | ((q1 & 0x03) << 2);
        }
        nibbles
    }
}

/// The 8-dimensional Gosset E8 lattice structure and Conway-Sloane closest-point decoder.
pub struct E8Lattice;

impl E8Lattice {
    /// Generates all 240 minimal root vectors of the E8 lattice in R^8.
    /// - 112 roots in D8: permutations of (+-1, +-1, 0, 0, 0, 0, 0, 0).
    /// - 128 roots in D8 + 1/2*1: (+-1/2, ..., +-1/2) with an even number of minus signs.
    ///
    /// Every root satisfies ||r||^2 = 2.0.
    pub fn generate_240_roots() -> [[f32; E8_DIMENSION]; E8_NUM_ROOTS] {
        let mut roots = [[0.0f32; E8_DIMENSION]; E8_NUM_ROOTS];
        let mut idx = 0;

        // 1. Generate 112 integer roots in D8: (+-1, +-1, 0^6)
        for i in 0..E8_DIMENSION {
            for j in (i + 1)..E8_DIMENSION {
                for &s_i in &[1.0f32, -1.0f32] {
                    for &s_j in &[1.0f32, -1.0f32] {
                        let mut root = [0.0f32; E8_DIMENSION];
                        root[i] = s_i;
                        root[j] = s_j;
                        roots[idx] = root;
                        idx += 1;
                    }
                }
            }
        }
        assert_eq!(idx, 112, "Group 1 must contain exactly 112 roots");

        // 2. Generate 128 half-integer roots in D8 + 1/2*1: (+-1/2)^8 with even negative count
        for mask in 0..256u32 {
            // Parity of negative signs: count number of set bits in mask
            if mask.count_ones() % 2 == 0 {
                let mut root = [0.0f32; E8_DIMENSION];
                for (d, dst) in root.iter_mut().enumerate() {
                    *dst = if ((mask >> d) & 1) == 0 {
                        0.5f32
                    } else {
                        -0.5f32
                    };
                }
                roots[idx] = root;
                idx += 1;
            }
        }
        assert_eq!(idx, 240, "Total E8 minimal roots must equal exactly 240");

        roots
    }

    /// Fast Conway & Sloane closest vector algorithm for the D8 checkerboard lattice.
    /// In D8, coordinates are integers and their sum is even.
    pub fn project_to_d8(x: &[f32; E8_DIMENSION]) -> [i32; E8_DIMENSION] {
        let mut f = [0i32; E8_DIMENSION];
        let mut errors = [0.0f32; E8_DIMENSION];
        let mut sum_f = 0i32;

        for i in 0..E8_DIMENSION {
            f[i] = x[i].round() as i32;
            errors[i] = (x[i] - f[i] as f32).abs();
            sum_f += f[i];
        }

        // If sum of rounded integers is odd, flip the coordinate with the largest error
        if sum_f % 2 != 0 {
            let mut max_err_idx = 0;
            let mut max_err = errors[0];
            for (i, &err) in errors.iter().enumerate().skip(1) {
                if err > max_err {
                    max_err = err;
                    max_err_idx = i;
                }
            }

            if x[max_err_idx] >= f[max_err_idx] as f32 {
                f[max_err_idx] += 1;
            } else {
                f[max_err_idx] -= 1;
            }
        }

        f
    }

    /// Fast Conway & Sloane Closest Vector Algorithm for E8:
    /// E8 = D8 U (D8 + 1/2 * 1)
    /// Finds the globally closest lattice point in strictly linear O(8) operations.
    pub fn project_to_e8(x: &[f32; E8_DIMENSION]) -> [f32; E8_DIMENSION] {
        // Candidate 1: Nearest point in D8
        let p1_int = Self::project_to_d8(x);
        let mut p1 = [0.0f32; E8_DIMENSION];
        let mut dist_sq1 = 0.0f32;
        for i in 0..E8_DIMENSION {
            p1[i] = p1_int[i] as f32;
            let diff = x[i] - p1[i];
            dist_sq1 += diff * diff;
        }

        // Candidate 2: Nearest point in shifted coset D8 + 1/2 * 1
        let mut shifted_x = [0.0f32; E8_DIMENSION];
        for i in 0..E8_DIMENSION {
            shifted_x[i] = x[i] - 0.5;
        }
        let p2_int = Self::project_to_d8(&shifted_x);
        let mut p2 = [0.0f32; E8_DIMENSION];
        let mut dist_sq2 = 0.0f32;
        for i in 0..E8_DIMENSION {
            p2[i] = p2_int[i] as f32 + 0.5;
            let diff = x[i] - p2[i];
            dist_sq2 += diff * diff;
        }

        // Return candidate with minimal Euclidean distance
        if dist_sq1 <= dist_sq2 {
            p1
        } else {
            p2
        }
    }
}
