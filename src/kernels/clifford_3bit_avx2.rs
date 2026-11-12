// ============================================================================
// VERSORC-MATH: AVX2 SPECIALIZED KERNEL FOR CLIFFORD 3-BIT DOT PRODUCTS
// License: Apache-2.0 OR MIT
//
// Eliminates ALU decompression bottlenecks on x86_64 via:
// 1. vpbroadcastd + vpsrlvd + vpand: Parallel extraction of 8 x 3-bit codes.
// 2. vpermps (_mm256_permutevar8x32_ps): 1-cycle gather across the 8-float
//    Lloyd-Max centroid table [+c0, +c1, +c2, +c3, -c0, -c1, -c2, -c3].
//    Reconstructs 8 signed floats in a single CPU cycle.
// 3. FMA accumulation: _mm256_fmadd_ps directly into vector registers.
// ============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::quantization::clifford_3bit::{
    Clifford3BitCacheLine64, Clifford3BitWord64, LLOYD_MAX_3BIT_CENTROIDS,
};
use crate::versor::DyadicWalshVersor;

/// Evaluates dot product of a single 16-weight word against a slice of 16 activations using AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_clifford_3bit_word_avx2(word: Clifford3BitWord64, x: &[f32; 16]) -> f32 {
    let table = _mm256_setr_ps(
        LLOYD_MAX_3BIT_CENTROIDS[0],
        LLOYD_MAX_3BIT_CENTROIDS[1],
        LLOYD_MAX_3BIT_CENTROIDS[2],
        LLOYD_MAX_3BIT_CENTROIDS[3],
        -LLOYD_MAX_3BIT_CENTROIDS[0],
        -LLOYD_MAX_3BIT_CENTROIDS[1],
        -LLOYD_MAX_3BIT_CENTROIDS[2],
        -LLOYD_MAX_3BIT_CENTROIDS[3],
    );

    let shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    let mask_7 = _mm256_set1_epi32(7);

    // Weights 0..7 (bits 0..23)
    let bits0 = (word.0 & 0x00FF_FFFF) as i32;
    let bcast0 = _mm256_set1_epi32(bits0);
    let shifted0 = _mm256_srlv_epi32(bcast0, shifts);
    let idx0 = _mm256_and_si256(shifted0, mask_7);
    let w0 = _mm256_permutevar8x32_ps(table, idx0);

    // Weights 8..15 (bits 24..47)
    let bits1 = ((word.0 >> 24) & 0x00FF_FFFF) as i32;
    let bcast1 = _mm256_set1_epi32(bits1);
    let shifted1 = _mm256_srlv_epi32(bcast1, shifts);
    let idx1 = _mm256_and_si256(shifted1, mask_7);
    let w1 = _mm256_permutevar8x32_ps(table, idx1);

    let x0 = _mm256_loadu_ps(x.as_ptr());
    let x1 = _mm256_loadu_ps(x.as_ptr().add(8));

    let dot0 = _mm256_mul_ps(w0, x0);
    let dot1 = _mm256_fmadd_ps(w1, x1, dot0);

    let scale_vec = _mm256_set1_ps(word.scale());
    let scaled = _mm256_mul_ps(dot1, scale_vec);

    hsum256_ps(scaled)
}

/// Evaluates dot product of an entire 64-byte cache line (128 weights) using AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_clifford_3bit_cache_line_avx2(line: &Clifford3BitCacheLine64, x: &[f32; 128]) -> f32 {
    let table = _mm256_setr_ps(
        LLOYD_MAX_3BIT_CENTROIDS[0],
        LLOYD_MAX_3BIT_CENTROIDS[1],
        LLOYD_MAX_3BIT_CENTROIDS[2],
        LLOYD_MAX_3BIT_CENTROIDS[3],
        -LLOYD_MAX_3BIT_CENTROIDS[0],
        -LLOYD_MAX_3BIT_CENTROIDS[1],
        -LLOYD_MAX_3BIT_CENTROIDS[2],
        -LLOYD_MAX_3BIT_CENTROIDS[3],
    );

    let shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    let mask_7 = _mm256_set1_epi32(7);
    let mut total_acc = _mm256_setzero_ps();

    for b in 0..8 {
        let word = line.words[b];
        let x_ptr = x.as_ptr().add(b * 16);

        // Weights 0..7
        let bits0 = (word.0 & 0x00FF_FFFF) as i32;
        let bcast0 = _mm256_set1_epi32(bits0);
        let shifted0 = _mm256_srlv_epi32(bcast0, shifts);
        let idx0 = _mm256_and_si256(shifted0, mask_7);
        let w0 = _mm256_permutevar8x32_ps(table, idx0);

        // Weights 8..15
        let bits1 = ((word.0 >> 24) & 0x00FF_FFFF) as i32;
        let bcast1 = _mm256_set1_epi32(bits1);
        let shifted1 = _mm256_srlv_epi32(bcast1, shifts);
        let idx1 = _mm256_and_si256(shifted1, mask_7);
        let w1 = _mm256_permutevar8x32_ps(table, idx1);

        let x0 = _mm256_loadu_ps(x_ptr);
        let x1 = _mm256_loadu_ps(x_ptr.add(8));

        let dot0 = _mm256_mul_ps(w0, x0);
        let dot_both = _mm256_fmadd_ps(w1, x1, dot0);

        let scale_vec = _mm256_set1_ps(word.scale());
        total_acc = _mm256_fmadd_ps(dot_both, scale_vec, total_acc);
    }

    hsum256_ps(total_acc)
}

/// Fallback for non-x86_64 architectures.
#[cfg(not(target_arch = "x86_64"))]
pub fn dot_clifford_3bit_cache_line_avx2(line: &Clifford3BitCacheLine64, x: &[f32; 128]) -> f32 {
    line.dot_128(x)
}

/// Fused MatVec over pre-rotated activation: Evaluates y = W * x_rot with zero heap allocation.
pub fn matvec_clifford_3bit_prerotated_avx2(
    cache_lines: &[Clifford3BitCacheLine64],
    rows: usize,
    cols: usize,
    x_rot: &[f32],
    out: &mut [f32],
) {
    assert_eq!(x_rot.len(), cols);
    assert_eq!(out.len(), rows);
    assert_eq!(cols % 128, 0);
    let lines_per_row = cols / 128;
    assert_eq!(cache_lines.len(), rows * lines_per_row);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            for r in 0..rows {
                let mut sum = 0.0f32;
                let offset = r * lines_per_row;
                for l in 0..lines_per_row {
                    let line = &cache_lines[offset + l];
                    let chunk_x: &[f32; 128] = x_rot[l * 128..(l + 1) * 128].try_into().unwrap();
                    sum += unsafe { dot_clifford_3bit_cache_line_avx2(line, chunk_x) };
                }
                out[r] = sum;
            }
            return;
        }
    }

    // Scalar fallback
    for r in 0..rows {
        let mut sum = 0.0f32;
        let offset = r * lines_per_row;
        for l in 0..lines_per_row {
            let line = &cache_lines[offset + l];
            let chunk_x: &[f32; 128] = x_rot[l * 128..(l + 1) * 128].try_into().unwrap();
            sum += line.dot_128(chunk_x);
        }
        out[r] = sum;
    }
}

/// Fused MatVec over raw byte slices: Zero-copy cast and evaluate.
pub fn matvec_clifford_3bit_bytes_prerotated_avx2(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    x_rot: &[f32],
    out: &mut [f32],
) {
    let lines_per_row = cols / 128;
    let expected_bytes = rows * lines_per_row * 64;
    assert!(bytes.len() >= expected_bytes);
    let count = rows * lines_per_row;
    let cache_lines = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const Clifford3BitCacheLine64, count)
    };
    matvec_clifford_3bit_prerotated_avx2(cache_lines, rows, cols, x_rot, out);
}

/// Fused MatVec: Evaluates y = W * x for an entire matrix using Clifford 3-bit AVX2.
pub fn matvec_clifford_3bit_avx2(
    cache_lines: &[Clifford3BitCacheLine64],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
    versor: &DyadicWalshVersor,
) {
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    assert_eq!(cols % 128, 0);

    // 1. Single activation rotation: R(x)
    let mut x_rot = x.to_vec();
    versor.apply_in_place(&mut x_rot);

    // 2. Direct streaming dot products using AVX2
    matvec_clifford_3bit_prerotated_avx2(cache_lines, rows, cols, &x_rot, out);
}

/// Horizontal sum of 8 floats in an AVX2 register.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}
