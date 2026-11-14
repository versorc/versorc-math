// ============================================================================
// VERSORC-MATH: AVX2 SPECIALIZED KERNEL FOR CLIFFORD 3-BIT DOT PRODUCTS
// License: Apache-2.0
//
// Eliminates ALU decompression bottlenecks on x86_64 via:
// 1. vpbroadcastd + vpsrlvd + vpand: Parallel extraction of 8 x 3-bit codes.
// 2. vpermps (_mm256_permutevar8x32_ps): 1-cycle gather across the 8-float
//    Lloyd-Max centroid table [+c0, +c1, +c2, +c3, -c0, -c1, -c2, -c3].
// 3. FMA accumulation with hoisted YMM constants and dual accumulators.
// ============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::quantization::clifford_3bit::{
    Clifford3BitCacheLine64, Clifford3BitWord64, LLOYD_MAX_3BIT_CENTROIDS,
};
use crate::versor::DyadicWalshVersor;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn avx2_centroid_table() -> __m256 {
    _mm256_setr_ps(
        LLOYD_MAX_3BIT_CENTROIDS[0],
        LLOYD_MAX_3BIT_CENTROIDS[1],
        LLOYD_MAX_3BIT_CENTROIDS[2],
        LLOYD_MAX_3BIT_CENTROIDS[3],
        -LLOYD_MAX_3BIT_CENTROIDS[0],
        -LLOYD_MAX_3BIT_CENTROIDS[1],
        -LLOYD_MAX_3BIT_CENTROIDS[2],
        -LLOYD_MAX_3BIT_CENTROIDS[3],
    )
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn accumulate_word_avx2(
    word: u64,
    x_ptr: *const f32,
    table: __m256,
    shifts: __m256i,
    mask_7: __m256i,
    acc: __m256,
) -> __m256 {
    let bits0 = (word & 0x00FF_FFFF) as i32;
    let bcast0 = _mm256_set1_epi32(bits0);
    let shifted0 = _mm256_srlv_epi32(bcast0, shifts);
    let idx0 = _mm256_and_si256(shifted0, mask_7);
    let w0 = _mm256_permutevar8x32_ps(table, idx0);

    let bits1 = ((word >> 24) & 0x00FF_FFFF) as i32;
    let bcast1 = _mm256_set1_epi32(bits1);
    let shifted1 = _mm256_srlv_epi32(bcast1, shifts);
    let idx1 = _mm256_and_si256(shifted1, mask_7);
    let w1 = _mm256_permutevar8x32_ps(table, idx1);

    let x0 = _mm256_loadu_ps(x_ptr);
    let x1 = _mm256_loadu_ps(x_ptr.add(8));

    let dot0 = _mm256_mul_ps(w0, x0);
    let dot_both = _mm256_fmadd_ps(w1, x1, dot0);

    let scale = f32::from_bits(((word >> 32) as u32) & 0xFFFF_0000);
    let scale_vec = _mm256_set1_ps(scale);
    _mm256_fmadd_ps(dot_both, scale_vec, acc)
}

/// Evaluates dot product of a single 16-weight word against a slice of 16 activations using AVX2.
///
/// # Safety
/// Caller must guarantee that the host CPU supports AVX2 and FMA instructions (`is_x86_feature_detected!("avx2")`
/// and `is_x86_feature_detected!("fma")`), and that `x` points to a valid 16-element float array.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_clifford_3bit_word_avx2(word: Clifford3BitWord64, x: &[f32; 16]) -> f32 {
    let table = avx2_centroid_table();
    let shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    let mask_7 = _mm256_set1_epi32(7);
    let acc = accumulate_word_avx2(
        word.0,
        x.as_ptr(),
        table,
        shifts,
        mask_7,
        _mm256_setzero_ps(),
    );
    hsum256_ps(acc)
}

/// Scalar fallback for non-x86_64 architectures.
#[cfg(not(target_arch = "x86_64"))]
pub fn dot_clifford_3bit_word_avx2(word: Clifford3BitWord64, x: &[f32; 16]) -> f32 {
    word.dot_with_slice(x)
}

/// Evaluates dot product of an entire 64-byte cache line (128 weights) using AVX2.
///
/// # Safety
/// Caller must guarantee that the host CPU supports AVX2 and FMA instructions (`is_x86_feature_detected!("avx2")`
/// and `is_x86_feature_detected!("fma")`), and that `x` points to a valid 128-element float array.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_clifford_3bit_cache_line_avx2(
    line: &Clifford3BitCacheLine64,
    x: &[f32; 128],
) -> f32 {
    let table = avx2_centroid_table();
    let shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    let mask_7 = _mm256_set1_epi32(7);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    for pair in 0..4 {
        let b0 = pair * 2;
        let b1 = b0 + 1;
        acc0 = accumulate_word_avx2(
            line.words[b0].0,
            x.as_ptr().add(b0 * 16),
            table,
            shifts,
            mask_7,
            acc0,
        );
        acc1 = accumulate_word_avx2(
            line.words[b1].0,
            x.as_ptr().add(b1 * 16),
            table,
            shifts,
            mask_7,
            acc1,
        );
    }

    hsum256_ps(_mm256_add_ps(acc0, acc1))
}

/// Fallback for non-x86_64 architectures.
#[cfg(not(target_arch = "x86_64"))]
pub fn dot_clifford_3bit_cache_line_avx2(line: &Clifford3BitCacheLine64, x: &[f32; 128]) -> f32 {
    line.dot_128(x)
}

/// Hoists YMM constants outside all loops so the inner 128-weight cache-line kernel is fully inlined.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matvec_rows_avx2(
    cache_lines: &[Clifford3BitCacheLine64],
    rows: usize,
    lines_per_row: usize,
    x_rot: &[f32],
    out: &mut [f32],
) {
    let table = avx2_centroid_table();
    let shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    let mask_7 = _mm256_set1_epi32(7);

    for (r, out_val) in out.iter_mut().enumerate().take(rows) {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let row_lines = &cache_lines[r * lines_per_row..(r + 1) * lines_per_row];

        for (l, line) in row_lines.iter().enumerate() {
            let line_x_ptr = x_rot.as_ptr().add(l * 128);
            for pair in 0..4 {
                let b0 = pair * 2;
                let b1 = b0 + 1;
                acc0 = accumulate_word_avx2(
                    line.words[b0].0,
                    line_x_ptr.add(b0 * 16),
                    table,
                    shifts,
                    mask_7,
                    acc0,
                );
                acc1 = accumulate_word_avx2(
                    line.words[b1].0,
                    line_x_ptr.add(b1 * 16),
                    table,
                    shifts,
                    mask_7,
                    acc1,
                );
            }
        }

        *out_val = hsum256_ps(_mm256_add_ps(acc0, acc1));
    }
}

/// Reads cache-line bytes safely via unaligned loads (zero UB) and evaluates with hoisted AVX2 constants.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matvec_rows_unaligned_bytes_avx2(
    bytes: &[u8],
    rows: usize,
    lines_per_row: usize,
    x_rot: &[f32],
    out: &mut [f32],
) {
    let table = avx2_centroid_table();
    let shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    let mask_7 = _mm256_set1_epi32(7);
    let row_bytes = lines_per_row * 64;

    for (r, out_val) in out.iter_mut().enumerate().take(rows) {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let row_slice = &bytes[r * row_bytes..(r + 1) * row_bytes];

        for (l, line_bytes) in row_slice.chunks_exact(64).enumerate() {
            let words_ptr = line_bytes.as_ptr() as *const [u64; 8];
            let words = std::ptr::read_unaligned(words_ptr);
            let line_x_ptr = x_rot.as_ptr().add(l * 128);
            for pair in 0..4 {
                let b0 = pair * 2;
                let b1 = b0 + 1;
                let w0 = u64::from_le(words[b0]);
                let w1 = u64::from_le(words[b1]);
                acc0 =
                    accumulate_word_avx2(w0, line_x_ptr.add(b0 * 16), table, shifts, mask_7, acc0);
                acc1 =
                    accumulate_word_avx2(w1, line_x_ptr.add(b1 * 16), table, shifts, mask_7, acc1);
            }
        }

        *out_val = hsum256_ps(_mm256_add_ps(acc0, acc1));
    }
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
            unsafe {
                matvec_rows_avx2(cache_lines, rows, lines_per_row, x_rot, out);
            }
            return;
        }
    }

    for (r, out_val) in out.iter_mut().enumerate().take(rows) {
        let mut sum = 0.0f32;
        let offset = r * lines_per_row;
        for l in 0..lines_per_row {
            let line = &cache_lines[offset + l];
            let chunk_x: &[f32; 128] = x_rot[l * 128..(l + 1) * 128].try_into().unwrap();
            sum += line.dot_128(chunk_x);
        }
        *out_val = sum;
    }
}

/// Fused MatVec over raw byte slices: fast-path zero-copy when 64-byte aligned,
/// or UB-free unaligned streaming otherwise.
pub fn matvec_clifford_3bit_bytes_prerotated_avx2(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    x_rot: &[f32],
    out: &mut [f32],
) {
    assert_eq!(x_rot.len(), cols);
    assert_eq!(out.len(), rows);
    assert_eq!(cols % 128, 0);
    let lines_per_row = cols / 128;
    let expected_bytes = rows * lines_per_row * 64;
    assert!(bytes.len() >= expected_bytes);
    let used_bytes = &bytes[..expected_bytes];

    if let Ok(cache_lines) = Clifford3BitCacheLine64::slice_from_bytes(used_bytes) {
        matvec_clifford_3bit_prerotated_avx2(cache_lines, rows, cols, x_rot, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                matvec_rows_unaligned_bytes_avx2(used_bytes, rows, lines_per_row, x_rot, out);
            }
            return;
        }
    }

    for (r, out_val) in out.iter_mut().enumerate().take(rows) {
        let mut sum = 0.0f32;
        let row_bytes = lines_per_row * 64;
        let row_slice = &used_bytes[r * row_bytes..(r + 1) * row_bytes];
        for (l, chunk) in row_slice.chunks_exact(64).enumerate() {
            let line_arr: &[u8; 64] = chunk.try_into().unwrap();
            let line = Clifford3BitCacheLine64::from_le_bytes(line_arr);
            let chunk_x: &[f32; 128] = x_rot[l * 128..(l + 1) * 128].try_into().unwrap();
            sum += line.dot_128(chunk_x);
        }
        *out_val = sum;
    }
}

/// Fused MatVec reusing a caller-supplied scratch buffer of length `cols`.
pub fn matvec_clifford_3bit_with_scratch_avx2(
    cache_lines: &[Clifford3BitCacheLine64],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
    versor: &DyadicWalshVersor,
    scratch: &mut [f32],
) {
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    assert_eq!(scratch.len(), cols);
    assert_eq!(cols % 128, 0);

    scratch.copy_from_slice(x);
    versor.apply_in_place(scratch);

    matvec_clifford_3bit_prerotated_avx2(cache_lines, rows, cols, scratch, out);
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
    assert_eq!(cols % 128, 0);
    let mut scratch = vec![0.0f32; cols];
    matvec_clifford_3bit_with_scratch_avx2(cache_lines, rows, cols, x, out, versor, &mut scratch);
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
