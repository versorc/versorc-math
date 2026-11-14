// ============================================================================
// VERSORC-MATH: AVX2 SIMD DIFFERENTIAL TEST SUITE (HARDWARE PARITY)
// License: Apache-2.0
//
// Rigorously validates AVX2 vectorized implementations against scalar reference
// algorithms across multiple dimensions, bit-packing layouts, and dual isometries.
// ============================================================================

use versorc_math::kernels::clifford_3bit_avx2::{
    dot_clifford_3bit_cache_line_avx2, dot_clifford_3bit_word_avx2,
    matvec_clifford_3bit_bytes_prerotated_avx2, matvec_clifford_3bit_prerotated_avx2,
};
use versorc_math::prelude::*;
use versorc_math::quantization::clifford_3bit::{Clifford3BitCacheLine64, Clifford3BitWord64};

#[test]
fn test_avx2_fwht_differential_multiple_dimensions() {
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 FWHT test: host CPU lacks AVX2");
            return;
        }

        // Test across all power-of-two dimensions from 8 to 4096
        let dimensions = [8usize, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

        for &n in &dimensions {
            let mut v_scalar = Vec::with_capacity(n);
            let mut v_avx2 = Vec::with_capacity(n);

            for i in 0..n {
                let val = (i as f32 * 0.17 + 0.5).sin() * 2.0 - (i as f32 * 0.05).cos();
                v_scalar.push(val);
                v_avx2.push(val);
            }

            // Reference scalar FWHT
            DyadicWalshVersor::fwht_scalar(&mut v_scalar);

            // AVX2 accelerated FWHT
            unsafe {
                DyadicWalshVersor::fwht_avx2(&mut v_avx2);
            }

            // Assert exact numerical parity
            for i in 0..n {
                let diff = (v_scalar[i] - v_avx2[i]).abs();
                assert!(
                    diff < 1e-5,
                    "FWHT AVX2 vs Scalar mismatch at dim={}, idx={}: scalar={}, avx2={}, diff={}",
                    n,
                    i,
                    v_scalar[i],
                    v_avx2[i],
                    diff
                );
            }
        }
    }
}

#[test]
fn test_avx2_clifford_3bit_word_and_cache_line_differential() {
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 Clifford 3-bit test: host lacks AVX2/FMA");
            return;
        }

        // 1. Single Word (16 weights) test over 1,000 distinct pseudo-random combinations
        for seed in 0..1000u64 {
            let raw_u64 = seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x55AA55AA55AA55AA;
            // Ensure bits 0..48 have valid centroids, scale != 0
            let scale_bits = (seed % 250) + 5; // scale != 0
            let word = Clifford3BitWord64((raw_u64 & 0x0000_FFFF_FFFF_FFFF) | (scale_bits << 48));

            let mut x16 = [0.0f32; 16];
            for (i, dst) in x16.iter_mut().enumerate() {
                let t = (seed as f32 * 0.1) + (i as f32 * 0.4);
                *dst = t.sin() * 1.5;
            }

            let scalar_dot = word.dot_with_slice(&x16);
            let avx2_dot = unsafe { dot_clifford_3bit_word_avx2(word, &x16) };

            let diff = (scalar_dot - avx2_dot).abs();
            assert!(
                diff < 1e-4,
                "Word AVX2 vs Scalar dot product mismatch at seed {}: scalar={}, avx2={}, diff={}",
                seed,
                scalar_dot,
                avx2_dot,
                diff
            );
        }

        // 2. Cache Line (128 weights, 64 bytes) test over 100 cache lines
        for seed in 0..100u64 {
            let mut line_input = [0.0f32; 128];
            let mut x128 = [0.0f32; 128];

            for i in 0..128 {
                let t = (seed as f32 * 0.2) + (i as f32 * 0.08);
                line_input[i] = t.cos() * 2.5;
                x128[i] = (t * 1.3).sin() * 1.2;
            }

            let line = Clifford3BitCacheLine64::pack_128(&line_input);
            let scalar_dot = line.dot_128(&x128);
            let avx2_dot = unsafe { dot_clifford_3bit_cache_line_avx2(&line, &x128) };

            let diff = (scalar_dot - avx2_dot).abs();
            assert!(
                diff < 1e-4,
                "CacheLine AVX2 vs Scalar dot product mismatch at seed {}: scalar={}, avx2={}, diff={}",
                seed, scalar_dot, avx2_dot, diff
            );
        }
    }
}

#[test]
fn test_avx2_matvec_bytes_prerotated_vs_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 matvec bytes test: host lacks AVX2/FMA");
            return;
        }

        let rows = 128usize;
        let cols = 256usize; // 2 cache lines per row
        let lines_per_row = cols / 128;

        let mut lines = Vec::with_capacity(rows * lines_per_row);
        for r in 0..rows {
            for l in 0..lines_per_row {
                let mut chunk = [0.0f32; 128];
                for (i, dst) in chunk.iter_mut().enumerate() {
                    let idx = r * cols + l * 128 + i;
                    *dst = (idx as f32 * 0.013).sin() * 2.0;
                }
                lines.push(Clifford3BitCacheLine64::pack_128(&chunk));
            }
        }

        let mut x_rot = Vec::with_capacity(cols);
        for i in 0..cols {
            x_rot.push((i as f32 * 0.021 + 0.3).cos() * 1.5);
        }

        // 1. Structured AVX2 execution
        let mut out_avx2 = vec![0.0f32; rows];
        matvec_clifford_3bit_prerotated_avx2(&lines, rows, cols, &x_rot, &mut out_avx2);

        // 2. Zero-copy aligned bytes AVX2 execution + unaligned offset slice soundness check
        let raw_bytes = Clifford3BitCacheLine64::slice_as_bytes(&lines);
        let mut out_bytes = vec![0.0f32; rows];
        matvec_clifford_3bit_bytes_prerotated_avx2(raw_bytes, rows, cols, &x_rot, &mut out_bytes);

        let mut unaligned_storage = vec![0u8; raw_bytes.len() + 1];
        unaligned_storage[1..].copy_from_slice(raw_bytes);
        let mut out_unaligned = vec![0.0f32; rows];
        matvec_clifford_3bit_bytes_prerotated_avx2(
            &unaligned_storage[1..],
            rows,
            cols,
            &x_rot,
            &mut out_unaligned,
        );

        // 3. Scalar reference execution
        let mut out_scalar = vec![0.0f32; rows];
        for (r, dst) in out_scalar.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            let offset = r * lines_per_row;
            for l in 0..lines_per_row {
                let line = &lines[offset + l];
                let chunk_x: &[f32; 128] = x_rot[l * 128..(l + 1) * 128].try_into().unwrap();
                sum += line.dot_128(chunk_x);
            }
            *dst = sum;
        }

        // Compare all three outputs
        for r in 0..rows {
            let diff_avx = (out_scalar[r] - out_avx2[r]).abs();
            let diff_bytes = (out_scalar[r] - out_bytes[r]).abs();

            assert!(
                diff_avx < 1e-4,
                "Row {} mismatch between Scalar and AVX2: scalar={}, avx2={}, diff={}",
                r,
                out_scalar[r],
                out_avx2[r],
                diff_avx
            );
            assert!(
                diff_bytes < 1e-4,
                "Row {} mismatch between Scalar and Zero-Copy Bytes AVX2: scalar={}, bytes={}, diff={}",
                r, out_scalar[r], out_bytes[r], diff_bytes
            );
            let diff_unaligned = (out_scalar[r] - out_unaligned[r]).abs();
            assert!(
                diff_unaligned < 1e-4,
                "Row {} mismatch between Scalar and Unaligned Bytes AVX2: scalar={}, unaligned={}, diff={}",
                r, out_scalar[r], out_unaligned[r], diff_unaligned
            );
        }
    }
}

#[test]
fn test_avx2_clifford_tensor_matvec_dual_isometry_avx2_vs_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 dual isometry test: host lacks AVX2/FMA");
            return;
        }

        let rows = 64usize;
        let cols = 128usize;
        let seed = 12345;
        let versor = DyadicWalshVersor::new(cols, seed);

        // Dense unquantized float matrix
        let mut dense_w = Vec::with_capacity(rows * cols);
        for i in 0..(rows * cols) {
            dense_w.push((i as f32 * 0.007).sin() * 1.8);
        }

        let tensor = Clifford3BitTensor::compile_aot(&dense_w, rows, cols, &versor);

        let mut x = Vec::with_capacity(cols);
        for i in 0..cols {
            x.push((i as f32 * 0.035).cos() * 1.1);
        }

        let mut out_avx2 = vec![0.0f32; rows];
        let mut out_scalar = vec![0.0f32; rows];

        tensor.matvec(&x, &mut out_avx2, &versor);
        tensor.matvec_scalar(&x, &mut out_scalar, &versor);

        for r in 0..rows {
            let diff = (out_avx2[r] - out_scalar[r]).abs();
            assert!(
                diff < 1e-4,
                "Tensor MatVec AVX2 vs Scalar mismatch at row {}: avx2={}, scalar={}, diff={}",
                r,
                out_avx2[r],
                out_scalar[r],
                diff
            );
        }
    }
}
