// ============================================================================
// VERSORC-MATH: EMPIRICAL SILICON AUDIT OF 4 KEY RISK VECTORS
// License: Apache-2.0
//
// 1. W_down Invariance: Proves y = W_down * h requires zero layer unrotation
//    when rotating rows of W_down by R_{d_ff} and h by R_{d_ff}.
// 2. Dyadic Restriction on d_ff_exp = 1408: Verifies exact 11 x H_128 isometry.
// 3. Silicon Throughput Benchmark: Measures ns/row and effective GB/s of
//    Asym64 (72 B / 128 w, 4.5 bpw) vs Clifford3Bit (64 B / 128 w, 4.0 bpw).
// 4. Angular Fidelity: Cosine similarity over a 1408 x 2048 matrix (2,883,584 weights).
// ============================================================================

use std::time::Instant;
use versorc_math::prelude::*;

#[test]
fn test_risk_1_w_down_invariance_and_zero_desrotation() {
    println!("====================================================================");
    println!(" RISK 1: SILICON VERIFICATION OF W_down INVARIANCE");
    println!("====================================================================");

    let d_model = 2048usize;
    let d_ff = 1408usize; // Non-power of 2

    // Versor over inner column dimension d_ff
    let versor_ff = DyadicWalshVersor::new(d_ff, 9999);

    // Generate W_down matrix (d_model rows x d_ff cols) and activation h (d_ff)
    let mut w_down = vec![0.0f32; d_model * d_ff];
    for r in 0..d_model {
        for c in 0..d_ff {
            let t = (r * d_ff + c) as f32 * 0.01;
            w_down[r * d_ff + c] = (t.sin() * 0.6) + (t * 0.5).cos() * 0.4;
        }
    }
    let mut h = vec![0.0f32; d_ff];
    for c in 0..d_ff {
        let t = (c as f32 + 42.0) * 0.05;
        h[c] = (t.sin() * 0.8) + (t * 0.3).cos() * 0.2;
    }

    // 1. Exact canonical product: y_exact = W_down * h (vector in R^d_model)
    let mut y_exact = vec![0.0f32; d_model];
    for r in 0..d_model {
        let row = &w_down[r * d_ff..(r + 1) * d_ff];
        y_exact[r] = row.iter().zip(h.iter()).map(|(&w, &x)| w * x).sum();
    }

    // 2. Clifford row isometry on W_down with R_{d_ff}:
    // Rotate each row of W_down by R_{d_ff} (performed by AOT compiler)
    let mut w_down_rot = w_down.clone();
    for r in 0..d_model {
        let row = &mut w_down_rot[r * d_ff..(r + 1) * d_ff];
        versor_ff.apply_in_place(row);
    }

    // Rotate activation h by R_{d_ff} (performed once at runtime)
    let mut h_rot = h.clone();
    versor_ff.apply_in_place(&mut h_rot);

    // 3. Evaluate dot product in rotated domain:
    let mut y_clifford = vec![0.0f32; d_model];
    for r in 0..d_model {
        let row_rot = &w_down_rot[r * d_ff..(r + 1) * d_ff];
        y_clifford[r] = row_rot.iter().zip(h_rot.iter()).map(|(&w, &x)| w * x).sum();
    }

    // Compare y_exact against y_clifford with ZERO output unrotation
    let mut max_diff = 0.0f32;
    for r in 0..d_model {
        let diff = (y_exact[r] - y_clifford[r]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    let norm_exact = (y_exact.iter().map(|&v| v * v).sum::<f32>()).sqrt();
    let rel_err = max_diff / norm_exact;

    println!("  - Exact norm ||y|| (R^2048)              : {:.6}", norm_exact);
    println!("  - Max absolute silicon difference        : {:.2e}", max_diff);
    println!("  - Relative error with ZERO unrotation    : {:.2e}", rel_err);
    println!("  -> VERDICT: Output y[r] already resides natively in d_model space.");
    println!("     Layer unrotation is mathematically redundant.");

    assert!(rel_err < 1e-5, "y_clifford must match y_exact without output unrotation");
}

#[test]
fn test_risk_2_diadic_restriction_on_1408() {
    println!("====================================================================");
    println!(" RISK 2: AUDIT OF DYADIC RESTRICTION ON d_ff_exp = 1408");
    println!("====================================================================");

    let d_ff = 1408usize;
    assert_eq!(d_ff % 128, 0, "1408 must be a multiple of 128");
    let n_blocks = d_ff / 128;
    assert_eq!(n_blocks, 11, "1408 must decompose into exactly 11 blocks of 128");

    let versor = DyadicWalshVersor::new(d_ff, 42);

    // Vector with outliers injected in distinct 128-element blocks
    let mut v = vec![0.0f32; d_ff];
    for i in 0..d_ff {
        v[i] = (i as f32 * 0.1).sin() * 0.5;
    }
    v[42] = 50.0;     // Outlier in block 0
    v[200] = -45.0;   // Outlier in block 1
    v[1350] = 60.0;   // Outlier in block 10

    let norm_orig: f32 = (v.iter().map(|&x| x * x).sum::<f32>()).sqrt();

    // Apply transformation
    let mut v_rot = v.clone();
    versor.apply_in_place(&mut v_rot);

    let norm_rot: f32 = (v_rot.iter().map(|&x| x * x).sum::<f32>()).sqrt();
    let norm_diff = (norm_orig - norm_rot).abs();

    // Outlier 60.0 / sqrt(128) ~= 60.0 / 11.31 ~= 5.30
    let max_rot_val = v_rot.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    // Invert transformation
    let mut v_recon = v_rot.clone();
    versor.invert_in_place(&mut v_recon);

    let mut max_recon_err = 0.0f32;
    for i in 0..d_ff {
        let err = (v[i] - v_recon[i]).abs();
        if err > max_recon_err {
            max_recon_err = err;
        }
    }

    println!("  - Hadamard blocks applied                : 11 x H_128 (exact 11 cache lines)");
    println!("  - Original norm ||x||                    : {:.6}", norm_orig);
    println!("  - Rotated norm ||R(x)||                  : {:.6}", norm_rot);
    println!("  - Isometry error (||x|| - ||R(x)||)      : {:.2e}", norm_diff);
    println!("  - Max input outlier                      : 60.0");
    println!("  - Max value after H_128                  : {:.4} (CLT smearing by sqrt(128))", max_rot_val);
    println!("  - Inversion error R^-1(R(x))             : {:.2e}", max_recon_err);
    println!("  -> VERDICT: 128-block transformation is an exact norm-preserving isometry.");

    assert!(norm_diff < 1e-4, "Isometry must be preserved with error < 1e-4");
    assert!(max_recon_err < 1e-5, "Reversibility must be exact with error < 1e-5");
}

// Strict reference emulation of VrcAsym64 (4.50 bpw, 36 bytes / 64 weights) for head-to-head comparison
fn pack_asym64_row(weights: &[f32]) -> Vec<u8> {
    assert_eq!(weights.len() % 64, 0);
    let n_blocks = weights.len() / 64;
    let mut bytes = vec![0u8; n_blocks * 36];

    for b in 0..n_blocks {
        let chunk = &weights[b * 64..(b + 1) * 64];
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &w in chunk {
            if w < min_val { min_val = w; }
            if w > max_val { max_val = w; }
        }
        let scale = if max_val > min_val { (max_val - min_val) / 15.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;

        let off = b * 36;
        let d_bf16 = f32_to_bf16(scale);
        let m_bf16 = f32_to_bf16(min_val);
        bytes[off..off + 2].copy_from_slice(&d_bf16.to_le_bytes());
        bytes[off + 2..off + 4].copy_from_slice(&m_bf16.to_le_bytes());

        for i in 0..32 {
            let q0 = ((chunk[i * 2] - min_val) * inv_scale).round().clamp(0.0, 15.0) as u8;
            let q1 = ((chunk[i * 2 + 1] - min_val) * inv_scale).round().clamp(0.0, 15.0) as u8;
            bytes[off + 4 + i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
    }
    bytes
}

#[inline(never)]
fn benchmark_dot_row_asym64(bytes: &[u8], x: &[f32], x_sums: &[f32]) -> f32 {
    let n_blocks = x.len() / 64;
    let mut acc = 0.0f32;
    for b in 0..n_blocks {
        let off = b * 36;
        let d = bf16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
        let m = bf16_to_f32(u16::from_le_bytes([bytes[off + 2], bytes[off + 3]]));

        let mut inner = 0.0f32;
        let chunk_x = &x[b * 64..(b + 1) * 64];
        for i in 0..32 {
            let byte = bytes[off + 4 + i];
            let q0 = (byte & 0x0F) as f32;
            let q1 = (byte >> 4) as f32;
            inner += q0 * chunk_x[i * 2] + q1 * chunk_x[i * 2 + 1];
        }
        acc += d * inner + m * x_sums[b];
    }
    acc
}

#[test]
fn test_risk_3_throughput_silicon_benchmark_1408x2048() {
    println!("====================================================================");
    println!(" RISK 3: EMPIRICAL SILICON BENCHMARK (MoE Matrix 1408 x 2048)");
    println!(" Total weights: 2,883,584 (Representative Expert Projection Layer)");
    println!("====================================================================");

    let rows = 1408usize;
    let cols = 2048usize;
    let total_weights = rows * cols; // 2,883,584 weights

    // 1. Generate deterministic weight matrix and input activation
    let mut weights = vec![0.0f32; total_weights];
    let mut state = 0xabcdef1234567890u64;
    for i in 0..total_weights {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (state >> 32) as f32 / 4294967296.0;
        weights[i] = (u - 0.5) * 0.1;
    }
    // Outliers
    weights[42] = 35.0;
    weights[350] = -28.0;

    let mut x = vec![0.0f32; cols];
    for i in 0..cols {
        x[i] = (i as f32 * 0.05).sin() * 0.8;
    }
    // Precompute block sums for Asym64
    let x_sums: Vec<f32> = x.chunks_exact(64).map(|c| c.iter().sum()).collect();

    let versor = DyadicWalshVersor::new(cols, 1234);

    // --- Pack into VrcAsym64 (4.50 bpw) ---
    let bytes_per_row_asym64 = (cols / 64) * 36; // 32 * 36 = 1152 bytes per row
    let total_bytes_asym64 = rows * bytes_per_row_asym64; // 1,622,016 bytes
    let mut asym64_matrix = vec![0u8; total_bytes_asym64];
    for r in 0..rows {
        let row_slice = &weights[r * cols..(r + 1) * cols];
        let row_bytes = pack_asym64_row(row_slice);
        asym64_matrix[r * bytes_per_row_asym64..(r + 1) * bytes_per_row_asym64].copy_from_slice(&row_bytes);
    }

    // --- Pack into Clifford3Bit (4.00 effective bpw, 64B / 128w) ---
    let tensor_3bit = Clifford3BitTensor::compile_aot(&weights, rows, cols, &versor);
    let total_bytes_3bit = tensor_3bit.cache_lines.len() * 64; // 1408 * 16 * 64 = 1,441,792 bytes

    println!(" Physical Memory Footprint:");
    println!("  - Asym64 Bytes per row (2048 w)      : {} B (4.50 bpw)", bytes_per_row_asym64);
    println!("  - Clifford3Bit Bytes per row (2048 w): {} B (4.00 bpw)", (cols / 128) * 64);
    println!("  - Net bandwidth reduction            : {:.2}% (from 1,152 B to 1,024 B per row)",
             (1.0 - (1024.0 / 1152.0)) * 100.0);
    println!("  - Total footprint Asym64             : {:.3} MiB", total_bytes_asym64 as f64 / 1048576.0);
    println!("  - Total footprint Clifford3Bit       : {:.3} MiB", total_bytes_3bit as f64 / 1048576.0);
    println!("--------------------------------------------------------------------");

    // Warmup
    let mut out_asym = vec![0.0f32; rows];
    let mut out_3bit = vec![0.0f32; rows];

    // Benchmark VrcAsym64
    let iters = 20usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        for r in 0..rows {
            let row_bytes = &asym64_matrix[r * bytes_per_row_asym64..(r + 1) * bytes_per_row_asym64];
            out_asym[r] = benchmark_dot_row_asym64(row_bytes, &x, &x_sums);
        }
    }
    let elapsed_asym = t0.elapsed() / (iters as u32);
    let us_asym = elapsed_asym.as_micros() as f64;
    let bw_asym = (total_bytes_asym64 as f64 / 1e9) / (elapsed_asym.as_secs_f64());

    // Benchmark Clifford3Bit (Scalar loop)
    let t1 = Instant::now();
    for _ in 0..iters {
        tensor_3bit.matvec(&x, &mut out_3bit, &versor);
    }
    let elapsed_3bit = t1.elapsed() / (iters as u32);
    let us_3bit = elapsed_3bit.as_micros() as f64;
    let bw_3bit = (total_bytes_3bit as f64 / 1e9) / (elapsed_3bit.as_secs_f64());

    // Benchmark Clifford3Bit (AVX2 vpermps kernel)
    let mut out_3bit_avx2 = vec![0.0f32; rows];
    let t2 = Instant::now();
    for _ in 0..iters {
        matvec_clifford_3bit_avx2(&tensor_3bit.cache_lines, rows, cols, &x, &mut out_3bit_avx2, &versor);
    }
    let elapsed_3bit_avx2 = t2.elapsed() / (iters as u32);
    let us_3bit_avx2 = elapsed_3bit_avx2.as_micros() as f64;
    let bw_3bit_avx2 = (total_bytes_3bit as f64 / 1e9) / (elapsed_3bit_avx2.as_secs_f64());

    // Exact FP32 reference for cosine similarity
    let mut out_exact = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &weights[r * cols..(r + 1) * cols];
        out_exact[r] = row.iter().zip(x.iter()).map(|(&w, &x)| w * x).sum();
    }

    let cos_asym = cosine_similarity(&out_exact, &out_asym);
    let cos_3bit = cosine_similarity(&out_exact, &out_3bit);
    let cos_3bit_avx2 = cosine_similarity(&out_exact, &out_3bit_avx2);

    println!(" SILICON MEASURED RESULTS:");
    println!("  [VrcAsym64 - 4.50 bpw (v5 Reference)]");
    println!("    - MatVec Latency (1408 x 2048)  : {:.1} us", us_asym);
    println!("    - Measured Bandwidth            : {:.2} GB/s", bw_asym);
    println!("    - Output Cosine vs FP32         : {:.5}", cos_asym);
    println!("  [Clifford3Bit - 4.00 bpw (Scalar Unpack)]");
    println!("    - MatVec Latency (1408 x 2048)  : {:.1} us", us_3bit);
    println!("    - Measured Bandwidth            : {:.2} GB/s", bw_3bit);
    println!("    - Output Cosine vs FP32         : {:.5}", cos_3bit);
    println!("  [Clifford3Bit - 4.00 bpw (AVX2 vpermps Kernel)]");
    println!("    - MatVec Latency (1408 x 2048)  : {:.1} us (including rotation)", us_3bit_avx2);
    println!("    - Measured Bandwidth            : {:.2} GB/s", bw_3bit_avx2);
    println!("    - Output Cosine vs FP32         : {:.5}", cos_3bit_avx2);
    println!("--------------------------------------------------------------------");
    println!(" AVX2 vs ASYM64 COMPARISON:");
    println!("  -> AVX2 Speedup over Scalar       : {:.2}x", us_3bit / us_3bit_avx2);
    if us_3bit_avx2 < us_asym {
        println!("  -> SILICON WIN: Clifford3Bit AVX2 is {:.2}x FASTER than VrcAsym64 ({:.1} us vs {:.1} us)!",
                 us_asym / us_3bit_avx2, us_3bit_avx2, us_asym);
    } else {
        println!("  -> Clifford3Bit AVX2 takes {:.1} us vs {:.1} us for Asym64 (ratio {:.2}x).",
                 us_3bit_avx2, us_asym, us_3bit_avx2 / us_asym);
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt()).max(1e-12)
}
