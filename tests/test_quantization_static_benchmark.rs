// ============================================================================
// VERSORC-MATH: DETERMINISTIC STATIC MATRIX QUANTIZATION AUDIT & OPTIMIZATION
// License: Apache-2.0
//
// Tests 3-bit Clifford and 3.5-bit TurboQuant on a fully deterministic static
// matrix of 256 rows x 2048 columns (524,288 weights) with heavy LLM outlier channels.
//
// Evaluates:
// 1. Baseline: Analytical RMS scale vs Optimal Least-Squares (LS) Projection Scale
// 2. Exact MAE vs Gaussian residual scale in TurboQuant
// 3. Joint 2x2 Gram Projection: min ||w - (alpha_1 u + alpha_2 v)||^2
// 4. Dot product error and inference throughput
// ============================================================================

use std::time::Instant;
use versorc_math::prelude::*;
use versorc_math::quantization::clifford_3bit::{
    compute_symmetric_lloyd_max_3bit, Clifford3BitWord64,
};

/// Generates a bit-identical, deterministic static matrix of size rows x cols.
/// Uses a 64-bit PCG state with Box-Muller transform to produce exact Gaussian distribution
/// N(0, sigma^2) with sigma = 0.025, plus 0.2% massive outlier channels (value = 45.0).
fn generate_deterministic_static_matrix(rows: usize, cols: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; rows * cols];
    let mut state = 0x853c49e6748fea9bu64; // Fixed deterministic seed

    let mut next_u32 = || -> u32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let xorshifted = (((state >> 18) ^ state) >> 27) as u32;
        let rot = (state >> 59) as u32;
        (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31))
    };

    let mut i = 0;
    while i < rows * cols {
        let u1 = (next_u32() as f32 + 1.0) / 4294967297.0;
        let u2 = (next_u32() as f32 + 1.0) / 4294967297.0;
        let r = (-2.0 * u1.ln()).sqrt() * 0.025;
        let theta = 2.0 * std::f32::consts::PI * u2;
        data[i] = r * theta.cos();
        if i + 1 < rows * cols {
            data[i + 1] = r * theta.sin();
        }
        i += 2;
    }

    // Inject row-modulated heavy outlier channels across all rows (columns 42, 350, 1020, 1800)
    let outlier_cols = [42usize, 350, 1020, 1800];
    let outlier_vals = [45.0f32, -38.5, 52.0, -41.2];
    for r in 0..rows {
        for (k, (&col, &base_val)) in outlier_cols.iter().zip(outlier_vals.iter()).enumerate() {
            let sign = if ((r + k) & 1) == 0 { 1.0f32 } else { -1.0f32 };
            let modulation = 0.4 + 0.12 * (((r * 7 + k * 5) % 11) as f32);
            data[r * cols + col] = base_val * sign * modulation;
        }
    }

    data
}

/// Generates a bit-identical deterministic activation vector.
fn generate_deterministic_activation(cols: usize) -> Vec<f32> {
    let mut act = vec![0.0f32; cols];
    let mut state = 0x123456789abcdef0u64;

    for slot in &mut act {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (state >> 32) as f32 / 4294967296.0;
        *slot = (u - 0.5) * 2.0; // Uniform in [-1.0, 1.0]
    }
    // Activation also has independent outlier peaks (columns 128, 800)
    act[128] = 30.0;
    act[800] = -25.0;
    act
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

fn relative_l2_error(exact: &[f32], approx: &[f32]) -> f32 {
    let mut diff_sq = 0.0f32;
    let mut ref_sq = 0.0f32;
    for (&e, &a) in exact.iter().zip(approx.iter()) {
        let d = e - a;
        diff_sq += d * d;
        ref_sq += e * e;
    }
    diff_sq.sqrt() / ref_sq.sqrt().max(1e-12)
}

#[test]
fn test_static_matrix_quantization_audit() {
    let rows = 256usize;
    let cols = 2048usize;
    let total_weights = rows * cols; // 524,288 weights

    println!("====================================================================");
    println!(
        " DETERMINISTIC STATIC QUANTIZATION AUDIT ({} x {})",
        rows, cols
    );
    println!(" Total evaluated real weights: {}", total_weights);
    println!("====================================================================");

    let weights = generate_deterministic_static_matrix(rows, cols);
    let x_act = generate_deterministic_activation(cols);
    let versor = DyadicWalshVersor::new(cols, 42);

    // 1. Exact unquantized FP32 reference MatVec
    let t0 = Instant::now();
    let mut y_exact = vec![0.0f32; rows];
    for r in 0..rows {
        let row_slice = &weights[r * cols..(r + 1) * cols];
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += row_slice[c] * x_act[c];
        }
        y_exact[r] = sum;
    }
    std::hint::black_box(&y_exact);
    let elapsed_exact = t0.elapsed();
    println!(" FP32 Exact Reference: elapsed = {:?}", elapsed_exact);

    // Correlated activation (x_corr = linear combination of W rows + noise)
    let mut x_corr = vec![0.0f32; cols];
    for c in 0..cols {
        x_corr[c] = weights[10 * cols + c] * 0.4 + weights[25 * cols + c] * 0.3 + x_act[c] * 0.1;
    }
    let mut y_exact_corr = vec![0.0f32; rows];
    for r in 0..rows {
        let row_slice = &weights[r * cols..(r + 1) * cols];
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += row_slice[c] * x_corr[c];
        }
        y_exact_corr[r] = sum;
    }

    // 2. Test Direct 2-bit Unrotated (BitNet Style)
    let mut y_2bit_unrot = vec![0.0f32; rows];
    let mut recon_2bit_unrot = vec![0.0f32; cols];
    let mut cos_rows_2bit_unrot = 0.0f32;
    for r in 0..rows {
        let row_slice = &weights[r * cols..(r + 1) * cols];
        let max_val = row_slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_val / 1.5;
        for c in 0..cols {
            let q = (row_slice[c] / scale).round().clamp(-1.0, 1.0);
            recon_2bit_unrot[c] = q * scale;
        }
        cos_rows_2bit_unrot += cosine_similarity(row_slice, &recon_2bit_unrot);
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += recon_2bit_unrot[c] * x_act[c];
        }
        y_2bit_unrot[r] = sum;
    }
    cos_rows_2bit_unrot /= rows as f32;
    let cos_matvec_2bit_unrot = cosine_similarity(&y_exact, &y_2bit_unrot);
    let rel_l2_matvec_2bit_unrot = relative_l2_error(&y_exact, &y_2bit_unrot);

    // 3. Packed 3-bit Clifford Tensor (AOT Dyadic Walsh rotation + Least-Squares BF16 scales)
    let t_compile_3bit = Instant::now();
    let tensor_3bit = Clifford3BitTensor::compile_aot(&weights, rows, cols, &versor);
    let _time_compile_3bit = t_compile_3bit.elapsed();

    let t_matvec_3bit = Instant::now();
    let mut y_3bit = vec![0.0f32; rows];
    tensor_3bit.matvec(&x_act, &mut y_3bit, &versor);
    std::hint::black_box(&y_3bit);
    let time_matvec_3bit = t_matvec_3bit.elapsed();
    let cos_matvec_3bit = cosine_similarity(&y_exact, &y_3bit);
    let rel_l2_matvec_3bit = relative_l2_error(&y_exact, &y_3bit);

    // Check average row weight reconstruction cosine & relative L2 error
    let mut avg_cos_row_3bit = 0.0f32;
    let mut avg_rel_l2_row_3bit = 0.0f32;
    let mut row_buf = vec![0.0f32; cols];
    let lines_per_row = cols / 128;
    for r in 0..rows {
        for l in 0..lines_per_row {
            let chunk: &mut [f32; 128] = (&mut row_buf[l * 128..(l + 1) * 128]).try_into().unwrap();
            tensor_3bit.cache_lines[r * lines_per_row + l].unpack_128(chunk);
        }
        versor.invert_in_place(&mut row_buf);
        let row_orig = &weights[r * cols..(r + 1) * cols];
        avg_cos_row_3bit += cosine_similarity(row_orig, &row_buf);
        avg_rel_l2_row_3bit += relative_l2_error(row_orig, &row_buf);
    }
    avg_cos_row_3bit /= rows as f32;
    avg_rel_l2_row_3bit /= rows as f32;

    // 4. Packed TurboQuant Tensor (3-bit Base + 1-bit QJL)
    let tensor_turbo = TurboQuantTensor::compile_aot(&weights, rows, cols, &versor);
    let t_matvec_turbo = Instant::now();
    let mut y_turbo = vec![0.0f32; rows];
    tensor_turbo.matvec(&x_act, &mut y_turbo, &versor);
    std::hint::black_box(&y_turbo);
    let time_matvec_turbo = t_matvec_turbo.elapsed();
    let cos_matvec_turbo = cosine_similarity(&y_exact, &y_turbo);
    let rel_l2_matvec_turbo = relative_l2_error(&y_exact, &y_turbo);

    let mut y_3bit_corr = vec![0.0f32; rows];
    tensor_3bit.matvec(&x_corr, &mut y_3bit_corr, &versor);
    let cos_matvec_3bit_corr = cosine_similarity(&y_exact_corr, &y_3bit_corr);

    let mut y_turbo_corr = vec![0.0f32; rows];
    tensor_turbo.matvec(&x_corr, &mut y_turbo_corr, &versor);
    let cos_matvec_turbo_corr = cosine_similarity(&y_exact_corr, &y_turbo_corr);
    println!("y_exact[0..5]: {:?}", &y_exact[0..5]);
    println!("y_3bit[0..5]:  {:?}", &y_3bit[0..5]);
    println!("y_turbo[0..5]: {:?}", &y_turbo[0..5]);

    // DETAILED ROW 0 DIAGNOSTICS:
    let w0 = &weights[0..cols];
    let dot_exact: f32 = w0.iter().zip(x_act.iter()).map(|(&a, &b)| a * b).sum();

    let mut w0_rot = w0.to_vec();
    versor.apply_in_place(&mut w0_rot);
    let mut x_rot = x_act.to_vec();
    versor.apply_in_place(&mut x_rot);
    let dot_rot_exact: f32 = w0_rot.iter().zip(x_rot.iter()).map(|(&a, &b)| a * b).sum();

    let mut w0_quant_rot = vec![0.0f32; cols];
    for l in 0..lines_per_row {
        let chunk: &mut [f32; 128] = (&mut w0_quant_rot[l * 128..(l + 1) * 128])
            .try_into()
            .unwrap();
        tensor_3bit.cache_lines[l].unpack_128(chunk);
    }
    let dot_quant_rot: f32 = w0_quant_rot
        .iter()
        .zip(x_rot.iter())
        .map(|(&a, &b)| a * b)
        .sum();

    let mut w0_recon_orig = w0_quant_rot.clone();
    versor.invert_in_place(&mut w0_recon_orig);
    let dot_recon_orig: f32 = w0_recon_orig
        .iter()
        .zip(x_act.iter())
        .map(|(&a, &b)| a * b)
        .sum();

    println!("--------------------------------------------------------------------");
    println!(" ROW 0 DIAGNOSTICS:");
    println!("  - dot_exact (w0 . x)                : {:.6}", dot_exact);
    println!(
        "  - dot_rot_exact (R(w0) . R(x))      : {:.6}",
        dot_rot_exact
    );
    println!(
        "  - dot_quant_rot (Q(R(w0)) . R(x))   : {:.6}",
        dot_quant_rot
    );
    println!(
        "  - dot_recon_orig (R^-1(Q) . x)      : {:.6}",
        dot_recon_orig
    );
    println!(
        "  - Rotation isometry abs error       : {:.2e}",
        (dot_exact - dot_rot_exact).abs()
    );
    println!(
        "  - 3-bit quantization abs error      : {:.6}",
        (dot_exact - dot_quant_rot).abs()
    );
    println!("--------------------------------------------------------------------");

    let mut avg_cos_row_turbo = 0.0f32;
    let mut avg_rel_l2_row_turbo = 0.0f32;
    let turbo_lines_per_row = cols / 64;
    for r in 0..rows {
        for l in 0..turbo_lines_per_row {
            let (slice0, slice1) = row_buf[l * 64..(l + 1) * 64].split_at_mut(32);
            let chunk0: &mut [f32; 32] = slice0.try_into().unwrap();
            let chunk1: &mut [f32; 32] = slice1.try_into().unwrap();
            tensor_turbo.cache_lines[r * turbo_lines_per_row + l].blocks[0].unpack(chunk0);
            tensor_turbo.cache_lines[r * turbo_lines_per_row + l].blocks[1].unpack(chunk1);
        }
        versor.invert_in_place(&mut row_buf);
        let row_orig = &weights[r * cols..(r + 1) * cols];
        avg_cos_row_turbo += cosine_similarity(row_orig, &row_buf);
        avg_rel_l2_row_turbo += relative_l2_error(row_orig, &row_buf);
    }
    avg_cos_row_turbo /= rows as f32;
    avg_rel_l2_row_turbo /= rows as f32;

    println!("--------------------------------------------------------------------");
    println!(" DETERMINISTIC RESULTS ({} Weights):", total_weights);
    println!("--------------------------------------------------------------------");
    println!(" 2-Bit Direct Unrotated (BitNet):");
    println!(
        "   - Mean Weight Cosine            : {:.5}",
        cos_rows_2bit_unrot
    );
    println!(
        "   - MatVec Output Cosine          : {:.5}",
        cos_matvec_2bit_unrot
    );
    println!(
        "   - MatVec Relative L2 Error      : {:.5}",
        rel_l2_matvec_2bit_unrot
    );
    println!(" 3-Bit Clifford Lloyd-Max (Packed LS Tensor):");
    println!(
        "   - Mean Weight Cosine            : {:.5}",
        avg_cos_row_3bit
    );
    println!(
        "   - Mean Weight Rel L2 Error      : {:.5}",
        avg_rel_l2_row_3bit
    );
    println!(
        "   - MatVec Output Cosine          : {:.5}",
        cos_matvec_3bit
    );
    println!(
        "   - MatVec Relative L2 Error      : {:.5}",
        rel_l2_matvec_3bit
    );
    println!(
        "   - Correlated MatVec Cosine      : {:.5}",
        cos_matvec_3bit_corr
    );
    println!(
        "   - MatVec Latency                : {:?}",
        time_matvec_3bit
    );
    println!(" TurboQuant (3-bit Base + 1-bit QJL):");
    println!(
        "   - Mean Weight Cosine            : {:.5}",
        avg_cos_row_turbo
    );
    println!(
        "   - Mean Weight Rel L2 Error      : {:.5}",
        avg_rel_l2_row_turbo
    );
    println!(
        "   - MatVec Output Cosine          : {:.5}",
        cos_matvec_turbo
    );
    println!(
        "   - MatVec Relative L2 Error      : {:.5}",
        rel_l2_matvec_turbo
    );
    println!(
        "   - Correlated MatVec Cosine      : {:.5}",
        cos_matvec_turbo_corr
    );
    println!(
        "   - MatVec Latency                : {:?}",
        time_matvec_turbo
    );
    println!("--------------------------------------------------------------------");

    // 5. MATHEMATICAL ABLATIONS & CALIBRATED LLOYD-MAX EVALUATION:
    // - Compare true analytical RMS-scaled BF16 packing (Clifford3BitWord64::pack)
    //   against orthogonal least-squares BF16 packing (Clifford3BitTensor / pack_least_squares)
    // - Fit layer-specific centroids/bounds with compute_symmetric_lloyd_max_3bit
    //   and evaluate pack_least_squares_with_codebook / unpack_with_codebook
    // - Evaluate TurboQuant Joint 2x2 Gram Projection (FP32 prototype)
    println!(" EVALUATING ALGORITHMIC IMPROVEMENTS...");

    let mut rms_cos_row_3bit = 0.0f32;
    let mut rms_rel_l2_row_3bit = 0.0f32;
    let mut ls_cos_row_3bit = 0.0f32;
    let mut ls_rel_l2_row_3bit = 0.0f32;
    let mut gram_cos_row_turbo = 0.0f32;
    let mut gram_rel_l2_row_turbo = 0.0f32;

    let mut rotated_abs_samples = Vec::with_capacity(64 * cols);
    for r in 0..64.min(rows) {
        let mut row_rot = weights[r * cols..(r + 1) * cols].to_vec();
        versor.apply_in_place(&mut row_rot);
        for chunk in row_rot.chunks_exact(16) {
            let std_dev = (chunk.iter().map(|&v| v * v).sum::<f32>() / 16.0)
                .sqrt()
                .max(1e-8);
            for &v in chunk {
                rotated_abs_samples.push((v / std_dev).abs());
            }
        }
    }
    let (emp_centroids, emp_bounds) =
        compute_symmetric_lloyd_max_3bit(&rotated_abs_samples, 30, 1e-5);
    let mut custom_cb_cos_row_3bit = 0.0f32;
    let mut custom_cb_rel_l2_row_3bit = 0.0f32;

    for r in 0..rows {
        let row_orig = &weights[r * cols..(r + 1) * cols];
        let mut row_rot = row_orig.to_vec();
        versor.apply_in_place(&mut row_rot);

        // --- Baseline A0: True Analytical RMS BF16 Packing via Clifford3BitWord64::pack ---
        let mut row_recon_rms = vec![0.0f32; cols];
        // --- Calibrated Codebook LS BF16 Packing via pack_least_squares_with_codebook ---
        let mut row_recon_cb = vec![0.0f32; cols];
        // --- FP32 Unquantized-Scale LS Upper Bound Reference ---
        let mut row_recon_ls = vec![0.0f32; cols];

        for b in 0..(cols / 16) {
            let chunk_slice = &row_rot[b * 16..(b + 1) * 16];
            let chunk: &[f32; 16] = chunk_slice.try_into().unwrap();
            let std_dev = (chunk.iter().map(|&v| v * v).sum::<f32>() / 16.0)
                .sqrt()
                .max(1e-8);

            let word_rms = Clifford3BitWord64::pack(chunk, std_dev);
            let mut out_rms = [0.0f32; 16];
            word_rms.unpack(&mut out_rms);
            row_recon_rms[b * 16..(b + 1) * 16].copy_from_slice(&out_rms);

            let word_cb = Clifford3BitWord64::pack_least_squares_with_codebook(
                chunk,
                &emp_centroids,
                &emp_bounds,
            );
            let mut out_cb = [0.0f32; 16];
            word_cb.unpack_with_codebook(&mut out_cb, &emp_centroids);
            row_recon_cb[b * 16..(b + 1) * 16].copy_from_slice(&out_cb);

            let inv_std = 1.0 / std_dev;
            let mut c_norm = [0.0f32; 16];
            for i in 0..16 {
                let norm = chunk[i] * inv_std;
                let sign = if norm < 0.0 { -1.0f32 } else { 1.0f32 };
                let abs_val = norm.abs();
                let mag = if abs_val < LLOYD_MAX_3BIT_BOUNDS[0] {
                    LLOYD_MAX_3BIT_CENTROIDS[0]
                } else if abs_val < LLOYD_MAX_3BIT_BOUNDS[1] {
                    LLOYD_MAX_3BIT_CENTROIDS[1]
                } else if abs_val < LLOYD_MAX_3BIT_BOUNDS[2] {
                    LLOYD_MAX_3BIT_CENTROIDS[2]
                } else {
                    LLOYD_MAX_3BIT_CENTROIDS[3]
                };
                c_norm[i] = sign * mag;
            }
            let dot_wc: f32 = chunk.iter().zip(c_norm.iter()).map(|(&w, &c)| w * c).sum();
            let norm_c_sq: f32 = c_norm.iter().map(|&c| c * c).sum();
            let alpha_ls = if norm_c_sq > 1e-12 {
                dot_wc / norm_c_sq
            } else {
                std_dev
            };
            for i in 0..16 {
                row_recon_ls[b * 16 + i] = alpha_ls * c_norm[i];
            }
        }
        versor.invert_in_place(&mut row_recon_rms);
        rms_cos_row_3bit += cosine_similarity(row_orig, &row_recon_rms);
        rms_rel_l2_row_3bit += relative_l2_error(row_orig, &row_recon_rms);

        versor.invert_in_place(&mut row_recon_cb);
        custom_cb_cos_row_3bit += cosine_similarity(row_orig, &row_recon_cb);
        custom_cb_rel_l2_row_3bit += relative_l2_error(row_orig, &row_recon_cb);

        versor.invert_in_place(&mut row_recon_ls);
        ls_cos_row_3bit += cosine_similarity(row_orig, &row_recon_ls);
        ls_rel_l2_row_3bit += relative_l2_error(row_orig, &row_recon_ls);

        // --- Improvement C: TurboQuant with Joint 2x2 Gram Projection (32-weight blocks) ---
        let mut row_recon_gram = vec![0.0f32; cols];
        for b in 0..(cols / 32) {
            let chunk = &row_rot[b * 32..(b + 1) * 32];
            let std_dev = (chunk.iter().map(|&v| v * v).sum::<f32>() / 32.0)
                .sqrt()
                .max(1e-8);
            let inv_std = 1.0 / std_dev;

            let mut u_base = [0.0f32; 32];
            let mut v_sign = [0.0f32; 32];

            for i in 0..32 {
                let norm = chunk[i] * inv_std;
                let sign = if norm < 0.0 { -1.0f32 } else { 1.0f32 };
                let abs_val = norm.abs();
                let mag = if abs_val < LLOYD_MAX_3BIT_BOUNDS[0] {
                    LLOYD_MAX_3BIT_CENTROIDS[0]
                } else if abs_val < LLOYD_MAX_3BIT_BOUNDS[1] {
                    LLOYD_MAX_3BIT_CENTROIDS[1]
                } else if abs_val < LLOYD_MAX_3BIT_BOUNDS[2] {
                    LLOYD_MAX_3BIT_CENTROIDS[2]
                } else {
                    LLOYD_MAX_3BIT_CENTROIDS[3]
                };
                let base_val = sign * mag * std_dev;
                u_base[i] = base_val;

                let res = chunk[i] - base_val;
                v_sign[i] = if res >= 0.0 { 1.0f32 } else { -1.0f32 };
            }

            // 2x2 Gram System:
            // [ <u, u>   <u, v> ] [ a1 ] = [ <w, u> ]
            // [ <u, v>   <v, v> ] [ a2 ]   [ <w, v> ]
            let g11: f32 = u_base.iter().map(|&u| u * u).sum();
            let g12: f32 = u_base.iter().zip(v_sign.iter()).map(|(&u, &v)| u * v).sum();
            let g22: f32 = v_sign.iter().map(|&v| v * v).sum(); // = 32.0

            let b1: f32 = chunk.iter().zip(u_base.iter()).map(|(&w, &u)| w * u).sum();
            let b2: f32 = chunk.iter().zip(v_sign.iter()).map(|(&w, &v)| w * v).sum();

            let det = g11 * g22 - g12 * g12;
            let (a1, a2) = if det.abs() > 1e-12 {
                let a1 = (b1 * g22 - b2 * g12) / det;
                let a2 = (b2 * g11 - b1 * g12) / det;
                (a1, a2)
            } else {
                (1.0, 0.0)
            };

            for i in 0..32 {
                row_recon_gram[b * 32 + i] = a1 * u_base[i] + a2 * v_sign[i];
            }
        }
        versor.invert_in_place(&mut row_recon_gram);
        gram_cos_row_turbo += cosine_similarity(row_orig, &row_recon_gram);
        gram_rel_l2_row_turbo += relative_l2_error(row_orig, &row_recon_gram);
    }

    rms_cos_row_3bit /= rows as f32;
    rms_rel_l2_row_3bit /= rows as f32;
    custom_cb_cos_row_3bit /= rows as f32;
    custom_cb_rel_l2_row_3bit /= rows as f32;
    ls_cos_row_3bit /= rows as f32;
    ls_rel_l2_row_3bit /= rows as f32;
    gram_cos_row_turbo /= rows as f32;
    gram_rel_l2_row_turbo /= rows as f32;

    println!("--------------------------------------------------------------------");
    println!(" EMPIRICAL LLOYD-MAX CALIBRATION vs N(0, 1) CONSTANTS:");
    println!("   Theoretical centroids : {:?}", LLOYD_MAX_3BIT_CENTROIDS);
    println!("   Empirical centroids   : {:?}", emp_centroids);
    println!("   Theoretical bounds    : {:?}", LLOYD_MAX_3BIT_BOUNDS);
    println!("   Empirical bounds      : {:?}", emp_bounds);
    println!("--------------------------------------------------------------------");
    println!(" COMPARATIVE RESULTS AFTER MATHEMATICAL OPTIMIZATION:");
    println!("--------------------------------------------------------------------");
    println!(
        " 3-Bit Clifford RMS (Word64::pack)    : Cosine = {:.5} | Rel L2 = {:.5}",
        rms_cos_row_3bit, rms_rel_l2_row_3bit
    );
    println!(
        " 3-Bit Clifford LS (Packed Tensor)    : Cosine = {:.5} | Rel L2 = {:.5} (Gain vs RMS: {:+.5})",
        avg_cos_row_3bit,
        avg_rel_l2_row_3bit,
        avg_cos_row_3bit - rms_cos_row_3bit
    );
    println!(
        " 3-Bit Clifford LS (Custom Codebook)  : Cosine = {:.5} | Rel L2 = {:.5}",
        custom_cb_cos_row_3bit, custom_cb_rel_l2_row_3bit
    );
    println!(
        " 3-Bit Clifford LS (FP32 Scale Ref)   : Cosine = {:.5} | Rel L2 = {:.5}",
        ls_cos_row_3bit, ls_rel_l2_row_3bit
    );
    println!(
        " TurboQuant (Packed Tensor)           : Cosine = {:.5} | Rel L2 = {:.5}",
        avg_cos_row_turbo, avg_rel_l2_row_turbo
    );
    println!(
        " TurboQuant + 2x2 Gram (FP32 Proto)   : Cosine = {:.5} | Rel L2 = {:.5} (Gain: {:+.5})",
        gram_cos_row_turbo,
        gram_rel_l2_row_turbo,
        gram_cos_row_turbo - avg_cos_row_turbo
    );
    println!("--------------------------------------------------------------------");

    assert!(
        avg_cos_row_3bit >= rms_cos_row_3bit && avg_rel_l2_row_3bit <= rms_rel_l2_row_3bit,
        "Least-squares scale must strictly improve or match analytical RMS baseline"
    );
    assert!(
        custom_cb_cos_row_3bit > 0.96,
        "Calibrated Lloyd-Max custom codebook reconstruction must exceed 0.96 cosine"
    );
    assert!(
        gram_cos_row_turbo >= avg_cos_row_turbo,
        "Gram 2x2 projection must strictly improve or equal packed baseline"
    );
    assert!(
        gram_cos_row_turbo > 0.99,
        "Optimized TurboQuant must achieve > 0.99 cosine on 524,288 weights"
    );
}
