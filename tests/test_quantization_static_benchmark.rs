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

use versorc_math::prelude::*;
use std::time::Instant;

/// Generates a bit-identical, deterministic static matrix of size rows x cols.
/// Uses a 64-bit PCG state with Box-Muller transform to produce exact Gaussian distribution
/// N(0, sigma^2) with sigma = 0.025, plus 0.2% massive outlier channels (value = 45.0).
fn generate_deterministic_static_matrix(rows: usize, cols: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; rows * cols];
    let mut state = 0x853c49e6748fea9bu64; // Fixed deterministic seed

    let mut next_u32 = || -> u32 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

    // Inject fixed, realistic outlier channels across all rows (columns 42, 350, 1020, 1800)
    let outlier_cols = [42usize, 350, 1020, 1800];
    let outlier_vals = [45.0f32, -38.5, 52.0, -41.2];
    for r in 0..rows {
        for (&col, &val) in outlier_cols.iter().zip(outlier_vals.iter()) {
            data[r * cols + col] = val;
        }
    }

    data
}

/// Generates a bit-identical deterministic activation vector.
fn generate_deterministic_activation(cols: usize) -> Vec<f32> {
    let mut act = vec![0.0f32; cols];
    let mut state = 0x123456789abcdef0u64;

    for i in 0..cols {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (state >> 32) as f32 / 4294967296.0;
        act[i] = (u - 0.5) * 2.0; // Uniform in [-1.0, 1.0]
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

#[test]
fn test_static_matrix_quantization_audit() {
    let rows = 256usize;
    let cols = 2048usize;
    let total_weights = rows * cols; // 524,288 weights

    println!("====================================================================");
    println!(" AUDITORÍA DE CUANTIZACIÓN DETERMINISTA ESTÁTICA ({} x {})", rows, cols);
    println!(" Total de pesos reales evaluados: {}", total_weights);
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
    let elapsed_exact = t0.elapsed();
    println!(" FP32 Referencia Exacta: tiempo = {:?}", elapsed_exact);

    // Activación correlacionada (x_corr = combinación de filas de W + ruido)
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

    // 3. Current 3-bit Clifford Implementation (Baseline Analytical RMS Scale)
    let t_compile_3bit = Instant::now();
    let tensor_3bit = Clifford3BitTensor::compile_aot(&weights, rows, cols, &versor);
    let _time_compile_3bit = t_compile_3bit.elapsed();

    let t_matvec_3bit = Instant::now();
    let mut y_3bit = vec![0.0f32; rows];
    tensor_3bit.matvec(&x_act, &mut y_3bit, &versor);
    let time_matvec_3bit = t_matvec_3bit.elapsed();
    let cos_matvec_3bit = cosine_similarity(&y_exact, &y_3bit);

    // Check average row weight reconstruction cosine
    let mut avg_cos_row_3bit = 0.0f32;
    let mut row_buf = vec![0.0f32; cols];
    let lines_per_row = cols / 128;
    for r in 0..rows {
        for l in 0..lines_per_row {
            let chunk: &mut [f32; 128] = (&mut row_buf[l * 128..(l + 1) * 128]).try_into().unwrap();
            tensor_3bit.cache_lines[r * lines_per_row + l].unpack_128(chunk);
        }
        versor.invert_in_place(&mut row_buf);
        avg_cos_row_3bit += cosine_similarity(&weights[r * cols..(r + 1) * cols], &row_buf);
    }
    avg_cos_row_3bit /= rows as f32;

    // 4. Current TurboQuant Implementation
    let tensor_turbo = TurboQuantTensor::compile_aot(&weights, rows, cols, &versor);
    let t_matvec_turbo = Instant::now();
    let mut y_turbo = vec![0.0f32; rows];
    tensor_turbo.matvec(&x_act, &mut y_turbo, &versor);
    let time_matvec_turbo = t_matvec_turbo.elapsed();
    let cos_matvec_turbo = cosine_similarity(&y_exact, &y_turbo);

    let mut y_3bit_corr = vec![0.0f32; rows];
    tensor_3bit.matvec(&x_corr, &mut y_3bit_corr, &versor);
    let cos_matvec_3bit_corr = cosine_similarity(&y_exact_corr, &y_3bit_corr);

    let mut y_turbo_corr = vec![0.0f32; rows];
    tensor_turbo.matvec(&x_corr, &mut y_turbo_corr, &versor);
    let cos_matvec_turbo_corr = cosine_similarity(&y_exact_corr, &y_turbo_corr);
    println!("y_exact[0..5]: {:?}", &y_exact[0..5]);
    println!("y_3bit[0..5]:  {:?}", &y_3bit[0..5]);
    println!("y_turbo[0..5]: {:?}", &y_turbo[0..5]);

    // DIAGNÓSTICO DETALLADO FILA 0:
    let w0 = &weights[0..cols];
    let dot_exact: f32 = w0.iter().zip(x_act.iter()).map(|(&a, &b)| a * b).sum();

    let mut w0_rot = w0.to_vec();
    versor.apply_in_place(&mut w0_rot);
    let mut x_rot = x_act.to_vec();
    versor.apply_in_place(&mut x_rot);
    let dot_rot_exact: f32 = w0_rot.iter().zip(x_rot.iter()).map(|(&a, &b)| a * b).sum();

    let mut w0_quant_rot = vec![0.0f32; cols];
    for l in 0..lines_per_row {
        let chunk: &mut [f32; 128] = (&mut w0_quant_rot[l * 128..(l + 1) * 128]).try_into().unwrap();
        tensor_3bit.cache_lines[l].unpack_128(chunk);
    }
    let dot_quant_rot: f32 = w0_quant_rot.iter().zip(x_rot.iter()).map(|(&a, &b)| a * b).sum();

    let mut w0_recon_orig = w0_quant_rot.clone();
    versor.invert_in_place(&mut w0_recon_orig);
    let dot_recon_orig: f32 = w0_recon_orig.iter().zip(x_act.iter()).map(|(&a, &b)| a * b).sum();

    println!("--------------------------------------------------------------------");
    println!(" DIAGNÓSTICO FILA 0:");
    println!("  - dot_exact (w0 . x)                : {:.6}", dot_exact);
    println!("  - dot_rot_exact (R(w0) . R(x))      : {:.6}", dot_rot_exact);
    println!("  - dot_quant_rot (Q(R(w0)) . R(x))   : {:.6}", dot_quant_rot);
    println!("  - dot_recon_orig (R^-1(Q) . x)      : {:.6}", dot_recon_orig);
    println!("  - Error absoluto rotación isometría : {:.2e}", (dot_exact - dot_rot_exact).abs());
    println!("  - Error por cuantización 3-bit      : {:.6}", (dot_exact - dot_quant_rot).abs());
    println!("--------------------------------------------------------------------");

    let mut avg_cos_row_turbo = 0.0f32;
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
        avg_cos_row_turbo += cosine_similarity(&weights[r * cols..(r + 1) * cols], &row_buf);
    }
    avg_cos_row_turbo /= rows as f32;

    println!("--------------------------------------------------------------------");
    println!(" RESULTADOS DETERMINISTAS (Sin Mock Data, 524,288 Pesos):");
    println!("--------------------------------------------------------------------");
    println!(" 2-Bit Directo Sin Rotar (BitNet):");
    println!("   - Cosine Promedio Pesos (w)     : {:.5}", cos_rows_2bit_unrot);
    println!("   - Cosine Salida MatVec (y)      : {:.5}", cos_matvec_2bit_unrot);
    println!(" 3-Bit Clifford Lloyd-Max:");
    println!("   - Cosine Promedio Pesos (w)     : {:.5}", avg_cos_row_3bit);
    println!("   - Cosine Salida MatVec (y)      : {:.5}", cos_matvec_3bit);
    println!("   - Cosine MatVec Correlacionada  : {:.5}", cos_matvec_3bit_corr);
    println!("   - Tiempo MatVec                 : {:?}", time_matvec_3bit);
    println!(" TurboQuant (3-bit Base + 1-bit QJL):");
    println!("   - Cosine Promedio Pesos (w)     : {:.5}", avg_cos_row_turbo);
    println!("   - Cosine Salida MatVec (y)      : {:.5}", cos_matvec_turbo);
    println!("   - Correlated MatVec Cosine      : {:.5}", cos_matvec_turbo_corr);
    println!("   - MatVec Latency                : {:?}", time_matvec_turbo);
    println!("--------------------------------------------------------------------");

    // 5. EVALUATION OF ENHANCEMENT HYPOTHESES:
    // Improvement A: Optimal Least-Squares Projection Scale: alpha* = <w, c> / ||c||^2
    // Improvement B: Exact MAE Residual Scale: mean(|r|)
    // Improvement C: Joint 2x2 Gram Projection in TurboQuant:
    //   min ||w - (alpha_1 * base + alpha_2 * res_sign)||^2
    println!(" EVALUATING ALGORITHMIC IMPROVEMENTS...");

    let mut ls_cos_row_3bit = 0.0f32;
    let mut gram_cos_row_turbo = 0.0f32;

    for r in 0..rows {
        let mut row_rot = weights[r * cols..(r + 1) * cols].to_vec();
        versor.apply_in_place(&mut row_rot);

        // --- Improvement A: 3-bit with Least-Squares Scale (16-weight blocks) ---
        let mut row_recon_ls = vec![0.0f32; cols];
        for b in 0..(cols / 16) {
            let chunk = &row_rot[b * 16..(b + 1) * 16];
            let std_dev = (chunk.iter().map(|&v| v * v).sum::<f32>() / 16.0).sqrt().max(1e-8);
            let inv_std = 1.0 / std_dev;

            // Extract normalized centroid codes
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

            // Exact least-squares scale: alpha* = <w, c_norm> / ||c_norm||^2
            let dot_wc: f32 = chunk.iter().zip(c_norm.iter()).map(|(&w, &c)| w * c).sum();
            let norm_c_sq: f32 = c_norm.iter().map(|&c| c * c).sum();
            let alpha_ls = if norm_c_sq > 1e-12 { dot_wc / norm_c_sq } else { std_dev };

            for i in 0..16 {
                row_recon_ls[b * 16 + i] = alpha_ls * c_norm[i];
            }
        }
        versor.invert_in_place(&mut row_recon_ls);
        ls_cos_row_3bit += cosine_similarity(&weights[r * cols..(r + 1) * cols], &row_recon_ls);

        // --- Improvement C: TurboQuant with Joint 2x2 Gram Projection (32-weight blocks) ---
        let mut row_recon_gram = vec![0.0f32; cols];
        for b in 0..(cols / 32) {
            let chunk = &row_rot[b * 32..(b + 1) * 32];
            let std_dev = (chunk.iter().map(|&v| v * v).sum::<f32>() / 32.0).sqrt().max(1e-8);
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
        gram_cos_row_turbo += cosine_similarity(&weights[r * cols..(r + 1) * cols], &row_recon_gram);
    }

    ls_cos_row_3bit /= rows as f32;
    gram_cos_row_turbo /= rows as f32;

    println!("--------------------------------------------------------------------");
    println!(" COMPARATIVE RESULTS AFTER MATHEMATICAL OPTIMIZATION:");
    println!("--------------------------------------------------------------------");
    println!(" 3-Bit Clifford Base (RMS Scale)      : Cosine = {:.5}", avg_cos_row_3bit);
    println!(" 3-Bit Clifford + Least-Squares Scale : Cosine = {:.5}  (Gain: +{:.4})",
             ls_cos_row_3bit, ls_cos_row_3bit - avg_cos_row_3bit);
    println!(" TurboQuant Base                      : Cosine = {:.5}", avg_cos_row_turbo);
    println!(" TurboQuant + 2x2 Gram Projection    : Cosine = {:.5}  (Gain: +{:.4})",
             gram_cos_row_turbo, gram_cos_row_turbo - avg_cos_row_turbo);
    println!("--------------------------------------------------------------------");

    assert!(ls_cos_row_3bit >= avg_cos_row_3bit, "Least-squares scale must strictly improve or equal baseline");
    assert!(gram_cos_row_turbo >= avg_cos_row_turbo, "Gram 2x2 projection must strictly improve or equal baseline");
    assert!(gram_cos_row_turbo > 0.995, "Optimized TurboQuant must achieve > 0.995 on 524,288 weights");
}
