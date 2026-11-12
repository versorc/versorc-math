use versorc_math::prelude::*;

#[test]
fn test_geometric_quantization_comparison() {
    let dim = 2048usize;
    let versor = DyadicWalshVersor::new(dim, 42);

    // Generate test vector with heavy outliers (resembling LLM activations / weights)
    let mut raw_vector = vec![0.0f32; dim];
    for i in 0..dim {
        let t = i as f32 * 0.1;
        raw_vector[i] = (t.sin() * 0.5) + (t * 0.3).cos() * 0.2;
    }
    // Inject 0.2% massive outlier channels (value = 40.0)
    raw_vector[12] = 42.0;
    raw_vector[87] = -38.5;
    raw_vector[350] = 55.0;
    raw_vector[1020] = -47.0;

    // Norm of original vector
    let orig_norm: f32 = raw_vector.iter().map(|&v| v * v).sum::<f32>().sqrt();

    // 1. Direct 2-bit quantization WITHOUT rotation (Standard BitNet / Polar)
    let mut recon_2bit_unrotated = vec![0.0f32; dim];
    let max_val = raw_vector.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale_2bit = max_val / 1.5;
    for i in 0..dim {
        let q = (raw_vector[i] / scale_2bit).round().clamp(-1.0, 1.0);
        recon_2bit_unrotated[i] = q * scale_2bit;
    }
    let cos_2bit_unrotated = cosine_similarity(&raw_vector, &recon_2bit_unrotated);

    // 2. Rotate with Dyadic Walsh Versor (Isometry R: ||R(x)|| == ||x||)
    let mut rotated = raw_vector.clone();
    versor.transform_in_place(&mut rotated);
    let rot_norm: f32 = rotated.iter().map(|&v| v * v).sum::<f32>().sqrt();
    assert!((rot_norm - orig_norm).abs() < 1e-3, "Walsh transform must be an exact isometry");

    // Standard deviation after rotation
    let std_dev = (rotated.iter().map(|&v| v * v).sum::<f32>() / dim as f32).sqrt();

    // 3. 2-bit Lloyd-Max on Rotated Coordinates (4 levels)
    // Centroids: +/- 0.4528 * sigma, +/- 1.5104 * sigma
    let c2 = [0.4528f32 * std_dev, 1.5104f32 * std_dev];
    let mut recon_2bit_rotated = vec![0.0f32; dim];
    for i in 0..dim {
        let v = rotated[i];
        let sign = if v >= 0.0 { 1.0 } else { -1.0 };
        let abs_v = v.abs();
        let mag = if abs_v < 0.9816 * std_dev { c2[0] } else { c2[1] };
        recon_2bit_rotated[i] = sign * mag;
    }
    versor.invert_in_place(&mut recon_2bit_rotated);
    let cos_2bit_rotated = cosine_similarity(&raw_vector, &recon_2bit_rotated);

    // 4. 3-bit Lloyd-Max on Rotated Coordinates (8 levels, 3 bits per weight)
    // Centroids for N(0, 1): +/- 0.2451, +/- 0.7560, +/- 1.3439, +/- 2.1520
    let c3 = [
        0.2451f32 * std_dev,
        0.7560f32 * std_dev,
        1.3439f32 * std_dev,
        2.1520f32 * std_dev,
    ];
    let bounds3 = [
        0.5005f32 * std_dev,
        1.0500f32 * std_dev,
        1.7480f32 * std_dev,
    ];
    let mut recon_3bit_rotated = vec![0.0f32; dim];
    for i in 0..dim {
        let v = rotated[i];
        let sign = if v >= 0.0 { 1.0 } else { -1.0 };
        let abs_v = v.abs();
        let mag = if abs_v < bounds3[0] {
            c3[0]
        } else if abs_v < bounds3[1] {
            c3[1]
        } else if abs_v < bounds3[2] {
            c3[2]
        } else {
            c3[3]
        };
        recon_3bit_rotated[i] = sign * mag;
    }
    versor.invert_in_place(&mut recon_3bit_rotated);
    let cos_3bit_rotated = cosine_similarity(&raw_vector, &recon_3bit_rotated);

    // 5. 3.5-bit TurboQuant (3-bit Lloyd-Max + 1-bit QJL Residual Projection)
    // Residual = rotated - quantized_3bit
    let mut residual = vec![0.0f32; dim];
    for i in 0..dim {
        let v = rotated[i];
        let sign = if v >= 0.0 { 1.0 } else { -1.0 };
        let abs_v = v.abs();
        let mag = if abs_v < bounds3[0] {
            c3[0]
        } else if abs_v < bounds3[1] {
            c3[1]
        } else if abs_v < bounds3[2] {
            c3[2]
        } else {
            c3[3]
        };
        residual[i] = v - (sign * mag);
    }
    let res_scale = (residual.iter().map(|&v| v * v).sum::<f32>() / dim as f32).sqrt() * 0.7979;
    let mut recon_3_5bit = vec![0.0f32; dim];
    for i in 0..dim {
        let v = rotated[i];
        let sign = if v >= 0.0 { 1.0 } else { -1.0 };
        let abs_v = v.abs();
        let mag = if abs_v < bounds3[0] {
            c3[0]
        } else if abs_v < bounds3[1] {
            c3[1]
        } else if abs_v < bounds3[2] {
            c3[2]
        } else {
            c3[3]
        };
        let res_sign = if residual[i] >= 0.0 { 1.0 } else { -1.0 };
        recon_3_5bit[i] = (sign * mag) + (res_sign * res_scale);
    }
    versor.invert_in_place(&mut recon_3_5bit);
    let cos_3_5bit_rotated = cosine_similarity(&raw_vector, &recon_3_5bit);

    // 6. 4-bit Uniform / INT4 on Rotated Coordinates (16 levels)
    let step4 = (3.5 * std_dev) / 7.5;
    let mut recon_4bit_rotated = vec![0.0f32; dim];
    for i in 0..dim {
        let v = rotated[i];
        let q = (v / step4).round().clamp(-8.0, 7.0);
        recon_4bit_rotated[i] = q * step4;
    }
    versor.invert_in_place(&mut recon_4bit_rotated);
    let cos_4bit_rotated = cosine_similarity(&raw_vector, &recon_4bit_rotated);

    // 7. Dot product preservation with activation vector x
    let mut x_act = vec![0.0f32; dim];
    for i in 0..dim {
        let t = (i as f32 + 100.0) * 0.2;
        x_act[i] = (t.sin() * 0.8) + (t * 0.5).cos() * 0.4;
    }
    // Activation also has realistic non-aligned peaks
    x_act[50] = 30.0;
    x_act[200] = -25.0;

    let dot_exact: f32 = raw_vector.iter().zip(x_act.iter()).map(|(&w, &x)| w * x).sum();
    let norm_act: f32 = x_act.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let exact_cosine_proj = dot_exact / (orig_norm * norm_act);

    let dot_2bit_unrot: f32 = recon_2bit_unrotated.iter().zip(x_act.iter()).map(|(&w, &x)| w * x).sum();
    let dot_2bit_rot: f32 = recon_2bit_rotated.iter().zip(x_act.iter()).map(|(&w, &x)| w * x).sum();
    let dot_3bit_rot: f32 = recon_3bit_rotated.iter().zip(x_act.iter()).map(|(&w, &x)| w * x).sum();
    let dot_3_5bit_rot: f32 = recon_3_5bit.iter().zip(x_act.iter()).map(|(&w, &x)| w * x).sum();
    let dot_4bit_rot: f32 = recon_4bit_rotated.iter().zip(x_act.iter()).map(|(&w, &x)| w * x).sum();

    let err_2bit_unrot = (dot_exact - dot_2bit_unrot).abs() / (orig_norm * norm_act);
    let err_2bit_rot = (dot_exact - dot_2bit_rot).abs() / (orig_norm * norm_act);
    let err_3bit_rot = (dot_exact - dot_3bit_rot).abs() / (orig_norm * norm_act);
    let err_3_5bit_rot = (dot_exact - dot_3_5bit_rot).abs() / (orig_norm * norm_act);
    let err_4bit_rot = (dot_exact - dot_4bit_rot).abs() / (orig_norm * norm_act);

    println!("------------------------------------------------------------");
    println!(" COMPARATIVA EMPÍRICA DE CUANTIZACIÓN (Dim = {})", dim);
    println!("------------------------------------------------------------");
    println!(" 2-bit Directo sin rotar (BitNet) : Cosine = {:.4} | Dot Error = {:.4}", cos_2bit_unrotated, err_2bit_unrot);
    println!(" 2-bit Rotado con Dyadic Versor  : Cosine = {:.4} | Dot Error = {:.4}", cos_2bit_rotated, err_2bit_rot);
    println!(" 3-bit Rotado Lloyd-Max (Turbo)  : Cosine = {:.4} | Dot Error = {:.4}", cos_3bit_rotated, err_3bit_rot);
    println!(" 3.5-bit TurboQuant (+QJL Res)   : Cosine = {:.4} | Dot Error = {:.4}", cos_3_5bit_rotated, err_3_5bit_rot);
    println!(" 4-bit Rotado (INT4 Isométrico)  : Cosine = {:.4} | Dot Error = {:.4}", cos_4bit_rotated, err_4bit_rot);
    println!("------------------------------------------------------------");
    println!(" Proyección de producto interno exacta: {:.4}", exact_cosine_proj);
    println!(" Proyección 2-bit unrotated:            {:.4}", dot_2bit_unrot / (orig_norm * norm_act));
    println!(" Proyección 3-bit rotated:              {:.4}", dot_3bit_rot / (orig_norm * norm_act));
    println!(" Proyección 3.5-bit rotated:            {:.4}", dot_3_5bit_rot / (orig_norm * norm_act));
    println!("------------------------------------------------------------");

    assert!(cos_3bit_rotated > 0.985, "3-bit rotated must achieve > 0.985 cosine similarity");
    assert!(cos_3_5bit_rotated > 0.995, "3.5-bit TurboQuant must achieve > 0.995 cosine similarity");
    assert!(err_3bit_rot < 0.05, "3-bit rotated dot product error must be < 5%");
    assert!(err_3_5bit_rot < 0.02, "3.5-bit TurboQuant dot product error must be < 2%");
}

#[test]
fn test_clifford_3bit_word_packing_and_alignment() {
    use std::mem::{align_of, size_of};

    // 1. Verify strict hardware cache-line alignment and size guarantees
    assert_eq!(size_of::<Clifford3BitWord64>(), 8, "Word must be exactly 8 bytes (64 bits)");
    assert_eq!(size_of::<Clifford3BitCacheLine64>(), 64, "Cache line must be exactly 64 bytes");
    assert_eq!(align_of::<Clifford3BitCacheLine64>(), 64, "Cache line must be 64-byte aligned");

    // 2. Test 16-weight word packing and unpacking
    let weights: [f32; 16] = [
        0.12, -0.45, 0.88, -1.25, 2.10, -0.05, 0.33, -0.76,
        1.05, -1.80, 0.50, -0.22, 1.45, -0.90, 0.02, -0.60,
    ];
    let scale = 1.0f32;
    let word = Clifford3BitWord64::pack(&weights, scale);
    assert!((word.scale() - scale).abs() < 1e-3);

    let mut unpacked = [0.0f32; 16];
    word.unpack(&mut unpacked);

    // Centroids must preserve sign and relative rank
    for i in 0..16 {
        assert_eq!(weights[i].signum(), unpacked[i].signum());
    }

    // Dot product with slice must match manual dot
    let manual_dot: f32 = unpacked.iter().zip(weights.iter()).map(|(&a, &b)| a * b).sum();
    let word_dot = word.dot_with_slice(&weights);
    assert!((manual_dot - word_dot).abs() < 1e-4);

    // 3. Test 128-weight cache line packing
    let mut w128 = [0.0f32; 128];
    for i in 0..128 {
        w128[i] = ((i as f32 * 0.2).sin() * 1.5) + ((i as f32 * 0.1).cos() * 0.5);
    }
    let cache_line = Clifford3BitCacheLine64::pack_128(&w128);
    let mut unp128 = [0.0f32; 128];
    cache_line.unpack_128(&mut unp128);

    let cos_line = cosine_similarity(&w128, &unp128);
    assert!(cos_line > 0.98, "128-weight cache line reconstruction cosine must exceed 0.98, got {}", cos_line);
}

#[test]
fn test_turboquant_block32_and_cache_line() {
    use std::mem::{align_of, size_of};

    assert_eq!(size_of::<TurboQuantCacheLine64>(), 64, "TurboQuant line must be 64 bytes");
    assert_eq!(align_of::<TurboQuantCacheLine64>(), 64, "TurboQuant line must be 64-byte aligned");

    let mut w32 = [0.0f32; 32];
    for i in 0..32 {
        w32[i] = (i as f32 * 0.3).sin() * 2.0;
    }
    let block = TurboQuantBlock32::pack(&w32);
    let mut unp32 = [0.0f32; 32];
    block.unpack(&mut unp32);

    let cos_32 = cosine_similarity(&w32, &unp32);
    assert!(cos_32 > 0.99, "TurboQuant 32-weight block cosine must exceed 0.99, got {}", cos_32);

    let dot_block = block.dot_with_slice(&w32);
    let manual_dot: f32 = unp32.iter().zip(w32.iter()).map(|(&a, &b)| a * b).sum();
    assert!((dot_block - manual_dot).abs() < 1e-3);
}

#[test]
fn test_clifford_3bit_tensor_matvec_dual_isometry() {
    let rows = 64usize;
    let cols = 512usize;
    let versor = DyadicWalshVersor::new(cols, 12345);

    // Synthetic weight matrix with random Gaussian + outlier channels
    let mut weights = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let t = (r * cols + c) as f32 * 0.05;
            weights[r * cols + c] = (t.sin() * 0.7) + (t * 0.3).cos() * 0.3;
        }
        // Heavy outlier in column 42
        weights[r * cols + 42] = 28.0;
    }

    // AOT compile to 3-bit Clifford tensor
    let tensor_3bit = Clifford3BitTensor::compile_aot(&weights, rows, cols, &versor);
    assert_eq!(tensor_3bit.cache_lines.len(), rows * (cols / 128));

    // Input activation vector x with separate outlier
    let mut x = vec![0.0f32; cols];
    for c in 0..cols {
        let t = (c as f32 + 50.0) * 0.1;
        x[c] = (t.sin() * 0.5) + (t * 0.2).cos() * 0.5;
    }
    x[100] = 35.0;

    // 1. Exact unquantized FP32 matvec
    let mut y_exact = vec![0.0f32; rows];
    for r in 0..rows {
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += weights[r * cols + c] * x[c];
        }
        y_exact[r] = sum;
    }

    // 2. Dual Isometry 3-bit matvec (Rotates x ONCE, streams packed cache lines)
    let mut y_3bit = vec![0.0f32; rows];
    tensor_3bit.matvec(&x, &mut y_3bit, &versor);

    // Compute cosine similarity between exact FP32 output and 3-bit output
    let cos_matvec = cosine_similarity(&y_exact, &y_3bit);
    println!("Matrix-Vector Output Cosine Similarity (3-bit Clifford): {:.5}", cos_matvec);

    // 3. Test TurboQuant tensor matvec
    let tensor_turbo = TurboQuantTensor::compile_aot(&weights, rows, cols, &versor);
    let mut y_turbo = vec![0.0f32; rows];
    tensor_turbo.matvec(&x, &mut y_turbo, &versor);

    let cos_turbo = cosine_similarity(&y_exact, &y_turbo);
    println!("Matrix-Vector Output Cosine Similarity (TurboQuant):      {:.5}", cos_turbo);

    // Check row 0 reconstruction cosine
    let mut row0_recon = vec![0.0f32; cols];
    let lines_per_row = cols / 128;
    for l in 0..lines_per_row {
        let chunk: &mut [f32; 128] = (&mut row0_recon[l * 128..(l + 1) * 128]).try_into().unwrap();
        tensor_3bit.cache_lines[l].unpack_128(chunk);
    }
    versor.invert_in_place(&mut row0_recon);
    let cos_row0 = cosine_similarity(&weights[0..cols], &row0_recon);
    println!("Row 0 Weight Reconstruction Cosine:                     {:.5}", cos_row0);

    assert!(cos_matvec > 0.94, "Matvec cosine similarity with 3-bit Clifford must exceed 0.94");
    assert!(cos_turbo > 0.98, "Matvec cosine similarity with TurboQuant must exceed 0.98");
    assert!(cos_row0 > 0.984, "Row 0 reconstruction cosine must exceed 0.984");
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b).max(1e-12)
}
