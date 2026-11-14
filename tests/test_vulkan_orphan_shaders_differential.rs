// ============================================================================
// VERSORC-MATH: VULKAN SHADER DIFFERENTIAL GROUND-TRUTH TEST BATTERY
// License: Apache-2.0
//
// Tests newly connected GPU shaders (RoPE, RMSNorm, SiLU-Mul) on physical GPU
// against formal analytical CPU ground truths and PlanarSpinorBank.
// ============================================================================

#![cfg(feature = "vulkan")]

use versorc_math::gpu::vulkan::GpuComputeContext;
use versorc_math::prelude::*;

#[test]
fn test_vulkan_rope_differential_neox_and_llama_cpu_vs_gpu() {
    let ctx = match GpuComputeContext::init_compute() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping Vulkan RoPE test: no compatible GPU ({})", e);
            return;
        }
    };

    let n_heads = 8u32;
    let head_dim = 64u32;
    let total_elements = (n_heads * head_dim) as usize;
    let pos = 17u32;
    let rope_theta = 10000.0f32;

    // Deterministic pseudo-random input
    let mut initial = Vec::with_capacity(total_elements);
    for i in 0..total_elements {
        let t = i as f32 * 0.05;
        initial.push(t.sin() * 0.8 + (t * 0.3).cos() * 0.2);
    }

    // ------------------------------------------------------------------------
    // 1. NeoX Layout Test: [i, i + half_dim]
    // ------------------------------------------------------------------------
    let mut cpu_neox = initial.clone();
    let bank = PlanarSpinorBank::from_rope(head_dim as usize, pos as usize, rope_theta);
    for h in 0..(n_heads as usize) {
        let head_slice = &mut cpu_neox[h * (head_dim as usize)..(h + 1) * (head_dim as usize)];
        bank.apply_in_place_neox(head_slice);
    }

    let mut gpu_neox = initial.clone();
    ctx.dispatch_rope(&mut gpu_neox, pos, n_heads, head_dim, rope_theta, true)
        .expect("dispatch_rope NeoX failed on GPU");

    for i in 0..total_elements {
        let diff = (cpu_neox[i] - gpu_neox[i]).abs();
        assert!(
            diff < 1e-4,
            "RoPE NeoX discrepancy at {}: CPU={}, GPU={}, diff={}",
            i,
            cpu_neox[i],
            gpu_neox[i],
            diff
        );
    }

    // ------------------------------------------------------------------------
    // 2. LLaMA Layout Test: interleaved [2*i, 2*i + 1]
    // ------------------------------------------------------------------------
    let mut cpu_llama = initial.clone();
    for h in 0..(n_heads as usize) {
        let head_slice = &mut cpu_llama[h * (head_dim as usize)..(h + 1) * (head_dim as usize)];
        bank.apply_in_place(head_slice);
    }

    let mut gpu_llama = initial.clone();
    ctx.dispatch_rope(&mut gpu_llama, pos, n_heads, head_dim, rope_theta, false)
        .expect("dispatch_rope LLaMA failed on GPU");

    for i in 0..total_elements {
        let diff = (cpu_llama[i] - gpu_llama[i]).abs();
        assert!(
            diff < 1e-4,
            "RoPE LLaMA discrepancy at {}: CPU={}, GPU={}, diff={}",
            i,
            cpu_llama[i],
            gpu_llama[i],
            diff
        );
    }
}

#[test]
fn test_vulkan_rmsnorm_differential_cpu_vs_gpu() {
    let ctx = match GpuComputeContext::init_compute() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping Vulkan RMSNorm test: no compatible GPU ({})", e);
            return;
        }
    };

    let d_model = 1024usize;
    let eps = 1e-5f32;

    let mut x = Vec::with_capacity(d_model);
    let mut gamma = Vec::with_capacity(d_model);
    for i in 0..d_model {
        let t = i as f32 * 0.02;
        x.push(t.sin() * 1.5 - (t * 0.4).cos() * 0.7);
        gamma.push(1.0 + (t * 0.1).sin() * 0.2);
    }

    // CPU Ground Truth: y_i = x_i / sqrt(mean(x^2) + eps) * gamma_i
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let mean_sq = sum_sq / (d_model as f32);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let cpu_out: Vec<f32> = x
        .iter()
        .zip(&gamma)
        .map(|(&xi, &gi)| xi * inv_rms * gi)
        .collect();

    let gpu_out = ctx
        .dispatch_rmsnorm(&x, &gamma, eps)
        .expect("dispatch_rmsnorm failed on GPU");

    assert_eq!(gpu_out.len(), d_model);
    for i in 0..d_model {
        let diff = (cpu_out[i] - gpu_out[i]).abs();
        assert!(
            diff < 1e-4,
            "RMSNorm discrepancy at {}: CPU={}, GPU={}, diff={}",
            i,
            cpu_out[i],
            gpu_out[i],
            diff
        );
    }
}

#[test]
fn test_vulkan_silu_mul_swiglu_differential_cpu_vs_gpu() {
    let ctx = match GpuComputeContext::init_compute() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping Vulkan SiLU-Mul test: no compatible GPU ({})", e);
            return;
        }
    };

    let n = 2048usize;
    let mut gate = Vec::with_capacity(n);
    let mut up = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 * 0.01;
        gate.push(t.sin() * 3.0);
        up.push((t * 0.7).cos() * 2.0);
    }

    // CPU Ground Truth: silu(gate) * up = gate / (1 + exp(-gate)) * up
    let cpu_out: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect();

    let gpu_out = ctx
        .dispatch_silu_mul(&gate, &up)
        .expect("dispatch_silu_mul failed on GPU");

    assert_eq!(gpu_out.len(), n);
    for i in 0..n {
        let diff = (cpu_out[i] - gpu_out[i]).abs();
        assert!(
            diff < 1e-5,
            "SiLU-Mul discrepancy at {}: CPU={}, GPU={}, diff={}",
            i,
            cpu_out[i],
            gpu_out[i],
            diff
        );
    }
}

#[test]
fn test_vulkan_rope_multi_position_and_dimension_stress() {
    let ctx = match GpuComputeContext::init_compute() {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "Skipping Vulkan RoPE stress test: no compatible GPU ({})",
                e
            );
            return;
        }
    };

    let configs: &[(u32, u32, &[u32])] = &[
        (4, 64, &[0, 1, 13, 128, 512]),
        (8, 128, &[0, 7, 64, 256, 2048]),
        (16, 64, &[1, 42, 1024]),
    ];

    let rope_theta = 10000.0f32;

    for &(n_heads, head_dim, positions) in configs {
        let total_elements = (n_heads * head_dim) as usize;
        let mut initial = Vec::with_capacity(total_elements);
        for i in 0..total_elements {
            let t = i as f32 * 0.031 + 0.1;
            initial.push(t.cos() * 1.2 - (t * 0.5).sin() * 0.4);
        }

        for &pos in positions {
            let bank = PlanarSpinorBank::from_rope(head_dim as usize, pos as usize, rope_theta);

            // 1. NeoX mode
            let mut cpu_neox = initial.clone();
            for h in 0..(n_heads as usize) {
                let slice = &mut cpu_neox[h * (head_dim as usize)..(h + 1) * (head_dim as usize)];
                bank.apply_in_place_neox(slice);
            }

            let mut gpu_neox = initial.clone();
            ctx.dispatch_rope(&mut gpu_neox, pos, n_heads, head_dim, rope_theta, true)
                .expect("dispatch_rope NeoX stress failed on GPU");

            for i in 0..total_elements {
                let diff = (cpu_neox[i] - gpu_neox[i]).abs();
                assert!(
                    diff < 1e-3,
                    "RoPE NeoX stress diff at [heads={}, dim={}, pos={}, idx={}]: CPU={}, GPU={}, diff={}",
                    n_heads, head_dim, pos, i, cpu_neox[i], gpu_neox[i], diff
                );
            }

            // 2. LLaMA mode
            let mut cpu_llama = initial.clone();
            for h in 0..(n_heads as usize) {
                let slice = &mut cpu_llama[h * (head_dim as usize)..(h + 1) * (head_dim as usize)];
                bank.apply_in_place(slice);
            }

            let mut gpu_llama = initial.clone();
            ctx.dispatch_rope(&mut gpu_llama, pos, n_heads, head_dim, rope_theta, false)
                .expect("dispatch_rope LLaMA stress failed on GPU");

            for i in 0..total_elements {
                let diff = (cpu_llama[i] - gpu_llama[i]).abs();
                assert!(
                    diff < 1e-3,
                    "RoPE LLaMA stress diff at [heads={}, dim={}, pos={}, idx={}]: CPU={}, GPU={}, diff={}",
                    n_heads, head_dim, pos, i, cpu_llama[i], gpu_llama[i], diff
                );
            }
        }
    }
}

#[test]
fn test_vulkan_rmsnorm_extreme_eps_and_dimensions() {
    let ctx = match GpuComputeContext::init_compute() {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "Skipping Vulkan RMSNorm stress test: no compatible GPU ({})",
                e
            );
            return;
        }
    };

    let dimensions = [256usize, 512, 2048, 4096];
    let epsilons = [1e-4f32, 1e-5f32, 1e-6f32];

    for &d_model in &dimensions {
        let mut x = Vec::with_capacity(d_model);
        let mut gamma = Vec::with_capacity(d_model);
        for i in 0..d_model {
            let t = i as f32 * 0.007;
            x.push(t.sin() * 2.5 + (t * 2.0).cos() * 0.5);
            gamma.push(0.5 + (t * 0.3).sin().abs());
        }

        for &eps in &epsilons {
            let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
            let mean_sq = sum_sq / (d_model as f32);
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            let cpu_out: Vec<f32> = x
                .iter()
                .zip(&gamma)
                .map(|(&xi, &gi)| xi * inv_rms * gi)
                .collect();

            let gpu_out = ctx
                .dispatch_rmsnorm(&x, &gamma, eps)
                .expect("dispatch_rmsnorm stress failed on GPU");

            assert_eq!(gpu_out.len(), d_model);
            for i in 0..d_model {
                let diff = (cpu_out[i] - gpu_out[i]).abs();
                assert!(
                    diff < 1e-4,
                    "RMSNorm stress diff at [d_model={}, eps={}, idx={}]: CPU={}, GPU={}, diff={}",
                    d_model,
                    eps,
                    i,
                    cpu_out[i],
                    gpu_out[i],
                    diff
                );
            }
        }
    }
}

#[test]
fn test_vulkan_silu_mul_high_volume_boundary_cases() {
    let ctx = match GpuComputeContext::init_compute() {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "Skipping Vulkan SiLU-Mul boundary test: no compatible GPU ({})",
                e
            );
            return;
        }
    };

    let n = 16384usize;
    let mut gate = Vec::with_capacity(n);
    let mut up = Vec::with_capacity(n);

    // Boundary cases: extreme negatives, zeros, extreme positives, subnormals
    for i in 0..n {
        let g = match i % 8 {
            0 => 0.0f32,
            1 => -20.0f32,
            2 => 20.0f32,
            3 => -100.0f32,
            4 => 100.0f32,
            5 => 1e-6f32,
            6 => -1e-6f32,
            _ => (i as f32 * 0.01).sin() * 4.0,
        };
        let u = ((i as f32) * 0.03).cos() * 1.5;
        gate.push(g);
        up.push(u);
    }

    let cpu_out: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| {
            let sig = 1.0 / (1.0 + (-g).exp());
            (g * sig) * u
        })
        .collect();

    let gpu_out = ctx
        .dispatch_silu_mul(&gate, &up)
        .expect("dispatch_silu_mul high volume failed on GPU");

    assert_eq!(gpu_out.len(), n);
    for i in 0..n {
        let diff = (cpu_out[i] - gpu_out[i]).abs();
        assert!(
            diff < 1e-4,
            "SiLU-Mul boundary diff at index {}: CPU={}, GPU={}, diff={}",
            i,
            cpu_out[i],
            gpu_out[i],
            diff
        );
    }
}
