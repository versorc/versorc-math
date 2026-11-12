// ============================================================================
// VERSORC-MATH: COMPREHENSIVE VULKAN GA OPERATOR TEST BATTERY (GPU vs CPU)
// License: Apache-2.0
//
// Calibrates all 15 Geometric Algebra operators executing on physical GPU
// against the CPU ground-truth engine and mathematical axiomatic invariants.
// ============================================================================

#![cfg(feature = "vulkan")]

use versorc_math::prelude::*;
use versorc_math::gpu::vulkan::GpuComputeContext;

/// Tolerance for single-precision 32-bit floating-point GPU comparisons
const EPSILON_F32: f32 = 1e-4;

fn create_test_multivector_batches(n: usize) -> (Vec<[f32; 8]>, Vec<[f32; 8]>) {
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);

    for i in 0..n {
        let fi = i as f32;
        a.push([
            (fi * 0.1).sin(),
            (fi * 0.2).cos(),
            (fi * 0.3 + 1.0).sin(),
            (fi * 0.15).cos() * 0.5,
            (fi * 0.25 - 0.5).sin(),
            (fi * 0.05).cos(),
            (fi * 0.18 + 0.2).sin(),
            (fi * 0.12).cos() * 0.3,
        ]);

        b.push([
            (fi * 0.14 + 0.3).cos(),
            (fi * 0.22 - 0.1).sin(),
            (fi * 0.08).cos(),
            (fi * 0.31).sin() * 0.4,
            (fi * 0.17 + 0.8).cos(),
            (fi * 0.21).sin(),
            (fi * 0.11 - 0.4).cos(),
            (fi * 0.09).sin() * 0.2,
        ]);
    }

    (a, b)
}

#[test]
fn test_vulkan_all_12_cl3_operators_differential_cpu_vs_gpu() {
    let ctx = GpuComputeContext::init_compute()
        .expect("Vulkan compute context initialization failed on physical GPU");

    let n = 256;
    let (a_batch, b_batch) = create_test_multivector_batches(n);

    // Operator mapping:
    // 0: Geometric Product
    // 1: Wedge Product
    // 2: Left Contraction
    // 3: Hestenes Inner
    // 4: Regressive Meet
    // 5: Lie Commutator
    // 6: Jordan Anticommutator
    // 7: Versor Sandwich
    // 8: Hodge Dual
    // 9: Reversion
    // 10: Grade Involution
    // 11: Clifford Conjugation
    for op_code in 0..=11 {
        let gpu_results = ctx.dispatch_cl3_batch(&a_batch, &b_batch, op_code)
            .unwrap_or_else(|e| panic!("GPU dispatch failed for op_code {}: {}", op_code, e));

        assert_eq!(gpu_results.len(), n);

        for i in 0..n {
            let cpu_a = Cl3 { coords: a_batch[i] };
            let cpu_b = Cl3 { coords: b_batch[i] };

            let cpu_expected: Cl3 = match op_code {
                0 => cpu_a.geometric_product(&cpu_b),
                1 => cpu_a.wedge(&cpu_b),
                2 => cpu_a.left_contraction(&cpu_b),
                3 => cpu_a.hestenes_inner_product(&cpu_b),
                4 => cpu_a.regressive_product(&cpu_b).expect("Cl3 regressive meet failed"),
                5 => cpu_a.commutator(&cpu_b),
                6 => cpu_a.anticommutator(&cpu_b),
                7 => cpu_a.sandwich(&cpu_b),
                8 => cpu_a.dual().expect("Cl3 dual failed"),
                9 => cpu_a.reverse(),
                10 => cpu_a.grade_involution(),
                11 => cpu_a.conjugation(),
                _ => unreachable!(),
            };

            let gpu_res = gpu_results[i];
            for k in 0..8 {
                let diff = (gpu_res[k] - cpu_expected.coords[k]).abs();
                assert!(
                    diff < EPSILON_F32,
                    "Mismatch at op_code {} element {} blade {}: GPU {:?} vs CPU {:?} (diff: {})",
                    op_code, i, k, gpu_res[k], cpu_expected.coords[k], diff
                );
            }
        }
    }
}

#[test]
fn test_vulkan_poincare_mobius_and_inversion_differential() {
    let ctx = GpuComputeContext::init_compute()
        .expect("Vulkan compute context initialization failed on physical GPU");

    let n = 256;
    let mut u_pts = Vec::with_capacity(n);
    let mut v_pts = Vec::with_capacity(n);

    for i in 0..n {
        let fi = i as f32;
        // Keep strictly inside unit Poincare ball: ||u|| < 0.9, ||v|| < 0.9
        let scale = 0.45;
        u_pts.push([
            (fi * 0.1).sin() * scale,
            (fi * 0.2).cos() * scale,
            (fi * 0.3).sin() * scale,
        ]);
        v_pts.push([
            (fi * 0.15 + 0.2).cos() * scale,
            (fi * 0.25 - 0.1).sin() * scale,
            (fi * 0.05).cos() * scale,
        ]);
    }

    // 1. Test Mobius Gyrovector Addition: op_code 0
    let gpu_mobius = ctx.dispatch_poincare_batch(&u_pts, &v_pts, 0, [0.0; 3], 1.0)
        .expect("GPU Mobius batch dispatch failed");

    assert_eq!(gpu_mobius.len(), n);
    for i in 0..n {
        let cpu_u = PoincarePoint::new(u_pts[i].to_vec());
        let cpu_v = PoincarePoint::new(v_pts[i].to_vec());
        let cpu_res = cpu_u.mobius_add(&cpu_v);

        for k in 0..3 {
            let diff = (gpu_mobius[i][k] - cpu_res.coords[k]).abs();
            assert!(
                diff < EPSILON_F32,
                "Mobius Add mismatch at index {} coord {}: GPU {:?} vs CPU {:?}",
                i, k, gpu_mobius[i][k], cpu_res.coords[k]
            );
        }
    }

    // 2. Test Conformal Spherical Inversion: op_code 2
    let sphere_center = [1.0f32, -0.5, 0.5];
    let sphere_radius = 2.0f32;

    let gpu_inversion = ctx.dispatch_poincare_batch(&u_pts, &[], 2, sphere_center, sphere_radius)
        .expect("GPU Conformal Inversion batch dispatch failed");

    for i in 0..n {
        let p = u_pts[i];
        let diff = [
            p[0] - sphere_center[0],
            p[1] - sphere_center[1],
            p[2] - sphere_center[2],
        ];
        let d2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
        let factor = (sphere_radius * sphere_radius) / d2;
        let expected = [
            sphere_center[0] + diff[0] * factor,
            sphere_center[1] + diff[1] * factor,
            sphere_center[2] + diff[2] * factor,
        ];

        for k in 0..3 {
            let err = (gpu_inversion[i][k] - expected[k]).abs();
            assert!(
                err < EPSILON_F32,
                "Conformal Inversion mismatch at index {} coord {}: GPU {:?} vs CPU {:?}",
                i, k, gpu_inversion[i][k], expected[k]
            );
        }
    }
}

#[test]
fn test_vulkan_algebraic_invariants_and_axiomatic_fuzzing() {
    let ctx = GpuComputeContext::init_compute()
        .expect("Vulkan compute context initialization failed on physical GPU");

    let n = 128;
    // 1. Invariant: Wedge Nilpotence on 1-vectors (v ^ v == 0)
    let mut vector_batch = Vec::with_capacity(n);
    for i in 0..n {
        let fi = i as f32;
        let mut v = [0.0f32; 8];
        v[1] = (fi * 0.1).sin();
        v[2] = (fi * 0.2).cos();
        v[4] = (fi * 0.3).sin();
        vector_batch.push(v);
    }

    let wedge_self = ctx.dispatch_cl3_batch(&vector_batch, &vector_batch, 1)
        .expect("Wedge self dispatch failed");

    for i in 0..n {
        for k in 0..8 {
            assert!(
                wedge_self[i][k].abs() < 1e-6,
                "Wedge nilpotency failed on GPU at index {} blade {}: value = {}",
                i, k, wedge_self[i][k]
            );
        }
    }

    // 2. Invariant: Rotor Isometry (||R x R~|| == ||x||)
    let mut rotor_batch = Vec::with_capacity(n);
    let mut point_batch = Vec::with_capacity(n);

    for i in 0..n {
        let fi = i as f32;
        // Rotor in Cl3: cos(theta/2) + sin(theta/2) * e12
        let theta = fi * 0.1;
        let mut r = [0.0f32; 8];
        r[0] = (theta * 0.5).cos();
        r[3] = (theta * 0.5).sin(); // e12 bivector
        rotor_batch.push(r);

        let mut pt = [0.0f32; 8];
        pt[1] = (fi * 0.3).sin() * 2.0;
        pt[2] = (fi * 0.4).cos() * 2.0;
        pt[4] = (fi * 0.5).sin() * 2.0;
        point_batch.push(pt);
    }

    let rotated = ctx.dispatch_cl3_batch(&rotor_batch, &point_batch, 7)
        .expect("Sandwich dispatch failed");

    for i in 0..n {
        let original_norm = (point_batch[i][1].powi(2) + point_batch[i][2].powi(2) + point_batch[i][4].powi(2)).sqrt();
        let rotated_norm = (rotated[i][1].powi(2) + rotated[i][2].powi(2) + rotated[i][4].powi(2)).sqrt();
        let diff = (original_norm - rotated_norm).abs();
        assert!(
            diff < EPSILON_F32,
            "Rotor isometry violated at index {}: orig {} vs rot {}",
            i, original_norm, rotated_norm
        );
    }
}
