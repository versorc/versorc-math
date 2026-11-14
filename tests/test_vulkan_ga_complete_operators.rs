// ============================================================================
// VERSORC-MATH: COMPREHENSIVE VULKAN GA OPERATOR TEST BATTERY (GPU vs CPU)
// License: Apache-2.0
//
// Calibrates all 15 Geometric Algebra operators executing on physical GPU
// against the CPU ground-truth engine, closed-form literature benchmarks,
// and mathematical axiomatic invariants.
// ============================================================================

#![cfg(feature = "vulkan")]

use versorc_math::gpu::shaders::CgaTransformPushConstants;
use versorc_math::gpu::vulkan::GpuComputeContext;
use versorc_math::prelude::*;

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
fn test_vulkan_all_13_cl3_operators_differential_cpu_vs_gpu() {
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
    // 12: Undual
    for op_code in 0..=12 {
        let gpu_results = ctx
            .dispatch_cl3_batch(&a_batch, &b_batch, op_code)
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
                4 => cpu_a
                    .regressive_product(&cpu_b)
                    .expect("Cl3 regressive meet failed"),
                5 => cpu_a.commutator(&cpu_b),
                6 => cpu_a.anticommutator(&cpu_b),
                7 => cpu_a.sandwich(&cpu_b),
                8 => cpu_a.dual().expect("Cl3 dual failed"),
                9 => cpu_a.reverse(),
                10 => cpu_a.grade_involution(),
                11 => cpu_a.conjugation(),
                12 => cpu_a.undual(),
                _ => unreachable!(),
            };

            let gpu_res = gpu_results[i];
            for (k, (&gpu_v, &cpu_v)) in gpu_res.iter().zip(cpu_expected.coords.iter()).enumerate()
            {
                let diff = (gpu_v - cpu_v).abs();
                assert!(
                    diff < EPSILON_F32,
                    "Mismatch at op_code {} element {} blade {}: GPU {:?} vs CPU {:?} (diff: {})",
                    op_code,
                    i,
                    k,
                    gpu_v,
                    cpu_v,
                    diff
                );
            }
        }
    }
}

#[test]
fn test_vulkan_all_6_poincare_operators_differential() {
    let ctx = GpuComputeContext::init_compute()
        .expect("Vulkan compute context initialization failed on physical GPU");

    let n = 256;
    let mut u_pts = Vec::with_capacity(n);
    let mut v_pts = Vec::with_capacity(n);

    for i in 0..n {
        let fi = i as f32;
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

    // 0. Mobius Gyrovector Addition: op_code 0
    let gpu_mobius = ctx
        .dispatch_poincare_batch(&u_pts, &v_pts, 0, [0.0; 3], 1.0)
        .expect("GPU Mobius batch dispatch failed");
    assert_eq!(gpu_mobius.len(), n);
    for i in 0..n {
        let cpu_u = PoincarePoint::new(u_pts[i].to_vec());
        let cpu_v = PoincarePoint::new(v_pts[i].to_vec());
        let cpu_res = cpu_u.mobius_add(&cpu_v);
        for (k, (&gpu_v, &cpu_v)) in gpu_mobius[i].iter().zip(cpu_res.coords.iter()).enumerate() {
            let diff = (gpu_v - cpu_v).abs();
            assert!(
                diff < EPSILON_F32,
                "Mobius Add mismatch at index {} coord {}: GPU {:?} vs CPU {:?}",
                i,
                k,
                gpu_v,
                cpu_v
            );
        }
    }

    // 1. Geodesic Distance: op_code 1 via dedicated helper
    let gpu_dists = ctx
        .dispatch_poincare_distance(&u_pts, &v_pts)
        .expect("GPU Geodesic Distance dispatch failed");
    assert_eq!(gpu_dists.len(), n);
    for i in 0..n {
        let cpu_u = PoincarePoint::new(u_pts[i].to_vec());
        let cpu_v = PoincarePoint::new(v_pts[i].to_vec());
        let cpu_d = cpu_u.geodesic_distance(&cpu_v);
        let diff = (gpu_dists[i] - cpu_d).abs();
        assert!(
            diff < EPSILON_F32,
            "Geodesic Distance mismatch at index {}: GPU {} vs CPU {}",
            i,
            gpu_dists[i],
            cpu_d
        );
    }

    // 2. Conformal Spherical Inversion: op_code 2
    let sphere_center = [1.0f32, -0.5, 0.5];
    let sphere_radius = 2.0f32;
    let gpu_inversion = ctx
        .dispatch_poincare_batch(&u_pts, &[], 2, sphere_center, sphere_radius)
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
                i,
                k,
                gpu_inversion[i][k],
                expected[k]
            );
        }
    }

    // 3 & 4. Poincare <-> Klein Bijections: op_codes 3 & 4
    let gpu_klein = ctx
        .dispatch_poincare_batch(&u_pts, &[], 3, [0.0; 3], 1.0)
        .expect("GPU Poincare to Klein dispatch failed");
    for i in 0..n {
        let cpu_u = PoincarePoint::new(u_pts[i].to_vec());
        let cpu_k = cpu_u.to_klein();
        for k in 0..3 {
            let diff = (gpu_klein[i][k] - cpu_k[k]).abs();
            assert!(
                diff < EPSILON_F32,
                "Poincare to Klein mismatch at index {} coord {}: GPU {} vs CPU {}",
                i,
                k,
                gpu_klein[i][k],
                cpu_k[k]
            );
        }
    }

    let gpu_back_to_poincare = ctx
        .dispatch_poincare_batch(&gpu_klein, &[], 4, [0.0; 3], 1.0)
        .expect("GPU Klein to Poincare dispatch failed");
    for i in 0..n {
        for k in 0..3 {
            let diff = (gpu_back_to_poincare[i][k] - u_pts[i][k]).abs();
            assert!(
                diff < EPSILON_F32,
                "Klein -> Poincare roundtrip failed at index {} coord {}: got {} vs orig {}",
                i,
                k,
                gpu_back_to_poincare[i][k],
                u_pts[i][k]
            );
        }
    }

    // 5. Einstein Midpoint: op_code 5
    let gpu_midpoints = ctx
        .dispatch_poincare_batch(&u_pts, &v_pts, 5, [0.0; 3], 1.0)
        .expect("GPU Einstein Midpoint dispatch failed");
    for i in 0..n {
        let cpu_u = PoincarePoint::new(u_pts[i].to_vec());
        let cpu_v = PoincarePoint::new(v_pts[i].to_vec());
        let cpu_m = PoincarePoint::einstein_midpoint(&[cpu_u, cpu_v], &[1.0, 1.0]);
        for (k, (&gpu_v, &cpu_v)) in gpu_midpoints[i].iter().zip(cpu_m.coords.iter()).enumerate() {
            let diff = (gpu_v - cpu_v).abs();
            assert!(
                diff < EPSILON_F32,
                "Einstein Midpoint mismatch at index {} coord {}: GPU {} vs CPU {}",
                i,
                k,
                gpu_v,
                cpu_v
            );
        }
    }
}

#[test]
fn test_vulkan_cga_transforms_differential() {
    let ctx = GpuComputeContext::init_compute()
        .expect("Vulkan compute context initialization failed on physical GPU");

    let points = vec![
        [1.0f32, 2.0, 3.0],
        [-2.0, 0.5, 1.0],
        [0.0, 0.0, 4.0],
        [3.0, -1.0, -2.0],
    ];

    // Dilation by 2.5
    let pc_dil = CgaTransformPushConstants::dilation(points.len() as u32, 2.5);
    let gpu_dil = ctx
        .dispatch_cga_transform(&points, &pc_dil)
        .expect("CGA dilation dispatch failed");
    for (i, p) in points.iter().enumerate() {
        for k in 0..3 {
            let expected = p[k] * 2.5;
            assert!((gpu_dil[i][k] - expected).abs() < EPSILON_F32);
        }
    }

    // Translation by [10.0, -5.0, 2.0]
    let shift = [10.0f32, -5.0, 2.0];
    let pc_trans = CgaTransformPushConstants::translation(points.len() as u32, shift);
    let gpu_trans = ctx
        .dispatch_cga_transform(&points, &pc_trans)
        .expect("CGA translation dispatch failed");
    for (i, p) in points.iter().enumerate() {
        for k in 0..3 {
            let expected = p[k] + shift[k];
            assert!((gpu_trans[i][k] - expected).abs() < EPSILON_F32);
        }
    }
}

#[test]
fn test_vulkan_literature_fixtures_exact() {
    let ctx = GpuComputeContext::init_compute()
        .expect("Vulkan compute context initialization failed on physical GPU");

    // 1. UNGAR (2008) Chapter 2 / Einstein Velocity Addition in B^3:
    // u = (0.5, 0.0, 0.0), v = (0.0, 0.5, 0.0)
    // Closed-form exact result: u (+) v = (10/17, 6/17, 0) ~= (0.5882353, 0.3529412, 0.0)
    let u_lit = vec![[0.5f32, 0.0, 0.0]];
    let v_lit = vec![[0.0f32, 0.5, 0.0]];
    let gpu_mob = ctx
        .dispatch_poincare_batch(&u_lit, &v_lit, 0, [0.0; 3], 1.0)
        .expect("Ungar 2008 literature test dispatch failed");

    let expected_x = 10.0f32 / 17.0f32;
    let expected_y = 6.0f32 / 17.0f32;
    assert!(
        (gpu_mob[0][0] - expected_x).abs() < 1e-5,
        "Ungar fixture mismatch X: got {}, expected {}",
        gpu_mob[0][0],
        expected_x
    );
    assert!(
        (gpu_mob[0][1] - expected_y).abs() < 1e-5,
        "Ungar fixture mismatch Y: got {}, expected {}",
        gpu_mob[0][1],
        expected_y
    );
    assert!(gpu_mob[0][2].abs() < 1e-5);

    // Ungar Geodesic distance from origin d_B(0, u) = ln((1+||u||)/(1-||u||)) = ln(1.5 / 0.5) = ln(3) ~= 1.0986123
    let zero_pt = vec![[0.0f32, 0.0, 0.0]];
    let dist_gpu = ctx
        .dispatch_poincare_distance(&zero_pt, &u_lit)
        .expect("Ungar distance dispatch failed");
    let expected_dist = 3.0f32.ln();
    assert!(
        (dist_gpu[0] - expected_dist).abs() < 1e-5,
        "Ungar distance mismatch: got {}, expected {}",
        dist_gpu[0],
        expected_dist
    );

    // 2. HESTENES & SOBCZYK (1984) Basis Squares & Duals in Cl(3, 0):
    // e1: coords[1] = 1.0; e2: coords[2] = 1.0; e3: coords[4] = 1.0
    let mut e1 = [0.0f32; 8];
    e1[1] = 1.0;
    let mut e2 = [0.0f32; 8];
    e2[2] = 1.0;
    let mut e3 = [0.0f32; 8];
    e3[4] = 1.0;

    // e1 * e1 = 1 (scalar)
    let e1_sq = ctx.dispatch_cl3_batch(&[e1], &[e1], 0).unwrap();
    assert!((e1_sq[0][0] - 1.0).abs() < 1e-6);
    for &v in &e1_sq[0][1..] {
        assert!(v.abs() < 1e-6);
    }

    // e1 * e2 = e12 (bivector coords[3])
    let e1_mul_e2 = ctx.dispatch_cl3_batch(&[e1], &[e2], 0).unwrap();
    assert!((e1_mul_e2[0][3] - 1.0).abs() < 1e-6);
    for k in [0, 1, 2, 4, 5, 6, 7] {
        assert!(e1_mul_e2[0][k].abs() < 1e-6);
    }

    // e2 * e1 = -e12 (anticommutativity)
    let e2_mul_e1 = ctx.dispatch_cl3_batch(&[e2], &[e1], 0).unwrap();
    assert!((e2_mul_e1[0][3] + 1.0).abs() < 1e-6);

    // e1 ^ e2 = e12
    let e1_wedge_e2 = ctx.dispatch_cl3_batch(&[e1], &[e2], 1).unwrap();
    assert!((e1_wedge_e2[0][3] - 1.0).abs() < 1e-6);

    // e1 . e2 = 0 (orthogonal)
    let e1_dot_e2 = ctx.dispatch_cl3_batch(&[e1], &[e2], 3).unwrap();
    for &v in &e1_dot_e2[0] {
        assert!(v.abs() < 1e-6);
    }

    // e3 * e3 = 1 (scalar)
    let e3_sq = ctx.dispatch_cl3_batch(&[e3], &[e3], 0).unwrap();
    assert!((e3_sq[0][0] - 1.0).abs() < 1e-6);

    // e12 ^ e3 = e123 (pseudoscalar at coords[7])
    let e12_wedge_e3 = ctx.dispatch_cl3_batch(&e1_wedge_e2, &[e3], 1).unwrap();
    assert!((e12_wedge_e3[0][7] - 1.0).abs() < 1e-6);

    // 3. DORST (2007) Conformal Spherical Inversion:
    // Sphere at origin with radius 2: point (4, 0, 0) -> (1, 0, 0)
    // Point on sphere surface (2, 0, 0) -> (2, 0, 0) [fixed invariant point]
    let pts_cga = vec![[4.0f32, 0.0, 0.0], [2.0f32, 0.0, 0.0]];
    let inv_dorst = ctx
        .dispatch_poincare_batch(&pts_cga, &[], 2, [0.0; 3], 2.0)
        .unwrap();
    assert!((inv_dorst[0][0] - 1.0).abs() < 1e-6);
    assert!((inv_dorst[1][0] - 2.0).abs() < 1e-6);
}

#[test]
fn test_vulkan_irregular_batch_sizes_tail_masking() {
    let ctx = GpuComputeContext::init_compute()
        .expect("Vulkan compute context initialization failed on physical GPU");

    // Stress test batch sizes across warp/workgroup boundaries (local_size_x = 64)
    let test_sizes = [1, 3, 7, 31, 63, 64, 65, 127, 128, 129, 255, 256, 513];

    for &size in &test_sizes {
        let (a_batch, b_batch) = create_test_multivector_batches(size);
        let res = ctx
            .dispatch_cl3_batch(&a_batch, &b_batch, 0)
            .unwrap_or_else(|e| panic!("Failed at irregular batch size {}: {}", size, e));
        assert_eq!(res.len(), size, "Length mismatch for size {}", size);

        // Verify first and last elements match CPU
        let cpu_first =
            (Cl3 { coords: a_batch[0] }).geometric_product(&(Cl3 { coords: b_batch[0] }));
        let cpu_last = (Cl3 {
            coords: a_batch[size - 1],
        })
        .geometric_product(
            &(Cl3 {
                coords: b_batch[size - 1],
            }),
        );

        for ((&f, &l), (&cf, &cl)) in res[0]
            .iter()
            .zip(res[size - 1].iter())
            .zip(cpu_first.coords.iter().zip(cpu_last.coords.iter()))
        {
            assert!((f - cf).abs() < EPSILON_F32);
            assert!((l - cl).abs() < EPSILON_F32);
        }
    }
}

#[test]
fn test_vulkan_boundary_singularity_and_error_handling() {
    let ctx = GpuComputeContext::init_compute()
        .expect("Vulkan compute context initialization failed on physical GPU");

    // 1. Points near hyperbolic boundary ||u|| = 0.9999
    let near_boundary = vec![[0.9999f32, 0.0, 0.0], [0.0, 0.9999f32, 0.0]];
    let d = ctx
        .dispatch_poincare_distance(&near_boundary, &near_boundary)
        .expect("Near boundary distance dispatch should succeed without NaN");
    assert!(
        !d[0].is_nan() && !d[0].is_infinite(),
        "Distance returned NaN or Inf: {}",
        d[0]
    );
    assert!(
        d[0].abs() < 1e-3,
        "Distance to self should be ~0: got {}",
        d[0]
    );

    // 2. Conformal inversion of point at exact sphere center
    let center = [1.0f32, 2.0, 3.0];
    let pt_at_center = vec![center];
    let inv_center = ctx
        .dispatch_poincare_batch(&pt_at_center, &[], 2, center, 2.0)
        .expect("Inversion at center should not panic or divide by zero");
    assert!(!inv_center[0][0].is_nan() && !inv_center[0][1].is_nan() && !inv_center[0][2].is_nan());

    // 3. Slice length mismatch validation on binary operators
    let a_20 = vec![[1.0f32; 8]; 20];
    let b_5 = vec![[1.0f32; 8]; 5];
    let err_cl3 = ctx.dispatch_cl3_batch(&a_20, &b_5, 0);
    assert!(
        err_cl3.is_err(),
        "Expected error on mismatched binary batch length"
    );

    let u_10 = vec![[0.1f32; 3]; 10];
    let v_2 = vec![[0.1f32; 3]; 2];
    let err_poincare = ctx.dispatch_poincare_batch(&u_10, &v_2, 0, [0.0; 3], 1.0);
    assert!(
        err_poincare.is_err(),
        "Expected error on mismatched Poincare batch length"
    );
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

    let wedge_self = ctx
        .dispatch_cl3_batch(&vector_batch, &vector_batch, 1)
        .expect("Wedge self dispatch failed");

    for (i, row) in wedge_self.iter().enumerate() {
        for (k, &val) in row.iter().enumerate() {
            assert!(
                val.abs() < 1e-6,
                "Wedge nilpotency failed on GPU at index {} blade {}: value = {}",
                i,
                k,
                val
            );
        }
    }

    // 2. Invariant: Rotor Isometry (||R x R~|| == ||x||)
    let mut rotor_batch = Vec::with_capacity(n);
    let mut point_batch = Vec::with_capacity(n);

    for i in 0..n {
        let fi = i as f32;
        let theta = fi * 0.1;
        let mut r = [0.0f32; 8];
        r[0] = (theta * 0.5).cos();
        r[3] = (theta * 0.5).sin();
        rotor_batch.push(r);

        let mut pt = [0.0f32; 8];
        pt[1] = (fi * 0.3).sin() * 2.0;
        pt[2] = (fi * 0.4).cos() * 2.0;
        pt[4] = (fi * 0.5).sin() * 2.0;
        point_batch.push(pt);
    }

    let rotated = ctx
        .dispatch_cl3_batch(&rotor_batch, &point_batch, 7)
        .expect("Sandwich dispatch failed");

    for i in 0..n {
        let original_norm =
            (point_batch[i][1].powi(2) + point_batch[i][2].powi(2) + point_batch[i][4].powi(2))
                .sqrt();
        let rotated_norm =
            (rotated[i][1].powi(2) + rotated[i][2].powi(2) + rotated[i][4].powi(2)).sqrt();
        let diff = (original_norm - rotated_norm).abs();
        assert!(
            diff < EPSILON_F32,
            "Rotor isometry violated at index {}: orig {} vs rot {}",
            i,
            original_norm,
            rotated_norm
        );
    }
}
