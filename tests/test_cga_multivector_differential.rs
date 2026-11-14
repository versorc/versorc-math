// ============================================================================
// VERSORC-MATH: CGA 5D MULTIVECTOR SANDWICH & QUANTIZATION DYNAMICS
// License: Apache-2.0
//
// Formal differential verification establishing exact parity between 5D multivector
// sandwiches M P M~ in Cl(4, 1, 0), Euclidean analytical formulas, and dynamic
// non-Gaussian Lloyd-Max iterative convergence.
// ============================================================================

use versorc_math::cga::{CgaConformalMotor, CgaPlane, CgaPoint, CgaSphere, CgaVector5D};
use versorc_math::quantization::clifford_3bit::compute_lloyd_max_1d;

#[test]
fn test_cga_sphere_inversion_multivector_sandwich() {
    let center = [1.0f32, 2.0, -1.0];
    let radius = 3.0f32;
    let sphere = CgaSphere::sphere(center[0], center[1], center[2], radius);

    let test_points = [
        [4.0f32, 2.0, -1.0], // distance = 3, on surface -> unchanged
        [1.0, 5.0, -1.0],    // distance = 3, on surface -> unchanged
        [1.0, 2.0, 2.0],     // distance = 3, on surface -> unchanged
        [1.0, 2.0, 0.5],     // inside sphere, distance = 1.5 -> moves to distance 6.0
        [7.0, 2.0, -1.0],    // outside sphere, distance = 6.0 -> moves to distance 1.5
    ];

    for &pt_coords in &test_points {
        let pt = CgaPoint::point(pt_coords[0], pt_coords[1], pt_coords[2]);

        // 1. Analytical geometric inversion
        let analytical = pt.invert_in_sphere(&sphere);
        let [ax, ay, az] = analytical.to_euclidean();

        // 2. Full Cl(4, 1, 0) 1-versor sandwich: P' = S * P * S^-1
        // In Cl(4, 1), the dual sphere is a 1-vector: S = c + 0.5*(||c||^2 - r^2)*ni + no
        let s_mv = sphere.to_multivector();
        let p_mv = pt.to_multivector();
        let s_inv = s_mv.inverse().expect("Sphere 1-versor must be invertible");
        let p_prime_mv = s_mv * p_mv * s_inv;

        let v5 = CgaVector5D::from_multivector(&p_prime_mv);
        let [mx, my, mz] = v5.to_euclidean();

        // 3. Differential comparison
        assert!(
            (ax - mx).abs() < 1e-4,
            "Spherical inversion X mismatch: analytical={}, mv={}",
            ax,
            mx
        );
        assert!(
            (ay - my).abs() < 1e-4,
            "Spherical inversion Y mismatch: analytical={}, mv={}",
            ay,
            my
        );
        assert!(
            (az - mz).abs() < 1e-4,
            "Spherical inversion Z mismatch: analytical={}, mv={}",
            az,
            mz
        );
    }
}

#[test]
fn test_cga_plane_reflection_multivector_sandwich() {
    let normal = [0.0f32, 0.0, 1.0];
    let d = 2.0f32; // Plane: z = 2.0
    let plane = CgaPlane::plane(normal[0], normal[1], normal[2], d);

    let test_points = [
        [0.0f32, 0.0, 0.0], // z=0 -> reflected to z=4
        [1.5, -2.0, 2.0],   // on plane -> stays z=2
        [3.0, 1.0, 5.0],    // z=5 -> reflected to z=-1
    ];

    for &pt_coords in &test_points {
        let pt = CgaPoint::point(pt_coords[0], pt_coords[1], pt_coords[2]);

        // 1. Analytical reflection: p' = p - 2*(p.n - d)*n
        let dist =
            pt_coords[0] * normal[0] + pt_coords[1] * normal[1] + pt_coords[2] * normal[2] - d;
        let expected = [
            pt_coords[0] - 2.0 * dist * normal[0],
            pt_coords[1] - 2.0 * dist * normal[1],
            pt_coords[2] - 2.0 * dist * normal[2],
        ];

        // 2. Multivector sandwich in Cl(4, 1, 0): P' = Pi * P * Pi^-1
        // Plane is a 1-vector Pi = nx*e1 + ny*e2 + nz*e3 + d*ni
        let pi_mv = plane.to_multivector();
        let p_mv = pt.to_multivector();
        let pi_inv = pi_mv.inverse().expect("Plane 1-versor must be invertible");
        let p_prime_mv = pi_mv * p_mv * pi_inv;

        let v5 = CgaVector5D::from_multivector(&p_prime_mv);
        let [mx, my, mz] = v5.to_euclidean();

        assert!(
            (expected[0] - mx).abs() < 1e-4,
            "Plane reflection X mismatch: exp={}, mv={}",
            expected[0],
            mx
        );
        assert!(
            (expected[1] - my).abs() < 1e-4,
            "Plane reflection Y mismatch: exp={}, mv={}",
            expected[1],
            my
        );
        assert!(
            (expected[2] - mz).abs() < 1e-4,
            "Plane reflection Z mismatch: exp={}, mv={}",
            expected[2],
            mz
        );
    }
}

#[test]
fn test_cga_conformal_motor_composite_sandwich_differential() {
    for seed in 0..20 {
        let fi = seed as f32;
        let axis = [
            (fi * 0.3).cos(),
            (fi * 0.5).sin(),
            (fi * 0.7).cos() * 0.5 + 0.5,
        ];
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let unit_axis = [axis[0] / norm, axis[1] / norm, axis[2] / norm];

        let angle = (fi * 0.4).sin() * 2.0;
        let trans = [
            (fi * 0.1).sin() * 5.0,
            (fi * 0.2).cos() * 3.0,
            (fi * 0.15).sin() * 4.0,
        ];
        let scale = 1.0 + (fi * 0.1).sin().abs() * 1.5; // lambda > 0

        let motor = CgaConformalMotor::new(unit_axis, angle, trans, scale);

        let pt = CgaPoint::point(
            fi * 0.5 - 2.0,
            (fi * 0.3).cos() * 3.0,
            (fi * 0.2).sin() * 4.0,
        );

        // Analytical Rodrigues fast-path
        let p_analytical = motor.transform_point(&pt);
        let [ax, ay, az] = p_analytical.to_euclidean();

        // 5D Multivector sandwich: M * P * M~
        let p_mv = motor.transform_point_multivector(&pt);
        let [mx, my, mz] = p_mv.to_euclidean();

        assert!(
            (ax - mx).abs() < 5e-4,
            "Seed {}: Conformal Motor X diff: analytical={}, mv={}",
            seed,
            ax,
            mx
        );
        assert!(
            (ay - my).abs() < 5e-4,
            "Seed {}: Conformal Motor Y diff: analytical={}, mv={}",
            seed,
            ay,
            my
        );
        assert!(
            (az - mz).abs() < 5e-4,
            "Seed {}: Conformal Motor Z diff: analytical={}, mv={}",
            seed,
            az,
            mz
        );
    }
}

#[test]
fn test_lloyd_max_general_non_gaussian_distributions() {
    // 1. Uniform distribution on [-6, 6]
    let n_samples = 100_000;
    let mut uniform_samples = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let u = (i as f32 + 0.5) / n_samples as f32; // [0, 1]
        uniform_samples.push(u * 12.0 - 6.0);
    }

    let num_levels = 8;
    let (u_centroids, u_bounds) = compute_lloyd_max_1d(&uniform_samples, num_levels, 50, 1e-5);

    // For uniform [-6, 6] with 8 levels, step = 12/8 = 1.5. Centroids: -5.25, -3.75, ..., +5.25
    for (k, &c) in u_centroids.iter().enumerate() {
        let expected_c = -6.0 + 1.5 * (k as f32 + 0.5);
        assert!(
            (c - expected_c).abs() < 0.05,
            "Uniform level {} centroid discrepancy: got={}, expected={}",
            k,
            c,
            expected_c
        );
    }
    for (k, &b) in u_bounds.iter().enumerate() {
        let expected_b = -6.0 + 1.5 * (k as f32 + 1.0);
        assert!(
            (b - expected_b).abs() < 0.05,
            "Uniform boundary {} discrepancy: got={}, expected={}",
            k,
            b,
            expected_b
        );
    }

    // 2. Heavy-tailed Laplacian distribution: f(x) = 0.5 * exp(-|x|)
    let mut laplace_samples = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let u = ((i as f32 + 0.5) / n_samples as f32) - 0.5; // (-0.5, 0.5)
        let sgn = if u < 0.0 { -1.0 } else { 1.0 };
        let x = -sgn * (1.0 - 2.0 * u.abs().min(0.49999)).ln();
        laplace_samples.push(x);
    }

    let (l_centroids, l_bounds) = compute_lloyd_max_1d(&laplace_samples, num_levels, 50, 1e-5);
    // Symmetry check: c[k] == -c[num_levels - 1 - k]
    for k in 0..num_levels / 2 {
        let left = l_centroids[k];
        let right = l_centroids[num_levels - 1 - k];
        assert!(
            (left + right).abs() < 0.05,
            "Laplace centroids must be symmetric around zero: c[{}]={}, c[{}]={}",
            k,
            left,
            num_levels - 1 - k,
            right
        );
    }

    // Centroids must be strictly ordered
    for k in 0..num_levels - 1 {
        assert!(l_centroids[k] < l_centroids[k + 1]);
        assert!(l_bounds[k] > l_centroids[k] && l_bounds[k] < l_centroids[k + 1]);
    }
}
