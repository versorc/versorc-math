use versorc_math::prelude::*;

#[test]
fn test_pga_plane_properties_and_reflection() {
    assert_eq!(std::mem::size_of::<PgaPlane>(), 16);
    assert_eq!(std::mem::align_of::<PgaPlane>(), 16);

    // Plane z = 2.0 (normal [0, 0, 1], distance -2.0)
    let plane = PgaPlane::from_point_normal(&[0.0, 0.0, 2.0], &[0.0, 0.0, 1.0]);
    assert_eq!(plane.normal(), [0.0, 0.0, 1.0]);
    assert_eq!(plane.d, -2.0);

    // Point at (1, 1, 5)
    let p = PgaPoint::new(1.0, 1.0, 5.0);
    let dist = plane.distance_to_point(&p);
    assert!((dist - 3.0).abs() < 1e-6);

    // 1-versor reflection across z = 2
    let p_ref = plane.reflect_point(&p);
    assert!((p_ref.x - 1.0).abs() < 1e-6);
    assert!((p_ref.y - 1.0).abs() < 1e-6);
    assert!((p_ref.z - (-1.0)).abs() < 1e-6); // 2 - (5 - 2) = -1

    // Double reflection returns to original point (involution)
    let p_double = plane.reflect_point(&p_ref);
    assert!((p_double.z - 5.0).abs() < 1e-6);
}

#[test]
fn test_pga_point_and_line_join_meet() {
    assert_eq!(std::mem::size_of::<PgaPoint>(), 16);
    assert_eq!(std::mem::align_of::<PgaPoint>(), 16);
    assert_eq!(std::mem::size_of::<PgaLine>(), 32);
    assert_eq!(std::mem::align_of::<PgaLine>(), 32);

    let p1 = PgaPoint::new(0.0, 0.0, 0.0);
    let p2 = PgaPoint::new(0.0, 0.0, 5.0);

    // Distance
    assert!((p1.distance(&p2) - 5.0).abs() < 1e-6);
    assert_eq!(p1.midpoint(&p2).to_euclidean(), [0.0, 0.0, 2.5]);

    // Line joining p1 and p2 (Z axis)
    let line = p1.join(&p2).normalized();
    assert_eq!(line.direction(), [0.0, 0.0, 1.0]);
    assert_eq!(line.moment(), [0.0, 0.0, 0.0]); // Passes through origin

    // Distance from point (3, 4, 10) to Z axis line must be 5.0
    let test_pt = PgaPoint::new(3.0, 4.0, 10.0);
    assert!((line.distance_to_point(&test_pt) - 5.0).abs() < 1e-6);

    // Meet with plane z = 10
    let plane_z10 = PgaPlane::new(0.0, 0.0, 1.0, -10.0);
    let intersect = line.meet_plane(&plane_z10);
    assert!((intersect.x - 0.0).abs() < 1e-6);
    assert!((intersect.y - 0.0).abs() < 1e-6);
    assert!((intersect.z - 10.0).abs() < 1e-6);
}

#[test]
fn test_pga_motor_rigid_body_transforms() {
    assert_eq!(std::mem::size_of::<PgaMotor>(), 32);
    assert_eq!(std::mem::align_of::<PgaMotor>(), 32);

    // 90 degree rotation around Z axis followed by translation by [10, 20, 30]
    let motor = PgaMotor::from_rotation_translation(
        &[0.0, 0.0, 1.0],
        std::f32::consts::FRAC_PI_2,
        &[10.0, 20.0, 30.0],
    );

    let pt = PgaPoint::new(1.0, 0.0, 0.0);
    let pt_transformed = motor.transform_point(&pt);
    let [x, y, z] = pt_transformed.to_euclidean();

    // (1, 0, 0) rotated by 90 deg around Z is (0, 1, 0).
    // Translated by (10, 20, 30) yields (10, 21, 30).
    assert!((x - 10.0).abs() < 1e-5, "Expected x = 10.0, got {}", x);
    assert!((y - 21.0).abs() < 1e-5, "Expected y = 21.0, got {}", y);
    assert!((z - 30.0).abs() < 1e-5, "Expected z = 30.0, got {}", z);

    // Inverse motor restores original point
    let inv_motor = motor.inverse();
    let pt_restored = inv_motor.transform_point(&pt_transformed);
    let [rx, ry, rz] = pt_restored.to_euclidean();
    assert!((rx - 1.0).abs() < 1e-5);
    assert!((ry - 0.0).abs() < 1e-5);
    assert!((rz - 0.0).abs() < 1e-5);
}

#[test]
fn test_pga_motor_screw_axis_and_sclerp() {
    let z_line = PgaLine::z_axis();
    // Screw motion: 180 deg rotation + 10 units translation along Z
    let angle = std::f32::consts::PI;
    let pitch = 10.0 / angle;
    let screw_motor = PgaMotor::from_screw_axis(&z_line, angle, pitch);

    let pt = PgaPoint::new(2.0, 0.0, 0.0);
    let pt_t = screw_motor.transform_point(&pt);
    let [x, y, z] = pt_t.to_euclidean();

    // Rotated 180 deg: x = -2.0, y = 0.0, translated along z: z = 10.0
    assert!((x - (-2.0)).abs() < 1e-5);
    assert!((y - 0.0).abs() < 1e-5);
    assert!((z - 10.0).abs() < 1e-5);

    // ScLERP at t = 0.5 (midpoint of screw trajectory)
    let mid_motor = PgaMotor::identity().sclerp(&screw_motor, 0.5);
    let mid_pt = mid_motor.transform_point(&pt);
    let [mx, my, mz] = mid_pt.to_euclidean();

    // At t = 0.5: 90 deg rotation -> x = 0, y = 2; translation = 5.0
    assert!((mx - 0.0).abs() < 1e-4, "Expected mx = 0, got {}", mx);
    assert!((my - 2.0).abs() < 1e-4, "Expected my = 2, got {}", my);
    assert!((mz - 5.0).abs() < 1e-4, "Expected mz = 5, got {}", mz);
}

#[test]
fn test_pga_point_negation_ideal_unnormalized_and_antipodal_sclerp() {
    // 1. P + (-P) == 0 across all homogeneous coordinates including w
    let p = PgaPoint::from_homogeneous(2.5, -1.25, 4.0, 2.0);
    let sum = p + (-p);
    assert_eq!(sum, PgaPoint::from_homogeneous(0.0, 0.0, 0.0, 0.0));

    // 2. Ideal point (w = 0) rotates without absorbing translation
    let motor = PgaMotor::from_rotation_translation(
        &[0.0, 0.0, 1.0],
        std::f32::consts::FRAC_PI_2,
        &[10.0, 20.0, 30.0],
    );
    let ideal_dir = PgaPoint::ideal(1.0, 0.0, 0.0);
    let rot_ideal = motor.transform_point(&ideal_dir);
    assert!(rot_ideal.is_ideal(), "Ideal point must remain at w = 0");
    assert!((rot_ideal.x - 0.0).abs() < 1e-5);
    assert!((rot_ideal.y - 1.0).abs() < 1e-5);
    assert!((rot_ideal.z - 0.0).abs() < 1e-5);
    assert_eq!(rot_ideal.w, 0.0);

    // 3. Unnormalized & Study-perturbed motor normalization and transform invariance
    let scaled_motor = PgaMotor::new(
        motor.s * 3.5,
        motor.b23 * 3.5,
        motor.b31 * 3.5,
        motor.b12 * 3.5,
        motor.d01 * 3.5,
        motor.d02 * 3.5,
        motor.d03 * 3.5,
        motor.e0123 * 3.5,
    );
    let p_aff = PgaPoint::new(1.0, 2.0, -3.0);
    let ref_pt = motor.transform_point(&p_aff).to_euclidean();
    let scaled_pt = scaled_motor.transform_point(&p_aff).to_euclidean();
    for i in 0..3 {
        assert!((ref_pt[i] - scaled_pt[i]).abs() < 1e-5);
    }

    let plane = PgaPlane::from_point_normal(&[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0]);
    let ref_pl = motor.transform_plane(&plane);
    let scaled_pl = scaled_motor.transform_plane(&plane);
    assert!((ref_pl.a - scaled_pl.a).abs() < 1e-5);
    assert!((ref_pl.b - scaled_pl.b).abs() < 1e-5);
    assert!((ref_pl.c - scaled_pl.c).abs() < 1e-5);
    assert!((ref_pl.d - scaled_pl.d).abs() < 1e-5);

    // Perturbed motor off the Study quadric must satisfy M * M~ = 1 after normalized()
    let dirty = PgaMotor::new(1.2, 0.5, -0.3, 0.8, 0.4, -0.7, 0.2, 1.1);
    let clean = dirty.normalized();
    let check = clean.geometric_product(&clean.reverse());
    assert!(
        (check.s - 1.0).abs() < 1e-5,
        "Normalized scalar must be 1, got {}",
        check.s
    );
    assert!(
        check.e0123.abs() < 1e-5,
        "Study quadric e0123 residual must be 0, got {}",
        check.e0123
    );

    // 4. ScLERP double-cover antipodal shortest path: sclerp(M, -M, 0.5) must stay on M's pose
    let antipodal = -motor;
    let mid_same = motor.sclerp(&antipodal, 0.5);
    let mid_pt = mid_same.transform_point(&p_aff).to_euclidean();
    for i in 0..3 {
        assert!(
            (ref_pt[i] - mid_pt[i]).abs() < 1e-4,
            "Antipodal ScLERP must stay stationary"
        );
    }
}
