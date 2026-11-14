use versorc_math::prelude::*;

#[test]
fn test_cga_point_embedding_and_distance() {
    assert_eq!(std::mem::size_of::<CgaVector5D>(), 32);
    assert_eq!(std::mem::align_of::<CgaVector5D>(), 32);

    let p1 = CgaPoint::point(1.0, 2.0, 3.0);
    let p2 = CgaPoint::point(4.0, 6.0, 3.0);

    // Conformal points must be null vectors: P . P = 0
    let dot_self = p1.inner_product(&p1);
    assert!(dot_self.abs() < 1e-6, "CGA point must satisfy P . P = 0");

    // Euclidean distance = sqrt((4-1)^2 + (6-2)^2 + (3-3)^2) = sqrt(9 + 16) = 5.0
    let dist = p1.distance(&p2);
    assert!(
        (dist - 5.0).abs() < 1e-6,
        "Expected distance 5.0, got {}",
        dist
    );

    // Point recovery
    let coords = p1.to_euclidean();
    assert_eq!(coords, [1.0, 2.0, 3.0]);
}

#[test]
fn test_cga_sphere_radius_and_inversion() {
    // Sphere with center (0, 0, 0) and radius 5.0
    let sphere = CgaSphere::sphere(0.0, 0.0, 0.0, 5.0);
    let radius = sphere.sphere_radius().expect("Valid sphere radius");
    assert!((radius - 5.0).abs() < 1e-6);

    // Point on sphere surface (0, 3, 4) -> distance = 5.0
    let pt_on = CgaPoint::point(0.0, 3.0, 4.0);
    let dot_on = pt_on.inner_product(&sphere);
    assert!(
        dot_on.abs() < 1e-5,
        "Point on sphere must have P . S = 0, got {}",
        dot_on
    );

    // Point inside sphere (0, 0, 2) -> distance = 2.0
    let pt_in = CgaPoint::point(0.0, 0.0, 2.0);
    let dot_in = pt_in.inner_product(&sphere);
    assert!(
        dot_in > 0.0,
        "Point inside sphere must have positive inner product"
    );

    // Classical spherical inversion across sphere of radius 5:
    // Point (0, 0, 2) maps to (0, 0, 5^2 / 2) = (0, 0, 12.5)
    let inverted = pt_in.invert_in_sphere(&sphere);
    let inv_coords = inverted.to_euclidean();
    assert!((inv_coords[0] - 0.0).abs() < 1e-5);
    assert!((inv_coords[1] - 0.0).abs() < 1e-5);
    assert!(
        (inv_coords[2] - 12.5).abs() < 1e-5,
        "Expected 12.5, got {}",
        inv_coords[2]
    );
}

#[test]
fn test_cga_conformal_transformations() {
    let p = CgaPoint::point(1.0, 2.0, 3.0);

    // Translation by (10, 20, 30)
    let p_trans = p.translate(10.0, 20.0, 30.0);
    assert_eq!(p_trans.to_euclidean(), [11.0, 22.0, 33.0]);

    // Dilation by 2.0
    let p_dilated = p.dilate(2.0);
    assert_eq!(p_dilated.to_euclidean(), [2.0, 4.0, 6.0]);
}

#[test]
fn test_cga_point_pair_and_circle() {
    let p1 = CgaPoint::point(0.0, 0.0, 0.0);
    let p2 = CgaPoint::point(6.0, 8.0, 0.0);

    let pair = CgaPointPair::new(p1, p2);
    assert_eq!(pair.distance(), 10.0);
    assert_eq!(pair.midpoint(), [3.0, 4.0, 0.0]);

    // Circle from Sphere (radius 5 at origin) and Plane z = 3 (normal (0,0,1), d = -3)
    let sphere = CgaSphere::sphere(0.0, 0.0, 0.0, 5.0);
    let plane = CgaPlane::plane(0.0, 0.0, 1.0, -3.0);

    let circle = CgaCircle::from_sphere_plane_intersection(&sphere, &plane)
        .expect("Intersection circle must exist");

    // Circle center should be at (0, 0, 3) and radius = sqrt(5^2 - 3^2) = 4.0
    assert_eq!(circle.center, [0.0, 0.0, 3.0]);
    assert!((circle.radius - 4.0).abs() < 1e-5);
    assert!((circle.area() - std::f32::consts::PI * 16.0).abs() < 1e-4);
}

#[test]
fn test_cga_conformal_motor_transformations() {
    let pt = CgaPoint::point(1.0, 0.0, 0.0);

    // Motor: Dilation x2, Rotation 90 deg around Z, Translation (0, 5, 0)
    let motor = CgaConformalMotor {
        dilation_factor: 2.0, // (1, 0, 0) -> (2, 0, 0)
        rotation_axis: [0.0, 0.0, 1.0],
        rotation_angle_rad: std::f32::consts::FRAC_PI_2, // (2, 0, 0) -> (0, 2, 0)
        translation: [0.0, 5.0, 0.0],                    // (0, 2, 0) -> (0, 7, 0)
    };

    let transformed = motor.transform_point(&pt);
    let coords = transformed.to_euclidean();

    assert!((coords[0] - 0.0).abs() < 1e-5);
    assert!((coords[1] - 7.0).abs() < 1e-5);
    assert!((coords[2] - 0.0).abs() < 1e-5);

    // 1. Verify individual primitive actions
    let m_dil = CgaConformalMotor::from_dilation(2.0);
    let p_dil = m_dil.transform_point_multivector(&pt);
    assert!((p_dil.to_euclidean()[0] - 2.0).abs() < 1e-5);

    let m_rot = CgaConformalMotor::from_rotation([0.0, 0.0, 1.0], std::f32::consts::FRAC_PI_2);
    let p_rot = m_rot.transform_point_multivector(&pt);
    assert!((p_rot.to_euclidean()[1] - 1.0).abs() < 1e-5);

    let m_trans = CgaConformalMotor::from_translation(0.0, 5.0, 0.0);
    let p_trans = m_trans.transform_point_multivector(&pt);
    assert!((p_trans.to_euclidean()[1] - 5.0).abs() < 1e-5);

    // 2. Verify exact equivalence of combined motor with 5D Clifford multivector sandwich product M * P * M~
    let transformed_mv = motor.transform_point_multivector(&pt);
    let coords_mv = transformed_mv.to_euclidean();

    assert!(
        (coords_mv[0] - coords[0]).abs() < 1e-4,
        "5D Multivector sandwich must match fast-path X"
    );
    assert!(
        (coords_mv[1] - coords[1]).abs() < 1e-4,
        "5D Multivector sandwich must match fast-path Y"
    );
    assert!(
        (coords_mv[2] - coords[2]).abs() < 1e-4,
        "5D Multivector sandwich must match fast-path Z"
    );
}

#[test]
fn test_cga_sphere_inversion_scaling_invariance_and_duality() {
    let sphere = CgaSphere::sphere(1.0, -2.0, 0.5, 3.0);
    let pt = CgaPoint::point(2.5, -2.0, 0.5);
    let expected = pt.invert_in_sphere(&sphere).to_euclidean();

    for lambda in [0.5f32, 7.3f32, -2.0f32] {
        let scaled_sphere = sphere * lambda;
        let got = pt.invert_in_sphere(&scaled_sphere).to_euclidean();
        for i in 0..3 {
            assert!(
                (got[i] - expected[i]).abs() < 1e-4,
                "Inversion must be invariant under lambda={}, axis {}: got {}, expected {}",
                lambda,
                i,
                got[i],
                expected[i]
            );
        }
    }

    // Center maps to infinity, and infinity maps back to center
    let center = CgaPoint::point(1.0, -2.0, 0.5);
    let center_inv = center.invert_in_sphere(&sphere);
    assert!(center_inv.no.abs() < 1e-6);
    assert!(center_inv.ni > 0.0);

    let inf_inv = CgaPoint::infinity()
        .invert_in_sphere(&sphere)
        .to_euclidean();
    assert!((inf_inv[0] - 1.0).abs() < 1e-5);
    assert!((inf_inv[1] - (-2.0)).abs() < 1e-5);
    assert!((inf_inv[2] - 0.5).abs() < 1e-5);

    // Conformal plane reflection (S.no == 0): pi = n + d * n_inf satisfies P . pi = n.x - d = 0,
    // so z = 2 is CgaPlane::plane(0, 0, 1, 2). Reflecting (1, 2, 5) gives (1, 2, -1).
    let plane_z2 = CgaPlane::plane(0.0, 0.0, 1.0, 2.0);
    assert!(
        CgaPoint::point(1.0, 2.0, 2.0)
            .inner_product(&plane_z2)
            .abs()
            < 1e-6
    );
    let reflected = CgaPoint::point(1.0, 2.0, 5.0)
        .invert_in_sphere(&plane_z2)
        .to_euclidean();
    assert!((reflected[0] - 1.0).abs() < 1e-5);
    assert!((reflected[1] - 2.0).abs() < 1e-5);
    assert!((reflected[2] - (-1.0)).abs() < 1e-5);
}
