use versorc_math::prelude::*;

#[test]
fn test_ck_multivector_cache_line_and_reversion() {
    assert_eq!(std::mem::size_of::<CkMultivector3D>(), 64, "CkMultivector3D must measure exactly 64 bytes");
    assert_eq!(std::mem::align_of::<CkMultivector3D>(), 64, "CkMultivector3D must be aligned to 64 bytes");

    let e1 = CkMultivector3D::e1();
    let e0 = CkMultivector3D::e0();
    assert_eq!(e1.c[1], 1.0);
    assert_eq!(e0.c[4], 1.0);

    // Test reversion on bivector
    let mut bivec = CkMultivector3D::zero();
    bivec.c[5] = 2.0; // e12
    let rev = bivec.reverse();
    assert_eq!(rev.c[5], -2.0, "Bivector reverse must flip sign: A~ = -A");
}

#[test]
fn test_ck_point_inner_product_and_distance() {
    let p1 = CkPoint3D::new(0.2, 0.1, 0.0);
    let p2 = CkPoint3D::new(0.4, -0.1, 0.2);

    // Inner product under Euclidean tau = 0
    let dot_0 = p1.inner_product(&p2, 0.0);
    assert_eq!(dot_0, 0.2 * 0.4 + 0.1 * (-0.1) + 0.0 * 0.2);

    // Geodesic distance under tau = 0 matches Euclidean distance
    let d0 = p1.geodesic_distance(&p2, 0.0);
    let expected_d0 = ((0.2 - 0.4f32).powi(2) + (0.1 - (-0.1f32)).powi(2) + (0.0 - 0.2f32).powi(2)).sqrt();
    assert!((d0 - expected_d0).abs() < 1e-6);

    // Geodesic distance under Hyperbolic tau = -1.0
    let d_hyp = p1.geodesic_distance(&p2, -1.0);
    assert!(d_hyp > d0, "Hyperbolic metric must expand distances relative to Euclidean");

    // Geodesic distance under Spherical tau = +1.0
    let d_sph = p1.geodesic_distance(&p2, 1.0);
    assert!(d_sph > 0.0);
}

#[test]
fn test_ck_plane_1_versor_reflection() {
    // Plane x = 0 (normal e1, distance 0)
    let plane_x = CkPlane3D::new(1.0, 0.0, 0.0, 0.0);
    let pt = CkPoint3D::new(2.5, 1.0, -0.5);

    // Reflect across x = 0 under Euclidean metric (tau = 0)
    let pt_ref_euc = plane_x.reflect_point(&pt, 0.0);
    assert!((pt_ref_euc.x - (-2.5)).abs() < 1e-6);
    assert!((pt_ref_euc.y - 1.0).abs() < 1e-6);
    assert!((pt_ref_euc.z - (-0.5)).abs() < 1e-6);

    // Reflect across x = 0 under Hyperbolic metric (tau = -1.0)
    let pt_ref_hyp = plane_x.reflect_point(&pt, -1.0);
    assert!((pt_ref_hyp.x - (-2.5)).abs() < 1e-6);
    assert!((pt_ref_hyp.y - 1.0).abs() < 1e-6);

    // Involutory property of 1-versor: H * (H * P) == P
    let pt_double_ref = plane_x.reflect_point(&pt_ref_hyp, -1.0);
    assert!((pt_double_ref.x - pt.x).abs() < 1e-5);
    assert!((pt_double_ref.y - pt.y).abs() < 1e-5);
    assert!((pt_double_ref.z - pt.z).abs() < 1e-5);
}

#[test]
fn test_ck_line_exterior_join_and_plucker() {
    let p1 = CkPoint3D::new(1.0, 0.0, 0.0);
    let p2 = CkPoint3D::new(0.0, 1.0, 0.0);

    // Exterior wedge Join: L = P1 ^ P2
    let line = p1.join(&p2);

    // Verify Plücker geometric condition: v . m == 0
    let plucker = line.plucker_inner();
    assert!(plucker.abs() < 1e-6, "Plucker condition v . m must be zero: got {}", plucker);

    // Line direction along z: e1 ^ e2 = e12
    assert_eq!(line.vz, 1.0);
}

#[test]
fn test_ck_motor_universal_isometries() {
    let pt = CkPoint3D::new(1.0, 0.0, 0.0);

    // 1. Pure 90-degree rotation around Z axis
    let motor_rot = CkMotor3D::from_rotation([0.0, 0.0, 1.0], std::f32::consts::FRAC_PI_2);
    let pt_rot = motor_rot.transform_point(&pt, 0.0);
    assert!((pt_rot.x - 0.0).abs() < 1e-5);
    assert!((pt_rot.y - 1.0).abs() < 1e-5);
    assert!((pt_rot.z - 0.0).abs() < 1e-5);

    // 2. Pure Euclidean Translation (tau = 0.0) along X axis by 3.0 units
    let motor_trans_euc = CkMotor3D::from_translation([1.0, 0.0, 0.0], 3.0, 0.0);
    let pt_trans = motor_trans_euc.transform_point(&pt, 0.0);
    assert!((pt_trans.x - 4.0).abs() < 1e-4);

    // 3. Hyperbolic Translation / Boost (tau = -1.0) along X axis
    let origin = CkPoint3D::origin();
    let motor_trans_hyp = CkMotor3D::from_translation([1.0, 0.0, 0.0], 0.5, -1.0);
    let pt_hyp = motor_trans_hyp.transform_point(&origin, -1.0);
    assert!(pt_hyp.x > 0.0 && pt_hyp.x < 1.0, "Hyperbolic translation must stay inside unit ball");

    // 4. Spherical Translation (tau = +1.0) along X axis
    let motor_trans_sph = CkMotor3D::from_translation([1.0, 0.0, 0.0], 0.5, 1.0);
    let pt_sph = motor_trans_sph.transform_point(&origin, 1.0);
    assert!(pt_sph.x > 0.0);
}
