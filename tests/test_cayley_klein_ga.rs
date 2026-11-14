use versorc_math::prelude::*;

#[test]
fn test_ck_multivector_cache_line_and_reversion() {
    assert_eq!(
        std::mem::size_of::<CkMultivector3D>(),
        64,
        "CkMultivector3D must measure exactly 64 bytes"
    );
    assert_eq!(
        std::mem::align_of::<CkMultivector3D>(),
        64,
        "CkMultivector3D must be aligned to 64 bytes"
    );

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
    let expected_d0 =
        ((0.2 - 0.4f32).powi(2) + (0.1 - (-0.1f32)).powi(2) + (0.0 - 0.2f32).powi(2)).sqrt();
    assert!((d0 - expected_d0).abs() < 1e-6);

    // Geodesic distance under Hyperbolic tau = -1.0
    let d_hyp = p1.geodesic_distance(&p2, -1.0);
    assert!(
        d_hyp > d0,
        "Hyperbolic metric must expand distances relative to Euclidean"
    );

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
    assert!(
        plucker.abs() < 1e-6,
        "Plucker condition v . m must be zero: got {}",
        plucker
    );

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
    assert!(
        pt_hyp.x > 0.0 && pt_hyp.x < 1.0,
        "Hyperbolic translation must stay inside unit ball"
    );

    // 4. Spherical Translation (tau = +1.0) along X axis
    let motor_trans_sph = CkMotor3D::from_translation([1.0, 0.0, 0.0], 0.5, 1.0);
    let pt_sph = motor_trans_sph.transform_point(&origin, 1.0);
    assert!(pt_sph.x > 0.0);
}

#[test]
fn test_ck_multivector_product_and_composite_motor_isometries() {
    let pt_start = CkPoint3D::new(0.2, -0.15, 0.1);
    let pt_other = CkPoint3D::new(-0.1, 0.25, -0.05);

    for tau in [-1.0f32, 0.0f32, 1.0f32] {
        // 1. Multivector algebra checks: e0^2 = tau, e1^2 = 1, e12^2 = -1, associativity
        let e0 = CkMultivector3D::e0();
        let e1 = CkMultivector3D::e1();
        let e2 = CkMultivector3D::e2();
        let e3 = CkMultivector3D::e3();
        assert!((e0.geometric_product(&e0, tau).c[0] - tau).abs() < 1e-6);
        assert!((e1.geometric_product(&e1, tau).c[0] - 1.0).abs() < 1e-6);
        let e12 = e1.wedge(&e2);
        assert!((e12.geometric_product(&e12, tau).c[0] - (-1.0)).abs() < 1e-6);

        let assoc_lhs = (e0 + e1 * 2.0)
            .geometric_product(&(e2 - e3 * 0.5), tau)
            .geometric_product(&(e3 + e0 * 0.3), tau);
        let assoc_rhs = (e0 + e1 * 2.0).geometric_product(
            &(e2 - e3 * 0.5).geometric_product(&(e3 + e0 * 0.3), tau),
            tau,
        );
        assert!((assoc_lhs - assoc_rhs).euclidean_norm() < 1e-5);

        // 2. Single-axis geodesic distance matches delta in all three curvatures
        let m_trans = CkMotor3D::from_translation([1.0, 0.0, 0.0], 0.5, tau);
        let origin_moved = m_trans.transform_point(&CkPoint3D::origin(), tau);
        let d_origin = CkPoint3D::origin().geodesic_distance(&origin_moved, tau);
        assert!(
            (d_origin - 0.5).abs() < 1e-4,
            "tau={}: expected geodesic displacement 0.5, got {}",
            tau,
            d_origin
        );

        // 3. Composite motor M = T2 * R * T1 (non-zero pseudoscalar M.p and non-origin P):
        // M(P) must equal T2(R(T1(P))) and strictly preserve geodesic distances.
        let t1 = CkMotor3D::from_translation([0.6, 0.8, 0.0], 0.35, tau);
        let r = CkMotor3D::from_rotation([0.0, 0.0, 1.0], 0.7);
        let t2 = CkMotor3D::from_translation([0.0, 0.0, 1.0], 0.25, tau);

        let composed = t2.geometric_product(&r, tau).geometric_product(&t1, tau);
        assert!(
            composed.p.abs() > 1e-3,
            "Composite skew motor must generate non-zero pseudoscalar p"
        );

        let via_seq = t2.transform_point(
            &r.transform_point(&t1.transform_point(&pt_start, tau), tau),
            tau,
        );
        let via_comp = composed.transform_point(&pt_start, tau);
        assert!((via_seq.x - via_comp.x).abs() < 1e-4);
        assert!((via_seq.y - via_comp.y).abs() < 1e-4);
        assert!((via_seq.z - via_comp.z).abs() < 1e-4);

        // Distance preservation (isometry) on non-origin pair
        let d_before = pt_start.geodesic_distance(&pt_other, tau);
        let q_a = composed.transform_point(&pt_start, tau);
        let q_b = composed.transform_point(&pt_other, tau);
        let d_after = q_a.geodesic_distance(&q_b, tau);
        assert!(
            (d_before - d_after).abs() < 1e-4,
            "tau={}: composite motor must preserve geodesic distance ({} vs {})",
            tau,
            d_before,
            d_after
        );

        // 4. Motor inverse and Study normalization
        let inv = composed.inverse(tau);
        let id = composed.geometric_product(&inv, tau);
        assert!((id.to_multivector() - CkMultivector3D::scalar(1.0)).euclidean_norm() < 1e-4);

        let restored = inv.transform_point(&via_comp, tau);
        assert!((restored.x - pt_start.x).abs() < 1e-4);
        assert!((restored.y - pt_start.y).abs() < 1e-4);
        assert!((restored.z - pt_start.z).abs() < 1e-4);

        let perturbed = CkMotor3D {
            s: composed.s * 2.0,
            rot: [
                composed.rot[0] * 2.0,
                composed.rot[1] * 2.0,
                composed.rot[2] * 2.0,
            ],
            trans: [
                composed.trans[0] * 2.0,
                composed.trans[1] * 2.0,
                composed.trans[2] * 2.0,
            ],
            p: composed.p * 2.0 + 0.15,
        };
        let normed = perturbed.normalized(tau);
        let check = normed.geometric_product(&normed.reverse(), tau);
        assert!((check.to_multivector() - CkMultivector3D::scalar(1.0)).euclidean_norm() < 1e-4);
    }
}
