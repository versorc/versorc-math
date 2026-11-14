use versorc_math::prelude::*;

#[test]
fn test_poincare_origin_and_geodesic_distance() {
    let p0 = PoincarePoint::origin(4);
    assert_eq!(
        p0.lambda(),
        2.0,
        "Conformal factor at origin must equal 2.0"
    );

    let p1 = PoincarePoint::new(vec![0.5, 0.0, 0.0, 0.0]);
    let dist = p0.geodesic_distance(&p1);

    // Analytical distance d(0, x) = 2 * artanh(||x||) = ln((1 + ||x||) / (1 - ||x||))
    // For ||x|| = 0.5: ln(1.5 / 0.5) = ln(3) ~ 1.098612
    let expected = 3.0f32.ln();
    assert!(
        (dist - expected).abs() < 1e-5,
        "Poincare distance to origin must match analytical artanh formula, got {}, expected {}",
        dist,
        expected
    );
}

#[test]
fn test_poincare_mobius_addition_identity() {
    let p0 = PoincarePoint::origin(3);
    let p = PoincarePoint::new(vec![0.2, -0.3, 0.1]);

    // 0 (+) p == p
    let res = p0.mobius_add(&p);
    for (a, b) in p.coords.iter().zip(&res.coords) {
        assert!((a - b).abs() < 1e-6);
    }

    // p (+) 0 == p
    let res2 = p.mobius_add(&p0);
    for (a, b) in p.coords.iter().zip(&res2.coords) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_poincare_klein_bijection_and_einstein_midpoint() {
    let p1 = PoincarePoint::new(vec![0.4, 0.2, -0.1]);
    let k = p1.to_klein();
    let p_recon = PoincarePoint::from_klein(&k);

    for (a, b) in p1.coords.iter().zip(&p_recon.coords) {
        assert!(
            (a - b).abs() < 1e-5,
            "Klein roundtrip failed: {} vs {}",
            a,
            b
        );
    }

    // Symmetric barycenter with equal weights must converge to origin
    let pa = PoincarePoint::new(vec![0.5, 0.0, 0.0]);
    let pb = PoincarePoint::new(vec![-0.5, 0.0, 0.0]);
    let mid = PoincarePoint::einstein_midpoint(&[pa, pb], &[0.5, 0.5]);
    assert!(
        mid.norm() < 1e-4,
        "Symmetric midpoint norm must be ~0, got {}",
        mid.norm()
    );
}

#[test]
fn test_poincare_ranking_key_and_rsgd_optimization() {
    let origin = PoincarePoint::origin(3);
    let p_near = PoincarePoint::new(vec![0.1, 0.0, 0.0]);
    let p_far = PoincarePoint::new(vec![0.5, 0.0, 0.0]);

    let key_near = origin.hyperbolic_distance_key(&p_near);
    let key_far = origin.hyperbolic_distance_key(&p_far);
    let dist_near = origin.geodesic_distance(&p_near);
    let dist_far = origin.geodesic_distance(&p_far);

    assert!(dist_near < dist_far);
    assert!(
        key_near < key_far,
        "Ranking key must preserve distance order"
    );

    // RSGD step towards target
    let target = PoincarePoint::new(vec![0.6, 0.0, 0.0]);
    let mut current = PoincarePoint::origin(3);
    let initial_dist = current.geodesic_distance(&target);

    for _ in 0..30 {
        let grad: Vec<f32> = current
            .coords
            .iter()
            .zip(target.coords.iter())
            .map(|(&c, &t)| 2.0 * (c - t))
            .collect();
        current.rsgd_step(&grad, 0.1);
    }

    let final_dist = current.geodesic_distance(&target);
    assert!(
        final_dist < initial_dist,
        "RSGD should decrease distance: {} vs {}",
        final_dist,
        initial_dist
    );

    // Tangent map roundtrip
    let v_tangent = vec![0.3, -0.2, 0.1];
    let pt_exp = PoincarePoint::exp_map_zero(&v_tangent);
    let v_recon = pt_exp.log_map_zero();
    for (a, b) in v_tangent.iter().zip(&v_recon) {
        assert!(
            (a - b).abs() < 1e-4,
            "Exp/Log tangent roundtrip failed: {} vs {}",
            a,
            b
        );
    }
}
