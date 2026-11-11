use versorc_math::prelude::*;

#[test]
fn test_poincare_origin_and_geodesic_distance() {
    let p0 = PoincarePoint::origin(4);
    assert_eq!(p0.lambda(), 2.0, "Conformal factor at origin must equal 2.0");

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
