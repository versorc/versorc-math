use versorc_math::prelude::*;

#[test]
fn test_clifford_basis_squares_and_anticommutativity() {
    // In Cl(3, 0, 0): e_1^2 = e_2^2 = e_3^2 = +1
    let e1 = Cl3::basis(1);
    let e2 = Cl3::basis(2);
    let e3 = Cl3::basis(3);

    // Squares
    assert_eq!((e1 * e1).scalar(), 1.0);
    assert_eq!((e2 * e2).scalar(), 1.0);
    assert_eq!((e3 * e3).scalar(), 1.0);

    // Anticommutativity: e1 * e2 + e2 * e1 == 0
    let e12 = e1 * e2;
    let e21 = e2 * e1;
    let sum = e12 + e21;
    assert!(sum.norm() < 1e-7, "e1*e2 + e2*e1 must be zero");
    assert_eq!(e12, -e21);

    // e1 * e2 * e3 is the unit pseudoscalar
    let i = e1 * e2 * e3;
    let i_sq = i * i;
    // In Cl(3, 0), I^2 = -1
    assert_eq!(i_sq.scalar(), -1.0);
}

#[test]
fn test_minkowski_spacetime_signature() {
    // In Cl(1, 3, 0) Spacetime: e1^2 = +1 (time), e2^2 = e3^2 = e4^2 = -1 (space)
    let e1 = ClMinkowski::basis(1);
    let e2 = ClMinkowski::basis(2);
    let e3 = ClMinkowski::basis(3);
    let e4 = ClMinkowski::basis(4);

    assert_eq!((e1 * e1).scalar(), 1.0);
    assert_eq!((e2 * e2).scalar(), -1.0);
    assert_eq!((e3 * e3).scalar(), -1.0);
    assert_eq!((e4 * e4).scalar(), -1.0);
}

#[test]
fn test_clifford_associativity_axiom() {
    let a = Cl3::from_vector(&[1.5, -2.0, 0.5]);
    let b = Cl3::from_vector(&[0.2, 1.0, -3.1]);
    let c = Cl3::from_vector(&[-1.0, 0.4, 2.2]);

    let lhs = (a * b) * c;
    let rhs = a * (b * c);

    let diff = (lhs - rhs).norm();
    assert!(diff < 1e-5, "Geometric product must be associative, diff = {}", diff);
}

#[test]
fn test_reversion_anti_automorphism() {
    let a = Cl3::from_vector(&[1.0, 2.0, 3.0]);
    let b = Cl3::from_vector(&[4.0, 5.0, 6.0]);

    // (A * B)~ == B~ * A~
    let lhs = (a * b).reverse();
    let rhs = b.reverse() * a.reverse();

    let diff = (lhs - rhs).norm();
    assert!(diff < 1e-6, "Reversion must satisfy (A*B)~ = B~*A~, diff = {}", diff);
}

#[test]
fn test_wedge_and_contraction_geometric_decomposition() {
    // For 1-vectors a, b: a * b = a . b + a ^ b
    let a = Cl3::from_vector(&[2.0, 1.0, 0.0]);
    let b = Cl3::from_vector(&[1.0, 3.0, 0.0]);

    let prod = a * b;
    let dot = a.scalar_product(&b);
    let wedge = a.wedge(&b);

    assert_eq!(dot, 2.0 * 1.0 + 1.0 * 3.0); // 5.0
    assert_eq!(prod.scalar(), dot);
    assert_eq!(prod.grade_projection(2), wedge);
}

#[test]
fn test_pseudoscalar_and_hodge_dual() {
    let e1 = Cl3::basis(1);
    let e2 = Cl3::basis(2);
    let e3 = Cl3::basis(3);

    let i = Cl3::pseudoscalar();
    let i_inv = Cl3::pseudoscalar_inverse().expect("Cl(3) must have invertible pseudoscalar");

    // I * I^{-1} == 1
    let unit = i * i_inv;
    assert_eq!(unit.scalar(), 1.0);
    assert!((unit - Cl3::from_scalar(1.0)).norm() < 1e-6);

    // Hodge dual *(e1 ^ e2) in Cl(3) should be e3
    let e12 = e1.wedge(&e2);
    let dual_e12 = e12.dual().expect("Dual must exist");
    // e12 * I^{-1} = e12 * (-e123) = - (e12 e123) = - (-e3) = e3
    assert!((dual_e12 - e3).norm() < 1e-6, "Dual of e12 must be e3");

    // Dual of e1: *(e1) = e1 * (-e123) = - e1 (e1 e2 e3) = - e23
    let dual_e1 = e1.dual().expect("Dual must exist");
    let e23 = e2.wedge(&e3);
    assert!((dual_e1 - (-e23)).norm() < 1e-6, "Dual of e1 must be -e23");
}

#[test]
fn test_lie_commutator_and_jacobi_identity() {
    // Lie commutator [A, B] = 1/2 (AB - BA)
    let e1 = Cl3::basis(1);
    let e2 = Cl3::basis(2);
    let e3 = Cl3::basis(3);

    let b12 = e1.wedge(&e2);
    let b23 = e2.wedge(&e3);
    let b13 = e1.wedge(&e3);

    // Commutator of bivectors closes on bivectors (so(3) Lie algebra)
    // [e12, e23] = 1/2 (e12 e23 - e23 e12) = 1/2 (e13 - (-e13)) = e13
    let comm = b12.commutator(&b23);
    assert!((comm - b13).norm() < 1e-6, "Commutator of e12 and e23 must be e13");

    // Anticommutativity: [A, B] = -[B, A]
    let comm_rev = b23.commutator(&b12);
    assert!((comm + comm_rev).norm() < 1e-6);

    // Jacobi Identity: [A, [B, C]] + [B, [C, A]] + [C, [A, B]] == 0
    let a = Cl3::from_vector(&[1.2, -0.5, 0.8]);
    let b = Cl3::from_vector(&[0.3, 2.1, -1.4]);
    let c = Cl3::from_vector(&[-0.7, 0.4, 1.9]);

    let term1 = a.commutator(&b.commutator(&c));
    let term2 = b.commutator(&c.commutator(&a));
    let term3 = c.commutator(&a.commutator(&b));
    let jacobi_sum = term1 + term2 + term3;
    assert!(jacobi_sum.norm() < 1e-5, "Jacobi sum must be zero, got {}", jacobi_sum.norm());

    // Jordan anticommutator A ⊙ B = 1/2 (AB + BA)
    let anticomm = a.anticommutator(&b);
    let expected = (a * b + b * a) * 0.5;
    assert!((anticomm - expected).norm() < 1e-6);
}

#[test]
fn test_regressive_meet_product() {
    let e1 = Cl3::basis(1);
    let e2 = Cl3::basis(2);
    let e3 = Cl3::basis(3);

    // Plane 1: xy-plane (e1 ^ e2)
    let p_xy = e1.wedge(&e2);
    // Plane 2: yz-plane (e2 ^ e3)
    let p_yz = e2.wedge(&e3);

    // Their intersection line (meet) should be along e2 (y-axis)
    let meet = p_xy.regressive_product(&p_yz).expect("Regressive product must exist");
    // Meet is parallel to e2
    let e2_component = meet.scalar_product(&e2);
    assert!(e2_component.abs() > 0.99, "Intersection line must lie along e2");
    // No other grade or component
    assert!((meet.grade_projection(1) - meet).norm() < 1e-6);
}

#[test]
fn test_cl2_and_cl3_general_multivector_inverse() {
    // 1. Test Cl(2) general multivector inverse
    let a_cl2 = Cl2 {
        coords: [2.0, -1.5, 0.8, 3.2], // scalar + e1 + e2 + e12
    };
    let inv_cl2 = a_cl2.inverse().expect("a_cl2 must be invertible");
    let prod_cl2 = a_cl2 * inv_cl2;
    assert!((prod_cl2 - Cl2::from_scalar(1.0)).norm() < 1e-5, "A * A^-1 must be 1 in Cl(2)");

    // 2. Test Cl(3) general multivector inverse with all 8 blades populated
    let a_cl3 = Cl3 {
        coords: [1.5, 0.4, -0.8, 1.1, 0.7, -1.2, 0.5, 2.0],
    };
    let inv_cl3 = a_cl3.inverse().expect("a_cl3 must be invertible");
    let prod_cl3_right = a_cl3 * inv_cl3;
    let prod_cl3_left = inv_cl3 * a_cl3;

    assert!((prod_cl3_right - Cl3::from_scalar(1.0)).norm() < 1e-5, "A * A^-1 must be 1 in Cl(3)");
    assert!((prod_cl3_left - Cl3::from_scalar(1.0)).norm() < 1e-5, "A^-1 * A must be 1 in Cl(3)");
}
