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
