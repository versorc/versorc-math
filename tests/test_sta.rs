use versorc_math::prelude::*;

#[test]
fn test_sta_vector_intervals_and_proper_time() {
    assert_eq!(std::mem::size_of::<StaVector>(), 16);
    assert_eq!(std::mem::align_of::<StaVector>(), 16);

    // Timelike 4-vector: (ct = 5, x = 3, y = 0, z = 0)
    let v_time = StaVector::new(5.0, 3.0, 0.0, 0.0);
    assert_eq!(v_time.interval_sq(), 25.0 - 9.0); // 16.0
    assert!(v_time.is_timelike());
    assert!(!v_time.is_spacelike());
    assert!(!v_time.is_lightlike());
    assert_eq!(v_time.proper_time(), Some(4.0));

    // Spacelike 4-vector: (ct = 3, x = 5, y = 0, z = 0)
    let v_space = StaVector::new(3.0, 5.0, 0.0, 0.0);
    assert_eq!(v_space.interval_sq(), 9.0 - 25.0); // -16.0
    assert!(v_space.is_spacelike());
    assert_eq!(v_space.proper_time(), None);

    // Lightlike / null 4-vector: (ct = 3, x = 0, y = 3, z = 0)
    let v_null = StaVector::new(3.0, 0.0, 3.0, 0.0);
    assert!(v_null.is_lightlike());
}

#[test]
fn test_sta_lorentz_boost_time_dilation_and_length_contraction() {
    // Boost along X with beta = 0.6 (gamma = 1 / sqrt(1 - 0.36) = 1 / 0.8 = 1.25)
    let boost = StaLorentzBoost::from_velocity(0.6, 0.0, 0.0);

    // Rest-frame clock ticking for Delta t = 1.0 at origin: (ct = 1.0, x = 0)
    let clock_rest = StaVector::new(1.0, 0.0, 0.0, 0.0);
    let clock_moving = boost.transform_vector(&clock_rest);

    // In moving frame, ct' = gamma * ct = 1.25 (time dilation)
    assert!(
        (clock_moving.ct - 1.25).abs() < 1e-5,
        "Expected ct' = 1.25, got {}",
        clock_moving.ct
    );

    // Invariant interval must be strictly preserved under Lorentz boost!
    assert!((clock_rest.interval_sq() - clock_moving.interval_sq()).abs() < 1e-5);
}

#[test]
fn test_sta_electromagnetic_faraday_bivector_invariants() {
    assert_eq!(std::mem::size_of::<StaBivector>(), 32);
    assert_eq!(std::mem::align_of::<StaBivector>(), 32);

    // Orthogonal E and B fields (e.g. plane EM wave with ||E|| = ||B||)
    let em_wave = StaBivector::new(0.0, 1.0, 0.0, 0.0, 0.0, 1.0); // E_y = 1, B_z = 1
                                                                  // I_1 = E^2 - B^2 = 1 - 1 = 0
    assert_eq!(em_wave.invariant_1(), 0.0);
    // I_2 = 2 * (E . B) = 0
    assert_eq!(em_wave.invariant_2(), 0.0);

    // Poynting vector direction: E x B = (0, 1, 0) x (0, 0, 1) = (1, 0, 0) (X axis propagation)
    let s = em_wave.poynting_vector();
    assert_eq!(s, [1.0, 0.0, 0.0]);
}

#[test]
fn test_sta_spinor_and_lounesto_classification() {
    assert_eq!(std::mem::size_of::<StaSpinor>(), 32);
    assert_eq!(std::mem::align_of::<StaSpinor>(), 32);

    // 1. Class 1: Regular Dirac Spinor (sigma != 0, omega != 0)
    let sp_dirac_gen = StaSpinor::new(2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.5);
    assert_eq!(
        sp_dirac_gen.lounesto_class(1e-4),
        LounestoClass::Class1DiracGeneral
    );

    // 2. Class 2: Dirac Electron Spinor (sigma != 0, omega == 0)
    // When pseudoscalar p == 0 and E . B == 0
    let sp_electron = StaSpinor::new(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    assert_eq!(
        sp_electron.lounesto_class(1e-4),
        LounestoClass::Class2DiracElectron
    );

    // 3. Bilinear Covariants: Dirac probability current J^0 must be strictly positive
    let cov = sp_dirac_gen.bilinear_covariants();
    assert!(
        cov.j.ct > 0.0,
        "Probability density J^0 must be positive, got {}",
        cov.j.ct
    );

    // Current J must be timelike or lightlike: J^2 >= 0
    assert!(
        cov.j.interval_sq() >= -1e-5,
        "Current vector J must be causal"
    );
}

#[test]
fn test_sta_bilinear_covariants_fierz_and_all_lounesto_classes() {
    // Compare fast-path StaSpinor::bilinear_covariants() against direct Cl(1, 3) multivector evaluation
    let sp = StaSpinor::new(1.3, -0.4, 0.7, -0.2, 0.5, -0.3, 0.9, -0.6);
    let cov = sp.bilinear_covariants();

    let g0 = ClMinkowski::basis(1);
    let g1 = ClMinkowski::basis(2);
    let g2 = ClMinkowski::basis(3);
    let g3 = ClMinkowski::basis(4);
    let pseudoscalar = g0 * g1 * g2 * g3;

    let psi = ClMinkowski::from_scalar(sp.scalar)
        + (g1 * g0) * sp.ex
        + (g2 * g0) * sp.ey
        + (g3 * g0) * sp.ez
        + (g3 * g2) * sp.bx
        + (g1 * g3) * sp.by
        + (g2 * g1) * sp.bz
        + pseudoscalar * sp.pseudoscalar;
    let psi_rev = psi.reverse();

    let psi_prod = psi * psi_rev;
    assert!((cov.sigma - psi_prod.scalar()).abs() < 1e-5);
    assert!((cov.omega - psi_prod.coords[15]).abs() < 1e-5);

    let j_mv = psi * g0 * psi_rev;
    // Project onto g_mu: x_mu = <J g_mu>_0 * g_mu^2
    assert!((cov.j.ct - (j_mv * g0).scalar()).abs() < 1e-5);
    assert!((cov.j.x - (-(j_mv * g1).scalar())).abs() < 1e-5);
    assert!((cov.j.y - (-(j_mv * g2).scalar())).abs() < 1e-5);
    assert!((cov.j.z - (-(j_mv * g3).scalar())).abs() < 1e-5);

    let k_mv = psi * g3 * psi_rev;
    assert!((cov.k.ct - (k_mv * g0).scalar()).abs() < 1e-5);
    assert!((cov.k.x - (-(k_mv * g1).scalar())).abs() < 1e-5);
    assert!((cov.k.y - (-(k_mv * g2).scalar())).abs() < 1e-5);
    assert!((cov.k.z - (-(k_mv * g3).scalar())).abs() < 1e-5);

    let s_mv = psi * (g2 * g1) * psi_rev;
    // Extract each bivector component using B * (basis)^-1:
    // (g_k g_0)^2 = +1, (g_i g_j)^2 = -1
    assert!((cov.s.ex - (s_mv * (g1 * g0)).scalar()).abs() < 1e-5);
    assert!((cov.s.ey - (s_mv * (g2 * g0)).scalar()).abs() < 1e-5);
    assert!((cov.s.ez - (s_mv * (g3 * g0)).scalar()).abs() < 1e-5);
    assert!((cov.s.bx - (-(s_mv * (g3 * g2)).scalar())).abs() < 1e-5);
    assert!((cov.s.by - (-(s_mv * (g1 * g3)).scalar())).abs() < 1e-5);
    assert!((cov.s.bz - (-(s_mv * (g2 * g1)).scalar())).abs() < 1e-5);

    // Pauli-Fierz identities:
    let r_sq = cov.sigma * cov.sigma + cov.omega * cov.omega;
    assert!((cov.j.interval_sq() - r_sq).abs() < 1e-4);
    assert!((cov.k.interval_sq() + r_sq).abs() < 1e-4);
    assert!(cov.j.dot(&cov.k).abs() < 1e-4);
    assert!((cov.s.invariant_1() + (cov.sigma * cov.sigma - cov.omega * cov.omega)).abs() < 1e-4);
    assert!((cov.s.invariant_2() + 2.0 * cov.sigma * cov.omega).abs() < 1e-4);

    // Exhaustive Lounesto classes 1..6 (including singular classes 4, 5, 6):
    assert_eq!(sp.lounesto_class(1e-4), LounestoClass::Class1DiracGeneral);
    let c2 = StaSpinor::new(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    assert_eq!(c2.lounesto_class(1e-4), LounestoClass::Class2DiracElectron);
    let c3 = StaSpinor::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0); // sigma = 0, omega = 2
    assert_eq!(
        c3.lounesto_class(1e-4),
        LounestoClass::Class3DiracPseudoscalar
    );
    let c4 = StaSpinor::new(1.0, 0.8, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0); // sigma=omega=0, S!=0, K!=0
    assert_eq!(c4.lounesto_class(1e-4), LounestoClass::Class4FlagDipole);
    let c5 = StaSpinor::new(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0); // sigma=omega=0, K=0, S!=0
    assert_eq!(
        c5.lounesto_class(1e-4),
        LounestoClass::Class5MajoranaFlagPole
    );
    let c6 = StaSpinor::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0); // sigma=omega=0, S=0, K!=0
    assert_eq!(c6.lounesto_class(1e-4), LounestoClass::Class6WeylDipole);
}
