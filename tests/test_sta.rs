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
    assert!((clock_moving.ct - 1.25).abs() < 1e-5, "Expected ct' = 1.25, got {}", clock_moving.ct);

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
    assert_eq!(sp_dirac_gen.lounesto_class(1e-4), LounestoClass::Class1DiracGeneral);

    // 2. Class 2: Dirac Electron Spinor (sigma != 0, omega == 0)
    // When pseudoscalar p == 0 and E . B == 0
    let sp_electron = StaSpinor::new(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    assert_eq!(sp_electron.lounesto_class(1e-4), LounestoClass::Class2DiracElectron);

    // 3. Bilinear Covariants: Dirac probability current J^0 must be strictly positive
    let cov = sp_dirac_gen.bilinear_covariants();
    assert!(cov.j.ct > 0.0, "Probability density J^0 must be positive, got {}", cov.j.ct);

    // Current J must be timelike or lightlike: J^2 >= 0
    assert!(cov.j.interval_sq() >= -1e-5, "Current vector J must be causal");
}
