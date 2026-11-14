use versorc_math::prelude::*;

#[test]
fn test_cayley_klein_tau_metric_transitions() {
    let x = [0.2f32, 0.1, 0.0];
    let y = [0.4f32, -0.1, 0.2];

    // 1. Hyperbolic branch (tau = -1.0)
    let space_hyp = CayleyKleinSpace::new(-1.0);
    let d_hyp = space_hyp.geodesic_distance(&x, &y);
    assert!(d_hyp > 0.0);

    // 2. Euclidean branch (tau = 0.0)
    let space_euc = CayleyKleinSpace::new(0.0);
    let d_euc = space_euc.geodesic_distance(&x, &y);
    let expected_euc =
        ((0.2 - 0.4f32).powi(2) + (0.1 - (-0.1f32)).powi(2) + (0.0 - 0.2f32).powi(2)).sqrt();
    assert!((d_euc - expected_euc).abs() < 1e-6);

    // 3. Spherical / Elliptic branch (tau = +1.0)
    let space_ell = CayleyKleinSpace::new(1.0);
    let d_ell = space_ell.geodesic_distance(&x, &y);
    assert!(d_ell > 0.0);

    // Invariants: All metrics are strictly positive for non-coincident points
    assert!(d_hyp > 0.0, "Hyperbolic distance must be positive");
    assert!(d_euc > 0.0, "Euclidean distance must be positive");
    assert!(d_ell > 0.0, "Elliptic distance must be positive");

    // Hyperbolic metric expands distances relative to Euclidean coordinate distance
    assert!(
        d_hyp > d_euc,
        "Hyperbolic metric should expand distances compared to Euclidean: {} vs {}",
        d_hyp,
        d_euc
    );

    // Conformal projection test
    let proj = space_hyp.project_to_euclidean(&x);
    assert_eq!(proj.len(), 3);
}

#[test]
fn test_eisenstein_a3_kissing_number_and_conversions() {
    // 1. Validate all 12 kissing root neighbors have discrete squared distance D_{A_3}^2 == 1
    let neighbors = EisensteinA3Lattice::generate_12_neighbors();
    assert_eq!(neighbors.len(), EISENSTEIN_KISSING_NUMBER);

    for n in &neighbors {
        let d_sq = EisensteinA3Lattice::distance_sq(n[0], n[1], n[2]);
        assert_eq!(
            d_sq, 1,
            "Nearest neighbor {:?} must have distance squared == 1",
            n
        );
    }

    // 2. Continuous <-> Discrete roundtrip
    let origin_coords = EisensteinA3Lattice::to_euclidean(0, 0, 0);
    assert_eq!(origin_coords, [0.0, 0.0, 0.0]);

    let (u, v, w) = (2, -1, 3);
    let pt3d = EisensteinA3Lattice::to_euclidean(u, v, w);
    let rounded = EisensteinA3Lattice::from_euclidean(pt3d[0], pt3d[1], pt3d[2]);
    assert_eq!(
        rounded,
        [u, v, w],
        "Eisenstein lattice roundtrip failed: {:?} vs {:?}",
        rounded,
        [u, v, w]
    );

    // 3. Poincaré projection
    let p3d = EisensteinA3Lattice::to_poincare(u, v, w, 0.1);
    let norm_p3d = (p3d[0] * p3d[0] + p3d[1] * p3d[1] + p3d[2] * p3d[2]).sqrt();
    assert!(
        norm_p3d < 1.0,
        "Eisenstein Poincaré projection must strictly reside inside unit ball: norm={}",
        norm_p3d
    );
}

#[test]
fn test_data_oriented_design_64byte_cache_alignment() {
    // 1. Memory alignment invariants: strictly 64 bytes (1 L1d cache line)
    assert_eq!(std::mem::size_of::<Universal64Accumulator>(), 64);
    assert_eq!(std::mem::align_of::<Universal64Accumulator>(), 64);

    assert_eq!(std::mem::size_of::<Universal64Chunk>(), 64);
    assert_eq!(std::mem::align_of::<Universal64Chunk>(), 64);

    // 2. SoABatch16: 4 contiguous 64-byte cache lines = 256 bytes
    assert_eq!(std::mem::size_of::<SoABatch16>(), 256);
    assert_eq!(std::mem::align_of::<SoABatch16>(), 64);

    // 3. Test balanced horizontal reduction on 64-byte accumulator
    let mut acc = Universal64Accumulator::zero();
    for (i, lane) in acc.lanes.iter_mut().enumerate() {
        *lane = (i + 1) as f32; // Sum 1..16 = 136
    }
    assert_eq!(acc.horizontal_sum(), 136.0f32);
}

#[test]
fn test_soa_batch_cache_line_operations() {
    // Create two batches of 16 3D points
    let mut pts_a = [[0.0f32; 3]; 16];
    let mut pts_b = [[0.0f32; 3]; 16];

    for i in 0..16 {
        pts_a[i] = [0.1 * (i as f32), 0.05 * (i as f32), -0.02 * (i as f32)];
        pts_b[i] = [0.05 * (i as f32), -0.05 * (i as f32), 0.04 * (i as f32)];
    }

    let batch_a = SoABatch16::from_aos_3d(&pts_a);
    let batch_b = SoABatch16::from_aos_3d(&pts_b);

    // 1. Batch geodesic distance under Hyperbolic Cayley-Klein space (tau = -0.5)
    let space_hyp = CayleyKleinSpace::new(-0.5);
    let dist_accum = batch_a.batch_geodesic_distance(&batch_b, &space_hyp);
    assert_eq!(std::mem::size_of_val(&dist_accum), 64);

    for (i, &val) in dist_accum.lanes.iter().enumerate().skip(1) {
        assert!(val > 0.0, "Geodesic distance lane {} must be positive", i);
    }

    // 2. Batch Eisenstein A3 conversion
    let (u_arr, v_arr, w_arr) = batch_a.batch_to_eisenstein_a3();
    assert_eq!(u_arr.len(), 16);
    assert_eq!(v_arr.len(), 16);
    assert_eq!(w_arr.len(), 16);

    // 3. Batch planar spinor rotation in cache
    let mut batch_rot = batch_a;
    let cos_45 = std::f32::consts::FRAC_1_SQRT_2;
    let sin_45 = std::f32::consts::FRAC_1_SQRT_2;
    batch_rot.batch_planar_spinor_rotate(cos_45, sin_45);

    // Check L2 norm preservation for lane 1
    let norm_before = (batch_a.x[1].powi(2) + batch_a.y[1].powi(2)).sqrt();
    let norm_after = (batch_rot.x[1].powi(2) + batch_rot.y[1].powi(2)).sqrt();
    assert!(
        (norm_after - norm_before).abs() < 1e-6,
        "Planar spinor rotation in batch must preserve norm"
    );
}
