use versorc_math::prelude::*;

#[test]
fn test_e8_minimal_240_roots_geometry() {
    let roots = E8Lattice::generate_240_roots();
    assert_eq!(roots.len(), 240);

    let mut integer_count = 0;
    let mut half_integer_count = 0;

    for (idx, r) in roots.iter().enumerate() {
        // Invariant 1: Squared Euclidean norm of every minimal root must equal 2.0
        let norm_sq: f32 = r.iter().map(|&x| x * x).sum();
        assert!(
            (norm_sq - 2.0).abs() < 1e-6,
            "Root {} norm squared must equal 2.0, got {}",
            idx,
            norm_sq
        );

        // Classify into D8 (integers) or D8 + 1/2*1 (half-integers)
        let is_half_int = (r[0].abs() - 0.5).abs() < 1e-5;
        if is_half_int {
            half_integer_count += 1;
            // Half-integer roots must have an even number of negative signs
            let neg_count = r.iter().filter(|&&x| x < 0.0).count();
            assert_eq!(
                neg_count % 2,
                0,
                "Root {} in D8 + 1/2*1 must have even negative signs, got {}",
                idx,
                neg_count
            );
        } else {
            integer_count += 1;
            // Integer roots must have an even sum of coordinates
            let sum_int: i32 = r.iter().map(|&x| x.round() as i32).sum();
            assert_eq!(
                sum_int % 2,
                0,
                "Root {} in D8 must have even coordinate sum, got {}",
                idx,
                sum_int
            );
        }
    }

    assert_eq!(integer_count, 112, "Must contain exactly 112 integer roots in D8");
    assert_eq!(half_integer_count, 128, "Must contain exactly 128 half-integer roots");
}

#[test]
fn test_conway_sloane_closest_vector_exactness() {
    // Perturb a known D8 root by a small noise vector and verify projection returns the exact root
    let root = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let perturbed = [1.1, -0.9, 0.05, -0.05, 0.02, -0.01, 0.0, 0.03];

    let projected = E8Lattice::project_to_e8(&perturbed);
    assert_eq!(projected, root, "Conway-Sloane must recover the true nearest E8 lattice point");

    // Perturb a half-integer root
    let half_root = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]; // 0 negative signs (even)
    let half_perturbed = [0.45, 0.55, 0.48, 0.52, 0.49, 0.51, 0.50, 0.49];

    let projected_half = E8Lattice::project_to_e8(&half_perturbed);
    assert_eq!(projected_half, half_root, "Conway-Sloane must recover half-integer E8 point");
}
