use versorc_math::prelude::*;

#[test]
fn test_single_householder_1_versor_involution() {
    // A single Householder reflection must satisfy H(H(x)) == x
    let normal = vec![1.0, -2.0, 3.0, 0.5];
    let h = HouseholderVersor::new(normal);

    let original = vec![2.5, 1.1, -0.4, 3.3];
    let mut x = original.clone();

    // First reflection
    h.reflect_in_place(&mut x);
    assert_ne!(x, original);

    // Second reflection (involution)
    h.reflect_in_place(&mut x);

    for (a, b) in original.iter().zip(&x) {
        assert!((a - b).abs() < 1e-6, "Householder 1-versor must be an exact involution H^2 = I");
    }
}

#[test]
fn test_householder_chain_matches_dense_orthogonal_matrix() {
    let dim = 8;
    let mut chain = HouseholderChain::new(dim);

    // Add 4 random reflection versores
    chain.push(HouseholderVersor::new(vec![1.0, 0.0, -1.0, 2.0, 0.5, -0.5, 1.2, 0.0]));
    chain.push(HouseholderVersor::new(vec![0.2, 1.1, 0.0, -0.4, 0.0, 2.0, -1.0, 0.8]));
    chain.push(HouseholderVersor::new(vec![-0.5, 0.0, 1.5, 0.3, 0.7, -0.2, 0.0, 1.0]));

    let q = chain.to_dense_matrix();

    // Verify Q^T * Q == I_dim (Orthonormality)
    for r in 0..dim {
        for c in 0..dim {
            let mut dot = 0.0f32;
            for k in 0..dim {
                dot += q[k * dim + r] * q[k * dim + c];
            }
            let expected = if r == c { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1e-5, "Reconstructed matrix must be orthonormal at ({}, {})", r, c);
        }
    }

    // Verify vector evaluation equivalence: H_chain(x) == Q * x
    let test_vec = vec![0.5, -1.2, 2.0, 0.1, -0.4, 1.8, -0.9, 0.7];
    let mut vec_chain = test_vec.clone();
    chain.apply_in_place(&mut vec_chain);

    let mut vec_matrix = vec![0.0f32; dim];
    for r in 0..dim {
        for c in 0..dim {
            vec_matrix[r] += q[r * dim + c] * test_vec[c];
        }
    }

    for i in 0..dim {
        assert!(
            (vec_chain[i] - vec_matrix[i]).abs() < 1e-5,
            "HouseholderChain evaluation must match dense matrix multiplication at index {}",
            i
        );
    }
}
