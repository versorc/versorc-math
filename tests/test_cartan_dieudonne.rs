use versorc_math::prelude::*;

#[test]
fn test_cartan_dieudonne_orthogonal_decomposition_parity() {
    // Construct a known 4x4 orthogonal matrix via Householder cascade
    let dim = 4;
    let mut initial_chain = HouseholderChain::new(dim);
    initial_chain.push(HouseholderVersor::new(vec![1.0, -1.0, 0.5, 0.0]));
    initial_chain.push(HouseholderVersor::new(vec![0.0, 1.0, 2.0, -1.0]));

    let q_matrix = initial_chain.to_dense_matrix();

    // Now decompose Q back into 1-versores using Cartan-Dieudonne
    let decomposed_chain = CartanDieudonneDecomposer::decompose_orthogonal(&q_matrix, dim);
    assert!(
        decomposed_chain.len() <= dim,
        "Cartan-Dieudonne guarantees at most n={} reflections, got {}",
        dim,
        decomposed_chain.len()
    );

    // Assert that applying the decomposed chain to any vector matches Q * x
    let test_vectors = [
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.5, -2.1, 1.3, 0.8],
    ];

    for v in &test_vectors {
        let mut v_chain = v.clone();
        decomposed_chain.apply_in_place(&mut v_chain);

        let mut v_mat = vec![0.0f32; dim];
        for r in 0..dim {
            for c in 0..dim {
                v_mat[r] += q_matrix[r * dim + c] * v[c];
            }
        }

        for i in 0..dim {
            assert!(
                (v_chain[i] - v_mat[i]).abs() < 1e-4,
                "Cartan-Dieudonne decomposed versors must match original matrix action at index {}",
                i
            );
        }
    }
}

#[test]
fn test_cartan_dieudonne_strict_upper_bound_and_identity() {
    // 1. Identity I_n requires 0 reflections
    for dim in [2, 3, 4, 6] {
        let mut id = vec![0.0f32; dim * dim];
        for i in 0..dim {
            id[i * dim + i] = 1.0;
        }
        let chain = CartanDieudonneDecomposer::decompose_orthogonal(&id, dim);
        assert_eq!(
            chain.len(),
            0,
            "Identity I_{} must require 0 reflections",
            dim
        );
    }

    // 2. Improper O(n) transformation (3 reflections in R^5): decomposed count <= 5
    let dim = 5;
    let mut ref_chain = HouseholderChain::new(dim);
    ref_chain.push(HouseholderVersor::new(vec![1.0, 2.0, -0.5, 0.3, 1.1]));
    ref_chain.push(HouseholderVersor::new(vec![0.0, -1.0, 1.5, 0.8, -0.4]));
    ref_chain.push(HouseholderVersor::new(vec![-0.7, 0.2, 0.9, -1.3, 0.6]));
    let q = ref_chain.to_dense_matrix();

    let dec = CartanDieudonneDecomposer::decompose_orthogonal(&q, dim);
    assert!(
        dec.len() <= dim,
        "Must satisfy k <= n={}, got {}",
        dim,
        dec.len()
    );
    let dec_mat = dec.to_dense_matrix();
    for i in 0..(dim * dim) {
        assert!((dec_mat[i] - q[i]).abs() < 1e-4);
    }

    // 3. Negative identity -I_3 (three flips) takes exactly 3 reflections (<= 3)
    let neg_id = vec![-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0];
    let neg_dec = CartanDieudonneDecomposer::decompose_orthogonal(&neg_id, 3);
    assert_eq!(neg_dec.len(), 3);
    assert_eq!(neg_dec.to_dense_matrix(), neg_id);
}
