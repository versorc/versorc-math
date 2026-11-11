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
