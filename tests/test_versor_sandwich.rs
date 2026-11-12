use versorc_math::prelude::*;

#[test]
fn test_versor_sandwich_norm_and_distance_preservation() {
    let rotor = PlanarRotor::from_angle(0.785398); // 45 degrees
    let v1 = [3.0f32, 4.0f32]; // Norm = 5.0
    let v2 = [-1.0f32, 2.0f32];

    let norm_v1_orig = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
    let dist_orig = ((v1[0] - v2[0]).powi(2) + (v1[1] - v2[1]).powi(2)).sqrt();

    let v1_rot = rotor.rotate_2d(&v1);
    let v2_rot = rotor.rotate_2d(&v2);

    let norm_v1_rot = (v1_rot[0] * v1_rot[0] + v1_rot[1] * v1_rot[1]).sqrt();
    let dist_rot = ((v1_rot[0] - v2_rot[0]).powi(2) + (v1_rot[1] - v2_rot[1]).powi(2)).sqrt();

    assert!((norm_v1_rot - norm_v1_orig).abs() < 1e-6, "Versor must strictly preserve vector norms");
    assert!((dist_rot - dist_orig).abs() < 1e-6, "Versor must strictly preserve Euclidean distances");
}

#[test]
fn test_planar_spinor_bank_rope_equivalence() {
    let dim = 8;
    let pos = 5;
    let bank = PlanarSpinorBank::from_rope(dim, pos, 10000.0);

    let mut vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let norm_before: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

    bank.apply_in_place(&mut vec);

    let norm_after: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm_after - norm_before).abs() < 1e-6, "PlanarSpinorBank must be an exact L2 isometry");

    // NeoX split-half layout test
    let mut vec_neox = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let norm_neox_before: f32 = vec_neox.iter().map(|x| x * x).sum::<f32>().sqrt();
    bank.apply_in_place_neox(&mut vec_neox);
    let norm_neox_after: f32 = vec_neox.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm_neox_after - norm_neox_before).abs() < 1e-6, "NeoX RoPE must also preserve exact L2 isometry");
}
