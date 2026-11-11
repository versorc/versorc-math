use versorc_math::prelude::*;

#[test]
fn test_hamilton_fundamental_formula_1843() {
    // i^2 = j^2 = k^2 = i*j*k = -1
    let i = Quaternion::new(0.0, 1.0, 0.0, 0.0);
    let j = Quaternion::new(0.0, 0.0, 1.0, 0.0);
    let k = Quaternion::new(0.0, 0.0, 0.0, 1.0);

    assert_eq!((i * i).w, -1.0);
    assert_eq!((j * j).w, -1.0);
    assert_eq!((k * k).w, -1.0);

    let ijk = i * j * k;
    assert_eq!(ijk.w, -1.0);
    assert_eq!(ijk.x, 0.0);
    assert_eq!(ijk.y, 0.0);
    assert_eq!(ijk.z, 0.0);

    // ij = k, jk = i, ki = j
    assert_eq!(i * j, k);
    assert_eq!(j * k, i);
    assert_eq!(k * i, j);

    // ji = -k, kj = -i, ik = -j
    assert_eq!(j * i, -k);
    assert_eq!(k * j, -i);
    assert_eq!(i * k, -j);
}

#[test]
fn test_quaternion_3d_vector_rotations() {
    // Rotate vector [1, 0, 0] by 90 deg around Z-axis -> must equal [0, 1, 0]
    let z_axis = [0.0, 0.0, 1.0];
    let q_rot_z = Quaternion::from_axis_angle(&z_axis, std::f32::consts::FRAC_PI_2);
    let v_x = [1.0, 0.0, 0.0];
    let v_rotated = q_rot_z.rotate_vector3(&v_x);

    assert!((v_rotated[0] - 0.0).abs() < 1e-6);
    assert!((v_rotated[1] - 1.0).abs() < 1e-6);
    assert!((v_rotated[2] - 0.0).abs() < 1e-6);

    // Rotate vector [0, 1, 0] by 90 deg around X-axis -> must equal [0, 0, 1]
    let x_axis = [1.0, 0.0, 0.0];
    let q_rot_x = Quaternion::from_axis_angle(&x_axis, std::f32::consts::FRAC_PI_2);
    let v_y = [0.0, 1.0, 0.0];
    let v_rotated_x = q_rot_x.rotate_vector3(&v_y);

    assert!((v_rotated_x[0] - 0.0).abs() < 1e-6);
    assert!((v_rotated_x[1] - 0.0).abs() < 1e-6);
    assert!((v_rotated_x[2] - 1.0).abs() < 1e-6);
}

#[test]
fn test_slerp_geodesic_interpolation_endpoints_and_midpoint() {
    let q0 = Quaternion::identity();
    let q1 = Quaternion::from_axis_angle(&[0.0, 1.0, 0.0], std::f32::consts::FRAC_PI_2);

    // Endpoint t = 0
    let s0 = Quaternion::slerp(q0, q1, 0.0);
    assert!((s0.w - q0.w).abs() < 1e-5);
    assert!((s0.y - q0.y).abs() < 1e-5);

    // Endpoint t = 1
    let s1 = Quaternion::slerp(q0, q1, 1.0);
    assert!((s1.w - q1.w).abs() < 1e-5);
    assert!((s1.y - q1.y).abs() < 1e-5);

    // Midpoint t = 0.5 (must be 45 deg rotation)
    let s_mid = Quaternion::slerp(q0, q1, 0.5);
    let q_expected = Quaternion::from_axis_angle(&[0.0, 1.0, 0.0], std::f32::consts::FRAC_PI_4);
    assert!((s_mid.w - q_expected.w).abs() < 1e-5);
    assert!((s_mid.y - q_expected.y).abs() < 1e-5);
}
