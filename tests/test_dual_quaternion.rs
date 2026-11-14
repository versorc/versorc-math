use versorc_math::prelude::*;

#[test]
fn test_dual_quaternion_rigid_transformation_and_inversion() {
    assert_eq!(std::mem::size_of::<DualQuaternion>(), 32);
    assert_eq!(std::mem::align_of::<DualQuaternion>(), 32);

    let rot = Quaternion::from_axis_angle(&[0.0, 1.0, 0.0], std::f32::consts::FRAC_PI_2);
    let trans = [10.0, 20.0, 30.0];

    let dq = DualQuaternion::from_rotation_translation(&rot, &trans);

    // Decomposing back into rotation and translation
    let (rec_rot, rec_trans) = dq.to_rotation_translation();
    assert!((rec_rot.w - rot.w).abs() < 1e-6);
    assert!((rec_rot.y - rot.y).abs() < 1e-6);
    assert!((rec_trans[0] - 10.0).abs() < 1e-5);
    assert!((rec_trans[1] - 20.0).abs() < 1e-5);
    assert!((rec_trans[2] - 30.0).abs() < 1e-5);

    // Transform 3D point (1, 0, 0):
    // Rotated 90 deg around Y: (0, 0, -1)
    // Translated by (10, 20, 30): (10, 20, 29)
    let pt = [1.0, 0.0, 0.0];
    let pt_t = dq.transform_point(&pt);
    assert!(
        (pt_t[0] - 10.0).abs() < 1e-5,
        "Expected 10.0, got {}",
        pt_t[0]
    );
    assert!(
        (pt_t[1] - 20.0).abs() < 1e-5,
        "Expected 20.0, got {}",
        pt_t[1]
    );
    assert!(
        (pt_t[2] - 29.0).abs() < 1e-5,
        "Expected 29.0, got {}",
        pt_t[2]
    );

    // Vector rotation (no translation applied)
    let v_t = dq.transform_vector(&pt);
    assert!((v_t[0] - 0.0).abs() < 1e-5);
    assert!((v_t[1] - 0.0).abs() < 1e-5);
    assert!((v_t[2] - (-1.0)).abs() < 1e-5);

    // Inverse dual quaternion
    let inv_dq = dq.inverse();
    let pt_restored = inv_dq.transform_point(&pt_t);
    assert!((pt_restored[0] - 1.0).abs() < 1e-5);
    assert!((pt_restored[1] - 0.0).abs() < 1e-5);
    assert!((pt_restored[2] - 0.0).abs() < 1e-5);
}

#[test]
fn test_dual_quaternion_dlb_and_sclerp() {
    let dq1 = DualQuaternion::from_translation(0.0, 0.0, 0.0);
    let dq2 = DualQuaternion::from_translation(10.0, 20.0, 30.0);

    // DLB blending with weights 0.5, 0.5
    let blended = DualQuaternion::dlb(&[0.5, 0.5], &[dq1, dq2]);
    let (_, t_blend) = blended.to_rotation_translation();
    assert!((t_blend[0] - 5.0).abs() < 1e-4);
    assert!((t_blend[1] - 10.0).abs() < 1e-4);
    assert!((t_blend[2] - 15.0).abs() < 1e-4);

    // ScLERP at t = 0.5
    let sclerped = dq1.sclerp(&dq2, 0.5);
    let (_, t_sclerp) = sclerped.to_rotation_translation();
    assert!((t_sclerp[0] - 5.0).abs() < 1e-4);
    assert!((t_sclerp[1] - 10.0).abs() < 1e-4);
    assert!((t_sclerp[2] - 15.0).abs() < 1e-4);
}

#[test]
fn test_dual_quaternion_pga_motor_bijection() {
    let rot = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], std::f32::consts::FRAC_PI_3);
    let trans = [5.0, -2.0, 7.0];
    let dq = DualQuaternion::from_rotation_translation(&rot, &trans);

    // Convert to PgaMotor
    let motor: PgaMotor = dq.into();

    // Convert back to DualQuaternion
    let dq_back: DualQuaternion = motor.into();

    assert!((dq.real.w - dq_back.real.w).abs() < 1e-6);
    assert!((dq.real.x - dq_back.real.x).abs() < 1e-6);
    assert!((dq.real.y - dq_back.real.y).abs() < 1e-6);
    assert!((dq.real.z - dq_back.real.z).abs() < 1e-6);

    assert!((dq.dual.w - dq_back.dual.w).abs() < 1e-6);
    assert!((dq.dual.x - dq_back.dual.x).abs() < 1e-6);
    assert!((dq.dual.y - dq_back.dual.y).abs() < 1e-6);
    assert!((dq.dual.z - dq_back.dual.z).abs() < 1e-6);

    // Ensure motor and dual quaternion produce identical transformation of test point
    let pt = [3.0, 4.0, 5.0];
    let pt_pga = PgaPoint::new(pt[0], pt[1], pt[2]);

    let res_dq = dq.transform_point(&pt);
    let res_motor = motor.transform_point(&pt_pga).to_euclidean();

    assert!((res_dq[0] - res_motor[0]).abs() < 1e-5);
    assert!((res_dq[1] - res_motor[1]).abs() < 1e-5);
    assert!((res_dq[2] - res_motor[2]).abs() < 1e-5);
}
