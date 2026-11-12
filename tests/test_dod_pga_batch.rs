use versorc_math::prelude::*;

#[test]
fn test_soa_pga_motor_batch_alignment_and_transforms() {
    // SoAPgaMotorBatch16 must be exactly 512 bytes (8 x 64-byte cache lines)
    assert_eq!(std::mem::size_of::<SoAPgaMotorBatch16>(), 512);
    assert_eq!(std::mem::align_of::<SoAPgaMotorBatch16>(), 64);

    // Create 16 points along the X axis
    let mut batch_points = SoABatch16::zero();
    for i in 0..16 {
        batch_points.x[i] = i as f32;
        batch_points.y[i] = 0.0;
        batch_points.z[i] = 0.0;
    }

    // Broadcast a translation by [10, 20, 30] to all 16 motors
    let motor = PgaMotor::from_translation(10.0, 20.0, 30.0);
    let batch_motors = SoAPgaMotorBatch16::broadcast(&motor);

    // Apply 16 motor transforms in parallel
    batch_motors.transform_points(&mut batch_points);

    for i in 0..16 {
        let expected_x = (i as f32) + 10.0;
        assert!((batch_points.x[i] - expected_x).abs() < 1e-5);
        assert!((batch_points.y[i] - 20.0).abs() < 1e-5);
        assert!((batch_points.z[i] - 30.0).abs() < 1e-5);
    }
}
