use versorc_math::prelude::*;

#[test]
fn test_vulkan_spirv_shaders_integrity() {
    // Assert all embedded SPIR-V binaries exist, are non-empty, and adhere to 4-byte alignment
    let shaders: &[(&str, &[u8])] = &[
        ("ROPE_SPV", ROPE_SPV),
        ("SILU_MUL_SPV", SILU_MUL_SPV),
        ("RMSNORM_SPV", RMSNORM_SPV),
        ("VEC_ADD_SPV", VEC_ADD_SPV),
        ("HOT_STRATUM_E8_SPV", HOT_STRATUM_E8_SPV),
        ("PGA_TRANSFORM_SPV", PGA_TRANSFORM_SPV),
        ("CGA_TRANSFORM_SPV", CGA_TRANSFORM_SPV),
        ("GA_CL3_BATCH_SPV", GA_CL3_BATCH_SPV),
        ("POINCARE_MOBIUS_BATCH_SPV", POINCARE_MOBIUS_BATCH_SPV),
    ];

    for &(name, spv) in shaders {
        assert!(!spv.is_empty(), "SPIR-V binary {} cannot be empty", name);
        assert_eq!(
            spv.len() % 4,
            0,
            "SPIR-V binary {} length ({}) must be a multiple of 4 bytes",
            name,
            spv.len()
        );

        // Verify SPIR-V magic number (0x07230203 in Little-Endian)
        let magic = u32::from_le_bytes([spv[0], spv[1], spv[2], spv[3]]);
        assert_eq!(
            magic, 0x0723_0203,
            "Shader {} does not match standard SPIR-V magic number 0x07230203",
            name
        );
    }
}

#[test]
fn test_vulkan_push_constants_layouts() {
    let rope_pc = RopePushConstants::new(10, 32, 128, 10000.0, true);
    assert_eq!(rope_pc.pos, 10);
    assert_eq!(rope_pc.is_neox, 1);
    assert_eq!(std::mem::size_of::<RopePushConstants>(), 32);

    let silu_pc = SiluMulPushConstants::new(2048);
    assert_eq!(silu_pc.num_elements, 2048);
    assert_eq!(std::mem::size_of::<SiluMulPushConstants>(), 16);

    let rms_pc = RmsNormPushConstants::new(4096, 1e-5, 0);
    assert_eq!(rms_pc.d_model, 4096);
    assert_eq!(std::mem::size_of::<RmsNormPushConstants>(), 16);

    let add_pc = VecAddPushConstants::new(1024);
    assert_eq!(add_pc.num_elements, 1024);
    assert_eq!(std::mem::size_of::<VecAddPushConstants>(), 16);
}

#[cfg(feature = "vulkan")]
#[test]
fn test_vulkan_gpu_probe_and_shader_compilation() {
    let gpus = GpuComputeContext::probe_devices().expect("Vulkan driver must probe available GPUs");
    assert!(!gpus.is_empty(), "At least one Vulkan GPU must be discovered on this host");

    println!("Discovered {} Vulkan device(s):", gpus.len());
    for gpu in &gpus {
        println!(" - {} (Discrete: {})", gpu.name, gpu.is_discrete);
    }

    let ctx = GpuComputeContext::init_compute().expect("Vulkan compute context initialization failed");

    // Verify creating shader modules directly from embedded SPIR-V
    let rope_module = ctx.create_shader_module(ROPE_SPV).expect("ROPE shader module creation failed");
    let silu_module = ctx.create_shader_module(SILU_MUL_SPV).expect("SILU shader module creation failed");
    let rms_module = ctx.create_shader_module(RMSNORM_SPV).expect("RMSNorm shader module creation failed");

    unsafe {
        ctx.device.destroy_shader_module(rope_module, None);
        ctx.device.destroy_shader_module(silu_module, None);
        ctx.device.destroy_shader_module(rms_module, None);
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn test_vulkan_gpu_pga_tensor_execution() {
    let ctx = GpuComputeContext::init_compute().expect("Vulkan compute context initialization failed");

    // 1. Create a test tensor of 256 3D points
    let n = 256;
    let mut host_points = Vec::<[f32; 3]>::with_capacity(n);
    for i in 0..n {
        let fi = i as f32;
        host_points.push([fi * 0.1, (fi * 0.05).sin(), (fi * 0.02).cos()]);
    }

    // 2. Define a 3D PGA motor (rotation 45 deg around [1, 1, 0] axis + translation [10, -5, 20])
    let motor = PgaMotor::from_rotation_translation(
        &[1.0, 1.0, 0.0],
        std::f32::consts::FRAC_PI_4,
        &[10.0, -5.0, 20.0],
    );

    // 3. Dispatch computation on physical GPU
    let gpu_results = ctx.dispatch_pga_transform(&host_points, &motor)
        .expect("GPU PGA tensor transform dispatch failed");

    assert_eq!(gpu_results.len(), n);

    // 4. Verify against CPU reference implementation
    for i in 0..n {
        let cpu_pt = PgaPoint::new(host_points[i][0], host_points[i][1], host_points[i][2]);
        let cpu_transformed = motor.transform_point(&cpu_pt).to_euclidean();

        let gpu_pt = gpu_results[i];

        let err_x = (gpu_pt[0] - cpu_transformed[0]).abs();
        let err_y = (gpu_pt[1] - cpu_transformed[1]).abs();
        let err_z = (gpu_pt[2] - cpu_transformed[2]).abs();

        assert!(
            err_x < 1e-4 && err_y < 1e-4 && err_z < 1e-4,
            "GPU vs CPU mismatch at index {}: GPU {:?} vs CPU {:?}",
            i, gpu_pt, cpu_transformed
        );
    }

    // 5. Also verify dense vector addition on GPU
    let a: Vec<f32> = (0..512).map(|x| x as f32 * 0.5).collect();
    let b: Vec<f32> = (0..512).map(|x| 100.0 - x as f32).collect();

    let add_results = ctx.dispatch_vec_add(&a, &b).expect("GPU vector addition failed");
    for i in 0..512 {
        let expected = a[i] + b[i];
        assert!((add_results[i] - expected).abs() < 1e-5);
    }
}
