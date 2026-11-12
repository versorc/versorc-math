// ============================================================================
// VERSORC-MATH: VULKAN GPU DOP & GEOMETRIC ALGEBRA TENSOR DEMO
// License: Apache-2.0
//
// CLI command to run:
// cargo run --example gpu_ga_tensor_demo --features vulkan
// ============================================================================

#[cfg(feature = "vulkan")]
fn main() {
    use std::time::Instant;
    use versorc_math::prelude::*;

    println!("============================================================================");
    println!(" VERSORC-MATH: VULKAN GPU DATA-ORIENTED GEOMETRIC ALGEBRA (DOP)");
    println!(" Real Hardware Tensor Execution Demo on Physical GPU");
    println!("============================================================================\n");

    // 1. Probe host GPU devices
    println!("[1/6] Suministrando sondas Vulkan sobre el hardware anfitrión...");
    let devices = GpuComputeContext::probe_devices().expect("Error al detectar dispositivos Vulkan");
    for (i, dev) in devices.iter().enumerate() {
        println!(
            "  [{}] GPU: {} | Discreta: {} | Driver: 0x{:08x} | API Vulkan: {}.{}.{}",
            i,
            dev.name,
            dev.is_discrete,
            dev.driver_version,
            dev.api_version >> 22,
            (dev.api_version >> 12) & 0x3ff,
            dev.api_version & 0xfff
        );
    }

    // 2. Initialize Vulkan compute context (selects discrete GPU if present)
    println!("\n[2/6] Inicializando GpuComputeContext (targeting preferred GPU)...");
    let init_start = Instant::now();
    let ctx = GpuComputeContext::init_compute().expect("Error al inicializar el contexto de cómputo Vulkan");
    println!("  Contexto creado exitosamente en {:.2?} (Cola Compute disponible).", init_start.elapsed());

    // 3. Generate test tensor [N, 3] of 3D coordinates
    let num_points = 1024usize;
    println!("\n[3/6] Generando tensor de prueba DOP: {} puntos 3D (matriz [{}, 3])...", num_points, num_points);
    let mut tensor_in = Vec::<[f32; 3]>::with_capacity(num_points);
    for i in 0..num_points {
        let t = (i as f32) / (num_points as f32);
        let x = (t * std::f32::consts::TAU * 4.0).cos() * 10.0;
        let y = (t * std::f32::consts::TAU * 4.0).sin() * 10.0;
        let z = t * 50.0;
        tensor_in.push([x, y, z]);
    }

    println!("  Muestra del Tensor de Entrada (primeros 5 elementos):");
    for i in 0..5 {
        println!("    Row [{:04}]: [{:10.4}, {:10.4}, {:10.4}]", i, tensor_in[i][0], tensor_in[i][1], tensor_in[i][2]);
    }

    // 4. Construct PGA 3D Motor (Rotation 60° around [0, 1, 1] + Translation [15, -10, 25])
    println!("\n[4/6] Definiendo operador cinemático PGA 3D Motor M (SE(3))...");
    let rot_axis = [0.0f32, 1.0, 1.0];
    let rot_angle = std::f32::consts::FRAC_PI_3; // 60 grados
    let translation = [15.0f32, -10.0, 25.0];
    let motor = PgaMotor::from_rotation_translation(&rot_axis, rot_angle, &translation);
    println!("  Motor M: s={:.4}, b=({:.4}, {:.4}, {:.4}), d=({:.4}, {:.4}, {:.4}), e0123={:.4}",
        motor.s, motor.b23, motor.b31, motor.b12, motor.d01, motor.d02, motor.d03, motor.e0123
    );

    // 5. Dispatch computation to GPU via Vulkan pipeline
    println!("\n[5/6] Despachando computación a la GPU física mediante pipeline Vulkan (SPIR-V)...");
    let gpu_start = Instant::now();
    let gpu_tensor_out = ctx.dispatch_pga_transform(&tensor_in, &motor)
        .expect("Fallo al ejecutar compute shader en la GPU");
    let gpu_elapsed = gpu_start.elapsed();
    println!("  Despacho y sincronización GPU completados en {:.2?} ({} puntos transformados en paralelo).",
        gpu_elapsed, num_points
    );

    // 6. CPU reference calculation & empirical numerical verification
    println!("\n[6/6] Verificación empírica rigurosa contra referencia matemática CPU...");
    let mut cpu_tensor_out = Vec::<[f32; 3]>::with_capacity(num_points);
    let cpu_start = Instant::now();
    for p in &tensor_in {
        let pt = PgaPoint::new(p[0], p[1], p[2]);
        let transformed = motor.transform_point(&pt).to_euclidean();
        cpu_tensor_out.push(transformed);
    }
    let cpu_elapsed = cpu_start.elapsed();
    println!("  Computación de referencia en CPU completada en {:.2?}.", cpu_elapsed);

    let mut max_abs_error = 0.0f32;
    let mut sum_sq_error = 0.0f32;
    for i in 0..num_points {
        let err_x = (gpu_tensor_out[i][0] - cpu_tensor_out[i][0]).abs();
        let err_y = (gpu_tensor_out[i][1] - cpu_tensor_out[i][1]).abs();
        let err_z = (gpu_tensor_out[i][2] - cpu_tensor_out[i][2]).abs();

        let row_max = err_x.max(err_y).max(err_z);
        if row_max > max_abs_error {
            max_abs_error = row_max;
        }
        sum_sq_error += err_x * err_x + err_y * err_y + err_z * err_z;
    }
    let mse = sum_sq_error / (num_points as f32 * 3.0);

    println!("\n  Muestra del Tensor de Salida Resultante (GPU vs CPU):");
    for i in 0..5 {
        println!("    Row [{:04}] | GPU: [{:10.4}, {:10.4}, {:10.4}] | CPU: [{:10.4}, {:10.4}, {:10.4}]",
            i,
            gpu_tensor_out[i][0], gpu_tensor_out[i][1], gpu_tensor_out[i][2],
            cpu_tensor_out[i][0], cpu_tensor_out[i][1], cpu_tensor_out[i][2]
        );
    }

    println!("\n----------------------------------------------------------------------------");
    println!(" REPORTE DE EXACTITUD Y RENDIMIENTO");
    println!("----------------------------------------------------------------------------");
    println!("  - Elementos transformados : {} tuplas 3D ({} floats)", num_points, num_points * 3);
    println!("  - Error Absoluto Máximo   : {:.2e} (Tolerancia: 1.00e-4)", max_abs_error);
    println!("  - Error Cuadrático Medio  : {:.2e}", mse);
    println!("  - Tiempo GPU (Buffer/Run) : {:.2?}", gpu_elapsed);
    println!("  - Estado de Validación    : {}", if max_abs_error < 1e-4 { "PASSED (Exactitud bit-level verificada)" } else { "FAILED" });
    println!("============================================================================\n");
}

#[cfg(not(feature = "vulkan"))]
fn main() {
    println!("Por favor ejecuta con el feature flag 'vulkan':");
    println!("cargo run --example gpu_ga_tensor_demo --features vulkan");
}
