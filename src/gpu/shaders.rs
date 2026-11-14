// ============================================================================
// VERSORC-MATH: VULKAN & SPIR-V COMPUTE SHADERS FOR GEOMETRIC OPERATIONS
// License: Apache-2.0
//
// Precompiled SPIR-V 1.3+ (Vulkan 1.1 target) bytecode for GPU-accelerated versor,
// Clifford, conformal, hyperbolic, and dense compute operations. Can be embedded
// directly by any Vulkan runtime or external compute pipeline.
// ============================================================================

pub const ROPE_SPV: &[u8] = include_bytes!("../../shaders/rope.spv");
pub const SILU_MUL_SPV: &[u8] = include_bytes!("../../shaders/silu_mul.spv");
pub const RMSNORM_SPV: &[u8] = include_bytes!("../../shaders/rmsnorm.spv");
pub const VEC_ADD_SPV: &[u8] = include_bytes!("../../shaders/vec_add.spv");
pub const PGA_TRANSFORM_SPV: &[u8] = include_bytes!("../../shaders/pga_transform.spv");
pub const CGA_TRANSFORM_SPV: &[u8] = include_bytes!("../../shaders/cga_transform.spv");
pub const GA_CL3_BATCH_SPV: &[u8] = include_bytes!("../../shaders/ga_cl3_batch.spv");
pub const POINCARE_MOBIUS_BATCH_SPV: &[u8] =
    include_bytes!("../../shaders/poincare_mobius_batch.spv");

/// Push constant payload for Planar Spinor Bank (RoPE) compute shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct RopePushConstants {
    pub pos: u32,
    pub n_heads: u32,
    pub head_dim: u32,
    pub rope_theta: f32,
    pub is_neox: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl RopePushConstants {
    pub fn new(pos: u32, n_heads: u32, head_dim: u32, rope_theta: f32, is_neox: bool) -> Self {
        Self {
            pos,
            n_heads,
            head_dim,
            rope_theta,
            is_neox: if is_neox { 1 } else { 0 },
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Push constant payload for SwiGLU / SiLU-Mul fused activation compute shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct SiluMulPushConstants {
    pub num_elements: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl SiluMulPushConstants {
    pub fn new(num_elements: u32) -> Self {
        Self {
            num_elements,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Push constant payload for RMSNorm compute shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct RmsNormPushConstants {
    pub d_model: u32,
    pub eps: f32,
    pub gamma_offset_f32: u32,
    pub _pad1: u32,
}

impl RmsNormPushConstants {
    pub fn new(d_model: u32, eps: f32, gamma_offset_f32: u32) -> Self {
        Self {
            d_model,
            eps,
            gamma_offset_f32,
            _pad1: 0,
        }
    }
}

/// Push constant payload for Vector Addition (Residual Stream Accumulation) compute shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct VecAddPushConstants {
    pub num_elements: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl VecAddPushConstants {
    pub fn new(num_elements: u32) -> Self {
        Self {
            num_elements,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Push constant payload for 3D Projective Geometric Algebra (PGA) rigid tensor transform on GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct PgaTransformPushConstants {
    pub num_points: u32,
    pub motor_s: f32,
    pub motor_b23: f32,
    pub motor_b31: f32,
    pub motor_b12: f32,
    pub motor_d01: f32,
    pub motor_d02: f32,
    pub motor_d03: f32,
    pub motor_e0123: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl PgaTransformPushConstants {
    pub fn new(num_points: u32, motor: &crate::pga::PgaMotor) -> Self {
        Self {
            num_points,
            motor_s: motor.s,
            motor_b23: motor.b23,
            motor_b31: motor.b31,
            motor_b12: motor.b12,
            motor_d01: motor.d01,
            motor_d02: motor.d02,
            motor_d03: motor.d03,
            motor_e0123: motor.e0123,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Push constant payload for 3D Conformal Geometric Algebra (CGA) spherical inversion on GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct CgaTransformPushConstants {
    pub num_points: u32,
    pub op_type: u32,
    pub sphere_cx: f32,
    pub sphere_cy: f32,
    pub sphere_cz: f32,
    pub sphere_radius: f32,
    pub param_lambda: f32,
    pub param_dy: f32,
    pub param_dz: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl CgaTransformPushConstants {
    pub fn spherical_inversion(num_points: u32, center: [f32; 3], radius: f32) -> Self {
        Self {
            num_points,
            op_type: 0,
            sphere_cx: center[0],
            sphere_cy: center[1],
            sphere_cz: center[2],
            sphere_radius: radius,
            param_lambda: 0.0,
            param_dy: 0.0,
            param_dz: 0.0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    pub fn dilation(num_points: u32, factor: f32) -> Self {
        Self {
            num_points,
            op_type: 1,
            sphere_cx: 0.0,
            sphere_cy: 0.0,
            sphere_cz: 0.0,
            sphere_radius: 0.0,
            param_lambda: factor,
            param_dy: 0.0,
            param_dz: 0.0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    pub fn translation(num_points: u32, shift: [f32; 3]) -> Self {
        Self {
            num_points,
            op_type: 2,
            sphere_cx: 0.0,
            sphere_cy: 0.0,
            sphere_cz: 0.0,
            sphere_radius: 0.0,
            param_lambda: shift[0],
            param_dy: shift[1],
            param_dz: shift[2],
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Push constant payload for unified Cl(3, 0) multivector batch operations on GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct GaCl3BatchPushConstants {
    pub num_elements: u32,
    pub op_code: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl GaCl3BatchPushConstants {
    pub fn new(num_elements: u32, op_code: u32) -> Self {
        Self {
            num_elements,
            op_code,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Push constant payload for Poincare hyperbolic and conformal batch operations on GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct PoincareBatchPushConstants {
    pub num_elements: u32,
    pub op_code: u32,
    pub sphere_cx: f32,
    pub sphere_cy: f32,
    pub sphere_cz: f32,
    pub sphere_radius: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl PoincareBatchPushConstants {
    pub fn new(
        num_elements: u32,
        op_code: u32,
        sphere_center: [f32; 3],
        sphere_radius: f32,
    ) -> Self {
        Self {
            num_elements,
            op_code,
            sphere_cx: sphere_center[0],
            sphere_cy: sphere_center[1],
            sphere_cz: sphere_center[2],
            sphere_radius,
            _pad0: 0,
            _pad1: 0,
        }
    }
}
