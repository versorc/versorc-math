// ============================================================================
// VERSORC-MATH: VULKAN & SPIR-V COMPUTE SHADERS FOR GEOMETRIC OPERATIONS
// License: Apache-2.0
//
// Precompiled SPIR-V 1.3+ bytecode for GPU-accelerated versor, lattice,
// and dense vector operations. Can be embedded directly by any Vulkan runtime
// (including `versorc-tiering` and external graphics/compute pipelines).
// ============================================================================

pub const ROPE_SPV: &[u8] = include_bytes!("../../shaders/rope.spv");
pub const SILU_MUL_SPV: &[u8] = include_bytes!("../../shaders/silu_mul.spv");
pub const RMSNORM_SPV: &[u8] = include_bytes!("../../shaders/rmsnorm.spv");
pub const VEC_ADD_SPV: &[u8] = include_bytes!("../../shaders/vec_add.spv");
pub const HOT_STRATUM_E8_SPV: &[u8] = include_bytes!("../../shaders/hot_stratum_e8.spv");
pub const PGA_TRANSFORM_SPV: &[u8] = include_bytes!("../../shaders/pga_transform.spv");
pub const CGA_TRANSFORM_SPV: &[u8] = include_bytes!("../../shaders/cga_transform.spv");

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

/// Push constant payload for Gosset E8 Lattice projection on GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct HotStratumE8PushConstants {
    pub d_model: u32,
    pub neuron_count: u32,
    pub pass_id: u32,
    pub layer_gamma: f32,
    pub beta: f32,
    pub gate_gamma: f32,
    pub up_gamma: f32,
    pub words_per_row: u32,
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
