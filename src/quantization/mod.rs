// ============================================================================
// VERSORC-MATH: CLIFFORD GEOMETRIC QUANTIZATION MODULE
// License: Apache-2.0 OR MIT
// ============================================================================

pub mod clifford_3bit;
pub mod turboquant;

pub use clifford_3bit::{
    bf16_to_f32, f32_to_bf16, Clifford3BitCacheLine64, Clifford3BitTensor, Clifford3BitWord64,
    LLOYD_MAX_3BIT_BOUNDS, LLOYD_MAX_3BIT_CENTROIDS,
};
pub use turboquant::{TurboQuantBlock32, TurboQuantCacheLine64, TurboQuantTensor};
