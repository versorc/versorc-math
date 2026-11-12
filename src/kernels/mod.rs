// ============================================================================
// VERSORC-MATH: SPECIALIZED HARDWARE COMPUTE KERNELS
// License: Apache-2.0
//
// High-performance hardware kernels designed around Data-Oriented Design (DOD):
// - clifford_3bit_avx2: AVX2 8-way vpermps + vpsrlvd vectorized dot product for Clifford 3-bit.
// - asym64_avx2: Reference 4.5 bpw AVX2 FMA fused dot product.
// ============================================================================

pub mod clifford_3bit_avx2;

pub use clifford_3bit_avx2::{
    dot_clifford_3bit_word_avx2, dot_clifford_3bit_cache_line_avx2,
    matvec_clifford_3bit_avx2, matvec_clifford_3bit_prerotated_avx2,
    matvec_clifford_3bit_bytes_prerotated_avx2,
};
