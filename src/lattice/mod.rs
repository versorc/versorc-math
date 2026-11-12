// ============================================================================
// VERSORC-MATH: ROOT LATTICES & CONWAY-SLOANE VECTOR QUANTIZATION
// License: Apache-2.0 OR MIT
// ============================================================================

pub mod e8;
pub mod eisenstein;

pub use e8::{
    E8Lattice, E8LatticeCodebook, E8_CODEBOOK_SIZE, E8_DIMENSION, E8_NUM_CENTROIDS,
    E8_NUM_ROOTS,
};
pub use eisenstein::{
    EisensteinA3Lattice, EISENSTEIN_DIMENSION, EISENSTEIN_KISSING_NUMBER,
};
