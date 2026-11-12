// ============================================================================
// VERSORC-MATH: CLIFFORD GEOMETRIC ALGEBRA & VERSOR OPERATORS ENGINE
// License: Apache-2.0
// ============================================================================

#![allow(clippy::needless_range_loop)]

pub mod signature;
pub mod blade;
pub mod multivector;
pub mod versor;
pub mod lattice;
pub mod poincare;
pub mod reduction;
pub mod gpu;
pub mod dod;
pub mod cayley_klein;
pub mod pga;
pub mod cga;
pub mod sta;
pub mod quantization;
pub mod kernels;
#[cfg(feature = "python")]
pub mod python;

pub use signature::{
    MetricSignature, SignatureCGA3D, SignatureEuclidean2D, SignatureEuclidean3D,
    SignatureHamilton2D, SignatureMinkowski, SignaturePGA3D,
};
pub use blade::Blade;
pub use multivector::{
    Cl2, Cl3, ClCGA3D, ClHamilton, ClMinkowski, ClPGA3D, Multivector,
};
pub use versor::{
    DualQuaternion, DyadicWalshVersor, HouseholderChain, HouseholderVersor, PlanarRotor,
    PlanarSpinorBank, Quaternion, VersorAction,
};
pub use lattice::{
    E8Lattice, E8LatticeCodebook, E8_CODEBOOK_SIZE, E8_DIMENSION, E8_NUM_CENTROIDS,
    E8_NUM_ROOTS, EisensteinA3Lattice, EISENSTEIN_DIMENSION, EISENSTEIN_KISSING_NUMBER,
};
pub use poincare::{CayleyKleinSpace, PoincarePoint};
pub use reduction::CartanDieudonneDecomposer;
pub use gpu::shaders::*;
#[cfg(feature = "vulkan")]
pub use gpu::vulkan::*;
pub use dod::*;
pub use cayley_klein::*;
pub use pga::*;
pub use cga::*;
pub use sta::*;
pub use quantization::*;
pub use kernels::*;

pub mod prelude {
    pub use crate::signature::*;
    pub use crate::blade::Blade;
    pub use crate::multivector::*;
    pub use crate::versor::*;
    pub use crate::lattice::*;
    pub use crate::poincare::*;
    pub use crate::reduction::*;
    pub use crate::gpu::shaders::*;
    #[cfg(feature = "vulkan")]
    pub use crate::gpu::vulkan::*;
    pub use crate::dod::*;
    pub use crate::cayley_klein::*;
    pub use crate::pga::*;
    pub use crate::cga::*;
    pub use crate::sta::*;
    pub use crate::quantization::*;
    pub use crate::kernels::*;
}
