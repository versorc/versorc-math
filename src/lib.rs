// ============================================================================
// VERSORC-MATH: CLIFFORD GEOMETRIC ALGEBRA & VERSOR OPERATORS ENGINE
// License: Apache-2.0
// ============================================================================

pub mod blade;
pub mod cayley_klein;
pub mod cga;
pub mod dod;
pub mod gpu;
pub mod kernels;
pub mod lattice;
pub mod multivector;
pub mod pga;
pub mod poincare;
#[cfg(feature = "python")]
pub mod python;
pub mod quantization;
pub mod reduction;
pub mod signature;
pub mod sta;
pub mod versor;

pub use blade::Blade;
pub use cayley_klein::*;
pub use cga::*;
pub use dod::*;
pub use gpu::shaders::*;
#[cfg(feature = "vulkan")]
pub use gpu::vulkan::*;
pub use kernels::*;
pub use lattice::{
    E8Lattice, E8LatticeCodebook, EisensteinA3Lattice, E8_CODEBOOK_SIZE, E8_DIMENSION,
    E8_NUM_CENTROIDS, E8_NUM_ROOTS, EISENSTEIN_DIMENSION, EISENSTEIN_KISSING_NUMBER,
};
pub use multivector::{Cl2, Cl3, ClCGA3D, ClHamilton, ClMinkowski, ClPGA3D, Multivector};
pub use pga::*;
pub use poincare::{CayleyKleinSpace, PoincarePoint};
pub use quantization::*;
pub use reduction::CartanDieudonneDecomposer;
pub use signature::{
    MetricSignature, SignatureCGA3D, SignatureEuclidean2D, SignatureEuclidean3D,
    SignatureHamilton2D, SignatureMinkowski, SignaturePGA3D,
};
pub use sta::*;
pub use versor::{
    DualQuaternion, DyadicWalshVersor, HouseholderChain, HouseholderVersor, PlanarRotor,
    PlanarSpinorBank, Quaternion, VersorAction,
};

pub mod prelude {
    pub use crate::blade::Blade;
    pub use crate::cayley_klein::*;
    pub use crate::cga::*;
    pub use crate::dod::*;
    pub use crate::gpu::shaders::*;
    #[cfg(feature = "vulkan")]
    pub use crate::gpu::vulkan::*;
    pub use crate::kernels::*;
    pub use crate::lattice::*;
    pub use crate::multivector::*;
    pub use crate::pga::*;
    pub use crate::poincare::*;
    pub use crate::quantization::*;
    pub use crate::reduction::*;
    pub use crate::signature::*;
    pub use crate::sta::*;
    pub use crate::versor::*;
}
