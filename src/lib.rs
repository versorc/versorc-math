// ============================================================================
// VERSORC-MATH: CLIFFORD GEOMETRIC ALGEBRA & VERSOR OPERATORS ENGINE
// License: Apache-2.0 OR MIT
// ============================================================================

#![allow(clippy::needless_range_loop)]

pub mod signature;
pub mod blade;
pub mod multivector;
pub mod versor;
pub mod lattice;
pub mod poincare;
pub mod reduction;

pub use signature::{
    MetricSignature, SignatureCGA3D, SignatureEuclidean2D, SignatureEuclidean3D,
    SignatureHamilton2D, SignatureMinkowski, SignaturePGA3D,
};
pub use blade::Blade;
pub use multivector::{
    Cl2, Cl3, ClCGA3D, ClHamilton, ClMinkowski, ClPGA3D, Multivector,
};
pub use versor::{
    DyadicWalshVersor, HouseholderChain, HouseholderVersor, PlanarRotor,
    PlanarSpinorBank, Quaternion, VersorAction,
};
pub use lattice::{E8Lattice, E8_DIMENSION, E8_NUM_ROOTS};
pub use poincare::PoincarePoint;
pub use reduction::CartanDieudonneDecomposer;

pub mod prelude {
    pub use crate::signature::*;
    pub use crate::blade::Blade;
    pub use crate::multivector::*;
    pub use crate::versor::*;
    pub use crate::lattice::*;
    pub use crate::poincare::*;
    pub use crate::reduction::*;
}
