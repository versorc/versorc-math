// ============================================================================
// VERSORC-MATH: THE VERSOR GROUP & SANDWICH OPERATORS
// License: Apache-2.0
// ============================================================================

pub mod dual_quaternion;
pub mod dyadic;
pub mod hamilton;
pub mod householder;
pub mod planar_bank;
pub mod rotor;

pub use dual_quaternion::DualQuaternion;
pub use dyadic::DyadicWalshVersor;
pub use hamilton::Quaternion;
pub use householder::{HouseholderChain, HouseholderVersor};
pub use planar_bank::PlanarSpinorBank;
pub use rotor::PlanarRotor;

/// Trait defining the two-sided sandwich action of a Versor: x |-> V * x * V~.
pub trait VersorAction<T: ?Sized> {
    /// Applies the versor sandwich transformation in-place: x = V * x * V~.
    fn transform_in_place(&self, target: &mut T);

    /// Applies the versor sandwich transformation out-of-place.
    fn transform(&self, target: &T) -> T
    where
        T: Clone + Sized,
    {
        let mut copy = target.clone();
        self.transform_in_place(&mut copy);
        copy
    }
}
