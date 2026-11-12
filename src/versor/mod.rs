// ============================================================================
// VERSORC-MATH: THE VERSOR GROUP & SANDWICH OPERATORS
// License: Apache-2.0 OR MIT
// ============================================================================

pub mod hamilton;
pub mod dual_quaternion;
pub mod rotor;
pub mod householder;
pub mod dyadic;
pub mod planar_bank;

pub use hamilton::Quaternion;
pub use dual_quaternion::DualQuaternion;
pub use rotor::PlanarRotor;
pub use householder::{HouseholderVersor, HouseholderChain};
pub use dyadic::DyadicWalshVersor;
pub use planar_bank::PlanarSpinorBank;

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
