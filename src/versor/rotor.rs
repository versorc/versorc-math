// ============================================================================
// VERSORC-MATH: GENERAL EXPONENTIAL PLANAR ROTOR R = exp(-B/2)
// License: Apache-2.0
// ============================================================================

use super::VersorAction;

/// Planar 2-Versor (Rotor) operating in a single bivector plane.
/// R = cos(theta / 2) + B * sin(theta / 2)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanarRotor {
    pub cos_half: f32,
    pub sin_half: f32,
    pub angle_rad: f32,
}

impl Default for PlanarRotor {
    #[inline(always)]
    fn default() -> Self {
        Self::identity()
    }
}

impl PlanarRotor {
    #[inline(always)]
    pub const fn identity() -> Self {
        Self {
            cos_half: 1.0,
            sin_half: 0.0,
            angle_rad: 0.0,
        }
    }

    /// Creates a planar rotor from rotation angle theta in radians.
    #[inline]
    pub fn from_angle(theta_rad: f32) -> Self {
        let half = theta_rad * 0.5;
        Self {
            cos_half: half.cos(),
            sin_half: half.sin(),
            angle_rad: theta_rad,
        }
    }

    /// Reverse of the rotor: R~ = cos(theta/2) - B*sin(theta/2).
    #[inline(always)]
    pub fn reverse(&self) -> Self {
        Self {
            cos_half: self.cos_half,
            sin_half: -self.sin_half,
            angle_rad: -self.angle_rad,
        }
    }

    /// Composes this rotor with another planar rotor in the same plane: R_total = R1 * R2.
    #[inline]
    pub fn compose(&self, other: &Self) -> Self {
        Self::from_angle(self.angle_rad + other.angle_rad)
    }

    /// Rotates a 2D vector [u, v] in the plane of the rotor.
    #[inline(always)]
    pub fn rotate_2d(&self, v: &[f32; 2]) -> [f32; 2] {
        let cos_theta = self.angle_rad.cos();
        let sin_theta = self.angle_rad.sin();
        [
            v[0] * cos_theta - v[1] * sin_theta,
            v[0] * sin_theta + v[1] * cos_theta,
        ]
    }
}

impl VersorAction<[f32; 2]> for PlanarRotor {
    #[inline]
    fn transform_in_place(&self, target: &mut [f32; 2]) {
        *target = self.rotate_2d(target);
    }
}
