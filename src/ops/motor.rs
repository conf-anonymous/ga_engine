use crate::{rotor::Rotor3, vector::Vec3};

/// A 3-D rigid‐body motor: rotation + translation
#[derive(Clone, Debug, PartialEq)]
pub struct Motor3 {
    pub rotor: Rotor3,
    pub translation: Vec3,
}

impl Motor3 {
    pub fn new(rotor: Rotor3, translation: Vec3) -> Self {
        Self { rotor, translation }
    }

    /// Apply to a point: R·p + t
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        self.rotor.rotate_fast(p) + self.translation
    }

    /// Compose with another motor: self then `other`
    pub fn then(&self, other: &Motor3) -> Motor3 {
        let r_prod = other.rotor.mul(&self.rotor);
        let t = other.transform_point(self.translation);
        Motor3::new(r_prod, t)
    }
}
