// multivector.rs

use crate::{bivector::Bivector3, vector::Vec3};
use std::fmt;
use std::ops::{Add, Sub};

/// Full 8-component multivector in 3-D: scalar + Vec3 + Bivector3 + pseudoscalar.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Multivector3 {
    /// Grade‑0 component
    pub scalar: f64,
    /// Grade‑1 component
    pub vector: Vec3,
    /// Grade‑2 component
    pub bivector: Bivector3,
    /// Grade‑3 (pseudoscalar) component
    pub pseudo: f64,
}

impl Multivector3 {
    /// Create the zero multivector (all components zero).
    pub fn zero() -> Self {
        Self {
            scalar: 0.0,
            vector: Vec3::new(0.0, 0.0, 0.0),
            bivector: Bivector3::new(0.0, 0.0, 0.0),
            pseudo: 0.0,
        }
    }

    /// Create a pure scalar multivector.
    pub fn from_scalar(s: f64) -> Self {
        Self {
            scalar: s,
            ..Self::zero()
        }
    }

    /// Create a pure vector multivector.
    pub fn from_vector(v: Vec3) -> Self {
        Self {
            vector: v,
            ..Self::zero()
        }
    }

    /// Create a pure bivector multivector.
    pub fn from_bivector(b: Bivector3) -> Self {
        Self {
            bivector: b,
            ..Self::zero()
        }
    }

    /// Create a pure pseudoscalar multivector.
    pub fn from_pseudoscalar(p: f64) -> Self {
        Self {
            pseudo: p,
            ..Self::zero()
        }
    }

    /// Geometric product of two multivectors.
    pub fn gp(&self, other: &Self) -> Self {
        let mut out = [0.0; 8];
        super::ga::geometric_product_full(
            &[
                self.scalar,
                self.vector.x,
                self.vector.y,
                self.vector.z,
                self.bivector.xy,
                self.bivector.yz,
                self.bivector.zx,
                self.pseudo,
            ],
            &[
                other.scalar,
                other.vector.x,
                other.vector.y,
                other.vector.z,
                other.bivector.xy,
                other.bivector.yz,
                other.bivector.zx,
                other.pseudo,
            ],
            &mut out,
        );
        Multivector3 {
            scalar: out[0],
            vector: Vec3::new(out[1], out[2], out[3]),
            bivector: Bivector3::new(out[4], out[5], out[6]),
            pseudo: out[7],
        }
    }

    /// Reverse involution: flips sign of bivector and pseudoscalar parts.
    pub fn reverse(&self) -> Self {
        Self {
            scalar: self.scalar,
            vector: self.vector,
            bivector: Bivector3::new(-self.bivector.xy, -self.bivector.yz, -self.bivector.zx),
            pseudo: -self.pseudo,
        }
    }
}

impl Add for Multivector3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            scalar: self.scalar + rhs.scalar,
            vector: Vec3::new(
                self.vector.x + rhs.vector.x,
                self.vector.y + rhs.vector.y,
                self.vector.z + rhs.vector.z,
            ),
            bivector: Bivector3::new(
                self.bivector.xy + rhs.bivector.xy,
                self.bivector.yz + rhs.bivector.yz,
                self.bivector.zx + rhs.bivector.zx,
            ),
            pseudo: self.pseudo + rhs.pseudo,
        }
    }
}

impl Sub for Multivector3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            scalar: self.scalar - rhs.scalar,
            vector: Vec3::new(
                self.vector.x - rhs.vector.x,
                self.vector.y - rhs.vector.y,
                self.vector.z - rhs.vector.z,
            ),
            bivector: Bivector3::new(
                self.bivector.xy - rhs.bivector.xy,
                self.bivector.yz - rhs.bivector.yz,
                self.bivector.zx - rhs.bivector.zx,
            ),
            pseudo: self.pseudo - rhs.pseudo,
        }
    }
}

impl fmt::Display for Multivector3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "M{{ s: {s}, v: ({:.3}, {:.3}, {:.3}), b: ({:.3}, {:.3}, {:.3}), p: {p:.3} }}",
            self.vector.x,
            self.vector.y,
            self.vector.z,
            self.bivector.xy,
            self.bivector.yz,
            self.bivector.zx,
            s = self.scalar,
            p = self.pseudo
        )
    }
}
