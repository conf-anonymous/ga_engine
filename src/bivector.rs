//! src/bivector.rs
//! A grade-2 multivector (bivector) in 3-D: e23, e31, e12.

use crate::vector::Vec3;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A bivector in 3D, representing oriented plane segments:
/// - `xy` is e₂₃ component (plane perpendicular to x-axis)
/// - `yz` is e₃₁ component (plane perpendicular to y-axis)
/// - `zx` is e₁₂ component (plane perpendicular to z-axis)
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Bivector3 {
    /// e₂₃ component
    pub xy: f64,
    /// e₃₁ component
    pub yz: f64,
    /// e₁₂ component
    pub zx: f64,
}

impl Bivector3 {
    /// Construct a new bivector from its three components.
    #[inline(always)]
    pub fn new(xy: f64, yz: f64, zx: f64) -> Self {
        Self { xy, yz, zx }
    }

    /// Wedge (outer) product of two vectors: a ∧ b
    #[inline(always)]
    pub fn from_wedge(a: Vec3, b: Vec3) -> Self {
        Self {
            xy: a.y * b.z - a.z * b.y,
            yz: a.z * b.x - a.x * b.z,
            zx: a.x * b.y - a.y * b.x,
        }
    }

    /// Scale this bivector by a scalar.
    #[inline(always)]
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.xy * s, self.yz * s, self.zx * s)
    }

    /// Magnitude (norm) of the bivector: sqrt(xy² + yz² + zx²)
    #[inline(always)]
    pub fn norm(&self) -> f64 {
        (self.xy * self.xy + self.yz * self.yz + self.zx * self.zx).sqrt()
    }
}

// Arithmetic operators for Bivector3
impl Add for Bivector3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.xy + rhs.xy, self.yz + rhs.yz, self.zx + rhs.zx)
    }
}

impl Sub for Bivector3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.xy - rhs.xy, self.yz - rhs.yz, self.zx - rhs.zx)
    }
}

impl Neg for Bivector3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.xy, -self.yz, -self.zx)
    }
}

impl Mul<f64> for Bivector3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        self.scale(rhs)
    }
}

// Display with debug style
impl fmt::Display for Bivector3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Bivector3 {{ xy: {:.6}, yz: {:.6}, zx: {:.6} }}",
            self.xy, self.yz, self.zx
        )
    }
}

// Conversions
impl From<[f64; 3]> for Bivector3 {
    #[inline(always)]
    fn from(arr: [f64; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }
}

impl From<Bivector3> for [f64; 3] {
    #[inline(always)]
    fn from(b: Bivector3) -> Self {
        [b.xy, b.yz, b.zx]
    }
}
