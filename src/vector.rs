// vector.rs

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A 3-D Euclidean vector.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Vec3 {
    /// Create a new `Vec3` from components.
    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Dot product of two 3-vectors.
    #[inline(always)]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product of two 3-vectors.
    #[inline(always)]
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Euclidean norm (length) of the vector.
    #[inline(always)]
    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Scale the vector by a scalar.
    #[inline(always)]
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Project this vector onto `axis`:  
    /// projₐ(self) = axis * (self·axis) / (axis·axis)
    #[inline(always)]
    pub fn project_onto(&self, axis: &Vec3) -> Vec3 {
        let denom = axis.dot(axis);
        if denom == 0.0 {
            return Vec3::default();
        }
        let scale = self.dot(axis) / denom;
        *axis * scale
    }

    /// Reject this vector from `axis` (component orthogonal to `axis`).
    #[inline(always)]
    pub fn reject_from(&self, axis: &Vec3) -> Vec3 {
        *self - self.project_onto(axis)
    }

    /// Scalar projection (signed) of this vector onto `axis`.
    #[inline(always)]
    pub fn scalar_proj_onto(&self, axis: &Vec3) -> f64 {
        let len = axis.norm();
        if len == 0.0 {
            0.0
        } else {
            self.dot(axis) / len
        }
    }
}

impl Default for Vec3 {
    fn default() -> Vec3 {
        Vec3::new(0.0, 0.0, 0.0)
    }
}

impl From<[f64; 3]> for Vec3 {
    fn from(arr: [f64; 3]) -> Vec3 {
        Vec3::new(arr[0], arr[1], arr[2])
    }
}

impl From<Vec3> for [f64; 3] {
    fn from(v: Vec3) -> [f64; 3] {
        [v.x, v.y, v.z]
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Vec3 {
        Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

/// A tiny wrapper for printing a `Vec3` rounded to `decimals` places.
pub struct Rounded<'a>(&'a Vec3, usize);

impl<'a> Rounded<'a> {
    /// Wrap a `&Vec3` for pretty-printing with `decimals` digits.
    #[inline(always)]
    pub fn new(v: &'a Vec3, decimals: usize) -> Self {
        Rounded(v, decimals)
    }
}

impl fmt::Display for Rounded<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Rounded(v, dec) = *self;
        write!(
            f,
            "Vec3 {{ x: {x:.dec$}, y: {y:.dec$}, z: {z:.dec$} }}",
            x = v.x,
            y = v.y,
            z = v.z,
            dec = dec
        )
    }
}
