//! Cl(3,0) Multivector type with true geometric product
//!
//! This module provides a multivector type for 3D Euclidean geometric algebra
//! with the full geometric product implementation.
//!
//! Basis: {1, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃}
//! Metric: e₁² = e₂² = e₃² = +1 (Euclidean signature)

use crate::ga::{geometric_product_full, geometric_product_full_simd};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::ops::{Add, Sub, Mul, Neg};

/// Cl(3,0) multivector with 8 components
///
/// Component ordering: [scalar, e₁, e₂, e₃, e₂₃, e₃₁, e₁₂, e₁₂₃]
/// (Note: e₃₁ is stored, which equals -e₁₃ in canonical ordering)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Multivector {
    pub components: [f64; 8],
}

impl Multivector {
    /// Create zero multivector
    pub fn zero() -> Self {
        Multivector { components: [0.0; 8] }
    }

    /// Create from components
    pub fn new(components: [f64; 8]) -> Self {
        Multivector { components }
    }

    /// Create scalar multivector
    pub fn scalar(s: f64) -> Self {
        let mut components = [0.0; 8];
        components[0] = s;
        Multivector { components }
    }

    /// Create vector (grade-1) multivector from (x, y, z)
    pub fn vector(x: f64, y: f64, z: f64) -> Self {
        Multivector {
            components: [0.0, x, y, z, 0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Create from 3D point
    pub fn from_point(x: f64, y: f64, z: f64) -> Self {
        Self::vector(x, y, z)
    }

    /// Create random multivector with Xavier initialization
    pub fn random_xavier(fan_in: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / fan_in as f64).sqrt();
        let mut components = [0.0; 8];
        for c in &mut components {
            *c = rng.gen_range(-scale..scale);
        }
        Multivector { components }
    }

    /// Create random multivector from normal distribution
    pub fn random_normal(std: f64) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, std).unwrap();
        let mut components = [0.0; 8];
        for c in &mut components {
            *c = normal.sample(&mut rng);
        }
        Multivector { components }
    }

    /// Scalar component (grade 0)
    #[inline]
    pub fn scalar_part(&self) -> f64 {
        self.components[0]
    }

    /// Vector components (grade 1): (e₁, e₂, e₃)
    #[inline]
    pub fn vector_part(&self) -> [f64; 3] {
        [self.components[1], self.components[2], self.components[3]]
    }

    /// Bivector components (grade 2): (e₂₃, e₃₁, e₁₂)
    #[inline]
    pub fn bivector_part(&self) -> [f64; 3] {
        [self.components[4], self.components[5], self.components[6]]
    }

    /// Pseudoscalar component (grade 3): e₁₂₃
    #[inline]
    pub fn pseudoscalar_part(&self) -> f64 {
        self.components[7]
    }

    /// Geometric product: self ⊗ other
    ///
    /// Uses the optimized lookup table implementation from ga.rs
    #[inline]
    pub fn gp(&self, other: &Multivector) -> Multivector {
        let mut out = [0.0; 8];
        geometric_product_full(&self.components, &other.components, &mut out);
        Multivector { components: out }
    }

    /// SIMD-accelerated geometric product
    #[inline]
    pub fn gp_simd(&self, other: &Multivector) -> Multivector {
        let mut out = [0.0; 8];
        geometric_product_full_simd(&self.components, &other.components, &mut out);
        Multivector { components: out }
    }

    /// Reverse (reversion): reverses order of basis vectors
    ///
    /// For Cl(3,0):
    /// - Grade 0 (scalar): unchanged
    /// - Grade 1 (vectors): unchanged
    /// - Grade 2 (bivectors): negated
    /// - Grade 3 (pseudoscalar): negated
    #[inline]
    pub fn reverse(&self) -> Multivector {
        Multivector {
            components: [
                self.components[0],   // scalar: +
                self.components[1],   // e1: +
                self.components[2],   // e2: +
                self.components[3],   // e3: +
                -self.components[4],  // e23: -
                -self.components[5],  // e31: -
                -self.components[6],  // e12: -
                -self.components[7],  // e123: -
            ],
        }
    }

    /// Grade involution (main involution)
    ///
    /// Negates odd-grade components
    #[inline]
    pub fn involute(&self) -> Multivector {
        Multivector {
            components: [
                self.components[0],   // scalar: +
                -self.components[1],  // e1: -
                -self.components[2],  // e2: -
                -self.components[3],  // e3: -
                self.components[4],   // e23: +
                self.components[5],   // e31: +
                self.components[6],   // e12: +
                -self.components[7],  // e123: -
            ],
        }
    }

    /// Clifford conjugate: reverse followed by grade involution
    #[inline]
    pub fn conjugate(&self) -> Multivector {
        self.reverse().involute()
    }

    /// Squared norm: M · M̃ (scalar part)
    ///
    /// For normalized multivectors, this equals 1
    #[inline]
    pub fn norm_squared(&self) -> f64 {
        let rev = self.reverse();
        let product = self.gp(&rev);
        product.scalar_part()
    }

    /// Euclidean norm
    #[inline]
    pub fn norm(&self) -> f64 {
        self.norm_squared().abs().sqrt()
    }

    /// Normalize to unit multivector
    #[inline]
    pub fn normalize(&self) -> Multivector {
        let n = self.norm();
        if n > 1e-10 {
            self.scale(1.0 / n)
        } else {
            *self
        }
    }

    /// Scale by scalar
    #[inline]
    pub fn scale(&self, s: f64) -> Multivector {
        let mut components = [0.0; 8];
        for i in 0..8 {
            components[i] = self.components[i] * s;
        }
        Multivector { components }
    }

    /// Component-wise addition
    #[inline]
    pub fn add(&self, other: &Multivector) -> Multivector {
        let mut components = [0.0; 8];
        for i in 0..8 {
            components[i] = self.components[i] + other.components[i];
        }
        Multivector { components }
    }

    /// Component-wise subtraction
    #[inline]
    pub fn sub(&self, other: &Multivector) -> Multivector {
        let mut components = [0.0; 8];
        for i in 0..8 {
            components[i] = self.components[i] - other.components[i];
        }
        Multivector { components }
    }

    /// Inner product (symmetric part of geometric product)
    ///
    /// a · b = (ab + ba) / 2 for vectors
    #[inline]
    pub fn inner(&self, other: &Multivector) -> Multivector {
        let ab = self.gp(other);
        let ba = other.gp(self);
        Multivector::add(&ab, &ba).scale(0.5)
    }

    /// Outer product (antisymmetric part of geometric product)
    ///
    /// a ∧ b = (ab - ba) / 2 for vectors
    #[inline]
    pub fn outer(&self, other: &Multivector) -> Multivector {
        let ab = self.gp(other);
        let ba = other.gp(self);
        Multivector::sub(&ab, &ba).scale(0.5)
    }

    /// Dot product of all 8 components (for similarity)
    #[inline]
    pub fn dot(&self, other: &Multivector) -> f64 {
        let mut sum = 0.0;
        for i in 0..8 {
            sum += self.components[i] * other.components[i];
        }
        sum
    }

    /// L2 squared distance between multivectors
    #[inline]
    pub fn distance_squared(&self, other: &Multivector) -> f64 {
        let mut sum = 0.0;
        for i in 0..8 {
            let d = self.components[i] - other.components[i];
            sum += d * d;
        }
        sum
    }

    /// Square activation (polynomial, FHE-friendly)
    #[inline]
    pub fn square_activation(&self) -> Multivector {
        let mut components = [0.0; 8];
        for i in 0..8 {
            components[i] = self.components[i] * self.components[i];
        }
        Multivector { components }
    }

    /// Cube activation (polynomial, FHE-friendly)
    #[inline]
    pub fn cube_activation(&self) -> Multivector {
        let mut components = [0.0; 8];
        for i in 0..8 {
            let x = self.components[i];
            components[i] = x * x * x;
        }
        Multivector { components }
    }

    /// Polynomial activation: ax² + bx + c (configurable)
    #[inline]
    pub fn poly_activation(&self, a: f64, b: f64, c: f64) -> Multivector {
        let mut components = [0.0; 8];
        for i in 0..8 {
            let x = self.components[i];
            components[i] = a * x * x + b * x + c;
        }
        Multivector { components }
    }

    /// Approximate GELU using polynomial (FHE-friendly)
    ///
    /// GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    /// We use a degree-3 polynomial approximation
    #[inline]
    pub fn gelu_approx(&self) -> Multivector {
        // Approximation: 0.5x + 0.398942x³ - 0.0357x⁵ ≈ GELU for |x| < 2
        // Simplified: 0.5x + 0.17x³ for basic approximation
        let mut components = [0.0; 8];
        for i in 0..8 {
            let x = self.components[i];
            components[i] = 0.5 * x + 0.17 * x * x * x;
        }
        Multivector { components }
    }

    /// Clamp components to range (for numerical stability)
    #[inline]
    pub fn clamp(&self, min: f64, max: f64) -> Multivector {
        let mut components = [0.0; 8];
        for i in 0..8 {
            components[i] = self.components[i].clamp(min, max);
        }
        Multivector { components }
    }

    /// Sum of all component magnitudes (L1 norm)
    #[inline]
    pub fn l1_norm(&self) -> f64 {
        self.components.iter().map(|x| x.abs()).sum()
    }

    /// Maximum component magnitude (L∞ norm)
    #[inline]
    pub fn linf_norm(&self) -> f64 {
        self.components.iter().map(|x| x.abs()).fold(0.0f64, f64::max)
    }

    /// Convert to flat array (for serialization)
    pub fn to_array(&self) -> [f64; 8] {
        self.components
    }

    /// Create from flat array
    pub fn from_array(arr: [f64; 8]) -> Self {
        Multivector { components: arr }
    }
}

impl Default for Multivector {
    fn default() -> Self {
        Self::zero()
    }
}

impl Add for Multivector {
    type Output = Multivector;

    fn add(self, other: Multivector) -> Multivector {
        Multivector::add(&self, &other)
    }
}

impl Sub for Multivector {
    type Output = Multivector;

    fn sub(self, other: Multivector) -> Multivector {
        Multivector::sub(&self, &other)
    }
}

impl Mul for Multivector {
    type Output = Multivector;

    /// Geometric product
    fn mul(self, other: Multivector) -> Multivector {
        self.gp(&other)
    }
}

impl Mul<f64> for Multivector {
    type Output = Multivector;

    fn mul(self, scalar: f64) -> Multivector {
        self.scale(scalar)
    }
}

impl Neg for Multivector {
    type Output = Multivector;

    fn neg(self) -> Multivector {
        self.scale(-1.0)
    }
}

impl std::fmt::Display for Multivector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.4} + {:.4}e₁ + {:.4}e₂ + {:.4}e₃ + {:.4}e₂₃ + {:.4}e₃₁ + {:.4}e₁₂ + {:.4}e₁₂₃",
            self.components[0],
            self.components[1],
            self.components[2],
            self.components[3],
            self.components[4],
            self.components[5],
            self.components[6],
            self.components[7]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_product() {
        let a = Multivector::scalar(2.0);
        let b = Multivector::scalar(3.0);
        let c = a.gp(&b);
        assert!((c.scalar_part() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_product() {
        // e1 * e1 = 1 (positive signature)
        let e1 = Multivector::vector(1.0, 0.0, 0.0);
        let result = e1.gp(&e1);
        assert!((result.scalar_part() - 1.0).abs() < 1e-10);

        // e1 * e2 = e12 (bivector)
        let e2 = Multivector::vector(0.0, 1.0, 0.0);
        let result = e1.gp(&e2);
        assert!((result.bivector_part()[2] - 1.0).abs() < 1e-10); // e12 component
    }

    #[test]
    fn test_anticommutativity() {
        // e1 * e2 = -e2 * e1
        let e1 = Multivector::vector(1.0, 0.0, 0.0);
        let e2 = Multivector::vector(0.0, 1.0, 0.0);

        let ab = e1.gp(&e2);
        let ba = e2.gp(&e1);
        let sum = Multivector::add(&ab, &ba);

        for i in 0..8 {
            assert!(sum.components[i].abs() < 1e-10, "Component {} should be 0", i);
        }
    }

    #[test]
    fn test_reverse() {
        let mv = Multivector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let rev = mv.reverse();

        // Scalar and vectors unchanged
        assert_eq!(rev.components[0], 1.0);
        assert_eq!(rev.components[1], 2.0);
        assert_eq!(rev.components[2], 3.0);
        assert_eq!(rev.components[3], 4.0);

        // Bivectors and pseudoscalar negated
        assert_eq!(rev.components[4], -5.0);
        assert_eq!(rev.components[5], -6.0);
        assert_eq!(rev.components[6], -7.0);
        assert_eq!(rev.components[7], -8.0);
    }

    #[test]
    fn test_norm() {
        // Unit vector should have norm 1
        let e1 = Multivector::vector(1.0, 0.0, 0.0);
        assert!((e1.norm() - 1.0).abs() < 1e-10);

        // Vector (3, 4, 0) should have norm 5
        let v = Multivector::vector(3.0, 4.0, 0.0);
        assert!((v.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let v = Multivector::vector(3.0, 4.0, 0.0);
        let normalized = v.normalize();
        assert!((normalized.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_activations() {
        let mv = Multivector::new([2.0, -1.0, 0.5, 0.0, 1.0, -0.5, 0.25, -0.25]);

        // Square activation
        let sq = mv.square_activation();
        assert!((sq.components[0] - 4.0).abs() < 1e-10);
        assert!((sq.components[1] - 1.0).abs() < 1e-10);
        assert!((sq.components[2] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_simd_matches_scalar() {
        let a = Multivector::random_normal(1.0);
        let b = Multivector::random_normal(1.0);

        let scalar_result = a.gp(&b);
        let simd_result = a.gp_simd(&b);

        for i in 0..8 {
            assert!(
                (scalar_result.components[i] - simd_result.components[i]).abs() < 1e-10,
                "Component {} differs: {} vs {}",
                i,
                scalar_result.components[i],
                simd_result.components[i]
            );
        }
    }
}
