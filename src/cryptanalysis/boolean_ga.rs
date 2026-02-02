//! Boolean Geometric Algebra (Multivectors over GF(2))
//!
//! This module implements geometric algebra over the field GF(2) = {0, 1}
//! with XOR as addition and AND as multiplication.
//!
//! # Boolean Algebra Properties
//!
//! - Addition: XOR (⊕)
//! - Multiplication: AND (∧)
//! - No subtraction needed (x ⊕ x = 0)
//! - Idempotent: x ∧ x = x (different from real GA!)
//!
//! # Multivector Structure
//!
//! For dimension n, a multivector has 2^n blades:
//! - 1 scalar (grade 0)
//! - n vectors (grade 1): e_i
//! - C(n,2) bivectors (grade 2): e_i ∧ e_j
//! - ...
//! - 1 pseudoscalar (grade n): e_1 ∧ e_2 ∧ ... ∧ e_n
//!
//! Example for n=3:
//! M = a_0 + a_1*e_1 + a_2*e_2 + a_3*e_3 + a_12*e_12 + a_13*e_13 + a_23*e_23 + a_123*e_123
//! Where each a_i ∈ {0, 1}

/// Boolean multivector over GF(2)
///
/// Represents a multivector in Cl(n, 0) over the field GF(2).
/// Each blade coefficient is a boolean (0 or 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanMultivector {
    /// Dimension of the underlying vector space
    dimension: usize,

    /// Blade coefficients (0 or 1)
    /// Indexed by subset representation: blade for subset S has index Σ_{i∈S} 2^i
    /// For n=3: [scalar, e1, e2, e12, e3, e13, e23, e123]
    /// Indices:  [0,     1,  2,  3,   4,  5,   6,   7]
    coeffs: Vec<bool>,
}

impl BooleanMultivector {
    /// Create zero multivector (all coefficients = 0)
    pub fn zero(dimension: usize) -> Self {
        let num_blades = 1 << dimension;
        Self {
            dimension,
            coeffs: vec![false; num_blades],
        }
    }

    /// Create scalar multivector
    pub fn scalar(dimension: usize, value: bool) -> Self {
        let mut mv = Self::zero(dimension);
        mv.coeffs[0] = value;
        mv
    }

    /// Create basis vector e_i
    pub fn basis_vector(dimension: usize, i: usize) -> Self {
        assert!(i < dimension, "Basis vector index out of range");
        let mut mv = Self::zero(dimension);
        mv.coeffs[1 << i] = true;
        mv
    }

    /// Create from bit vector
    ///
    /// Interprets a bit vector as a grade-1 multivector.
    /// For example, 0b10110011 (8 bits) becomes:
    /// e_0 + e_1 + e_4 + e_5 + e_7
    ///
    /// # Arguments
    ///
    /// * `bits` - Bit vector as u8
    /// * `dimension` - Dimension (must be ≤ 8 for u8)
    pub fn from_bitvec(bits: u8, dimension: usize) -> Self {
        assert!(dimension <= 8, "Dimension too large for u8");
        let mut mv = Self::zero(dimension);

        // Set coefficients for basis vectors where bit is 1
        for i in 0..dimension {
            if (bits >> i) & 1 == 1 {
                mv.coeffs[1 << i] = true;
            }
        }

        mv
    }

    /// Convert grade-1 part to bit vector
    ///
    /// Returns the bit vector representation of the grade-1 part.
    /// Only valid if this multivector has only grade-1 components.
    pub fn to_bitvec(&self) -> u8 {
        assert!(self.dimension <= 8, "Dimension too large for u8");

        let mut bits = 0u8;
        for i in 0..self.dimension {
            if self.coeffs[1 << i] {
                bits |= 1 << i;
            }
        }
        bits
    }

    /// Get coefficient for a specific blade
    ///
    /// # Arguments
    ///
    /// * `blade_index` - Index of blade (0 = scalar, 1 = e1, 2 = e2, 3 = e12, ...)
    pub fn get_coeff(&self, blade_index: usize) -> bool {
        self.coeffs[blade_index]
    }

    /// Set coefficient for a specific blade
    pub fn set_coeff(&mut self, blade_index: usize, value: bool) {
        self.coeffs[blade_index] = value;
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get number of blades (2^n)
    pub fn num_blades(&self) -> usize {
        self.coeffs.len()
    }

    /// XOR (addition in GF(2))
    ///
    /// Component-wise XOR of coefficients.
    pub fn xor(&self, other: &Self) -> Self {
        assert_eq!(self.dimension, other.dimension);

        let mut result = Self::zero(self.dimension);
        for i in 0..self.coeffs.len() {
            result.coeffs[i] = self.coeffs[i] ^ other.coeffs[i];
        }
        result
    }

    /// Geometric product over GF(2)
    ///
    /// Computes the geometric product a·b where:
    /// - e_i · e_j = e_i ∧ e_j (for i ≠ j, since e_i · e_i = 1 in Cl(n,0))
    /// - Multiplication is over GF(2) (XOR for addition, AND for multiplication)
    ///
    /// Note: In Boolean GA, e_i · e_i = 1 (not 0 as in standard GA!)
    /// This is because we're working over GF(2) where 1 + 1 = 0.
    pub fn geometric_product(&self, other: &Self) -> Self {
        assert_eq!(self.dimension, other.dimension);

        let mut result = Self::zero(self.dimension);

        // For each pair of blades
        for i in 0..self.coeffs.len() {
            if !self.coeffs[i] {
                continue;
            }

            for j in 0..other.coeffs.len() {
                if !other.coeffs[j] {
                    continue;
                }

                // Multiply blades i and j
                // Blade i represents subset S_i, blade j represents subset S_j
                // Product is symmetric difference (XOR) in Boolean GA
                let product_blade = i ^ j;

                // XOR into result (addition in GF(2))
                result.coeffs[product_blade] ^= true;
            }
        }

        result
    }

    /// Wedge product (outer product) over GF(2)
    ///
    /// For Boolean GA, the wedge product is antisymmetric:
    /// e_i ∧ e_j = -e_j ∧ e_i (but -1 = 1 in GF(2), so still antisymmetric)
    /// e_i ∧ e_i = 0
    pub fn wedge(&self, other: &Self) -> Self {
        assert_eq!(self.dimension, other.dimension);

        let mut result = Self::zero(self.dimension);

        for i in 0..self.coeffs.len() {
            if !self.coeffs[i] {
                continue;
            }

            for j in 0..other.coeffs.len() {
                if !other.coeffs[j] {
                    continue;
                }

                // Check if blades i and j share any basis vectors
                if i & j == 0 {
                    // No overlap - valid wedge product
                    let product_blade = i | j; // Union of basis vectors

                    // XOR into result
                    result.coeffs[product_blade] ^= true;
                }
                // If overlap, wedge product is 0 (already false in result)
            }
        }

        result
    }

    /// Fast wedge product for grade-1 multivectors (bit vectors)
    ///
    /// Optimized O(1) version when both inputs are known to be grade-1.
    /// For grade-1 vectors represented as bit vectors a and b:
    /// a ∧ b = 0 if and only if a & b != 0 (share a common bit)
    ///
    /// This avoids the O(2^n × 2^n) double loop of the general wedge product.
    pub fn wedge_grade1_fast(&self, other: &Self, a_bits: u8, b_bits: u8) -> Self {
        assert_eq!(self.dimension, other.dimension);
        assert!(self.dimension <= 8);

        // Fast check: if bits overlap, wedge is zero
        if a_bits & b_bits != 0 {
            return Self::zero(self.dimension);
        }

        // Otherwise compute the general wedge product
        self.wedge(other)
    }

    /// Check if a grade-1 multivector (bit vector) is zero
    ///
    /// Fast O(1) check for bit vectors
    pub fn is_zero_grade1_fast(&self, bits: u8) -> bool {
        bits == 0
    }

    /// Inner product (contraction) over GF(2)
    ///
    /// For Boolean GA, the inner product contracts common basis vectors.
    pub fn inner(&self, other: &Self) -> Self {
        assert_eq!(self.dimension, other.dimension);

        let mut result = Self::zero(self.dimension);

        for i in 0..self.coeffs.len() {
            if !self.coeffs[i] {
                continue;
            }

            for j in 0..other.coeffs.len() {
                if !other.coeffs[j] {
                    continue;
                }

                // Inner product: contract common basis vectors
                let common = i & j;
                let remaining = i ^ j; // XOR removes common basis vectors

                if common != 0 {
                    // There's something to contract
                    result.coeffs[remaining] ^= true;
                }
            }
        }

        result
    }

    /// Extract grade k part
    ///
    /// Returns a new multivector containing only blades of grade k.
    /// Grade k means k basis vectors in the blade.
    pub fn grade(&self, k: usize) -> Self {
        let mut result = Self::zero(self.dimension);

        for i in 0..self.coeffs.len() {
            if self.coeffs[i] && i.count_ones() as usize == k {
                result.coeffs[i] = true;
            }
        }

        result
    }

    /// Check if multivector is zero
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| !c)
    }

    /// Count number of non-zero coefficients
    pub fn count_nonzero(&self) -> usize {
        self.coeffs.iter().filter(|&&c| c).count()
    }
}

/// Compute parity (sum in GF(2)) of bit vector
///
/// Returns true if odd number of 1s, false if even.
pub fn parity(bits: u8) -> bool {
    bits.count_ones() % 2 == 1
}

/// Compute inner product (dot product) of two bit vectors in GF(2)
///
/// ⟨a, b⟩ = Σ a_i · b_i (mod 2) = parity(a ∧ b)
pub fn dot_product(a: u8, b: u8) -> bool {
    parity(a & b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let mv = BooleanMultivector::zero(3);
        assert_eq!(mv.dimension(), 3);
        assert_eq!(mv.num_blades(), 8);
        assert!(mv.is_zero());
    }

    #[test]
    fn test_scalar() {
        let mv = BooleanMultivector::scalar(3, true);
        assert!(!mv.is_zero());
        assert_eq!(mv.get_coeff(0), true);
        assert_eq!(mv.get_coeff(1), false);
    }

    #[test]
    fn test_basis_vector() {
        let e1 = BooleanMultivector::basis_vector(3, 0);
        assert_eq!(e1.get_coeff(1), true); // e1 is at index 1 (2^0)
        assert_eq!(e1.count_nonzero(), 1);
    }

    #[test]
    fn test_from_bitvec() {
        // 0b011 = e0 + e1
        let mv = BooleanMultivector::from_bitvec(0b011, 3);
        assert_eq!(mv.get_coeff(1), true); // e0 at index 1
        assert_eq!(mv.get_coeff(2), true); // e1 at index 2
        assert_eq!(mv.get_coeff(4), false); // e2 at index 4
    }

    #[test]
    fn test_to_bitvec() {
        let mv = BooleanMultivector::from_bitvec(0b10110011, 8);
        assert_eq!(mv.to_bitvec(), 0b10110011);
    }

    #[test]
    fn test_xor() {
        let a = BooleanMultivector::from_bitvec(0b101, 3);
        let b = BooleanMultivector::from_bitvec(0b011, 3);
        let c = a.xor(&b);

        // 0b101 ⊕ 0b011 = 0b110
        assert_eq!(c.to_bitvec(), 0b110);
    }

    #[test]
    fn test_wedge_product() {
        // e1 ∧ e2 = e12
        let e1 = BooleanMultivector::basis_vector(3, 0);
        let e2 = BooleanMultivector::basis_vector(3, 1);
        let e12 = e1.wedge(&e2);

        // e12 is at index 3 (binary 011 = 2^0 + 2^1)
        assert_eq!(e12.get_coeff(3), true);
        assert_eq!(e12.count_nonzero(), 1);
    }

    #[test]
    fn test_wedge_product_antisymmetric() {
        // e1 ∧ e1 = 0
        let e1 = BooleanMultivector::basis_vector(3, 0);
        let result = e1.wedge(&e1);
        assert!(result.is_zero());
    }

    #[test]
    fn test_geometric_product() {
        // e1 · e1 = 1 (scalar)
        let e1 = BooleanMultivector::basis_vector(3, 0);
        let result = e1.geometric_product(&e1);

        assert_eq!(result.get_coeff(0), true); // Scalar component
        assert_eq!(result.count_nonzero(), 1);
    }

    #[test]
    fn test_grade_extraction() {
        // Create multivector with mixed grades
        let mut mv = BooleanMultivector::zero(3);
        mv.set_coeff(0, true);  // Scalar
        mv.set_coeff(1, true);  // e1 (grade 1)
        mv.set_coeff(3, true);  // e12 (grade 2)

        let grade0 = mv.grade(0);
        assert_eq!(grade0.get_coeff(0), true);
        assert_eq!(grade0.count_nonzero(), 1);

        let grade1 = mv.grade(1);
        assert_eq!(grade1.get_coeff(1), true);
        assert_eq!(grade1.count_nonzero(), 1);

        let grade2 = mv.grade(2);
        assert_eq!(grade2.get_coeff(3), true);
        assert_eq!(grade2.count_nonzero(), 1);
    }

    #[test]
    fn test_parity() {
        assert_eq!(parity(0b0000), false); // 0 ones
        assert_eq!(parity(0b0001), true);  // 1 one
        assert_eq!(parity(0b0011), false); // 2 ones
        assert_eq!(parity(0b0111), true);  // 3 ones
    }

    #[test]
    fn test_dot_product() {
        // (1,0,1)·(1,1,0) = 1×1 + 0×1 + 1×0 = 1 (mod 2) = true
        assert_eq!(dot_product(0b101, 0b011), true);
        // (1,1,1)·(1,1,1) = 1×1 + 1×1 + 1×1 = 3 (mod 2) = true
        assert_eq!(dot_product(0b111, 0b111), true);
        // (1,0,1)·(0,1,0) = 0 (mod 2) = false
        assert_eq!(dot_product(0b101, 0b010), false);
    }
}
