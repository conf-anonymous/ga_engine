//! S-box representation using Geometric Algebra
//!
//! This module represents S-boxes (substitution boxes) using GA structures
//! to enable fast cryptanalytic operations (DDT, LAT computation).

use crate::cryptanalysis::boolean_ga::BooleanMultivector;

/// S-box represented using geometric algebra
#[derive(Clone)]
pub struct SBoxGA {
    /// Input/output dimension (in bits)
    pub n: usize,

    /// Lookup table (standard representation)
    /// lut[x] = S(x)
    pub lut: Vec<u8>,

    /// Precomputed multivector representations (if beneficial)
    /// For each input x, store S(x) as a multivector
    pub mv_table: Option<Vec<BooleanMultivector>>,
}

impl SBoxGA {
    /// Create S-box from lookup table
    ///
    /// # Arguments
    ///
    /// * `lut` - Lookup table where lut[x] = S(x)
    /// * `n` - Bit width (must be ≤ 8 for u8)
    pub fn from_lut(lut: Vec<u8>, n: usize) -> Self {
        assert_eq!(lut.len(), 1 << n, "LUT size must be 2^n");
        assert!(n <= 8, "Only up to 8-bit S-boxes supported");

        Self {
            n,
            lut,
            mv_table: None,
        }
    }

    /// Apply S-box to a value
    pub fn apply(&self, x: u8) -> u8 {
        let index = (x as usize) & ((1 << self.n) - 1);
        self.lut[index]
    }

    /// Get S-box size (2^n)
    pub fn size(&self) -> usize {
        1 << self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_sbox() {
        // Identity S-box: S(x) = x
        let lut: Vec<u8> = (0..16).collect();
        let sbox = SBoxGA::from_lut(lut, 4);

        assert_eq!(sbox.apply(5), 5);
        assert_eq!(sbox.apply(10), 10);
    }

    #[test]
    fn test_inversion_sbox() {
        // Bit inversion: S(x) = ~x (for 4 bits)
        let lut: Vec<u8> = (0..16).map(|x| (!x) & 0xF).collect();
        let sbox = SBoxGA::from_lut(lut, 4);

        assert_eq!(sbox.apply(0b0000), 0b1111);
        assert_eq!(sbox.apply(0b1010), 0b0101);
    }
}
