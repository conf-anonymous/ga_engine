//! Bit-sliced S-box representation for SIMD-friendly operations
//!
//! This module implements bit-slicing, where we process multiple S-box
//! evaluations in parallel by organizing data by bit position rather than
//! by input value.
//!
//! # Bit-Slicing Concept
//!
//! Instead of processing each input sequentially:
//!   input[0] -> sbox -> output[0]
//!   input[1] -> sbox -> output[1]
//!   ...
//!
//! We organize by bit positions:
//!   bit0[0..31] = all bit-0 values from 32 inputs
//!   bit1[0..31] = all bit-1 values from 32 inputs
//!   ...
//!
//! This allows SIMD operations across 32 inputs simultaneously.

/// Number of inputs processed in parallel
const BATCH_SIZE: usize = 32;

/// Bit-sliced S-box for parallel evaluation
///
/// Stores the S-box in a format optimized for batch evaluation.
/// Each layer represents one bit position across multiple inputs.
pub struct BitSlicedSBox {
    /// Number of bits (typically 8)
    n_bits: usize,

    /// Original lookup table (for fallback)
    lut: Vec<u8>,

    /// Bit-sliced representation
    /// For each possible input pattern (256 for 8-bit),
    /// store which output bits are set across batch
    /// This is precomputed for fast lookup
    bit_layers: Option<Vec<[u32; 8]>>,
}

impl BitSlicedSBox {
    /// Create bit-sliced S-box from lookup table
    pub fn from_lut(lut: Vec<u8>, n_bits: usize) -> Self {
        assert_eq!(lut.len(), 1 << n_bits);
        assert!(n_bits <= 8, "Only up to 8-bit S-boxes supported");

        Self {
            n_bits,
            lut,
            bit_layers: None,
        }
    }

    /// Apply S-box to a batch of inputs (up to 32)
    ///
    /// This uses scalar lookups but processes them efficiently.
    /// For true SIMD, we'd need platform-specific intrinsics.
    pub fn apply_batch(&self, inputs: &[u8]) -> Vec<u8> {
        assert!(inputs.len() <= BATCH_SIZE);

        inputs.iter().map(|&x| {
            let idx = (x as usize) & ((1 << self.n_bits) - 1);
            self.lut[idx]
        }).collect()
    }

    /// Compute DDT row for a specific delta_x using batched operations
    ///
    /// This processes multiple x values at once for better cache locality.
    pub fn compute_ddt_row_batched(&self, delta_x: u8) -> Vec<usize> {
        let size = 1 << self.n_bits;
        let mut row = vec![0usize; size];

        // Process in batches for better cache behavior
        for batch_start in (0..size).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(size);
            let batch_len = batch_end - batch_start;

            // Prepare batch of inputs
            let inputs: Vec<u8> = (batch_start..batch_end)
                .map(|x| x as u8)
                .collect();

            // Apply S-box to original inputs
            let outputs1 = self.apply_batch(&inputs);

            // Prepare XORed inputs
            let inputs_xor: Vec<u8> = inputs.iter()
                .map(|&x| x ^ delta_x)
                .collect();

            // Apply S-box to XORed inputs
            let outputs2 = self.apply_batch(&inputs_xor);

            // Compute differences and accumulate
            for i in 0..batch_len {
                let delta_y = (outputs1[i] ^ outputs2[i]) as usize;
                row[delta_y] += 1;
            }
        }

        row
    }
}

/// SIMD-friendly XOR operation (scalar fallback)
///
/// In a full implementation, this would use AVX2/NEON intrinsics.
#[inline]
pub fn xor_batch(a: &[u8], b: &[u8]) -> Vec<u8> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x ^ y).collect()
}

/// SIMD-friendly XOR with scalar broadcast
#[inline]
pub fn xor_scalar_batch(a: &[u8], scalar: u8) -> Vec<u8> {
    a.iter().map(|&x| x ^ scalar).collect()
}

/// Histogram accumulation for batch of values
///
/// Updates histogram with multiple values efficiently.
#[inline]
pub fn histogram_accumulate(hist: &mut [usize], values: &[u8]) {
    for &val in values {
        hist[val as usize] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitsliced_sbox_identity() {
        let lut: Vec<u8> = (0..16).collect();
        let sbox = BitSlicedSBox::from_lut(lut, 4);

        let inputs = vec![0, 1, 5, 10, 15];
        let outputs = sbox.apply_batch(&inputs);

        assert_eq!(outputs, inputs);
    }

    #[test]
    fn test_bitsliced_sbox_inversion() {
        let lut: Vec<u8> = (0..16).map(|x| (!x) & 0xF).collect();
        let sbox = BitSlicedSBox::from_lut(lut, 4);

        let inputs = vec![0b0000, 0b1111, 0b1010];
        let outputs = sbox.apply_batch(&inputs);

        assert_eq!(outputs, vec![0b1111, 0b0000, 0b0101]);
    }

    #[test]
    fn test_xor_batch() {
        let a = vec![0b1010, 0b1100, 0b1111];
        let b = vec![0b0011, 0b0101, 0b1111];
        let result = xor_batch(&a, &b);

        assert_eq!(result, vec![0b1001, 0b1001, 0b0000]);
    }

    #[test]
    fn test_xor_scalar_batch() {
        let a = vec![0b0000, 0b0001, 0b0010];
        let result = xor_scalar_batch(&a, 0b1111);

        assert_eq!(result, vec![0b1111, 0b1110, 0b1101]);
    }

    #[test]
    fn test_histogram_accumulate() {
        let mut hist = vec![0usize; 16];
        let values = vec![0, 1, 1, 5, 5, 5, 10];

        histogram_accumulate(&mut hist, &values);

        assert_eq!(hist[0], 1);
        assert_eq!(hist[1], 2);
        assert_eq!(hist[5], 3);
        assert_eq!(hist[10], 1);
    }

    #[test]
    fn test_ddt_row_batched() {
        let lut: Vec<u8> = (0..16).collect(); // Identity
        let sbox = BitSlicedSBox::from_lut(lut, 4);

        let row = sbox.compute_ddt_row_batched(0x5);

        // For identity S-box, DDT[Δx][Δy] = 16 if Δx = Δy, else 0
        assert_eq!(row[0x5], 16);
        for i in 0..16 {
            if i != 0x5 {
                assert_eq!(row[i], 0, "row[{}] should be 0", i);
            }
        }
    }
}
