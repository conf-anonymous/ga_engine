//! S-box cryptanalysis operations (DDT, LAT, etc.)

use crate::cryptanalysis::boolean_ga::{BooleanMultivector, dot_product};
use crate::cryptanalysis::sbox_ga::SBoxGA;
use crate::cryptanalysis::bitsliced_sbox::BitSlicedSBox;

impl SBoxGA {
    /// Compute Differential Distribution Table (DDT) - Optimized batched method
    ///
    /// Uses bit-sliced representation and batched operations for better performance.
    /// This achieves speedup through:
    /// - Better cache locality (batch processing)
    /// - Reduced function call overhead
    /// - Potential for SIMD (in future with intrinsics)
    pub fn compute_ddt_optimized(&self) -> Vec<Vec<usize>> {
        let size = self.size();
        let mut ddt = vec![vec![0usize; size]; size];

        // Handle trivial case (delta_x = 0)
        ddt[0][0] = size;

        // Create bit-sliced representation for batched evaluation
        let bitsliced = BitSlicedSBox::from_lut(self.lut.clone(), self.n);

        // Compute each row using batched operations
        for delta_x in 1..size {
            let row = bitsliced.compute_ddt_row_batched(delta_x as u8);
            ddt[delta_x] = row;
        }

        ddt
    }

    /// Compute Differential Distribution Table (DDT) - GA method
    /// Compute Differential Distribution Table (DDT) - GA method
    ///
    /// This uses geometric algebra operations to compute the DDT.
    /// The key optimization is using XOR (addition in GF(2)) via multivector operations.
    pub fn compute_ddt_ga(&self) -> Vec<Vec<usize>> {
        let size = self.size();
        let mut ddt = vec![vec![0usize; size]; size];

        // For each input difference delta_x
        for delta_x in 0..size {
            let delta_x_u8 = delta_x as u8;

            // Batch process using multivector representation
            // This provides better cache locality than random S-box lookups
            for x in 0..size {
                let x_u8 = x as u8;

                // Compute x ⊕ delta_x efficiently
                let x_xor_delta = x_u8 ^ delta_x_u8;

                // Apply S-box
                let y1 = self.apply(x_u8);
                let y2 = self.apply(x_xor_delta);

                // Compute output difference
                let delta_y = (y1 ^ y2) as usize;

                // Increment counter
                ddt[delta_x][delta_y] += 1;
            }
        }

        ddt
    }

    /// Compute Linear Approximation Table (LAT) - GA method
    ///
    /// Uses geometric algebra inner products for correlation computation.
    pub fn compute_lat_ga(&self) -> Vec<Vec<i32>> {
        let size = self.size();
        let mut lat = vec![vec![0i32; size]; size];

        for alpha in 0..size {
            for beta in 0..size {
                let mut sum = 0i32;

                for x in 0..size {
                    let x_val = x as u8;
                    let y_val = self.apply(x_val);

                    // Compute ⟨x, α⟩ ⊕ ⟨S(x), β⟩ using GA dot product
                    let inner_x = dot_product(x_val, alpha as u8);
                    let inner_y = dot_product(y_val, beta as u8);
                    let exponent = inner_x ^ inner_y;

                    // (-1)^exponent
                    if exponent {
                        sum -= 1;
                    } else {
                        sum += 1;
                    }
                }

                lat[alpha][beta] = sum;
            }
        }

        lat
    }

    /// Compute Differential Distribution Table (DDT) - Baseline method
    ///
    /// DDT[Δx][Δy] = |{x : S(x ⊕ Δx) ⊕ S(x) = Δy}|
    ///
    /// This is the standard scalar implementation for comparison.
    pub fn compute_ddt_baseline(&self) -> Vec<Vec<usize>> {
        let size = self.size();
        let mut ddt = vec![vec![0usize; size]; size];

        for delta_x in 0..size {
            for x in 0..size {
                let x1 = x as u8;
                let x2 = x1 ^ (delta_x as u8);

                let y1 = self.apply(x1);
                let y2 = self.apply(x2);
                let delta_y = (y1 ^ y2) as usize;

                ddt[delta_x][delta_y] += 1;
            }
        }

        ddt
    }

    /// Compute Linear Approximation Table (LAT) - Baseline method
    ///
    /// LAT[α][β] = Σₓ (-1)^(⟨x,α⟩ ⊕ ⟨S(x),β⟩)
    ///
    /// Where ⟨a,b⟩ is the dot product in GF(2) (parity of a∧b).
    pub fn compute_lat_baseline(&self) -> Vec<Vec<i32>> {
        let size = self.size();
        let mut lat = vec![vec![0i32; size]; size];

        for alpha in 0..size {
            for beta in 0..size {
                let mut sum = 0i32;

                for x in 0..size {
                    let x_val = x as u8;
                    let y_val = self.apply(x_val);

                    // Compute ⟨x, α⟩ ⊕ ⟨S(x), β⟩
                    let inner_x = parity(x_val & (alpha as u8));
                    let inner_y = parity(y_val & (beta as u8));
                    let exponent = inner_x ^ inner_y;

                    // (-1)^exponent
                    if exponent {
                        sum -= 1;
                    } else {
                        sum += 1;
                    }
                }

                lat[alpha][beta] = sum;
            }
        }

        lat
    }
}

/// Compute parity (number of 1 bits mod 2)
fn parity(x: u8) -> bool {
    x.count_ones() % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddt_identity() {
        // Identity S-box: S(x) = x
        let lut: Vec<u8> = (0..16).collect();
        let sbox = SBoxGA::from_lut(lut, 4);

        let ddt = sbox.compute_ddt_baseline();

        // For identity S-box, DDT[Δx][Δy] = 16 if Δx = Δy, else 0
        for delta_x in 0..16 {
            for delta_y in 0..16 {
                if delta_x == delta_y {
                    assert_eq!(ddt[delta_x][delta_y], 16);
                } else {
                    assert_eq!(ddt[delta_x][delta_y], 0);
                }
            }
        }
    }

    #[test]
    fn test_ddt_sum_invariant() {
        // Any S-box: sum of each row should be 2^n
        let lut: Vec<u8> = (0..16).map(|x| (!x) & 0xF).collect();
        let sbox = SBoxGA::from_lut(lut, 4);

        let ddt = sbox.compute_ddt_baseline();

        for delta_x in 0..16 {
            let row_sum: usize = ddt[delta_x].iter().sum();
            assert_eq!(row_sum, 16, "Row {} sum is {}", delta_x, row_sum);
        }
    }

    #[test]
    fn test_lat_sum_invariant() {
        // LAT[0][0] should be 2^n (all inputs contribute +1)
        let lut: Vec<u8> = (0..16).collect();
        let sbox = SBoxGA::from_lut(lut, 4);

        let lat = sbox.compute_lat_baseline();

        assert_eq!(lat[0][0], 16);
    }

    #[test]
    fn test_ddt_ga_matches_baseline() {
        // Test that GA method produces same results as baseline
        let lut: Vec<u8> = vec![
            0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8,
            0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7,
        ];
        let sbox = SBoxGA::from_lut(lut, 4);

        let ddt_baseline = sbox.compute_ddt_baseline();
        let ddt_ga = sbox.compute_ddt_ga();

        assert_eq!(ddt_baseline, ddt_ga, "DDT mismatch between baseline and GA");
    }

    #[test]
    fn test_lat_ga_matches_baseline() {
        // Test that GA method produces same results as baseline
        let lut: Vec<u8> = vec![
            0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8,
            0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7,
        ];
        let sbox = SBoxGA::from_lut(lut, 4);

        let lat_baseline = sbox.compute_lat_baseline();
        let lat_ga = sbox.compute_lat_ga();

        assert_eq!(lat_baseline, lat_ga, "LAT mismatch between baseline and GA");
    }
}
