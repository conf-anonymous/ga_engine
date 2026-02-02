/// SIMD Batching for Clifford FHE
///
/// Implements slot packing/unpacking to process 512 multivectors in parallel:
/// - Each multivector has 8 components (Cl(3,0))
/// - Each ciphertext has 512 slots
/// - Use 8 ciphertexts to encode 512 multivectors (one ciphertext per component)
///
/// This achieves 512× throughput multiplier compared to single-sample encryption.

use super::clifford_encoding::Multivector3D;
use std::f64::consts::PI;

/// Batch of multivectors encoded across 8 ciphertexts
///
/// Structure:
/// - `components[0]`: 512 scalar values (m₀)
/// - `components[1]`: 512 e₁ values (m₁)
/// - `components[2]`: 512 e₂ values (m₂)
/// - `components[3]`: 512 e₃ values (m₃)
/// - `components[4]`: 512 e₁₂ values (m₄)
/// - `components[5]`: 512 e₁₃ values (m₅)
/// - `components[6]`: 512 e₂₃ values (m₆)
/// - `components[7]`: 512 e₁₂₃ values (m₇)
#[derive(Debug, Clone)]
pub struct BatchedMultivectors {
    pub components: [Vec<f64>; 8],
    pub batch_size: usize,
}

impl BatchedMultivectors {
    /// Maximum batch size (determined by polynomial degree N=1024)
    pub const MAX_BATCH_SIZE: usize = 512;

    /// Create new batched multivectors from a slice
    ///
    /// # Panics
    /// Panics if batch size exceeds MAX_BATCH_SIZE (512)
    pub fn from_multivectors(multivectors: &[Multivector3D]) -> Self {
        assert!(
            multivectors.len() <= Self::MAX_BATCH_SIZE,
            "Batch size {} exceeds maximum {}",
            multivectors.len(),
            Self::MAX_BATCH_SIZE
        );

        let batch_size = multivectors.len();

        // Initialize 8 component vectors
        let mut components: [Vec<f64>; 8] = Default::default();
        for comp in &mut components {
            *comp = Vec::with_capacity(batch_size);
        }

        // Pack multivectors into component-wise vectors (Structure of Arrays)
        for mv in multivectors {
            for (comp_idx, comp_vec) in components.iter_mut().enumerate() {
                comp_vec.push(mv.components[comp_idx]);
            }
        }

        BatchedMultivectors {
            components,
            batch_size,
        }
    }

    /// Extract individual multivectors from batch
    pub fn to_multivectors(&self) -> Vec<Multivector3D> {
        let mut result = Vec::with_capacity(self.batch_size);

        for i in 0..self.batch_size {
            let mut mv_components = [0.0; 8];
            for (comp_idx, comp_vec) in self.components.iter().enumerate() {
                mv_components[comp_idx] = comp_vec[i];
            }
            result.push(Multivector3D::new(mv_components));
        }

        result
    }

    /// Pad batch to full 512 slots with zeros
    ///
    /// Required for encryption since ciphertexts always have 512 slots.
    /// Padding slots are ignored during unpacking.
    pub fn pad_to_full_slots(&mut self) {
        let padding = Self::MAX_BATCH_SIZE - self.batch_size;
        if padding > 0 {
            for comp_vec in &mut self.components {
                comp_vec.resize(Self::MAX_BATCH_SIZE, 0.0);
            }
        }
    }

    /// Create empty batch (all zeros)
    pub fn zeros(batch_size: usize) -> Self {
        assert!(
            batch_size <= Self::MAX_BATCH_SIZE,
            "Batch size {} exceeds maximum {}",
            batch_size,
            Self::MAX_BATCH_SIZE
        );

        let components = [
            vec![0.0; batch_size],
            vec![0.0; batch_size],
            vec![0.0; batch_size],
            vec![0.0; batch_size],
            vec![0.0; batch_size],
            vec![0.0; batch_size],
            vec![0.0; batch_size],
            vec![0.0; batch_size],
        ];

        BatchedMultivectors {
            components,
            batch_size,
        }
    }
}

/// Batched geometric product (component-wise across slots)
///
/// Computes geometric product for each of 512 multivector pairs in parallel.
/// Currently uses simplified dot-product version.
///
/// # Arguments
/// * `a` - First batch of multivectors
/// * `b` - Second batch of multivectors
///
/// # Returns
/// Batched result of geometric products
pub fn batched_geometric_product(
    a: &BatchedMultivectors,
    b: &BatchedMultivectors,
) -> BatchedMultivectors {
    assert_eq!(
        a.batch_size, b.batch_size,
        "Batch sizes must match: {} vs {}",
        a.batch_size, b.batch_size
    );

    let batch_size = a.batch_size;

    // Simplified geometric product: dot product of 8D vectors
    let mut result_scalars = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let mut dot = 0.0;
        for comp_idx in 0..8 {
            dot += a.components[comp_idx][i] * b.components[comp_idx][i];
        }
        result_scalars.push(dot);
    }

    // Result is scalar multivectors (only component 0 non-zero)
    let mut components = [
        result_scalars,
        vec![0.0; batch_size],
        vec![0.0; batch_size],
        vec![0.0; batch_size],
        vec![0.0; batch_size],
        vec![0.0; batch_size],
        vec![0.0; batch_size],
        vec![0.0; batch_size],
    ];

    BatchedMultivectors {
        components,
        batch_size,
    }
}

/// Batched ReLU activation
pub fn batched_relu(input: &BatchedMultivectors) -> BatchedMultivectors {
    let mut result = input.clone();

    // Apply ReLU to each component independently
    for comp_vec in &mut result.components {
        for val in comp_vec.iter_mut() {
            *val = val.max(0.0);
        }
    }

    result
}

/// Batched addition
pub fn batched_add(a: &BatchedMultivectors, b: &BatchedMultivectors) -> BatchedMultivectors {
    assert_eq!(a.batch_size, b.batch_size);

    let batch_size = a.batch_size;
    let mut components: [Vec<f64>; 8] = Default::default();

    for comp_idx in 0..8 {
        let mut comp_vec = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            comp_vec.push(a.components[comp_idx][i] + b.components[comp_idx][i]);
        }
        components[comp_idx] = comp_vec;
    }

    BatchedMultivectors {
        components,
        batch_size,
    }
}

/// Batched scalar multiplication
pub fn batched_scalar_mul(scalar: f64, input: &BatchedMultivectors) -> BatchedMultivectors {
    let mut result = input.clone();

    for comp_vec in &mut result.components {
        for val in comp_vec.iter_mut() {
            *val *= scalar;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_identity() {
        // Create 10 random multivectors
        let original: Vec<Multivector3D> = (0..10)
            .map(|i| {
                Multivector3D::new([
                    i as f64,
                    i as f64 + 0.1,
                    i as f64 + 0.2,
                    i as f64 + 0.3,
                    i as f64 + 0.4,
                    i as f64 + 0.5,
                    i as f64 + 0.6,
                    i as f64 + 0.7,
                ])
            })
            .collect();

        // Pack and unpack
        let batched = BatchedMultivectors::from_multivectors(&original);
        let unpacked = batched.to_multivectors();

        // Verify identity
        assert_eq!(unpacked.len(), original.len());
        for (orig, result) in original.iter().zip(unpacked.iter()) {
            for i in 0..8 {
                assert!((orig.components[i] - result.components[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_max_batch_size() {
        let multivectors: Vec<Multivector3D> = (0..512)
            .map(|i| Multivector3D::new([i as f64; 8]))
            .collect();

        let batched = BatchedMultivectors::from_multivectors(&multivectors);
        assert_eq!(batched.batch_size, 512);
    }

    #[test]
    #[should_panic(expected = "exceeds maximum")]
    fn test_exceed_max_batch_size() {
        let multivectors: Vec<Multivector3D> = (0..513)
            .map(|i| Multivector3D::new([i as f64; 8]))
            .collect();

        BatchedMultivectors::from_multivectors(&multivectors);
    }

    #[test]
    fn test_padding() {
        let multivectors: Vec<Multivector3D> = (0..10)
            .map(|i| Multivector3D::new([i as f64; 8]))
            .collect();

        let mut batched = BatchedMultivectors::from_multivectors(&multivectors);
        assert_eq!(batched.components[0].len(), 10);

        batched.pad_to_full_slots();
        assert_eq!(batched.components[0].len(), 512);

        // Verify padding is zeros
        for comp_vec in &batched.components {
            for &val in &comp_vec[10..512] {
                assert_eq!(val, 0.0);
            }
        }
    }

    #[test]
    fn test_batched_geometric_product() {
        let a = vec![
            Multivector3D::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            Multivector3D::new([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        ];
        let b = vec![
            Multivector3D::new([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            Multivector3D::new([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        ];

        let batched_a = BatchedMultivectors::from_multivectors(&a);
        let batched_b = BatchedMultivectors::from_multivectors(&b);

        let result = batched_geometric_product(&batched_a, &batched_b);

        // Expected dot products: [36.0, 88.0]
        // First: (1+2+3+4+5+6+7+8) * 1 = 36
        // Second: (2+3+4+5+6+7+8+9) * 2 = 44 * 2 = 88
        assert_eq!(result.batch_size, 2);
        assert!((result.components[0][0] - 36.0).abs() < 1e-10);
        assert!((result.components[0][1] - 88.0).abs() < 1e-10);
    }

    #[test]
    fn test_batched_relu() {
        let multivectors = vec![
            Multivector3D::new([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]),
            Multivector3D::new([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]),
        ];

        let batched = BatchedMultivectors::from_multivectors(&multivectors);
        let result = batched_relu(&batched);

        // Verify ReLU applied
        let expected_0 = [1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0];
        let expected_1 = [0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0];

        let unpacked = result.to_multivectors();
        for (i, &exp) in expected_0.iter().enumerate() {
            assert_eq!(unpacked[0].components[i], exp);
        }
        for (i, &exp) in expected_1.iter().enumerate() {
            assert_eq!(unpacked[1].components[i], exp);
        }
    }

    #[test]
    fn test_batched_add() {
        let a = vec![Multivector3D::new([1.0; 8])];
        let b = vec![Multivector3D::new([2.0; 8])];

        let batched_a = BatchedMultivectors::from_multivectors(&a);
        let batched_b = BatchedMultivectors::from_multivectors(&b);

        let result = batched_add(&batched_a, &batched_b);
        let unpacked = result.to_multivectors();

        for &val in &unpacked[0].components {
            assert_eq!(val, 3.0);
        }
    }
}
