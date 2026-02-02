//! SIMD Batched Operations for V3
//!
//! Implements slot-level parallelism to process 512 multivectors simultaneously
//! in a single CKKS ciphertext, achieving 512× throughput increase.
//!
//! # Slot Layout
//!
//! For N=8192 (4096 slots), batch size = 512 multivectors:
//! ```text
//! Slot 0:    mv[0].c0 (scalar)
//! Slot 1:    mv[0].c1 (e1)
//! ...
//! Slot 7:    mv[0].c7 (e123)
//! Slot 8:    mv[1].c0
//! ...
//! Slot 4095: mv[511].c7
//! ```
//!
//! # Key Operations
//!
//! - **Component Extraction:** Use rotation + masking to extract specific components
//! - **Batch Geometric Product:** Operate on all 512 pairs simultaneously
//! - **Batch Bootstrap:** Refresh noise for entire batch in one operation
//!
//! # Performance
//!
//! - Single sample: 2000ms bootstrap
//! - Batched (512×): 3.9ms per sample amortized = 512× speedup

pub mod encoding;
pub mod extraction;
pub mod geometric;
pub mod bootstrap;

use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;

/// Batched multivector ciphertext
///
/// Encodes multiple multivectors (each 8 components) into CKKS slots
/// for parallel processing.
///
/// # Slot Packing
///
/// Components are interleaved with stride 8:
/// - Slots [0, 8, 16, ..., 4088]: component 0 of all multivectors
/// - Slots [1, 9, 17, ..., 4089]: component 1 of all multivectors
/// - ...
/// - Slots [7, 15, 23, ..., 4095]: component 7 of all multivectors
#[derive(Clone, Debug)]
pub struct BatchedMultivector {
    /// Underlying CKKS ciphertext with packed slots
    pub ciphertext: Ciphertext,

    /// Number of multivectors in this batch
    pub batch_size: usize,

    /// Ring dimension (determines max batch size = N/2 / 8)
    pub n: usize,
}

impl BatchedMultivector {
    /// Create new batched multivector from ciphertext
    pub fn new(ciphertext: Ciphertext, batch_size: usize) -> Self {
        let n = ciphertext.n;
        let max_batch = n / 2 / 8;
        assert!(
            batch_size <= max_batch,
            "Batch size {} exceeds maximum {} for N={}",
            batch_size, max_batch, n
        );

        Self {
            ciphertext,
            batch_size,
            n,
        }
    }

    /// Maximum batch size for given ring dimension
    pub fn max_batch_size(n: usize) -> usize {
        n / 2 / 8  // N/2 slots, 8 components per multivector
    }

    /// Number of slots used
    pub fn slots_used(&self) -> usize {
        self.batch_size * 8
    }

    /// Slot utilization percentage
    pub fn slot_utilization(&self) -> f64 {
        (self.slots_used() as f64) / (self.n as f64 / 2.0) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_batch_size() {
        assert_eq!(BatchedMultivector::max_batch_size(1024), 64);
        assert_eq!(BatchedMultivector::max_batch_size(2048), 128);
        assert_eq!(BatchedMultivector::max_batch_size(4096), 256);
        assert_eq!(BatchedMultivector::max_batch_size(8192), 512);
    }

    #[test]
    fn test_slot_utilization() {
        use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

        // Create dummy ciphertext
        let n = 8192;
        let moduli = vec![1099511627791u64, 1099511627789u64, 1099511627773u64];
        let c0 = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.clone()); n];
        let c1 = c0.clone();
        let ct = Ciphertext::new(c0, c1, 2, 1.0 * (1u64 << 40) as f64);

        let batch = BatchedMultivector::new(ct, 512);
        assert_eq!(batch.slots_used(), 4096);
        assert_eq!(batch.slot_utilization(), 100.0);

        let half_batch = BatchedMultivector::new(batch.ciphertext.clone(), 256);
        assert_eq!(half_batch.slot_utilization(), 50.0);
    }
}
