/// Packed Multivector Representation
///
/// A single CKKS ciphertext storing all 8 Clifford algebra components in interleaved slots.
///
/// Slot layout: [s₀, e1₀, e2₀, e3₀, e12₀, e23₀, e31₀, I₀, s₁, e1₁, ...]
///
/// Each batch of 8 consecutive slots represents one complete multivector.
/// This achieves 8× memory reduction compared to V2/V3's component-separate layout.

#[cfg(feature = "v2-gpu-cuda")]
use crate::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext as Ciphertext;

#[cfg(all(feature = "v2-gpu-metal", not(feature = "v2-gpu-cuda")))]
use crate::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCiphertext as Ciphertext;

#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;

/// PackedMultivector: All 8 components in a single ciphertext
///
/// Memory usage: Same as 1 CKKS ciphertext (vs 8 in V2/V3)
/// Batch size: Number of multivectors packed (typically n/8 or n/16)
#[derive(Clone)]
pub struct PackedMultivector {
    /// Single ciphertext with interleaved components
    pub ct: Ciphertext,
    
    /// Number of multivectors packed (n/8 or n/16)
    pub batch_size: usize,
    
    /// Ring dimension (1024, 2048, etc)
    pub n: usize,
    
    /// Number of RNS primes at current level
    pub num_primes: usize,
    
    /// Current level (decreases after rescaling)
    pub level: usize,
    
    /// Current scale (2^precision)
    pub scale: f64,
}

impl PackedMultivector {
    /// Create a new packed multivector from an existing ciphertext
    pub fn new(
        ct: Ciphertext,
        batch_size: usize,
        n: usize,
        num_primes: usize,
        level: usize,
        scale: f64,
    ) -> Self {
        assert!(batch_size * 8 <= n / 2, 
            "Batch size {} × 8 components exceeds n/2 = {}", batch_size, n / 2);
        
        PackedMultivector {
            ct,
            batch_size,
            n,
            num_primes,
            level,
            scale,
        }
    }
    
    /// Get slot index for component i of multivector j
    ///
    /// Slot layout: [s₀, e1₀, e2₀, e3₀, e12₀, e23₀, e31₀, I₀, s₁, e1₁, ...]
    /// 
    /// Component mapping:
    /// - 0: scalar (s)
    /// - 1: e1
    /// - 2: e2
    /// - 3: e3
    /// - 4: e12
    /// - 5: e23
    /// - 6: e31
    /// - 7: pseudoscalar (I)
    pub fn slot_index(batch_idx: usize, component: usize) -> usize {
        assert!(component < 8, "Component must be 0-7");
        batch_idx * 8 + component
    }
    
    /// Get the total number of active slots
    pub fn num_slots(&self) -> usize {
        self.batch_size * 8
    }
    
    /// Check if this packed multivector is compatible with another for operations
    pub fn is_compatible(&self, other: &PackedMultivector) -> bool {
        self.batch_size == other.batch_size
            && self.n == other.n
            && self.level == other.level
            && (self.scale - other.scale).abs() < 1e-6
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_slot_index() {
        // First multivector components
        assert_eq!(PackedMultivector::slot_index(0, 0), 0);  // s₀
        assert_eq!(PackedMultivector::slot_index(0, 1), 1);  // e1₀
        assert_eq!(PackedMultivector::slot_index(0, 7), 7);  // I₀
        
        // Second multivector components
        assert_eq!(PackedMultivector::slot_index(1, 0), 8);   // s₁
        assert_eq!(PackedMultivector::slot_index(1, 1), 9);   // e1₁
        assert_eq!(PackedMultivector::slot_index(1, 7), 15);  // I₁
    }
}
