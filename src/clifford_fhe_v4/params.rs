/// V4 Parameters for Packed Layout
///
/// Uses same base parameters as V2/V3 but with adjusted batch sizes.

/// Packed parameter set
pub struct PackedParams {
    /// Ring dimension (1024, 2048, etc)
    pub n: usize,
    
    /// Number of RNS primes
    pub num_primes: usize,
    
    /// Bit sizes of RNS primes
    pub prime_bits: Vec<usize>,
    
    /// Precision bits (determines scale = 2^precision)
    pub precision_bits: usize,
    
    /// Batch size (number of multivectors)
    /// For n=1024: batch_size = 64 (512 slots / 8 components)
    /// For n=2048: batch_size = 128 (1024 slots / 8 components)
    pub batch_size: usize,
}

impl PackedParams {
    /// Standard parameters for N=1024 (matches V2/V3)
    pub fn n1024_standard() -> Self {
        PackedParams {
            n: 1024,
            num_primes: 4,
            prime_bits: vec![60, 60, 60, 60],
            precision_bits: 40,
            batch_size: 64,  // 512 slots / 8 components = 64 multivectors
        }
    }
    
    /// Bootstrap parameters for N=1024 (matches V3)
    pub fn n1024_bootstrap() -> Self {
        PackedParams {
            n: 1024,
            num_primes: 30,  // 1× 60-bit + 29× 45-bit
            prime_bits: {
                let mut bits = vec![60];
                bits.extend(vec![45; 29]);
                bits
            },
            precision_bits: 40,
            batch_size: 64,
        }
    }
    
    /// Large ring dimension for higher throughput
    pub fn n2048_standard() -> Self {
        PackedParams {
            n: 2048,
            num_primes: 4,
            prime_bits: vec![60, 60, 60, 60],
            precision_bits: 40,
            batch_size: 128,  // 1024 slots / 8 components = 128 multivectors
        }
    }
    
    /// Get the number of slots (= n/2 for CKKS)
    pub fn num_slots(&self) -> usize {
        self.n / 2
    }
    
    /// Get the maximum batch size (slots / 8 components)
    pub fn max_batch_size(&self) -> usize {
        self.num_slots() / 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_n1024_batch_size() {
        let params = PackedParams::n1024_standard();
        assert_eq!(params.num_slots(), 512);
        assert_eq!(params.max_batch_size(), 64);
        assert_eq!(params.batch_size, 64);
    }
    
    #[test]
    fn test_n2048_batch_size() {
        let params = PackedParams::n2048_standard();
        assert_eq!(params.num_slots(), 1024);
        assert_eq!(params.max_batch_size(), 128);
        assert_eq!(params.batch_size, 128);
    }
}
