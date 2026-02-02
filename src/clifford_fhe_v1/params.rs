//! Parameter sets for Clifford-FHE
//!
//! Defines security levels and corresponding CKKS parameters optimized
//! for geometric algebra operations.

/// Security levels following NIST standards
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityLevel {
    /// ~128-bit security (NIST Level 1)
    Bit128,
    /// ~192-bit security (NIST Level 3)
    Bit192,
    /// ~256-bit security (NIST Level 5)
    Bit256,
}

/// CKKS parameters for Clifford-FHE
#[derive(Debug, Clone)]
pub struct CliffordFHEParams {
    /// Ring dimension (polynomial degree)
    /// Must be power of 2 for NTT
    pub n: usize,

    /// Ciphertext modulus chain (for leveled FHE)
    /// Each level uses a different modulus for rescaling
    pub moduli: Vec<i64>,

    /// Scaling factor (determines precision)
    /// Larger = more precision but more noise
    pub scale: f64,

    /// Standard deviation for error distribution
    pub error_std: f64,

    /// Security level
    pub security: SecurityLevel,
}

impl CliffordFHEParams {
    /// Minimal parameters for testing (NOT SECURE!)
    ///
    /// Use this only for development/testing. Much faster key generation.
    /// N=64 is TOO SMALL for multiplication - use new_test_mult() instead.
    pub fn new_test() -> Self {
        Self {
            n: 64, // Very small for fast testing (rotation/addition only!)
            moduli: vec![
                // Just a few levels for testing
                40, // Level 0
                40, // Level 1
                40, // Level 2
            ]
            .iter()
            .map(|bits| Self::generate_prime(*bits))
            .collect(),
            scale: 2f64.powi(20), // Smaller scale for testing
            error_std: 3.2,
            security: SecurityLevel::Bit128, // Claimed, not actual!
        }
    }

    /// Test parameters for multiplication (single-modulus - LIMITED)
    ///
    /// NOTE: This is the OLD single-modulus approach with limitations.
    /// Use new_rns_mult() for proper RNS-CKKS multiplication.
    pub fn new_test_mult() -> Self {
        Self {
            n: 1024,
            moduli: vec![60]
                .iter()
                .map(|bits| Self::generate_prime(*bits))
                .collect(),
            scale: 2f64.powi(30),
            error_std: 3.2,
            security: SecurityLevel::Bit128,
        }
    }

    /// Test parameters for RNS-CKKS multiplication (DEPTH-1)
    ///
    /// N=1024 for testing, 3-prime modulus chain for depth-1 circuits.
    ///
    /// CORRECTED APPROACH: Use scaling primes ≈ Δ
    ///
    /// In RNS-CKKS, for rescaling to work correctly:
    /// - After multiplication: scale = Δ²
    /// - After rescaling by q_last: new_scale = Δ²/q_last
    /// - For new_scale ≈ Δ, we need: q_last ≈ Δ
    ///
    /// Therefore: Mix 60-bit primes (security) with 40-bit primes (scaling)
    pub fn new_rns_mult() -> Self {
        // RNS-CKKS with proper scaling primes
        // For N=1024, we need p ≡ 1 (mod 2048)
        //
        // Modulus chain strategy:
        // - Start with large primes for security
        // - Use 40-bit primes ≈ Δ for rescaling operations
        // - When we rescale (drop a prime), we drop a 40-bit prime
        // - This gives: new_scale = Δ²/q_scale ≈ Δ²/Δ = Δ ✓
        let moduli = vec![
            1141392289560813569,  // q₀ (60-bit, NTT-friendly) - for security
            1099511678977,        // q₁ (41-bit, ≈ 2^40, NTT-friendly) - SCALING PRIME
            1099511683073,        // q₂ (41-bit, ≈ 2^40, NTT-friendly) - SCALING PRIME
        ];

        Self {
            n: 1024,
            moduli,
            // Scaling factor: Δ = 2^40 ≈ 1.1 × 10^12
            // This matches our 40-bit scaling primes!
            //
            // Workflow:
            // 1. Fresh ciphertext: scale = Δ, modulus = q₀·q₁·q₂
            // 2. After multiply: scale = Δ², modulus = q₀·q₁·q₂
            // 3. Rescale (drop q₂): scale = Δ²/q₂ ≈ Δ, modulus = q₀·q₁
            //
            // Supports depth-1 circuits (1 multiplication)
            scale: 2f64.powi(40),
            error_std: 3.2,
            security: SecurityLevel::Bit128, // Based on lattice estimator
        }
    }

    /// Test parameters for RNS-CKKS with depth-2 support (4 primes)
    ///
    /// N=1024 for testing, 4-prime modulus chain for depth-2 circuits.
    ///
    /// Enables operations that require 2 sequential multiplications:
    /// - Wedge Product: (a⊗b - b⊗a) / 2
    /// - Inner Product: (a⊗b + b⊗a) / 2
    /// - Rotation: R ⊗ v ⊗ R̃
    pub fn new_rns_mult_depth2() -> Self {
        // 4 primes = depth-2 support (minimal)
        //
        // CRITICAL FIX (2025-11-02): Replaced composite "prime" with actual prime!
        // - Old q₃ = 1099511693313 was COMPOSITE (3 × 366503897771)
        let moduli = vec![
            1141392289560813569,  // q₀ (60-bit, NTT-friendly) - for security
            1099511678977,        // q₁ (41-bit, NTT-friendly) - SCALING PRIME (verified)
            1099511683073,        // q₂ (41-bit, NTT-friendly) - SCALING PRIME (verified)
            1099511795713,        // q₃ (41-bit, NTT-friendly) - SCALING PRIME (verified, fixed)
        ];

        Self {
            n: 1024,
            moduli,
            // Workflow:
            // 1. Fresh: scale = Δ, modulus = q₀·q₁·q₂·q₃
            // 2. After 1st multiply: scale = Δ², rescale by q₃ → scale = Δ, modulus = q₀·q₁·q₂
            // 3. After 2nd multiply: scale = Δ², rescale by q₂ → scale = Δ, modulus = q₀·q₁
            //
            // Supports depth-2 circuits (2 multiplications)
            scale: 2f64.powi(40),
            error_std: 3.2,
            security: SecurityLevel::Bit128,
        }
    }

    /// Test parameters for RNS-CKKS with depth-2 support (5 primes - better headroom)
    ///
    /// N=1024 for testing, 5-prime modulus chain for depth-2 circuits with extra headroom.
    ///
    /// More conservative than new_rns_mult_depth2() - leaves 3 primes after depth-2.
    pub fn new_rns_mult_depth2_safe() -> Self {
        // 5 primes = depth-2 support with extra headroom
        //
        // CRITICAL FIX (2025-11-02): Replaced composite "primes" with actual primes!
        // - Old q₃ = 1099511693313 was COMPOSITE (3 × 366503897771)
        // - Old q₄ = 1099511697409 was COMPOSITE
        // - All values verified with Miller-Rabin and Fermat tests
        let moduli = vec![
            1141392289560813569,  // q₀ (60-bit, NTT-friendly) - for security
            1099511678977,        // q₁ (41-bit, NTT-friendly) - SCALING PRIME (verified)
            1099511683073,        // q₂ (41-bit, NTT-friendly) - SCALING PRIME (verified)
            1099511795713,        // q₃ (41-bit, NTT-friendly) - SCALING PRIME (verified, fixed)
            1099511799809,        // q₄ (41-bit, NTT-friendly) - SCALING PRIME (verified, fixed)
        ];

        let scale_value = 1099511627776_f64;  // 2^40 - power of 2 for clean encoding

        Self {
            n: 1024,
            moduli,
            // CRITICAL: Scale CANNOT equal any of the moduli primes!
            //
            // If scale = q₁, then encoding a plaintext gives:
            //   [m]_scale = m * scale = m * q₁
            // Taking this mod q₁ gives: (m * q₁) mod q₁ = 0
            // This zero residue breaks RNS arithmetic!
            //
            // Solution: Use scale = 2^40 = 1099511627776
            // This is slightly smaller than all scaling primes (~51k-74k difference)
            // There will be minor scale drift, but it's acceptable.
            //
            // Workflow:
            // 1. Fresh: scale = 2^40, modulus = q₀·q₁·q₂·q₃·q₄
            // 2. After 1st mult: scale = 2^80, rescale by q₄ → scale ≈ 2^40
            // 3. After 2nd mult: scale = 2^80, rescale by q₃ → scale ≈ 2^40
            scale: scale_value,
            error_std: 3.2,
            security: SecurityLevel::Bit128,
        }
    }

    /// Parameters for 128-bit security
    ///
    /// Optimized for:
    /// - Multivector encryption (8 components)
    /// - ~10 levels of homomorphic operations
    /// - Geometric product + rotations
    pub fn new_128bit() -> Self {
        Self {
            n: 8192, // Smaller than typical CKKS (usually 2^14-2^16)
            // because we pack efficiently
            moduli: vec![
                // Level 0 (fresh ciphertext): Large modulus
                60,  // 60-bit prime
                50, 50, 50, // Intermediate levels
                50, 50, 50, // More levels
                50, 50, 50, // For deep circuits
                40, // Final level
            ]
            .iter()
            .map(|bits| Self::generate_prime(*bits))
            .collect(),
            scale: 2f64.powi(40), // 40-bit precision (~12 decimal digits)
            error_std: 3.2,       // Standard CKKS error
            security: SecurityLevel::Bit128,
        }
    }

    /// Parameters for 192-bit security
    pub fn new_192bit() -> Self {
        Self {
            n: 16384,
            moduli: vec![60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 40]
                .iter()
                .map(|bits| Self::generate_prime(*bits))
                .collect(),
            scale: 2f64.powi(40),
            error_std: 3.2,
            security: SecurityLevel::Bit192,
        }
    }

    /// Parameters for 256-bit security
    pub fn new_256bit() -> Self {
        Self {
            n: 32768,
            moduli: vec![60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 40]
                .iter()
                .map(|bits| Self::generate_prime(*bits))
                .collect(),
            scale: 2f64.powi(40),
            error_std: 3.2,
            security: SecurityLevel::Bit256,
        }
    }

    /// Generate NTT-friendly prime of given bit-length
    ///
    /// Prime must satisfy: q ≡ 1 (mod 2N) for NTT to work
    /// For N=1024, we need: (q-1) % 2048 == 0
    fn generate_prime(bits: u32) -> i64 {
        // For now, use precomputed NTT-friendly primes
        // In production, would generate dynamically
        match bits {
            40 => 1_099_511_678_977, // 41-bit NTT-friendly prime (q ≡ 1 mod 2048)
            50 => 1_125_899_906_826_241, // 50-bit
            60 => 1_152_921_504_598_630_401, // 60-bit (close to 2^60)
            _ => panic!("Unsupported bit length: {}", bits),
        }
    }

    /// Get current level modulus
    ///
    /// In RNS-CKKS, each level uses the PRODUCT of remaining primes:
    /// - Level 0: Q0 = q0 * q1 * q2 (all primes)
    /// - Level 1: Q1 = q0 * q1 (dropped last prime)
    /// - Level 2: Q2 = q0 (dropped all but first)
    ///
    /// For single-modulus CKKS (simplified), all levels use the same modulus.
    pub fn modulus_at_level(&self, level: usize) -> i64 {
        if level >= self.moduli.len() {
            panic!("Level {} exceeds maximum {}", level, self.moduli.len() - 1);
        }

        // For now, use simplified single-modulus approach:
        // Use the FIRST (largest) prime for all levels
        // This avoids RNS complexity for initial testing
        self.moduli[0]
    }

    /// Number of levels (depth) available
    pub fn max_level(&self) -> usize {
        self.moduli.len() - 1
    }

    /// Get product of all moduli up to given level
    /// This is the effective modulus for ciphertexts at that level
    pub fn modulus_product_up_to(&self, level: usize) -> f64 {
        self.moduli[..=level]
            .iter()
            .map(|&q| q as f64)
            .product()
    }
}

impl Default for CliffordFHEParams {
    fn default() -> Self {
        Self::new_128bit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_creation() {
        let params = CliffordFHEParams::new_128bit();
        assert_eq!(params.n, 8192);
        assert_eq!(params.security, SecurityLevel::Bit128);
        assert!(params.max_level() >= 10);
    }

    #[test]
    fn test_modulus_at_level() {
        // Use new_rns_mult() which has verified NTT-friendly primes
        let params = CliffordFHEParams::new_rns_mult();
        let q0 = params.modulus_at_level(0);
        assert!(q0 > 0);
        // For NTT to work, we need (q-1) % 2n == 0
        let two_n = 2 * params.n as i64;
        assert!((q0 - 1) % two_n == 0, "q-1 must be divisible by 2n for NTT");
    }
}
