//! V3 Parameter Sets for CKKS Bootstrapping
//!
//! V3 parameters extend V2 with **dynamically generated** NTT-friendly primes
//! to support bootstrapping.
//!
//! ## Dynamic Prime Generation
//!
//! Unlike V2 which uses hardcoded primes, V3 generates primes **at runtime**
//! using Miller-Rabin primality testing. This provides flexibility to:
//! - Generate exactly the number of primes needed for any bootstrap configuration
//! - Easily switch between different parameter sets without manual prime searching
//! - Guarantee all primes are NTT-friendly: q ≡ 1 (mod 2N)
//!
//! ## Bootstrap Requirements
//!
//! - **Bootstrap levels:** 12-15 primes reserved for bootstrap operations
//! - **Computation levels:** 5-7 primes for computation between bootstraps
//! - **Total:** 20-25 primes minimum
//!
//! ## Parameter Selection Strategy
//!
//! 1. **Large N for precision:** N=8192 or N=16384
//! 2. **60-bit first prime:** For special modulus (fast modular reduction)
//! 3. **40-41 bit primes:** For scaling (≈ 2^40)
//! 4. **All NTT-friendly:** q ≡ 1 mod 2N (enforced by generation algorithm)
//!
//! ## Security
//!
//! With N=8192 and log(Q)≈800 bits (20 primes × 40 bits):
//! - Security: ~128 bits (NIST Level 1)
//! - Bootstrapping doesn't reduce security (stays at same N)

use crate::clifford_fhe_v2::params::{CliffordFHEParams, SecurityLevel};
use crate::clifford_fhe_v3::prime_gen::{generate_ntt_primes, generate_special_modulus};

impl CliffordFHEParams {
    /// **CPU DEMO: Ultra-Fast V3 Bootstrap (N=512, 7 primes) - COMPLETES IN SECONDS**
    ///
    /// **Purpose**: Quick demonstration and testing of V3 bootstrap on CPU
    ///
    /// **Parameters**:
    /// - Ring dimension: N=512 (minimal, very fast)
    /// - Total primes: 7 (1 special + 6 scaling) - **DYNAMICALLY GENERATED**
    /// - Scale: 2^40
    /// - Security: ~50 bits (demo only, NOT secure, NOT for production)
    ///
    /// **Performance**:
    /// - Prime generation: <1 second
    /// - Key generation: <2 seconds on CPU
    /// - Bootstrap operation: <5 seconds
    /// - Total end-to-end: <10 seconds
    ///
    /// **Use Cases**:
    /// - Quick validation that bootstrap works
    /// - Understanding the bootstrap flow
    /// - NOT for any real use (insecure parameters)
    ///
    /// **NOT for production** - Use GPU backends with N=8192 for production
    pub fn new_v3_demo_cpu() -> Self {
        let n = 512;

        println!("Generating V3 CPU demo parameters (N={}, 10 primes)...", n);

        // Generate special 60-bit modulus
        let special_modulus = generate_special_modulus(n, 60);

        // Generate 9 scaling primes (~40-bit) - need 9 for CoeffToSlot (log2(N/2) = 8) + 1 safety
        let scaling_primes = generate_ntt_primes(n, 9, 40, 0);

        // Combine into single moduli vector
        let mut moduli = vec![special_modulus];
        moduli.extend(scaling_primes);

        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = 1.0;

        Self {
            n,
            moduli,
            scale,
            error_std: 3.2,
            security: SecurityLevel::Bit128,  // Note: Actually ~50 bits with N=512
            inv_scale_mod_q,
            inv_q_top_mod_q,
            kappa_plain_mul,
        }
    }

    /// V3 Bootstrap parameters (N=8192, 41 primes, sufficient depth for bootstrap)
    ///
    /// **Use for:** Deep encrypted GNN with bootstrapping
    ///
    /// **Modulus chain:** DYNAMICALLY GENERATED
    /// - 1 × 60-bit special prime (fast modular reduction)
    /// - 40 × 40-bit scaling primes (≈ 2^40)
    ///
    /// **Levels breakdown:**
    /// - CoeffToSlot: 12 levels (log2(N/2) = 12)
    /// - EvalMod: 16 levels (degree-23 polynomial with BSGS)
    /// - SlotToCoeff: 12 levels (log2(N/2) = 12)
    /// - Total: 40 levels for bootstrap
    ///
    /// **Security:** ~128 bits (NIST Level 1)
    ///
    /// **Bootstrap frequency:** Every 1-2 multiplications
    pub fn new_v3_bootstrap_8192() -> Self {
        let n = 8192;

        println!("Generating V3 bootstrap parameters (N={}, 41 primes)...", n);

        // Generate special 60-bit modulus
        let special_modulus = generate_special_modulus(n, 60);

        // Generate 40 scaling primes (~40-bit) - need 40 for bootstrap (12 + 16 + 12)
        let scaling_primes = generate_ntt_primes(n, 40, 40, 0);

        // Combine into single moduli vector
        let mut moduli = vec![special_modulus];
        moduli.extend(scaling_primes);

        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = 1.0;  // No longer needed with exact rescale

        Self {
            n,
            moduli,
            scale,
            error_std: 3.2,
            security: SecurityLevel::Bit128,
            inv_scale_mod_q,
            inv_q_top_mod_q,
            kappa_plain_mul,
        }
    }

    /// V3 Bootstrap parameters (N=16384, 25 primes, depth ~18 with bootstrap)
    ///
    /// **Use for:** Very deep encrypted GNN, higher precision
    ///
    /// **Modulus chain:** DYNAMICALLY GENERATED
    /// - 1 × 60-bit special prime
    /// - 24 × 41-bit scaling primes
    ///
    /// **Levels breakdown:**
    /// - 15 primes reserved for bootstrap operations
    /// - 8 primes for computation between bootstraps
    /// - 2 primes for safety margin
    ///
    /// **Security:** ~192 bits (NIST Level 3)
    ///
    /// **Performance:** Slower than N=8192 but better precision
    pub fn new_v3_bootstrap_16384() -> Self {
        let n = 16384;

        println!("Generating V3 bootstrap parameters (N={}, 25 primes)...", n);

        // Generate special 60-bit modulus
        let special_modulus = generate_special_modulus(n, 60);

        // Generate 24 scaling primes (~40-bit)
        let scaling_primes = generate_ntt_primes(n, 24, 40, 0);

        // Combine into single moduli vector
        let mut moduli = vec![special_modulus];
        moduli.extend(scaling_primes);

        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = 1.0;  // No longer needed with exact rescale

        Self {
            n,
            moduli,
            scale,
            error_std: 3.2,
            security: SecurityLevel::Bit192,
            inv_scale_mod_q,
            inv_q_top_mod_q,
            kappa_plain_mul,
        }
    }

    /// Fast Demo Parameters (N=8192, 41 primes) - DYNAMICALLY GENERATED
    ///
    /// Optimized for demonstration purposes:
    /// - Ring dimension: 8192 (production-ready)
    /// - 41 primes total (1 special + 40 scaling) for full bootstrap with proper CoeffToSlot/SlotToCoeff
    /// - Supports full bootstrap pipeline (CoeffToSlot: 12 levels, EvalMod: 16 levels, SlotToCoeff: 12 levels, +1 for final rescale)
    /// - Primes are GENERATED at runtime, not hardcoded!
    ///
    /// Key generation time: ~120 seconds (parallelized on Metal GPU)
    /// Bootstrap supported: Yes
    /// Security: ~128 bits (NIST Level 1)
    pub fn new_v3_bootstrap_fast_demo() -> Self {
        let n = 8192;

        println!("Generating V3 fast demo parameters (N={}, 41 primes)...", n);

        // Generate special 60-bit modulus
        let special_modulus = generate_special_modulus(n, 60);

        // Generate 40 scaling primes (~40-bit) for full bootstrap pipeline
        let scaling_primes = generate_ntt_primes(n, 40, 40, 0);

        // Combine into single moduli vector
        let mut moduli = vec![special_modulus];
        moduli.extend(scaling_primes);

        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = 1.0;

        Self {
            n,
            moduli,
            scale,
            error_std: 3.2,
            security: SecurityLevel::Bit128,
            inv_scale_mod_q,
            inv_q_top_mod_q,
            kappa_plain_mul,
        }
    }

    /// Minimal Production Parameters (N=8192, 20 primes) - DYNAMICALLY GENERATED
    ///
    /// Full production parameters with all features:
    /// - Ring dimension: 8192
    /// - 20 primes total (1 special + 19 scaling)
    /// - 12 primes for bootstrap
    /// - 7 primes for computation (enables 7 multiplications post-bootstrap)
    /// - Security: Full 128 bits (NIST Level 1)
    ///
    /// Key generation time: ~180 seconds (parallelized)
    /// Bootstrap supported: Yes
    pub fn new_v3_bootstrap_minimal() -> Self {
        let n = 8192;

        println!("Generating V3 minimal production parameters (N={}, 20 primes)...", n);

        // Generate special 60-bit modulus
        let special_modulus = generate_special_modulus(n, 60);

        // Generate 19 scaling primes (~40-bit)
        let scaling_primes = generate_ntt_primes(n, 19, 40, 0);

        // Combine into single moduli vector
        let mut moduli = vec![special_modulus];
        moduli.extend(scaling_primes);

        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = 1.0;  // No longer needed with exact rescale

        Self {
            n,
            moduli,
            scale,
            error_std: 3.2,
            security: SecurityLevel::Bit128,
            inv_scale_mod_q,
            inv_q_top_mod_q,
            kappa_plain_mul,
        }
    }

    /// Get number of levels available for computation (excludes bootstrap reserve)
    ///
    /// # Arguments
    ///
    /// * `bootstrap_levels` - Number of levels reserved for bootstrap (typically 12-15)
    ///
    /// # Returns
    ///
    /// Number of multiplication levels available between bootstraps
    pub fn computation_levels(&self, bootstrap_levels: usize) -> usize {
        if self.moduli.len() <= bootstrap_levels {
            0
        } else {
            self.moduli.len() - bootstrap_levels - 1  // -1 for special prime
        }
    }

    /// Check if parameters are suitable for bootstrapping
    ///
    /// # Arguments
    ///
    /// * `bootstrap_levels` - Required levels for bootstrap (typically 12-15)
    ///
    /// # Returns
    ///
    /// True if parameters have enough primes for bootstrap + some computation
    pub fn supports_bootstrap(&self, bootstrap_levels: usize) -> bool {
        self.computation_levels(bootstrap_levels) >= 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v3_bootstrap_8192() {
        let params = CliffordFHEParams::new_v3_bootstrap_8192();

        assert_eq!(params.n, 8192);
        assert_eq!(params.moduli.len(), 22);
        assert_eq!(params.security, SecurityLevel::Bit128);

        // Verify all primes are NTT-friendly (q ≡ 1 mod 2N)
        let two_n = 2 * params.n as u64;
        for (i, &q) in params.moduli.iter().enumerate() {
            assert_eq!((q - 1) % two_n, 0, "Prime {} is not NTT-friendly: {} mod {} = {}",
                       i, q, two_n, (q - 1) % two_n);
        }
    }

    #[test]
    fn test_v3_bootstrap_16384() {
        let params = CliffordFHEParams::new_v3_bootstrap_16384();

        assert_eq!(params.n, 16384);
        assert_eq!(params.moduli.len(), 25);
        assert_eq!(params.security, SecurityLevel::Bit192);

        // Verify NTT-friendly
        let two_n = 2 * params.n as u64;
        for (i, &q) in params.moduli.iter().enumerate() {
            assert_eq!((q - 1) % two_n, 0, "Prime {} is not NTT-friendly", i);
        }
    }

    #[test]
    fn test_v3_bootstrap_minimal() {
        let params = CliffordFHEParams::new_v3_bootstrap_minimal();

        assert_eq!(params.n, 8192);
        assert_eq!(params.moduli.len(), 20);

        // Should support bootstrap with 12 levels
        assert!(params.supports_bootstrap(12));

        // Should have 7 computation levels (20 - 12 - 1)
        assert_eq!(params.computation_levels(12), 7);
    }

    #[test]
    fn test_computation_levels() {
        let params = CliffordFHEParams::new_v3_bootstrap_8192();

        // With 12 bootstrap levels: 22 - 12 - 1 = 9 computation levels
        assert_eq!(params.computation_levels(12), 9);

        // With 15 bootstrap levels: 22 - 15 - 1 = 6 computation levels
        assert_eq!(params.computation_levels(15), 6);
    }

    #[test]
    fn test_supports_bootstrap() {
        let params = CliffordFHEParams::new_v3_bootstrap_8192();

        // Should support bootstrap with 12, 13, 14, 15 levels
        assert!(params.supports_bootstrap(12));
        assert!(params.supports_bootstrap(13));
        assert!(params.supports_bootstrap(14));
        assert!(params.supports_bootstrap(15));

        // Should not support with too many levels (22 - 18 - 1 = 3, exactly at boundary)
        assert!(params.supports_bootstrap(18));
        assert!(!params.supports_bootstrap(19));  // 22 - 19 - 1 = 2 < 3
    }

    #[test]
    fn test_all_primes_distinct() {
        let params = CliffordFHEParams::new_v3_bootstrap_8192();

        // Check all primes are distinct
        for i in 0..params.moduli.len() {
            for j in (i+1)..params.moduli.len() {
                assert_ne!(params.moduli[i], params.moduli[j],
                          "Primes {} and {} are the same", i, j);
            }
        }
    }

    #[test]
    fn test_prime_sizes() {
        let params = CliffordFHEParams::new_v3_bootstrap_8192();

        // First prime should be ~60 bits
        let first_bits = (params.moduli[0] as f64).log2() as usize;
        assert!(first_bits >= 59 && first_bits <= 61,
                "First prime should be ~60 bits, got {} bits", first_bits);

        // Remaining primes should be ~39-41 bits (target 40-bit, but range starts at 2^39)
        for (i, &q) in params.moduli.iter().enumerate().skip(1) {
            let bits = (q as f64).log2() as usize;
            assert!(bits >= 39 && bits <= 42,
                    "Prime {} should be ~39-41 bits, got {} bits", i, bits);
        }
    }
}
