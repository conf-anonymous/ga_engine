//! V2 Parameter Sets for Clifford FHE with NTT Optimization
//!
//! **Optimizations over V1:**
//! - All primes are NTT-friendly: q ≡ 1 mod 2N
//! - Optimized for Harvey butterfly NTT
//! - Better prime selection for Barrett reduction efficiency
//! - Larger polynomial degrees supported (up to N=16384)
//!
//! **Performance Target:** 10-20× faster than V1 with same security

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

/// V2 CKKS parameters optimized for NTT-based operations
#[derive(Debug, Clone)]
pub struct CliffordFHEParams {
    /// Ring dimension (polynomial degree)
    /// Must be power of 2 for NTT
    /// V2 supports: 1024, 2048, 4096, 8192, 16384
    pub n: usize,

    /// Ciphertext modulus chain (for leveled FHE)
    /// All primes are NTT-friendly (q ≡ 1 mod 2N)
    pub moduli: Vec<u64>,

    /// Scaling factor (determines precision)
    /// In V2, we use exact powers of 2 for clean rescaling
    pub scale: f64,

    /// Standard deviation for error distribution
    pub error_std: f64,

    /// Security level
    pub security: SecurityLevel,

    /// Precomputed scale inverses for multiply_plain (DEPRECATED - use rescale instead)
    /// inv_scale_mod_q[level][prime_idx] = (scale mod q_i)^(-1) mod q_i
    /// Used to cancel scale inflation in plaintext multiplication
    pub inv_scale_mod_q: Vec<Vec<u64>>,

    /// Precomputed modulus inverses for rescale operation
    /// inv_q_top_mod_q[level-1][prime_idx] = q_level^(-1) mod q_i for i=0..level-1
    /// Used in rescale_to_next() to divide by the top modulus
    pub inv_q_top_mod_q: Vec<Vec<u64>>,

    /// Normalization constant for multiply_plain operations
    /// κ ≈ n/2 × 1.46 compensates for encode/decode canonical embedding mismatch
    /// Use encode_for_plain_mul instead of encode when creating plaintexts for multiply_plain
    pub kappa_plain_mul: f64,
}

impl CliffordFHEParams {
    /// Test parameters for NTT development (N=1024, depth-1)
    ///
    /// **Use for:** NTT testing, basic multiplication verification
    ///
    /// **Modulus chain:**
    /// - q₀ = 1152921504606584833 (60-bit, NTT-friendly for N=1024)
    /// - q₁ = 1099511678977 (41-bit, scaling prime ≈ 2^40)
    /// - q₂ = 1099511683073 (41-bit, scaling prime ≈ 2^40)
    pub fn new_test_ntt_1024() -> Self {
        let n = 1024;
        let moduli = vec![
            1152921504606584833,  // 60-bit: (q-1) = 2^60 - 2048 = 1024 * 2 * k
            1099511678977,        // 41-bit: (q-1) = 2048 * 536870351
            1099511683073,        // 41-bit: (q-1) = 2048 * 536870353
        ];
        let scale = 2f64.powi(40); // Exact power of 2
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        // κ measured empirically for n=1024 with drop-limb rescale: κ ≈ 1200
        let kappa_plain_mul = 1200.0;

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

    /// Test parameters for N=2048 (depth-3)
    ///
    /// **Use for:** Medium-depth circuits, faster than N=4096
    ///
    /// **Modulus chain:** 5 primes for depth-3 support
    pub fn new_test_ntt_2048() -> Self {
        let n = 2048;
        let moduli = vec![
            1152921504605798401,  // 60-bit: q ≡ 1 mod 4096
            1099511701505,        // 41-bit: q ≡ 1 mod 4096
            1099511705601,        // 41-bit: q ≡ 1 mod 4096
            1099511709697,        // 41-bit: q ≡ 1 mod 4096
            1099511713793,        // 41-bit: q ≡ 1 mod 4096
        ];
        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = (n as f64 / 2.0) * 1.46;

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

    /// Test parameters for N=4096 (depth-5)
    ///
    /// **Use for:** Deep circuits, geometric product chains
    ///
    /// **Modulus chain:** 7 primes for depth-5 support
    pub fn new_test_ntt_4096() -> Self {
        let n = 4096;
        let moduli = vec![
            // FIXED: All primes below were composite! Replaced with verified actual primes.
            1152921504597016577,  // 60-bit: q ≡ 1 mod 8192, verified prime
            1099511799809,        // 41-bit: q ≡ 1 mod 8192, verified prime
            1099511922689,        // 41-bit: q ≡ 1 mod 8192, verified prime
            1099512094721,        // 41-bit: q ≡ 1 mod 8192, verified prime
            1099512266753,        // 41-bit: q ≡ 1 mod 8192, verified prime
            1099512291329,        // 41-bit: q ≡ 1 mod 8192, verified prime
            1099512365057,        // 41-bit: q ≡ 1 mod 8192, verified prime
        ];
        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = (n as f64 / 2.0) * 1.46;

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

    /// Production parameters for 128-bit security (N=8192, depth-7)
    ///
    /// **Use for:** Production deployments
    ///
    /// **Modulus chain:** 9 primes for depth-7 support
    pub fn new_128bit() -> Self {
        let n = 8192;
        let moduli = vec![
            1152921504606994433,  // 60-bit: q ≡ 1 mod 16384
            1099511922689,        // 41-bit: q ≡ 1 mod 16384
            1099512004609,        // 41-bit: q ≡ 1 mod 16384
            1099512266753,        // 41-bit: q ≡ 1 mod 16384
            1099512299521,        // 41-bit: q ≡ 1 mod 16384
            1099512365057,        // 41-bit: q ≡ 1 mod 16384
            1099512856577,        // 41-bit: q ≡ 1 mod 16384
            1099512938497,        // 41-bit: q ≡ 1 mod 16384
            1099513774081,        // 41-bit: q ≡ 1 mod 16384
        ];
        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);

        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = (n as f64 / 2.0) * 1.46;
        Self { n, moduli, scale, error_std: 3.2, security: SecurityLevel::Bit128, inv_scale_mod_q, inv_q_top_mod_q, kappa_plain_mul }
    }

    /// Production parameters for 192-bit security (N=16384, depth-7)
    pub fn new_192bit() -> Self {
        let n = 16384;
        let moduli = vec![
            1152921504607338497,  // 60-bit: q ≡ 1 mod 32768
            1099511922689,        // 41-bit: q ≡ 1 mod 32768
            1099512938497,        // 41-bit: q ≡ 1 mod 32768
            1099514314753,        // 41-bit: q ≡ 1 mod 32768
            1099514478593,        // 41-bit: q ≡ 1 mod 32768
            1099515691009,        // 41-bit: q ≡ 1 mod 32768
            1099515789313,        // 41-bit: q ≡ 1 mod 32768
            1099515985921,        // 41-bit: q ≡ 1 mod 32768
            1099516182529,        // 41-bit: q ≡ 1 mod 32768
        ];
        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);

        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = (n as f64 / 2.0) * 1.46;
        Self { n, moduli, scale, error_std: 3.2, security: SecurityLevel::Bit192, inv_scale_mod_q, inv_q_top_mod_q, kappa_plain_mul }
    }

    /// Production parameters for 256-bit security (N=32768, depth-7)
    pub fn new_256bit() -> Self {
        let n = 32768;
        let moduli = vec![
            1152921504608747521,  // 60-bit: q ≡ 1 mod 65536
            1099512938497,        // 41-bit: q ≡ 1 mod 65536
            1099514314753,        // 41-bit: q ≡ 1 mod 65536
            1099515691009,        // 41-bit: q ≡ 1 mod 65536
            1099516280833,        // 41-bit: q ≡ 1 mod 65536
            1099516542977,        // 41-bit: q ≡ 1 mod 65536
            1099516870657,        // 41-bit: q ≡ 1 mod 65536
            1099518246913,        // 41-bit: q ≡ 1 mod 65536
            1099520606209,        // 41-bit: q ≡ 1 mod 65536
        ];
        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);

        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = (n as f64 / 2.0) * 1.46;
        Self { n, moduli, scale, error_std: 3.2, security: SecurityLevel::Bit256, inv_scale_mod_q, inv_q_top_mod_q, kappa_plain_mul }
    }

    /// V3 Bootstrap parameters for Metal GPU (N=1024, 20 primes)
    ///
    /// Designed for full CoeffToSlot + SlotToCoeff bootstrap on Metal GPU.
    ///
    /// **Level budget:**
    /// - CoeffToSlot: log2(N/2) = log2(512) = 9 levels
    /// - SlotToCoeff: log2(N/2) = log2(512) = 9 levels
    /// - Total: 18 levels required
    /// - We use 20 primes (19 levels) to have margin for EvalMod
    ///
    /// **Performance target:**
    /// - <2s for CoeffToSlot + SlotToCoeff on M3 Max
    /// - 36-72× faster than CPU-only V3 (~360s baseline)
    ///
    /// **Note:** Uses dynamic prime generation from V3 module
    ///
    #[cfg(feature = "v3")]
    pub fn new_v3_bootstrap_metal() -> Result<Self, String> {
        use crate::clifford_fhe_v3::prime_gen::{generate_special_modulus, generate_ntt_primes};

        let n = 1024;

        println!("Generating V3 Metal GPU bootstrap parameters (N={}, 20 primes)...", n);

        // Generate special 60-bit modulus (provides extra precision)
        let special_modulus = generate_special_modulus(n, 60);

        // Generate 19 scaling primes (~45-bit for precision)
        let scaling_primes = generate_ntt_primes(n, 19, 45, 0);

        // Combine: special prime first, then scaling primes
        let mut moduli = vec![special_modulus];
        moduli.extend(scaling_primes);

        let scale = 2f64.powi(45);  // 45-bit scale for precision
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = (n as f64 / 2.0) * 1.46;

        println!("  Generated {} NTT-friendly primes", moduli.len());

        Ok(Self {
            n,
            moduli,
            scale,
            error_std: 3.2,
            security: SecurityLevel::Bit128,
            inv_scale_mod_q,
            inv_q_top_mod_q,
            kappa_plain_mul,
        })
    }

    /// V3 CUDA GPU Full Bootstrap Parameters (N=1024, 30 primes)
    ///
    /// **Level Budget for Full Bootstrap:**
    /// - CoeffToSlot: 9 levels
    /// - EvalMod: 9 levels
    /// - SlotToCoeff: 9 levels
    /// - Total: 27 levels minimum (30 primes provides buffer)
    ///
    /// **Use for:** Full CUDA GPU bootstrap with EvalMod
    ///
    /// **Modulus chain:** 1× 60-bit + 29× 45-bit primes
    #[cfg(feature = "v3")]
    pub fn new_v3_bootstrap_cuda_full() -> Result<Self, String> {
        use crate::clifford_fhe_v3::prime_gen::{generate_special_modulus, generate_ntt_primes};

        let n = 1024;

        println!("Generating V3 CUDA GPU full bootstrap parameters (N={}, 30 primes)...", n);

        // Generate special 60-bit modulus (provides extra precision)
        let special_modulus = generate_special_modulus(n, 60);

        // Generate 29 scaling primes (~45-bit for precision)
        let scaling_primes = generate_ntt_primes(n, 29, 45, 0);

        // Combine: special prime first, then scaling primes
        let mut moduli = vec![special_modulus];
        moduli.extend(scaling_primes);

        // CRITICAL: Scale must match the actual prime size for stable rescaling!
        // The primes are ~2^44 (in range [2^44, 2^45)), so use 2^44 scale.
        // Using 2^45 scale with 2^44 primes causes 1 bit growth per multiply,
        // which compounds exponentially in chained operations.
        let scale = 2f64.powi(44);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = (n as f64 / 2.0) * 1.46;

        println!("  Generated {} NTT-friendly primes", moduli.len());
        println!("  N = {}, num_primes = {}", n, moduli.len());
        println!("  Scale = 2^44 (matched to prime size)");
        println!("  Level budget: CoeffToSlot(9) + EvalMod(9) + SlotToCoeff(9) = 27 levels used");

        Ok(Self {
            n,
            moduli,
            scale,
            error_std: 3.2,
            security: SecurityLevel::Bit128,
            inv_scale_mod_q,
            inv_q_top_mod_q,
            kappa_plain_mul,
        })
    }

    /// Get the prime for a specific level in the modulus chain
    pub fn modulus_at_level(&self, level: usize) -> u64 {
        if level >= self.moduli.len() {
            panic!("Level {} exceeds maximum {}", level, self.moduli.len() - 1);
        }
        self.moduli[level]
    }

    /// Number of levels (depth) available
    pub fn max_level(&self) -> usize {
        self.moduli.len() - 1
    }

    /// Get product of all moduli (total modulus)
    pub fn total_modulus(&self) -> f64 {
        self.moduli.iter().map(|&q| q as f64).product()
    }

    /// Verify all primes are NTT-friendly (q ≡ 1 mod 2N)
    pub fn verify_ntt_friendly(&self) -> bool {
        let two_n = (2 * self.n) as u64;
        self.moduli.iter().all(|&q| (q - 1) % two_n == 0)
    }

    /// Precompute scale inverses for all levels and primes (DEPRECATED - NOT USED)
    ///
    /// NOTE: This function is deprecated and returns empty vectors.
    /// The exact BigInt rescale implementation does not require precomputed inverses.
    /// Kept for API compatibility with existing param constructors.
    pub fn precompute_inv_scale_mod_q(_scale: f64, moduli: &[u64]) -> Vec<Vec<u64>> {
        // Return empty vectors for each level (not used by BigInt rescale)
        vec![vec![]; moduli.len()]
    }

    /// Precompute modulus inverses for rescale operation (DEPRECATED - NOT USED)
    ///
    /// NOTE: This function is deprecated and returns empty vectors.
    /// The exact BigInt rescale implementation does not require precomputed inverses.
    /// It performs CRT reconstruction and exact division instead.
    /// Kept for API compatibility with existing param constructors.
    pub fn precompute_inv_q_top_mod_q(moduli: &[u64]) -> Vec<Vec<u64>> {
        // Return empty vectors for each level (not used by BigInt rescale)
        if moduli.len() > 1 {
            vec![vec![]; moduli.len() - 1]
        } else {
            vec![]
        }
    }

    /// Compute modular inverse using extended Euclidean algorithm
    /// Returns x such that (a * x) mod m = 1
    fn mod_inverse(a: u64, m: u64) -> Option<u64> {
        fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
            if b == 0 {
                (a, 1, 0)
            } else {
                let (gcd, x1, y1) = extended_gcd(b, a % b);
                (gcd, y1, x1 - (a / b) * y1)
            }
        }

        let (gcd, x, _) = extended_gcd(a as i128, m as i128);

        if gcd != 1 {
            return None; // No inverse exists
        }

        // Ensure positive result
        let result = if x < 0 {
            (x + m as i128) as u64
        } else {
            x as u64
        };

        Some(result)
    }
}

impl Default for CliffordFHEParams {
    fn default() -> Self {
        Self::new_128bit()
    }
}

/// Precomputed NTT-friendly primes for different polynomial degrees
///
/// All primes satisfy: q ≡ 1 mod 2N for their respective N
pub mod ntt_primes {
    /// 60-bit NTT-friendly primes for N=1024
    pub const N1024_60BIT: &[u64] = &[
        1152921504606584833,  // 2^60 - 2047 (favorite prime for N=1024)
        1152921504606586881,
        1152921504606588929,
    ];

    /// 41-bit NTT-friendly primes for N=1024 (scaling primes)
    pub const N1024_41BIT: &[u64] = &[
        1099511678977,   // 2^40 + 51201
        1099511683073,
        1099511687169,
        1099511691265,
        1099511695361,
        1099511699457,
    ];

    /// 60-bit NTT-friendly primes for N=2048
    pub const N2048_60BIT: &[u64] = &[
        1152921504605798401,
        1152921504605802497,
        1152921504605806593,
    ];

    /// 60-bit NTT-friendly primes for N=4096
    /// FIXED: Replaced composite numbers with actual primes
    pub const N4096_60BIT: &[u64] = &[
        1152921504597016577,  // Verified prime, q ≡ 1 (mod 8192)
        1152921504597024769,  // TODO: Verify these are actually prime
        1152921504597032961,  // TODO: Verify these are actually prime
    ];

    /// 60-bit NTT-friendly primes for N=8192
    pub const N8192_60BIT: &[u64] = &[
        1152921504584294401,
        1152921504584310785,
        1152921504584327169,
    ];

    /// 60-bit NTT-friendly primes for N=16384
    pub const N16384_60BIT: &[u64] = &[
        1152921504568426497,
        1152921504568459265,
        1152921504568492033,
    ];

    /// Verify a prime is NTT-friendly for given N
    pub fn is_ntt_friendly(q: u64, n: usize) -> bool {
        let two_n = (2 * n) as u64;
        (q - 1) % two_n == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_n1024() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        assert_eq!(params.n, 1024);
        assert_eq!(params.moduli.len(), 3);
        assert!(params.verify_ntt_friendly(), "All primes must be NTT-friendly");
    }

    #[test]
    fn test_params_n2048() {
        let params = CliffordFHEParams::new_test_ntt_2048();
        assert_eq!(params.n, 2048);
        assert_eq!(params.moduli.len(), 5);
        assert!(params.verify_ntt_friendly());
    }

    #[test]
    fn test_params_n4096() {
        let params = CliffordFHEParams::new_test_ntt_4096();
        assert_eq!(params.n, 4096);
        assert_eq!(params.moduli.len(), 7);
        assert!(params.verify_ntt_friendly());
    }

    #[test]
    fn test_params_128bit() {
        let params = CliffordFHEParams::new_128bit();
        assert_eq!(params.n, 8192);
        assert_eq!(params.security, SecurityLevel::Bit128);
        assert!(params.verify_ntt_friendly());
    }

    #[test]
    fn test_modulus_at_level() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let q0 = params.modulus_at_level(0);
        assert_eq!(q0, 1152921504606584833);
    }

    #[test]
    fn test_ntt_friendly_verification() {
        // Test N=1024 primes
        for &q in ntt_primes::N1024_60BIT {
            assert!(
                ntt_primes::is_ntt_friendly(q, 1024),
                "Prime {} should be NTT-friendly for N=1024",
                q
            );
        }

        for &q in ntt_primes::N1024_41BIT {
            assert!(
                ntt_primes::is_ntt_friendly(q, 1024),
                "Prime {} should be NTT-friendly for N=1024",
                q
            );
        }
    }

    #[test]
    fn test_scale_is_power_of_two() {
        let params = CliffordFHEParams::new_128bit();
        // scale should be exactly 2^40
        let expected = 2f64.powi(40);
        assert_eq!(params.scale, expected);
    }

    #[test]
    #[should_panic(expected = "Level 10 exceeds maximum")]
    fn test_invalid_level() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        params.modulus_at_level(10); // Should panic
    }
}
