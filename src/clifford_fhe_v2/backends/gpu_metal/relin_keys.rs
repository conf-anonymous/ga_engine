//! Metal GPU Relinearization Keys for CKKS
//!
//! **Purpose:** Enable degree reduction after ciphertext multiplication on Metal GPU.
//!
//! **Background:**
//! After multiplying two degree-1 ciphertexts (c₀, c₁) × (d₀, d₁), we get a degree-2 ciphertext:
//! - ct_mult = (c₀×d₀, c₀×d₁ + c₁×d₀, c₁×d₁)
//! - Decrypts as: c₀×d₀ + (c₀×d₁ + c₁×d₀)·s + c₁×d₁·s²
//!
//! We need to convert this back to degree-1: (c'₀, c'₁) where:
//! - c'₀ + c'₁·s ≈ c₀×d₀ + (c₀×d₁ + c₁×d₀)·s + c₁×d₁·s²
//!
//! **Solution: Relinearization**
//! Relinearization keys encode s² in an encrypted form, allowing us to transform c₂·s² → c'₁·s
//! using key switching with gadget decomposition to minimize noise growth.
//!
//! **Relinearization Key Structure:**
//! We store (rlk0[], rlk1[]) where:
//! - rlk0[i] = -a_i·s + e_i + B^i·s²  (for i = 0..num_digits)
//! - rlk1[i] = a_i
//! - B = 2^base_w (gadget base, typically w=16-20)
//!
//! **Memory Layout:**
//! For N=1024, 41 primes, base_w=16 (→3 digits):
//! - Total size: 2 × num_digits × N × num_primes × 8 bytes ≈ 2 MB
//! - Acceptable on Apple Silicon unified memory
//!
//! **Usage:**
//! ```rust,ignore
//! // Generate relinearization keys
//! let relin_keys = MetalRelinKeys::generate(
//!     device,
//!     &sk,
//!     &params,
//!     &ntt_contexts,
//!     16,  // base_w
//! )?;
//!
//! // Use for ciphertext multiplication
//! let ct_mult = ct1.multiply(&ct2, &relin_keys, &ctx)?;
//! ```
//!
//! **Reference:** Gentry, Halevi, Smart 2012 (BGV relinearization), Cheon et al. 2017 (CKKS adaptation)

use super::device::MetalDevice;
use super::ntt::MetalNttContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use rand::Rng;
use std::sync::Arc;

/// Metal GPU Relinearization Keys (with Gadget Decomposition)
///
/// Stores key-switching keys for degree reduction using gadget decomposition.
/// This ensures minimal noise growth during relinearization (additive instead of multiplicative).
///
/// # Key Structure
///
/// We store (rlk0[], rlk1[]) where:
/// - rlk0[t] = -a_t·s + e_t + B^t·s²  (for t = 0..num_digits)
/// - rlk1[t] = a_t
/// - B = 2^base_w (gadget base)
///
/// # Memory Layout
///
/// Each component is Vec<u64> in flat RNS layout: [coeff0_mod_q0, coeff0_mod_q1, ...]
pub struct MetalRelinKeys {
    /// Metal device (shared)
    device: Arc<MetalDevice>,

    /// Relinearization keys in coefficient domain (per level)
    /// Vec indexed by level, each containing num_digits elements
    /// Each element is Vec<u64> in flat RNS layout [n × num_primes_for_level]
    rlk0_coeff: Vec<Vec<Vec<u64>>>,
    rlk1_coeff: Vec<Vec<Vec<u64>>>,

    /// Pre-computed NTT-transformed relinearization keys (per level)
    /// Vec indexed by level, containing (rlk0_ntt[], rlk1_ntt[])
    /// This eliminates runtime NTT transforms during multiplication
    rlk0_ntt: Vec<Vec<Vec<u64>>>,
    rlk1_ntt: Vec<Vec<Vec<u64>>>,

    /// Gadget base exponent (e.g., 16 → B = 2^16)
    base_w: u32,

    /// Number of decomposition digits
    num_digits: usize,

    /// Ring dimension N
    n: usize,

    /// Number of RNS primes
    num_primes: usize,

    /// Maximum level these keys support
    max_level: usize,
}

impl MetalRelinKeys {
    /// Generate relinearization keys for a secret key
    ///
    /// # Arguments
    /// * `device` - Metal device context
    /// * `sk` - Secret key
    /// * `params` - FHE parameters
    /// * `ntt_contexts` - NTT contexts for each RNS prime
    /// * `base_w` - Gadget base exponent (e.g., 16 for B = 2^16)
    pub fn generate(
        device: Arc<MetalDevice>,
        sk: &SecretKey,
        params: &CliffordFHEParams,
        ntt_contexts: &[MetalNttContext],
        base_w: u32,
    ) -> Result<Self, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║        Generating Metal GPU Relinearization Keys             ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let n = params.n;
        let num_primes = params.moduli.len();
        let max_level = num_primes - 1;
        let moduli = &params.moduli;

        // Calculate number of gadget digits
        // dnum = ceil(log_B(Q)) where Q = product of all primes
        let total_bits: u32 = moduli.iter()
            .map(|&q| 64 - q.leading_zeros())
            .sum();
        let num_digits = ((total_bits + base_w - 1) / base_w) as usize;

        println!("Relinearization key parameters:");
        println!("  Ring dimension (N): {}", n);
        println!("  Number of RNS primes: {}", num_primes);
        println!("  Gadget base: B = 2^{}", base_w);
        println!("  Number of digits: {}", num_digits);
        println!("  Max level: {}", max_level);

        // Compute s² in RNS representation
        let s_squared = Self::multiply_secret_keys(sk, sk, moduli, n)?;

        println!("\nGenerating {} gadget components...", num_digits);

        let mut rlk0_coeff = Vec::with_capacity(num_digits);
        let mut rlk1_coeff = Vec::with_capacity(num_digits);

        // CRITICAL FIX: Precompute B^t mod q for each prime and each digit
        // This avoids u128 overflow when t is large (e.g., B^7 = 2^140 > 2^128)
        // Matches CPU implementation in keys.rs lines 458-468
        let base = 1u64 << base_w;
        let mut bpow_t_mod_q = vec![vec![0u64; num_primes]; num_digits];
        for (j, &q) in moduli.iter().enumerate() {
            let q_u128 = q as u128;
            let mut p = 1u128;  // B^0 = 1
            for t in 0..num_digits {
                bpow_t_mod_q[t][j] = (p % q_u128) as u64;
                p = (p * (base as u128)) % q_u128;  // Incrementally compute B^(t+1) mod q
            }
        }

        for digit_idx in 0..num_digits {
            // Sample random polynomial a_i
            let a_i = Self::sample_uniform_poly(n, moduli);

            // Compute b_i = -a_i·s + e_i - B^digit_idx·s² (matches CPU EVK generation)
            // Use precomputed B^t mod q values to avoid overflow
            let b_i = Self::compute_rlk_component_rns(
                &a_i,
                sk,
                &s_squared,
                &bpow_t_mod_q[digit_idx],  // B^t mod q for each prime
                moduli,
                n,
                params.error_std,
            )?;

            // Debug: print first coefficient of first digit
            if std::env::var("EVK_GEN_DEBUG").is_ok() && digit_idx == 0 {
                print!("[EVK_GEN_DEBUG] digit[0] b_i[coeff=0]: ");
                for prime_idx in 0..num_primes {
                    print!("{} ", b_i[0 * num_primes + prime_idx]);
                }
                println!();
                print!("[EVK_GEN_DEBUG] digit[0] a_i[coeff=0]: ");
                for prime_idx in 0..num_primes {
                    print!("{} ", a_i[0 * num_primes + prime_idx]);
                }
                println!();
            }

            rlk0_coeff.push(b_i);
            rlk1_coeff.push(a_i);

            if (digit_idx + 1) % 5 == 0 || digit_idx + 1 == num_digits {
                println!("  Generated {}/{} components", digit_idx + 1, num_digits);
            }
        }

        println!("\nGenerating level-specific keys and NTT transforms...");

        // Generate both coefficient and NTT keys for each level
        let mut rlk0_coeff_levels = Vec::with_capacity(max_level + 1);
        let mut rlk1_coeff_levels = Vec::with_capacity(max_level + 1);
        let mut rlk0_ntt = Vec::with_capacity(max_level + 1);
        let mut rlk1_ntt = Vec::with_capacity(max_level + 1);

        for level in 0..=max_level {
            let num_primes_level = level + 1;
            let mut rlk0_coeff_level = Vec::with_capacity(num_digits);
            let mut rlk1_coeff_level = Vec::with_capacity(num_digits);
            let mut rlk0_ntt_level = Vec::with_capacity(num_digits);
            let mut rlk1_ntt_level = Vec::with_capacity(num_digits);

            for digit_idx in 0..num_digits {
                // Debug: print original values before extraction
                if std::env::var("EVK_EXTRACT_DEBUG").is_ok() && level == 2 && digit_idx == 0 {
                    print!("[EVK_EXTRACT_DEBUG] level={}, digit={}, BEFORE extract rlk0[coeff=0]: ", level, digit_idx);
                    for prime_idx in 0..num_primes.min(3) {
                        print!("{} ", rlk0_coeff[digit_idx][0 * num_primes + prime_idx]);
                    }
                    println!();
                }

                // Extract coefficient keys for this level (first num_primes_level primes)
                let rlk0_coeff_digit = Self::extract_primes_flat(
                    &rlk0_coeff[digit_idx],
                    n,
                    num_primes,
                    num_primes_level,
                );
                let rlk1_coeff_digit = Self::extract_primes_flat(
                    &rlk1_coeff[digit_idx],
                    n,
                    num_primes,
                    num_primes_level,
                );

                // Debug: print extracted values
                if std::env::var("EVK_EXTRACT_DEBUG").is_ok() && level == 2 && digit_idx == 0 {
                    print!("[EVK_EXTRACT_DEBUG] level={}, digit={}, AFTER extract rlk0[coeff=0]: ", level, digit_idx);
                    for prime_idx in 0..num_primes_level.min(3) {
                        print!("{} ", rlk0_coeff_digit[0 * num_primes_level + prime_idx]);
                    }
                    println!();
                }

                // Compute NTT transform for this level (use level-specific coefficients)
                let rlk0_digit_ntt = Self::forward_ntt_flat(
                    &rlk0_coeff_digit,
                    n,
                    num_primes_level,
                    ntt_contexts,
                )?;
                let rlk1_digit_ntt = Self::forward_ntt_flat(
                    &rlk1_coeff_digit,
                    n,
                    num_primes_level,
                    ntt_contexts,
                )?;

                rlk0_coeff_level.push(rlk0_coeff_digit);
                rlk1_coeff_level.push(rlk1_coeff_digit);
                rlk0_ntt_level.push(rlk0_digit_ntt);
                rlk1_ntt_level.push(rlk1_digit_ntt);
            }

            rlk0_coeff_levels.push(rlk0_coeff_level);
            rlk1_coeff_levels.push(rlk1_coeff_level);
            rlk0_ntt.push(rlk0_ntt_level);
            rlk1_ntt.push(rlk1_ntt_level);

            if (level + 1) % 10 == 0 || level == max_level {
                println!("  Processed {}/{} levels", level + 1, max_level + 1);
            }
        }

        let memory_mb = (2 * num_digits * n * num_primes * 8) as f64 / 1_048_576.0;
        println!("\nRelinearization keys generated successfully.");
        println!("   Memory footprint: {:.2} MB", memory_mb);

        Ok(Self {
            device,
            rlk0_coeff: rlk0_coeff_levels,
            rlk1_coeff: rlk1_coeff_levels,
            rlk0_ntt,
            rlk1_ntt,
            base_w,
            num_digits,
            n,
            num_primes,
            max_level,
        })
    }

    /// Get NTT-transformed relinearization keys for a specific level
    pub fn get_ntt_keys(&self, level: usize) -> Result<(&[Vec<u64>], &[Vec<u64>]), String> {
        if level > self.max_level {
            return Err(format!("Level {} exceeds max level {}", level, self.max_level));
        }
        Ok((&self.rlk0_ntt[level], &self.rlk1_ntt[level]))
    }

    /// Get coefficient-form relinearization keys at specified level
    ///
    /// Returns keys with exactly the right number of primes for the level.
    /// Each key is [n × (level+1)] in flat RNS layout.
    pub fn get_coeff_keys(&self, level: usize) -> Result<(&[Vec<u64>], &[Vec<u64>]), String> {
        if level > self.max_level {
            return Err(format!("Level {} exceeds max level {}", level, self.max_level));
        }

        // Debug: print EVK values being returned
        if std::env::var("EVK_GET_DEBUG").is_ok() && level == 2 {
            println!("[EVK_GET_DEBUG] get_coeff_keys(level={})", level);
            println!("[EVK_GET_DEBUG] rlk0_coeff[{}].len() = {}", level, self.rlk0_coeff[level].len());
            if !self.rlk0_coeff[level].is_empty() {
                print!("[EVK_GET_DEBUG] rlk0_coeff[{}][0][coeff=0] primes: ", level);
                let num_primes = self.rlk0_coeff[level][0].len() / self.n;
                for prime_idx in 0..num_primes.min(3) {
                    print!("{} ", self.rlk0_coeff[level][0][0 * num_primes + prime_idx]);
                }
                println!();
            }
        }

        // Coefficient keys are now stored per-level with correct prime count
        Ok((&self.rlk0_coeff[level], &self.rlk1_coeff[level]))
    }

    /// Get gadget decomposition parameters
    pub fn gadget_params(&self) -> (u32, usize) {
        (self.base_w, self.num_digits)
    }

    /// Multiply two secret keys in RNS representation: s1 · s2
    ///
    /// CRITICAL: Uses NTT-based polynomial multiplication, not coefficient-wise!
    /// This matches CPU line 443: s_squared = self.multiply_polynomials(&sk.coeffs, &sk.coeffs, moduli)
    fn multiply_secret_keys(
        sk1: &SecretKey,
        sk2: &SecretKey,
        moduli: &[u64],
        n: usize,
    ) -> Result<Vec<RnsRepresentation>, String> {
        use crate::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;

        let mut result = Vec::with_capacity(n);

        // For each prime, multiply the polynomials using NTT
        let num_primes = moduli.len();
        let mut products_per_prime = vec![vec![0u64; n]; num_primes];

        for (prime_idx, &q) in moduli.iter().enumerate() {
            // Extract polynomials for this prime
            let mut s1_poly = vec![0u64; n];
            let mut s2_poly = vec![0u64; n];
            for coeff_idx in 0..n {
                s1_poly[coeff_idx] = sk1.coeffs[coeff_idx].values[prime_idx];
                s2_poly[coeff_idx] = sk2.coeffs[coeff_idx].values[prime_idx];
            }

            // Multiply using NTT (negacyclic convolution)
            let ntt_ctx = NttContext::new(n, q);
            let product = ntt_ctx.multiply_polynomials(&s1_poly, &s2_poly);
            products_per_prime[prime_idx] = product;
        }

        // Convert back to Vec<RnsRepresentation>
        for coeff_idx in 0..n {
            let mut product_rns = Vec::with_capacity(num_primes);
            for prime_idx in 0..num_primes {
                product_rns.push(products_per_prime[prime_idx][coeff_idx]);
            }
            result.push(RnsRepresentation::new(product_rns, moduli.to_vec()));
        }

        Ok(result)
    }

    /// Sample uniform random polynomial in flat RNS layout
    fn sample_uniform_poly(n: usize, moduli: &[u64]) -> Vec<u64> {
        let num_primes = moduli.len();
        let mut rng = rand::thread_rng();
        let mut poly = vec![0u64; n * num_primes];

        for coeff_idx in 0..n {
            for (prime_idx, &q) in moduli.iter().enumerate() {
                poly[coeff_idx * num_primes + prime_idx] = rng.gen::<u64>() % q;
            }
        }

        poly
    }

    /// Compute rlk component: b = -B^t·s² + a·s + e
    /// Uses per-prime precomputed B^t mod q values to avoid u128 overflow
    fn compute_rlk_component_rns(
        a: &[u64],
        sk: &SecretKey,
        s_squared: &[RnsRepresentation],
        power_mod_q: &[u64],  // B^t mod q for each prime (precomputed)
        moduli: &[u64],
        n: usize,
        error_std: f64,
    ) -> Result<Vec<u64>, String> {
        use rand_distr::{Distribution, Normal};
        use crate::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;

        let num_primes = moduli.len();
        let mut b = vec![0u64; n * num_primes];
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, error_std).unwrap();

        // Compute a·s using NTT-based polynomial multiplication
        // Matches CPU line 498: a_t_times_s = self.multiply_polynomials(&a_t, &sk.coeffs, moduli)
        let mut a_times_s_flat = vec![0u64; n * num_primes];
        for (prime_idx, &q) in moduli.iter().enumerate() {
            // Extract polynomials for this prime
            let mut a_poly = vec![0u64; n];
            let mut s_poly = vec![0u64; n];
            for coeff_idx in 0..n {
                a_poly[coeff_idx] = a[coeff_idx * num_primes + prime_idx];
                s_poly[coeff_idx] = sk.coeffs[coeff_idx].values[prime_idx];
            }

            // Multiply using NTT (negacyclic convolution)
            let ntt_ctx = NttContext::new(n, q);
            let product = ntt_ctx.multiply_polynomials(&a_poly, &s_poly);

            // Store result back in flat layout
            for coeff_idx in 0..n {
                a_times_s_flat[coeff_idx * num_primes + prime_idx] = product[coeff_idx];
            }
        }

        // Now compute b = -B^t·s² + a·s + e per coefficient
        // CRITICAL: Sample ONE error per coefficient, then reduce mod each prime
        // (Matches CPU sample_error which samples one Gaussian value per coefficient)
        for coeff_idx in 0..n {
            // Sample error ONCE per coefficient (same integer for all primes)
            let e_float: f64 = normal.sample(&mut rng);
            let e_int: i64 = e_float.round() as i64;

            for (prime_idx, &q) in moduli.iter().enumerate() {
                let idx = coeff_idx * num_primes + prime_idx;

                // a·s (from polynomial multiplication above)
                let a_times_s = a_times_s_flat[idx];

                // Convert error to mod q (same integer value for all primes)
                let e = if e_int >= 0 {
                    (e_int as u64) % q
                } else {
                    let abs_e = (-e_int) as u64;
                    let remainder = abs_e % q;
                    if remainder == 0 { 0 } else { q - remainder }
                };

                // -B^t·s² mod q (negative, matches CPU line 499: neg_bt_s2)
                let s2_val = s_squared[coeff_idx].values[prime_idx];
                let bt_mod_q = power_mod_q[prime_idx];  // Use precomputed B^t mod q
                let power_s2_pos = ((bt_mod_q as u128 * s2_val as u128) % q as u128) as u64;
                let neg_power_s2 = if power_s2_pos == 0 { 0 } else { q - power_s2_pos };

                // b = -B^t·s² + a·s + e (matches CPU line 496: evk0[t] = -B^t*s^2 + a_t*s + e_t)
                let sum = (neg_power_s2 as u128 + a_times_s as u128 + e as u128) % q as u128;
                b[idx] = sum as u64;
            }
        }

        Ok(b)
    }

    /// Extract first `num_primes_out` primes from flat RNS polynomial
    ///
    /// Input: flat array [n × num_primes_in]
    /// Output: flat array [n × num_primes_out]
    fn extract_primes_flat(
        poly_flat: &[u64],
        n: usize,
        num_primes_in: usize,
        num_primes_out: usize,
    ) -> Vec<u64> {
        let mut result = vec![0u64; n * num_primes_out];

        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes_out {
                let src_idx = coeff_idx * num_primes_in + prime_idx;
                let dst_idx = coeff_idx * num_primes_out + prime_idx;
                result[dst_idx] = poly_flat[src_idx];
            }
        }

        result
    }

    /// Forward NTT transform on flat RNS polynomial (CPU-side, for precomputation)
    ///
    /// **State-of-the-art optimization**: Pre-applies twist for negacyclic convolution, storing
    /// EVK in "twisted-NTT" domain to eliminate runtime twist operations during relinearization.
    fn forward_ntt_flat(
        poly_flat: &[u64],
        n: usize,
        num_primes_level: usize,
        ntt_contexts: &[MetalNttContext],
    ) -> Result<Vec<u64>, String> {
        let mut result = vec![0u64; n * num_primes_level];

        for prime_idx in 0..num_primes_level {
            let ntt_ctx = &ntt_contexts[prime_idx];
            let q = ntt_ctx.q();

            // Extract coefficients for this prime
            // Note: poly_flat has stride = num_primes_level (it was already extracted)
            let mut poly_prime = vec![0u64; n];
            for coeff_idx in 0..n {
                poly_prime[coeff_idx] = poly_flat[coeff_idx * num_primes_level + prime_idx];
            }

            // Forward NTT (for later asymmetric multiplication optimization)
            ntt_ctx.forward(&mut poly_prime)?;

            // Store in flat layout (now in twisted-NTT domain)
            for coeff_idx in 0..n {
                result[coeff_idx * num_primes_level + prime_idx] = poly_prime[coeff_idx];
            }
        }

        Ok(result)
    }

    /// Get device reference
    pub fn device(&self) -> &Arc<MetalDevice> {
        &self.device
    }

    /// Get number of digits
    pub fn num_digits(&self) -> usize {
        self.num_digits
    }

    /// Get gadget base
    pub fn base_w(&self) -> u32 {
        self.base_w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_relin_keys_generation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let device = Arc::new(MetalDevice::new().expect("Failed to create Metal device"));

        // Generate secret key
        let key_ctx = KeyContext::new(params.clone());
        let (_, sk, _) = key_ctx.keygen();

        // Create NTT contexts
        let ntt_contexts: Vec<MetalNttContext> = params.moduli.iter()
            .map(|&q| MetalNttContext::new(params.n, q, device.clone()).unwrap())
            .collect();

        // Generate relinearization keys
        let result = MetalRelinKeys::generate(
            device.clone(),
            &sk,
            &params,
            &ntt_contexts,
            16,
        );

        assert!(result.is_ok(), "Failed to generate relinearization keys: {:?}", result.err());

        let relin_keys = result.unwrap();
        assert_eq!(relin_keys.n, params.n);
        assert_eq!(relin_keys.num_primes, params.moduli.len());
        assert!(relin_keys.num_digits > 0);
    }
}
