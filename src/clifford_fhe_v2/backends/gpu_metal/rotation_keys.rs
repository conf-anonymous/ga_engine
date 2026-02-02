//! Metal GPU Rotation Keys for Homomorphic Slot Rotations
//!
//! **Purpose:** Enable key switching after Galois automorphisms on Metal GPU ciphertexts.
//!
//! **Background:**
//! When we apply a Galois automorphism σ_k to a ciphertext (c₀, c₁), we get:
//! - σ_k(c₀ + c₁·s) = σ_k(c₀) + σ_k(c₁)·σ_k(s)
//!
//! But we need the result in the form (c'₀, c'₁) where decryption uses the ORIGINAL secret key s:
//! - c'₀ + c'₁·s ≈ σ_k(m)
//!
//! **Solution: Key Switching**
//! Rotation keys allow us to convert σ_k(c₁)·σ_k(s) back to c'₁·s using a key-switching key
//! that encodes the relationship between σ_k(s) and s.
//!
//! **Key Switching Key Structure:**
//! For each Galois element k, we store (a_k, b_k) where:
//! - b_k ≈ -a_k·s + e + σ_k(s)
//!
//! This allows us to transform σ_k(c₁)·σ_k(s) → c'₁·s using polynomial multiplication.
//!
//! **Reference:** Halevi & Shoup 2014, Chen et al. 2018 (CKKS rotation keys)

use super::device::MetalDevice;
use super::ntt::MetalNttContext;
use super::rotation::{compute_galois_map, rotation_step_to_galois_element};
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use std::collections::HashMap;
use std::sync::Arc;

/// Metal GPU Rotation Keys (with Gadget Decomposition)
///
/// Stores key-switching keys for each rotation step using gadget decomposition.
/// This ensures minimal noise growth during rotation (additive instead of multiplicative).
///
/// # Key Structure
///
/// For each Galois element k, we store (rlk0[], rlk1[]) where:
/// - rlk0[t] = -a_t·s + e_t + B^t·σ_k(s)  (for t = 0..num_digits)
/// - rlk1[t] = a_t
/// - B = 2^base_w (gadget base, typically w=20-30)
///
/// # Memory Layout
///
/// For N=1024, 41 primes, base_w=20 (→3 digits), 24 rotation keys:
/// - Each key: 2 × num_digits × N × num_primes × 8 bytes = ~3.9 MB
/// - Total: 24 × 3.9 MB = ~94 MB (acceptable on Apple Silicon unified memory)
///
/// # Usage
///
/// ```rust,ignore
/// // Generate rotation keys for bootstrap
/// let rot_steps = compute_bootstrap_rotation_steps(n);
/// let rot_keys = MetalRotationKeys::generate(
///     device,
///     &sk,
///     &rot_steps,
///     &params,
///     &ntt_contexts,
///     20,  // base_w
/// )?;
///
/// // Use for rotation (handles gadget decomposition internally)
/// let rotated_ct = ct.rotate_by_steps(1, &rot_keys, &ctx)?;
/// ```
pub struct MetalRotationKeys {
    /// Metal device (shared)
    device: Arc<MetalDevice>,

    /// Rotation keys for each Galois element k (coefficient domain)
    /// Maps k → (rlk0[], rlk1[]) where each array has num_digits elements
    /// Each element is Vec<u64> in flat RNS layout: [coeff0_mod_q0, coeff0_mod_q1, ...]
    keys: HashMap<usize, (Vec<Vec<u64>>, Vec<Vec<u64>>)>,

    /// Pre-computed NTT-transformed rotation keys (NTT domain, Montgomery)
    /// Maps k → level → (rlk0_ntt[], rlk1_ntt[]) for each level
    /// This eliminates the need to transform keys at runtime during rotations
    /// Memory cost: 2× key size per level, but saves 15-20% of rotation time
    keys_ntt: HashMap<usize, Vec<(Vec<Vec<u64>>, Vec<Vec<u64>>)>>,

    /// Gadget base exponent (e.g., 20 → B = 2^20)
    base_w: u32,

    /// Number of decomposition digits
    num_digits: usize,

    /// Ring dimension N
    n: usize,

    /// Number of RNS primes
    num_primes: usize,

    /// Level these keys were generated for
    level: usize,
}

impl MetalRotationKeys {
    /// Generate rotation keys for a set of rotation steps
    ///
    /// For each rotation step r, computes the Galois element k = 5^r (mod 2N)
    /// and generates gadget decomposition key-switching keys.
    ///
    /// # Algorithm
    ///
    /// For each Galois element k:
    /// 1. Compute s_k = σ_k(s) by applying Galois automorphism to secret key
    /// 2. For each decomposition digit t = 0..num_digits:
    ///    a. Sample uniform random polynomial a_t
    ///    b. Sample error polynomial e_t from Gaussian (σ=3.2)
    ///    c. Compute rlk0[t] = -a_t·s + e_t + B^t·s_k
    ///    d. Set rlk1[t] = a_t
    ///
    /// where B = 2^base_w is the gadget base.
    ///
    /// # Arguments
    ///
    /// * `device` - Shared Metal device
    /// * `sk` - Secret key
    /// * `rotation_steps` - List of rotation steps (e.g., [±1, ±2, ±4, ..., ±N/2])
    /// * `params` - FHE parameters
    /// * `ntt_contexts` - Metal NTT contexts for each prime
    /// * `base_w` - Gadget base exponent (typically 20-30)
    ///
    /// # Returns
    ///
    /// MetalRotationKeys ready for use with rotate_by_steps()
    ///
    /// # Performance
    ///
    /// - For N=1024, 41 primes, 24 keys, base_w=20: ~45-60 seconds on M3 Max
    /// - 3× slower than single-key but produces correct results
    pub fn generate(
        device: Arc<MetalDevice>,
        sk: &SecretKey,
        rotation_steps: &[i32],
        params: &CliffordFHEParams,
        ntt_contexts: &[MetalNttContext],
        base_w: u32,
    ) -> Result<Self, String> {
        let n = params.n;
        let level = sk.level;
        let moduli = &params.moduli[..=level];
        let num_primes = moduli.len();

        // Compute number of decomposition digits needed
        let q_bits = Self::compute_total_modulus_bits(moduli);
        let num_digits = ((q_bits + base_w - 1) / base_w) as usize;

        let start = std::time::Instant::now();

        let mut keys = HashMap::new();

        for (_idx, &step) in rotation_steps.iter().enumerate() {
            // Convert rotation step to Galois element k
            let k = rotation_step_to_galois_element(step, n);

            // Skip if already generated (handles duplicates)
            if keys.contains_key(&k) {
                continue;
            }

            // Generate rotation key with gadget decomposition for σ_k
            let (rlk0, rlk1) = Self::generate_rotation_key_for_k_gadget(
                k,
                sk,
                moduli,
                n,
                ntt_contexts,
                base_w,
                num_digits,
            )?;

            keys.insert(k, (rlk0, rlk1));
        }

        println!("  Rotation keys generated in {:.3}s", start.elapsed().as_secs_f64());
        println!("  Pre-computing NTT-transformed keys for all levels...");
        let ntt_start = std::time::Instant::now();

        // Pre-compute NTT-transformed keys for all levels (0..=level)
        let keys_ntt = Self::precompute_ntt_keys(&keys, n, num_digits, params, ntt_contexts, level)?;

        println!("  NTT keys pre-computed in {:.3}s", ntt_start.elapsed().as_secs_f64());

        Ok(Self {
            device,
            keys,
            keys_ntt,
            base_w,
            num_digits,
            n,
            num_primes,
            level,
        })
    }

    /// Pre-compute NTT-transformed rotation keys for all levels
    ///
    /// For each Galois element and each level, transforms rotation keys to NTT domain
    /// using Metal GPU. This eliminates the need to transform keys at runtime.
    ///
    /// # Performance
    /// - Done once at key generation time
    /// - Uses Metal GPU for twist + NTT transformation
    /// - Memory cost: 2× key size per level
    /// - Runtime savings: 15-20% per rotation
    fn precompute_ntt_keys(
        keys: &HashMap<usize, (Vec<Vec<u64>>, Vec<Vec<u64>>)>,
        n: usize,
        num_digits: usize,
        params: &CliffordFHEParams,
        ntt_contexts: &[MetalNttContext],
        max_level: usize,
    ) -> Result<HashMap<usize, Vec<(Vec<Vec<u64>>, Vec<Vec<u64>>)>>, String> {
        let mut keys_ntt = HashMap::new();

        for (&k, (rlk0_coeff, rlk1_coeff)) in keys.iter() {
            // For this Galois element, pre-compute for all levels
            let mut keys_by_level = Vec::new();

            for level in 0..=max_level {
                let num_primes = level + 1;
                let moduli = &params.moduli[..num_primes];

                // Transform rlk0 and rlk1 to NTT domain for this level
                let mut rlk0_ntt = Vec::with_capacity(num_digits);
                let mut rlk1_ntt = Vec::with_capacity(num_digits);

                for t in 0..num_digits {
                    // Transform rlk0[t] to NTT domain (extract level primes)
                    let rlk0_t_ntt = Self::transform_key_digit_to_ntt(
                        &rlk0_coeff[t],
                        n,
                        moduli,
                        ntt_contexts,
                    )?;

                    // Transform rlk1[t] to NTT domain (extract level primes)
                    let rlk1_t_ntt = Self::transform_key_digit_to_ntt(
                        &rlk1_coeff[t],
                        n,
                        moduli,
                        ntt_contexts,
                    )?;

                    rlk0_ntt.push(rlk0_t_ntt);
                    rlk1_ntt.push(rlk1_t_ntt);
                }

                keys_by_level.push((rlk0_ntt, rlk1_ntt));
            }

            keys_ntt.insert(k, keys_by_level);
        }

        Ok(keys_ntt)
    }

    /// Transform a single key digit to NTT domain using Metal GPU
    ///
    /// Extracts the specified level primes and applies twist + NTT transform.
    fn transform_key_digit_to_ntt(
        key_digit_coeff: &[u64],
        n: usize,
        moduli: &[u64],
        ntt_contexts: &[MetalNttContext],
    ) -> Result<Vec<u64>, String> {
        let num_primes = moduli.len();
        let num_primes_full = key_digit_coeff.len() / n;
        let mut key_ntt_flat = vec![0u64; n * num_primes];

        // For each prime at this level
        for (prime_idx, _q) in moduli.iter().enumerate() {
            let ntt_ctx = &ntt_contexts[prime_idx];

            // Extract key polynomial for this prime from full key
            let mut key_poly = vec![0u64; n];
            for i in 0..n {
                key_poly[i] = key_digit_coeff[i * num_primes_full + prime_idx];
            }

            // Transform to NTT domain using Metal GPU (twist + forward NTT)
            ntt_ctx.coeff_to_ntt_gpu(&mut key_poly)?;

            // Store in flat layout for this level
            for i in 0..n {
                key_ntt_flat[i * num_primes + prime_idx] = key_poly[i];
            }
        }

        Ok(key_ntt_flat)
    }

    /// Compute bits in product of all moduli Q
    ///
    /// For gadget decomposition, we need to decompose values mod Q where Q = ∏ q_i.
    /// This computes the total bits needed to represent Q.
    fn compute_total_modulus_bits(moduli: &[u64]) -> u32 {
        // Sum bits of all primes (approximation of log2(Q))
        // Q = ∏ q_i → log2(Q) ≈ Σ log2(q_i)
        let mut total_bits = 0u32;
        for &q in moduli {
            total_bits += 64 - q.leading_zeros();
        }
        total_bits
    }

    /// Generate rotation key with gadget decomposition for Galois element k
    ///
    /// Creates key-switching keys (rlk0[], rlk1[]) where:
    /// - For t = 0..num_digits:
    ///   - rlk1[t] = a_t (uniform random)
    ///   - rlk0[t] ≈ -a_t·s + e_t + B^t·σ_k(s)
    ///
    /// This allows low-noise key switching during rotation.
    fn generate_rotation_key_for_k_gadget(
        k: usize,
        sk: &SecretKey,
        moduli: &[u64],
        n: usize,
        ntt_contexts: &[MetalNttContext],
        base_w: u32,
        num_digits: usize,
    ) -> Result<(Vec<Vec<u64>>, Vec<Vec<u64>>), String> {
        use rand::{thread_rng, Rng};
        use rand_distr::{Distribution, Normal};
        use num_bigint::BigInt;
        use num_traits::ToPrimitive;

        // Step 1: Apply Galois automorphism to secret key: s_k = σ_k(s)
        let (galois_map, galois_signs) = compute_galois_map(n, k);
        let s_k = Self::apply_galois_to_secret_key(sk, &galois_map, &galois_signs, moduli);

        // Convert secret key to flat layout
        let sk_flat = Self::sk_to_flat(sk, moduli);

        // Prepare gadget powers: B^t for t = 0..num_digits
        let base_big = BigInt::from(1u64) << base_w;  // B = 2^base_w

        let mut rlk0 = Vec::with_capacity(num_digits);
        let mut rlk1 = Vec::with_capacity(num_digits);

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 3.2).unwrap();

        // Generate one key pair for each decomposition digit
        for t in 0..num_digits {
            // Compute B^t
            let mut power_t_big = BigInt::from(1u64);
            for _ in 0..t {
                power_t_big *= &base_big;
            }

            // Convert B^t · s_k to flat RNS representation
            let mut bt_sk = vec![0u64; n * moduli.len()];
            for i in 0..n {
                for (j, &q) in moduli.iter().enumerate() {
                    let idx = i * moduli.len() + j;
                    let sk_val = s_k[idx];

                    // Multiply s_k[i] by B^t mod q
                    let power_t_mod_q = (&power_t_big % q).to_u64().unwrap_or(0);
                    bt_sk[idx] = Self::mul_mod(sk_val, power_t_mod_q, q);
                }
            }

            // Sample uniform random polynomial a_t
            let a_t: Vec<u64> = (0..(n * moduli.len()))
                .map(|i| {
                    let prime_idx = i % moduli.len();
                    rng.gen_range(0..moduli[prime_idx])
                })
                .collect();

            // Sample error polynomial e_t from Gaussian
            let e_t: Vec<i64> = (0..n).map(|_| {
                let sample: f64 = normal.sample(&mut rng);
                sample.round() as i64
            }).collect();
            let e_t_flat = Self::coeffs_to_flat_rns(&e_t, moduli);

            // Compute a_t · s using NTT multiplication
            let at_s = Self::multiply_polys_ntt(&a_t, &sk_flat, moduli, ntt_contexts)?;

            // Compute rlk0[t] = -B^t·s_k + a_t·s + e_t (in coefficient domain)
            // This ensures: rlk0[t] - a_t·s ≈ -B^t·s_k + e_t
            let mut rlk0_t = vec![0u64; n * moduli.len()];
            for i in 0..(n * moduli.len()) {
                let prime_idx = i % moduli.len();
                let q = moduli[prime_idx];

                // -B^t·s_k (negate in mod q)
                let neg_bt_sk = if bt_sk[i] == 0 { 0 } else { q - bt_sk[i] };

                // -B^t·s_k + a_t·s (mod q)
                let temp = Self::add_mod(neg_bt_sk, at_s[i], q);

                // -B^t·s_k + a_t·s + e_t (mod q)
                rlk0_t[i] = Self::add_mod(temp, e_t_flat[i], q);
            }

            // Store keys in COEFFICIENT DOMAIN (not NTT)
            // We'll transform during multiplication to match the twist convention
            rlk0.push(rlk0_t);
            rlk1.push(a_t);
        }

        Ok((rlk0, rlk1))
    }

    /// Get rotation key for a rotation step
    ///
    /// Returns None if the key was not generated (step not in original list).
    pub fn get_key_for_step(&self, step: i32) -> Option<&(Vec<Vec<u64>>, Vec<Vec<u64>>)> {
        let k = rotation_step_to_galois_element(step, self.n);
        self.keys.get(&k)
    }

    /// Get pre-computed NTT-transformed rotation key for a step and level
    ///
    /// Returns keys already in NTT domain (Montgomery), eliminating the need
    /// for runtime transformation. This saves 15-20% of rotation time.
    ///
    /// # Arguments
    /// * `step` - Rotation step (e.g., 1, 2, -1, etc.)
    /// * `level` - Ciphertext level (0-indexed, determines number of primes)
    ///
    /// # Returns
    /// Option with (rlk0_ntt[], rlk1_ntt[]) already in NTT domain for this level
    pub fn get_key_ntt_for_step(&self, step: i32, level: usize) -> Option<&(Vec<Vec<u64>>, Vec<Vec<u64>>)> {
        let k = rotation_step_to_galois_element(step, self.n);
        self.keys_ntt.get(&k).and_then(|keys_by_level| keys_by_level.get(level))
    }

    /// Get rotation key for a Galois element k directly
    pub fn get_key_for_galois_element(&self, k: usize) -> Option<&(Vec<Vec<u64>>, Vec<Vec<u64>>)> {
        self.keys.get(&k)
    }

    /// Get gadget base exponent
    pub fn base_w(&self) -> u32 {
        self.base_w
    }

    /// Get number of decomposition digits
    pub fn num_digits(&self) -> usize {
        self.num_digits
    }

    /// Check if rotation key exists for a step
    pub fn has_key_for_step(&self, step: i32) -> bool {
        let k = rotation_step_to_galois_element(step, self.n);
        self.keys.contains_key(&k)
    }

    /// Number of unique rotation keys stored
    pub fn num_keys(&self) -> usize {
        self.keys.len()
    }

    // ==================== Helper Functions ====================

    /// Apply Galois automorphism to secret key coefficients
    ///
    /// Applies the permutation σ_k to the secret key to get σ_k(s).
    fn apply_galois_to_secret_key(
        sk: &SecretKey,
        galois_map: &[u32],
        galois_signs: &[i32],
        moduli: &[u64],
    ) -> Vec<u64> {
        let n = sk.coeffs.len();
        let num_primes = moduli.len();
        let mut s_k = vec![0u64; n * num_primes];

        for i in 0..n {
            let target_idx = galois_map[i] as usize;
            let sign = galois_signs[i];

            for (j, &q) in moduli.iter().enumerate() {
                // Get coefficient value at RNS component j
                let val = sk.coeffs[i].values[j];

                // Apply sign correction
                let val_signed = if sign < 0 && val != 0 {
                    q - val // Negate: -x ≡ q - x (mod q)
                } else {
                    val
                };

                // Store in flat layout at target position
                let out_idx = target_idx * num_primes + j;
                s_k[out_idx] = val_signed;
            }
        }

        s_k
    }

    /// Convert signed coefficient vector to flat RNS representation
    fn coeffs_to_flat_rns(coeffs: &[i64], moduli: &[u64]) -> Vec<u64> {
        let n = coeffs.len();
        let num_primes = moduli.len();
        let mut flat = vec![0u64; n * num_primes];

        for (i, &coeff) in coeffs.iter().enumerate() {
            for (j, &q) in moduli.iter().enumerate() {
                // Convert signed coefficient to unsigned mod q
                let val_mod_q = if coeff >= 0 {
                    (coeff as u64) % q
                } else {
                    let abs_coeff = (-coeff) as u64;
                    let remainder = abs_coeff % q;
                    if remainder == 0 {
                        0
                    } else {
                        q - remainder
                    }
                };

                // Flat index: i * num_primes + j
                flat[i * num_primes + j] = val_mod_q;
            }
        }

        flat
    }

    /// Convert secret key to flat layout
    fn sk_to_flat(sk: &SecretKey, moduli: &[u64]) -> Vec<u64> {
        let n = sk.coeffs.len();
        let num_primes = moduli.len();
        let mut flat = vec![0u64; n * num_primes];

        for (i, rns) in sk.coeffs.iter().enumerate() {
            for j in 0..num_primes {
                flat[i * num_primes + j] = rns.values[j];
            }
        }

        flat
    }

    /// Transform polynomial to NTT domain (for storing keys in NTT form)
    ///
    /// This is called during key generation to pre-transform rotation keys to NTT domain.
    /// Storing keys in NTT domain saves 2 forward NTT operations per rotation!
    fn transform_to_ntt(
        poly_flat: &[u64],
        moduli: &[u64],
        ntt_contexts: &[MetalNttContext],
    ) -> Result<Vec<u64>, String> {
        let num_primes = moduli.len();
        let n = poly_flat.len() / num_primes;
        let mut result_flat = vec![0u64; n * num_primes];

        // For each RNS component (each prime), apply NTT
        for (prime_idx, &q) in moduli.iter().enumerate() {
            // Extract polynomial for this prime
            let mut poly = vec![0u64; n];
            for i in 0..n {
                poly[i] = poly_flat[i * num_primes + prime_idx];
            }

            let ntt_ctx = &ntt_contexts[prime_idx];

            // Apply twist for negacyclic convolution (mod x^n + 1)
            for i in 0..n {
                poly[i] = Self::mul_mod(poly[i], ntt_ctx.psi_powers()[i], q);
            }

            // Forward NTT
            ntt_ctx.forward(&mut poly)?;

            // Store back in flat layout (still in NTT domain!)
            for i in 0..n {
                result_flat[i * num_primes + prime_idx] = poly[i];
            }
        }

        Ok(result_flat)
    }

    /// Multiply two polynomials using NTT (CPU-side for now)
    ///
    /// Uses the existing MetalNttContext infrastructure.
    /// For rotation key generation, CPU-side is acceptable since it's a one-time operation.
    fn multiply_polys_ntt(
        a_flat: &[u64],
        b_flat: &[u64],
        moduli: &[u64],
        ntt_contexts: &[MetalNttContext],
    ) -> Result<Vec<u64>, String> {
        let num_primes = moduli.len();
        let n = a_flat.len() / num_primes;
        let mut result_flat = vec![0u64; n * num_primes];

        // For each RNS component (each prime), multiply using Metal NTT
        for (prime_idx, &q) in moduli.iter().enumerate() {
            // Extract polynomial for this prime
            let mut a_poly = vec![0u64; n];
            let mut b_poly = vec![0u64; n];

            for i in 0..n {
                a_poly[i] = a_flat[i * num_primes + prime_idx];
                b_poly[i] = b_flat[i * num_primes + prime_idx];
            }

            let ntt_ctx = &ntt_contexts[prime_idx];

            // Apply twist for negacyclic convolution (mod x^n + 1)
            for i in 0..n {
                a_poly[i] = Self::mul_mod(a_poly[i], ntt_ctx.psi_powers()[i], q);
                b_poly[i] = Self::mul_mod(b_poly[i], ntt_ctx.psi_powers()[i], q);
            }

            // Forward NTT
            ntt_ctx.forward(&mut a_poly)?;
            ntt_ctx.forward(&mut b_poly)?;

            // Pointwise multiply in NTT domain
            let mut result_poly = vec![0u64; n];
            ntt_ctx.pointwise_multiply(&a_poly, &b_poly, &mut result_poly)?;

            // Inverse NTT
            ntt_ctx.inverse(&mut result_poly)?;

            // Untwist for negacyclic convolution
            for i in 0..n {
                result_poly[i] = Self::mul_mod(result_poly[i], ntt_ctx.psi_inv_powers()[i], q);
            }

            // Store back in flat layout
            for i in 0..n {
                result_flat[i * num_primes + prime_idx] = result_poly[i];
            }
        }

        Ok(result_flat)
    }

    /// Modular multiplication: (a * b) mod q
    fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
        ((a as u128 * b as u128) % q as u128) as u64
    }

    /// Modular addition: (a + b) mod q
    fn add_mod(a: u64, b: u64, q: u64) -> u64 {
        let sum = (a as u128 + b as u128) % q as u128;
        sum as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use crate::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;
    use super::super::rotation::compute_bootstrap_rotation_steps;

    #[test]
    fn test_rotation_keys_generation() {
        // Use small parameters for quick test
        let params = CliffordFHEParams::new_test_ntt_1024();

        // Generate secret key using CPU keygen (faster for testing)
        let mut key_ctx = KeyContext::new(params.clone());
        let (_pk, sk, _evk) = key_ctx.keygen();

        // Create Metal device and NTT contexts
        let device = match MetalDevice::new() {
            Ok(dev) => Arc::new(dev),
            Err(e) => {
                println!("Skipping test: Metal not available: {}", e);
                return;
            }
        };

        // Create Metal key context to get NTT contexts
        // (MetalKeyContext::new creates NTT contexts using private find_primitive_2n_root)
        let metal_key_ctx = match MetalKeyContext::new(params.clone()) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: Failed to create Metal key context: {}", e);
                return;
            }
        };

        // Generate rotation keys for a few steps
        let rotation_steps = vec![1, -1, 2, -2];

        let rot_keys = MetalRotationKeys::generate(
            metal_key_ctx.device().clone(),
            &sk,
            &rotation_steps,
            &params,
            metal_key_ctx.ntt_contexts(),
            20,  // base_w for gadget decomposition
        ).expect("Failed to generate rotation keys");

        // Verify keys were generated
        assert_eq!(rot_keys.num_keys(), rotation_steps.len());

        // Verify we can retrieve keys
        for &step in &rotation_steps {
            assert!(rot_keys.has_key_for_step(step), "Missing key for step {}", step);
            let key = rot_keys.get_key_for_step(step).expect("Key not found");

            // With gadget decomposition:
            // key.0 = Vec<Vec<u64>> where outer Vec = num_digits, inner Vec = n × num_primes
            assert_eq!(key.0.len(), rot_keys.num_digits(), "Wrong number of digits in rlk0");
            assert_eq!(key.1.len(), rot_keys.num_digits(), "Wrong number of digits in rlk1");

            // Check each digit has correct polynomial size
            for digit_idx in 0..rot_keys.num_digits() {
                assert_eq!(key.0[digit_idx].len(), params.n * (sk.level + 1),
                          "Wrong polynomial size in rlk0 digit {}", digit_idx);
                assert_eq!(key.1[digit_idx].len(), params.n * (sk.level + 1),
                          "Wrong polynomial size in rlk1 digit {}", digit_idx);
            }
        }

        println!("Rotation keys test passed!");
    }

    #[test]
    fn test_rotation_keys_bootstrap_steps() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let n = params.n;

        // Compute all rotation steps needed for bootstrap
        let rotation_steps = compute_bootstrap_rotation_steps(n);

        // For N=1024: should have ~20-24 steps (powers of 2 in both directions)
        assert!(rotation_steps.len() >= 18 && rotation_steps.len() <= 24,
                "Bootstrap should need 18-24 rotation steps for N=1024, got {}", rotation_steps.len());

        println!("Bootstrap rotation steps for N={}: {:?}", n, rotation_steps);
    }

    #[test]
    fn test_apply_galois_to_secret_key() {
        let params = CliffordFHEParams::new_test_ntt_1024();

        // Generate secret key
        let mut key_ctx = KeyContext::new(params.clone());
        let (_pk, sk, _evk) = key_ctx.keygen();

        let moduli = &params.moduli[..=sk.level];

        // Apply identity automorphism (k=1)
        let (galois_map_id, galois_signs_id) = compute_galois_map(params.n, 1);
        let s_id = MetalRotationKeys::apply_galois_to_secret_key(&sk, &galois_map_id, &galois_signs_id, moduli);

        // Convert original sk to flat for comparison
        let sk_flat = MetalRotationKeys::sk_to_flat(&sk, moduli);

        // Identity should give same result
        assert_eq!(s_id, sk_flat, "Identity automorphism should preserve secret key");

        // Apply rotation by 1 (k=5)
        let (galois_map_rot, galois_signs_rot) = compute_galois_map(params.n, 5);
        let s_rot = MetalRotationKeys::apply_galois_to_secret_key(&sk, &galois_map_rot, &galois_signs_rot, moduli);

        // Should be different from original (unless by chance)
        assert_ne!(s_rot, sk_flat, "Rotation should change secret key representation");

        println!("Galois automorphism on secret key test passed!");
    }
}
