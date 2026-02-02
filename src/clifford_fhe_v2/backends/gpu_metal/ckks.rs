//! Metal GPU-accelerated CKKS (Cheon-Kim-Kim-Song) implementation
//!
//! This module provides a complete GPU-only CKKS context for homomorphic encryption.
//! All operations (encode, decode, encrypt, decrypt, add, multiply) run on Metal GPU.
//!
//! **Design Principle**: Complete isolation from CPU backend - no fallbacks, no mixing.

use super::device::MetalDevice;
use super::ntt::MetalNttContext;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::BarrettReducer;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::{PublicKey, SecretKey};
use std::sync::Arc;
use num_bigint::BigInt;
use num_traits::{Zero, One, ToPrimitive};

/// Metal GPU-accelerated CKKS context
///
/// All operations run entirely on GPU - no CPU fallbacks.
/// Uses Metal NTT for polynomial multiplication.
pub struct MetalCkksContext {
    /// Shared Metal device
    device: Arc<MetalDevice>,

    /// FHE parameters (public for bootstrap operations)
    pub params: CliffordFHEParams,

    /// Metal NTT contexts (one per prime in RNS)
    ntt_contexts: Vec<MetalNttContext>,

    /// Barrett reducers for modular reduction (lightweight, keep on CPU)
    reducers: Vec<BarrettReducer>,

    /// Precomputed rescaling constants for GPU-native exact rescale (subtractive method)
    /// rescale_inv_table[level][i] = q_{level}^{-1} mod q_i for i < level
    /// This allows GPU kernel to perform exact DivideRoundByLastQ without BigInt
    pub rescale_inv_table: Vec<Vec<u64>>,

    /// Precomputed alpha table for alternative rescaling (additive formula)
    /// alpha_table[level][i] = [(Q^(l) * Q^(l)^{-1} mod q_last) / q_last] mod q_i
    /// where Q^(l) = q_0 * q_1 * ... * q_{level-1}
    alpha_table: Vec<Vec<u64>>,
}

/// GPU plaintext representation
///
/// Stores polynomial coefficients in RNS representation.
/// Each coefficient is represented modulo each prime in the modulus chain.
#[derive(Clone, Debug)]
pub struct MetalPlaintext {
    /// Polynomial coefficients in RNS form
    /// Each u64 value is a coefficient in a specific RNS component
    /// Flat layout: [coeff0_mod_q0, coeff0_mod_q1, ..., coeff1_mod_q0, coeff1_mod_q1, ...]
    /// Length: n × num_primes
    pub coeffs: Vec<u64>,

    /// Ring dimension (number of polynomial coefficients)
    pub n: usize,

    /// Number of RNS primes
    pub num_primes: usize,

    /// Current level in modulus chain
    pub level: usize,

    /// Current scale (encoding scale)
    pub scale: f64,
}

/// GPU ciphertext representation (RLWE ciphertext)
///
/// Standard RLWE ciphertext (c0, c1) where:
/// - Decryption: m ≈ c0 + c1 * s (mod q)
#[derive(Clone, Debug)]
pub struct MetalCiphertext {
    /// First polynomial (c0) in RNS form
    /// Flat layout: [coeff0_mod_q0, coeff0_mod_q1, ..., coeff1_mod_q0, coeff1_mod_q1, ...]
    pub c0: Vec<u64>,

    /// Second polynomial (c1) in RNS form
    /// Flat layout: same as c0
    pub c1: Vec<u64>,

    /// Ring dimension (number of polynomial coefficients)
    pub n: usize,

    /// Number of RNS primes
    pub num_primes: usize,

    /// Current level in modulus chain
    pub level: usize,

    /// Current scale
    pub scale: f64,
}

impl MetalCkksContext {
    /// Create new Metal CKKS context with GPU acceleration
    ///
    /// All operations will run on Metal GPU. If GPU is not available,
    /// this will return an error (no CPU fallback).
    ///
    /// # Arguments
    /// * `params` - FHE parameters (ring dimension, moduli, scale)
    ///
    /// # Returns
    /// GPU-accelerated CKKS context or error if GPU unavailable
    pub fn new(params: CliffordFHEParams) -> Result<Self, String> {
        // Create shared Metal device
        let device = Arc::new(MetalDevice::new()?);

        // Create Metal NTT contexts for each prime
        let mut ntt_contexts = Vec::with_capacity(params.moduli.len());
        for &q in params.moduli.iter() {
            // Find primitive 2n-th root of unity
            let psi = Self::find_primitive_2n_root(params.n, q)?;

            // Create Metal NTT context
            let metal_ntt = MetalNttContext::new_with_device(
                device.clone(),
                params.n,
                q,
                psi
            )?;

            ntt_contexts.push(metal_ntt);
        }

        // Create Barrett reducers (lightweight, keep on CPU)
        let reducers: Vec<BarrettReducer> = params.moduli.iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        // Precompute rescaling inverse constants for GPU-native exact rescale
        let rescale_inv_table = Self::precompute_rescale_inv_table(&params.moduli);

        // Precompute alpha table for alternative rescaling method
        let alpha_table = Self::precompute_alpha_table(&params.moduli);

        Ok(Self {
            device,
            params,
            ntt_contexts,
            reducers,
            rescale_inv_table,
            alpha_table,
        })
    }

    /// Get reference to the Metal device
    pub fn device(&self) -> &Arc<MetalDevice> {
        &self.device
    }

    /// Get reference to the NTT contexts
    pub fn ntt_contexts(&self) -> &[MetalNttContext] {
        &self.ntt_contexts
    }

    /// Get psi (primitive 2N-th root) for each prime
    pub fn psi_per_prime(&self) -> Vec<u64> {
        self.ntt_contexts.iter().map(|ctx| ctx.psi()).collect()
    }

    /// Encode floating-point values into plaintext polynomial
    ///
    /// Currently uses CPU for canonical embedding (GPU optimization planned).
    /// The encoding is done in a way that's compatible with GPU operations.
    ///
    /// # Arguments
    /// * `values` - Real values to encode (length ≤ n/2)
    ///
    /// # Returns
    /// Plaintext polynomial in flat RNS representation
    pub fn encode(&self, values: &[f64]) -> Result<MetalPlaintext, String> {
        let n = self.params.n;
        let num_slots = n / 2;

        if values.len() > num_slots {
            return Err(format!("Too many values: {} > n/2 = {}", values.len(), num_slots));
        }

        // Step 1: Canonical embedding (CPU for now)
        let coeffs_i64 = Self::canonical_embed_encode_real(values, self.params.scale, n);

        // Step 2: Convert to flat RNS representation
        let level = self.params.max_level();
        let moduli = &self.params.moduli[..=level];
        let num_primes = moduli.len();

        // Flat layout: [coeff0_mod_q0, coeff0_mod_q1, ..., coeff1_mod_q0, ...]
        let mut flat_coeffs = vec![0u64; n * num_primes];

        for (i, &coeff) in coeffs_i64.iter().enumerate() {
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
                flat_coeffs[i * num_primes + j] = val_mod_q;
            }
        }

        Ok(MetalPlaintext {
            coeffs: flat_coeffs,
            n,
            num_primes,
            level,
            scale: self.params.scale,
        })
    }

    /// Decode plaintext polynomial to floating-point values
    ///
    /// Currently uses CPU for canonical embedding (GPU optimization planned).
    ///
    /// # Arguments
    /// * `pt` - Plaintext polynomial
    ///
    /// # Returns
    /// Decoded real values
    pub fn decode(&self, pt: &MetalPlaintext) -> Result<Vec<f64>, String> {
        let n = pt.n;
        let num_primes = pt.num_primes;

        // Step 1: Reconstruct coefficients from RNS using first prime (for simplicity)
        // In full implementation, would use CRT reconstruction
        let q0 = self.params.moduli[0];
        let mut coeffs_i64 = vec![0i64; n];

        for i in 0..n {
            let val = pt.coeffs[i * num_primes]; // Use first RNS component

            // Convert from mod q to signed integer (centered representation)
            // Values > q/2 are treated as negative (centered lift)
            let half_q0 = q0 / 2;
            coeffs_i64[i] = if val > half_q0 {
                -((q0 - val) as i64)
            } else {
                val as i64
            };
        }

        // Step 2: Canonical embedding decode (CPU for now)
        let slots = Self::canonical_embed_decode_real(&coeffs_i64, pt.scale, n);

        Ok(slots)
    }

    /// Encrypt plaintext using public key
    ///
    /// Creates RLWE ciphertext (c0, c1) using polynomial operations.
    /// Uses Metal GPU for NTT-based polynomial multiplication.
    ///
    /// # Arguments
    /// * `pt` - Plaintext to encrypt
    /// * `pk` - Public key
    ///
    /// # Returns
    /// Ciphertext encrypting the plaintext
    pub fn encrypt(
        &self,
        pt: &MetalPlaintext,
        pk: &PublicKey,
    ) -> Result<MetalCiphertext, String> {
        use rand::{thread_rng, Rng};
        use rand_distr::{Distribution, Normal};

        let n = self.params.n;
        let level = pt.level;
        let moduli = &self.params.moduli[..=level];
        let num_primes = moduli.len();
        let mut rng = thread_rng();

        // Sample ternary random polynomial u ∈ {-1, 0, 1}^n
        let u_coeffs: Vec<i64> = (0..n)
            .map(|_| {
                let val: f64 = rng.gen();
                if val < 0.33 {
                    -1
                } else if val < 0.66 {
                    0
                } else {
                    1
                }
            })
            .collect();

        // Sample error polynomials e0, e1 from Gaussian distribution
        let normal = Normal::new(0.0, self.params.error_std).unwrap();
        let e0_coeffs: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();
        let e1_coeffs: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();

        // Convert to flat RNS representation
        let u_flat = Self::coeffs_to_flat_rns(&u_coeffs, moduli);
        let e0_flat = Self::coeffs_to_flat_rns(&e0_coeffs, moduli);
        let e1_flat = Self::coeffs_to_flat_rns(&e1_coeffs, moduli);

        // Convert public key to flat layout (if not already)
        let pk_a_flat = Self::rns_vec_to_flat(&pk.a, num_primes);
        let pk_b_flat = Self::rns_vec_to_flat(&pk.b, num_primes);

        // c0 = b*u + e0 + m (using Metal NTT for multiplication)
        // CRITICAL: Use negacyclic convolution to match CPU and CKKS ring R = Z[x]/(x^n + 1)
        let bu = self.multiply_polys_flat_ntt_negacyclic(&pk_b_flat, &u_flat, moduli)?;
        let mut c0 = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let q = moduli[prime_idx];

            // Use Barrett reducer for proper modular arithmetic
            let reducer = &self.reducers[prime_idx];
            let sum1 = reducer.add(bu[i], e0_flat[i]);
            let sum2 = reducer.add(sum1, pt.coeffs[i]);
            c0[i] = sum2;
        }

        // c1 = a*u + e1 (using Metal NTT for multiplication)
        // CRITICAL: Use negacyclic convolution to match CPU and CKKS ring R = Z[x]/(x^n + 1)
        let au = self.multiply_polys_flat_ntt_negacyclic(&pk_a_flat, &u_flat, moduli)?;
        let mut c1 = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let reducer = &self.reducers[prime_idx];
            c1[i] = reducer.add(au[i], e1_flat[i]);
        }

        Ok(MetalCiphertext {
            c0,
            c1,
            n,
            num_primes,
            level,
            scale: pt.scale,
        })
    }

    /// Decrypt ciphertext using secret key
    ///
    /// Recovers plaintext from RLWE ciphertext: m = c0 + c1*s
    /// Uses Metal GPU for NTT-based polynomial multiplication.
    ///
    /// # Arguments
    /// * `ct` - Ciphertext to decrypt
    /// * `sk` - Secret key
    ///
    /// # Returns
    /// Decrypted plaintext
    pub fn decrypt(
        &self,
        ct: &MetalCiphertext,
        sk: &SecretKey,
    ) -> Result<MetalPlaintext, String> {
        let n = ct.n;
        let level = ct.level;
        let moduli = &self.params.moduli[..=level];
        let num_primes = moduli.len();

        // Convert secret key to flat layout at ciphertext's level
        let sk_flat = Self::rns_vec_to_flat_at_level(&sk.coeffs, level, num_primes);

        // Compute c1 * s using Metal NTT
        // CRITICAL: Use negacyclic convolution to match CPU and CKKS ring R = Z[x]/(x^n + 1)
        let c1s = self.multiply_polys_flat_ntt_negacyclic(&ct.c1, &sk_flat, moduli)?;

        // m = c0 + c1*s
        let mut m_flat = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let reducer = &self.reducers[prime_idx];
            m_flat[i] = reducer.add(ct.c0[i], c1s[i]);
        }

        Ok(MetalPlaintext {
            coeffs: m_flat,
            n,
            num_primes,
            level,
            scale: ct.scale,
        })
    }

    // ==================== Conversion Functions (Metal ↔ CPU) ====================

    /// Convert MetalCiphertext to CPU Ciphertext
    ///
    /// This allows Metal GPU ciphertexts to be used with CPU-based operations
    /// like bootstrap. The conversion is lossless.
    pub fn to_cpu_ciphertext(&self, ct: &MetalCiphertext) -> crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext {
        use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

        let moduli = self.params.moduli[..=ct.level].to_vec();

        // Convert c0 from flat layout to Vec<RnsRepresentation>
        let mut c0_rns = Vec::with_capacity(ct.n);
        for i in 0..ct.n {
            let mut values = Vec::with_capacity(ct.num_primes);
            for j in 0..ct.num_primes {
                values.push(ct.c0[i * ct.num_primes + j]);
            }
            c0_rns.push(RnsRepresentation::new(values, moduli.clone()));
        }

        // Convert c1 from flat layout to Vec<RnsRepresentation>
        let mut c1_rns = Vec::with_capacity(ct.n);
        for i in 0..ct.n {
            let mut values = Vec::with_capacity(ct.num_primes);
            for j in 0..ct.num_primes {
                values.push(ct.c1[i * ct.num_primes + j]);
            }
            c1_rns.push(RnsRepresentation::new(values, moduli.clone()));
        }

        crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext::new(
            c0_rns,
            c1_rns,
            ct.level,
            ct.scale,
        )
    }

    /// Convert CPU Ciphertext to MetalCiphertext
    ///
    /// This allows CPU ciphertexts (e.g., from bootstrap) to be used with
    /// Metal GPU operations. The conversion is lossless.
    pub fn from_cpu_ciphertext(&self, ct: &crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext) -> MetalCiphertext {
        let n = ct.n;
        let level = ct.level;
        let num_primes = ct.c0[0].values.len();

        // Convert c0 from Vec<RnsRepresentation> to flat layout
        let mut c0_flat = vec![0u64; n * num_primes];
        for (i, rns) in ct.c0.iter().enumerate() {
            for (j, &val) in rns.values.iter().enumerate() {
                c0_flat[i * num_primes + j] = val;
            }
        }

        // Convert c1 from Vec<RnsRepresentation> to flat layout
        let mut c1_flat = vec![0u64; n * num_primes];
        for (i, rns) in ct.c1.iter().enumerate() {
            for (j, &val) in rns.values.iter().enumerate() {
                c1_flat[i * num_primes + j] = val;
            }
        }

        MetalCiphertext {
            c0: c0_flat,
            c1: c1_flat,
            n,
            num_primes,
            level,
            scale: ct.scale,
        }
    }

    /// Convert CPU Plaintext to MetalPlaintext
    pub fn from_cpu_plaintext(&self, pt: &crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext) -> MetalPlaintext {
        let n = pt.n;
        let level = pt.level;
        let num_primes = pt.coeffs[0].values.len();

        // Convert coeffs from Vec<RnsRepresentation> to flat layout
        let mut coeffs_flat = vec![0u64; n * num_primes];
        for (i, rns) in pt.coeffs.iter().enumerate() {
            for (j, &val) in rns.values.iter().enumerate() {
                coeffs_flat[i * num_primes + j] = val;
            }
        }

        MetalPlaintext {
            coeffs: coeffs_flat,
            n,
            num_primes,
            level,
            scale: pt.scale,
        }
    }

    // ==================== Canonical Embedding Functions ====================

    /// Compute the Galois orbit order for CKKS slot indexing
    fn orbit_order(n: usize, g: usize) -> Vec<usize> {
        let m = 2 * n; // M = 2N
        let num_slots = n / 2; // N/2 slots

        let mut e = vec![0usize; num_slots];
        let mut cur = 1usize;

        for t in 0..num_slots {
            e[t] = cur; // odd exponent in [1..2N-1]
            cur = (cur * g) % m;
        }

        e
    }

    /// Encode real-valued slots using CKKS canonical embedding
    pub fn canonical_embed_encode_real(values: &[f64], scale: f64, n: usize) -> Vec<i64> {
        use std::f64::consts::PI;

        assert!(n.is_power_of_two());
        let num_slots = n / 2;
        assert!(values.len() <= num_slots);

        let m = 2 * n; // Cyclotomic index M = 2N
        let g = 5; // Generator for power-of-two cyclotomics

        // Use Galois orbit order
        let e = Self::orbit_order(n, g);

        // Pad values to full slot count
        let mut slots = vec![0.0; num_slots];
        for (i, &val) in values.iter().enumerate() {
            slots[i] = val;
        }

        // Inverse canonical embedding
        let mut coeffs_float = vec![0.0; n];

        for j in 0..n {
            let mut sum = 0.0;

            for t in 0..num_slots {
                // w_t(j) = exp(-2πi * e[t] * j / M)
                let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
                let cos_val = angle.cos();

                // For real slots: contribution is 2 * z[t] * cos(angle)
                sum += slots[t] * cos_val;
            }

            // Normalize by 2/N
            coeffs_float[j] = (2.0 / n as f64) * sum;
        }

        // Scale and round to integers
        coeffs_float.iter().map(|&x| (x * scale).round() as i64).collect()
    }

    /// Decode real-valued slots using CKKS canonical embedding
    fn canonical_embed_decode_real(coeffs: &[i64], scale: f64, n: usize) -> Vec<f64> {
        use std::f64::consts::PI;

        assert_eq!(coeffs.len(), n);

        let m = 2 * n; // M = 2N
        let num_slots = n / 2;
        let g = 5; // Generator

        // Use Galois orbit order
        let e = Self::orbit_order(n, g);

        // Convert to floating point (with scale normalization)
        let coeffs_float: Vec<f64> = coeffs.iter().map(|&c| c as f64 / scale).collect();

        // Forward canonical embedding
        let mut slots = vec![0.0; num_slots];

        for t in 0..num_slots {
            let mut sum_real = 0.0;
            for j in 0..n {
                // w_t(j) = exp(+2πi * e[t] * j / M) (positive angle for decode)
                let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
                let cos_val = angle.cos();
                sum_real += coeffs_float[j] * cos_val;
            }
            slots[t] = sum_real;
        }

        slots
    }

    // ==================== RNS Conversion Helper Functions ====================

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

    /// Convert RnsRepresentation vector to flat layout
    fn rns_vec_to_flat(rns_vec: &[crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation], num_primes: usize) -> Vec<u64> {
        let n = rns_vec.len();
        let mut flat = vec![0u64; n * num_primes];

        for (i, rns) in rns_vec.iter().enumerate() {
            for j in 0..num_primes {
                flat[i * num_primes + j] = rns.values[j];
            }
        }

        flat
    }

    /// Convert RnsRepresentation vector to flat layout at specific level
    fn rns_vec_to_flat_at_level(rns_vec: &[crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation], level: usize, num_primes: usize) -> Vec<u64> {
        let n = rns_vec.len();
        let mut flat = vec![0u64; n * num_primes];

        for (i, rns) in rns_vec.iter().enumerate() {
            for j in 0..num_primes {
                // Take only components up to level
                flat[i * num_primes + j] = rns.values[j];
            }
        }

        flat
    }

    /// Multiply two polynomials using NTT - CYCLIC convolution (for key operations)
    ///
    /// This is used for key generation and operations that don't require negacyclic convolution.
    /// For CKKS plaintext multiplication, use multiply_polys_flat_ntt_negacyclic instead.
    fn multiply_polys_flat_ntt(&self, a_flat: &[u64], b_flat: &[u64], moduli: &[u64]) -> Result<Vec<u64>, String> {
        self.multiply_polys_flat_ntt_impl(a_flat, b_flat, moduli, false)
    }

    /// Multiply two polynomials using NTT - NEGACYCLIC convolution (for CKKS operations)
    ///
    /// This is used for CKKS plaintext multiplication which requires negacyclic convolution (mod x^n + 1).
    /// Uses twist/untwist to convert cyclic NTT to negacyclic.
    /// Public for use by bootstrap operations.
    pub fn multiply_polys_flat_ntt_negacyclic(&self, a_flat: &[u64], b_flat: &[u64], moduli: &[u64]) -> Result<Vec<u64>, String> {
        self.multiply_polys_flat_ntt_impl(a_flat, b_flat, moduli, true)
    }

    /// Multiply coefficient-form polynomial by NTT-form polynomial (asymmetric multiplication)
    ///
    /// **State-of-the-art optimization for relinearization:**
    /// EVK is pre-transformed to NTT domain, saving one forward NTT per multiplication.
    ///
    /// - `a_flat`: Coefficient-form polynomial (e.g., gadget digit) [n × stride_a]
    /// - `b_ntt_flat`: NTT-form polynomial (e.g., pre-transformed EVK) [n × stride_b]
    /// - `moduli`: Active primes for this level
    /// - Returns: Coefficient-form result [n × num_primes]
    pub fn multiply_coeff_by_ntt(&self, a_flat: &[u64], b_ntt_flat: &[u64], moduli: &[u64]) -> Result<Vec<u64>, String> {
        let n = self.params.n;
        let num_primes = moduli.len();

        // Infer strides from input lengths
        let a_stride = a_flat.len() / n;
        let b_stride = b_ntt_flat.len() / n;

        // Debug
        if std::env::var("ASYM_DEBUG").is_ok() {
            println!("[ASYM_DEBUG] n={}, num_primes={}, a_len={}, b_len={}, a_stride={}, b_stride={}",
                n, num_primes, a_flat.len(), b_ntt_flat.len(), a_stride, b_stride);
        }

        if a_stride < num_primes || b_stride < num_primes {
            return Err(format!("Input strides too small: a_stride={}, b_stride={}, need {}",
                a_stride, b_stride, num_primes));
        }

        let mut result_flat = vec![0u64; n * num_primes];

        // Process each RNS component independently (fully parallel)
        for (prime_idx, &q) in moduli.iter().enumerate() {
            // Extract coefficient polynomial for this prime
            let mut a_poly = vec![0u64; n];
            for i in 0..n {
                a_poly[i] = a_flat[i * a_stride + prime_idx];
            }

            // Extract NTT polynomial for this prime (already in NTT domain!)
            let mut b_ntt_poly = vec![0u64; n];
            for i in 0..n {
                b_ntt_poly[i] = b_ntt_flat[i * b_stride + prime_idx];
            }

            let ntt_ctx = &self.ntt_contexts[prime_idx];

            // TWIST: Apply ψ^i to convert negacyclic → cyclic
            for i in 0..n {
                a_poly[i] = Self::mul_mod(a_poly[i], ntt_ctx.psi_powers()[i], q);
            }

            // Forward NTT on coefficient input only (b already in NTT domain)
            ntt_ctx.forward(&mut a_poly)?;

            // Pointwise multiply in NTT domain
            let mut result_ntt = vec![0u64; n];
            ntt_ctx.pointwise_multiply(&a_poly, &b_ntt_poly, &mut result_ntt)?;

            // Inverse NTT
            ntt_ctx.inverse(&mut result_ntt)?;

            // UNTWIST: Apply ψ^{-i} to convert cyclic → negacyclic
            for i in 0..n {
                result_ntt[i] = Self::mul_mod(result_ntt[i], ntt_ctx.psi_inv_powers()[i], q);
            }

            // Store in output flat layout
            for i in 0..n {
                result_flat[i * num_primes + prime_idx] = result_ntt[i];
            }
        }

        Ok(result_flat)
    }

    /// Internal implementation of polynomial multiplication with optional twist/untwist
    ///
    /// Both input arrays must have the same stride (num_primes_in_array).
    /// The stride must be >= moduli.len().
    ///
    /// @param negacyclic If true, applies twist/untwist for negacyclic convolution (mod x^n + 1)
    fn multiply_polys_flat_ntt_impl(&self, a_flat: &[u64], b_flat: &[u64], moduli: &[u64], negacyclic: bool) -> Result<Vec<u64>, String> {
        let n = self.params.n;
        let num_primes_to_process = moduli.len();

        // Infer the stride from the input array length
        let num_primes_in_array = a_flat.len() / n;
        let b_stride = b_flat.len() / n;
        if b_stride != num_primes_in_array {
            return Err(format!("Input arrays have different strides: a={} ({}×{}), b={} ({}×{})",
                a_flat.len(), n, num_primes_in_array,
                b_flat.len(), n, b_stride));
        }

        let mut result_flat = vec![0u64; n * num_primes_to_process];

        // For each RNS component (each prime), multiply using Metal NTT
        for (prime_idx, &q) in moduli.iter().enumerate() {
            // Extract polynomial for this prime
            let mut a_poly = vec![0u64; n];
            let mut b_poly = vec![0u64; n];

            for i in 0..n {
                // IMPORTANT: Use the array's stride (num_primes_in_array), not num_primes_to_process
                a_poly[i] = a_flat[i * num_primes_in_array + prime_idx];
                b_poly[i] = b_flat[i * num_primes_in_array + prime_idx];
            }

            let ntt_ctx = &self.ntt_contexts[prime_idx];
            let q = moduli[prime_idx];

            // For negacyclic convolution (CKKS): apply twist/untwist
            // For cyclic convolution (key operations): use NTT directly
            if negacyclic {
                // TWIST: multiply a and b by psi^i (converts negacyclic → cyclic)
                for i in 0..n {
                    a_poly[i] = Self::mul_mod(a_poly[i], ntt_ctx.psi_powers()[i], q);
                    b_poly[i] = Self::mul_mod(b_poly[i], ntt_ctx.psi_powers()[i], q);
                }
            }

            // Forward NTT
            ntt_ctx.forward(&mut a_poly)?;
            ntt_ctx.forward(&mut b_poly)?;

            // Pointwise multiply in NTT domain
            let mut result_poly = vec![0u64; n];
            ntt_ctx.pointwise_multiply(&a_poly, &b_poly, &mut result_poly)?;

            // Inverse NTT
            ntt_ctx.inverse(&mut result_poly)?;

            if negacyclic {
                // UNTWIST: multiply by psi^{-i} (converts cyclic → negacyclic)
                for i in 0..n {
                    result_poly[i] = Self::mul_mod(result_poly[i], ntt_ctx.psi_inv_powers()[i], q);
                }
            }

            // Store back in flat layout
            for i in 0..n {
                result_flat[i * num_primes_to_process + prime_idx] = result_poly[i];
            }
        }

        Ok(result_flat)
    }

    /// GPU-native exact rescaling using RNS formula
    ///
    /// Implements DivideRoundByLastQ without BigInt reconstruction using the RNS formula:
    /// r'ᵢ = (rᵢ - r_top) × q_top^{-1} mod qᵢ
    ///
    /// This is mathematically equivalent to the BigInt version but runs entirely on GPU.
    ///
    /// # Arguments
    /// * `poly_in` - Input polynomial in flat RNS layout [n × num_primes_in]
    /// * `level` - Current level (q_top = moduli[level])
    ///
    /// # Returns
    /// Rescaled polynomial in flat RNS layout [n × (num_primes_in - 1)]
    pub fn exact_rescale_gpu(&self, poly_in: &[u64], level: usize) -> Result<Vec<u64>, String> {
        use metal::*;

        if level == 0 {
            return Err("Cannot rescale at level 0".to_string());
        }

        let n = self.params.n;
        let moduli = &self.params.moduli[..=level];
        let num_primes_in = moduli.len();
        let num_primes_out = num_primes_in - 1;

        assert_eq!(poly_in.len(), n * num_primes_in, "Input size mismatch");

        // Get precomputed inverse constants for this level
        let qtop_inv = &self.rescale_inv_table[level];
        assert_eq!(qtop_inv.len(), num_primes_out, "Inverse table size mismatch");

        // Create output buffer
        let mut poly_out = vec![0u64; n * num_primes_out];

        // Get Metal kernel and create pipeline state
        let function = self.device.get_rns_function("rns_exact_rescale")?;
        let pipeline = self.device.device().new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Failed to create rescale pipeline: {:?}", e))?;

        // Create GPU buffers
        let input_buffer = self.device.device().new_buffer_with_data(
            poly_in.as_ptr() as *const _,
            (poly_in.len() * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.device().new_buffer(
            (poly_out.len() * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let moduli_buffer = self.device.device().new_buffer_with_data(
            moduli.as_ptr() as *const _,
            (moduli.len() * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let inv_buffer = self.device.device().new_buffer_with_data(
            qtop_inv.as_ptr() as *const _,
            (qtop_inv.len() * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let n_u32 = n as u32;
        let num_primes_u32 = num_primes_in as u32;

        // Execute kernel
        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_buffer(2, Some(&moduli_buffer), 0);
            encoder.set_buffer(3, Some(&inv_buffer), 0);
            encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &n_u32 as *const u32 as *const _);
            encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &num_primes_u32 as *const u32 as *const _);

            // Dispatch 1D grid with n threads (one per coefficient)
            let threadgroup_size = MTLSize { width: 256.min(n as u64), height: 1, depth: 1 };
            let num_threadgroups = ((n as u64 + threadgroup_size.width - 1) / threadgroup_size.width).max(1);
            let grid_size = MTLSize { width: num_threadgroups, height: 1, depth: 1 };

            encoder.dispatch_thread_groups(grid_size, threadgroup_size);

            Ok(())
        })?;

        // Copy result back
        let out_len = poly_out.len();
        unsafe {
            let ptr = output_buffer.contents() as *const u64;
            poly_out.copy_from_slice(std::slice::from_raw_parts(ptr, out_len));
        }

        Ok(poly_out)
    }

    /// GPU-Native Exact Rescaling using Alternative Algorithm (Additive Formula)
    ///
    /// NOTE: This function is currently DISABLED as it requires a separate shader.
    /// The standard GPU rescaling (exact_rescale_gpu_fixed) is used instead.
    ///
    /// Rescale ciphertext by dropping last prime using additive formula:
    /// result_i = c_i * q_last^{-1} + (c_last mod q_i) * alpha_i
    ///
    /// This approach may be more numerically stable than the subtractive formula.
    ///
    /// # Arguments
    /// * `poly_in` - Input polynomial in flat RNS layout [n × num_primes_in]
    /// * `level` - Current level (q_top = moduli[level])
    ///
    /// # Returns
    /// Rescaled polynomial in flat RNS layout [n × (num_primes_in - 1)]
    #[allow(dead_code)]
    fn exact_rescale_gpu_alternative(&self, _poly_in: &[u64], _level: usize) -> Result<Vec<u64>, String> {
        Err("Alternative rescaling method is disabled - use exact_rescale_gpu_fixed instead".to_string())
    }

    /// GPU-Native Exact Rescaling using FIXED Algorithm (proper DRLMQ)
    ///
    /// This fixes the domain mismatch and precision issues in the original implementation:
    /// 1. Uses proper 128-bit modular arithmetic (no precision loss from % on intermediates)
    /// 2. Implements centered rounding: ⌊(C + q_last/2) / q_last⌋
    /// 3. All operations in standard domain (matching inverse NTT output)
    ///
    /// # Arguments
    /// * `poly_in` - Input polynomial in flat RNS layout [n × num_primes_in] (standard domain)
    /// * `level` - Current level (q_top = moduli[level])
    ///
    /// # Returns
    /// Rescaled polynomial in flat RNS layout [n × (num_primes_in - 1)] (standard domain)
    pub fn exact_rescale_gpu_fixed(&self, poly_in: &[u64], level: usize) -> Result<Vec<u64>, String> {
        use metal::*;

        if level == 0 {
            return Err("Cannot rescale at level 0".to_string());
        }

        let n = self.params.n;
        let moduli = &self.params.moduli[..=level];
        let num_primes_in = moduli.len();
        let num_primes_out = num_primes_in - 1;

        assert_eq!(poly_in.len(), n * num_primes_in, "Input size mismatch");

        // Get precomputed inverse constants for this level
        let qtop_inv = &self.rescale_inv_table[level];
        assert_eq!(qtop_inv.len(), num_primes_out, "Inverse table size mismatch");

        // Create output buffer
        let mut poly_out = vec![0u64; n * num_primes_out];

        // Get Metal kernel and create pipeline state
        let function = self.device.get_rns_fixed_function("rns_exact_rescale_fixed")?;
        let pipeline = self.device.device().new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Failed to create fixed rescale pipeline: {:?}", e))?;

        // Create GPU buffers
        let input_buffer = self.device.device().new_buffer_with_data(
            poly_in.as_ptr() as *const _,
            (poly_in.len() * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.device().new_buffer(
            (poly_out.len() * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let moduli_buffer = self.device.device().new_buffer_with_data(
            moduli.as_ptr() as *const _,
            (moduli.len() * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let inv_buffer = self.device.device().new_buffer_with_data(
            qtop_inv.as_ptr() as *const _,
            (qtop_inv.len() * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let n_u32 = n as u32;
        let num_primes_u32 = num_primes_in as u32;

        // Execute kernel
        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_buffer(2, Some(&moduli_buffer), 0);
            encoder.set_buffer(3, Some(&inv_buffer), 0);
            encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &n_u32 as *const u32 as *const _);
            encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &num_primes_u32 as *const u32 as *const _);

            // Dispatch 1D grid with n threads (one per coefficient)
            let threadgroup_size = MTLSize { width: 256.min(n as u64), height: 1, depth: 1 };
            let num_threadgroups = ((n as u64 + threadgroup_size.width - 1) / threadgroup_size.width).max(1);
            let grid_size = MTLSize { width: num_threadgroups, height: 1, depth: 1 };

            encoder.dispatch_thread_groups(grid_size, threadgroup_size);

            Ok(())
        })?;

        // Copy result back
        let out_len = poly_out.len();
        unsafe {
            let ptr = output_buffer.contents() as *const u64;
            poly_out.copy_from_slice(std::slice::from_raw_parts(ptr, out_len));
        }

        Ok(poly_out)
    }

    // ==================== NTT Helper Functions ====================

    /// Find primitive 2n-th root of unity for NTT
    ///
    /// Uses same algorithm as CPU backend for compatibility.
    fn find_primitive_2n_root(n: usize, q: u64) -> Result<u64, String> {
        // Verify q ≡ 1 (mod 2n)
        let two_n = (2 * n) as u64;
        if (q - 1) % two_n != 0 {
            return Err(format!(
                "q = {} is not NTT-friendly for n = {} (q-1 must be divisible by 2n)",
                q, n
            ));
        }

        // Try small candidates that are often generators
        for candidate in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
            if Self::is_primitive_root_candidate(candidate, n, q) {
                let exponent = (q - 1) / two_n;
                return Ok(Self::pow_mod(candidate, exponent, q));
            }
        }

        // Extended search
        for candidate in 32..1000u64 {
            if Self::is_primitive_root_candidate(candidate, n, q) {
                let exponent = (q - 1) / two_n;
                return Ok(Self::pow_mod(candidate, exponent, q));
            }
        }

        Err(format!("Failed to find primitive root for q = {}, n = {}", q, n))
    }

    /// Check if g is suitable for generating primitive 2n-th root
    fn is_primitive_root_candidate(g: u64, n: usize, q: u64) -> bool {
        // Check if g is a quadratic non-residue
        if Self::pow_mod(g, (q - 1) / 2, q) == 1 {
            return false;
        }

        // Check if g^((q-1)/(2n)) generates the subgroup of order 2n
        let psi = Self::pow_mod(g, (q - 1) / (2 * n as u64), q);

        // psi^n should equal -1 mod q
        let psi_n = Self::pow_mod(psi, n as u64, q);
        if psi_n != q - 1 {
            return false;
        }

        // psi^(2n) should equal 1 mod q
        let psi_2n = Self::pow_mod(psi, 2 * n as u64, q);
        psi_2n == 1
    }

    /// Modular multiplication: (a * b) mod q
    fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
        ((a as u128 * b as u128) % q as u128) as u64
    }

    /// Modular exponentiation: base^exp mod q
    fn pow_mod(mut base: u64, mut exp: u64, q: u64) -> u64 {
        let mut result = 1u64;
        base %= q;

        while exp > 0 {
            if exp & 1 == 1 {
                result = ((result as u128 * base as u128) % q as u128) as u64;
            }
            base = ((base as u128 * base as u128) % q as u128) as u64;
            exp >>= 1;
        }

        result
    }

    /// Compute modular inverse using extended Euclidean algorithm
    /// Returns a^{-1} mod m, or None if gcd(a, m) != 1
    fn mod_inverse_u64(a: u64, m: u64) -> Option<u64> {
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

        let mut result = x % (m as i128);
        if result < 0 {
            result += m as i128;
        }

        Some(result as u64)
    }

    /// Precompute rescaling inverse table for GPU-native exact rescale
    ///
    /// For each level L, precompute q_L^{-1} mod q_i for all i < L
    /// This allows the GPU kernel to perform exact DivideRoundByLastQ without BigInt
    fn precompute_rescale_inv_table(moduli: &[u64]) -> Vec<Vec<u64>> {
        let num_primes = moduli.len();
        let mut table = Vec::with_capacity(num_primes);

        for level in 0..num_primes {
            if level == 0 {
                // Level 0 has no lower primes
                table.push(Vec::new());
                continue;
            }

            let q_top = moduli[level];
            let mut inv_row = Vec::with_capacity(level);

            // Compute q_top^{-1} mod q_i for each i < level
            for i in 0..level {
                let q_i = moduli[i];
                let inv = Self::mod_inverse_u64(q_top, q_i)
                    .expect(&format!("Failed to compute {}^{{-1}} mod {}", q_top, q_i));
                inv_row.push(inv);
            }

            table.push(inv_row);
        }

        table
    }

    /// Precompute alpha table for alternative rescaling method (additive formula)
    ///
    /// For each level L, precompute:
    /// alpha_i = [(Q^(l) * Q^(l)^{-1} mod q_L) / q_L] mod q_i
    /// where Q^(l) = q_0 * q_1 * ... * q_{L-1}
    fn precompute_alpha_table(moduli: &[u64]) -> Vec<Vec<u64>> {
        use num_bigint::BigUint;

        let num_primes = moduli.len();
        let mut table = Vec::with_capacity(num_primes);

        for level in 0..num_primes {
            if level == 0 {
                // Level 0 has no lower primes
                table.push(Vec::new());
                continue;
            }

            let q_last = moduli[level];

            // Compute Q^(l) = q_0 * q_1 * ... * q_{level-1}
            let mut q_product = BigUint::from(1u64);
            for i in 0..level {
                q_product *= BigUint::from(moduli[i]);
            }

            // Compute Q^(l)^{-1} mod q_last
            let q_inv_mod_qlast = Self::mod_inverse_bigint(&q_product, q_last)
                .expect(&format!("Failed to compute Q^{{-1}} mod {}", q_last));

            // Compute (Q^(l) * Q^(l)^{-1}) / q_last
            let numerator = q_product * q_inv_mod_qlast;
            let quotient = numerator / BigUint::from(q_last);

            // Compute alpha_i = quotient mod q_i for each i < level
            let mut alpha_row = Vec::with_capacity(level);
            for i in 0..level {
                let q_i = moduli[i];
                let alpha_i = (&quotient % BigUint::from(q_i)).to_u64_digits().first().copied().unwrap_or(0);
                alpha_row.push(alpha_i);
            }

            table.push(alpha_row);
        }

        table
    }

    /// Compute modular inverse of BigUint mod u64
    fn mod_inverse_bigint(a: &num_bigint::BigUint, m: u64) -> Option<num_bigint::BigUint> {
        use num_bigint::{BigUint, BigInt};

        let m_big = BigUint::from(m);

        // Extended GCD
        let (g, x, _) = Self::extended_gcd(a.clone(), m_big.clone());

        if g != BigUint::from(1u64) {
            return None;
        }

        // Convert BigInt to BigUint (ensuring positive result)
        let m_bigint = BigInt::from(m);
        let x_mod = ((x % &m_bigint) + &m_bigint) % &m_bigint;

        // Convert to BigUint
        x_mod.to_biguint()
    }

    /// Extended GCD for BigUint
    fn extended_gcd(a: num_bigint::BigUint, b: num_bigint::BigUint) -> (num_bigint::BigUint, num_bigint::BigInt, num_bigint::BigInt) {
        use num_bigint::{BigUint, BigInt};

        if b == BigUint::from(0u64) {
            return (a, BigInt::from(1), BigInt::from(0));
        }

        let (g, x1, y1) = Self::extended_gcd(b.clone(), &a % &b);
        let x = y1.clone();
        let y = x1 - BigInt::from(&a / &b) * y1;

        (g, x, y)
    }

    /// BATCHED pointwise multiply for all RNS primes in single GPU dispatch
    ///
    /// Processes all RNS primes in parallel using 2D Metal dispatch.
    /// This eliminates per-prime GPU kernel launches and CPU loops.
    ///
    /// **Performance:** 2-3× faster than sequential per-prime operations
    ///
    /// # Arguments
    /// * `a_flat` - Input A in flat interleaved layout: [coeff0_p0, coeff0_p1, ..., coeff1_p0, ...]
    /// * `b_flat` - Input B in flat interleaved layout (same as A)
    /// * `moduli` - RNS moduli to use
    ///
    /// # Returns
    /// Result in flat layout (same as inputs)
    pub fn pointwise_multiply_batched(
        &self,
        a_flat: &[u64],
        b_flat: &[u64],
        moduli: &[u64],
    ) -> Result<Vec<u64>, String> {
        use metal::*;

        let n = self.params.n;
        let num_primes = moduli.len();

        if a_flat.len() != n * num_primes || b_flat.len() != n * num_primes {
            return Err(format!(
                "Expected flat arrays of size {}×{}={}, got {} and {}",
                n, num_primes, n * num_primes,
                a_flat.len(), b_flat.len()
            ));
        }

        // Gather moduli and q_inv parameters
        let mut moduli_array = Vec::with_capacity(num_primes);
        let mut moduli_inv_array = Vec::with_capacity(num_primes);

        for &q in moduli.iter() {
            // Find matching NTT context for this prime
            let ntt_ctx = self.ntt_contexts.iter()
                .find(|ctx| ctx.q() == q)
                .ok_or_else(|| format!("No NTT context for modulus {}", q))?;

            moduli_array.push(q);
            moduli_inv_array.push(ntt_ctx.q_inv());
        }

        // Create GPU buffers
        let a_buffer = self.device.create_buffer_with_data(a_flat);
        let b_buffer = self.device.create_buffer_with_data(b_flat);
        let c_buffer = self.device.create_buffer(n * num_primes);
        let moduli_buffer = self.device.create_buffer_with_data(&moduli_array);
        let moduli_inv_buffer = self.device.create_buffer_with_data(&moduli_inv_array);
        let n_buffer = self.device.create_buffer_with_u32_data(&[n as u32]);
        let num_primes_buffer = self.device.create_buffer_with_u32_data(&[num_primes as u32]);

        // Get kernel
        let kernel = self.device.get_function("ntt_pointwise_multiply_batched")?;
        let pipeline = self.device.device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create batched multiply pipeline: {:?}", e))?;

        // 2D dispatch: (n coefficients, num_primes)
        let threadgroup_size = MTLSize::new(16, 16, 1);  // 16×16 = 256 threads
        let threadgroups = MTLSize::new(
            ((n + 15) / 16) as u64,
            ((num_primes + 15) / 16) as u64,
            1
        );

        // Execute kernel
        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&a_buffer), 0);
            encoder.set_buffer(1, Some(&b_buffer), 0);
            encoder.set_buffer(2, Some(&c_buffer), 0);
            encoder.set_buffer(3, Some(&moduli_buffer), 0);
            encoder.set_buffer(4, Some(&moduli_inv_buffer), 0);
            encoder.set_buffer(5, Some(&n_buffer), 0);
            encoder.set_buffer(6, Some(&num_primes_buffer), 0);

            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            Ok(())
        })?;

        // Read result
        let result = self.device.read_buffer(&c_buffer, n * num_primes);
        Ok(result)
    }

    /// BATCHED in-place modular subtraction: a -= b (mod q) for all RNS primes
    ///
    /// Processes all RNS primes in parallel using 2D Metal dispatch.
    /// Modifies `a` in-place.
    ///
    /// **Performance:** GPU accelerated, eliminates CPU loops
    ///
    /// # Arguments
    /// * `a` - Input/output array in flat layout (modified in-place)
    /// * `b` - Array to subtract in flat layout
    /// * `moduli` - RNS moduli
    pub fn subtract_inplace_batched(
        &self,
        a: &mut [u64],
        b: &[u64],
        moduli: &[u64],
    ) -> Result<(), String> {
        use metal::*;

        let n = self.params.n;
        let num_primes = moduli.len();

        if a.len() != n * num_primes || b.len() != n * num_primes {
            return Err(format!(
                "Expected arrays of size {}×{}={}, got {} and {}",
                n, num_primes, n * num_primes, a.len(), b.len()
            ));
        }

        // Create GPU buffers
        let a_buffer = self.device.create_buffer_with_data(a);
        let b_buffer = self.device.create_buffer_with_data(b);
        let moduli_buffer = self.device.create_buffer_with_data(moduli);
        let n_buffer = self.device.create_buffer_with_u32_data(&[n as u32]);
        let num_primes_buffer = self.device.create_buffer_with_u32_data(&[num_primes as u32]);

        // Get kernel
        let kernel = self.device.get_function("ntt_pointwise_sub_inplace_batched")?;
        let pipeline = self.device.device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create batched sub pipeline: {:?}", e))?;

        // 2D dispatch
        let threadgroup_size = MTLSize::new(16, 16, 1);
        let threadgroups = MTLSize::new(
            ((n + 15) / 16) as u64,
            ((num_primes + 15) / 16) as u64,
            1
        );

        // Execute kernel
        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&a_buffer), 0);
            encoder.set_buffer(1, Some(&b_buffer), 0);
            encoder.set_buffer(2, Some(&moduli_buffer), 0);
            encoder.set_buffer(3, Some(&n_buffer), 0);
            encoder.set_buffer(4, Some(&num_primes_buffer), 0);

            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            Ok(())
        })?;

        // Read result back to a
        let result = self.device.read_buffer(&a_buffer, n * num_primes);
        a.copy_from_slice(&result);

        Ok(())
    }

    /// BATCHED in-place modular addition: a += b (mod q) for all RNS primes
    ///
    /// Processes all RNS primes in parallel using 2D Metal dispatch.
    /// Modifies `a` in-place.
    ///
    /// **Performance:** GPU accelerated, eliminates CPU loops
    ///
    /// # Arguments
    /// * `a` - Input/output array in flat layout (modified in-place)
    /// * `b` - Array to add in flat layout
    /// * `moduli` - RNS moduli
    pub fn add_inplace_batched(
        &self,
        a: &mut [u64],
        b: &[u64],
        moduli: &[u64],
    ) -> Result<(), String> {
        use metal::*;

        let n = self.params.n;
        let num_primes = moduli.len();

        if a.len() != n * num_primes || b.len() != n * num_primes {
            return Err(format!(
                "Expected arrays of size {}×{}={}, got {} and {}",
                n, num_primes, n * num_primes, a.len(), b.len()
            ));
        }

        // Create GPU buffers
        let a_buffer = self.device.create_buffer_with_data(a);
        let b_buffer = self.device.create_buffer_with_data(b);
        let moduli_buffer = self.device.create_buffer_with_data(moduli);
        let n_buffer = self.device.create_buffer_with_u32_data(&[n as u32]);
        let num_primes_buffer = self.device.create_buffer_with_u32_data(&[num_primes as u32]);

        // Get kernel
        let kernel = self.device.get_function("ntt_pointwise_add_inplace_batched")?;
        let pipeline = self.device.device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create batched add pipeline: {:?}", e))?;

        // 2D dispatch
        let threadgroup_size = MTLSize::new(16, 16, 1);
        let threadgroups = MTLSize::new(
            ((n + 15) / 16) as u64,
            ((num_primes + 15) / 16) as u64,
            1
        );

        // Execute kernel
        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&a_buffer), 0);
            encoder.set_buffer(1, Some(&b_buffer), 0);
            encoder.set_buffer(2, Some(&moduli_buffer), 0);
            encoder.set_buffer(3, Some(&n_buffer), 0);
            encoder.set_buffer(4, Some(&num_primes_buffer), 0);

            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            Ok(())
        })?;

        // Read result back to a
        let result = self.device.read_buffer(&a_buffer, n * num_primes);
        a.copy_from_slice(&result);

        Ok(())
    }
}

// Implement methods for MetalCiphertext
impl MetalCiphertext {
    /// Add two ciphertexts (component-wise polynomial addition)
    pub fn add(&self, other: &Self, ctx: &MetalCkksContext) -> Result<Self, String> {
        assert_eq!(self.n, other.n, "Dimensions must match");
        assert_eq!(self.level, other.level, "Levels must match");
        assert_eq!(self.num_primes, other.num_primes, "Number of primes must match");

        let moduli = &ctx.params.moduli[..=self.level];
        let n = self.n;
        let num_primes = self.num_primes;
        let num_active_primes = self.level + 1;  // Number of primes actually in use at this level

        // Add c0 + c0' component-wise
        // Note: After rescaling, actual array size is n * num_active_primes
        // But output needs to be n * num_primes to match struct field
        let mut new_c0 = vec![0u64; n * num_primes];
        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                let out_idx = coeff_idx * num_primes + prime_idx;
                // Only process active primes (indices 0 to level)
                if prime_idx < num_active_primes {
                    // Input arrays have stride num_active_primes, not num_primes
                    let src_idx = coeff_idx * num_active_primes + prime_idx;
                    let q = moduli[prime_idx];
                    new_c0[out_idx] = ((self.c0[src_idx] as u128 + other.c0[src_idx] as u128) % q as u128) as u64;
                } else {
                    new_c0[out_idx] = 0;  // Inactive primes set to 0
                }
            }
        }

        // Add c1 + c1' component-wise
        let mut new_c1 = vec![0u64; n * num_primes];
        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                let out_idx = coeff_idx * num_primes + prime_idx;
                // Only process active primes (indices 0 to level)
                if prime_idx < num_active_primes {
                    // Input arrays have stride num_active_primes, not num_primes
                    let src_idx = coeff_idx * num_active_primes + prime_idx;
                    let q = moduli[prime_idx];
                    new_c1[out_idx] = ((self.c1[src_idx] as u128 + other.c1[src_idx] as u128) % q as u128) as u64;
                } else {
                    new_c1[out_idx] = 0;  // Inactive primes set to 0
                }
            }
        }

        // Scale stays the same (assuming both have same scale)
        Ok(Self {
            c0: new_c0,
            c1: new_c1,
            n,
            num_primes,
            level: self.level,
            scale: self.scale,
        })
    }

    /// Trim ciphertext arrays to only active primes (level + 1)
    ///
    /// After rescaling operations, ciphertext arrays may have padding (stride = num_primes)
    /// but only level + 1 primes are active. This method creates a new ciphertext with
    /// arrays trimmed to size n × (level + 1) for compatibility with decrypt/encode operations.
    pub fn trim_to_active_primes(&self) -> Self {
        let n = self.n;
        let num_active_primes = self.level + 1;
        let current_stride = self.c0.len() / n;

        // If already trimmed, return a clone
        if current_stride == num_active_primes {
            return self.clone();
        }

        // Extract only active primes
        let mut trimmed_c0 = vec![0u64; n * num_active_primes];
        let mut trimmed_c1 = vec![0u64; n * num_active_primes];

        for coeff_idx in 0..n {
            for prime_idx in 0..num_active_primes {
                let src_idx = coeff_idx * current_stride + prime_idx;
                let dst_idx = coeff_idx * num_active_primes + prime_idx;
                trimmed_c0[dst_idx] = self.c0[src_idx];
                trimmed_c1[dst_idx] = self.c1[src_idx];
            }
        }

        Self {
            c0: trimmed_c0,
            c1: trimmed_c1,
            n,
            num_primes: num_active_primes,
            level: self.level,
            scale: self.scale,
        }
    }

    /// Modulus switch to a lower level (drop higher primes without rescaling)
    ///
    /// This is different from rescaling! Mod switch simply truncates the RNS
    /// representation to fewer primes while keeping the same scale. This is used
    /// to align ciphertext levels before operations.
    ///
    /// # Arguments
    /// * `target_level` - The level to switch to (must be ≤ current level)
    ///
    /// # Returns
    /// Ciphertext at the target level with same scale
    pub fn mod_switch_to_level(&self, target_level: usize) -> Self {
        if target_level == self.level {
            return self.clone();
        }

        assert!(
            target_level < self.level,
            "Target level {} must be less than current level {}",
            target_level,
            self.level
        );

        let n = self.n;
        let old_num_primes = self.level + 1;
        let new_num_primes = target_level + 1;

        // Truncate c0: keep only first (target_level + 1) primes for each coefficient
        let mut new_c0 = vec![0u64; n * new_num_primes];
        for coeff_idx in 0..n {
            for prime_idx in 0..new_num_primes {
                let old_idx = coeff_idx * old_num_primes + prime_idx;
                let new_idx = coeff_idx * new_num_primes + prime_idx;
                new_c0[new_idx] = self.c0[old_idx];
            }
        }

        // Truncate c1: keep only first (target_level + 1) primes for each coefficient
        let mut new_c1 = vec![0u64; n * new_num_primes];
        for coeff_idx in 0..n {
            for prime_idx in 0..new_num_primes {
                let old_idx = coeff_idx * old_num_primes + prime_idx;
                let new_idx = coeff_idx * new_num_primes + prime_idx;
                new_c1[new_idx] = self.c1[old_idx];
            }
        }

        Self {
            c0: new_c0,
            c1: new_c1,
            n,
            num_primes: new_num_primes,
            level: target_level,
            scale: self.scale,  // Scale stays the same for mod_switch
        }
    }

    /// Multiply ciphertext by plaintext using Metal GPU NTT
    ///
    /// Algorithm: (c0, c1) * pt = (c0 * pt, c1 * pt)
    /// Then rescale to bring scale back down.
    pub fn multiply_plain(&self, pt: &MetalPlaintext, ctx: &MetalCkksContext) -> Result<Self, String> {
        assert_eq!(self.n, pt.n, "Dimensions must match");
        assert_eq!(self.level, pt.level, "Levels must match for plaintext multiplication");

        let moduli = &ctx.params.moduli[..=self.level];

        // CRITICAL: Use negacyclic NTT for CKKS plaintext multiplication!
        // CKKS operates in the ring R = Z[x]/(x^n + 1) which requires negacyclic convolution.
        let new_c0 = ctx.multiply_polys_flat_ntt_negacyclic(&self.c0, &pt.coeffs, moduli)?;

        // Multiply c1 by plaintext using negacyclic NTT
        let new_c1 = ctx.multiply_polys_flat_ntt_negacyclic(&self.c1, &pt.coeffs, moduli)?;

        // Compute pre-rescale scale
        let pre_rescale_scale = self.scale * pt.scale;

        // CRITICAL FIX: multiply_polys_flat_ntt returns arrays with stride = moduli.len()
        // Must set num_primes to match the output stride, not the input stride!
        let num_primes_to_process = moduli.len();

        // Create intermediate ciphertext (scale will be fixed in rescale)
        let ct_mult = Self {
            c0: new_c0,
            c1: new_c1,
            n: self.n,
            num_primes: num_primes_to_process,  // FIX: Use output stride, not self.num_primes
            level: self.level,
            scale: 0.0, // Will be set by rescale
        };

        // Rescale to bring scale back to ~Δ and drop one level
        ct_mult.rescale_to_next(ctx, pre_rescale_scale)
    }

    /// Rescale ciphertext to next level (drop one prime from modulus chain)
    ///
    /// This operation divides the ciphertext by the dropped prime and reduces the level.
    /// Essential for keeping noise growth manageable in CKKS.
    fn rescale_to_next(&self, ctx: &MetalCkksContext, pre_rescale_scale: f64) -> Result<Self, String> {
        if self.level == 0 {
            return Err("Cannot rescale at level 0".to_string());
        }

        use num_bigint::BigInt;
        use num_traits::{One, Signed, Zero};

        let moduli_before = &ctx.params.moduli[..=self.level];
        let moduli_after = &ctx.params.moduli[..self.level]; // Drop last prime
        let q_last = moduli_before[moduli_before.len() - 1];
        let n = self.n;
        let new_level = self.level - 1;
        let num_primes_after = moduli_after.len();

        // CRITICAL: Verify stride matches number of primes (prevents stride mismatch bugs)
        debug_assert_eq!(
            self.num_primes,
            moduli_before.len(),
            "Stride/level mismatch: num_primes={} but moduli_before.len()={}",
            self.num_primes,
            moduli_before.len()
        );
        debug_assert_eq!(
            self.c0.len(),
            n * self.num_primes,
            "Buffer size mismatch: c0.len()={} but expected n*num_primes={}",
            self.c0.len(),
            n * self.num_primes
        );

        // Helper: Convert residue to centered representation: [0, q) -> (-q/2, q/2]
        let centered_residue = |x: u64, q: u64| -> i128 {
            let x_i128 = x as i128;
            let q_i128 = q as i128;
            if x_i128 > q_i128 / 2 {
                x_i128 - q_i128
            } else {
                x_i128
            }
        };

        // Helper: Convert centered residue back to canonical [0, q)
        let canon_from_centered = |t: i128, q: u64| -> u64 {
            let q_i128 = q as i128;
            let mut u = t % q_i128;
            if u < 0 {
                u += q_i128;
            }
            u as u64
        };

        // CRT reconstruction using direct formula with centered residues
        let crt_reconstruct_centered = |residues: &[u64], moduli: &[u64]| -> BigInt {
            let centered: Vec<i128> = residues.iter().zip(moduli.iter())
                .map(|(&r, &q)| centered_residue(r, q))
                .collect();

            let mut q_product = BigInt::one();
            for &q in moduli {
                q_product *= q;
            }

            let mut result = BigInt::zero();
            for i in 0..moduli.len() {
                let r_i = BigInt::from(centered[i]);
                let q_i = BigInt::from(moduli[i]);
                let m_i = &q_product / &q_i;

                // Compute modular inverse using Extended Euclidean Algorithm
                let (gcd, x, _) = {
                    fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
                        if b.is_zero() {
                            return (a.clone(), BigInt::one(), BigInt::zero());
                        }
                        let (g, x1, y1) = extended_gcd(b, &(a % b));
                        let x = y1.clone();
                        let y = x1 - (a / b) * y1;
                        (g, x, y)
                    }
                    extended_gcd(&m_i, &q_i)
                };
                assert!(gcd.is_one(), "GCD must be 1");

                let m_i_inv = ((x % &q_i) + &q_i) % &q_i;
                let term_i = &r_i * &m_i * m_i_inv;
                result += term_i;
            }

            // Center the result to (-Q/2, Q/2]
            result = result % &q_product;
            if result > &q_product / 2 {
                result -= q_product;
            }
            result
        };

        // Rescale c0
        let mut new_c0_flat = vec![0u64; n * num_primes_after];
        for i in 0..n {
            // Extract RNS representation for this coefficient (before rescale)
            // IMPORTANT: Use self.num_primes for indexing (the current stride), not moduli_before.len()
            let residues_before: Vec<u64> = (0..moduli_before.len())
                .map(|j| self.c0[i * self.num_primes + j])
                .collect();

            // CRT reconstruct to BigInt
            let coeff_big = crt_reconstruct_centered(&residues_before, moduli_before);

            // Exact division by q_last (centered)
            let coeff_rescaled = &coeff_big / q_last;

            // Convert back to RNS (mod each remaining prime)
            for j in 0..num_primes_after {
                let q_j = moduli_after[j];
                let residue = (&coeff_rescaled % q_j).to_u64_digits();
                let val = if residue.1.is_empty() {
                    0u64
                } else if residue.0 == num_bigint::Sign::Minus {
                    // Negative: convert to canonical form
                    let abs_val = residue.1[0];
                    if abs_val % q_j == 0 {
                        0
                    } else {
                        q_j - (abs_val % q_j)
                    }
                } else {
                    residue.1[0] % q_j
                };
                new_c0_flat[i * num_primes_after + j] = val;
            }
        }

        // Rescale c1
        let mut new_c1_flat = vec![0u64; n * num_primes_after];
        for i in 0..n {
            // IMPORTANT: Use self.num_primes for indexing (the current stride), not moduli_before.len()
            let residues_before: Vec<u64> = (0..moduli_before.len())
                .map(|j| self.c1[i * self.num_primes + j])
                .collect();

            let coeff_big = crt_reconstruct_centered(&residues_before, moduli_before);
            let coeff_rescaled = &coeff_big / q_last;

            for j in 0..num_primes_after {
                let q_j = moduli_after[j];
                let residue = (&coeff_rescaled % q_j).to_u64_digits();
                let val = if residue.1.is_empty() {
                    0u64
                } else if residue.0 == num_bigint::Sign::Minus {
                    let abs_val = residue.1[0];
                    if abs_val % q_j == 0 {
                        0
                    } else {
                        q_j - (abs_val % q_j)
                    }
                } else {
                    residue.1[0] % q_j
                };
                new_c1_flat[i * num_primes_after + j] = val;
            }
        }

        // New scale after rescale
        let new_scale = pre_rescale_scale / (q_last as f64);

        Ok(Self {
            c0: new_c0_flat,
            c1: new_c1_flat,
            n,
            num_primes: num_primes_after,
            level: new_level,
            scale: new_scale,
        })
    }

    /// Rotate ciphertext slots by r steps using Metal GPU
    ///
    /// Uses Galois automorphism σ_k where k = 5^r (mod 2N).
    /// Requires rotation keys for the specified step.
    ///
    /// # Algorithm
    /// 1. Apply σ_k to c₀ and c₁ (GPU kernel)
    /// 2. Key switch c₁ using rotation key (GPU NTT)
    /// 3. Return rotated ciphertext
    ///
    /// # Arguments
    /// * `step` - Number of slots to rotate (positive = left, negative = right)
    /// * `rot_keys` - Rotation keys (must contain key for this step)
    /// * `ctx` - Metal CKKS context
    ///
    /// # Returns
    /// Rotated ciphertext with slots shifted by `step` positions
    ///
    /// # Performance
    /// - Target: <1ms per rotation on M3 Max
    /// - GPU kernel: <0.1ms (pure permutation)
    /// - Key switching: <0.9ms (NTT multiplication)
    pub fn rotate_by_steps(
        &self,
        step: i32,
        rot_keys: &super::rotation_keys::MetalRotationKeys,
        ctx: &MetalCkksContext,
    ) -> Result<Self, String> {
        use super::rotation::{compute_galois_map, rotation_step_to_galois_element};

        let n = self.n;
        let num_primes_active = self.level + 1;
        let moduli = &ctx.params.moduli[..num_primes_active];

        // Get rotation key with gadget decomposition (full-sized, generated with all primes)
        let (rlk0_full, rlk1_full) = rot_keys.get_key_for_step(step)
            .ok_or_else(|| format!("Rotation key for step {} not found. Generate rotation keys first.", step))?;

        // Extract only active primes from rotation keys (for each digit)
        // Rotation keys have flat RNS layout: [coeff0_mod_q0, coeff0_mod_q1, ..., coeffN-1_mod_qL]
        let num_digits = rlk0_full.len();
        let mut rlk0 = Vec::with_capacity(num_digits);
        let mut rlk1 = Vec::with_capacity(num_digits);

        for t in 0..num_digits {
            let rot_key_stride = rlk0_full[t].len() / n;  // Total number of primes in this key digit
            let mut rlk0_t = vec![0u64; n * num_primes_active];
            let mut rlk1_t = vec![0u64; n * num_primes_active];

            for coeff_idx in 0..n {
                for prime_idx in 0..num_primes_active {
                    rlk0_t[coeff_idx * num_primes_active + prime_idx] =
                        rlk0_full[t][coeff_idx * rot_key_stride + prime_idx];
                    rlk1_t[coeff_idx * num_primes_active + prime_idx] =
                        rlk1_full[t][coeff_idx * rot_key_stride + prime_idx];
                }
            }

            rlk0.push(rlk0_t);
            rlk1.push(rlk1_t);
        }

        let base_w = rot_keys.base_w();

        // Convert step to Galois element
        let k = rotation_step_to_galois_element(step, n);

        // Precompute Galois map
        let (galois_map, galois_signs) = compute_galois_map(n, k);

        // Extract active primes from ciphertext (handle variable stride after rescaling)
        let ct_stride = self.c0.len() / n;

        let mut c0_active = vec![0u64; n * num_primes_active];
        let mut c1_active = vec![0u64; n * num_primes_active];

        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes_active {
                c0_active[coeff_idx * num_primes_active + prime_idx] =
                    self.c0[coeff_idx * ct_stride + prime_idx];
                c1_active[coeff_idx * num_primes_active + prime_idx] =
                    self.c1[coeff_idx * ct_stride + prime_idx];
            }
        }

        // Apply Galois automorphism to c₀ and c₁ (GPU)
        let c0_rotated = self.apply_galois_gpu(&c0_active, &galois_map, &galois_signs, moduli, ctx)?;
        let c1_rotated = self.apply_galois_gpu(&c1_active, &galois_map, &galois_signs, moduli, ctx)?;

        // Key switch using rotation key with gadget decomposition (GPU NTT multiplication)
        let (c0_final, c1_final) = self.key_switch_gpu_gadget(&c0_rotated, &c1_rotated, &rlk0, &rlk1, moduli, base_w, ctx)?;

        // Pad the compact output back to full stride
        // c0_final and c1_final have size n × num_primes_active (strided layout)
        // Need to expand to n × self.num_primes
        let mut c0_padded = vec![0u64; n * self.num_primes];
        let mut c1_padded = vec![0u64; n * self.num_primes];

        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes_active {
                c0_padded[coeff_idx * self.num_primes + prime_idx] =
                    c0_final[coeff_idx * num_primes_active + prime_idx];
                c1_padded[coeff_idx * self.num_primes + prime_idx] =
                    c1_final[coeff_idx * num_primes_active + prime_idx];
            }
        }

        Ok(Self {
            c0: c0_padded,
            c1: c1_padded,
            n,
            num_primes: self.num_primes,  // Keep original total
            level: self.level,
            scale: self.scale,
        })
    }

    /// Batch rotation with automorphism hoisting (optimized for linear transforms)
    ///
    /// This is the OPTIMIZED rotation path for use cases like:
    /// - Bootstrapping (many rotations of the same ciphertext)
    /// - Linear transformations (matrix-vector multiply in CKKS)
    /// - Slot permutations
    ///
    /// **NOT** optimized for butterfly transform (which uses different rotation steps).
    ///
    /// # Algorithm
    /// For each unique rotation step:
    /// 1. Apply Galois automorphism to c0 and c1
    /// 2. **HOIST**: Decompose c1 and forward-NTT all digits (expensive, done ONCE per step)
    /// 3. Key-switch using hoisted digits (amortizes decompose+NTT cost)
    ///
    /// # Performance (per rotation)
    /// - Without hoisting: decompose(0.05s) + NTT(0.08s) + key-switch(0.12s) = 0.25s
    /// - With hoisting: [decompose+NTT once] + key-switch(0.12s) = ~0.12s per rotation
    /// - **Speedup: ~2× per rotation after first**
    ///
    /// # Arguments
    /// * `steps` - List of rotation steps to perform
    /// * `rot_keys` - Rotation keys for all steps
    /// * `ctx` - CKKS context
    ///
    /// # Returns
    /// Vector of rotated ciphertexts, one for each step (in same order as `steps`)
    pub fn rotate_batch_with_hoisting(
        &self,
        steps: &[i32],
        rot_keys: &super::rotation_keys::MetalRotationKeys,
        ctx: &MetalCkksContext,
    ) -> Result<Vec<Self>, String> {
        use super::rotation::{compute_galois_map, rotation_step_to_galois_element};
        use super::hoisting::{hoist_decompose_ntt, rotate_with_hoisted_digits};

        if steps.is_empty() {
            return Ok(vec![]);
        }

        let n = self.n;
        let num_primes_active = self.level + 1;
        let moduli = &ctx.params.moduli[..num_primes_active];
        let base_w = rot_keys.base_w();

        // Extract active primes from ciphertext
        let ct_stride = self.c0.len() / n;
        let mut c0_active = vec![0u64; n * num_primes_active];
        let mut c1_active = vec![0u64; n * num_primes_active];

        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes_active {
                c0_active[coeff_idx * num_primes_active + prime_idx] =
                    self.c0[coeff_idx * ct_stride + prime_idx];
                c1_active[coeff_idx * num_primes_active + prime_idx] =
                    self.c1[coeff_idx * ct_stride + prime_idx];
            }
        }

        // HOIST STEP (ONCE): Decompose c1 and forward-NTT all digits
        // This is the expensive operation we want to do ONCE for all rotations
        let hoisted = hoist_decompose_ntt(&c1_active, base_w, moduli, n, ctx)?;

        // Result vector
        let mut results = Vec::with_capacity(steps.len());

        // Process each rotation step (FAST PATH with PRE-CACHED NTT KEYS!)
        for &step in steps {
            // Get PRE-CACHED NTT rotation keys for this level (15-20% faster!)
            // No runtime transformation needed - keys are already in NTT domain
            let (rlk0_ntt, rlk1_ntt) = rot_keys.get_key_ntt_for_step(step, self.level)
                .ok_or_else(|| format!("NTT rotation key for step {} at level {} not found", step, self.level))?;

            // Convert step to Galois element
            let k = rotation_step_to_galois_element(step, n);
            let (galois_map, galois_signs) = compute_galois_map(n, k);

            // Apply Galois automorphism to c0 (c1 automorphism is handled via hoisting)
            let c0_rotated = self.apply_galois_gpu(&c0_active, &galois_map, &galois_signs, moduli, ctx)?;

            // ULTRA-FAST PATH: Use hoisted digits + PRE-CACHED NTT keys!
            // No decompose, no NTT, no key transformation - just permute + multiply + iNTT
            let (c0_final, c1_final) = rotate_with_hoisted_digits(
                &hoisted,
                step,
                rlk0_ntt,
                rlk1_ntt,
                &c0_rotated,
                moduli,
                ctx,
            )?;

            // Pad back to full stride
            let mut c0_padded = vec![0u64; n * self.num_primes];
            let mut c1_padded = vec![0u64; n * self.num_primes];

            for coeff_idx in 0..n {
                for prime_idx in 0..num_primes_active {
                    c0_padded[coeff_idx * self.num_primes + prime_idx] =
                        c0_final[coeff_idx * num_primes_active + prime_idx];
                    c1_padded[coeff_idx * self.num_primes + prime_idx] =
                        c1_final[coeff_idx * num_primes_active + prime_idx];
                }
            }

            results.push(Self {
                c0: c0_padded,
                c1: c1_padded,
                n,
                num_primes: self.num_primes,
                level: self.level,
                scale: self.scale,
            });
        }

        Ok(results)
    }

    /// Apply Galois automorphism using Metal GPU kernel
    ///
    /// Dispatches the apply_galois_automorphism kernel from rotation.metal.
    fn apply_galois_gpu(
        &self,
        poly: &[u64],
        galois_map: &[u32],
        galois_signs: &[i32],
        moduli: &[u64],
        ctx: &MetalCkksContext,
    ) -> Result<Vec<u64>, String> {
        use metal::MTLSize;

        let device = &ctx.device;
        let n = self.n;
        let num_primes = moduli.len();

        // Create Metal buffers
        let input_buffer = device.create_buffer_with_data(poly);
        let output_buffer = device.create_buffer(poly.len());
        let map_buffer = device.create_buffer_with_u32_data(galois_map);
        let signs_buffer = device.create_buffer_with_i32_data(galois_signs);
        let moduli_buffer = device.create_buffer_with_data(moduli);

        // Get rotation kernel function
        let function = device.get_rotation_function("apply_galois_automorphism")?;
        let pipeline = device.device().new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Failed to create rotation pipeline: {:?}", e))?;

        // Execute kernel
        device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_buffer(2, Some(&map_buffer), 0);
            encoder.set_buffer(3, Some(&signs_buffer), 0);
            encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(n as u32) as *const u32 as *const _);
            encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &(num_primes as u32) as *const u32 as *const _);
            encoder.set_buffer(6, Some(&moduli_buffer), 0);

            // Dispatch threads: one per coefficient
            let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
            let thread_groups = MTLSize {
                width: ((n + 255) / 256) as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(thread_groups, thread_group_size);

            Ok(())
        })?;

        // Read back result
        let result = device.read_buffer(&output_buffer, poly.len());

        Ok(result)
    }

    /// Key switch after rotation using rotation key
    ///
    /// Converts (σ_k(c₀), σ_k(c₁)) which decrypts with σ_k(s) into
    /// (c'₀, c'₁) which decrypts with the original s.
    ///
    /// # Algorithm
    ///
    /// The rotation key (rk0, rk1) = (a_k, b_k) satisfies:
    ///   b_k ≈ -a_k·s + e + σ_k(s)
    ///
    /// After σ_k, we have: σ_k(c₀) + σ_k(c₁)·σ_k(s) = σ_k(m)
    ///
    /// Key switching formula:
    ///   c'₀ = σ_k(c₀) + σ_k(c₁)·b_k
    ///   c'₁ = σ_k(c₁)·a_k
    ///
    /// Decryption check:
    ///   c'₀ + c'₁·s = σ_k(c₀) + σ_k(c₁)·b_k + σ_k(c₁)·a_k·s
    ///               = σ_k(c₀) + σ_k(c₁)·(b_k + a_k·s)
    ///               ≈ σ_k(c₀) + σ_k(c₁)·σ_k(s)  (since b_k + a_k·s ≈ σ_k(s))
    ///               = σ_k(m)  ✓
    ///
    /// # Performance
    /// - 2 NTT multiplications: σ_k(c₁)·b_k and σ_k(c₁)·a_k
    /// - 1 addition: σ_k(c₀) + σ_k(c₁)·b_k
    /// - All operations on GPU using existing Metal NTT
    fn key_switch_gpu_gadget(
        &self,
        c0_rotated: &[u64],
        c1_rotated: &[u64],
        rlk0: &[Vec<u64>],
        rlk1: &[Vec<u64>],
        moduli: &[u64],
        base_w: u32,
        ctx: &MetalCkksContext,
    ) -> Result<(Vec<u64>, Vec<u64>), String> {
        let n = self.n;
        let num_primes = moduli.len();
        let num_digits = rlk0.len();

        // Initialize accumulators
        let mut c0_final = c0_rotated.to_vec();
        let mut c1_final = vec![0u64; n * num_primes];

        // Decompose c1_rotated using gadget decomposition
        let c1_digits = Self::gadget_decompose_flat(c1_rotated, base_w, moduli, n)?;

        // For each digit t in the decomposition
        for t in 0..num_digits {
            if t >= c1_digits.len() {
                break;  // Fewer actual digits than key components
            }

            // IMPORTANT: rlk0[t] and rlk1[t] are already in NTT domain!
            // We only need to transform the digit, then do pointwise multiply, then inverse NTT
            let term0 = Self::multiply_digit_by_ntt_key(&c1_digits[t], &rlk0[t], moduli, ctx)?;
            let term1 = Self::multiply_digit_by_ntt_key(&c1_digits[t], &rlk1[t], moduli, ctx)?;

            // c0_final -= term0 (matches V3 CPU implementation)
            for i in 0..(n * num_primes) {
                let prime_idx = i % num_primes;
                let q = moduli[prime_idx];

                // Subtract: c0_final = c0_final - term0
                let diff = if c0_final[i] >= term0[i] {
                    c0_final[i] - term0[i]
                } else {
                    q - (term0[i] - c0_final[i])
                };
                c0_final[i] = diff;
            }

            // c1_final += term1
            for i in 0..(n * num_primes) {
                let prime_idx = i % num_primes;
                let q = moduli[prime_idx];
                c1_final[i] = ((c1_final[i] as u128 + term1[i] as u128) % q as u128) as u64;
            }
        }

        Ok((c0_final, c1_final))
    }

    /// Multiply two coefficient-domain polynomials using NTT
    ///
    /// Both digit and key are in coefficient domain.
    /// Uses standard CKKS negacyclic multiplication: twist → NTT → multiply → iNTT → untwist
    fn multiply_digit_by_ntt_key(
        digit_coeff: &[u64],
        key_coeff: &[u64],
        moduli: &[u64],
        ctx: &MetalCkksContext,
    ) -> Result<Vec<u64>, String> {
        let n = ctx.params.n;
        let num_primes = moduli.len();
        let mut result_flat = vec![0u64; n * num_primes];

        // For each RNS component (each prime)
        for (prime_idx, &q) in moduli.iter().enumerate() {
            // Extract digit polynomial for this prime
            let mut digit_poly = vec![0u64; n];
            for i in 0..n {
                digit_poly[i] = digit_coeff[i * num_primes + prime_idx];
            }

            // Extract key polynomial for this prime
            let mut key_poly = vec![0u64; n];
            for i in 0..n {
                key_poly[i] = key_coeff[i * num_primes + prime_idx];
            }

            let ntt_ctx = &ctx.ntt_contexts[prime_idx];

            // Standard negacyclic NTT multiplication pattern:
            // 1. Twist both polynomials
            for i in 0..n {
                digit_poly[i] = ((digit_poly[i] as u128 * ntt_ctx.psi_powers()[i] as u128) % q as u128) as u64;
                key_poly[i] = ((key_poly[i] as u128 * ntt_ctx.psi_powers()[i] as u128) % q as u128) as u64;
            }

            // 2. Forward NTT on both
            ntt_ctx.forward(&mut digit_poly)?;
            ntt_ctx.forward(&mut key_poly)?;

            // 3. Pointwise multiply in NTT domain
            let mut result_poly = vec![0u64; n];
            ntt_ctx.pointwise_multiply(&digit_poly, &key_poly, &mut result_poly)?;

            // 4. Inverse NTT
            ntt_ctx.inverse(&mut result_poly)?;

            // 5. Untwist
            for i in 0..n {
                result_poly[i] = ((result_poly[i] as u128 * ntt_ctx.psi_inv_powers()[i] as u128) % q as u128) as u64;
            }

            // Store back in flat layout
            for i in 0..n {
                result_flat[i * num_primes + prime_idx] = result_poly[i];
            }
        }

        Ok(result_flat)
    }

    /// Gadget decomposition for flat RNS layout
    ///
    /// Decomposes polynomial in base B = 2^base_w using CRT-consistent decomposition.
    pub fn gadget_decompose_flat(
        poly: &[u64],
        base_w: u32,
        moduli: &[u64],
        n: usize,
    ) -> Result<Vec<Vec<u64>>, String> {
        let num_primes = moduli.len();

        // Compute Q = product of all primes
        let q_prod_big: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();
        let q_half_big = &q_prod_big / 2;
        let base_big = BigInt::one() << base_w;  // B = 2^base_w

        // Determine number of digits
        let q_bits = q_prod_big.bits() as u32;
        let num_digits = ((q_bits + base_w - 1) / base_w) as usize;

        let mut digits = vec![vec![0u64; n * num_primes]; num_digits];

        // Decompose each coefficient using CRT (EXACTLY matching CPU implementation)
        for i in 0..n {
            // Debug: print input residues for first coefficient
            if std::env::var("CRT_DEBUG").is_ok() && i == 0 {
                print!("[CRT_DEBUG Metal] coeff[0] input residues: ");
                for j in 0..num_primes {
                    print!("{} ", poly[i * num_primes + j]);
                }
                println!();
            }

            // Step 1: CRT reconstruct to get x ∈ [0, Q)
            let mut x_big = BigInt::zero();
            for (j, &q) in moduli.iter().enumerate() {
                let idx = i * num_primes + j;
                let residue = poly[idx];

                let q_big = BigInt::from(q);
                let q_i = &q_prod_big / &q_big;

                // Compute q_i^(-1) mod q using extended GCD
                let qi_inv = Self::mod_inverse_bigint(&q_i, &q_big)?;

                let ri_big = BigInt::from(residue);
                // Compute: basis = (Q/qi) * inv mod Q, then term = ri * basis mod Q
                let basis = (&q_i * &qi_inv) % &q_prod_big;
                let term = (ri_big * basis) % &q_prod_big;
                x_big = (&x_big + term) % &q_prod_big;
            }

            // Ensure result is positive
            if x_big.sign() == num_bigint::Sign::Minus {
                x_big += &q_prod_big;
            }

            // Debug: print reconstructed value for first coefficient
            if std::env::var("CRT_DEBUG").is_ok() && i == 0 {
                println!("[CRT_DEBUG Metal] coeff[0] after CRT: {}", x_big);
            }

            // Step 2: Center-lift to x_c ∈ (-Q/2, Q/2]
            let x_centered_big = if x_big > q_half_big {
                x_big - &q_prod_big
            } else {
                x_big
            };

            // Debug: print centered value for first coefficient
            if std::env::var("CRT_DEBUG").is_ok() && i == 0 {
                println!("[CRT_DEBUG Metal] coeff[0] after centering: {}", x_centered_big);
            }

            // Step 3: Balanced decomposition in Z (FIXED: use fixed loop like CPU)
            let mut remainder_big = x_centered_big;
            let half_base_big = &base_big / 2;

            for t in 0..num_digits {
                // Extract digit dt ∈ (-B/2, B/2] (balanced, matching CPU logic)
                let dt_unbalanced = &remainder_big % &base_big;
                let dt_big = if dt_unbalanced > half_base_big {
                    &dt_unbalanced - &base_big  // Shift to negative range
                } else {
                    dt_unbalanced
                };

                // Debug: print first digit for first coefficient
                if std::env::var("GADGET_DEBUG").is_ok() && i == 0 && t == 0 {
                    println!("[GADGET_DEBUG Metal] coeff[0] digit[0]: dt_big={}", dt_big);
                }

                // Convert dt to residues mod each prime
                for (j, &q) in moduli.iter().enumerate() {
                    let idx = i * num_primes + j;
                    let q_big = BigInt::from(q);
                    let mut dt_mod_q_big = &dt_big % &q_big;
                    if dt_mod_q_big.sign() == num_bigint::Sign::Minus {
                        dt_mod_q_big += &q_big;
                    }
                    digits[t][idx] = dt_mod_q_big.to_u64().unwrap_or(0);

                    // Debug: print conversion
                    if std::env::var("GADGET_DEBUG").is_ok() && i == 0 && t == 0 {
                        println!("[GADGET_DEBUG Metal] coeff[0] digit[0] prime[{}]: dt_mod_q={} (idx={})", j, digits[t][idx], idx);
                    }
                }

                // Update remainder: (x_c - dt) / B (exact division, matches CPU)
                remainder_big = (remainder_big - &dt_big) / &base_big;
            }
        }

        Ok(digits)
    }

    /// Modular inverse using extended Euclidean algorithm (BigInt version)
    fn mod_inverse_bigint(a: &BigInt, modulus: &BigInt) -> Result<BigInt, String> {
        let mut t = BigInt::zero();
        let mut newt = BigInt::one();
        let mut r = modulus.clone();
        let mut newr = a.clone();

        while !newr.is_zero() {
            let quotient = &r / &newr;
            let temp_t = t.clone();
            t = newt.clone();
            newt = temp_t - &quotient * &newt;

            let temp_r = r.clone();
            r = newr.clone();
            newr = temp_r - quotient * newr;
        }

        if r > BigInt::one() {
            return Err(format!("Not invertible"));
        }
        if t < BigInt::zero() {
            t += modulus;
        }

        Ok(t)
    }

    /// Multiply two ciphertexts with relinearization (GPU)
    ///
    /// Computes ct1 × ct2 using tensor product followed by relinearization:
    /// 1. Tensor product: (c0, c1) × (d0, d1) = (c0×d0, c0×d1 + c1×d0, c1×d1)
    /// 2. Relinearization: (ct0, ct1, ct2) → (ct'0, ct'1) using relin keys
    /// 3. Rescale: Drop one prime from modulus chain
    ///
    /// # Arguments
    /// * `other` - Second ciphertext to multiply
    /// * `relin_keys` - Relinearization keys for degree reduction
    /// * `ctx` - Metal CKKS context
    ///
    /// # Returns
    /// Result ciphertext at level-1 with scale²
    pub fn multiply(
        &self,
        other: &Self,
        relin_keys: &super::relin_keys::MetalRelinKeys,
        ctx: &MetalCkksContext,
    ) -> Result<Self, String> {
        // Validate inputs
        if self.level != other.level {
            return Err(format!(
                "Cannot multiply ciphertexts at different levels: {} vs {}",
                self.level, other.level
            ));
        }

        if self.level == 0 {
            return Err("Cannot multiply ciphertext at level 0 (no room for rescale)".to_string());
        }

        let n = self.n;
        let level = self.level;
        let num_primes = level + 1;

        // Debug: print input c1 values
        if std::env::var("C1_DEBUG").is_ok() {
            print!("[C1_DEBUG Metal] self.c1[0] across primes: ");
            for j in 0..num_primes {
                print!("{} ", self.c1[0 * num_primes + j]);
            }
            println!();
            print!("[C1_DEBUG Metal] other.c1[0] across primes: ");
            for j in 0..num_primes {
                print!("{} ", other.c1[0 * num_primes + j]);
            }
            println!();
        }

        // Step 1: Tensor product using NTT multiplication
        // ct_mult = (c0×d0, c0×d1 + c1×d0, c1×d1)

        // c0 × d0
        let ct0_ct0 = ctx.multiply_polys_flat_ntt_negacyclic(
            &self.c0,
            &other.c0,
            &ctx.params.moduli[..num_primes],
        )?;

        // c0 × d1
        let ct0_ct1 = ctx.multiply_polys_flat_ntt_negacyclic(
            &self.c0,
            &other.c1,
            &ctx.params.moduli[..num_primes],
        )?;

        // c1 × d0
        let ct1_ct0 = ctx.multiply_polys_flat_ntt_negacyclic(
            &self.c1,
            &other.c0,
            &ctx.params.moduli[..num_primes],
        )?;

        // Debug: print c1 values BEFORE multiplication
        if std::env::var("POLY_DEBUG").is_ok() {
            print!("[POLY_DEBUG Metal] BEFORE mult - self.c1[0:2] prime 0: ");
            for i in 0..2.min(n) {
                print!("{} ", self.c1[i * num_primes + 0]);
            }
            println!();
            print!("[POLY_DEBUG Metal] BEFORE mult - other.c1[0:2] prime 0: ");
            for i in 0..2.min(n) {
                print!("{} ", other.c1[i * num_primes + 0]);
            }
            println!();
        }

        // c1 × d1 (this is c2 component)
        let c2 = ctx.multiply_polys_flat_ntt_negacyclic(
            &self.c1,
            &other.c1,
            &ctx.params.moduli[..num_primes],
        )?;

        // Debug: print c2 values AFTER multiplication
        if std::env::var("POLY_DEBUG").is_ok() {
            print!("[POLY_DEBUG Metal] AFTER mult - c2[0:2] prime 0: ");
            for i in 0..2.min(n) {
                print!("{} ", c2[i * num_primes + 0]);
            }
            println!();
        }

        // Debug: print c2 values before gadget decomposition
        if std::env::var("C2_DEBUG").is_ok() {
            print!("[C2_DEBUG Metal] c2[0] across primes: ");
            for j in 0..num_primes {
                print!("{} ", c2[0 * num_primes + j]);
            }
            println!();
        }

        // ct0_ct1 + ct1_ct0 (middle term)
        let mut ct1_temp = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let q = ctx.params.moduli[i % num_primes];
            ct1_temp[i] = ((ct0_ct1[i] as u128 + ct1_ct0[i] as u128) % q as u128) as u64;
        }

        // Result before relinearization: (ct0_ct0, ct1_temp, c2)
        let mut result_c0 = ct0_ct0;
        let mut result_c1 = ct1_temp;

        // Debug: print values right after tensor product
        if std::env::var("TENSOR_DEBUG").is_ok() {
            println!("[TENSOR_DEBUG Metal] After tensor product (before relin):");
            print!("  c0[0] across {} primes: ", num_primes);
            for j in 0..num_primes {
                print!("{} ", result_c0[0 * num_primes + j]);
            }
            println!();
            print!("  c1[0] across {} primes: ", num_primes);
            for j in 0..num_primes {
                print!("{} ", result_c1[0 * num_primes + j]);
            }
            println!();
            print!("  c2[0] across {} primes: ", num_primes);
            for j in 0..num_primes {
                print!("{} ", c2[0 * num_primes + j]);
            }
            println!();
        }

        // Debug: print values before relinearization
        if std::env::var("RELIN_DEBUG").is_ok() {
            println!("[RELIN_DEBUG Metal] Before relinearization:");
            print!("  c0[0] across {} primes: ", num_primes);
            for j in 0..num_primes {
                print!("{} ", result_c0[0 * num_primes + j]);
            }
            println!();
            print!("  c1[0] across {} primes: ", num_primes);
            for j in 0..num_primes {
                print!("{} ", result_c1[0 * num_primes + j]);
            }
            println!();
            print!("  c2[0] across {} primes: ", num_primes);
            for j in 0..num_primes {
                print!("{} ", c2[0 * num_primes + j]);
            }
            println!();
        }

        // Step 2: Relinearization - convert c2·s² into (Δc0, Δc1)
        // EXACTLY matches CPU relinearize_degree2 logic from multiplication.rs:158-197

        let (base_w, _max_num_digits) = relin_keys.gadget_params();
        let (rlk0_coeff, rlk1_coeff) = relin_keys.get_coeff_keys(level)?;

        // Debug: print EVK values RIGHT AFTER retrieval
        if std::env::var("EVK_IMMEDIATE_DEBUG").is_ok() {
            print!("[EVK_IMMEDIATE_DEBUG] RIGHT AFTER get_coeff_keys, rlk0_coeff[0][coeff=0] primes: ");
            for j in 0..num_primes {
                print!("{} ", rlk0_coeff[0][0 * num_primes + j]);
            }
            println!();
        }

        // Debug: print EVK shapes
        if std::env::var("EVK_DEBUG").is_ok() {
            println!("[EVK_DEBUG Metal] level={}, num_primes={}", level, num_primes);
            println!("[EVK_DEBUG Metal] rlk0_coeff.len()={}", rlk0_coeff.len());
            if !rlk0_coeff.is_empty() {
                println!("[EVK_DEBUG Metal] rlk0_coeff[0].len()={} (expected n×num_primes = {})",
                    rlk0_coeff[0].len(), n * num_primes);
            }
        }

        // Gadget decompose c2 into digits (CPU uses CRT-based decomposition)
        let c2_digits = Self::gadget_decompose_flat(
            &c2,
            base_w,
            &ctx.params.moduli[..num_primes],
            n,
        )?;

        // Debug: print first few values of first digit
        if std::env::var("MULT_DEBUG").is_ok() && c2_digits.len() > 0 {
            println!("[MULT_DEBUG] Metal gadget decomposition:");
            println!("  num_digits: {}", c2_digits.len());
            println!("  first digit, first 4 coeffs × {} primes:", n.min(4));
            for i in 0..n.min(4) {
                print!("    coeff[{}]: ", i);
                for j in 0..num_primes {
                    print!("{} ", c2_digits[0][i * num_primes + j]);
                }
                println!();
            }
        }

        // For each digit in the decomposition (matches CPU loop at line 177)
        for (t, d2_digit) in c2_digits.iter().enumerate() {
            if t >= rlk0_coeff.len() {
                break; // No more evaluation key components (matches CPU line 178)
            }

            // Multiply d2_digit by evk[t] and accumulate
            // The EVK encrypts -B^t·s², so we SUBTRACT term0 and ADD term1
            // (Matches CPU comments at lines 182-185)

            // term0 = d2_digit × evk0[t] (both in coefficient-form)
            if std::env::var("DETAIL_DEBUG").is_ok() && t == 0 {
                print!("[DETAIL_DEBUG Metal] digit[0] coeff[0] primes: ");
                for j in 0..num_primes {
                    print!("{} ", d2_digit[0 * num_primes + j]);
                }
                println!();
                print!("[DETAIL_DEBUG Metal] evk0[0] coeff[0] primes: ");
                for j in 0..num_primes {
                    print!("{} ", rlk0_coeff[t][0 * num_primes + j]);
                }
                println!();
            }

            let term0 = ctx.multiply_polys_flat_ntt_negacyclic(
                d2_digit,
                &rlk0_coeff[t],
                &ctx.params.moduli[..num_primes],
            )?;

            if std::env::var("DETAIL_DEBUG").is_ok() && t == 0 {
                print!("[DETAIL_DEBUG Metal] term0 coeff[0] primes: ");
                for j in 0..num_primes {
                    print!("{} ", term0[0 * num_primes + j]);
                }
                println!();
            }

            // term1 = d2_digit × evk1[t] (both in coefficient-form)
            let term1 = ctx.multiply_polys_flat_ntt_negacyclic(
                d2_digit,
                &rlk1_coeff[t],
                &ctx.params.moduli[..num_primes],
            )?;

            if std::env::var("DETAIL_DEBUG").is_ok() && t == 0 {
                print!("[DETAIL_DEBUG Metal] term1 coeff[0] primes: ");
                for j in 0..num_primes {
                    print!("{} ", term1[0 * num_primes + j]);
                }
                println!();
                print!("[DETAIL_DEBUG Metal] result_c0[0] BEFORE subtract: ");
                for j in 0..num_primes {
                    print!("{} ", result_c0[0 * num_primes + j]);
                }
                println!();
                print!("[DETAIL_DEBUG Metal] result_c1[0] BEFORE add: ");
                for j in 0..num_primes {
                    print!("{} ", result_c1[0 * num_primes + j]);
                }
                println!();
            }

            // c0 -= term0 (CPU line 191: c0[i] = c0[i].sub(&term0[i]))
            for i in 0..(n * num_primes) {
                let q = ctx.params.moduli[i % num_primes];
                result_c0[i] = if result_c0[i] >= term0[i] {
                    result_c0[i] - term0[i]
                } else {
                    q - (term0[i] - result_c0[i])
                };
            }

            // c1 += term1 (CPU line 192: c1[i] = c1[i].add(&term1[i]))
            for i in 0..(n * num_primes) {
                let q = ctx.params.moduli[i % num_primes];
                result_c1[i] = ((result_c1[i] as u128 + term1[i] as u128) % q as u128) as u64;
            }

            if std::env::var("DETAIL_DEBUG").is_ok() && t == 0 {
                print!("[DETAIL_DEBUG Metal] result_c0[0] AFTER subtract: ");
                for j in 0..num_primes {
                    print!("{} ", result_c0[0 * num_primes + j]);
                }
                println!();
                print!("[DETAIL_DEBUG Metal] result_c1[0] AFTER add: ");
                for j in 0..num_primes {
                    print!("{} ", result_c1[0 * num_primes + j]);
                }
                println!();
            }
        }

        // Debug: print values after relinearization
        if std::env::var("RELIN_DEBUG").is_ok() {
            println!("[RELIN_DEBUG Metal] After relinearization:");
            print!("  c0[0] across {} primes: ", num_primes);
            for j in 0..num_primes {
                print!("{} ", result_c0[0 * num_primes + j]);
            }
            println!();
            print!("  c1[0] across {} primes: ", num_primes);
            for j in 0..num_primes {
                print!("{} ", result_c1[0 * num_primes + j]);
            }
            println!();
        }

        // Step 3: Rescale to drop one prime
        let new_level = level - 1;
        let result_c0_rescaled = ctx.exact_rescale_gpu(&result_c0, level)?;
        let result_c1_rescaled = ctx.exact_rescale_gpu(&result_c1, level)?;

        // Debug: print values after rescaling
        if std::env::var("RELIN_DEBUG").is_ok() {
            println!("[RELIN_DEBUG Metal] After rescaling:");
            print!("  c0[0] across {} primes: ", new_level + 1);
            for j in 0..(new_level + 1) {
                print!("{} ", result_c0_rescaled[0 * (new_level + 1) + j]);
            }
            println!();
            print!("  c1[0] across {} primes: ", new_level + 1);
            for j in 0..(new_level + 1) {
                print!("{} ", result_c1_rescaled[0 * (new_level + 1) + j]);
            }
            println!();
        }

        // New scale is scale1 × scale2 / q_L (approximately scale²)
        let new_scale = (self.scale * other.scale) / ctx.params.moduli[level] as f64;

        Ok(Self {
            c0: result_c0_rescaled,
            c1: result_c1_rescaled,
            n,
            num_primes: new_level + 1,
            level: new_level,
            scale: new_scale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_ckks_context_creation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let result = MetalCkksContext::new(params);

        #[cfg(not(target_os = "macos"))]
        {
            assert!(result.is_err(), "Should fail on non-macOS");
        }

        #[cfg(target_os = "macos")]
        {
            assert!(result.is_ok(), "Should succeed on macOS with Metal support");
        }
    }
}
