//! V2 CKKS Encryption/Decryption with NTT Optimization
//!
//! **Optimizations over V1:**
//! - Uses Harvey NTT for O(n log n) polynomial operations
//! - Barrett reduction for fast modular arithmetic
//! - Precomputed NTT contexts for each prime
//! - Optimized RNS representation
//!
//! **Performance Target:** 10× faster encryption/decryption vs V1

use crate::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::{
    BarrettReducer, RnsContext, RnsRepresentation,
};
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::{PublicKey, SecretKey};
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use num_bigint::BigInt;
use num_traits::{Zero, One, ToPrimitive, Signed};
use num_integer::Integer;

/// V2 Plaintext in RNS representation with NTT support
#[derive(Clone, Debug)]
pub struct Plaintext {
    /// Polynomial coefficients in RNS form
    /// coeffs[i] = polynomial mod q_i (for each prime q_i)
    pub coeffs: Vec<RnsRepresentation>,

    /// Scaling factor for fixed-point encoding
    pub scale: f64,

    /// Ring dimension (polynomial degree)
    pub n: usize,

    /// Current level (number of active primes)
    pub level: usize,
}

impl Plaintext {
    /// Create plaintext from RNS coefficients
    pub fn new(coeffs: Vec<RnsRepresentation>, scale: f64, level: usize) -> Self {
        let n = coeffs.len();
        Self {
            coeffs,
            scale,
            n,
            level,
        }
    }

    /// Encode a vector of floats into plaintext using CKKS canonical embedding
    ///
    /// Uses proper orbit-ordered canonical embedding to ensure Galois automorphisms
    /// correspond to slot rotations (critical for homomorphic rotation).
    ///
    /// # Arguments
    /// * `values` - Float values to encode (length ≤ n/2 for CKKS)
    /// * `scale` - Scaling factor for fixed-point representation
    /// * `params` - CKKS parameters
    ///
    /// # Returns
    /// Encoded plaintext ready for encryption
    pub fn encode(values: &[f64], scale: f64, params: &CliffordFHEParams) -> Self {
        let n = params.n;
        let level = params.max_level();

        assert!(
            values.len() <= n / 2,
            "Too many values: {} > n/2 = {}",
            values.len(),
            n / 2
        );

        // Encode using canonical embedding with orbit ordering
        let coeffs_vec = canonical_embed_encode_real(values, scale, n);

        // Convert to RNS representation for each prime
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();
        let mut rns_coeffs = Vec::with_capacity(n);

        for &coeff in &coeffs_vec {
            // Handle negative coefficients by converting to positive mod q
            let values: Vec<u64> = moduli
                .iter()
                .map(|&q| {
                    if coeff >= 0 {
                        (coeff as u64) % q
                    } else {
                        let abs_coeff = (-coeff) as u64;
                        let remainder = abs_coeff % q;
                        if remainder == 0 {
                            0
                        } else {
                            q - remainder
                        }
                    }
                })
                .collect();

            rns_coeffs.push(RnsRepresentation::new(values, moduli.clone()));
        }

        Self::new(rns_coeffs, scale, level)
    }

    /// Encode values at a specific level (for bootstrap operations)
    ///
    /// **CRITICAL for bootstrap**: This ensures the plaintext has the same RNS basis
    /// as a ciphertext at the given level, preventing moduli mismatch errors.
    ///
    /// During bootstrap operations like EvalMod, we need to encode helper plaintexts
    /// (sine coefficients, scaling constants, etc.) that will be added/multiplied
    /// with ciphertexts. These plaintexts MUST have the exact same number of primes
    /// as the ciphertext they operate with.
    ///
    /// # Arguments
    /// * `values` - Float values to encode (length ≤ n/2 for CKKS)
    /// * `scale` - Scaling factor for fixed-point representation
    /// * `params` - CKKS parameters
    /// * `level` - Target level (determines which moduli to use: [0..=level])
    ///
    /// # Returns
    /// Encoded plaintext with RNS representation matching the target level
    ///
    /// # Example
    /// ```
    /// // Encode for a ciphertext at level 15 (uses primes [0..=15])
    /// let pt = Plaintext::encode_at_level(&values, params.scale, &params, 15);
    /// let ct_result = ct.add_plain(&pt); // No moduli mismatch!
    /// ```
    pub fn encode_at_level(values: &[f64], scale: f64, params: &CliffordFHEParams, level: usize) -> Self {
        let n = params.n;

        assert!(
            values.len() <= n / 2,
            "Too many values: {} > n/2 = {}",
            values.len(),
            n / 2
        );

        assert!(
            level <= params.max_level(),
            "Level {} exceeds max_level {}",
            level,
            params.max_level()
        );

        // Encode using canonical embedding with orbit ordering
        let coeffs_vec = canonical_embed_encode_real(values, scale, n);

        // Convert to RNS representation using ONLY the moduli up to the target level
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();
        let mut rns_coeffs = Vec::with_capacity(n);

        for &coeff in &coeffs_vec {
            // Handle negative coefficients by converting to positive mod q
            let values: Vec<u64> = moduli
                .iter()
                .map(|&q| {
                    if coeff >= 0 {
                        (coeff as u64) % q
                    } else {
                        let abs_coeff = (-coeff) as u64;
                        let remainder = abs_coeff % q;
                        if remainder == 0 {
                            0
                        } else {
                            q - remainder
                        }
                    }
                })
                .collect();

            rns_coeffs.push(RnsRepresentation::new(values, moduli.clone()));
        }

        Self::new(rns_coeffs, scale, level)
    }

    /// Encode values for use in multiply_plain operations
    ///
    /// **IMPORTANT:** This is just an alias for the standard encode().
    /// The key insight: use the SAME encoder for both ciphertext data and plaintext
    /// multipliers. This automatically matches all normalization constants (including
    /// any canonical embedding factors), eliminating the need for κ compensation.
    ///
    /// For slot-wise multiplication by a diagonal/mask, encode the diagonal vector
    /// with the same scale Δ used for ciphertext encoding. The polynomial multiplication
    /// in the ring correctly implements component-wise multiplication in slots.
    ///
    /// # Arguments
    /// * `values` - Slot values (diagonal, mask, or constant vector)
    /// * `base_scale` - Same scale used for ciphertext (typically params.scale)
    /// * `params` - CKKS parameters
    ///
    /// # Example
    /// ```
    /// // Slot-wise multiply by constant 2.0
    /// let pt = Plaintext::encode_for_plain_mul(&vec![2.0; n/2], params.scale, &params);
    /// let ct_result = ct.multiply_plain(&pt, &ckks_ctx); // Implements 2× in each slot
    /// ```
    pub fn encode_for_plain_mul(values: &[f64], base_scale: f64, params: &CliffordFHEParams) -> Self {
        // Just use normal encoder - same normalization as ciphertext encoder
        Self::encode(values, base_scale, params)
    }

    /// Decode plaintext to vector of floats using canonical embedding
    ///
    /// Uses proper orbit-ordered canonical embedding to decode slots.
    ///
    /// # Arguments
    /// * `params` - CKKS parameters
    ///
    /// # Returns
    /// Decoded float values (N/2 slots)
    pub fn decode(&self, params: &CliffordFHEParams) -> Vec<f64> {
        // For simplicity, use the first (largest) prime for decoding
        // This avoids CRT overflow issues and is sufficient for testing
        // In production, would use proper multi-prime CRT or Garner's algorithm

        let q0 = params.moduli[0];
        let half_q0 = (q0 / 2) as i128;

        let mut coeffs_i64 = Vec::with_capacity(self.n);

        for rns_coeff in &self.coeffs {
            let val_mod_q0 = rns_coeff.values[0] as i128;

            // Convert to signed integer (centered lift)
            let signed_val = if val_mod_q0 > half_q0 {
                val_mod_q0 - (q0 as i128)
            } else {
                val_mod_q0
            };

            coeffs_i64.push(signed_val as i64);
        }

        // Decode using canonical embedding
        canonical_embed_decode_real(&coeffs_i64, self.scale, self.n)
    }
}

/// V2 Ciphertext in RNS representation with NTT support
#[derive(Clone, Debug)]
pub struct Ciphertext {
    /// First component c0 (RNS polynomial)
    pub c0: Vec<RnsRepresentation>,

    /// Second component c1 (RNS polynomial)
    pub c1: Vec<RnsRepresentation>,

    /// Current level (determines active primes)
    pub level: usize,

    /// Scaling factor
    pub scale: f64,

    /// Ring dimension
    pub n: usize,
}

impl Ciphertext {
    /// Create new ciphertext
    pub fn new(
        c0: Vec<RnsRepresentation>,
        c1: Vec<RnsRepresentation>,
        level: usize,
        scale: f64,
    ) -> Self {
        let n = c0.len();
        assert_eq!(c1.len(), n, "c0 and c1 must have same length");

        Self {
            c0,
            c1,
            level,
            scale,
            n,
        }
    }

    /// Add two ciphertexts (homomorphic addition)
    ///
    /// **Complexity:** O(n) per prime (no NTT needed for addition)
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n, "Ciphertexts must have same dimension");

        // Align levels if needed
        let (self_aligned, other_aligned) = if self.level != other.level {
            let min_level = self.level.min(other.level);
            let self_new = if self.level > min_level {
                self.mod_switch_to_level(min_level)
            } else {
                self.clone()
            };
            let other_new = if other.level > min_level {
                other.mod_switch_to_level(min_level)
            } else {
                other.clone()
            };
            (self_new, other_new)
        } else {
            (self.clone(), other.clone())
        };

        assert!(
            (self_aligned.scale - other_aligned.scale).abs() < 1.0,
            "Ciphertexts must have similar scale"
        );

        let c0: Vec<RnsRepresentation> = self_aligned
            .c0
            .iter()
            .zip(&other_aligned.c0)
            .map(|(a, b)| a.add(b))
            .collect();

        let c1: Vec<RnsRepresentation> = self_aligned
            .c1
            .iter()
            .zip(&other_aligned.c1)
            .map(|(a, b)| a.add(b))
            .collect();

        Self::new(c0, c1, self_aligned.level, self_aligned.scale)
    }

    /// Subtract two ciphertexts (homomorphic subtraction)
    ///
    /// **Complexity:** O(n) per prime
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n, "Ciphertexts must have same dimension");

        // Align levels if needed
        let (self_aligned, other_aligned) = if self.level != other.level {
            let min_level = self.level.min(other.level);
            let self_new = if self.level > min_level {
                self.mod_switch_to_level(min_level)
            } else {
                self.clone()
            };
            let other_new = if other.level > min_level {
                other.mod_switch_to_level(min_level)
            } else {
                other.clone()
            };
            (self_new, other_new)
        } else {
            (self.clone(), other.clone())
        };

        let c0: Vec<RnsRepresentation> = self_aligned
            .c0
            .iter()
            .zip(&other_aligned.c0)
            .map(|(a, b)| a.sub(b))
            .collect();

        let c1: Vec<RnsRepresentation> = self_aligned
            .c1
            .iter()
            .zip(&other_aligned.c1)
            .map(|(a, b)| a.sub(b))
            .collect();

        Self::new(c0, c1, self_aligned.level, self_aligned.scale)
    }

    /// Multiply ciphertext by scalar (homomorphic scalar multiplication)
    ///
    /// **Complexity:** O(n) per prime
    pub fn mul_scalar(&self, scalar: f64) -> Self {
        // For plaintext-ciphertext multiplication in CKKS:
        // - Ciphertext encrypts ⌊m * Δ⌉ with scale Δ
        // - We want result to encrypt ⌊(m * s) * Δ⌉ with SAME scale Δ
        //
        // Standard CKKS approach:
        // - Encode scalar s as plaintext polynomial (just the value, no scaling)
        // - Multiply: ct * s keeps scale Δ
        //
        // For special cases:
        // - Integer scalars: multiply RNS values directly
        // - 0.5: use modular inverse of 2

        let c0: Vec<RnsRepresentation>;
        let c1: Vec<RnsRepresentation>;

        if (scalar - 0.5).abs() < 1e-10 {
            // Special case: division by 2
            // Compute 2^(-1) mod q for each prime, then multiply
            c0 = self
                .c0
                .iter()
                .map(|rns| {
                    let values: Vec<u64> = rns
                        .values
                        .iter()
                        .zip(&rns.moduli)
                        .map(|(&val, &q)| {
                            // Compute 2^(-1) mod q = (q + 1) / 2
                            let inv2 = (q + 1) / 2;
                            ((val as u128 * inv2 as u128) % q as u128) as u64
                        })
                        .collect();
                    RnsRepresentation::new(values, rns.moduli.clone())
                })
                .collect();

            c1 = self
                .c1
                .iter()
                .map(|rns| {
                    let values: Vec<u64> = rns
                        .values
                        .iter()
                        .zip(&rns.moduli)
                        .map(|(&val, &q)| {
                            let inv2 = (q + 1) / 2;
                            ((val as u128 * inv2 as u128) % q as u128) as u64
                        })
                        .collect();
                    RnsRepresentation::new(values, rns.moduli.clone())
                })
                .collect();
        } else if scalar.abs() < 1e-10 {
            // Multiplication by 0 - return zero ciphertext
            let zero_rns = RnsRepresentation::from_u64(0, &self.c0[0].moduli);
            c0 = vec![zero_rns.clone(); self.n];
            c1 = vec![zero_rns; self.n];
        } else if (scalar - 1.0).abs() < 1e-10 {
            // Multiplication by 1 - return copy
            c0 = self.c0.clone();
            c1 = self.c1.clone();
        } else if (scalar - scalar.round()).abs() < 1e-10 {
            // Integer scalar: multiply directly
            let scalar_int = scalar.round() as u64;
            c0 = self.c0.iter().map(|rns| rns.mul_scalar(scalar_int)).collect();
            c1 = self.c1.iter().map(|rns| rns.mul_scalar(scalar_int)).collect();
        } else {
            // Fractional scalar (not 0.5): NOT IMPLEMENTED YET
            // This requires computing modular inverse which is complex for arbitrary fractions
            panic!("Multiplication by fractional scalar {} not implemented (only 0.5 supported)", scalar);
        }

        // Scale stays the same!
        Self::new(c0, c1, self.level, self.scale)
    }

    /// Add plaintext to ciphertext (homomorphic plaintext addition)
    pub fn add_plaintext(&self, pt: &Plaintext) -> Self {
        assert_eq!(self.n, pt.n, "Dimensions must match");
        assert_eq!(self.level, pt.level, "Levels must match");

        let c0: Vec<RnsRepresentation> = self
            .c0
            .iter()
            .zip(&pt.coeffs)
            .map(|(ct, pt)| ct.add(pt))
            .collect();

        Self::new(c0.clone(), self.c1.clone(), self.level, self.scale)
    }

    /// Multiply ciphertext by plaintext (homomorphic plaintext multiplication)
    ///
    /// This operation multiplies an encrypted value by a known plaintext value.
    /// Unlike ciphertext-ciphertext multiplication, this does NOT require
    /// relinearization, making it much faster and cheaper.
    ///
    /// **Algorithm:**
    /// (c0, c1) * pt = (c0 * pt, c1 * pt)
    ///
    /// **Complexity:** O(n log n) per prime (using NTT for polynomial multiplication)
    ///
    /// # Arguments
    /// * `pt` - Plaintext to multiply (must have same level)
    ///
    /// # Returns
    /// New ciphertext encrypting the product
    pub fn multiply_plain(&self, pt: &Plaintext, ckks_ctx: &CkksContext) -> Self {
        assert_eq!(self.n, pt.n, "Dimensions must match");
        assert_eq!(self.level, pt.level, "Levels must match for plaintext multiplication");

        // Note: Plaintext scale determines the effective scale after multiplication
        // Testing with pt.scale to see impact on results

        let moduli = &ckks_ctx.params.moduli[..=self.level];

        // Multiply both c0 and c1 by plaintext using NTT
        // After multiplication: effective scale is self.scale × pt.scale (≈ Δ²)
        let new_c0 = ckks_ctx.multiply_polys_ntt(&self.c0, &pt.coeffs, moduli);
        let new_c1 = ckks_ctx.multiply_polys_ntt(&self.c1, &pt.coeffs, moduli);

        // Compute the scale BEFORE rescale (this is the actual scale after polynomial multiply)
        let pre_rescale_scale = self.scale * pt.scale;

        // Create intermediate ciphertext (scale will be fixed in rescale)
        let ct_mult = Self::new(new_c0, new_c1, self.level, 0.0);

        // Rescale to bring scale back to ~Δ and drop one level
        ct_mult.rescale_to_next_with_scale(ckks_ctx, pre_rescale_scale)
    }

    // ==================== EXACT RESCALE HELPERS ====================
    // Following expert guidance: BigInt-based exact rescale with centered residues

    /// Convert residue to centered representation: [0, q) -> (-q/2, q/2]
    fn centered_residue(x: u64, q: u64) -> i128 {
        let x_i128 = x as i128;
        let q_i128 = q as i128;
        if x_i128 > q_i128 / 2 {
            x_i128 - q_i128
        } else {
            x_i128
        }
    }

    /// Convert centered residue back to canonical [0, q)
    fn canon_from_centered(t: i128, q: u64) -> u64 {
        let q_i128 = q as i128;
        let mut u = t % q_i128;
        if u < 0 {
            u += q_i128;
        }
        u as u64
    }

    /// CRT reconstruction using direct formula with centered residues
    ///
    /// Uses the explicit CRT formula: C = sum(r_i * M_i * (M_i^{-1} mod q_i)) mod Q
    /// where M_i = Q / q_i, and then centers the result to (-Q/2, Q/2]
    fn crt_reconstruct_centered(residues: &[u64], moduli: &[u64]) -> BigInt {
        assert_eq!(residues.len(), moduli.len());

        // Convert to centered representation
        let centered: Vec<i128> = residues.iter().zip(moduli.iter())
            .map(|(&r, &q)| Self::centered_residue(r, q))
            .collect();

        // Compute Q = product of all moduli
        let mut q_product = BigInt::one();
        for &q in moduli {
            q_product *= q;
        }

        // CRT reconstruction: C = sum(r_i * M_i * (M_i^{-1} mod q_i)) mod Q
        // where M_i = Q / q_i
        let mut result = BigInt::zero();

        for i in 0..moduli.len() {
            let r_i = BigInt::from(centered[i]);
            let q_i = BigInt::from(moduli[i]);

            // M_i = Q / q_i
            let m_i = &q_product / &q_i;

            // Compute M_i^{-1} mod q_i
            let m_i_inv = Self::mod_inverse(&m_i, &q_i);

            // term_i = r_i * M_i * (M_i^{-1} mod q_i)
            let term_i = r_i * &m_i * m_i_inv;

            result += term_i;
        }

        // Reduce mod Q and center to (-Q/2, Q/2]
        result %= &q_product;
        let half_q = &q_product >> 1;
        if result > half_q {
            result -= &q_product;
        }

        result
    }

    /// Compute modular inverse using extended Euclidean algorithm
    fn mod_inverse(a: &BigInt, m: &BigInt) -> BigInt {
        let (gcd, x, _) = Self::extended_gcd(a, m);
        assert_eq!(gcd, BigInt::one(), "Modular inverse does not exist");
        let mut result = x % m;
        if result < BigInt::zero() {
            result += m;
        }
        result
    }

    /// Extended Euclidean algorithm: returns (gcd, x, y) such that ax + my = gcd
    fn extended_gcd(a: &BigInt, m: &BigInt) -> (BigInt, BigInt, BigInt) {
        if a.is_zero() {
            return (m.clone(), BigInt::zero(), BigInt::one());
        }
        let (gcd, x1, y1) = Self::extended_gcd(&(m % a), a);
        let x = y1 - (m / a) * &x1;
        let y = x1;
        (gcd, x, y)
    }

    /// Rounded division by q_top with proper rounding
    ///
    /// For C >= 0: C' = floor((C + floor(q_top/2)) / q_top)
    /// For C < 0:  C' = -floor((-C + floor(q_top/2)) / q_top)
    fn rounded_div_by_qtop(c: &BigInt, q_top: u64) -> BigInt {
        let q_top_big = BigInt::from(q_top);
        let half: BigInt = &q_top_big >> 1; // floor(q_top / 2)

        if c.is_negative() {
            let c_pos: BigInt = -c;
            let numerator: BigInt = c_pos + &half;
            -(numerator.div_floor(&q_top_big))
        } else {
            let numerator: BigInt = c + &half;
            numerator.div_floor(&q_top_big)
        }
    }

    /// Rescale a single coefficient: CRT reconstruct, divide by q_top, re-encode to RNS
    pub fn rescale_coeff_bigint(limbs: &[u64], moduli: &[u64], q_top: u64) -> Vec<u64> {
        // 1. CRT reconstruct with centered residues
        let c_full = Self::crt_reconstruct_centered(limbs, moduli);

        // 2. Rounded division by q_top
        let c_divided = Self::rounded_div_by_qtop(&c_full, q_top);

        // 3. Re-encode to remaining moduli with centered reduction
        let new_moduli = &moduli[..moduli.len() - 1];
        new_moduli.iter()
            .map(|&q_j| {
                let q_j_big = BigInt::from(q_j as i128);
                let r = &c_divided % &q_j_big;
                let r_i128 = r.to_i128().unwrap_or(0);
                Self::canon_from_centered(r_i128, q_j)
            })
            .collect()
    }

    /// Rescale ciphertext by dropping the top modulus and dividing by it (EXACT VERSION)
    ///
    /// After ciphertext-plaintext multiplication where both have scale Δ,
    /// the result has effective scale Δ². This operation brings it back to ~Δ
    /// by dividing by the top modulus q_top and dropping that RNS limb.
    ///
    /// This is the standard CKKS rescale operation with EXACT rounded division.
    ///
    /// **Implementation:** Following expert guidance, this uses BigInt CRT reconstruction
    /// with centered residues, proper rounded division, and re-encoding to RNS.
    ///
    /// # Arguments
    /// * `ckks_ctx` - CKKS context containing parameters
    /// * `pre_rescale_scale` - The actual scale before rescale (e.g., Δ² after multiply)
    ///
    /// # Returns
    /// Rescaled ciphertext with level decreased by 1 and scale = pre_rescale_scale / q_top
    pub fn rescale_to_next_with_scale(&self, ckks_ctx: &CkksContext, pre_rescale_scale: f64) -> Self {
        let level = self.level;
        assert!(level > 0, "Cannot rescale at level 0");

        let moduli = &ckks_ctx.params.moduli[..=level];
        let q_top = moduli[level];

        let mut new_c0 = Vec::with_capacity(self.n);
        let mut new_c1 = Vec::with_capacity(self.n);

        // EXACT rescale: For each coefficient, do CRT reconstruct → divide → re-encode
        for i in 0..self.n {
            let new_c0_limbs = Self::rescale_coeff_bigint(&self.c0[i].values, moduli, q_top);
            let new_c1_limbs = Self::rescale_coeff_bigint(&self.c1[i].values, moduli, q_top);

            let new_moduli = moduli[..level].to_vec();
            new_c0.push(RnsRepresentation::new(new_c0_limbs, new_moduli.clone()));
            new_c1.push(RnsRepresentation::new(new_c1_limbs, new_moduli));
        }

        // Update scale metadata: divide by q_top
        let new_scale = pre_rescale_scale / (q_top as f64);
        Self::new(new_c0, new_c1, level - 1, new_scale)
    }

    /// Reconstruct a BigInt from RNS representation using Chinese Remainder Theorem
    /// For SLOW validation rescale only
    fn rns_to_bigint(limbs: &[u64], moduli: &[u64]) -> u128 {
        assert_eq!(limbs.len(), moduli.len());

        // Compute Q = product of all moduli
        let mut q_product: u128 = 1;
        for &q in moduli {
            q_product *= q as u128;
        }

        // CRT reconstruction: sum_i (limbs[i] * (Q/q_i) * ((Q/q_i)^{-1} mod q_i))
        let mut result: u128 = 0;
        for i in 0..limbs.len() {
            let q_i = moduli[i] as u128;
            let q_hat_i = q_product / q_i; // Q / q_i

            // Compute (Q/q_i)^{-1} mod q_i
            let q_hat_inv = Self::mod_inverse_u128(q_hat_i % q_i, q_i);

            // Add contribution: limbs[i] * (Q/q_i) * ((Q/q_i)^{-1} mod q_i)
            let contrib = (limbs[i] as u128 * q_hat_i % q_product) * q_hat_inv % q_product;
            result = (result + contrib) % q_product;
        }

        result
    }

    /// Compute modular inverse using extended Euclidean algorithm for u128
    fn mod_inverse_u128(a: u128, m: u128) -> u128 {
        fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
            if b == 0 {
                (a, 1, 0)
            } else {
                let (gcd, x1, y1) = extended_gcd(b, a % b);
                (gcd, y1, x1 - (a / b) * y1)
            }
        }

        let (gcd, x, _) = extended_gcd(a as i128, m as i128);
        assert_eq!(gcd, 1, "No modular inverse exists");

        if x < 0 {
            ((x + m as i128) as u128) % m
        } else {
            (x as u128) % m
        }
    }

    /// Rounded division: ⌊a / b⌉ (round to nearest, ties to even)
    fn rounded_div_u128(a: u128, b: u128) -> u128 {
        let quotient = a / b;
        let remainder = a % b;

        // Round to nearest: if remainder >= b/2, round up
        if remainder * 2 >= b {
            // Check for tie (remainder == b/2)
            if remainder * 2 == b {
                // Tie: round to even
                if quotient % 2 == 0 {
                    quotient
                } else {
                    quotient + 1
                }
            } else {
                // Not a tie: round up
                quotient + 1
            }
        } else {
            // Round down
            quotient
        }
    }

    /// Modulus switch to a target level (drop primes without rescaling)
    ///
    /// This operation brings a ciphertext from a higher level (more primes)
    /// to a lower level (fewer primes) without changing the scale.
    /// Used to align levels before multiplication.
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

        // Drop RNS components to match target level
        // target_level is index of last prime to keep, so we want [0..=target_level]
        let new_c0: Vec<RnsRepresentation> = self
            .c0
            .iter()
            .map(|rns| {
                let new_values = rns.values[..=target_level].to_vec();
                let new_moduli = rns.moduli[..=target_level].to_vec();
                RnsRepresentation::new(new_values, new_moduli)
            })
            .collect();

        let new_c1: Vec<RnsRepresentation> = self
            .c1
            .iter()
            .map(|rns| {
                let new_values = rns.values[..=target_level].to_vec();
                let new_moduli = rns.moduli[..=target_level].to_vec();
                RnsRepresentation::new(new_values, new_moduli)
            })
            .collect();

        Self::new(new_c0, new_c1, target_level, self.scale)
    }
}

/// CKKS Context with precomputed NTT transforms
pub struct CkksContext {
    /// Parameters
    pub params: CliffordFHEParams,

    /// NTT contexts for each prime
    pub ntt_contexts: Vec<NttContext>,

    /// Barrett reducers for each prime
    pub reducers: Vec<BarrettReducer>,

    /// Calibration constant for multiply_plain normalization
    /// This compensates for encode/decode canonical embedding mismatch
    /// κ ≈ n/2 typically; computed once per parameter set
    pub kappa: f64,

    /// RNS context for CRT
    pub rns_context: RnsContext,
}

impl CkksContext {
    /// Create new CKKS context with precomputed NTT data
    pub fn new(params: CliffordFHEParams) -> Self {
        let moduli = params.moduli.clone();

        // Create NTT context for each prime
        let ntt_contexts: Vec<NttContext> = moduli
            .iter()
            .map(|&q| NttContext::new(params.n, q))
            .collect();

        // Create Barrett reducers
        let reducers: Vec<BarrettReducer> = moduli
            .iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        // Create RNS context
        let rns_context = RnsContext::new(moduli);

        // κ ≈ n/2 × 1.46 (empirically measured normalization constant)
        // This compensates for encode/decode canonical embedding mismatch
        // TODO: Make canonical embedding unitary to eliminate this factor
        let n_over_2 = (params.n / 2) as f64;
        let kappa = n_over_2 * 1.46;

        Self {
            params,
            ntt_contexts,
            reducers,
            kappa,
            rns_context,
        }
    }

    /// Calibrate κ for multiply_plain normalization
    /// This compensates for encode/decode canonical embedding mismatch
    /// Should be called once per parameter set with a keypair
    pub fn calibrate_kappa(&mut self, pk: &PublicKey, sk: &SecretKey) {
        use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

        // Encrypt constant vector of 1.0 in all slots
        let num_slots = self.params.n / 2;
        let ones = vec![1.0; num_slots];
        let pt_x = Plaintext::encode(&ones, self.params.scale, &self.params);
        let ct_x = self.encrypt(&pt_x, pk);

        // Plaintext constant vector of 1.0
        let pt_c = Plaintext::encode(&ones, self.params.scale, &self.params);

        // ct_y := ct_x ⊙ pt_c with rescale
        let ct_y = ct_x.multiply_plain(&pt_c, self);

        // Decrypt and decode
        let pt_y = self.decrypt(&ct_y, sk);
        let y = pt_y.decode(&self.params);

        // κ = mean(y), since true answer is 1.0 in every slot
        let mean_y: f64 = y.iter().take(num_slots).sum::<f64>() / (num_slots as f64);
        self.kappa = mean_y;

        println!("Calibrated κ = {} for multiply_plain", self.kappa);
    }

    /// Encode float values to plaintext
    pub fn encode(&self, values: &[f64]) -> Plaintext {
        Plaintext::encode(values, self.params.scale, &self.params)
    }

    /// Encode values at a specific level (for bootstrap operations)
    ///
    /// This ensures the plaintext has the same RNS basis as a ciphertext at the given level.
    /// **CRITICAL for bootstrap**: All encode operations must match the ciphertext's level
    /// to avoid moduli mismatch errors.
    ///
    /// # Arguments
    /// * `values` - Float values to encode
    /// * `level` - Target level (determines which moduli to use: [0..=level])
    ///
    /// # Returns
    /// Plaintext with RNS representation matching the target level
    pub fn encode_at_level(&self, values: &[f64], level: usize) -> Plaintext {
        Plaintext::encode_at_level(values, self.params.scale, &self.params, level)
    }

    /// Decode plaintext to float values
    pub fn decode(&self, pt: &Plaintext) -> Vec<f64> {
        pt.decode(&self.params)
    }

    /// Encrypt plaintext to ciphertext using public key
    ///
    /// **Algorithm:**
    /// 1. Sample error polynomials e0, e1 ~ N(0, σ²)
    /// 2. Sample random u ~ {-1, 0, 1}
    /// 3. c0 = b*u + e0 + m (where b is from public key)
    /// 4. c1 = a*u + e1 (where a is from public key)
    ///
    /// **Security:** Relies on RLWE hardness
    ///
    /// # Arguments
    /// * `pt` - Plaintext to encrypt
    /// * `pk` - Public key (a, b) where b = -a*s + e
    ///
    /// # Returns
    /// Ciphertext (c0, c1) encrypting the plaintext
    pub fn encrypt(&self, pt: &Plaintext, pk: &crate::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey) -> Ciphertext {
        use rand::{thread_rng, Rng};
        use rand_distr::{Distribution, Normal};

        let n = self.params.n;
        let level = self.params.max_level();
        let moduli: Vec<u64> = self.params.moduli[..=level].to_vec();
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

        // Convert to RNS representation
        let u = self.coeffs_to_rns(&u_coeffs, &moduli);
        let e0 = self.coeffs_to_rns(&e0_coeffs, &moduli);
        let e1 = self.coeffs_to_rns(&e1_coeffs, &moduli);

        // c0 = b*u + e0 + m
        let bu = self.multiply_polys_ntt(&pk.b, &u.coeffs, &moduli);
        let c0: Vec<RnsRepresentation> = bu
            .iter()
            .zip(&e0.coeffs)
            .zip(&pt.coeffs)
            .map(|((bu_i, e0_i), m_i)| bu_i.add(e0_i).add(m_i))
            .collect();

        // c1 = a*u + e1
        let au = self.multiply_polys_ntt(&pk.a, &u.coeffs, &moduli);
        let c1: Vec<RnsRepresentation> = au
            .iter()
            .zip(&e1.coeffs)
            .map(|(au_i, e1_i)| au_i.add(e1_i))
            .collect();

        Ciphertext::new(c0, c1, level, pt.scale)
    }

    /// Decrypt ciphertext to plaintext using secret key
    ///
    /// **Algorithm:**
    /// m = c0 + c1*s (mod q)
    ///
    /// **Decryption correctness:**
    /// m ≈ (b*u + e0 + m) + (a*u + e1)*s
    ///   = b*u + a*u*s + e_total + m
    ///   = (-a*s + e_pk)*u + a*u*s + e_total + m
    ///   = e_pk*u + e_total + m
    ///   ≈ m (if noise is small)
    ///
    /// # Arguments
    /// * `ct` - Ciphertext to decrypt
    /// * `sk` - Secret key s
    ///
    /// # Returns
    /// Plaintext approximating the encrypted message
    pub fn decrypt(&self, ct: &Ciphertext, sk: &crate::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey) -> Plaintext {
        let moduli: Vec<u64> = self.params.moduli[..=ct.level].to_vec();

        // Extract secret key coefficients at the ciphertext's level
        // Secret key was generated at max_level, but ciphertext may be at lower level
        let sk_at_level: Vec<RnsRepresentation> = sk
            .coeffs
            .iter()
            .map(|rns| {
                // Take only the RNS components for active primes at this level
                let values = rns.values[..=ct.level].to_vec();
                let moduli_at_level = rns.moduli[..=ct.level].to_vec();
                RnsRepresentation::new(values, moduli_at_level)
            })
            .collect();

        // Compute c1 * s
        let c1s = self.multiply_polys_ntt(&ct.c1, &sk_at_level, &moduli);

        // m = c0 + c1*s
        let m: Vec<RnsRepresentation> = ct
            .c0
            .iter()
            .zip(&c1s)
            .map(|(c0_i, c1s_i)| c0_i.add(c1s_i))
            .collect();

        // Debug: print first coefficient
        if std::env::var("DECRYPT_DEBUG").is_ok() {
            println!("[DECRYPT_DEBUG CPU] c0[0] prime 0: {}", ct.c0[0].values[0]);
            println!("[DECRYPT_DEBUG CPU] c1[0] prime 0: {}", ct.c1[0].values[0]);
            println!("[DECRYPT_DEBUG CPU] sk[0] prime 0: {}", sk_at_level[0].values[0]);
            println!("[DECRYPT_DEBUG CPU] c1s[0] prime 0: {}", c1s[0].values[0]);
            println!("[DECRYPT_DEBUG CPU] m[0] prime 0: {}", m[0].values[0]);
        }

        Plaintext::new(m, ct.scale, ct.level)
    }

    /// Convert signed integer coefficients to RNS representation
    fn coeffs_to_rns(&self, coeffs: &[i64], moduli: &[u64]) -> Plaintext {
        let n = coeffs.len();
        let mut rns_coeffs = Vec::with_capacity(n);

        for &coeff in coeffs {
            let values: Vec<u64> = moduli
                .iter()
                .map(|&q| {
                    if coeff >= 0 {
                        (coeff as u64) % q
                    } else {
                        let abs_coeff = (-coeff) as u64;
                        let remainder = abs_coeff % q;
                        if remainder == 0 {
                            0
                        } else {
                            q - remainder
                        }
                    }
                })
                .collect();

            rns_coeffs.push(RnsRepresentation::new(values, moduli.to_vec()));
        }

        Plaintext::new(rns_coeffs, 1.0, moduli.len() - 1)
    }

    /// Multiply two polynomials using NTT (negacyclic convolution mod x^n + 1)
    ///
    /// Uses V2's fixed NTT implementation (twisted NTT for negacyclic convolution).
    fn multiply_polys_ntt(
        &self,
        a: &[RnsRepresentation],
        b: &[RnsRepresentation],
        moduli: &[u64],
    ) -> Vec<RnsRepresentation> {
        let n = a.len();
        let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.to_vec()); n];

        // For each prime modulus, multiply using V2's NTT
        for (prime_idx, &q) in moduli.iter().enumerate() {
            // Create NTT context for this prime
            let ntt_ctx = super::ntt::NttContext::new(n, q);

            // Extract coefficients for this prime (already in u64 mod q form)
            let a_mod_q: Vec<u64> = a.iter().map(|rns| rns.values[prime_idx]).collect();
            let b_mod_q: Vec<u64> = b.iter().map(|rns| rns.values[prime_idx]).collect();

            // Multiply using V2's NTT (negacyclic convolution)
            let product_mod_q = ntt_ctx.multiply_polynomials(&a_mod_q, &b_mod_q);

            // Store result (convert u64 back to i64)
            for i in 0..n {
                result[i].values[prime_idx] = product_mod_q[i];
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plaintext_encode_decode() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CkksContext::new(params.clone());

        let values = vec![1.5, 2.7, -3.2, 0.5];
        let pt = ctx.encode(&values);

        assert_eq!(pt.n, params.n);
        assert_eq!(pt.scale, params.scale);

        let decoded = ctx.decode(&pt);

        // Check first few values (rest should be ~0)
        for i in 0..values.len() {
            let error = (decoded[i] - values[i]).abs();
            assert!(
                error < 1e-6,
                "Value {} decode error: expected {}, got {} (error: {})",
                i,
                values[i],
                decoded[i],
                error
            );
        }
    }

    #[test]
    fn test_plaintext_encode_negative() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CkksContext::new(params);

        let values = vec![-10.5, -20.3, -5.0];
        let pt = ctx.encode(&values);
        let decoded = ctx.decode(&pt);

        for i in 0..values.len() {
            let error = (decoded[i] - values[i]).abs();
            assert!(error < 1e-6, "Negative value decode error at index {}", i);
        }
    }

    #[test]
    fn test_ciphertext_addition() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let n = params.n;
        let level = params.max_level();
        let moduli = params.moduli[..=level].to_vec();

        // Create dummy ciphertexts with simple RNS values
        let c0_a: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64(i as u64, &moduli))
            .collect();
        let c1_a: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((i + 1) as u64, &moduli))
            .collect();

        let c0_b: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((2 * i) as u64, &moduli))
            .collect();
        let c1_b: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((2 * i + 1) as u64, &moduli))
            .collect();

        let ct_a = Ciphertext::new(c0_a, c1_a, level, params.scale);
        let ct_b = Ciphertext::new(c0_b, c1_b, level, params.scale);

        let ct_sum = ct_a.add(&ct_b);

        // Check dimensions
        assert_eq!(ct_sum.n, n);
        assert_eq!(ct_sum.level, level);

        // Check first coefficient of c0: should be 0 + 0 = 0
        assert_eq!(ct_sum.c0[0].values[0], 0);

        // Check second coefficient of c0: should be 1 + 2 = 3
        assert_eq!(ct_sum.c0[1].values[0], 3);
    }

    #[test]
    fn test_ciphertext_subtraction() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let n = params.n;
        let level = params.max_level();
        let moduli = params.moduli[..=level].to_vec();

        let c0_a: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((i + 10) as u64, &moduli))
            .collect();
        let c1_a: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64(i as u64, &moduli))
            .collect();

        let c0_b: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64(i as u64, &moduli))
            .collect();
        let c1_b: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64(i as u64, &moduli))
            .collect();

        let ct_a = Ciphertext::new(c0_a, c1_a, level, params.scale);
        let ct_b = Ciphertext::new(c0_b, c1_b, level, params.scale);

        let ct_diff = ct_a.sub(&ct_b);

        // First coefficient: (0+10) - 0 = 10
        assert_eq!(ct_diff.c0[0].values[0], 10);

        // Second coefficient: (1+10) - 1 = 10
        assert_eq!(ct_diff.c0[1].values[0], 10);
    }

    #[test]
    fn test_ciphertext_scalar_multiplication() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let n = params.n;
        let level = params.max_level();
        let moduli = params.moduli[..=level].to_vec();

        let c0: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((i + 1) as u64, &moduli))
            .collect();
        let c1: Vec<RnsRepresentation> = (0..n)
            .map(|_| RnsRepresentation::from_u64(0, &moduli))
            .collect();

        let ct = Ciphertext::new(c0, c1, level, params.scale);
        let scalar = 3.0;
        let ct_scaled = ct.mul_scalar(scalar);

        // Scale should STAY THE SAME (plaintext-ciphertext multiplication)
        assert!((ct_scaled.scale - params.scale).abs() < 0.1);

        // Level should stay the same
        assert_eq!(ct_scaled.level, level);
    }

    #[test]
    fn test_ckks_context_creation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CkksContext::new(params.clone());

        assert_eq!(ctx.ntt_contexts.len(), params.moduli.len());
        assert_eq!(ctx.reducers.len(), params.moduli.len());
        assert_eq!(ctx.rns_context.moduli.len(), params.moduli.len());
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CkksContext::new(params.clone());
        let key_ctx = KeyContext::new(params.clone());

        // Generate keys
        let (pk, sk, _evk) = key_ctx.keygen();

        // Create simple plaintext with known coefficients
        let level = params.max_level();
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();
        let scale = params.scale;

        // Create plaintext with small integer values scaled up
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
        for i in 0..values.len().min(params.n) {
            let scaled_val = (values[i] * scale) as u64;
            coeffs[i] = RnsRepresentation::from_u64(scaled_val, &moduli);
        }
        let pt = Plaintext::new(coeffs, scale, level);

        // Encrypt
        let ct = ctx.encrypt(&pt, &pk);

        // Decrypt
        let pt_decrypted = ctx.decrypt(&ct, &sk);

        // Check first few coefficients are approximately correct
        // Encryption adds noise, so we allow some tolerance
        for i in 0..values.len() {
            let original = pt.coeffs[i].values[0];
            let decrypted = pt_decrypted.coeffs[i].values[0];
            let q = moduli[0];

            // Convert to signed representation (centered lift)
            let to_signed = |val: u64| -> i128 {
                if val > q / 2 {
                    val as i128 - q as i128
                } else {
                    val as i128
                }
            };

            let original_signed = to_signed(original);
            let decrypted_signed = to_signed(decrypted);
            let error = (decrypted_signed - original_signed).abs();

            // Allow noise up to 5% of the scaled value (RLWE adds noise)
            let tolerance = (scale * 0.05) as i128;
            assert!(error < tolerance,
                "Coefficient {} error too large: {} vs {} (error: {}), q={}",
                i, decrypted_signed, original_signed, error, q);
        }
    }
}

//
// ============================================================================
// CKKS Canonical Embedding (Orbit-Ordered for Rotation Correctness)
// ============================================================================
//
// This implements the proper canonical embedding from CKKS that ensures
// Galois automorphisms correspond to slot rotations.
//
// **Critical for V3 rotation:** Simple coefficient encoding does NOT work
// with Galois automorphisms. We need proper FFT-based slot encoding.
//

use std::f64::consts::PI;

/// Compute the Galois orbit order for CKKS slot indexing
///
/// For power-of-two cyclotomics M=2N, the odd residues mod M form orbits
/// under multiplication by generator g (typically g=5). This function computes
/// the orbit starting from 1: e[t] = g^t mod M.
///
/// With this ordering, automorphism σ_g acts as a left rotation by 1 slot!
///
/// # Arguments
/// * `n` - Ring dimension N
/// * `g` - Generator (typically 5 for power-of-two cyclotomics)
///
/// # Returns
/// Vector e where e[t] = g^t mod M for t=0..(N/2-1)
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
///
/// Evaluates slots at the specific primitive roots ζ_M^(e[t])
/// to ensure automorphisms correspond to slot rotations.
///
/// # Arguments
/// * `values` - N/2 real values to encode (will be treated as complex with zero imaginary part)
/// * `scale` - Scaling factor
/// * `n` - Ring dimension (N in the formula above)
///
/// # Returns
/// Polynomial coefficients (length N)
fn canonical_embed_encode_real(values: &[f64], scale: f64, n: usize) -> Vec<i64> {
    assert!(n.is_power_of_two());
    let num_slots = n / 2;
    assert!(values.len() <= num_slots);

    let m = 2 * n; // Cyclotomic index M = 2N
    let g = 5; // Generator for power-of-two cyclotomics

    // Use Galois orbit order to ensure automorphism σ_g acts as rotate-by-1
    let e = orbit_order(n, g);

    // Pad values to full slot count
    let mut slots = vec![0.0; num_slots];
    for (i, &val) in values.iter().enumerate() {
        slots[i] = val;
    }

    // Inverse canonical embedding (orbit-order compatible)
    // For each coefficient j, sum over slots with both the slot value and its conjugate
    // This handles the Hermitian symmetry required for real coefficients
    //
    // Formula: c[j] = (1/N) * Re( Σ_t ( z[t] * w_t(j) + conj(z[t]) * conj(w_t(j)) ) )
    // where w_t(j) = exp(-2πi * e[t] * j / M)
    //
    // For real values: z[t] = conj(z[t]), so this simplifies to:
    // c[j] = (2/N) * Re( Σ_t z[t] * w_t(j) )
    let mut coeffs_float = vec![0.0; n];

    for j in 0..n {
        let mut sum = 0.0;

        for t in 0..num_slots {
            // w_t(j) = exp(-2πi * e[t] * j / M)
            let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
            let cos_val = angle.cos();

            // For real slots: contribution is 2 * real_part(z[t] * w)
            // = 2 * z[t] * cos(angle)
            sum += slots[t] * cos_val;
        }

        // Normalize by 2/N (factor of 2 from Hermitian symmetry)
        coeffs_float[j] = (2.0 / n as f64) * sum;
    }

    // Scale and round to integers
    let result: Vec<i64> = coeffs_float.iter().map(|&x| (x * scale).round() as i64).collect();

    // DEBUG: Print first coefficient to verify encoding
    if result.len() > 0 {
        println!("  [encode DEBUG] values[0]={}, scale={}, coeffs_float[0]={}, result[0]={}",
                 if values.len() > 0 { values[0] } else { 0.0 },
                 scale,
                 coeffs_float[0],
                 result[0]);
    }

    result
}

/// Decode real-valued slots using CKKS canonical embedding with orbit ordering
///
/// Evaluates polynomial at the orbit-ordered primitive roots ζ_M^{e[t]}.
///
/// This is the adjoint of canonical_embed_encode_real.
///
/// # Arguments
/// * `coeffs` - Polynomial coefficients (length N)
/// * `scale` - Scaling factor
/// * `n` - Ring dimension
///
/// # Returns
/// N/2 real slot values
fn canonical_embed_decode_real(coeffs: &[i64], scale: f64, n: usize) -> Vec<f64> {
    assert_eq!(coeffs.len(), n);

    let m = 2 * n; // M = 2N
    let num_slots = n / 2;
    let g = 5; // Generator

    // Use Galois orbit order
    let e = orbit_order(n, g);

    // Convert to floating point (with scale normalization)
    let coeffs_float: Vec<f64> = coeffs.iter().map(|&c| c as f64 / scale).collect();

    // Forward canonical embedding: evaluate polynomial at ζ_M^{e[t]} for t = 0..N/2-1
    // Formula: y_t = Σ_{j=0}^{N-1} c[j] * exp(+2πi * e[t] * j / M)
    // For real results, take real part
    let mut slots = vec![0.0; num_slots];

    for t in 0..num_slots {
        let mut sum_real = 0.0;
        for j in 0..n {
            // w_t(j) = exp(+2πi * e[t] * j / M)  (note: positive angle for decode)
            let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
            let cos_val = angle.cos();
            sum_real += coeffs_float[j] * cos_val;
        }
        slots[t] = sum_real;
    }

    slots
}

#[cfg(test)]
mod multiply_plain_tests {
    use super::*;
    use crate::clifford_fhe_v2::params::CliffordFHEParams;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

    #[test]
    fn test_multiply_plain_simple() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let mut ckks_ctx = CkksContext::new(params.clone());

        // Encrypt constant vector [2.0, 2.0, ...] filling ALL slots
        let num_slots = params.n / 2;
        let values = vec![2.0; num_slots];
        let pt1 = Plaintext::encode(&values, params.scale, &params);
        let ct = ckks_ctx.encrypt(&pt1, &pk);

        println!("\n=== Scale Tracking ===");
        println!("Initial CT: level={}, scale={}", ct.level, ct.scale);

        // Encode plaintext for multiplication - use SAME encoder as ciphertext
        // This automatically matches normalization (no κ compensation needed)
        let multiplier = vec![2.0; num_slots];
        let pt2 = Plaintext::encode(&multiplier, params.scale, &params);

        println!("PT scale: {}", pt2.scale);
        println!("PT level: {}", pt2.level);
        println!("Pre-rescale scale should be: {} × {} = {}",
                 ct.scale, pt2.scale, ct.scale * pt2.scale);
        println!("q_top (level {}): {}", ct.level, params.moduli[ct.level]);
        println!("Expected post-rescale: {} / {} = {}",
                 ct.scale * pt2.scale,
                 params.moduli[ct.level],
                 (ct.scale * pt2.scale) / params.moduli[ct.level] as f64);

        let ct_result = ct.multiply_plain(&pt2, &ckks_ctx);

        println!("After multiply_plain: level={}, scale={}", ct_result.level, ct_result.scale);
        println!("======================\n");

        // Decrypt - should get [2.0, 4.0, 6.0, 8.0]
        let pt_result = ckks_ctx.decrypt(&ct_result, &sk);

        println!("PT result level: {}, scale: {}", pt_result.level, pt_result.scale);
        println!("Decoding with params.scale: {}", params.scale);

        let result = pt_result.decode(&params);

        println!("\n=== RESULTS ===");
        println!("Input: 2.0 × Multiplier: 2.0 → Expected: 4.0");
        println!("Result (first 4 slots): [{:.4}, {:.4}, {:.4}, {:.4}]",
                 result[0], result[1], result[2], result[3]);

        // Calculate mean absolute error
        let expected_value = 4.0; // 2×2
        let mut error_sum = 0.0;
        for i in 0..num_slots {
            error_sum += (result[i] - expected_value).abs();
        }
        let mean_abs_error = error_sum / (num_slots as f64);
        let relative_error = mean_abs_error / expected_value;
        println!("Mean absolute error: {:.6}", mean_abs_error);
        println!("Relative error: {:.2}%", relative_error * 100.0);

        // Test passes if relative error < 10% (accounting for CKKS noise + rescale approximation)
        println!("\n=== TEST VALIDATION ===");
        assert!(relative_error < 0.10,
               "Multiply-plain failed: relative error {:.2}% exceeds 10% threshold",
               relative_error * 100.0);
        println!("✓ Test PASSED: Relative error {:.2}% < 10%", relative_error * 100.0);
    }

    /// H1.1: UNIT TEST - Test encode → decode identity for single value
    #[test]
    fn test_h1_1_encode_decode_single() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let input = vec![2.0];

        let pt = Plaintext::encode(&input, params.scale, &params);
        let output = pt.decode(&params);

        let error = (output[0] - input[0]).abs();
        println!("\n=== H1.1: Encode/Decode Single Value ===");
        println!("Input: {}", input[0]);
        println!("Output: {}", output[0]);
        println!("Error: {}", error);

        assert!(error < 0.01, "Encode/decode single value failed: error = {}", error);
    }

    /// H7.1: UNIT TEST - Plaintext-only multiply (no encryption) to measure κ_plaintext
    #[test]
    fn test_h7_1_plaintext_multiply_no_encryption() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let num_slots = params.n / 2;

        // Encode two plaintexts
        let x_vals = vec![2.0; num_slots];
        let c_vals = vec![3.0; num_slots];

        let pt_x = Plaintext::encode(&x_vals, params.scale, &params);
        let pt_c = Plaintext::encode(&c_vals, params.scale, &params);

        // Manually multiply coefficients in RNS
        let mut result_coeffs = Vec::new();
        for i in 0..params.n {
            let mut values = Vec::new();
            for j in 0..pt_x.coeffs[i].values.len() {
                let q_j = pt_x.coeffs[i].moduli[j];
                let a = pt_x.coeffs[i].values[j];
                let b = pt_c.coeffs[i].values[j];

                // Multiply mod q_j
                let prod = ((a as u128) * (b as u128)) % (q_j as u128);
                values.push(prod as u64);
            }
            result_coeffs.push(RnsRepresentation::new(values, pt_x.coeffs[i].moduli.clone()));
        }

        // DEBUG: Check coefficients before creating result
        println!("\n=== H7.1: Plaintext-Only Multiply - DETAILED ===");
        println!("PT_x coeffs[0] (first limb): {}", pt_x.coeffs[0].values[0]);
        println!("PT_c coeffs[0] (first limb): {}", pt_c.coeffs[0].values[0]);
        println!("Result coeffs[0] (first limb): {}", result_coeffs[0].values[0]);
        println!("Δ = {}", params.scale);
        println!("Δ² = {}", params.scale * params.scale);

        // Create result plaintext with scale = Δ²
        let result_scale = params.scale * params.scale;
        let pt_result = Plaintext::new(result_coeffs, result_scale, pt_x.level);

        println!("pt_result.scale = {}", pt_result.scale);

        // Decode
        let output = pt_result.decode(&params);

        // Expected: 2 × 3 = 6
        let expected = 6.0;
        let mean_output: f64 = output.iter().take(num_slots).sum::<f64>() / (num_slots as f64);
        let kappa_plaintext = mean_output / expected;

        println!("Expected (2×3): {}", expected);
        println!("Mean output: {}", mean_output);
        println!("κ_plaintext = {:.2}", kappa_plaintext);
        println!("Output[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
                 output[0], output[1], output[2], output[3]);

        // This measures the gain from encode→multiply→decode WITHOUT encryption or rescale
    }

    /// H9.1: UNIT TEST - Isolated rescale test with known coefficient
    ///
    /// Tests that rescale_coeff_bigint correctly divides by q_top with rounding.
    /// Creates a synthetic coefficient C = k * q_top + r and verifies rescale produces k.
    #[test]
    fn test_h9_1_isolated_rescale() {
        use num_bigint::BigInt;

        let params = CliffordFHEParams::new_test_ntt_1024();
        let moduli = &params.moduli[..=params.max_level()];
        let q_top = moduli[moduli.len() - 1];

        println!("\n=== H9.1: Isolated Rescale Test ===");
        println!("Moduli: {:?}", moduli);
        println!("q_top: {}", q_top);

        // Test case 1: C = 5 * q_top + 100 (should round to 5)
        let k = 5i64;
        let r = 100i64;
        let c_full = BigInt::from(k) * BigInt::from(q_top) + BigInt::from(r);

        println!("\nTest 1: C = {} * {} + {} = {}", k, q_top, r, c_full);

        // Encode C to RNS
        let mut limbs = Vec::new();
        for &q_j in moduli {
            let limb = (&c_full % BigInt::from(q_j)).to_u64().unwrap();
            limbs.push(limb);
        }
        println!("C in RNS: {:?}", limbs);

        // Rescale using our function
        let rescaled_limbs = Ciphertext::rescale_coeff_bigint(&limbs, moduli, q_top);
        println!("After rescale limbs: {:?}", rescaled_limbs);

        // Reconstruct the rescaled value
        let new_moduli = &moduli[..moduli.len() - 1];
        let c_rescaled = Ciphertext::crt_reconstruct_centered(&rescaled_limbs, new_moduli);
        println!("Rescaled value: {}", c_rescaled);
        println!("Expected: {}", k);

        let diff = (&c_rescaled - BigInt::from(k)).abs();
        assert!(diff <= BigInt::from(1), "Rescale error too large: {} (expected {})", c_rescaled, k);

        // Test case 2: C = -3 * q_top + q_top/2 - 50 (negative, should round to -3)
        let k2 = -3i64;
        let r2 = (q_top / 2) as i64 - 50;
        let c_full2 = BigInt::from(k2) * BigInt::from(q_top) + BigInt::from(r2);

        println!("\nTest 2: C = {} * {} + {} = {}", k2, q_top, r2, c_full2);

        // Encode to RNS (handle negative)
        let mut limbs2 = Vec::new();
        for &q_j in moduli {
            let q_j_big = BigInt::from(q_j);
            let mut limb = &c_full2 % &q_j_big;
            if limb < BigInt::from(0) {
                limb += &q_j_big;
            }
            limbs2.push(limb.to_u64().unwrap());
        }

        let rescaled_limbs2 = Ciphertext::rescale_coeff_bigint(&limbs2, moduli, q_top);
        let c_rescaled2 = Ciphertext::crt_reconstruct_centered(&rescaled_limbs2, new_moduli);
        println!("Rescaled value: {}", c_rescaled2);
        println!("Expected: {}", k2);

        let diff2 = (&c_rescaled2 - BigInt::from(k2)).abs();
        assert!(diff2 <= BigInt::from(1), "Rescale error too large: {} (expected {})", c_rescaled2, k2);

        println!("\nH9.1 PASSED: Rescale correctly divides by q_top with rounding");
    }

    /// H9.2: UNIT TEST - Plain multiply sanity check (all 2s × all 3s = all 6s)
    ///
    /// Full integration test: encrypt 2.0 in all slots, multiply by plaintext 3.0,
    /// decrypt and verify result is 6.0 with high precision (error < 0.01).
    #[test]
    fn test_h9_2_plain_multiply_sanity() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let mut ckks_ctx = CkksContext::new(params.clone());

        let num_slots = params.n / 2;

        println!("\n=== H9.2: Plain Multiply Sanity (2 × 3 = 6) ===");

        // Encrypt all 2.0s
        let values = vec![2.0; num_slots];
        let pt_x = Plaintext::encode(&values, params.scale, &params);
        let ct = ckks_ctx.encrypt(&pt_x, &pk);

        println!("Encrypted: all 2.0s at level {}, scale {}", ct.level, ct.scale);

        // Create plaintext multiplier: all 3.0s
        let multiplier = vec![3.0; num_slots];
        let pt_c = Plaintext::encode(&multiplier, params.scale, &params);

        println!("Plaintext: all 3.0s at level {}, scale {}", pt_c.level, pt_c.scale);

        // Multiply
        let ct_result = ct.multiply_plain(&pt_c, &ckks_ctx);

        println!("After multiply_plain: level {}, scale {}", ct_result.level, ct_result.scale);

        // Decrypt and decode
        let pt_result = ckks_ctx.decrypt(&ct_result, &sk);
        let output = pt_result.decode(&params);

        println!("Output (first 8 slots): {:?}", &output[..8.min(output.len())]);

        // Verify all slots are close to 6.0
        let expected = 6.0;
        let mut max_error = 0.0;
        let mut sum = 0.0;

        for i in 0..num_slots {
            let error = (output[i] - expected).abs();
            if error > max_error {
                max_error = error;
            }
            sum += output[i];
        }

        let mean = sum / (num_slots as f64);

        println!("Expected: {}", expected);
        println!("Mean output: {}", mean);
        println!("Max error: {}", max_error);

        // With proper rescale, error should be tiny (< 0.01)
        assert!(max_error < 0.01, "Plain multiply failed: max error {} exceeds 0.01", max_error);

        let mean_error = (mean - expected).abs();
        assert!(mean_error < 0.001, "Mean error {} exceeds 0.001", mean_error);

        println!("\nH9.2 PASSED: Plain multiply produces correct results (error < 0.01)");
    }

    /// H9.3: SIMPLIFIED TEST - No encryption, just plaintext arithmetic
    ///
    /// encode(2) * encode(3) → rescale → decode should give 6
    #[test]
    fn test_h9_3_plaintext_only_with_rescale() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ckks_ctx = CkksContext::new(params.clone());
        let num_slots = params.n / 2;

        println!("\n=== H9.3: Plaintext-Only Arithmetic (No Encryption) ===");

        // Encode 2.0
        let x_vals = vec![2.0; num_slots];
        let pt_x = Plaintext::encode(&x_vals, params.scale, &params);
        println!("PT_x: level={}, scale={}, coeffs[0]={}", pt_x.level, pt_x.scale, pt_x.coeffs[0].values[0]);

        // Encode 3.0
        let c_vals = vec![3.0; num_slots];
        let pt_c = Plaintext::encode(&c_vals, params.scale, &params);
        println!("PT_c: level={}, scale={}, coeffs[0]={}", pt_c.level, pt_c.scale, pt_c.coeffs[0].values[0]);

        // Manually multiply using NTT (same as multiply_plain does)
        let moduli = &params.moduli[..=pt_x.level];
        let product_coeffs = ckks_ctx.multiply_polys_ntt(&pt_x.coeffs, &pt_c.coeffs, moduli);

        println!("After NTT multiply: coeffs[0]={}", product_coeffs[0].values[0]);

        // Check the full value before rescale
        let c0_before = Ciphertext::crt_reconstruct_centered(&product_coeffs[0].values, moduli);
        println!("Product coeffs[0] full value: {}", c0_before);
        println!("Expected: ~{}", 6.0 * params.scale * params.scale);

        // Create a "fake" ciphertext to use rescale
        let fake_ct = Ciphertext::new(product_coeffs.clone(), product_coeffs.clone(), pt_x.level, 0.0);

        // Rescale
        let pre_rescale_scale = pt_x.scale * pt_c.scale;
        let rescaled_ct = fake_ct.rescale_to_next_with_scale(&ckks_ctx, pre_rescale_scale);

        println!("After rescale: level={}, scale={}", rescaled_ct.level, rescaled_ct.scale);

        // Create plaintext from rescaled coefficients
        let pt_result = Plaintext::new(rescaled_ct.c0.clone(), rescaled_ct.scale, rescaled_ct.level);
        let output = pt_result.decode(&params);

        println!("Output (first 8 slots): {:?}", &output[..8.min(output.len())]);

        let expected = 6.0;
        let mut max_error = 0.0;
        for i in 0..num_slots {
            let error = (output[i] - expected).abs();
            if error > max_error {
                max_error = error;
            }
        }

        println!("Expected: {}", expected);
        println!("Max error: {}", max_error);

        assert!(max_error < 0.01, "Plaintext-only multiply failed: max error {} exceeds 0.01", max_error);

        println!("\nH9.3 PASSED: Plaintext-only multiply works correctly");
    }
}

