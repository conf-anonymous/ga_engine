//! Rotation Keys for CKKS Bootstrapping
//!
//! Rotation keys enable homomorphic rotation of ciphertext slots,
//! which is essential for CoeffToSlot and SlotToCoeff transformations.
//!
//! ## Galois Automorphisms
//!
//! A rotation by k slots corresponds to the Galois automorphism σ_g where:
//! - g = 5^k mod 2N (for rotation by k)
//! - σ_g(X^i) = X^(g·i mod 2N)
//!
//! ## Key Switching
//!
//! After applying σ_g, we need to key-switch from s(X^g) back to s(X).
//! This requires a key-switching key that encrypts s(X^g) under s(X).

use crate::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use std::collections::HashMap;

/// Rotation keys for CKKS bootstrapping
///
/// Contains key-switching keys for all required Galois automorphisms.
#[derive(Clone, Debug)]
pub struct RotationKeys {
    /// Map from Galois element g to key-switching key
    /// - Key: g (Galois element)
    /// - Value: (rlk0, rlk1) where rlk encrypts s(X^g) under s(X)
    pub keys: HashMap<usize, RotationKey>,

    /// Ring dimension
    pub n: usize,

    /// Current level
    pub level: usize,
}

/// A single rotation key (key-switching key for one Galois automorphism)
#[derive(Clone, Debug)]
pub struct RotationKey {
    /// Galois element g
    pub galois_element: usize,

    /// First component (encrypts B^t · s(X^g))
    pub rlk0: Vec<Vec<RnsRepresentation>>,

    /// Second component (uniform randomness)
    pub rlk1: Vec<Vec<RnsRepresentation>>,

    /// Gadget base (e.g., w=20 means base B=2^20)
    pub base_w: u32,
}

impl RotationKeys {
    /// Create new rotation keys
    pub fn new(n: usize, level: usize) -> Self {
        Self {
            keys: HashMap::new(),
            n,
            level,
        }
    }

    /// Add a rotation key for a specific Galois element
    pub fn add_key(&mut self, galois_element: usize, key: RotationKey) {
        self.keys.insert(galois_element, key);
    }

    /// Get rotation key for a specific Galois element
    pub fn get_key(&self, galois_element: usize) -> Option<&RotationKey> {
        self.keys.get(&galois_element)
    }

    /// Check if rotation key exists for a Galois element
    pub fn has_key(&self, galois_element: usize) -> bool {
        self.keys.contains_key(&galois_element)
    }

    /// Get number of rotation keys
    pub fn num_keys(&self) -> usize {
        self.keys.len()
    }
}

/// Compute Galois element for rotation by k slots
///
/// For rotation by k positions: g = 5^k mod 2N
///
/// # Arguments
///
/// * `k` - Number of slots to rotate (can be positive or negative)
/// * `n` - Ring dimension
///
/// # Returns
///
/// Galois element g for the rotation
///
/// # Example
///
/// ```ignore
/// let g = galois_element_for_rotation(1, 8192);  // Rotation by 1 slot
/// let g_inv = galois_element_for_rotation(-1, 8192);  // Rotation by -1 slot
/// ```
pub fn galois_element_for_rotation(k: i32, n: usize) -> usize {
    let two_n = 2 * n;

    // Normalize k to [0, N)
    let k_normalized = if k >= 0 {
        (k as usize) % n
    } else {
        let abs_k = (-k) as usize % n;
        if abs_k == 0 {
            0
        } else {
            n - abs_k
        }
    };

    // Compute 5^k mod 2N using modular exponentiation
    mod_pow(5, k_normalized, two_n)
}

/// Compute base^exp mod modulus using square-and-multiply
fn mod_pow(base: usize, mut exp: usize, modulus: usize) -> usize {
    if modulus == 1 {
        return 0;
    }

    let mut result = 1;
    let mut base = base % modulus;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }

    result
}

/// Apply Galois automorphism to a polynomial (coefficient permutation)
///
/// Given polynomial p(X) = Σ p_i X^i, computes p(X^g) = Σ p_i X^(g·i mod 2N)
///
/// # Arguments
///
/// * `poly` - Input polynomial coefficients
/// * `galois_element` - Galois element g
/// * `n` - Ring dimension
///
/// # Returns
///
/// Permuted polynomial coefficients
pub fn apply_galois_automorphism(
    poly: &[RnsRepresentation],
    galois_element: usize,
    n: usize,
) -> Vec<RnsRepresentation> {
    let two_n = 2 * n;

    // Initialize result to zero (not poly[0]!)
    // The Galois automorphism is a permutation, so every position will be written exactly once
    let moduli = &poly[0].moduli;
    let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.to_vec()); n];

    for i in 0..n {
        // Compute new index: (g * i) mod 2N
        let new_idx = (galois_element * i) % two_n;

        if new_idx < n {
            // X^i → X^new_idx (positive)
            result[new_idx] = poly[i].clone();
        } else {
            // X^i → -X^(new_idx - N) (negative, wrap around)
            result[new_idx - n] = poly[i].negate();
        }
    }

    result
}

/// Generate rotation keys for a list of rotation amounts
///
/// # Arguments
///
/// * `rotations` - List of rotation amounts (e.g., [1, 2, 4, 8, ...])
/// * `secret_key` - Secret key
/// * `params` - FHE parameters
///
/// # Returns
///
/// Rotation keys for all requested rotations
///
/// # Implementation
///
/// For each rotation k:
/// 1. Compute Galois element g = 5^k mod 2N
/// 2. Apply Galois automorphism to secret key: s(X^g)
/// 3. Generate key-switching key that encrypts s(X^g) under s(X)
/// 4. Use gadget decomposition (same structure as evaluation key)
///
/// The rotation key allows transforming:
///   (c0, c1) where Dec = c0 + c1·s
/// into:
///   (c0', c1') where Dec = c0' + c1'·s = c0(X^g) + c1(X^g)·s(X^g)
///
/// After applying Galois automorphism to ciphertext, we need to key-switch
/// from s(X^g) back to s(X).
pub fn generate_rotation_keys(
    rotations: &[i32],
    secret_key: &SecretKey,
    params: &CliffordFHEParams,
) -> RotationKeys {
    let n = params.n;
    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();

    let mut rotation_keys = RotationKeys::new(n, level);

    println!("Generating rotation keys for {} rotations...", rotations.len());

    for &k in rotations {
        let g = galois_element_for_rotation(k, n);
        println!("  Rotation {}: Galois element g = {} (5^{} mod {})",
                 k, g, k, 2 * n);

        // Skip if we already have a key for this Galois element
        // (can happen when ±k map to same g, e.g., rotation by ±N/2)
        if rotation_keys.has_key(g) {
            println!("    (skipping duplicate Galois element)");
            continue;
        }

        // Generate key-switching key for this rotation
        let rotation_key = generate_single_rotation_key(
            g,
            secret_key,
            &moduli,
            params,
        );

        rotation_keys.add_key(g, rotation_key);
    }

    println!("  ✓ Generated {} unique rotation keys (from {} rotations)",
             rotation_keys.num_keys(), rotations.len());

    rotation_keys
}

/// Generate a single rotation key for Galois element g
///
/// This creates a key-switching key that encrypts s(X^g) under s(X).
///
/// Similar to evaluation key generation, we use gadget decomposition:
/// - rlk0[t] encrypts B^t · s(X^g) under s(X)
/// - rlk1[t] = uniform randomness
///
/// Where B = 2^w (base for gadget decomposition)
fn generate_single_rotation_key(
    galois_element: usize,
    secret_key: &SecretKey,
    moduli: &[u64],
    params: &CliffordFHEParams,
) -> RotationKey {
    use rand::{thread_rng, Rng};
    use rand_distr::{Normal, Distribution};

    let base_w = 20u32;  // Use base 2^20 for gadget decomposition
    let n = params.n;

    // Step 1: Apply Galois automorphism to secret key: s(X) → s(X^g)
    let s_automorphism = apply_galois_automorphism(&secret_key.coeffs, galois_element, n);

    // Step 2: Compute number of digits needed for gadget decomposition
    use num_bigint::BigInt;
    let q_prod_big: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();
    let q_bits = q_prod_big.bits() as u32;
    let num_digits = ((q_bits + base_w - 1) / base_w) as usize;

    // Step 3: Precompute B^t mod q for each prime and each digit
    let base = 1u64 << base_w;
    let mut bpow_t_mod_q = vec![vec![0u64; moduli.len()]; num_digits];
    for (j, &q) in moduli.iter().enumerate() {
        let q_u128 = q as u128;
        let mut p = 1u128;
        for t in 0..num_digits {
            bpow_t_mod_q[t][j] = (p % q_u128) as u64;
            p = (p * (base as u128)) % q_u128;
        }
    }

    // Step 4: Generate key-switching key digits
    let mut rlk0 = Vec::with_capacity(num_digits);
    let mut rlk1 = Vec::with_capacity(num_digits);

    let mut rng = thread_rng();
    let normal = Normal::new(0.0, params.error_std)
        .expect("Invalid normal distribution parameters");

    for t in 0..num_digits {
        // Compute B^t * s(X^g) using precomputed B^t mod q
        let bt_s_automorphism: Vec<RnsRepresentation> = s_automorphism
            .iter()
            .map(|rns| {
                let values: Vec<u64> = rns.values.iter().enumerate()
                    .map(|(j, &val)| {
                        let q = moduli[j];
                        let bt_mod_q = bpow_t_mod_q[t][j];
                        // Compute val * B^t mod q
                        ((val as u128) * (bt_mod_q as u128) % (q as u128)) as u64
                    })
                    .collect();
                RnsRepresentation::new(values, moduli.to_vec())
            })
            .collect();

        // Sample uniform a_t
        let a_t: Vec<RnsRepresentation> = (0..n)
            .map(|_| {
                let values: Vec<u64> = moduli.iter().map(|&q| rng.gen_range(0..q)).collect();
                RnsRepresentation::new(values, moduli.to_vec())
            })
            .collect();

        // Sample error e_t from Gaussian distribution
        let e_t: Vec<RnsRepresentation> = (0..n)
            .map(|_| {
                let error_val = normal.sample(&mut rng).round() as i64;
                let values: Vec<u64> = moduli
                    .iter()
                    .map(|&q| {
                        if error_val >= 0 {
                            (error_val as u64) % q
                        } else {
                            let abs_val = (-error_val) as u64;
                            let remainder = abs_val % q;
                            if remainder == 0 {
                                0
                            } else {
                                q - remainder
                            }
                        }
                    })
                    .collect();
                RnsRepresentation::new(values, moduli.to_vec())
            })
            .collect();

        // Compute rlk0[t] = -B^t*s(X^g) + a_t*s + e_t
        // This ensures: rlk0[t] - rlk1[t]*s = -B^t*s(X^g) + e_t
        let a_t_times_s = multiply_polynomials_ntt(&a_t, &secret_key.coeffs, moduli, n);
        let neg_bt_s_auto = negate_polynomial(&bt_s_automorphism, moduli);
        let temp = add_polynomials(&neg_bt_s_auto, &a_t_times_s);
        let b_t: Vec<RnsRepresentation> = temp
            .iter()
            .zip(&e_t)
            .map(|(x, y)| x.add(y))
            .collect();

        rlk0.push(b_t);
        rlk1.push(a_t);
    }

    RotationKey {
        galois_element,
        rlk0,
        rlk1,
        base_w,
    }
}

/// Get required rotations for CoeffToSlot/SlotToCoeff
///
/// For FFT-like transformations, we need rotations by powers of 2:
/// - Rotations: ±1, ±2, ±4, ±8, ..., ±N/2
///
/// # Arguments
///
/// * `n` - Ring dimension
///
/// # Returns
///
/// List of rotation amounts needed for bootstrap
pub fn required_rotations_for_bootstrap(n: usize) -> Vec<i32> {
    let mut rotations = Vec::new();

    // Powers of 2: 1, 2, 4, 8, ..., N/2
    let mut k = 1;
    while k < n {
        rotations.push(k as i32);
        rotations.push(-(k as i32));
        k *= 2;
    }

    rotations
}

/// Helper: Multiply two polynomials using NTT (negacyclic convolution mod x^n + 1)
fn multiply_polynomials_ntt(
    a: &[RnsRepresentation],
    b: &[RnsRepresentation],
    moduli: &[u64],
    n: usize,
) -> Vec<RnsRepresentation> {
    use crate::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;

    assert_eq!(a.len(), n);
    assert_eq!(b.len(), n);

    let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.to_vec()); n];

    // Multiply for each prime separately using NTT
    for (prime_idx, &q) in moduli.iter().enumerate() {
        let ntt_ctx = NttContext::new(n, q);

        // Extract coefficients for this prime
        let a_mod_q: Vec<u64> = a.iter().map(|rns| rns.values[prime_idx]).collect();
        let b_mod_q: Vec<u64> = b.iter().map(|rns| rns.values[prime_idx]).collect();

        // Multiply using NTT (negacyclic convolution)
        let product_mod_q = ntt_ctx.multiply_polynomials(&a_mod_q, &b_mod_q);

        // Store results
        for (i, &val) in product_mod_q.iter().enumerate() {
            result[i].values[prime_idx] = val;
        }
    }

    result
}

/// Helper: Negate polynomial (compute -a mod q for each coefficient)
fn negate_polynomial(
    a: &[RnsRepresentation],
    moduli: &[u64],
) -> Vec<RnsRepresentation> {
    a.iter()
        .map(|rns| {
            let negated_values: Vec<u64> = rns
                .values
                .iter()
                .zip(moduli)
                .map(|(&val, &q)| if val == 0 { 0 } else { q - val })
                .collect();
            RnsRepresentation::new(negated_values, moduli.to_vec())
        })
        .collect()
}

/// Helper: Add two polynomials
fn add_polynomials(
    a: &[RnsRepresentation],
    b: &[RnsRepresentation],
) -> Vec<RnsRepresentation> {
    a.iter().zip(b).map(|(x, y)| x.add(y)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_galois_element_for_rotation() {
        let n = 8;  // Small ring for testing

        // Rotation by 0 should give g = 1
        let g0 = galois_element_for_rotation(0, n);
        assert_eq!(g0, 1);

        // Rotation by 1: g = 5^1 mod 16 = 5
        let g1 = galois_element_for_rotation(1, n);
        assert_eq!(g1, 5);

        // Rotation by 2: g = 5^2 mod 16 = 25 mod 16 = 9
        let g2 = galois_element_for_rotation(2, n);
        assert_eq!(g2, 9);

        // Rotation by -1 should be inverse of rotation by 1
        let g_neg1 = galois_element_for_rotation(-1, n);
        // For N=8: rotation by -1 = rotation by 7
        // 5^7 mod 16 = 78125 mod 16 = 13
        assert_eq!(g_neg1, 13);
    }

    #[test]
    fn test_galois_element_for_large_n() {
        let n = 8192;

        let g1 = galois_element_for_rotation(1, n);
        assert_eq!(g1, 5);  // 5^1 mod 16384 = 5

        let g2 = galois_element_for_rotation(2, n);
        assert_eq!(g2, 25);  // 5^2 mod 16384 = 25
    }

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(5, 0, 16), 1);   // 5^0 = 1
        assert_eq!(mod_pow(5, 1, 16), 5);   // 5^1 = 5
        assert_eq!(mod_pow(5, 2, 16), 9);   // 5^2 = 25 mod 16 = 9
        assert_eq!(mod_pow(5, 3, 16), 13);  // 5^3 = 125 mod 16 = 13
    }

    #[test]
    fn test_required_rotations_for_bootstrap() {
        let n = 8;
        let rotations = required_rotations_for_bootstrap(n);

        // For N=8, should have: ±1, ±2, ±4
        assert!(rotations.contains(&1));
        assert!(rotations.contains(&-1));
        assert!(rotations.contains(&2));
        assert!(rotations.contains(&-2));
        assert!(rotations.contains(&4));
        assert!(rotations.contains(&-4));

        // Should have 2*log2(N) = 2*3 = 6 rotations
        assert_eq!(rotations.len(), 6);
    }

    #[test]
    fn test_required_rotations_for_large_n() {
        let n = 8192;
        let rotations = required_rotations_for_bootstrap(n);

        // Should have: ±1, ±2, ±4, ..., ±4096
        // That's 2*log2(8192) = 2*13 = 26 rotations
        assert_eq!(rotations.len(), 26);

        // Check specific values
        assert!(rotations.contains(&1));
        assert!(rotations.contains(&-1));
        assert!(rotations.contains(&4096));
        assert!(rotations.contains(&-4096));
    }

    #[test]
    fn test_rotation_keys_creation() {
        use crate::clifford_fhe_v2::params::CliffordFHEParams;
        use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

        // Use smaller params for faster testing
        let params = CliffordFHEParams::new_128bit();
        let key_ctx = KeyContext::new(params.clone());
        let (_, secret_key, _) = key_ctx.keygen();

        // Generate rotation keys for small set
        let rotations = vec![1, 2, 4];
        let rotation_keys = generate_rotation_keys(&rotations, &secret_key, &params);

        // Should have 3 keys
        assert_eq!(rotation_keys.num_keys(), 3);

        // Check keys exist for expected Galois elements
        let g1 = galois_element_for_rotation(1, params.n);
        assert!(rotation_keys.has_key(g1));

        // Verify rotation key structure
        let rot_key = rotation_keys.get_key(g1).unwrap();
        assert_eq!(rot_key.galois_element, g1);
        assert_eq!(rot_key.base_w, 20);
        assert!(rot_key.rlk0.len() > 0, "rlk0 should have digits");
        assert_eq!(rot_key.rlk0.len(), rot_key.rlk1.len(), "rlk0 and rlk1 should have same number of digits");

        // Each digit should have N coefficients
        for digit in &rot_key.rlk0 {
            assert_eq!(digit.len(), params.n, "Each rlk0 digit should have N coefficients");
        }
        for digit in &rot_key.rlk1 {
            assert_eq!(digit.len(), params.n, "Each rlk1 digit should have N coefficients");
        }
    }
}
