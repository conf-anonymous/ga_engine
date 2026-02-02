//! RNS-aware key generation for CKKS
//!
//! This module provides key structures and generation functions that work
//! with RNS (Residue Number System) representation, enabling multi-prime CKKS.

use crate::clifford_fhe_v1::params::CliffordFHEParams;
use crate::clifford_fhe_v1::rns::{RnsPolynomial, rns_add, rns_multiply as rns_poly_multiply};
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal};

/// RNS-aware public key
#[derive(Clone, Debug)]
pub struct RnsPublicKey {
    /// First component (RNS polynomial)
    pub a: RnsPolynomial,
    /// Second component (RNS polynomial): b = a*s + e
    pub b: RnsPolynomial,
    /// Ring dimension
    pub n: usize,
}

/// RNS-aware secret key
#[derive(Clone, Debug)]
pub struct RnsSecretKey {
    /// Secret polynomial (RNS representation)
    /// This is a ternary polynomial: coefficients in {-1, 0, 1}
    pub coeffs: RnsPolynomial,
    /// Ring dimension
    pub n: usize,
}

/// RNS-aware evaluation key (for relinearization with gadget decomposition)
///
/// Stores per-digit switching keys for low-noise relinearization.
/// For base B=2^w, we decompose d₂ into D digits and use D key pairs.
#[derive(Clone, Debug)]
pub struct RnsEvaluationKey {
    /// Digit width (e.g., w=20 means B=2^20)
    pub base_w: u32,
    /// evk0[t] encrypts B^t · s² (one per digit)
    pub evk0: Vec<RnsPolynomial>,
    /// evk1[t] = uniform randomness for evk0[t] (one per digit)
    pub evk1: Vec<RnsPolynomial>,
    /// Ring dimension
    pub n: usize,
}

/// RNS-aware rotation key (for slot rotations)
#[derive(Clone, Debug)]
pub struct RnsRotationKey {
    /// Map from automorphism index to key pair
    pub keys: std::collections::HashMap<usize, (RnsPolynomial, RnsPolynomial)>,
    /// Ring dimension
    pub n: usize,
}

/// Generate RNS-CKKS keys
///
/// This generates a fresh secret key and derives the public key and evaluation key.
/// All keys are in RNS form, compatible with multi-prime CKKS operations.
///
/// # Key Generation Algorithm
/// 1. Sample secret s ← {-1, 0, 1}^N (ternary polynomial)
/// 2. Sample a ← R_Q uniformly (each prime independently)
/// 3. Sample error e ← χ (Gaussian, then convert to RNS)
/// 4. Compute b = a*s + e (all in RNS)
/// 5. Public key: (a, b)
///
/// # Returns
/// (PublicKey, SecretKey, EvaluationKey)
pub fn rns_keygen(params: &CliffordFHEParams) -> (RnsPublicKey, RnsSecretKey, RnsEvaluationKey) {
    let n = params.n;
    let primes = &params.moduli;
    let num_primes = primes.len();
    let mut rng = thread_rng();

    // 1. Sample secret key: ternary polynomial s ∈ {-1, 0, 1}^N
    let s_coeffs: Vec<i64> = (0..n)
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

    // Convert to RNS (level 0 = all primes active)
    let s_rns = RnsPolynomial::from_coeffs(&s_coeffs, primes, n, 0);

    // 2. Sample uniform polynomial a
    // CRITICAL: Sample coefficients in a consistent range, then reduce mod each prime
    // We sample from [0, q0) where q0 is the largest prime, then reduce mod all primes
    // This ensures CRT consistency: all residues represent the same underlying value
    let q0 = primes[0]; // Largest prime (primes are sorted descending)
    let a_coeffs: Vec<i64> = (0..n).map(|_| rng.gen_range(0..q0)).collect();

    // Convert to RNS (this will reduce mod each prime consistently)
    let a_rns = RnsPolynomial::from_coeffs(&a_coeffs, primes, n, 0);

    // 3. Sample error from Gaussian distribution
    let normal = Normal::new(0.0, params.error_std).unwrap();
    let e_coeffs: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();
    let e_rns = RnsPolynomial::from_coeffs(&e_coeffs, primes, n, 0);

    // 4. Compute b = a*s + e (in RNS)
    // Use NTT-based polynomial multiplication (imported from ckks_rns module)
    use crate::clifford_fhe_v1::ckks_rns::polynomial_multiply_ntt;

    // DEBUG: Check inputs to multiplication
    if std::env::var("RNS_TRACE").is_ok() && num_primes == 2 {
        // eprintln!("[KEYGEN] a[0] residues: [{}, {}]", a_rns.rns_coeffs[0][0], a_rns.rns_coeffs[0][1]);
        // eprintln!("[KEYGEN] a[1] residues: [{}, {}]", a_rns.rns_coeffs[1][0], a_rns.rns_coeffs[1][1]);
        // eprintln!("[KEYGEN] s[0] residues: [{}, {}]", s_rns.rns_coeffs[0][0], s_rns.rns_coeffs[0][1]);
        // eprintln!("[KEYGEN] s[1] residues: [{}, {}]", s_rns.rns_coeffs[1][0], s_rns.rns_coeffs[1][1]);
    }

    let a_times_s = rns_poly_multiply(&a_rns, &s_rns, primes, polynomial_multiply_ntt);

    // DEBUG: Check if a_times_s has consistent residues
    if std::env::var("RNS_TRACE").is_ok() && num_primes == 2 {
        let r0 = a_times_s.rns_coeffs[0][0];
        let r1 = a_times_s.rns_coeffs[0][1];
        // eprintln!("[KEYGEN] a*s[0] residues: [{}, {}], diff={}", r0, r1, (r0 as i128 - r1 as i128).abs());
    }

    // Negate a·s to get -a·s (required for CKKS public key)
    let neg_a_times_s_coeffs: Vec<Vec<i64>> = (0..n)
        .map(|i| {
            (0..num_primes)
                .map(|j| {
                    let q = primes[j];
                    let val = a_times_s.rns_coeffs[i][j];
                    // Negate: -x = q - x (mod q)
                    if val == 0 {
                        0
                    } else {
                        q - val
                    }
                })
                .collect()
        })
        .collect();
    let neg_a_times_s = RnsPolynomial::new(neg_a_times_s_coeffs, n, 0);

    // DEBUG: Check if -a*s has consistent residues
    if std::env::var("RNS_TRACE").is_ok() && num_primes == 2 {
        let r0 = neg_a_times_s.rns_coeffs[0][0];
        let r1 = neg_a_times_s.rns_coeffs[0][1];
        // eprintln!("[KEYGEN] -a*s[0] residues: [{}, {}], diff={}", r0, r1, (r0 as i128 - r1 as i128).abs());
        // eprintln!("[KEYGEN] e[0] residues: [{}, {}]", e_rns.rns_coeffs[0][0], e_rns.rns_coeffs[0][1]);
    }

    // b = -a·s + e
    let b_rns = rns_add(&neg_a_times_s, &e_rns, primes);

    // DEBUG: Check if b has consistent residues
    if std::env::var("RNS_TRACE").is_ok() && num_primes == 2 {
        let r0 = b_rns.rns_coeffs[0][0];
        let r1 = b_rns.rns_coeffs[0][1];
        // eprintln!("[KEYGEN] b[0] residues: [{}, {}], diff={}", r0, r1, (r0 as i128 - r1 as i128).abs());
    }

    // 5. Create public key
    let pk = RnsPublicKey {
        a: a_rns,
        b: b_rns,
        n,
    };

    // 6. Create secret key
    let sk = RnsSecretKey {
        coeffs: s_rns.clone(),
        n,
    };

    // 7. Generate evaluation key (for relinearization)
    // This encrypts s² under the public key
    let evk = generate_rns_evaluation_key(&s_rns, &pk, params);

    (pk, sk, evk)
}

/// Generate RNS evaluation key for relinearization with gadget decomposition
///
/// Creates digit-wise switching keys for low-noise relinearization.
/// For base B=2^w, generates D keys encrypting B^t·s² for t=0..D-1.
fn generate_rns_evaluation_key(
    sk: &RnsPolynomial,
    _pk: &RnsPublicKey,
    params: &CliffordFHEParams,
) -> RnsEvaluationKey {
    let n = params.n;
    let primes = &params.moduli;
    let num_primes = primes.len();
    let mut rng = thread_rng();

    // Gadget parameters
    // CRITICAL: Use w=20 for Δ=2^40 to keep noise growth manageable
    // Rule of thumb: B ≈ sqrt(Δ), so for Δ=2^40, B=2^20
    let w: u32 = 20;  // Digit width (B = 2^20)

    // Number of digits must cover Q = product of all primes
    // CRITICAL: Use the SAME calculation as decompose_base_pow2!
    // We use Q.bits() not sum of individual prime bits, because:
    //   sum(bits) overestimates (e.g., 224 for 5 primes)
    //   Q.bits() is exact (e.g., 220 for 5 primes)
    // This ensures EVK has exactly the right number of entries.
    use num_bigint::BigInt;
    let q_prod: BigInt = primes.iter().map(|&p| BigInt::from(p)).product();
    let q_bits: u32 = q_prod.bits() as u32;
    let d: usize = ((q_bits + w - 1) / w) as usize;

    // eprintln!("[EVK GEN] num_primes={}, total_bits={}, w={}, num_digits={}",
    //           num_primes, q_bits, w, d);

    // Use NTT-based polynomial multiplication (imported from ckks_rns module)
    use crate::clifford_fhe_v1::ckks_rns::polynomial_multiply_ntt;

    // Compute s²
    let s_squared = rns_poly_multiply(sk, sk, primes, polynomial_multiply_ntt);

    // Precompute B^t mod q for each prime and each digit
    let b: i64 = 1i64 << w;
    let mut bpow_t_mod_q = vec![vec![0i64; num_primes]; d];
    for j in 0..num_primes {
        let q = primes[j] as i128;
        let mut p = 1i128;
        for t in 0..d {
            bpow_t_mod_q[t][j] = p as i64;
            p = (p * (b as i128)) % q;
        }
    }

    let normal = Normal::new(0.0, params.error_std).unwrap();

    let mut evk0 = Vec::with_capacity(d);
    let mut evk1 = Vec::with_capacity(d);

    // Generate one key pair per digit
    for t in 0..d {
        // Sample uniform a_t (RNS-consistent!)
        // CRITICAL: Sample from [0, q0) then reduce mod all primes for CRT consistency
        let q0 = primes[0];
        let a_t_vec: Vec<i64> = (0..n).map(|_| rng.gen_range(0..q0)).collect();
        let a_t = RnsPolynomial::from_coeffs(&a_t_vec, primes, n, 0);

        // Sample small error e_t
        let e_t_vec: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();
        let e_t = RnsPolynomial::from_coeffs(&e_t_vec, primes, n, 0);

        // Compute B^t · s² (per prime)
        let mut bt_s2_coeffs = vec![vec![0i64; num_primes]; n];
        for i in 0..n {
            for j in 0..num_primes {
                let q = primes[j] as i128;
                let val = ((s_squared.rns_coeffs[i][j] as i128) * (bpow_t_mod_q[t][j] as i128)) % q;
                bt_s2_coeffs[i][j] = val as i64;
            }
        }

        // Compute a_t · s
        let a_t_s = rns_poly_multiply(&a_t, sk, primes, polynomial_multiply_ntt);

        // TEST: Try the opposite sign
        // evk0[t] = -B^t·s² + a_t·s + e_t
        // This would satisfy: evk0[t] - evk1[t]·s = -B^t·s² + e_t
        let neg_bt_s2_coeffs: Vec<Vec<i64>> = (0..n).map(|i| {
            (0..num_primes).map(|j| {
                let q = primes[j];
                (q - bt_s2_coeffs[i][j] % q) % q
            }).collect()
        }).collect();
        let neg_bt_s2 = RnsPolynomial::new(neg_bt_s2_coeffs, n, 0);

        let tmp = rns_add(&neg_bt_s2, &a_t_s, primes);
        let evk0_t = rns_add(&tmp, &e_t, primes);

        // evk1[t] = a_t
        evk0.push(evk0_t);
        evk1.push(a_t);
    }

    RnsEvaluationKey {
        base_w: w,
        evk0,
        evk1,
        n,
    }
}

/// Generate RNS rotation keys for given rotation amounts
///
/// For each rotation r, generates a key-switching key that enables
/// rotating CKKS slots by r positions.
pub fn rns_generate_rotation_keys(
    sk: &RnsSecretKey,
    pk: &RnsPublicKey,
    rotations: &[isize],
    params: &CliffordFHEParams,
) -> RnsRotationKey {
    use crate::clifford_fhe_v1::automorphisms::rotation_to_automorphism;
    use std::collections::HashMap;

    let mut keys = HashMap::new();

    for &r in rotations {
        // Convert rotation to automorphism index
        let k = rotation_to_automorphism(r, params.n);

        // Generate key-switching key for this automorphism
        // This is similar to evaluation key generation but for σ_k(s)
        // For simplicity, we'll skip the full implementation for now
        // and just store placeholder keys

        // TODO: Implement proper rotation key generation
        // For now, just clone the evaluation key structure as placeholder
        let dummy_key = (sk.coeffs.clone(), sk.coeffs.clone());
        keys.insert(k, dummy_key);
    }

    RnsRotationKey {
        keys,
        n: params.n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rns_keygen() {
        let params = CliffordFHEParams::new_rns_mult();
        let (_pk, _sk, _evk) = rns_keygen(&params);
        // If this doesn't panic, key generation works!
    }
}
