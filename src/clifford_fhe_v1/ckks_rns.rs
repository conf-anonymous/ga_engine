//! RNS-CKKS encryption scheme for Clifford algebra
//!
//! This is the RNS (Residue Number System) version of CKKS that enables
//! proper homomorphic multiplication with rescaling.
//!
//! Key differences from single-modulus CKKS:
//! - Coefficients stored as RNS tuples instead of single i64
//! - Rescaling drops a prime from the modulus chain
//! - Supports larger effective moduli (Q = q₀ · q₁ · ... can be 2^200+)

use crate::clifford_fhe_v1::keys_rns::{RnsPublicKey, RnsSecretKey, RnsEvaluationKey};
use crate::clifford_fhe_v1::params::CliffordFHEParams;
use crate::clifford_fhe_v1::rns::{RnsPolynomial, rns_add, rns_sub, rns_multiply as rns_poly_multiply, rns_rescale_exact, precompute_rescale_inv, decompose_base_pow2};

// ============================================================================
// NTT (Number Theoretic Transform) helpers for negacyclic convolution
// ============================================================================

#[inline(always)]
fn mod_add_u64(a: u64, b: u64, q: u64) -> u64 {
    let s = a.wrapping_add(b);
    if s >= q { s - q } else { s }
}

#[inline(always)]
fn mod_sub_u64(a: u64, b: u64, q: u64) -> u64 {
    if a >= b { a - b } else { a + q - b }
}

#[inline(always)]
fn mod_mul_u64(a: u64, b: u64, q: u64) -> u64 {
    // 128-bit exact multiply then reduce
    let p = (a as u128) * (b as u128);
    (p % (q as u128)) as u64
}

fn mod_pow_u64(mut base: u64, mut exp: u64, q: u64) -> u64 {
    let mut acc = 1u64;
    while exp > 0 {
        if (exp & 1) == 1 { acc = mod_mul_u64(acc, base, q); }
        base = mod_mul_u64(base, base, q);
        exp >>= 1;
    }
    acc
}

/// Find a primitive root of Z_q^* (q prime). Brute force is fine for a few primes.
fn primitive_root(q: u64) -> u64 {
    let phi = q - 1;

    // Get prime factors of phi
    let prime_factors = prime_factors_of(phi);

    // Test each candidate g
    for g in 2..q {
        let mut is_primitive = true;

        // g is a primitive root iff g^(phi/p) ≠ 1 for all prime factors p of phi
        for &p in &prime_factors {
            if mod_pow_u64(g, phi / p, q) == 1 {
                is_primitive = false;
                break;
            }
        }

        if is_primitive {
            return g;
        }
    }
    unreachable!("no primitive root found (unexpected for prime q)");
}

/// Find prime factors of n (simple trial division, sufficient for our use case)
fn prime_factors_of(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();

    // Factor out 2
    if n % 2 == 0 {
        factors.push(2);
        while n % 2 == 0 {
            n /= 2;
        }
    }

    // Factor out odd primes
    let mut p = 3u64;
    while p * p <= n {
        if n % p == 0 {
            factors.push(p);
            while n % p == 0 {
                n /= p;
            }
        }
        p += 2;
    }

    // If n > 1, then it's a prime factor
    if n > 1 {
        factors.push(n);
    }

    factors
}

/// Compute psi, omega for twisted NTT:
/// - psi is a 2N-th primitive root: psi^(2N) = 1, psi^N = q-1 (=-1)
/// - omega = psi^2 is an N-th primitive root.
fn negacyclic_roots(q: u64, n: usize) -> (u64, u64) {
    let g = primitive_root(q);
    let two_n = 2u64 * (n as u64);
    assert_eq!((q - 1) % two_n, 0, "q-1 must be divisible by 2N for NTT");
    let exp = (q - 1) / two_n;
    let psi = mod_pow_u64(g, exp, q);
    let omega = mod_mul_u64(psi, psi, q); // psi^2
    (psi, omega)
}

/// Bit reverse of k with logn bits
#[inline(always)]
fn bitrev(mut k: usize, logn: usize) -> usize {
    let mut r = 0usize;
    for _ in 0..logn {
        r = (r << 1) | (k & 1);
        k >>= 1;
    }
    r
}

/// In-place iterative Cooley–Tukey NTT with root `omega` of order N (cyclic).
fn ntt_in_place(a: &mut [u64], q: u64, omega: u64) {
    let n = a.len();
    let logn = n.trailing_zeros() as usize;

    // bit-reverse permutation
    for i in 0..n {
        let j = bitrev(i, logn);
        if j > i { a.swap(i, j); }
    }

    let mut m = 1;
    for _stage in 0..logn {
        let m2 = m << 1;
        // w_m = omega^(N/m2)
        let w_m = mod_pow_u64(omega, (n / m2) as u64, q);
        let mut k = 0;
        while k < n {
            let mut w = 1u64;
            for j in 0..m {
                let t = mod_mul_u64(w, a[k + j + m], q);
                let u = a[k + j];
                a[k + j]     = mod_add_u64(u, t, q);
                a[k + j + m] = mod_sub_u64(u, t, q);
                w = mod_mul_u64(w, w_m, q);
            }
            k += m2;
        }
        m = m2;
    }
}

/// Inverse NTT (cyclic) with inverse root omega^{-1} and scaling by n^{-1}.
fn intt_in_place(a: &mut [u64], q: u64, omega: u64) {
    let n = a.len();
    // omega_inv = omega^{-1} = omega^{q-2} by Fermat's Little Theorem
    // Equivalently: omega^{n-1} since omega has order n
    let omega_inv = mod_pow_u64(omega, q - 2, q);
    ntt_in_place(a, q, omega_inv);
    // scale by n^{-1} = n^{q-2} by Fermat
    let n_inv = mod_pow_u64(n as u64, q - 2, q);
    for v in a.iter_mut() {
        *v = mod_mul_u64(*v, n_inv, q);
    }
}

/// Twisted NTT for negacyclic convolution modulo X^N + 1
/// forward: a[i] <- a[i] * psi^i, then NTT with omega = psi^2
fn negacyclic_ntt(mut a: Vec<u64>, q: u64, psi: u64, omega: u64) -> Vec<u64> {
    let n = a.len();
    let mut pow = 1u64;
    for i in 0..n {
        a[i] = mod_mul_u64(a[i], pow, q);
        pow = mod_mul_u64(pow, psi, q);
    }
    ntt_in_place(&mut a, q, omega);
    a
}

/// inverse: inverse NTT, then a[i] <- a[i] * psi^{-i}
fn negacyclic_intt(mut a: Vec<u64>, q: u64, psi: u64, omega: u64) -> Vec<u64> {
    let n = a.len();
    intt_in_place(&mut a, q, omega);

    // multiply by psi^{-i}
    let psi_inv = mod_pow_u64(psi, (q - 1) - 1, q); // psi^{-1}
    let mut pow = 1u64;
    for i in 0..n {
        a[i] = mod_mul_u64(a[i], pow, q);
        pow = mod_mul_u64(pow, psi_inv, q);
    }
    a
}

// ============================================================================
// Polynomial multiplication using NTT
// ============================================================================

/// Normalize i64 to u64 in [0, q) - CRITICAL for correct NTT input
///
/// This ensures the value is strictly in [0, q) before passing to NTT.
/// Missing this final modulo reduction causes NTT to fail for certain primes.
#[inline]
fn norm_i64_to_u64_mod_q(x: i64, q: i64) -> u64 {
    let r = (x % q + q) % q;  // Guaranteed in [0, q)
    debug_assert!(r >= 0 && r < q, "norm_i64_to_u64_mod_q failed: r={}, q={}", r, q);
    r as u64
}

/// Negacyclic polynomial multiply c = a * b mod (x^n + 1, q) using twisted NTT.
/// `a`, `b` are length-n with coefficients in [0, q).
///
/// This is the PRODUCTION implementation for CKKS with 60-bit primes.
/// Uses O(N log N) NTT instead of naive O(N^2) schoolbook multiplication.
pub fn polynomial_multiply_ntt(a: &[i64], b: &[i64], q_i64: i64, n: usize) -> Vec<i64> {
    debug_assert!(n.is_power_of_two(), "N must be power of 2 for NTT");
    let q = q_i64 as u64;

    // Convert to u64 residues in [0, q) - FIXED: proper normalization
    let au: Vec<u64> = a.iter().copied().map(|x| norm_i64_to_u64_mod_q(x, q_i64)).collect();
    let bu: Vec<u64> = b.iter().copied().map(|x| norm_i64_to_u64_mod_q(x, q_i64)).collect();

    // Twisted NTT params
    let (psi, omega) = negacyclic_roots(q, n);

    // Sanity checks for NTT roots (from expert feedback)
    debug_assert_eq!(mod_pow_u64(psi, 2 * n as u64, q), 1,
        "psi must be 2N-th root of unity");
    debug_assert_eq!(mod_pow_u64(psi, n as u64, q), q - 1,
        "psi^N must equal -1 (mod q)");
    debug_assert_eq!(mod_pow_u64(omega, n as u64, q), 1,
        "omega must be N-th root of unity");
    debug_assert_ne!(mod_pow_u64(omega, n as u64 / 2, q), 1,
        "omega must be primitive N-th root (not a (N/2)-th root)");

    // Forward NTT (negacyclic)
    let a_ntt = negacyclic_ntt(au, q, psi, omega);
    let b_ntt = negacyclic_ntt(bu, q, psi, omega);

    // Pointwise multiply
    let mut c_ntt = vec![0u64; n];
    for i in 0..n {
        c_ntt[i] = mod_mul_u64(a_ntt[i], b_ntt[i], q);
    }

    // Inverse NTT (negacyclic)
    let c = negacyclic_intt(c_ntt, q, psi, omega);

    // Back to i64 in [0, q)
    c.into_iter().map(|v| v as i64).collect()
}

/// RNS-CKKS plaintext (polynomial in RNS representation)
#[derive(Debug, Clone)]
pub struct RnsPlaintext {
    /// Polynomial coefficients in RNS form
    pub coeffs: RnsPolynomial,
    /// Scaling factor used for encoding
    pub scale: f64,
    /// Ring dimension
    pub n: usize,
}

impl RnsPlaintext {
    /// Create plaintext from RNS polynomial
    pub fn new(coeffs: RnsPolynomial, scale: f64) -> Self {
        let n = coeffs.n;
        Self { coeffs, scale, n }
    }

    /// Create plaintext from regular coefficients
    pub fn from_coeffs(coeffs: Vec<i64>, scale: f64, primes: &[i64], level: usize) -> Self {
        let n = coeffs.len();
        let rns_coeffs = RnsPolynomial::from_coeffs(&coeffs, primes, n, level);
        Self::new(rns_coeffs, scale)
    }

    /// Convert to regular coefficients using CRT (full reconstruction)
    ///
    /// WARNING: This can overflow if Q = ∏qᵢ > i64_MAX!
    /// For CKKS decoding with many primes, use to_coeffs_f64() instead.
    ///
    /// # Deprecated
    /// Use to_coeffs_f64() for production code with >3 primes.
    pub fn to_coeffs(&self, primes: &[i64]) -> Vec<i64> {
        self.coeffs.to_coeffs(primes)
    }

    /// Convert to i128 coefficients using Garner's CRT (PRODUCTION VERSION)
    ///
    /// Uses Garner's algorithm entirely in i128 arithmetic:
    /// - Works correctly with up to 10 primes
    /// - Returns centered values in (-Q/2, Q/2]
    /// - All intermediate arithmetic stays in i128 (no f64 precision loss)
    ///
    /// This is the **RECOMMENDED** way to decode CKKS plaintexts with multiple primes.
    ///
    /// # Arguments
    /// * `primes` - Prime chain (2-10 primes)
    ///
    /// # Returns
    /// Coefficients as i128 in range (-Q/2, Q/2]
    pub fn to_coeffs_i128(&self, primes: &[i64]) -> Vec<i128> {
        self.coeffs.to_coeffs_crt_i128(primes)
    }

    /// Convert to f64 coefficients using Garner's CRT
    ///
    /// Uses Garner's algorithm which:
    /// - Works correctly with up to 10 primes (no i128 overflow)
    /// - Returns centered values in (-Q/2, Q/2]
    /// - Uses f64 arithmetic for numerical stability
    ///
    /// # Deprecated
    /// Use to_coeffs_i128() for production code to avoid f64 precision loss.
    ///
    /// # Arguments
    /// * `primes` - Prime chain
    ///
    /// # Returns
    /// Coefficients as f64 in range (-Q/2, Q/2]
    pub fn to_coeffs_f64(&self, primes: &[i64]) -> Vec<f64> {
        self.coeffs.to_coeffs_crt_centered(primes)
    }

    /// Convert to regular coefficients using single prime
    ///
    /// Extracts coefficients modulo the first prime (largest/base prime).
    /// This avoids CRT computation but may lose precision.
    ///
    /// # Arguments
    /// * `primes` - Prime chain (will use first prime)
    pub fn to_coeffs_single_prime(&self, primes: &[i64]) -> Vec<i64> {
        let prime_idx = 0; // Use first prime (largest, most precision)
        let prime_value = primes[prime_idx];
        self.coeffs.to_coeffs_single_prime(prime_idx, prime_value)
    }
}

/// RNS-CKKS ciphertext
///
/// A ciphertext is a pair (c0, c1) of RNS polynomials in R_q
/// Decryption: m ≈ c0 + c1*s (mod Q) where Q = q₀ · q₁ · ...
#[derive(Debug, Clone)]
pub struct RnsCiphertext {
    /// First component (RNS polynomial)
    pub c0: RnsPolynomial,
    /// Second component (RNS polynomial)
    pub c1: RnsPolynomial,
    /// Current level (determines which primes are active)
    /// Level 0: all primes [q₀, q₁, q₂, ...]
    /// Level 1: dropped last prime [q₀, q₁, ...]
    pub level: usize,
    /// Scaling factor (carries through homomorphic operations)
    pub scale: f64,
    /// Ring dimension
    pub n: usize,
}

impl RnsCiphertext {
    /// Create new RNS ciphertext
    pub fn new(c0: RnsPolynomial, c1: RnsPolynomial, level: usize, scale: f64) -> Self {
        let n = c0.n;
        assert_eq!(c1.n, n, "Ciphertext components must have same length");
        assert_eq!(c0.level, level, "c0 level mismatch");
        assert_eq!(c1.level, level, "c1 level mismatch");
        Self {
            c0,
            c1,
            level,
            scale,
            n,
        }
    }
}

/// Encrypt plaintext using public key (RNS version)
///
/// RNS-CKKS encryption:
/// 1. Sample random r, e0, e1 from error distribution
/// 2. Convert to RNS representation
/// 3. Compute (all in RNS):
///    c0 = b*r + e0 + m
///    c1 = a*r + e1
///
/// # Arguments
/// * `pk` - RNS public key
/// * `pt` - Plaintext in RNS form
/// * `params` - CKKS parameters with modulus chain
pub fn rns_encrypt(pk: &RnsPublicKey, pt: &RnsPlaintext, params: &CliffordFHEParams) -> RnsCiphertext {
    use rand::{thread_rng, Rng};
    use rand_distr::{Distribution, Normal};

    let n = params.n;
    let primes = &params.moduli;
    let num_primes = primes.len();
    let mut rng = thread_rng();

    // Sample ternary random polynomial r ∈ {-1, 0, 1}^n
    let r: Vec<i64> = (0..n)
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

    // Sample errors e0, e1 from Gaussian distribution
    let normal = Normal::new(0.0, params.error_std).unwrap();
    let e0: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();
    let e1: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();

    // Convert to RNS
    let r_rns = RnsPolynomial::from_coeffs(&r, primes, n, 0);
    let e0_rns = RnsPolynomial::from_coeffs(&e0, primes, n, 0);
    let e1_rns = RnsPolynomial::from_coeffs(&e1, primes, n, 0);

    // Use RNS public key directly (already in RNS form!)
    // Compute b*r using RNS multiplication
    let br = rns_poly_multiply(&pk.b, &r_rns, primes, polynomial_multiply_ntt);

    // Compute a*r using RNS multiplication
    let ar = rns_poly_multiply(&pk.a, &r_rns, primes, polynomial_multiply_ntt);

    // DEBUG: Trace encryption steps
    if std::env::var("RNS_TRACE").is_ok() {
        // eprintln!("\n[ENCRYPTION TRACE]");
        // eprintln!("  r (first 5 coeffs): {:?}", &r[..5]);
        // eprintln!("  r[0] residues: {:?}", &r_rns.rns_coeffs[0]);
        // eprintln!("  r[1] residues: {:?}", &r_rns.rns_coeffs[1]);
        // eprintln!("  r[63] residues: {:?}", &r_rns.rns_coeffs[63]);
        // eprintln!("  e0[0] residues: {:?}", &e0_rns.rns_coeffs[0]);
        // eprintln!("  e1[0] residues: {:?}", &e1_rns.rns_coeffs[0]);
        // eprintln!("  pt.coeffs[0] residues: {:?}", &pt.coeffs.rns_coeffs[0]);
        // eprintln!("  pk.b[0] residues: {:?}", &pk.b.rns_coeffs[0]);
        // eprintln!("  pk.b[1] residues: {:?}", &pk.b.rns_coeffs[1]);
        // eprintln!("  pk.b[63] residues: {:?}", &pk.b.rns_coeffs[63]);
        // eprintln!("  pk.a[0] residues: {:?}", &pk.a.rns_coeffs[0]);
        // eprintln!("  br[0] residues: {:?}", &br.rns_coeffs[0]);
        // eprintln!("  ar[0] residues: {:?}", &ar.rns_coeffs[0]);
    }

    // c0 = b*r + e0 + m
    let c0_temp = rns_add(&br, &e0_rns, primes);
    let c0 = rns_add(&c0_temp, &pt.coeffs, primes);

    // c1 = a*r + e1
    let c1 = rns_add(&ar, &e1_rns, primes);

    if std::env::var("RNS_TRACE").is_ok() {
        // eprintln!("  c0_temp[0] residues: {:?}", &c0_temp.rns_coeffs[0]);
        // eprintln!("  c0[0] residues: {:?}", &c0.rns_coeffs[0]);
        // eprintln!("  c1[0] residues: {:?}", &c1.rns_coeffs[0]);
    }

    RnsCiphertext::new(c0, c1, 0, pt.scale)
}

/// Decrypt ciphertext using secret key (RNS version)
///
/// RNS-CKKS decryption:
/// m' = c0 + c1*s (all in RNS, then convert back)
///
/// # Arguments
/// * `sk` - RNS secret key
/// * `ct` - Ciphertext in RNS form
/// * `params` - CKKS parameters
pub fn rns_decrypt(sk: &RnsSecretKey, ct: &RnsCiphertext, params: &CliffordFHEParams) -> RnsPlaintext {
    let n = ct.n;
    let primes = &params.moduli;
    let num_primes = primes.len() - ct.level; // Active primes at this level
    let active_primes = &primes[..num_primes];

    // Use RNS secret key directly
    // Note: If we've rescaled, the secret key needs to be at the same level
    // For level > 0, we need to use only the first num_primes components
    let sk_at_level = if ct.level > 0 {
        // Extract only active primes from secret key
        let mut rns_coeffs = vec![vec![0i64; num_primes]; n];
        for i in 0..n {
            for j in 0..num_primes {
                rns_coeffs[i][j] = sk.coeffs.rns_coeffs[i][j];
            }
        }
        RnsPolynomial::new(rns_coeffs, n, ct.level)
    } else {
        sk.coeffs.clone()
    };

    // Compute c1*s using RNS multiplication
    let c1s = rns_poly_multiply(&ct.c1, &sk_at_level, active_primes, polynomial_multiply_ntt);

    // DEBUG: Trace decryption steps
    if std::env::var("RNS_TRACE").is_ok() {
        // eprintln!("\n[DECRYPTION TRACE]");
        // eprintln!("  ct.c0[0] residues: {:?}", &ct.c0.rns_coeffs[0]);
        // eprintln!("  ct.c0[1] residues: {:?}", &ct.c0.rns_coeffs[1]);
        // eprintln!("  ct.c0[2] residues: {:?}", &ct.c0.rns_coeffs[2]);
        // eprintln!("  ct.c1[0] residues: {:?}", &ct.c1.rns_coeffs[0]);
        // eprintln!("  ct.c1[1] residues: {:?}", &ct.c1.rns_coeffs[1]);
        // eprintln!("  ct.c1[2] residues: {:?}", &ct.c1.rns_coeffs[2]);
        // eprintln!("  sk[0] residues: {:?}", &sk_at_level.rns_coeffs[0]);
        // eprintln!("  sk[1] residues: {:?}", &sk_at_level.rns_coeffs[1]);
        // eprintln!("  sk[2] residues: {:?}", &sk_at_level.rns_coeffs[2]);

        // CRT consistency check for inputs
        let check_crt = |name: &str, residues: &[i64], primes: &[i64]| {
            if residues.len() == 2 {
                let r0 = residues[0] as i128;
                let r1 = residues[1] as i128;
                let p = primes[0] as i128;
                let q = primes[1] as i128;

                // Check if they represent the same value by checking (r0 - r1) % gcd(p,q) == 0
                // Since p, q are coprime, gcd = 1, so we check via CRT
                let _crt_consistent = r0 % q == r1;
            }
        };

        check_crt("ct.c0[0]", &ct.c0.rns_coeffs[0], active_primes);
        check_crt("ct.c1[0]", &ct.c1.rns_coeffs[0], active_primes);
        check_crt("sk[0]", &sk_at_level.rns_coeffs[0], active_primes);
        check_crt("c1s[0]", &c1s.rns_coeffs[0], active_primes);
    }

    // m' = c0 + c1*s (because pk.b = -a*s + e, the decryption formula is c0 + c1*s)
    // Proof: c0 + c1*s = (b*r + e0 + m) + (a*r + e1)*s
    //                  = ((-a*s + e)*r + e0 + m) + (a*r)*s + e1*s
    //                  = -a*s*r + e*r + e0 + m + a*r*s + e1*s
    //                  = m + e*r + e0 + e1*s  (the a*s*r terms cancel!)
    let m_prime = rns_add(&ct.c0, &c1s, active_primes);

    if std::env::var("RNS_TRACE").is_ok() {
        // Only do CRT check if we have 2+ residues
        if num_primes >= 2 {
            let r0 = m_prime.rns_coeffs[0][0] as i128;
            let r1 = m_prime.rns_coeffs[0][1] as i128;
            let _p = active_primes[0] as i128;
            let q = active_primes[1] as i128;
            let _crt_consistent = r0 % q == r1;
        }
    }

    RnsPlaintext::new(m_prime, ct.scale)
}

/// Homomorphic addition (RNS version)
///
/// Simply add the RNS polynomials component-wise.
/// Scales must match!
pub fn rns_add_ciphertexts(
    ct1: &RnsCiphertext,
    ct2: &RnsCiphertext,
    params: &CliffordFHEParams,
) -> RnsCiphertext {
    assert_eq!(ct1.level, ct2.level, "Ciphertexts must be at same level");
    assert_eq!(ct1.n, ct2.n, "Ciphertexts must have same dimension");

    // TODO: Handle scale mismatch (need to rescale to match)
    // For now, require same scale
    assert!((ct1.scale - ct2.scale).abs() < 1e-6, "Scales must match (for now)");

    let primes = &params.moduli;
    let num_primes = primes.len() - ct1.level;
    let active_primes = &primes[..num_primes];

    let c0 = rns_add(&ct1.c0, &ct2.c0, active_primes);
    let c1 = rns_add(&ct1.c1, &ct2.c1, active_primes);

    RnsCiphertext::new(c0, c1, ct1.level, ct1.scale)
}

/// Homomorphic multiplication with rescaling (RNS version)
///
/// This is the KEY operation that requires RNS!
///
/// Steps:
/// 1. Multiply polynomials (tensored ciphertext): (c0, c1) ⊗ (d0, d1) = (c0d0, c0d1+c1d0, c1d1)
/// 2. Relinearize: convert degree-2 back to degree-1 using evaluation key
/// 3. **Rescale**: drop the last prime from the modulus chain
///
/// After rescaling:
/// - Level increases by 1 (one fewer prime)
/// - Scale divided by the dropped prime
/// - Coefficients properly normalized
pub fn rns_multiply_ciphertexts(
    ct1: &RnsCiphertext,
    ct2: &RnsCiphertext,
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> RnsCiphertext {
    assert_eq!(ct1.level, ct2.level, "Ciphertexts must be at same level");
    assert_eq!(ct1.n, ct2.n, "Ciphertexts must have same dimension");

    let n = ct1.n;
    let primes = &params.moduli;
    let num_primes = primes.len() - ct1.level;
    let active_primes = &primes[..num_primes];

    // eprintln!("\n[INPUT CIPHERTEXTS]");
    // eprintln!("  ct1.c0[0] residues: {:?}", &ct1.c0.rns_coeffs[0][..ct1.c0.num_primes().min(3)]);
    // eprintln!("  ct2.c0[0] residues: {:?}", &ct2.c0.rns_coeffs[0][..ct2.c0.num_primes().min(3)]);
    // eprintln!("  Expected ct1.c0[0] ≈ 2Δ ≈ {:.2e}", 2.0 * params.scale);
    // eprintln!("  Expected ct2.c0[0] ≈ 3Δ ≈ {:.2e}", 3.0 * params.scale);

    // Step 1: Multiply ciphertexts (tensored product)
    // Degree-2 ciphertext: (d0, d1, d2) where m1 * m2 = d0 + d1*s + d2*s²
    let c0d0 = rns_poly_multiply(&ct1.c0, &ct2.c0, active_primes, polynomial_multiply_ntt);
    let c0d1 = rns_poly_multiply(&ct1.c0, &ct2.c1, active_primes, polynomial_multiply_ntt);
    let c1d0 = rns_poly_multiply(&ct1.c1, &ct2.c0, active_primes, polynomial_multiply_ntt);
    let c1d1 = rns_poly_multiply(&ct1.c1, &ct2.c1, active_primes, polynomial_multiply_ntt);

    // eprintln!("\n[AFTER TENSOR PRODUCT]");
    // eprintln!("  c0d0[0] residues: {:?}", &c0d0.rns_coeffs[0][..c0d0.num_primes().min(3)]);
    // eprintln!("  c1d1[0] (=d2) residues: {:?}", &c1d1.rns_coeffs[0][..c1d1.num_primes().min(3)]);

    // d1 = c0*d1 + c1*d0
    let d_mid = rns_add(&c0d1, &c1d0, active_primes);

    // Step 2: Relinearization (degree-2 → degree-1)
    let (new_c0, new_c1) = rns_relinearize_degree2(&c0d0, &d_mid, &c1d1, evk, active_primes, n);

    // DEBUG: Check values BEFORE rescale
    // // eprintln!("\n[BEFORE RESCALE] Values before rescaling:");
    // // eprintln!("  new_c0[0] residues:");
    for j in 0..new_c0.num_primes().min(3) {
        let qi = active_primes[j];
        let r = new_c0.rns_coeffs[0][j];
        let centered = if r > qi / 2 { r - qi } else { r };
        // eprintln!("    j={} r={} centered={}", j, r, centered);
    }
    // // eprintln!("  new_c1[0] residues:");
    for j in 0..new_c1.num_primes().min(3) {
        let qi = active_primes[j];
        let r = new_c1.rns_coeffs[0][j];
        let centered = if r > qi / 2 { r - qi } else { r };
        // eprintln!("    j={} r={} centered={}", j, r, centered);
    }

    // Step 3: Exact rescaling with proper rounding
    let inv = precompute_rescale_inv(active_primes);
    let rescaled_c0 = rns_rescale_exact(&new_c0, active_primes, &inv);
    let rescaled_c1 = rns_rescale_exact(&new_c1, active_primes, &inv);

    // PROBE A: Verify DivideRoundByLastq identity
    fn verify_divide_round_by_lastq(
        pre: &RnsPolynomial,
        post: &RnsPolynomial,
        primes: &[i64],
        idx: usize,
    ) {
        let num_primes = pre.num_primes();
        let q_last = primes[num_primes - 1];

        let c_l = pre.rns_coeffs[idx][num_primes - 1];
        let c_l_center = if c_l > q_last / 2 { c_l - q_last } else { c_l };

        for j in 0..(num_primes - 1) {
            let qi = primes[j];

            let lhs = {
                let t = ((post.rns_coeffs[idx][j] as i128) * (q_last as i128)) % (qi as i128);
                let u = (t + (c_l_center as i128)) % (qi as i128);
                ((u + (qi as i128)) % (qi as i128)) as i64
            };

            let rhs = pre.rns_coeffs[idx][j] % qi;

            if lhs != rhs {
                // eprintln!(
                //     "[DIVROUND CHECK FAIL] coeff {}, prime j={} (qi={}): LHS={} != RHS={}",
                //     idx, j, qi, lhs, rhs
                // );
                // eprintln!("  c_l_center={}, post[{}][{}]={}, pre[{}][{}]={}",
                //     c_l_center, idx, j, post.rns_coeffs[idx][j], idx, j, pre.rns_coeffs[idx][j]);
            }
        }
    }

    // PROBE B: Residue magnitude sanity check
    fn dump_residues(name: &str, poly: &RnsPolynomial, primes: &[i64], idx: usize) {
        // eprintln!("{} coeff[{}] residues:", name, idx);
        for j in 0..poly.num_primes() {
            let qi = primes[j];
            let r = poly.rns_coeffs[idx][j];
            let centered = if r > qi / 2 { r - qi } else { r };
            // eprintln!("  j={}  r={}  centered={}", j, r, centered);
        }
    }

    // // eprintln!("\n[RESCALE VERIFICATION]");
    verify_divide_round_by_lastq(&new_c0, &rescaled_c0, active_primes, 0);
    verify_divide_round_by_lastq(&new_c1, &rescaled_c1, active_primes, 0);

    let new_primes = &active_primes[..active_primes.len()-1];
    dump_residues("c0_after_rescale", &rescaled_c0, new_primes, 0);
    dump_residues("c1_after_rescale", &rescaled_c1, new_primes, 0);

    // New scale: (scale1 * scale2) / q_last
    let q_last = active_primes[num_primes - 1];
    let new_scale = (ct1.scale * ct2.scale) / (q_last as f64);
    let new_level = ct1.level + 1;

    RnsCiphertext::new(rescaled_c0, rescaled_c1, new_level, new_scale)
}

/// Relinearize degree-2 ciphertext to degree-1 (RNS version)
///
/// Input: (d0, d1, d2) where m = d0 + d1*s + d2*s²
/// Output: (c0, c1) where m ≈ c0 + c1*s
///
/// Uses evaluation key which encrypts s²
fn rns_relinearize_degree2(
    d0: &RnsPolynomial,
    d1: &RnsPolynomial,
    d2: &RnsPolynomial,
    evk: &RnsEvaluationKey,
    primes: &[i64],
    _n: usize,
) -> (RnsPolynomial, RnsPolynomial) {
    // CORRECTED: Use gadget decomposition instead of direct multiplication
    // This is the key fix - decompose d2 in base 2^w to control noise

    // 1) Decompose d2 into D digits in base B = 2^w
    let d2_digits = decompose_base_pow2(d2, primes, evk.base_w);

    // // eprintln!("[RELINEARIZATION DEBUG]");
    // // eprintln!("  d2[0] residues: {:?}", &d2.rns_coeffs[0][..d2.num_primes().min(3)]);
    // // eprintln!("  base_w: {}, num_digits: {}", evk.base_w, d2_digits.len());
    // for t in 0..d2_digits.len() {
    //     // eprintln!("  digit[{}][0] residues: {:?}", t, &d2_digits[t].rns_coeffs[0][..d2_digits[t].num_primes().min(3)]);
    // }

    // 2) TEST: Try with subtraction for c1
    // If EVK satisfies: evk0[t] - evk1[t]·s = -B^t·s² + e_t
    // Then: -B^t·s² = evk0[t] - evk1[t]·s - e_t
    // So: d2·s² = -Σ d_t·(evk0[t] - evk1[t]·s - e_t)
    //           = -Σ d_t·evk0[t] + (Σ d_t·evk1[t])·s + noise
    // Therefore: m = d0 + d1·s + d2·s²
    //              = (d0 - Σ d_t·evk0[t]) + (d1 + Σ d_t·evk1[t])·s + noise
    let mut c0 = d0.clone();
    let mut c1 = d1.clone();

    for t in 0..d2_digits.len() {
        // LEVEL MATCHING: If digit is at level L, but EVK is at level 0,
        // we need to truncate EVK to match the digit's level.
        let digit_level = d2_digits[t].level;
        let digit_num_primes = d2_digits[t].num_primes();

        let evk0_matched = if evk.evk0[t].level != digit_level {
            // Truncate EVK to match digit level
            let mut new_coeffs = Vec::new();
            for i in 0..evk.evk0[t].n {
                new_coeffs.push(evk.evk0[t].rns_coeffs[i][..digit_num_primes].to_vec());
            }
            RnsPolynomial::new(new_coeffs, evk.evk0[t].n, digit_level)
        } else {
            evk.evk0[t].clone()
        };

        let evk1_matched = if evk.evk1[t].level != digit_level {
            // Truncate EVK to match digit level
            let mut new_coeffs = Vec::new();
            for i in 0..evk.evk1[t].n {
                new_coeffs.push(evk.evk1[t].rns_coeffs[i][..digit_num_primes].to_vec());
            }
            RnsPolynomial::new(new_coeffs, evk.evk1[t].n, digit_level)
        } else {
            evk.evk1[t].clone()
        };

        // Multiply small digit by corresponding EVK component (now at matching levels)
        let u0 = rns_poly_multiply(&d2_digits[t], &evk0_matched, primes, polynomial_multiply_ntt);
        let u1 = rns_poly_multiply(&d2_digits[t], &evk1_matched, primes, polynomial_multiply_ntt);

        // eprintln!("  After mult with evk[{}]: u0[0]={:?}, u1[0]={:?}",
        //           t, &u0.rns_coeffs[0][..u0.num_primes().min(3)], &u1.rns_coeffs[0][..u1.num_primes().min(3)]);

        // Accumulate: SUBTRACT u0 from c0, ADD u1 to c1
        c0 = rns_sub(&c0, &u0, primes);  // TEST: subtract instead of add
        c1 = rns_add(&c1, &u1, primes);
    }

    (c0, c1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rns_plaintext_conversion() {
        let coeffs = vec![123, 456, 789, -100];
        let scale = 1024.0;
        let primes = vec![1_099_511_627_689, 1_099_511_627_691];

        let pt = RnsPlaintext::from_coeffs(coeffs.clone(), scale, &primes, 0);
        let recovered = pt.to_coeffs(&primes);

        for i in 0..coeffs.len() {
            assert_eq!(coeffs[i], recovered[i]);
        }
    }
}
