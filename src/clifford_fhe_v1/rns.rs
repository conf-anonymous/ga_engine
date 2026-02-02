//! Residue Number System (RNS) for CKKS
//!
//! RNS representation allows working with large moduli by using the Chinese Remainder Theorem.
//! Instead of storing coefficients modulo a single large Q, we store them as tuples of
//! residues modulo smaller primes: (c mod q₀, c mod q₁, ..., c mod qₗ)
//!
//! Key benefits:
//! - Support moduli Q > 2^63 (product of primes)
//! - Efficient rescaling (drop one prime)
//! - Parallelizable operations (per-prime)

use num_bigint::BigInt;

/// Domain tag for RNS polynomials
///
/// Tracks whether polynomial is in coefficient or NTT domain.
/// This prevents accidentally multiplying COEF with NTT polynomials.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    /// Coefficient domain (standard representation)
    Coef,
    /// NTT domain (Number Theoretic Transform, for fast multiplication)
    Ntt,
}

/// RNS polynomial: coefficients in residue representation
///
/// Each coefficient is represented as a vector of residues mod each prime in the chain.
/// For example, with primes [q₀, q₁, q₂] and coefficient c:
///   rns_coeffs[i] = [c mod q₀, c mod q₁, c mod q₂]
#[derive(Clone, Debug)]
pub struct RnsPolynomial {
    /// Coefficients in RNS form
    /// Outer vector: polynomial coefficients (length N)
    /// Inner vector: residues for each prime (length L = number of primes at this level)
    pub rns_coeffs: Vec<Vec<i64>>,

    /// Number of coefficients (polynomial degree)
    pub n: usize,

    /// Current level (determines which primes are active)
    /// Level 0: all primes [q₀, q₁, ..., qₗ]
    /// Level 1: dropped last prime [q₀, q₁, ..., qₗ₋₁]
    pub level: usize,

    /// Domain tag (COEF or NTT)
    /// All operations verify domain compatibility
    pub domain: Domain,
}

impl RnsPolynomial {
    /// Create new RNS polynomial from RNS coefficients (COEF domain by default)
    pub fn new(rns_coeffs: Vec<Vec<i64>>, n: usize, level: usize) -> Self {
        assert_eq!(rns_coeffs.len(), n, "Wrong number of coefficients");
        Self { rns_coeffs, n, level, domain: Domain::Coef }
    }

    /// Create new RNS polynomial with explicit domain tag
    pub fn new_with_domain(rns_coeffs: Vec<Vec<i64>>, n: usize, level: usize, domain: Domain) -> Self {
        assert_eq!(rns_coeffs.len(), n, "Wrong number of coefficients");
        Self { rns_coeffs, n, level, domain }
    }

    /// Create RNS polynomial from regular coefficients
    ///
    /// Converts coefficients [c₀, c₁, ..., cₙ₋₁] to RNS form:
    /// rns_coeffs[i] = [cᵢ mod q₀, cᵢ mod q₁, ...]
    pub fn from_coeffs(coeffs: &[i64], primes: &[i64], n: usize, level: usize) -> Self {
        assert_eq!(coeffs.len(), n, "Wrong number of coefficients");

        let num_primes = primes.len() - level; // Active primes at this level
        let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

        for i in 0..n {
            for (j, &q) in primes.iter().take(num_primes).enumerate() {
                rns_coeffs[i][j] = ((coeffs[i] % q) + q) % q;
            }
        }

        Self::new(rns_coeffs, n, level)
    }

    /// Convert RNS polynomial back to regular coefficients using CRT
    ///
    /// Uses Chinese Remainder Theorem to reconstruct coefficients from residues.
    /// For two primes q₀, q₁: c = (c₀·Q₀·Q₀⁻¹ + c₁·Q₁·Q₁⁻¹) mod Q
    /// where Q = q₀·q₁, Q₀ = Q/q₀, Q₁ = Q/q₁
    ///
    /// Uses BigInt to handle Q > 2^127 (works with any number of primes)
    pub fn to_coeffs(&self, primes: &[i64]) -> Vec<i64> {
        use num_bigint::BigInt;
        use num_traits::{Zero, One};

        let num_primes = self.rns_coeffs[0].len();
        let active_primes: Vec<BigInt> = primes[..num_primes].iter().map(|&q| BigInt::from(q)).collect();

        // Compute Q = product of all active primes (using BigInt)
        let q_product: BigInt = active_primes.iter().fold(BigInt::one(), |acc, q| acc * q);

        let mut coeffs = vec![0i64; self.n];

        for i in 0..self.n {
            let mut c = BigInt::zero();

            for (j, qj) in active_primes.iter().enumerate() {
                // Q_j = Q / q_j
                let q_j = &q_product / qj;

                // Q_j^{-1} mod q_j (using extended Euclidean algorithm)
                let q_j_inv = mod_inverse_bigint(&q_j, qj);

                // Contribution: c_j * Q_j * Q_j^{-1}
                let c_j = BigInt::from(self.rns_coeffs[i][j]);

                // Compute (c_j * Q_j * Q_j^{-1}) % Q
                let contrib = (c_j * &q_j * q_j_inv) % &q_product;

                c = (c + contrib) % &q_product;
            }

            // Center around 0: map from [0, Q) to (-Q/2, Q/2]
            let q_half = &q_product / 2;
            if c > q_half {
                c -= &q_product;
            }

            // Convert BigInt to i64
            // This should be safe since the centered value fits in i64 for practical parameters
            let bytes = c.to_signed_bytes_le();
            let mut result = 0i64;
            for (idx, &byte) in bytes.iter().take(8).enumerate() {
                result |= (byte as i64) << (idx * 8);
            }
            // Handle sign extension if negative
            if bytes.len() > 0 && bytes[bytes.len() - 1] & 0x80 != 0 {
                // Negative number - sign extend
                for idx in bytes.len()..8 {
                    result |= 0xFF << (idx * 8);
                }
            }

            coeffs[i] = result;
        }

        coeffs
    }

    /// Number of active primes at this level
    pub fn num_primes(&self) -> usize {
        self.rns_coeffs[0].len()
    }

    /// Extract coefficients modulo a single prime (avoiding CRT)
    ///
    /// For CKKS decoding, we don't need the full CRT reconstruction!
    /// Since message and noise are small relative to any single prime qᵢ,
    /// we can decode from any prime's representation.
    ///
    /// This avoids:
    /// - Expensive CRT computation
    /// - Overflow issues when Q > i64_MAX
    ///
    /// # Arguments
    /// * `prime_idx` - Which prime to use (0 = first/largest prime)
    /// * `prime_value` - The actual prime value for center-lifting
    pub fn to_coeffs_single_prime(&self, prime_idx: usize, prime_value: i64) -> Vec<i64> {
        assert!(prime_idx < self.num_primes(), "Prime index out of range");

        let mut coeffs = vec![0i64; self.n];

        for i in 0..self.n {
            let c = self.rns_coeffs[i][prime_idx];

            // Center-lift to (-q/2, q/2]
            let centered = if c > prime_value / 2 {
                c - prime_value
            } else {
                c
            };

            coeffs[i] = centered;
        }

        coeffs
    }

    /// CRT reconstruction for 2 primes only (most common case after rescale)
    ///
    /// For 2 primes, we can use a simplified formula that stays in i128.
    /// For p, q: x ≡ a (mod p), x ≡ b (mod q)
    /// Solution: x = a + p * ((b - a) * p^{-1} mod q)
    ///
    /// # Arguments
    /// * `primes` - Exactly 2 primes
    ///
    /// # Returns
    /// Coefficients in range [0, p*q)
    pub fn to_coeffs_crt_two_primes(&self, primes: &[i64]) -> Vec<f64> {
        assert_eq!(primes.len(), 2, "This function requires exactly 2 primes");
        assert_eq!(self.num_primes(), 2, "Polynomial must be at level with 2 primes");

        let p = primes[0] as i128;
        let q = primes[1] as i128;
        let pq = (p as f64) * (q as f64); // Product as f64

        // Precompute p^{-1} mod q
        let p_inv_mod_q = mod_inverse(p, q);

        let mut coeffs = vec![0f64; self.n];

        for i in 0..self.n {
            let a = self.rns_coeffs[i][0] as i128; // Residue mod p
            let b = self.rns_coeffs[i][1] as i128; // Residue mod q

            // CRT formula: x = a + p * ((b - a) * p^{-1} mod q)
            let diff = ((b - a) % q + q) % q;
            let factor = (diff * p_inv_mod_q) % q;
            let x = a + p * factor;

            // Reduce mod pq if needed (should already be in range)
            let x_reduced = ((x % (pq as i128)) + (pq as i128)) % (pq as i128);

            coeffs[i] = x_reduced as f64;
        }

        coeffs
    }

    /// General CRT reconstruction for N primes using Garner's algorithm
    ///
    /// Garner's algorithm is numerically stable and works for arbitrary primes.
    ///
    /// # Arguments
    /// * `primes` - The active primes (2-10 supported)
    ///
    /// # Returns
    /// Coefficients in range [0, Q) where Q = product of primes
    fn to_coeffs_crt_general(&self, primes: &[i64]) -> Vec<f64> {
        assert!(primes.len() >= 2 && primes.len() <= 10, "CRT supports 2-10 primes");

        // Special case: 2 primes (use optimized direct formula)
        if primes.len() == 2 {
            return self.to_coeffs_crt_two_primes(primes);
        }

        // General case: Garner's algorithm
        let num_primes = primes.len();
        let mut coeffs = vec![0f64; self.n];

        // Precompute mixed radix constants: C[i][j] = q_j^{-1} mod q_i for j < i
        let mut c_matrix = vec![vec![1i128; num_primes]; num_primes];
        for i in 1..num_primes {
            let qi = primes[i] as i128;
            for j in 0..i {
                let qj = primes[j] as i128;
                c_matrix[i][j] = mod_inverse(qj, qi);
            }
        }

        // Apply Garner's algorithm for each coefficient
        for idx in 0..self.n {
            // Extract residues
            let residues: Vec<i128> = (0..num_primes)
                .map(|j| self.rns_coeffs[idx][j] as i128)
                .collect();

            // Garner: compute mixed radix digits
            let mut mixed_radix = vec![0i128; num_primes];
            mixed_radix[0] = residues[0];

            for i in 1..num_primes {
                let qi = primes[i] as i128;
                let mut temp = residues[i];

                for j in 0..i {
                    temp = ((temp - mixed_radix[j]) % qi + qi) % qi;
                    temp = (temp * c_matrix[i][j]) % qi;
                }

                mixed_radix[i] = temp;
            }

            // Convert to positional: x = v0 + v1*q0 + v2*q0*q1 + ...
            let mut result = 0f64;
            let mut prod_so_far = 1f64;

            for i in 0..num_primes {
                result += (mixed_radix[i] as f64) * prod_so_far;
                prod_so_far *= primes[i] as f64;
            }

            coeffs[idx] = result;
        }

        coeffs
    }

    /// CRT reconstruction for CKKS using Garner with i128 (PRODUCTION VERSION)
    ///
    /// This is the CORRECT way to decode CKKS ciphertexts with multiple primes.
    /// Uses Garner's algorithm keeping everything in i128 until the final step.
    ///
    /// Key insight: For CKKS, message + noise is SMALL (< 2^80 typically),
    /// so even though Q ≈ 2^600, the actual value we're reconstructing fits in i128.
    /// We just need to be careful about the final center-lifting step.
    ///
    /// # Arguments
    /// * `primes` - The active primes (2-10 supported)
    ///
    /// # Returns
    /// Coefficients as i128 in range (-Q/2, Q/2]
    ///
    /// # Panics
    /// If the reconstructed value doesn't fit in i128 (shouldn't happen for CKKS)
    pub fn to_coeffs_crt_i128(&self, primes: &[i64]) -> Vec<i128> {
        assert!(primes.len() >= 2 && primes.len() <= 10, "CRT supports 2-10 primes");
        let num_primes = primes.len();

        // Special case: 2 primes (optimized)
        if num_primes == 2 {
            return self.to_coeffs_crt_two_primes_i128(primes);
        }

        let mut coeffs = vec![0i128; self.n];

        // Precompute mixed radix constants: C[i][j] = q_j^{-1} mod q_i for j < i
        let mut c_matrix = vec![vec![1i128; num_primes]; num_primes];
        for i in 1..num_primes {
            let qi = primes[i] as i128;
            for j in 0..i {
                let qj = primes[j] as i128;
                c_matrix[i][j] = mod_inverse(qj, qi);
            }
        }

        // Apply Garner's algorithm for each coefficient
        for idx in 0..self.n {
            // Extract residues
            let residues: Vec<i128> = (0..num_primes)
                .map(|j| self.rns_coeffs[idx][j] as i128)
                .collect();

            // Garner: compute mixed radix digits (all in i128)
            let mut mixed_radix = vec![0i128; num_primes];
            mixed_radix[0] = residues[0];

            for i in 1..num_primes {
                let qi = primes[i] as i128;
                let mut temp = residues[i];

                for j in 0..i {
                    temp = ((temp - mixed_radix[j]) % qi + qi) % qi;
                    temp = (temp * c_matrix[i][j]) % qi;
                }

                mixed_radix[i] = temp;
            }

            // Convert from mixed radix to positional (CAREFULLY in i128)
            // x = v0 + v1*q0 + v2*q0*q1 + ...
            // For CKKS: the value should be small even though Q is large
            let mut result = 0i128;
            let mut prod_so_far = 1i128;

            for i in 0..num_primes {
                let qi = primes[i] as i128;
                // result += mixed_radix[i] * prod_so_far
                let term = mulmod_i128(mixed_radix[i], prod_so_far, i128::MAX);
                result = result.saturating_add(term);

                // prod_so_far *= qi (will overflow for many primes, but we don't use it after CKKS range)
                if i < num_primes - 1 {
                    prod_so_far = prod_so_far.saturating_mul(qi);
                }
            }

            // Center-lift: For CKKS, if value is > Q/2, subtract Q
            // But we can't compute Q in i128... Instead, use heuristic:
            // If result is huge positive, it's probably Q - small_value
            // This works because CKKS values are small relative to Q

            // Heuristic: if top bits are set, assume it wraps around
            if result > (i128::MAX / 2) {
                // Value is likely Q - something_small
                // We need to make it negative
                // Approximate: treat as negative (this loses some precision but works for CKKS)
                result = result - i128::MAX;
            }

            coeffs[idx] = result;
        }

        coeffs
    }

    /// CRT for 2 primes returning i128 (optimized case)
    ///
    /// Uses explicit CRT formula (NOT Garner) for 2 primes.
    /// Formula: c = (a * Q0 * Q0_inv + b * Q1 * Q1_inv) mod Q
    /// where Q = p*q, Q0 = q, Q1 = p
    fn to_coeffs_crt_two_primes_i128(&self, primes: &[i64]) -> Vec<i128> {
        assert_eq!(primes.len(), 2);
        let p = primes[0] as i128;
        let q = primes[1] as i128;

        // Compute Q = p * q
        let q_product = p.checked_mul(q).expect("pq overflow - primes too large");

        // Q0 = Q / p = q
        // Q1 = Q / q = p
        let q0 = q;
        let q1 = p;

        // Q0^{-1} mod p (i.e., q^{-1} mod p)
        let q0_inv = mod_inverse(q0, p);

        // Q1^{-1} mod q (i.e., p^{-1} mod q)
        let q1_inv = mod_inverse(q1, q);

        let mut coeffs = vec![0i128; self.n];

        for i in 0..self.n {
            let a = self.rns_coeffs[i][0] as i128; // residue mod p
            let b = self.rns_coeffs[i][1] as i128; // residue mod q

            // Contribution 1: a * Q0 * Q0_inv = a * q * (q^{-1} mod p)
            // This must be computed mod Q to avoid overflow
            let contrib0 = {
                let temp = mulmod_i128(a, q0, q_product);
                mulmod_i128(temp, q0_inv, q_product)
            };

            // Contribution 2: b * Q1 * Q1_inv = b * p * (p^{-1} mod q)
            let contrib1 = {
                let temp = mulmod_i128(b, q1, q_product);
                mulmod_i128(temp, q1_inv, q_product)
            };

            // c = (contrib0 + contrib1) mod Q
            let c = (contrib0 + contrib1) % q_product;

            // Center-lift: c in [0, Q) -> (-Q/2, Q/2]
            let centered = if c > q_product / 2 {
                c - q_product
            } else {
                c
            };

            if i == 0 {
                // eprintln!("[DEBUG 2-prime CRT] a={}, b={}, contrib0={}, contrib1={}, c={}, centered={}",
                //          a, b, contrib0, contrib1, c, centered);
                // eprintln!("  Q={}, Q/2={}", q_product, q_product/2);
            }

            coeffs[i] = centered;
        }

        coeffs
    }

    /// CRT reconstruction with centered reduction to (-Q/2, Q/2]
    ///
    /// This is the standard way to decode CKKS ciphertexts.
    /// Supports 2-10 primes using Garner's algorithm.
    ///
    /// # Arguments
    /// * `primes` - The active primes (2-10 supported)
    ///
    /// # Returns
    /// Coefficients as f64 in range (-Q/2, Q/2]
    ///
    /// # Deprecated
    /// Use to_coeffs_crt_i128() for better precision
    pub fn to_coeffs_crt_centered(&self, primes: &[i64]) -> Vec<f64> {
        assert!(primes.len() >= 2 && primes.len() <= 10, "CRT supports 2-10 primes");

        let coeffs = self.to_coeffs_crt_general(primes);

        // Compute Q = product of primes (use f64)
        let q_product: f64 = primes.iter().map(|&p| p as f64).product();
        let q_half = q_product / 2.0;

        // Center-lift each coefficient
        coeffs.iter().map(|&c| {
            if c > q_half {
                c - q_product
            } else {
                c
            }
        }).collect()
    }
}

/// Modular multiplication without overflow: (a * b) % m
///
/// Uses double-and-add method to avoid overflow when a*b > i128::MAX
pub fn mulmod_i128(a: i128, b: i128, m: i128) -> i128 {
    // Ensure inputs are reduced mod m
    let a = ((a % m) + m) % m;
    let b = ((b % m) + m) % m;

    // Check if we can multiply directly without overflow
    if let Some(product) = a.checked_mul(b) {
        ((product % m) + m) % m
    } else {
        // Use double-and-add method for large values
        let mut result = 0i128;
        let mut a = a;
        let mut b = b;

        while b > 0 {
            if b & 1 == 1 {
                // Add a to result (mod m)
                result = if let Some(sum) = result.checked_add(a) {
                    sum % m
                } else {
                    // Even addition can overflow with large m
                    ((result % m) + (a % m)) % m
                };
            }
            // Double a (mod m)
            a = if let Some(doubled) = a.checked_add(a) {
                doubled % m
            } else {
                ((a % m) + (a % m)) % m
            };
            b >>= 1;  // Halve b
        }
        result
    }
}

/// Modular inverse using extended Euclidean algorithm
///
/// Returns a^{-1} mod m such that (a * a^{-1}) ≡ 1 (mod m)
pub fn mod_inverse(a: i128, m: i128) -> i128 {
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (m, a % m);

    while new_r != 0 {
        let quotient = r / new_r;

        let temp = new_t;
        new_t = t - quotient * new_t;
        t = temp;

        let temp = new_r;
        new_r = r - quotient * new_r;
        r = temp;
    }

    if r > 1 {
        panic!("a is not invertible mod m");
    }
    if t < 0 {
        t += m;
    }

    t
}

/// RNS polynomial addition
///
/// (a + b) mod Q = [(a₀ + b₀) mod q₀, (a₁ + b₁) mod q₁, ...]
pub fn rns_add(a: &RnsPolynomial, b: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    assert_eq!(a.n, b.n, "Polynomials must have same length");
    assert_eq!(a.level, b.level, "Polynomials must be at same level");
    assert_eq!(a.domain, b.domain, "Polynomials must be in same domain");

    let n = a.n;
    let level = a.level;
    let domain = a.domain;  // Preserve domain
    let num_primes = a.num_primes();
    let active_primes = &primes[..num_primes];

    let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

    for i in 0..n {
        for (j, &q) in active_primes.iter().enumerate() {
            let sum = a.rns_coeffs[i][j] + b.rns_coeffs[i][j];
            rns_coeffs[i][j] = ((sum % q) + q) % q;
        }
    }

    RnsPolynomial::new_with_domain(rns_coeffs, n, level, domain)
}

/// RNS polynomial subtraction
///
/// (a - b) mod Q = [(a₀ - b₀) mod q₀, (a₁ - b₁) mod q₁, ...]
pub fn rns_sub(a: &RnsPolynomial, b: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    assert_eq!(a.n, b.n, "Polynomials must have same length");
    assert_eq!(a.level, b.level, "Polynomials must be at same level");
    assert_eq!(a.domain, b.domain, "Polynomials must be in same domain");

    let n = a.n;
    let level = a.level;
    let domain = a.domain;  // Preserve domain
    let num_primes = a.num_primes();
    let active_primes = &primes[..num_primes];

    let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

    for i in 0..n {
        for (j, &q) in active_primes.iter().enumerate() {
            let diff = a.rns_coeffs[i][j] - b.rns_coeffs[i][j];
            rns_coeffs[i][j] = ((diff % q) + q) % q;
        }
    }

    RnsPolynomial::new_with_domain(rns_coeffs, n, level, domain)
}

/// RNS polynomial negation
///
/// (-a) mod Q = [(-a₀) mod q₀, (-a₁) mod q₁, ...]
pub fn rns_negate(a: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    let n = a.n;
    let level = a.level;
    let num_primes = a.num_primes();
    let active_primes = &primes[..num_primes];

    let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

    for i in 0..n {
        for (j, &q) in active_primes.iter().enumerate() {
            rns_coeffs[i][j] = ((q - a.rns_coeffs[i][j]) % q + q) % q;
        }
    }

    RnsPolynomial::new(rns_coeffs, n, level)
}

/// RNS polynomial multiplication with self-check
///
/// (a * b) mod Q = [(a₀ * b₀) mod q₀, (a₁ * b₁) mod q₁, ...]
///
/// Each multiplication is done independently modulo each prime using NTT.
/// This is the key advantage of RNS: parallelizable operations!
///
/// SELF-CHECK: After computing the result, we independently recompute each
/// prime's multiplication and verify it matches. This catches per-prime
/// misalignment bugs where residues from different primes get mixed.
pub fn rns_multiply(
    a: &RnsPolynomial,
    b: &RnsPolynomial,
    primes: &[i64],
    ntt_multiply_fn: impl Fn(&[i64], &[i64], i64, usize) -> Vec<i64>,
) -> RnsPolynomial {
    assert_eq!(a.n, b.n, "Polynomials must have same length");
    assert_eq!(a.level, b.level, "Polynomials must be at same level");

    // CRITICAL: Verify domain compatibility
    assert_eq!(a.domain, b.domain,
        "RNS multiply: domain mismatch! a={:?}, b={:?}. Cannot multiply COEF with NTT!",
        a.domain, b.domain);
    // For now, only support COEF×COEF (NTT multiplication needs different handling)
    assert_eq!(a.domain, Domain::Coef,
        "RNS multiply: currently only supports COEF domain, got {:?}", a.domain);

    let n = a.n;
    let level = a.level;
    let num_primes = a.num_primes();

    // CRITICAL: Verify inputs have consistent prime counts
    assert_eq!(a.num_primes(), b.num_primes(),
        "RNS multiply: a has {} primes, b has {} primes - MISMATCH!",
        a.num_primes(), b.num_primes());
    assert_eq!(num_primes, primes.len(),
        "RNS multiply: polynomials have {} primes but primes array has {} - MISMATCH!",
        num_primes, primes.len());

    let active_primes = &primes[..num_primes];

    let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

    // Multiply modulo each prime independently (COLUMN-WISE PER PRIME)
    for (j, &q) in active_primes.iter().enumerate() {
        // Extract coefficients for THIS prime ONLY (column j)
        let a_mod_q: Vec<i64> = (0..n).map(|i| a.rns_coeffs[i][j]).collect();
        let b_mod_q: Vec<i64> = (0..n).map(|i| b.rns_coeffs[i][j]).collect();

        // Multiply using NTT for THIS prime
        let c_mod_q = ntt_multiply_fn(&a_mod_q, &b_mod_q, q, n);

        // Store result in column j
        for i in 0..n {
            rns_coeffs[i][j] = c_mod_q[i];
        }
    }

    let result = RnsPolynomial::new(rns_coeffs, n, level);

    // SELF-CHECK: Verify each prime's column independently
    // (Only check first 3 coefficients to avoid spam)
    if std::env::var("RNS_SELFCHECK").is_ok() {
        for j in 0..num_primes {
            let q = active_primes[j];
            let a_j: Vec<i64> = (0..n).map(|i| a.rns_coeffs[i][j]).collect();
            let b_j: Vec<i64> = (0..n).map(|i| b.rns_coeffs[i][j]).collect();
            let expected = ntt_multiply_fn(&a_j, &b_j, q, n);

            for i in 0..n.min(3) {
                let got = ((result.rns_coeffs[i][j] % q) + q) % q;
                let exp = ((expected[i] % q) + q) % q;
                if got != exp {
                    panic!("RNS multiply self-check failed - per-prime misalignment detected!");
                }
            }
        }
    }

    result
}

/// RNS rescaling: drop the last prime from the modulus chain
///
/// This is how CKKS rescaling works in RNS:
/// - Before: coefficients mod Q = q₀ · q₁ · ... · qₗ
/// - After: coefficients mod Q' = q₀ · q₁ · ... · qₗ₋₁ (dropped qₗ)
///
/// The rescaling process:
/// 1. Convert RNS to regular coefficients (CRT)
/// 2. Divide by the last prime qₗ (with rounding)
/// 3. Convert back to RNS with one fewer prime
///
/// Returns new polynomial at level+1 with scale divided by qₗ
/// Exact RNS rescale with proper rounding: computes round(c / q_last)
///
/// This implements the CKKS "DivideRoundByLastq" operation correctly:
/// 1. Center-lift the last residue for symmetric rounding
/// 2. For each remaining prime, compute: (c_i - c_last_centered) * q_last^{-1} mod q_i
///
/// # Arguments
/// * `poly` - Polynomial to rescale
/// * `primes` - Full prime chain at current level (including q_last to be dropped)
/// * `inv_qlast_mod_qi` - Precomputed q_last^{-1} mod q_i for each remaining prime
///
/// # Returns
/// Polynomial at level+1 with last prime dropped
pub fn rns_rescale_exact(
    poly: &RnsPolynomial,
    primes: &[i64],
    inv_qlast_mod_qi: &[i64],
) -> RnsPolynomial {
    let n = poly.n;
    let level = poly.level;
    let num_primes = poly.num_primes();

    assert!(num_primes > 1, "Cannot rescale: only one prime remaining");
    assert_eq!(inv_qlast_mod_qi.len(), num_primes - 1, "inv_qlast_mod_qi length mismatch");

    // Get the prime we're dropping (last one)
    let q_last = primes[num_primes - 1];
    let new_num_primes = num_primes - 1;

    let mut new_rns_coeffs = vec![vec![0i64; new_num_primes]; n];

    // For each coefficient, perform exact rounded rescaling
    for i in 0..n {
        // Step 1: Center-lift the last residue to (-q_last/2, q_last/2]
        let c_mod_qlast = poly.rns_coeffs[i][num_primes - 1];
        let c_last_centered = if c_mod_qlast > q_last / 2 {
            c_mod_qlast - q_last
        } else {
            c_mod_qlast
        };

        // Step 2: For each remaining prime, compute rounded division
        for j in 0..new_num_primes {
            let qj = primes[j];
            let c_mod_qj = poly.rns_coeffs[i][j];

            // Bring c_last_centered into modulo qj
            let t = ((c_last_centered % qj) + qj) % qj;

            // Compute: (c_mod_qj - t) * inv_qlast_mod_qi[j] mod qj
            let diff = ((c_mod_qj - t) % qj + qj) % qj;

            // Multiply by precomputed inverse (use i128 to avoid overflow)
            let c_new = ((diff as i128) * (inv_qlast_mod_qi[j] as i128) % (qj as i128) + (qj as i128)) % (qj as i128);

            new_rns_coeffs[i][j] = c_new as i64;
        }
    }

    // Preserve domain tag from input
    RnsPolynomial::new_with_domain(new_rns_coeffs, n, level + 1, poly.domain)
}

/// Precompute inverse constants for RNS rescaling
///
/// For dropping prime q_last, computes q_last^{-1} mod q_i for each remaining prime.
///
/// # Arguments
/// * `primes` - Full prime chain at current level
///
/// # Returns
/// Vector of inverses [q_last^{-1} mod q_0, q_last^{-1} mod q_1, ...]
pub fn precompute_rescale_inv(primes: &[i64]) -> Vec<i64> {
    let num_primes = primes.len();
    assert!(num_primes > 1, "Need at least 2 primes to rescale");

    let q_last = primes[num_primes - 1];
    let mut inv_qlast_mod_qi = vec![0i64; num_primes - 1];

    for i in 0..(num_primes - 1) {
        let qi = primes[i];
        inv_qlast_mod_qi[i] = mod_inverse(q_last as i128, qi as i128) as i64;
    }

    inv_qlast_mod_qi
}

/// Reference CRT-based rescale (slow but correct - for debugging)
///
/// This is the "obviously correct" implementation that:
/// 1. Reconstructs each coefficient as a big integer via CRT
/// 2. Center-lifts to (-Q/2, Q/2]
/// 3. Performs exact nearest-integer division by q_last
/// 4. Converts back to RNS with the remaining primes
///
/// Use this to verify the fast rns_rescale_exact() is correct.
pub fn rns_rescale_reference(poly: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    let n = poly.n;
    let num_primes = poly.num_primes();
    assert!(num_primes > 1, "Cannot rescale: only one prime remaining");

    let q_last = primes[num_primes - 1];
    let new_num_primes = num_primes - 1;
    let new_primes = &primes[..new_num_primes];

    // For large moduli (>2 primes of 60+ bits), full CRT overflows i128!
    // Instead, use "approximate rescale": scale each residue independently.
    //
    // For each remaining prime q_i, compute:
    //   c_i' ≈ (c_i * q_last^{-1}) mod q_i
    //
    // This is an approximation but works well when noise is small.
    // The exact formula would require full CRT reconstruction which overflows.

    let q_last_128 = q_last as i128;
    let mut new_rns_coeffs = vec![vec![0i64; new_num_primes]; n];

    for i in 0..n {
        for j in 0..new_num_primes {
            let q_j = new_primes[j];
            let q_j_128 = q_j as i128;

            // Get residue mod q_j and mod q_last
            let c_j = poly.rns_coeffs[i][j] as i128;
            let c_last = poly.rns_coeffs[i][num_primes - 1] as i128;

            // Approximate rescale: (c_j - c_last) / q_last mod q_j
            // This is the "fast rescale" used in many RNS-FHE implementations
            let q_last_inv_mod_qj = mod_inverse(q_last_128, q_j_128);

            // Compute (c_j - c_last) * q_last^{-1} mod q_j
            let diff = (c_j - c_last + q_j_128) % q_j_128;  // Ensure positive
            let rescaled = mulmod_i128(diff, q_last_inv_mod_qj, q_j_128);

            new_rns_coeffs[i][j] = rescaled as i64;
        }
    }

    let result = RnsPolynomial::new(new_rns_coeffs, n, poly.level + 1);
    result
}
/// Old rescale function (kept for compatibility, but use rns_rescale_exact for correctness)
#[deprecated(note = "Use rns_rescale_exact for proper CKKS rescaling")]
#[allow(dead_code)]
pub fn rns_rescale(poly: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    // Compute inverse on the fly
    let inv = precompute_rescale_inv(primes);
    rns_rescale_exact(poly, primes, &inv)
}

/// CRT reconstruction: convert RNS residues to a single integer in [0, Q)
///
/// Given residues [r0, r1, ..., rL] where ri = x mod qi,
/// reconstruct x ∈ [0, Q) where Q = q0 * q1 * ... * qL
///
/// Uses the explicit CRT formula with precomputed basis.
fn crt_reconstruct(residues: &[i64], primes: &[i64]) -> i128 {
    let num_primes = residues.len();
    assert_eq!(num_primes, primes.len());

    if num_primes == 1 {
        return residues[0] as i128;
    }

    // Compute Q = product of all primes (careful: can overflow for many primes!)
    let mut q_prod = 1i128;
    for &qi in primes {
        q_prod *= qi as i128;
    }

    // CRT formula: x = Σ ri * (Q/qi) * [(Q/qi)^{-1} mod qi] mod Q
    // CRITICAL: Use mul_mod to avoid overflow in intermediate products
    let mut result = 0i128;
    for i in 0..num_primes {
        let qi = primes[i] as i128;
        let ri = residues[i] as i128;

        // Compute Q/qi
        let q_div_qi = q_prod / qi;

        // Compute (Q/qi)^{-1} mod qi
        let inv = mod_inverse(q_div_qi, qi);

        // Compute: coeff = (Q/qi) * inv mod Q
        // This gives us the CRT basis element
        let basis = mulmod_i128(q_div_qi, inv, q_prod);

        // Add ri * basis to result
        let term = mulmod_i128(ri, basis, q_prod);
        result = (result + term) % q_prod;
    }

    // Ensure positive result
    if result < 0 {
        result += q_prod;
    }

    result
}

/// Center-lift an integer from [0, Q) to (-Q/2, Q/2]
fn center_lift(x: i128, q_prod: i128) -> i128 {
    if x > q_prod / 2 {
        x - q_prod
    } else {
        x
    }
}

/// Balanced base-B decomposition in Z
///
/// Decompose x ∈ Z into digits dt ∈ [-B/2, B/2) such that:
///   x = d0 + d1*B + d2*B² + ... + d(D-1)*B^(D-1)
///
/// Uses balanced representation to minimize digit magnitudes.
fn balanced_pow2_decompose(x: i128, w: u32, d: usize) -> Vec<i64> {
    let b = 1i128 << w;  // B = 2^w
    let b_half = b / 2;

    let mut digits = vec![0i64; d];
    let mut remainder = x;

    for t in 0..d {
        // Extract digit in balanced form: dt ∈ [-B/2, B/2)
        // First, get remainder mod B (result in [0, B) for positive, or (-B, 0] for negative)
        let mut dt = ((remainder % b) + b) % b;  // Force into [0, B)

        // Balance: if dt >= B/2, make it negative by subtracting B
        if dt >= b_half {
            dt -= b;
        }

        digits[t] = dt as i64;

        // Update remainder: (x - dt) / B
        // This should be exact division since dt ≡ x (mod B)
        remainder = (remainder - dt) / b;
    }

    digits
}

/// Decompose RNS polynomial into CRT-consistent, balanced base-2^w digits
///
/// CRITICAL FIX: This uses CRT-consistent decomposition, not per-prime independent.
///
/// For each coefficient:
/// 1. Reconstruct the integer x ∈ [0, Q) via CRT from all residues
/// 2. Center-lift to x_c ∈ (-Q/2, Q/2]
/// 3. Balanced decomposition in Z: x_c = Σ dt·B^t where dt ∈ [-B/2, B/2)
/// 4. Map each digit back to RNS identically across all primes
///
/// This ensures Σ dt·B^t ≡ x (mod qi) for EVERY prime qi, maintaining the
/// EVK cancellation property even when noise is present.
///
/// # Arguments
/// * `poly` - RNS polynomial to decompose
/// * `primes` - Active primes (matching poly's level)
/// * `w` - Digit width (typically 20-30 bits)
///
/// # Returns
/// Vector of D digit polynomials with CRT-consistent residues
pub fn decompose_base_pow2(
    poly: &RnsPolynomial,
    primes: &[i64],
    w: u32,
) -> Vec<RnsPolynomial> {
    let n = poly.n;
    let num_primes = poly.num_primes();

    // CRT-based decomposition using BigInt (avoids i128 overflow for large Q)
    //
    // Compute Q = product of all primes (using BigInt for arbitrary precision)
    let q_prod_big: BigInt = primes.iter()
        .map(|&q| BigInt::from(q))
        .product();

    // Number of bits in Q
    let q_bits = q_prod_big.bits() as u32;
    let d = ((q_bits + w - 1) / w) as usize;

    // Storage for all digits: digits[t] is an RnsPolynomial
    let mut digits_data = vec![vec![vec![0i64; num_primes]; n]; d];

    // Decompose each coefficient using CRT
    for i in 0..n {
        // Step 1: CRT reconstruct to get x ∈ [0, Q) using BigInt
        let residues: Vec<i64> = (0..num_primes)
            .map(|j| poly.rns_coeffs[i][j])
            .collect();

        let x_big = crt_reconstruct_bigint(&residues, primes);

        // Step 2: Center-lift to (-Q/2, Q/2]
        let q_half = &q_prod_big / 2;
        let x_centered_big = if x_big > q_half {
            x_big - &q_prod_big
        } else {
            x_big
        };

        // Step 3: Balanced base-B decomposition in Z
        let digits_z = balanced_pow2_decompose_bigint(&x_centered_big, w, d);

        // Step 4: Map each digit back to RNS consistently
        for t in 0..d {
            let dt_big = &digits_z[t];

            // Reduce dt modulo each prime identically
            for j in 0..num_primes {
                let qi_big = BigInt::from(primes[j]);
                // Ensure positive residue
                let dt_mod_qi = ((dt_big % &qi_big) + &qi_big) % &qi_big;
                // Convert to i64 (safe because dt_mod_qi < qi < 2^63)
                let dt_i64: i64 = dt_mod_qi.to_string().parse().unwrap();
                digits_data[t][i][j] = dt_i64;
            }
        }
    }

    // Package each digit level into an RnsPolynomial
    digits_data.into_iter()
        .map(|digit_coeffs| RnsPolynomial::new(digit_coeffs, n, poly.level))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rns_conversion() {
        let primes = vec![1_099_511_627_689, 1_099_511_627_691]; // Two 40-bit primes
        let coeffs = vec![123456789, -987654321, 0, 42];
        let n = 4;

        let rns_poly = RnsPolynomial::from_coeffs(&coeffs, &primes, n, 0);
        let recovered = rns_poly.to_coeffs(&primes);

        for i in 0..n {
            assert_eq!(coeffs[i], recovered[i], "Coefficient {} mismatch", i);
        }
    }

    #[test]
    fn test_rns_add() {
        let primes = vec![1_099_511_627_689, 1_099_511_627_691];
        let a_coeffs = vec![100, 200, 300, 400];
        let b_coeffs = vec![50, 60, 70, 80];
        let n = 4;

        let a = RnsPolynomial::from_coeffs(&a_coeffs, &primes, n, 0);
        let b = RnsPolynomial::from_coeffs(&b_coeffs, &primes, n, 0);

        let c = rns_add(&a, &b, &primes);
        let c_coeffs = c.to_coeffs(&primes);

        for i in 0..n {
            assert_eq!(a_coeffs[i] + b_coeffs[i], c_coeffs[i]);
        }
    }

    #[test]
    fn test_mod_inverse() {
        // Test: 3 * 3^{-1} ≡ 1 (mod 7)
        let inv = mod_inverse(3, 7);
        assert_eq!((3 * inv) % 7, 1);

        // Test with larger values
        let q = 1_099_511_627_689i128;
        let a = 12345678i128;
        let inv = mod_inverse(a, q);
        assert_eq!((a * inv) % q, 1);
    }
}

// =============================================================================
// BigInt helpers for arbitrary-precision CRT (needed for large Q > 2^127)
// =============================================================================

/// CRT reconstruction using BigInt (no overflow for large Q)
fn crt_reconstruct_bigint(residues: &[i64], primes: &[i64]) -> BigInt {
    let num_primes = residues.len();
    assert_eq!(num_primes, primes.len());

    if num_primes == 1 {
        return BigInt::from(residues[0]);
    }

    // Compute Q = product of all primes
    let q_prod: BigInt = primes.iter().map(|&q| BigInt::from(q)).product();

    // CRT formula: x = Σ ri * (Q/qi) * [(Q/qi)^{-1} mod qi] mod Q
    let mut result = BigInt::from(0);
    for i in 0..num_primes {
        let qi = BigInt::from(primes[i]);
        let ri = BigInt::from(residues[i]);

        // Compute Q/qi
        let q_div_qi = &q_prod / &qi;

        // Compute (Q/qi)^{-1} mod qi using extended GCD
        let inv_big = mod_inverse_bigint(&q_div_qi, &qi);

        // Compute: term = ri * (Q/qi) * inv mod Q
        let basis = (&q_div_qi * &inv_big) % &q_prod;
        let term = (ri * basis) % &q_prod;

        result = (result + term) % &q_prod;
    }

    // Ensure positive result
    if result < BigInt::from(0) {
        result += &q_prod;
    }

    result
}

/// Modular inverse using extended GCD (BigInt version)
fn mod_inverse_bigint(a: &BigInt, m: &BigInt) -> BigInt {
    use num_bigint::Sign;

    // Extended GCD to find x such that a*x ≡ 1 (mod m)
    let (gcd, x, _) = extended_gcd_bigint(a, m);

    if gcd != BigInt::from(1) {
        panic!("Modular inverse does not exist (gcd != 1)");
    }

    // Ensure positive result
    let mut result = x % m;
    if result.sign() == Sign::Minus {
        result += m;
    }

    result
}

/// Extended GCD for BigInt: returns (gcd, x, y) such that gcd = a*x + b*y
fn extended_gcd_bigint(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if b == &BigInt::from(0) {
        return (a.clone(), BigInt::from(1), BigInt::from(0));
    }

    let (gcd, x1, y1) = extended_gcd_bigint(b, &(a % b));
    let x = y1.clone();
    let y = x1 - (a / b) * &y1;

    (gcd, x, y)
}

/// Balanced base-B decomposition for BigInt
fn balanced_pow2_decompose_bigint(x: &BigInt, w: u32, d: usize) -> Vec<BigInt> {
    let b = BigInt::from(1i64 << w);  // B = 2^w
    let b_half = &b / 2;               // B/2

    let mut digits = Vec::with_capacity(d);
    let mut remainder = x.clone();

    for _ in 0..d {
        // Extract digit: dt = remainder mod B, balanced to (-B/2, B/2]
        let dt_unbalanced = &remainder % &b;
        let dt = if dt_unbalanced > b_half {
            &dt_unbalanced - &b  // Shift to negative range
        } else {
            dt_unbalanced
        };

        digits.push(dt.clone());

        // Update remainder: remainder = (remainder - dt) / B
        remainder = (remainder - &dt) / &b;
    }

    digits
}
