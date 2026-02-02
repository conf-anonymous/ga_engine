//! Galois Automorphisms for Homomorphic Rotation on Metal GPU
//!
//! **Purpose:** Enable CKKS slot rotations on GPU ciphertexts for bootstrap operations.
//!
//! **Mathematical Foundation:**
//! - CKKS operates in the ring R = ℤ[X]/(X^N + 1) where N is a power of 2
//! - Galois automorphism σ_k: X → X^k permutes CKKS slots
//! - For power-of-two cyclotomics, the Galois group is ℤ*_{2N} ≅ ℤ/2 × ℤ/(N/2)
//! - Column rotation by r steps uses k = 5^r (mod 2N) where 5 is the generator
//! - Conjugation (row rotation) uses k = 2N-1
//!
//! **Why This Module Exists:**
//! V3 bootstrap (CoeffToSlot/SlotToCoeff) requires 24 rotations for N=1024.
//! Without GPU rotation, these operations force GPU→CPU→GPU conversions,
//! killing performance. This module enables rotations entirely on Metal GPU.
//!
//! **Reference:** Halevi & Shoup 2014 (BGV/CKKS bootstrapping with automorphisms)

use std::collections::HashMap;

/// Precompute Galois automorphism map for X → X^k in R = ℤ[X]/(X^N + 1)
///
/// Computes the permutation and sign corrections needed to apply σ_k to a polynomial.
///
/// # Mathematical Details
///
/// For polynomial f(X) = Σ f_i X^i, the automorphism σ_k gives:
/// ```
/// σ_k(f)(X) = f(X^k) = Σ f_i X^(i·k)
/// ```
///
/// In the quotient ring R = ℤ[X]/(X^N + 1), we have X^N = -1, so:
/// - If i·k mod 2N < N:  X^(i·k) stays as X^(i·k mod N) with sign +1
/// - If i·k mod 2N ≥ N:  X^(i·k) becomes -X^(i·k mod N) with sign -1
///
/// # Returns
///
/// Tuple of (permutation_map, sign_map) where:
/// - `permutation_map[i]` = target index for coefficient i
/// - `sign_map[i]` = +1 (keep sign) or -1 (negate)
///
/// # Example
///
/// For N=4, k=3 (rotation by 1 step):
/// ```
/// i=0: 0·3 = 0  → index 0, sign +1
/// i=1: 1·3 = 3  → index 3, sign +1
/// i=2: 2·3 = 6  → 6 mod 4 = 2, sign -1 (since 6 ≥ N)
/// i=3: 3·3 = 9  → 9 mod 8 = 1, sign -1 (but 9 mod 4 = 1, and 9 ≥ N)
/// ```
/// Returns: ([0, 3, 2, 1], [1, 1, -1, -1])
///
/// # Arguments
///
/// * `n` - Ring dimension (polynomial degree N, must be power of 2)
/// * `k` - Galois element (must be in ℤ*_{2N}, i.e., gcd(k, 2N) = 1)
///
/// # Panics
///
/// Panics if n is not a power of 2 or if k is not coprime to 2N.
pub fn compute_galois_map(n: usize, k: usize) -> (Vec<u32>, Vec<i32>) {
    assert!(n.is_power_of_two(), "N must be a power of 2 for CKKS");

    let two_n = 2 * n;

    // Verify k is in the Galois group (coprime to 2N)
    assert!(
        gcd(k, two_n) == 1,
        "k = {} must be coprime to 2N = {} for valid automorphism",
        k,
        two_n
    );

    let mut perm = vec![0u32; n];
    let mut signs = vec![1i32; n];

    for i in 0..n {
        // Compute i·k mod 2N
        let ik = ((i * k) % two_n) as usize;

        if ik < n {
            // X^(i·k) stays positive
            perm[i] = ik as u32;
            signs[i] = 1;
        } else {
            // X^(i·k) = X^(ik) = X^(ik - N) · X^N = -X^(ik - N)
            // (since X^N = -1 in quotient ring)
            perm[i] = (ik - n) as u32;
            signs[i] = -1;
        }
    }

    (perm, signs)
}

/// Convert rotation step to Galois element k
///
/// For power-of-two cyclotomics (CKKS with N = 2^m), the Galois group generator is g = 5.
/// A rotation by r steps corresponds to the automorphism σ_k where:
/// - k = 5^r (mod 2N) for r ≥ 0
/// - k = 5^(N/2 + r) (mod 2N) for r < 0 (using φ(2N) = N for power-of-two N)
///
/// # Arguments
///
/// * `step` - Number of slots to rotate (positive = left, negative = right)
/// * `n` - Ring dimension N
///
/// # Returns
///
/// Galois element k such that σ_k rotates by `step` slots
///
/// # Example
///
/// ```
/// let n = 1024;
/// let k = rotation_step_to_galois_element(1, n);  // Rotate left by 1
/// // For N=1024: k = 5 (since 5^1 mod 2048 = 5)
///
/// let k_neg = rotation_step_to_galois_element(-1, n);  // Rotate right by 1
/// // For N=1024: k = 5^(512 - 1) mod 2048 = 5^511 mod 2048
/// ```
pub fn rotation_step_to_galois_element(step: i32, n: usize) -> usize {
    let two_n = 2 * n;
    let g = 5usize; // Generator for power-of-two cyclotomics

    if step >= 0 {
        // Positive rotation: k = 5^step mod 2N
        pow_mod(g, step as usize, two_n)
    } else {
        // Negative rotation: k = 5^(-step) = 5^(φ(2N) + step) mod 2N
        // For power-of-two N, φ(2N) = N
        let phi = n;
        let adjusted_step = (phi as i32 + step) as usize;
        pow_mod(g, adjusted_step, two_n)
    }
}

/// Get conjugation Galois element (flips complex conjugate in slots)
///
/// For CKKS, conjugation is the automorphism σ_{2N-1}: X → X^(2N-1) = X^(-1).
/// This swaps the complex conjugate pairs in the canonical embedding.
///
/// # Arguments
///
/// * `n` - Ring dimension N
///
/// # Returns
///
/// Galois element k for conjugation (always k = 2N-1)
pub fn conjugation_galois_element(n: usize) -> usize {
    2 * n - 1
}

/// Check if a Galois element is valid for ring dimension N
///
/// A Galois element k is valid if gcd(k, 2N) = 1 (i.e., k ∈ ℤ*_{2N}).
pub fn is_valid_galois_element(k: usize, n: usize) -> bool {
    gcd(k, 2 * n) == 1
}

/// Compute modular exponentiation: base^exp mod modulus
///
/// Uses binary exponentiation for efficiency.
fn pow_mod(base: usize, exp: usize, modulus: usize) -> usize {
    let mut result = 1;
    let mut b = base % modulus;
    let mut e = exp;

    while e > 0 {
        if e & 1 == 1 {
            result = (result * b) % modulus;
        }
        b = (b * b) % modulus;
        e >>= 1;
    }

    result
}

/// Compute greatest common divisor (Euclidean algorithm)
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Precompute all rotation keys needed for bootstrap
///
/// For N=1024 bootstrap (CoeffToSlot + SlotToCoeff), we need rotations:
/// - Powers of 2: [±1, ±2, ±4, ±8, ..., ±N/2]
/// - This is log₂(N) + 1 rotations per direction = ~24 total for N=1024
///
/// # Arguments
///
/// * `n` - Ring dimension N
///
/// # Returns
///
/// Vec of rotation steps needed for bootstrap
pub fn compute_bootstrap_rotation_steps(n: usize) -> Vec<i32> {
    let logn = (n as f64).log2() as usize;
    let mut steps = Vec::new();

    // Rotations by powers of 2 (both directions)
    for i in 0..=logn {
        let step = 1 << i; // 2^i
        if step <= (n / 2) {
            steps.push(step as i32);
            steps.push(-(step as i32));
        }
    }

    // Remove duplicates
    steps.sort_unstable();
    steps.dedup();

    steps
}

/// Galois element cache for fast lookup
///
/// Bootstrap operations repeatedly use the same rotations, so we cache
/// the Galois elements to avoid recomputation.
pub struct GaloisElementCache {
    /// Maps rotation step → Galois element k
    cache: HashMap<i32, usize>,

    /// Ring dimension N
    n: usize,
}

impl GaloisElementCache {
    /// Create new cache for ring dimension N
    pub fn new(n: usize) -> Self {
        Self {
            cache: HashMap::new(),
            n,
        }
    }

    /// Get Galois element for rotation step (with caching)
    pub fn get(&mut self, step: i32) -> usize {
        if let Some(&k) = self.cache.get(&step) {
            return k;
        }

        let k = rotation_step_to_galois_element(step, self.n);
        self.cache.insert(step, k);
        k
    }

    /// Precompute all bootstrap rotation elements
    pub fn precompute_bootstrap_rotations(&mut self) {
        let steps = compute_bootstrap_rotation_steps(self.n);
        for step in steps {
            self.get(step); // Populate cache
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_galois_map_identity() {
        // σ_1 should be the identity (no permutation)
        let n = 8;
        let k = 1;
        let (perm, signs) = compute_galois_map(n, k);

        // Identity: perm[i] = i, signs[i] = 1
        for i in 0..n {
            assert_eq!(perm[i], i as u32, "σ_1 should be identity");
            assert_eq!(signs[i], 1, "σ_1 should have all positive signs");
        }
    }

    #[test]
    fn test_compute_galois_map_rotation() {
        // For N=4, k=3 should rotate by 1 slot
        let n = 4;
        let k = 3;
        let (perm, signs) = compute_galois_map(n, k);

        // Expected permutation for k=3:
        // i=0: 0·3=0  → perm=0, sign=+1
        // i=1: 1·3=3  → perm=3, sign=+1
        // i=2: 2·3=6  → 6 mod 8 = 6 ≥ 4, so perm=(6-4)=2, sign=-1
        // i=3: 3·3=9  → 9 mod 8 = 1, but 9 ≥ 4, so perm=1, sign=-1
        // Wait, 9 mod 8 = 1, which is < 4, so perm=1, sign=+1? Let me recalculate:
        // Actually: i·k mod 2N gives us the exponent in [0, 2N)
        // 3·3 = 9, 9 mod 8 = 1 < 4, so perm=1, sign=+1
        // Let me recalculate all:
        // 0·3 mod 8 = 0 < 4 → perm=0, sign=+1
        // 1·3 mod 8 = 3 < 4 → perm=3, sign=+1
        // 2·3 mod 8 = 6 ≥ 4 → perm=6-4=2, sign=-1
        // 3·3 mod 8 = 1 < 4 → perm=1, sign=+1

        assert_eq!(perm, vec![0, 3, 2, 1]);
        assert_eq!(signs, vec![1, 1, -1, 1]);
    }

    #[test]
    fn test_rotation_step_to_galois_element() {
        let n = 1024;

        // Rotation by 0 should be identity (k=1)
        let k0 = rotation_step_to_galois_element(0, n);
        assert_eq!(k0, 1, "Rotation by 0 should give k=1 (identity)");

        // Rotation by 1 should give k=5
        let k1 = rotation_step_to_galois_element(1, n);
        assert_eq!(k1, 5, "Rotation by 1 should give k=5");

        // Rotation by 2 should give k=25
        let k2 = rotation_step_to_galois_element(2, n);
        assert_eq!(k2, 25, "Rotation by 2 should give k=5^2=25");

        // Verify negative rotations are valid
        let k_neg1 = rotation_step_to_galois_element(-1, n);
        assert!(
            is_valid_galois_element(k_neg1, n),
            "Negative rotation should give valid Galois element"
        );
    }

    #[test]
    fn test_conjugation_galois_element() {
        let n = 1024;
        let k_conj = conjugation_galois_element(n);

        // For N=1024, conjugation is k = 2·1024 - 1 = 2047
        assert_eq!(k_conj, 2047);

        // Verify it's a valid Galois element
        assert!(is_valid_galois_element(k_conj, n));
    }

    #[test]
    fn test_compute_bootstrap_rotation_steps() {
        let n = 16; // Small N for testing
        let steps = compute_bootstrap_rotation_steps(n);

        // For N=16: log₂(16) = 4, so rotations are ±1, ±2, ±4, ±8
        let expected = vec![-8, -4, -2, -1, 1, 2, 4, 8];
        assert_eq!(steps, expected);
    }

    #[test]
    fn test_galois_element_cache() {
        let n = 1024;
        let mut cache = GaloisElementCache::new(n);

        // First access should compute and cache
        let k1_first = cache.get(1);
        assert_eq!(k1_first, 5);

        // Second access should hit cache (verify by checking count)
        let k1_second = cache.get(1);
        assert_eq!(k1_second, 5);

        // Cache should have 1 entry
        assert_eq!(cache.cache.len(), 1);

        // Precompute bootstrap rotations
        cache.precompute_bootstrap_rotations();

        // Cache should now have ~24 entries for N=1024
        let expected_count = 2 * ((n as f64).log2() as usize + 1);
        assert!(
            cache.cache.len() >= expected_count - 2,
            "Cache should have ~{} entries, got {}",
            expected_count,
            cache.cache.len()
        );
    }

    #[test]
    fn test_pow_mod() {
        assert_eq!(pow_mod(5, 0, 2048), 1); // 5^0 = 1
        assert_eq!(pow_mod(5, 1, 2048), 5); // 5^1 = 5
        assert_eq!(pow_mod(5, 2, 2048), 25); // 5^2 = 25
        assert_eq!(pow_mod(2, 10, 1000), 24); // 2^10 mod 1000 = 1024 mod 1000 = 24
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 19), 1); // Coprime
        assert_eq!(gcd(100, 50), 50);
        assert_eq!(gcd(2048, 5), 1); // 5 is coprime to 2048 (valid generator)
    }
}
