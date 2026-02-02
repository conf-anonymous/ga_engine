//! Galois Automorphisms for CKKS Slot Rotations
//!
//! This module implements Galois automorphisms which are the mathematical foundation
//! for rotating SIMD slots in CKKS. These are ring homomorphisms that permute slots.
//!
//! # Mathematical Foundation
//!
//! A Galois automorphism σₖ is defined as:
//! ```text
//! σₖ: R → R
//! σₖ(x) = x^k
//! ```
//!
//! Where k must be odd and gcd(k, 2N) = 1 for it to be a valid automorphism.
//!
//! # Slot Rotation
//!
//! For power-of-2 ring dimensions, rotating slots by r positions corresponds to
//! applying automorphism σₖ where k = 5^r mod M (M = 2N).
//!
//! - Rotate left by 1:  k = 5
//! - Rotate left by 2:  k = 25
//! - Rotate right by 1: k = 5^(-1) mod M
//!
//! # References
//!
//! - "Homomorphic Encryption for Arithmetic of Approximate Numbers" (CKKS paper)
//! - Galois automorphism implementations in standard FHE libraries

use std::collections::HashMap;

/// Apply Galois automorphism σₖ to polynomial
///
/// Computes p(x^k) in the ring R = Z[x]/(x^N + 1)
///
/// # Arguments
/// * `poly` - Input polynomial coefficients
/// * `k` - Automorphism index (must be odd, gcd(k, 2N) = 1)
/// * `n` - Ring dimension
///
/// # Returns
/// Polynomial with automorphism applied
///
/// # Example
/// ```rust,ignore
/// // Apply σ₅ (rotate slots left by 1)
/// let poly_rotated = apply_automorphism(&poly, 5, n);
/// ```
pub fn apply_automorphism(poly: &[i64], k: usize, n: usize) -> Vec<i64> {
    assert!(k % 2 == 1, "Automorphism index must be odd");
    assert_eq!(poly.len(), n, "Polynomial must have length n");

    let mut result = vec![0i64; n];

    // For each coefficient position i, we need to find where x^i maps to
    // under the automorphism x → x^k
    for i in 0..n {
        // Compute k*i mod 2N (because we're in ring Z[x]/(x^N + 1))
        let new_idx = (k * i) % (2 * n);

        if new_idx < n {
            // Coefficient stays positive
            result[new_idx] = poly[i];
        } else {
            // Negacyclic reduction: x^N = -1, so x^(N+j) = -x^j
            result[new_idx % n] = -poly[i];
        }
    }

    result
}

/// Get automorphism index for rotating slots by r positions
///
/// For CKKS with power-of-2 ring dimension, rotation by r slots
/// corresponds to automorphism with index k = 5^r mod M.
///
/// # Arguments
/// * `r` - Number of positions to rotate (positive = left, negative = right)
/// * `n` - Ring dimension
///
/// # Returns
/// Automorphism index k
///
/// # Example
/// ```rust,ignore
/// let k = rotation_to_automorphism(1, 8192);  // Rotate left by 1
/// let k_inv = rotation_to_automorphism(-1, 8192);  // Rotate right by 1
/// ```
pub fn rotation_to_automorphism(r: isize, n: usize) -> usize {
    let m = 2 * n; // Cyclotomic index M = 2N

    // Compute 5^r mod M
    // For negative r, we compute 5^(-|r|) = (5^(-1))^|r| mod M
    if r >= 0 {
        power_mod(5, r as usize, m)
    } else {
        // Compute 5^(-1) mod M first
        let five_inv = mod_inverse(5, m);
        power_mod(five_inv, (-r) as usize, m)
    }
}

/// Check if k is a valid automorphism index
///
/// For k to be a valid Galois automorphism in Z[x]/(x^N + 1):
/// 1. k must be odd
/// 2. gcd(k, 2N) = 1
pub fn is_valid_automorphism(k: usize, n: usize) -> bool {
    if k % 2 == 0 {
        return false; // Must be odd
    }

    let m = 2 * n;
    gcd(k, m) == 1
}

/// Precompute all automorphism indices for common rotations
///
/// This is useful for key generation - we compute which automorphisms
/// we'll need ahead of time.
///
/// # Arguments
/// * `rotation_steps` - List of rotation amounts needed
/// * `n` - Ring dimension
///
/// # Returns
/// HashMap mapping rotation amount → automorphism index
pub fn precompute_rotation_automorphisms(
    rotation_steps: &[isize],
    n: usize,
) -> HashMap<isize, usize> {
    let mut automorphisms = HashMap::new();

    for &r in rotation_steps {
        let k = rotation_to_automorphism(r, n);
        automorphisms.insert(r, k);
    }

    automorphisms
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute (base^exp) mod m using fast exponentiation
fn power_mod(base: usize, exp: usize, m: usize) -> usize {
    if exp == 0 {
        return 1;
    }

    let mut result = 1usize;
    let mut base = base % m;
    let mut exp = exp;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % m;
        }
        base = (base * base) % m;
        exp /= 2;
    }

    result
}

/// Compute modular inverse using extended Euclidean algorithm
///
/// Returns a such that (a * x) mod m = 1
fn mod_inverse(x: usize, m: usize) -> usize {
    let (mut old_r, mut r) = (x as isize, m as isize);
    let (mut old_s, mut s) = (1isize, 0isize);

    while r != 0 {
        let quotient = old_r / r;
        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;
    }

    // Make sure result is positive
    let result = if old_s < 0 {
        (old_s + m as isize) as usize
    } else {
        old_s as usize
    };

    result
}

/// Compute greatest common divisor using Euclidean algorithm
fn gcd(a: usize, b: usize) -> usize {
    let mut a = a;
    let mut b = b;

    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }

    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_automorphism_identity() {
        let n = 64;
        let poly = vec![1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        // Apply identity automorphism (k = 1)
        let result = apply_automorphism(&poly, 1, n);

        // Should be unchanged
        assert_eq!(result, poly);
    }

    #[test]
    fn test_automorphism_composition() {
        let n = 64;
        let poly: Vec<i64> = (0..n as i64).collect();

        // σₖ₁ ∘ σₖ₂ = σ_{k₁·k₂ mod M}
        let k1 = 5;
        let k2 = 3;
        let m = 2 * n;

        let result1 = apply_automorphism(&apply_automorphism(&poly, k1, n), k2, n);
        let k_composed = (k1 * k2) % m;
        let result2 = apply_automorphism(&poly, k_composed, n);

        assert_eq!(result1, result2, "Automorphism composition should match");
    }

    #[test]
    fn test_rotation_to_automorphism() {
        let n = 64;

        // Rotate left by 1 should give k = 5
        let k = rotation_to_automorphism(1, n);
        assert_eq!(k, 5);

        // Rotate left by 2 should give k = 25
        let k = rotation_to_automorphism(2, n);
        assert_eq!(k, 25);
    }

    #[test]
    fn test_rotation_inverse() {
        let n = 64;

        // Rotating left then right should give identity
        let k_left = rotation_to_automorphism(1, n);
        let k_right = rotation_to_automorphism(-1, n);

        let m = 2 * n;
        assert_eq!((k_left * k_right) % m, 1, "Left and right should be inverses");
    }

    #[test]
    fn test_is_valid_automorphism() {
        let n = 64;

        // 5 is valid (odd, gcd(5, 128) = 1)
        assert!(is_valid_automorphism(5, n));

        // 1 is valid (identity)
        assert!(is_valid_automorphism(1, n));

        // 2 is invalid (even)
        assert!(!is_valid_automorphism(2, n));

        // 64 is invalid (gcd(64, 128) = 64 ≠ 1)
        assert!(!is_valid_automorphism(64, n));
    }

    #[test]
    fn test_power_mod() {
        // 5^2 mod 128 = 25
        assert_eq!(power_mod(5, 2, 128), 25);

        // 5^3 mod 128 = 125
        assert_eq!(power_mod(5, 3, 128), 125);

        // 5^0 mod 128 = 1
        assert_eq!(power_mod(5, 0, 128), 1);
    }

    #[test]
    fn test_mod_inverse() {
        // 5 * mod_inverse(5, 128) ≡ 1 (mod 128)
        let inv = mod_inverse(5, 128);
        assert_eq!((5 * inv) % 128, 1);

        // 3 * mod_inverse(3, 10) ≡ 1 (mod 10)
        let inv = mod_inverse(3, 10);
        assert_eq!((3 * inv) % 10, 1);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(5, 128), 1);
        assert_eq!(gcd(64, 128), 64);
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(17, 19), 1); // Coprime
    }

    #[test]
    fn test_precompute_rotation_automorphisms() {
        let n = 64;
        let rotations = vec![-2, -1, 0, 1, 2];

        let auto_map = precompute_rotation_automorphisms(&rotations, n);

        assert_eq!(auto_map.len(), 5);
        assert!(auto_map.contains_key(&1));
        assert!(auto_map.contains_key(&-1));

        // Verify k=5 for rotation 1
        assert_eq!(auto_map[&1], 5);
    }
}
