//! SIMD Backend Trait Abstraction
//!
//! This module defines the trait interface that all SIMD backends must implement.
//! This allows runtime selection of the best backend (AVX2, NEON, or scalar fallback)
//! while maintaining a uniform API.

/// SIMD Backend Interface
///
/// All SIMD implementations (AVX2, NEON, scalar) must implement this trait.
/// This enables runtime polymorphism for selecting the best backend based on CPU features.
pub trait SimdBackend: Send + Sync {
    /// Returns the name of this backend (for debugging/logging)
    fn name(&self) -> &'static str;

    /// NTT butterfly operation (vectorized)
    ///
    /// Computes the core butterfly operation for Cooley-Tukey NTT:
    /// - `a[i] = a[i] + w * b[i]` (mod q)
    /// - `b[i] = a[i] - w * b[i]` (mod q)
    ///
    /// # Arguments
    /// * `a` - First array (modified in-place)
    /// * `b` - Second array (modified in-place)
    /// * `twiddle` - Twiddle factor w
    /// * `q` - Modulus
    /// * `len` - Number of elements to process
    ///
    /// # Safety
    /// - `a` and `b` must have at least `len` elements
    /// - All values must be < q
    ///
    /// # Performance
    /// - Scalar: processes 1 element per iteration
    /// - AVX2: processes 4 elements per iteration
    /// - NEON: processes 2-4 elements per iteration
    unsafe fn ntt_butterfly(
        &self,
        a: *mut u64,
        b: *mut u64,
        twiddle: u64,
        q: u64,
        len: usize,
    );

    /// Modular multiplication using Barrett reduction (vectorized)
    ///
    /// Computes `(a * b) mod q` for arrays of values.
    ///
    /// # Arguments
    /// * `result` - Output array
    /// * `a` - First input array
    /// * `b` - Second input array
    /// * `q` - Modulus
    /// * `barrett_k` - Barrett constant for this modulus
    /// * `len` - Number of elements
    ///
    /// # Barrett Reduction
    /// Barrett reduction is a fast alternative to division for modular reduction.
    /// Precomputes k = ⌊2^128 / q⌋, then approximates `(a * b) mod q` without division.
    ///
    /// # Safety
    /// All arrays must have at least `len` elements
    unsafe fn barrett_mul_mod(
        &self,
        result: *mut u64,
        a: *const u64,
        b: *const u64,
        q: u64,
        barrett_k: u128,
        len: usize,
    );

    /// Modular addition (vectorized)
    ///
    /// Computes `(a + b) mod q` for arrays.
    ///
    /// # Safety
    /// All arrays must have at least `len` elements, values must be < q
    unsafe fn add_mod(
        &self,
        result: *mut u64,
        a: *const u64,
        b: *const u64,
        q: u64,
        len: usize,
    );

    /// Modular subtraction (vectorized)
    ///
    /// Computes `(a - b) mod q` for arrays.
    ///
    /// # Safety
    /// All arrays must have at least `len` elements, values must be < q
    unsafe fn sub_mod(
        &self,
        result: *mut u64,
        a: *const u64,
        b: *const u64,
        q: u64,
        len: usize,
    );

    /// Scalar multiplication with modular reduction (vectorized)
    ///
    /// Computes `(a[i] * scalar) mod q` for all elements in array.
    ///
    /// # Arguments
    /// * `result` - Output array
    /// * `a` - Input array
    /// * `scalar` - Scalar value to multiply by
    /// * `q` - Modulus
    /// * `barrett_k` - Barrett constant
    /// * `len` - Number of elements
    ///
    /// # Safety
    /// Arrays must have at least `len` elements
    unsafe fn scalar_mul_mod(
        &self,
        result: *mut u64,
        a: *const u64,
        scalar: u64,
        q: u64,
        barrett_k: u128,
        len: usize,
    );

    /// Montgomery multiplication (vectorized, exact arithmetic)
    ///
    /// Computes `(a * b * R^{-1}) mod q` for arrays of values in Montgomery form.
    /// **This provides exact modular multiplication** suitable for FHE operations,
    /// unlike Barrett reduction which has approximation errors for large primes.
    ///
    /// # Montgomery Form
    /// Values in Montgomery form: x̄ = x * R mod q, where R = 2^64.
    /// Montgomery multiplication: (ā * b̄) * R^{-1} mod q = (a * b) * R mod q
    ///
    /// # Arguments
    /// * `result` - Output array (in Montgomery form)
    /// * `a` - First input array (in Montgomery form)
    /// * `b` - Second input array (in Montgomery form)
    /// * `q` - Modulus (must be odd)
    /// * `q_prime` - Montgomery constant: -q^{-1} mod 2^64
    /// * `len` - Number of elements
    ///
    /// # Safety
    /// - All arrays must have at least `len` elements
    /// - Values must be in Montgomery form and < q
    /// - q must be odd (coprime with R = 2^64)
    ///
    /// # Performance
    /// - Exact arithmetic (no approximation errors)
    /// - SIMD-friendly (no 256-bit operations needed)
    /// - Faster than Barrett for large primes (60-bit FHE primes)
    unsafe fn montgomery_mul_mod(
        &self,
        result: *mut u64,
        a: *const u64,
        b: *const u64,
        q: u64,
        q_prime: u64,
        len: usize,
    );
}

/// Precompute Barrett constant for a given modulus
///
/// Barrett reduction for 64-bit multiplication uses k = ⌊2^128 / q⌋.
/// This allows fast modular reduction without actual division.
///
/// # Arguments
/// * `q` - The modulus (must be > 0)
///
/// # Returns
/// Barrett constant k (as u128)
///
/// # Algorithm
/// For product p = a * b (128-bit):
///   quotient ≈ ⌊(p * k) >> 128⌋
///   remainder = p - quotient * q
///
/// # Note
/// We return k mod 2^128, which is sufficient for the reduction algorithm.
#[inline]
pub fn compute_barrett_constant(q: u64) -> u128 {
    // We need k = ⌊2^128 / q⌋
    // Approximate as (2^128 - 1) / q + 1
    let q_128 = q as u128;
    (u128::MAX / q_128).wrapping_add(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrett_constant() {
        // Test with a 60-bit prime
        let q = 1152921504606830593u64; // 2^60 - 93 (prime)
        let k = compute_barrett_constant(q);

        // k = ⌊2^128 / q⌋ for 60-bit q should be approximately 2^68
        // For q ≈ 2^60, k ≈ 2^128 / 2^60 = 2^68 ≈ 295147905179352825856
        assert!(k > 0);
        assert!(k > 100_000_000_000_000_000_000u128); // Should be ~2^68

        // Test with smaller prime
        let q2 = 17u64;
        let k2 = compute_barrett_constant(q2);
        // 2^128 / 17 ≈ 20_000_000_000_000_000_000_000_000_000_000_000_000
        assert!(k2 > 1_000_000_000_000_000_000u128);
    }

    #[test]
    fn test_barrett_constant_properties() {
        let test_primes = vec![
            1099511678977u64,  // 41-bit prime from V2 params
            1141392289560813569u64, // 60-bit prime from V2 params
        ];

        for &q in &test_primes {
            let k = compute_barrett_constant(q);

            // Verify k is reasonable for this modulus size
            assert!(k > 0);

            // For a b-bit modulus, k should be approximately 2^(64-b)
            let log_q = 64 - q.leading_zeros();
            let expected_log_k = 64 - log_q;

            let actual_log_k = 64 - k.leading_zeros();

            // Allow some tolerance
            assert!(
                actual_log_k >= expected_log_k.saturating_sub(2),
                "k too small for modulus q={}, log_k={}, expected≈{}",
                q,
                actual_log_k,
                expected_log_k
            );
        }
    }
}
