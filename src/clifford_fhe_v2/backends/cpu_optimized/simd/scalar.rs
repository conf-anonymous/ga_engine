//! Scalar Fallback Implementation
//!
//! This module provides a pure Rust scalar implementation of all SIMD operations.
//! It serves two purposes:
//! 1. Fallback when no SIMD instructions are available
//! 2. Reference implementation for testing SIMD correctness
//!
//! All SIMD implementations must produce identical results to this scalar version.

use super::traits::SimdBackend;

/// Scalar backend (no SIMD, pure Rust)
///
/// This is the fallback implementation used when no SIMD instructions are available,
/// or for testing/verification purposes.
pub struct ScalarBackend;

impl ScalarBackend {
    /// Create a new scalar backend
    pub fn new() -> Self {
        ScalarBackend
    }
}

impl Default for ScalarBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdBackend for ScalarBackend {
    fn name(&self) -> &'static str {
        "Scalar"
    }

    unsafe fn ntt_butterfly(
        &self,
        a: *mut u64,
        b: *mut u64,
        twiddle: u64,
        q: u64,
        len: usize,
    ) {
        // Process one element at a time (no vectorization)
        for i in 0..len {
            let a_val = *a.add(i);
            let b_val = *b.add(i);

            // t = w * b[i] mod q
            let t = mul_mod(twiddle, b_val, q);

            // a[i] = a[i] + t mod q
            *a.add(i) = add_mod(a_val, t, q);

            // b[i] = a[i] - t mod q
            *b.add(i) = sub_mod(a_val, t, q);
        }
    }

    unsafe fn barrett_mul_mod(
        &self,
        result: *mut u64,
        a: *const u64,
        b: *const u64,
        q: u64,
        barrett_k: u128,
        len: usize,
    ) {
        for i in 0..len {
            let a_val = *a.add(i);
            let b_val = *b.add(i);

            // Use Barrett reduction for modular multiplication
            *result.add(i) = barrett_reduce(a_val, b_val, q, barrett_k);
        }
    }

    unsafe fn add_mod(
        &self,
        result: *mut u64,
        a: *const u64,
        b: *const u64,
        q: u64,
        len: usize,
    ) {
        for i in 0..len {
            let a_val = *a.add(i);
            let b_val = *b.add(i);
            *result.add(i) = add_mod(a_val, b_val, q);
        }
    }

    unsafe fn sub_mod(
        &self,
        result: *mut u64,
        a: *const u64,
        b: *const u64,
        q: u64,
        len: usize,
    ) {
        for i in 0..len {
            let a_val = *a.add(i);
            let b_val = *b.add(i);
            *result.add(i) = sub_mod(a_val, b_val, q);
        }
    }

    unsafe fn scalar_mul_mod(
        &self,
        result: *mut u64,
        a: *const u64,
        scalar: u64,
        q: u64,
        barrett_k: u128,
        len: usize,
    ) {
        for i in 0..len {
            let a_val = *a.add(i);
            *result.add(i) = barrett_reduce(a_val, scalar, q, barrett_k);
        }
    }

    unsafe fn montgomery_mul_mod(
        &self,
        result: *mut u64,
        a: *const u64,
        b: *const u64,
        q: u64,
        q_prime: u64,
        len: usize,
    ) {
        // Process one element at a time (no vectorization)
        for i in 0..len {
            let a_val = *a.add(i);
            let b_val = *b.add(i);
            *result.add(i) = montgomery_mul(a_val, b_val, q, q_prime);
        }
    }
}

// ============================================================================
// Helper Functions (Scalar Arithmetic)
// ============================================================================

/// Modular multiplication: (a * b) mod q
///
/// Uses 128-bit intermediate to prevent overflow.
#[inline(always)]
fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

/// Modular addition: (a + b) mod q
///
/// Optimized to avoid division when possible.
#[inline(always)]
fn add_mod(a: u64, b: u64, q: u64) -> u64 {
    let sum = a + b;
    if sum >= q {
        sum - q
    } else {
        sum
    }
}

/// Modular subtraction: (a - b) mod q
///
/// Handles underflow by adding q.
#[inline(always)]
fn sub_mod(a: u64, b: u64, q: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + q - b
    }
}

/// Barrett reduction for modular multiplication
///
/// Computes `(a * b) mod q` using Barrett's algorithm.
/// This avoids expensive division operations.
///
/// # Algorithm
/// 1. Compute product: `p = a * b` (128-bit)
/// 2. Approximate quotient: `q_hat = (p * k) >> 128` where `k = ⌊2^128 / q⌋`
/// 3. Compute remainder: `r = p - q_hat * q`
/// 4. Final correction: if `r >= q`, return `r - q`, else return `r`
///
/// # Arguments
/// * `a` - First operand (must be < q)
/// * `b` - Second operand (must be < q)
/// * `q` - Modulus
/// * `k` - Barrett constant (precomputed as `⌊2^128 / q⌋`)
///
/// # Returns
/// `(a * b) mod q`
#[inline(always)]
fn barrett_reduce(a: u64, b: u64, q: u64, k: u128) -> u64 {
    // Step 1: Compute full product (128-bit)
    let product = (a as u128) * (b as u128);

    // Step 2: Approximate quotient using Barrett constant
    // We need the upper 128 bits of (product * k) which is a 256-bit multiplication
    // Since k ≈ 2^128 / q, we can approximate: q_hat ≈ product / q
    // Using 256-bit math: q_hat = ((product * k) >> 128)

    // Multiply product (128-bit) × k (128-bit) and take upper 128 bits
    // This is equivalent to: ⌊(product * k) / 2^128⌋
    let q_hat = mulhi_u128(product, k);

    // Step 3: Compute remainder
    // r = product - q_hat * q
    let remainder = (product - (q_hat * (q as u128))) as u64;

    // Step 4: Final correction (at most one subtraction needed)
    if remainder >= q {
        remainder - q
    } else {
        remainder
    }
}

/// Compute the upper 128 bits of a 128×128→256 bit multiplication
///
/// Returns ⌊(a * b) / 2^128⌋
#[inline(always)]
fn mulhi_u128(a: u128, b: u128) -> u128 {
    // Split a and b into high and low 64-bit parts
    let a_lo = a as u64 as u128;
    let a_hi = (a >> 64) as u64 as u128;
    let b_lo = b as u64 as u128;
    let b_hi = (b >> 64) as u64 as u128;

    // Compute partial products
    // a * b = (a_hi * 2^64 + a_lo) * (b_hi * 2^64 + b_lo)
    //       = a_hi*b_hi*2^128 + (a_hi*b_lo + a_lo*b_hi)*2^64 + a_lo*b_lo

    let p_ll = a_lo * b_lo;
    let p_lh = a_lo * b_hi;
    let p_hl = a_hi * b_lo;
    let p_hh = a_hi * b_hi;

    // Combine for upper 128 bits
    // Upper 128 bits = p_hh + (p_lh >> 64) + (p_hl >> 64) + ((p_ll >> 64) + (p_lh & 0xFFFFFFFFFFFFFFFF) + (p_hl & 0xFFFFFFFFFFFFFFFF)) >> 64

    let mid_sum = (p_ll >> 64) + (p_lh & 0xFFFFFFFFFFFFFFFF) + (p_hl & 0xFFFFFFFFFFFFFFFF);
    let carry = mid_sum >> 64;

    p_hh + (p_lh >> 64) + (p_hl >> 64) + carry
}

/// Montgomery multiplication: (a * b) / R mod q
///
/// Computes (a * b) * R^(-1) mod q where R = 2^64.
/// **This is exact arithmetic** - no approximation like Barrett reduction.
///
/// # Arguments
/// * `a` - First operand (in Montgomery form)
/// * `b` - Second operand (in Montgomery form)
/// * `q` - Modulus (must be odd)
/// * `q_prime` - Precomputed -q^(-1) mod R
///
/// # Algorithm (CIOS - Coarsely Integrated Operand Scanning)
/// 1. Compute t = a * b (128-bit product)
/// 2. Compute m = (t mod R) * q' mod R
/// 3. Compute u = (t + m * q) / R
/// 4. If u ≥ q, return u - q, else return u
///
/// # Returns
/// Product in Montgomery form: (a * b) / R mod q
#[inline(always)]
fn montgomery_mul(a: u64, b: u64, q: u64, q_prime: u64) -> u64 {
    // Step 1: Compute full 128-bit product t = a * b
    let t = (a as u128) * (b as u128);

    // Step 2: Compute m = ((t mod R) * q') mod R
    // Since R = 2^64, "mod R" is just taking lower 64 bits
    let t_lo = t as u64;
    let m = t_lo.wrapping_mul(q_prime); // This is m mod R (wrapping = mod 2^64)

    // Step 3: Compute u = (t + m * q) / R
    // The division by R = 2^64 is just taking upper 64 bits
    let mq = (m as u128) * (q as u128);
    let u = ((t + mq) >> 64) as u64; // Right shift by 64 = divide by R

    // Step 4: Final conditional subtraction
    if u >= q {
        u - q
    } else {
        u
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::traits::compute_barrett_constant;

    #[test]
    fn test_scalar_backend_creation() {
        let backend = ScalarBackend::new();
        assert_eq!(backend.name(), "Scalar");
    }

    #[test]
    fn test_mul_mod() {
        let q = 17u64;
        assert_eq!(mul_mod(5, 7, q), 1); // 5*7 = 35 ≡ 1 (mod 17)
        assert_eq!(mul_mod(10, 10, q), 15); // 10*10 = 100 ≡ 15 (mod 17)
        assert_eq!(mul_mod(16, 16, q), 1); // 16*16 = 256 ≡ 1 (mod 17)
    }

    #[test]
    fn test_add_mod() {
        let q = 17u64;
        assert_eq!(add_mod(5, 7, q), 12);
        assert_eq!(add_mod(10, 10, q), 3); // 20 ≡ 3 (mod 17)
        assert_eq!(add_mod(16, 2, q), 1); // 18 ≡ 1 (mod 17)
    }

    #[test]
    fn test_sub_mod() {
        let q = 17u64;
        assert_eq!(sub_mod(10, 5, q), 5);
        assert_eq!(sub_mod(5, 10, q), 12); // -5 ≡ 12 (mod 17)
        assert_eq!(sub_mod(1, 2, q), 16); // -1 ≡ 16 (mod 17)
    }

    #[test]
    fn test_mulhi_u128() {
        // Test mulhi_u128 computes upper 128 bits correctly

        // Test 1: Simple case where result fits in 128 bits
        let a = 1u128 << 64;  // 2^64
        let b = 1u128 << 64;  // 2^64
        // a * b = 2^128, so upper 128 bits = 1
        assert_eq!(mulhi_u128(a, b), 1, "2^64 * 2^64 should give upper bits = 1");

        // Test 2: Maximum values
        let max = u128::MAX;
        // max * max has upper bits = max - 1
        assert_eq!(mulhi_u128(max, max), max - 1, "MAX * MAX upper bits");

        // Test 3: Verify against known computation
        // (2^64 + 1) * (2^64 + 1) = 2^128 + 2^65 + 1
        // Upper 128 bits = 1 + 0 = 1 (the 2^65 is in lower bits)
        let a = (1u128 << 64) + 1;
        let b = (1u128 << 64) + 1;
        assert_eq!(mulhi_u128(a, b), 1, "(2^64+1)^2 upper bits");

        // Test 4: Asymmetric case
        let a = 12345678901234567890u128;
        let b = 98765432109876543210u128;
        // Verify by checking that mulhi gives same result as manual 256-bit math
        // For now, just verify it doesn't panic
        let result = mulhi_u128(a, b);
        assert!(result > 0, "Large multiplication should have non-zero upper bits");
    }

    #[test]
    fn test_barrett_reduce() {
        let q = 1099511678977u64; // 41-bit prime from V2 params
        let k = compute_barrett_constant(q);

        // Test small values
        assert_eq!(barrett_reduce(2, 3, q, k), 6);
        assert_eq!(barrett_reduce(100, 100, q, k), 10000);

        // Test values close to q
        let large = q - 1;
        assert_eq!(barrett_reduce(large, 2, q, k), (2 * large) % q);

        // Test that it matches naive modular multiplication
        for a in [1u64, 10, 100, 1000, q - 1, q / 2] {
            for b in [1u64, 10, 100, 1000, q - 1, q / 2] {
                let expected = mul_mod(a, b, q);
                let actual = barrett_reduce(a, b, q, k);
                assert_eq!(
                    actual, expected,
                    "Barrett mismatch for {}*{} mod {}",
                    a, b, q
                );
            }
        }
    }

    #[test]
    fn test_barrett_reduce_large_prime() {
        let q = 1141392289560813569u64; // 60-bit prime from V2 params
        let k = compute_barrett_constant(q);

        // Test various values
        let test_cases = vec![
            (1, 1, 1),
            (2, 3, 6),
            (q - 1, 2, (2 * (q - 1)) % q),
            (q - 1, q - 1, mul_mod(q - 1, q - 1, q)),
        ];

        for (a, b, expected) in test_cases {
            let actual = barrett_reduce(a, b, q, k);
            assert_eq!(
                actual, expected,
                "Barrett failed for {}*{} mod {}",
                a, b, q
            );
        }
    }

    #[test]
    fn test_ntt_butterfly_scalar() {
        let backend = ScalarBackend::new();
        let q = 17u64;
        let twiddle = 3u64;

        let mut a = vec![5u64, 10u64, 3u64, 7u64];
        let mut b = vec![2u64, 4u64, 1u64, 6u64];

        unsafe {
            backend.ntt_butterfly(a.as_mut_ptr(), b.as_mut_ptr(), twiddle, q, 4);
        }

        // Verify results:
        // For each i:
        //   t = twiddle * b_old[i] mod q
        //   a_new[i] = (a_old[i] + t) mod q
        //   b_new[i] = (a_old[i] - t) mod q

        // i=0: t = 3*2 = 6, a = 5+6 = 11, b = 5-6 = -1 ≡ 16 (mod 17)
        assert_eq!(a[0], 11);
        assert_eq!(b[0], 16);

        // i=1: t = 3*4 = 12, a = 10+12 = 22 ≡ 5 (mod 17), b = 10-12 = -2 ≡ 15 (mod 17)
        assert_eq!(a[1], 5);
        assert_eq!(b[1], 15);
    }

    #[test]
    fn test_simd_operations_match_scalar() {
        let backend = ScalarBackend::new();
        let q = 1099511678977u64;
        let k = compute_barrett_constant(q);

        let a = vec![100u64, 200, 300, 400];
        let b = vec![50u64, 75, 125, 175];
        let scalar = 7u64;

        let mut result_mul = vec![0u64; 4];
        let mut result_add = vec![0u64; 4];
        let mut result_sub = vec![0u64; 4];
        let mut result_scalar_mul = vec![0u64; 4];

        unsafe {
            backend.barrett_mul_mod(
                result_mul.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                q,
                k,
                4,
            );

            backend.add_mod(result_add.as_mut_ptr(), a.as_ptr(), b.as_ptr(), q, 4);

            backend.sub_mod(result_sub.as_mut_ptr(), a.as_ptr(), b.as_ptr(), q, 4);

            backend.scalar_mul_mod(
                result_scalar_mul.as_mut_ptr(),
                a.as_ptr(),
                scalar,
                q,
                k,
                4,
            );
        }

        // Verify against naive implementations
        for i in 0..4 {
            assert_eq!(result_mul[i], mul_mod(a[i], b[i], q));
            assert_eq!(result_add[i], add_mod(a[i], b[i], q));
            assert_eq!(result_sub[i], sub_mod(a[i], b[i], q));
            assert_eq!(result_scalar_mul[i], mul_mod(a[i], scalar, q));
        }
    }
}
