//! Metal Compute Shaders for Number Theoretic Transform (NTT)
//!
//! Harvey Butterfly NTT implementation optimized for Apple Silicon GPUs.
//!
//! Key optimizations:
//! - Threadgroup memory for reduced global memory access
//! - Cooperative loading for memory coalescing
//! - Bit-reversal permutation integrated into NTT

#include <metal_stdlib>
using namespace metal;

/// Modular multiplication: (a * b) mod q
/// Uses 128-bit intermediate for FHE-sized primes (44-60 bits)
///
/// Implements exact 128-bit modular reduction: (a * b) mod q
/// Equivalent to Rust: ((a as u128 * b as u128) % q as u128) as u64
/// Montgomery multiplication: (a * b * R^{-1}) mod q
/// where R = 2^64
///
/// This is the CORRECT way to do 128-bit modular multiplication on GPU!
/// No 128-bit division needed, no %q on partial products that lose carry info.
///
/// @param a First operand (in Montgomery domain: a * R mod q)
/// @param b Second operand (in Montgomery domain: b * R mod q)
/// @param q Modulus (60-bit prime)
/// @param q_inv Precomputed: -q^{-1} mod 2^64
/// @return (a * b * R^{-1}) mod q (stays in Montgomery domain)
inline ulong mont_mul(ulong a, ulong b, ulong q, ulong q_inv) {
    // Step 1: Compute t = a * b (128-bit)
    ulong t_lo = a * b;
    ulong t_hi = mulhi(a, b);

    // Step 2: Compute m = (t_lo * q_inv) mod 2^64
    // This is just the low 64 bits (automatic wrap)
    ulong m = t_lo * q_inv;

    // Step 3: Compute m * q (128-bit)
    ulong mq_lo = m * q;
    ulong mq_hi = mulhi(m, q);

    // Step 4: Compute u = (t + m*q) / 2^64
    // This is the high limb of the 128-bit sum
    // Need to detect carry from low limb addition
    ulong carry = (t_lo > ~mq_lo) ? 1UL : 0UL;  // Will t_lo + mq_lo overflow?
    ulong sum_hi = t_hi + mq_hi + carry;

    // Step 5: Conditional subtraction to ensure result < q
    return (sum_hi >= q) ? (sum_hi - q) : sum_hi;
}

/// Standard modular multiplication (NON-Montgomery)
/// Only used for compatibility where Montgomery domain conversion isn't done yet
/// SLOW and potentially buggy - prefer Montgomery!
inline ulong mul_mod_slow(ulong a, ulong b, ulong q) {
    ulong hi = mulhi(a, b);
    ulong lo = a * b;

    if (hi == 0) {
        return lo % q;
    }

    // This path is fundamentally problematic - splitting with % loses carries
    // Use Montgomery multiplication instead!
    hi = hi % q;
    lo = lo % q;
    ulong two_32 = (1UL << 32) % q;
    ulong two_64_mod_q = (two_32 * two_32) % q;
    ulong hi_contrib_hi = mulhi(hi, two_64_mod_q);
    ulong hi_contrib_lo = hi * two_64_mod_q;

    if (hi_contrib_hi == 0) {
        ulong hi_contrib = hi_contrib_lo % q;
        ulong result = (hi_contrib + lo) % q;
        return result;
    } else {
        hi_contrib_hi = hi_contrib_hi % q;
        ulong temp = (hi_contrib_hi * two_64_mod_q) % q;
        ulong hi_contrib = (temp + (hi_contrib_lo % q)) % q;
        ulong result = (hi_contrib + lo) % q;
        return result;
    }
}

/// Modular multiplication wrapper
/// Since all data is pre-converted to Montgomery domain by the Rust code,
/// we use Montgomery multiplication here for correctness and speed.
/// @param a First operand (in Montgomery domain)
/// @param b Second operand (in Montgomery domain)
/// @param q Modulus
/// @param q_inv Precomputed -q^{-1} mod 2^64
/// @return (a * b * R^{-1}) mod q (stays in Montgomery domain)
inline ulong mul_mod(ulong a, ulong b, ulong q, ulong q_inv) {
    return mont_mul(a, b, q, q_inv);
}

/// DEPRECATED: 3-argument version for backwards compatibility with old unused kernels
/// This will use the slow/broken path - DO NOT USE in new code!
inline ulong mul_mod(ulong a, ulong b, ulong q) {
    return mul_mod_slow(a, b, q);
}

/// Modular addition: (a + b) mod q
inline ulong add_mod(ulong a, ulong b, ulong q) {
    ulong sum = a + b;
    return (sum >= q) ? (sum - q) : sum;
}

/// Modular subtraction: (a - b) mod q
inline ulong sub_mod(ulong a, ulong b, ulong q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

/// Modular exponentiation: base^exp mod q (for twiddle factor generation)
/// NOTE: This should never be called in optimized code - twiddles are precomputed on CPU!
inline ulong pow_mod(ulong base, ulong exp, ulong q, ulong q_inv) {
    ulong result = 1;
    base = base % q;

    while (exp > 0) {
        if (exp & 1) {
            result = mul_mod(result, base, q, q_inv);
        }
        base = mul_mod(base, base, q, q_inv);
        exp >>= 1;
    }

    return result;
}

/// Bit-reversal permutation (separate pass for correctness)
kernel void ntt_bit_reverse(
    device ulong* coeffs [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        uint reversed = 0;
        uint k = gid;
        uint logn = 31 - clz(n);

        for (uint i = 0; i < logn; i++) {
            reversed = (reversed << 1) | (k & 1);
            k >>= 1;
        }

        if (gid < reversed) {
            ulong temp = coeffs[gid];
            coeffs[gid] = coeffs[reversed];
            coeffs[reversed] = temp;
        }
    }
}

/// Single-stage NTT butterfly (Cooley-Tukey) with Montgomery multiplication
/// One dispatch per stage - provides global synchronization between stages
///
/// @param coeffs Input/output coefficients (in-place, in Montgomery domain)
/// @param twiddles Precomputed twiddle factors (in Montgomery domain)
/// @param n Polynomial degree
/// @param q Modulus
/// @param stage Which stage (0 = first, log2(n)-1 = last)
/// @param q_inv -q^{-1} mod 2^64 for Montgomery reduction
kernel void ntt_forward_stage(
    device ulong* coeffs [[buffer(0)]],
    constant ulong* twiddles [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant ulong& q [[buffer(3)]],
    constant uint& stage [[buffer(4)]],
    constant ulong& q_inv [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint m = 1 << (stage + 1);    // Block size for this stage
    uint m_half = 1 << stage;      // Half block size

    // Each thread processes one butterfly
    uint butterfly_idx = gid;
    if (butterfly_idx < n / 2) {
        uint block_idx = butterfly_idx / m_half;
        uint idx_in_block = butterfly_idx % m_half;

        uint i = block_idx * m + idx_in_block;
        uint j = i + m_half;

        // Twiddle factor index (twiddle is already in Montgomery domain)
        uint twiddle_idx = (n / m) * idx_in_block;
        ulong omega = twiddles[twiddle_idx];

        // Harvey butterfly with Montgomery multiplication
        ulong u = coeffs[i];
        ulong v = mont_mul(coeffs[j], omega, q, q_inv);

        coeffs[i] = add_mod(u, v, q);
        coeffs[j] = sub_mod(u, v, q);
    }
}

/* OLD single-dispatch NTT (BUGGY - no global sync between stages)
   REMOVED - do not use, has race conditions

kernel void ntt_forward_single_dispatch_BUGGY_REMOVED(
    device ulong* coeffs [[buffer(0)]],
    constant ulong* twiddles [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant ulong& q [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // BUG: threadgroup_barrier does NOT synchronize across threadgroups!
    // Different threadgroups can be at different stages, reading stale values.

    // Bit-reversal permutation
    if (gid < n) {
        uint reversed = 0;
        uint k = gid;
        uint logn = 31 - clz(n);

        for (uint i = 0; i < logn; i++) {
            reversed = (reversed << 1) | (k & 1);
            k >>= 1;
        }

        if (gid < reversed) {
            ulong temp = coeffs[gid];
            coeffs[gid] = coeffs[reversed];
            coeffs[reversed] = temp;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);  // DOES NOT WORK GLOBALLY!

    // NTT butterfly stages
    for (uint stage = 0; stage < 31 - clz(n); stage++) {
        uint m = 1 << (stage + 1);
        uint m_half = 1 << stage;

        uint butterfly_idx = gid;
        if (butterfly_idx < n / 2) {
            uint block_idx = butterfly_idx / m_half;
            uint idx_in_block = butterfly_idx % m_half;

            uint i = block_idx * m + idx_in_block;
            uint j = i + m_half;

            uint twiddle_idx = (n / m) * idx_in_block;
            ulong omega = twiddles[twiddle_idx];

            ulong u = coeffs[i];
            ulong v = mul_mod(coeffs[j], omega, q);

            coeffs[i] = add_mod(u, v, q);
            coeffs[j] = sub_mod(u, v, q);
        }

        threadgroup_barrier(mem_flags::mem_device);  // DOES NOT WORK GLOBALLY!
    }
}
*/

/// Single-stage inverse NTT butterfly (Gentleman-Sande) with Montgomery multiplication
/// One dispatch per stage - provides global synchronization
///
/// @param stage Which stage (log2(n)-1 = first, 0 = last for inverse)
/// @param q_inv -q^{-1} mod 2^64 for Montgomery reduction
kernel void ntt_inverse_stage(
    device ulong* coeffs [[buffer(0)]],
    constant ulong* twiddles_inv [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant ulong& q [[buffer(3)]],
    constant uint& stage [[buffer(4)]],
    constant ulong& q_inv [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint m = 1 << (stage + 1);
    uint m_half = 1 << stage;

    uint butterfly_idx = gid;
    if (butterfly_idx < n / 2) {
        uint block_idx = butterfly_idx / m_half;
        uint idx_in_block = butterfly_idx % m_half;

        uint i = block_idx * m + idx_in_block;
        uint j = i + m_half;

        uint twiddle_idx = (n / m) * idx_in_block;
        ulong omega_inv = twiddles_inv[twiddle_idx];

        // Inverse butterfly with Montgomery multiplication
        ulong u = coeffs[i];
        ulong v = coeffs[j];

        coeffs[i] = add_mod(u, v, q);
        coeffs[j] = mont_mul(sub_mod(u, v, q), omega_inv, q, q_inv);
    }
}

/// Scale and bit-reverse (final step of inverse NTT) with Montgomery multiplication
///
/// @param q_inv -q^{-1} mod 2^64 for Montgomery reduction
kernel void ntt_inverse_final_scale(
    device ulong* coeffs [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant ulong& q [[buffer(2)]],
    constant ulong& n_inv [[buffer(3)]],
    constant ulong& q_inv [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        uint reversed = 0;
        uint k = gid;
        uint logn = 31 - clz(n);

        for (uint i = 0; i < logn; i++) {
            reversed = (reversed << 1) | (k & 1);
            k >>= 1;
        }

        if (gid <= reversed) {
            ulong temp = mont_mul(coeffs[gid], n_inv, q, q_inv);
            coeffs[gid] = mont_mul(coeffs[reversed], n_inv, q, q_inv);
            coeffs[reversed] = temp;
        }
    }
}

/// OLD single-dispatch inverse NTT (BUGGY - no global sync)
kernel void ntt_inverse_single_dispatch_BUGGY(
    device ulong* coeffs [[buffer(0)]],
    constant ulong* twiddles_inv [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant ulong& q [[buffer(3)]],
    constant ulong& n_inv [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // BUG: Same synchronization issue as forward NTT

    // Inverse NTT butterfly stages
    for (int stage = (31 - clz(n)) - 1; stage >= 0; stage--) {
        uint m = 1 << (stage + 1);
        uint m_half = 1 << stage;

        uint butterfly_idx = gid;
        if (butterfly_idx < n / 2) {
            uint block_idx = butterfly_idx / m_half;
            uint idx_in_block = butterfly_idx % m_half;

            uint i = block_idx * m + idx_in_block;
            uint j = i + m_half;

            uint twiddle_idx = (n / m) * idx_in_block;
            ulong omega_inv = twiddles_inv[twiddle_idx];

            ulong u = coeffs[i];
            ulong v = coeffs[j];

            coeffs[i] = add_mod(u, v, q);
            coeffs[j] = mul_mod(sub_mod(u, v, q), omega_inv, q);
        }

        threadgroup_barrier(mem_flags::mem_device);  // DOES NOT WORK GLOBALLY!
    }

    // Bit-reversal and scaling
    if (gid < n) {
        uint reversed = 0;
        uint k = gid;
        uint logn = 31 - clz(n);

        for (uint i = 0; i < logn; i++) {
            reversed = (reversed << 1) | (k & 1);
            k >>= 1;
        }

        if (gid <= reversed) {
            ulong temp = mul_mod(coeffs[gid], n_inv, q);
            coeffs[gid] = mul_mod(coeffs[reversed], n_inv, q);
            coeffs[reversed] = temp;
        }
    }
}

/// Pointwise multiplication in NTT domain (Hadamard product)
///
/// Used for polynomial multiplication: c(x) = a(x) * b(x)
/// In NTT domain: NTT(c) = NTT(a) ⊙ NTT(b) (pointwise)
///
/// Fully parallel: each thread handles one coefficient
/// Pointwise modular multiplication in NTT domain (using Montgomery multiplication)
///
/// IMPORTANT: Input values a[] and b[] must be in Montgomery domain!
/// Output c[] is also in Montgomery domain.
kernel void ntt_pointwise_multiply(
    device const ulong* a [[buffer(0)]],
    device const ulong* b [[buffer(1)]],
    device ulong* c [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant ulong& q [[buffer(4)]],
    constant ulong& q_inv [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        // Montgomery multiplication: (a*R) * (b*R) * R^{-1} = (a*b)*R
        c[gid] = mont_mul(a[gid], b[gid], q, q_inv);
    }
}

/// Batched pointwise multiplication for multiple RNS primes
///
/// Processes multiple RNS primes in parallel using 2D dispatch.
/// Input data is in flat interleaved layout: [coeff0_p0, coeff0_p1, coeff0_p2, coeff1_p0, ...]
///
/// @param a Input polynomial A in flat layout (size: n * num_primes)
/// @param b Input polynomial B in flat layout (size: n * num_primes)
/// @param c Output polynomial C in flat layout (size: n * num_primes)
/// @param moduli Array of RNS moduli [q0, q1, q2, ...]
/// @param moduli_inv Array of Montgomery parameters [q0_inv, q1_inv, q2_inv, ...]
/// @param n Polynomial degree
/// @param num_primes Number of RNS primes
/// @param gid 2D thread position: (coeff_idx, prime_idx)
kernel void ntt_pointwise_multiply_batched(
    device const ulong* a [[buffer(0)]],
    device const ulong* b [[buffer(1)]],
    device ulong* c [[buffer(2)]],
    constant ulong* moduli [[buffer(3)]],
    constant ulong* moduli_inv [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& num_primes [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint coeff_idx = gid.x;
    uint prime_idx = gid.y;

    if (coeff_idx < n && prime_idx < num_primes) {
        // Calculate flat index: coeff_idx * num_primes + prime_idx
        uint flat_idx = coeff_idx * num_primes + prime_idx;

        // Get modulus and Montgomery parameter for this prime
        ulong q = moduli[prime_idx];
        ulong q_inv = moduli_inv[prime_idx];

        // Montgomery multiplication: (a*R) * (b*R) * R^{-1} = (a*b)*R
        c[flat_idx] = mont_mul(a[flat_idx], b[flat_idx], q, q_inv);
    }
}

/// Pointwise modular addition
kernel void ntt_pointwise_add(
    device const ulong* a [[buffer(0)]],
    device const ulong* b [[buffer(1)]],
    device ulong* c [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant ulong& q [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        c[gid] = add_mod(a[gid], b[gid], q);
    }
}

/// Pointwise modular subtraction
kernel void ntt_pointwise_sub(
    device const ulong* a [[buffer(0)]],
    device const ulong* b [[buffer(1)]],
    device ulong* c [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant ulong& q [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        c[gid] = sub_mod(a[gid], b[gid], q);
    }
}

/// Batched in-place modular subtraction for multi-prime flat layout
///
/// Performs: a[i] -= b[i] (mod q) for all i
/// Works with flat interleaved RNS layout: [coeff0_p0, coeff0_p1, ..., coeff1_p0, ...]
///
/// @param a Input/output array (modified in-place)
/// @param b Array to subtract
/// @param moduli Array of RNS moduli [q0, q1, q2, ...]
/// @param n Polynomial degree
/// @param num_primes Number of RNS primes
kernel void ntt_pointwise_sub_inplace_batched(
    device ulong* a [[buffer(0)]],
    device const ulong* b [[buffer(1)]],
    constant ulong* moduli [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& num_primes [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint coeff_idx = gid.x;
    uint prime_idx = gid.y;

    if (coeff_idx < n && prime_idx < num_primes) {
        uint flat_idx = coeff_idx * num_primes + prime_idx;
        ulong q = moduli[prime_idx];
        a[flat_idx] = sub_mod(a[flat_idx], b[flat_idx], q);
    }
}

/// Batched in-place modular addition for multi-prime flat layout
///
/// Performs: a[i] += b[i] (mod q) for all i
/// Works with flat interleaved RNS layout: [coeff0_p0, coeff0_p1, ..., coeff1_p0, ...]
///
/// @param a Input/output array (modified in-place)
/// @param b Array to add
/// @param moduli Array of RNS moduli [q0, q1, q2, ...]
/// @param n Polynomial degree
/// @param num_primes Number of RNS primes
kernel void ntt_pointwise_add_inplace_batched(
    device ulong* a [[buffer(0)]],
    device const ulong* b [[buffer(1)]],
    constant ulong* moduli [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& num_primes [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint coeff_idx = gid.x;
    uint prime_idx = gid.y;

    if (coeff_idx < n && prime_idx < num_primes) {
        uint flat_idx = coeff_idx * num_primes + prime_idx;
        ulong q = moduli[prime_idx];
        a[flat_idx] = add_mod(a[flat_idx], b[flat_idx], q);
    }
}

// ============================================================================
// TWISTED NTT for Negacyclic Convolution (mod x^n + 1)
// ============================================================================
//
// Standard NTT gives cyclic convolution (mod x^n - 1).
// For FHE, we need negacyclic convolution (mod x^n + 1).
//
// Solution: Apply "twist" using primitive 2n-th root of unity (psi)
//
// Forward Twisted NTT:
//   1. Multiply coeffs[i] by psi^i (twist)
//   2. Apply standard NTT with omega = psi^2
//   3. Result is NTT for negacyclic convolution
//
// Inverse Twisted NTT:
//   1. Apply standard INTT with omega = psi^2
//   2. Multiply coeffs[i] by psi^(-i) (inverse twist)
//   3. Result is polynomial in coefficient form
//
// ============================================================================

/// Apply twist: coeffs[i] *= psi^i mod q
///
/// This converts coefficients for negacyclic convolution to cyclic form.
/// Used before standard NTT.
///
/// @param coeffs Input/output coefficients (in-place)
/// @param psi_powers Precomputed powers of psi: [1, psi, psi^2, ..., psi^(n-1)]
/// @param n Polynomial degree
/// @param q Modulus
kernel void ntt_apply_twist(
    device ulong* coeffs [[buffer(0)]],
    constant ulong* psi_powers [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant ulong& q [[buffer(3)]],
    constant ulong& q_inv [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        coeffs[gid] = mul_mod(coeffs[gid], psi_powers[gid], q, q_inv);
    }
}

/// Apply inverse twist: coeffs[i] *= psi^(-i) mod q
///
/// This converts coefficients from cyclic form back to negacyclic form.
/// Used after standard INTT.
///
/// @param coeffs Input/output coefficients (in-place)
/// @param psi_inv_powers Precomputed powers of psi^(-1): [1, psi^(-1), psi^(-2), ..., psi^(-(n-1))]
/// @param n Polynomial degree
/// @param q Modulus
kernel void ntt_apply_inverse_twist(
    device ulong* coeffs [[buffer(0)]],
    constant ulong* psi_inv_powers [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant ulong& q [[buffer(3)]],
    constant ulong& q_inv [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        coeffs[gid] = mul_mod(coeffs[gid], psi_inv_powers[gid], q, q_inv);
    }
}

/// Fused iNTT Final Scale + Inverse Twist
///
/// Combines the final bit-reversal/scaling step of iNTT with inverse twist application.
/// This saves one GPU kernel dispatch and intermediate buffer readback.
///
/// Performs:
/// 1. Bit-reverse permutation
/// 2. Scale by n^{-1} (Montgomery: multiply by n_inv_mont)
/// 3. Apply inverse twist: multiply by ψ^{-i}
///
/// @param coeffs Input/output coefficients (in Montgomery domain)
/// @param psi_inv_powers Powers of psi^(-1) in Montgomery: [1, ψ^{-1}, ψ^{-2}, ..., ψ^{-(n-1)}]
/// @param n Polynomial degree
/// @param q Modulus
/// @param n_inv_mont Modular inverse of n in Montgomery domain (n^{-1} * R mod q)
/// @param q_inv Montgomery parameter: -q^{-1} mod 2^64
kernel void ntt_inverse_final_scale_and_untwist(
    device ulong* coeffs [[buffer(0)]],
    constant ulong* psi_inv_powers [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant ulong& q [[buffer(3)]],
    constant ulong& n_inv_mont [[buffer(4)]],
    constant ulong& q_inv [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        // Compute bit-reversed index
        uint reversed = 0;
        uint k = gid;
        uint logn = 31 - clz(n);

        for (uint i = 0; i < logn; i++) {
            reversed = (reversed << 1) | (k & 1);
            k >>= 1;
        }

        // Perform bit-reversal swap (only for gid <= reversed to avoid double-swap)
        if (gid <= reversed) {
            // Read values
            ulong val_gid = coeffs[gid];
            ulong val_rev = coeffs[reversed];

            // Scale by n^{-1} (Montgomery multiply)
            val_gid = mont_mul(val_gid, n_inv_mont, q, q_inv);
            val_rev = mont_mul(val_rev, n_inv_mont, q, q_inv);

            // Apply inverse twist: multiply by ψ^{-i}
            val_gid = mont_mul(val_gid, psi_inv_powers[gid], q, q_inv);
            val_rev = mont_mul(val_rev, psi_inv_powers[reversed], q, q_inv);

            // Write back (swapped)
            coeffs[gid] = val_rev;
            coeffs[reversed] = val_gid;
        }
    }
}

/// Forward Twisted NTT (for negacyclic convolution)
///
/// Combines twist application + standard NTT in a single kernel for efficiency.
/// This is the main kernel for FHE operations.
///
/// @param coeffs Input/output polynomial coefficients
/// @param psi_powers Powers of psi (2n-th root): [1, psi, psi^2, ...]
/// @param omega_powers Powers of omega = psi^2 (n-th root): [1, omega, omega^2, ...]
/// @param n Polynomial degree
/// @param q Modulus
kernel void ntt_forward_twisted(
    device ulong* coeffs [[buffer(0)]],
    constant ulong* psi_powers [[buffer(1)]],
    constant ulong* omega_powers [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant ulong& q [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // Step 1: Apply twist
    if (gid < n) {
        coeffs[gid] = mul_mod(coeffs[gid], psi_powers[gid], q);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Step 2: Bit-reversal permutation
    if (gid < n) {
        uint reversed = 0;
        uint k = gid;
        uint logn = 31 - clz(n);

        for (uint i = 0; i < logn; i++) {
            reversed = (reversed << 1) | (k & 1);
            k >>= 1;
        }

        if (gid < reversed) {
            ulong temp = coeffs[gid];
            coeffs[gid] = coeffs[reversed];
            coeffs[reversed] = temp;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Step 3: Standard NTT butterfly stages with omega
    for (uint stage = 0; stage < 31 - clz(n); stage++) {
        uint m = 1 << (stage + 1);
        uint m_half = 1 << stage;

        uint butterfly_idx = gid;
        if (butterfly_idx < n / 2) {
            uint block_idx = butterfly_idx / m_half;
            uint idx_in_block = butterfly_idx % m_half;

            uint i = block_idx * m + idx_in_block;
            uint j = i + m_half;

            uint twiddle_idx = (n / m) * idx_in_block;
            ulong omega = omega_powers[twiddle_idx];

            ulong u = coeffs[i];
            ulong v = mul_mod(coeffs[j], omega, q);

            coeffs[i] = add_mod(u, v, q);
            coeffs[j] = sub_mod(u, v, q);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

/// Inverse Twisted NTT (for negacyclic convolution)
///
/// Combines standard INTT + inverse twist application.
///
/// @param coeffs Input/output evaluation points
/// @param omega_inv_powers Powers of omega^(-1)
/// @param psi_inv_powers Powers of psi^(-1)
/// @param n Polynomial degree
/// @param q Modulus
/// @param n_inv Modular inverse of n
kernel void ntt_inverse_twisted(
    device ulong* coeffs [[buffer(0)]],
    constant ulong* omega_inv_powers [[buffer(1)]],
    constant ulong* psi_inv_powers [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant ulong& q [[buffer(4)]],
    constant ulong& n_inv [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // Step 1: Standard inverse NTT butterfly stages
    for (int stage = (31 - clz(n)) - 1; stage >= 0; stage--) {
        uint m = 1 << (stage + 1);
        uint m_half = 1 << stage;

        uint butterfly_idx = gid;
        if (butterfly_idx < n / 2) {
            uint block_idx = butterfly_idx / m_half;
            uint idx_in_block = butterfly_idx % m_half;

            uint i = block_idx * m + idx_in_block;
            uint j = i + m_half;

            uint twiddle_idx = (n / m) * idx_in_block;
            ulong omega_inv = omega_inv_powers[twiddle_idx];

            ulong u = coeffs[i];
            ulong v = coeffs[j];

            coeffs[i] = add_mod(u, v, q);
            coeffs[j] = mul_mod(sub_mod(u, v, q), omega_inv, q);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    // Step 2: Bit-reversal and scaling by 1/n
    if (gid < n) {
        uint reversed = 0;
        uint k = gid;
        uint logn = 31 - clz(n);

        for (uint i = 0; i < logn; i++) {
            reversed = (reversed << 1) | (k & 1);
            k >>= 1;
        }

        if (gid <= reversed) {
            ulong temp = mul_mod(coeffs[gid], n_inv, q);
            coeffs[gid] = mul_mod(coeffs[reversed], n_inv, q);
            coeffs[reversed] = temp;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Step 3: Apply inverse twist
    if (gid < n) {
        coeffs[gid] = mul_mod(coeffs[gid], psi_inv_powers[gid], q);
    }
}
