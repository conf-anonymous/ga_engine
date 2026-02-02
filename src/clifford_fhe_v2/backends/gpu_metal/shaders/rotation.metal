//! Metal Compute Shaders for Galois Automorphisms (Homomorphic Rotation)
//!
//! Implements permutation X → X^k for CKKS slot rotations in the ring R = Z[X]/(X^N + 1).
//!
//! **Mathematical Background:**
//! - Galois automorphism σ_k: X → X^k permutes CKKS slots
//! - For X^(i·k mod 2N): if i·k < N, coefficient stays positive
//!                         if i·k ≥ N, coefficient gets negated (since X^N = -1)
//! - Precomputed galois_map and galois_signs handle this permutation efficiently
//!
//! **Key Optimization:**
//! - Fully parallel: each thread handles one coefficient
//! - No synchronization needed (pure permutation)
//! - Coalesced memory access via flat RNS layout

#include <metal_stdlib>
using namespace metal;

/// Apply Galois automorphism: polynomial[i] → polynomial[galois_map[i]]
///
/// For σ_k: X → X^k in ring R = Z[X]/(X^N + 1):
/// - Input:  f(X) = Σ fᵢ X^i
/// - Output: f(X^k) = Σ fᵢ X^(i·k mod 2N) with sign correction
///
/// The galois_map precomputes the permutation and sign flips.
///
/// # Layout
///
/// Input/output use flat RNS layout (coeff-major):
/// ```
/// [coeff0_mod_q0, coeff0_mod_q1, ..., coeff0_mod_qL,
///  coeff1_mod_q0, coeff1_mod_q1, ..., coeff1_mod_qL,
///  ...]
/// ```
///
/// # Example
///
/// For N=4, k=3 (rotation by 1 step for N=4):
/// - X^0 → X^0 (stays)
/// - X^1 → X^3
/// - X^2 → X^6 = -X^2 (since X^4 = -1)
/// - X^3 → X^9 = -X^1 (since X^8 = 1, X^9 = -X^1)
///
/// galois_map = [0, 3, 2, 1]
/// galois_signs = [1, 1, -1, -1]
///
/// @param input Input polynomial coefficients (N × num_primes elements)
/// @param output Output polynomial after automorphism (N × num_primes elements)
/// @param galois_map Precomputed permutation map [N] (where each coefficient goes)
/// @param galois_signs Sign correction (+1 or -1) [N] (negate if -1)
/// @param n Polynomial degree N
/// @param num_primes Number of RNS components
/// @param moduli Array of RNS moduli [num_primes]
kernel void apply_galois_automorphism(
    device const ulong* input [[buffer(0)]],
    device ulong* output [[buffer(1)]],
    constant uint* galois_map [[buffer(2)]],
    constant int* galois_signs [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& num_primes [[buffer(5)]],
    constant ulong* moduli [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread handles one coefficient across all RNS components
    if (gid < n) {
        uint target_idx = galois_map[gid];  // Where this coefficient goes
        int sign = galois_signs[gid];        // Sign flip (+1 or -1)

        // Process all RNS components for this coefficient
        for (uint prime_idx = 0; prime_idx < num_primes; prime_idx++) {
            ulong q = moduli[prime_idx];

            // Flat layout index: coeff_idx * num_primes + prime_idx
            uint in_idx = gid * num_primes + prime_idx;
            uint out_idx = target_idx * num_primes + prime_idx;

            ulong val = input[in_idx];

            // Apply sign correction
            if (sign < 0 && val != 0) {
                val = q - val;  // Negate: -x ≡ q - x (mod q)
            }

            output[out_idx] = val;
        }
    }
}

/// Optimized version for when all signs are positive (common case for small rotations)
///
/// Skips sign checking for better performance when k is small.
kernel void apply_galois_automorphism_positive_only(
    device const ulong* input [[buffer(0)]],
    device ulong* output [[buffer(1)]],
    constant uint* galois_map [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& num_primes [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        uint target_idx = galois_map[gid];

        // Process all RNS components for this coefficient
        for (uint prime_idx = 0; prime_idx < num_primes; prime_idx++) {
            uint in_idx = gid * num_primes + prime_idx;
            uint out_idx = target_idx * num_primes + prime_idx;
            output[out_idx] = input[in_idx];
        }
    }
}

/// Apply Galois automorphism with vectorized memory access (experimental)
///
/// Attempts to use vector loads for better memory bandwidth.
/// May be faster for large N on M3 Max (128-bit vector units).
kernel void apply_galois_automorphism_vectorized(
    device const ulong* input [[buffer(0)]],
    device ulong* output [[buffer(1)]],
    constant uint* galois_map [[buffer(2)]],
    constant int* galois_signs [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& num_primes [[buffer(5)]],
    constant ulong* moduli [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread handles one coefficient
    if (gid < n) {
        uint target_idx = galois_map[gid];
        int sign = galois_signs[gid];

        // Try to process 2 RNS components at a time (if even number of primes)
        uint pairs = num_primes / 2;
        uint remainder = num_primes % 2;

        for (uint pair_idx = 0; pair_idx < pairs; pair_idx++) {
            uint prime_idx0 = pair_idx * 2;
            uint prime_idx1 = pair_idx * 2 + 1;

            ulong q0 = moduli[prime_idx0];
            ulong q1 = moduli[prime_idx1];

            uint in_idx0 = gid * num_primes + prime_idx0;
            uint in_idx1 = gid * num_primes + prime_idx1;
            uint out_idx0 = target_idx * num_primes + prime_idx0;
            uint out_idx1 = target_idx * num_primes + prime_idx1;

            ulong val0 = input[in_idx0];
            ulong val1 = input[in_idx1];

            if (sign < 0) {
                if (val0 != 0) val0 = q0 - val0;
                if (val1 != 0) val1 = q1 - val1;
            }

            output[out_idx0] = val0;
            output[out_idx1] = val1;
        }

        // Handle odd remainder if num_primes is odd
        if (remainder) {
            uint prime_idx = num_primes - 1;
            ulong q = moduli[prime_idx];
            uint in_idx = gid * num_primes + prime_idx;
            uint out_idx = target_idx * num_primes + prime_idx;

            ulong val = input[in_idx];
            if (sign < 0 && val != 0) {
                val = q - val;
            }
            output[out_idx] = val;
        }
    }
}
