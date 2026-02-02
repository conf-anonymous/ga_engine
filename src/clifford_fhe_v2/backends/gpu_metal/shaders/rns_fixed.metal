//! Fixed RNS Exact Rescaling for CKKS  
//! Uses simple but correct % operator throughout

#include <metal_stdlib>
using namespace metal;

inline ulong add_mod_lazy(ulong a, ulong b, ulong q) {
    ulong sum = a + b;
    return (sum >= q) ? (sum - q) : sum;
}

inline ulong sub_mod(ulong a, ulong b, ulong q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

/// 128-bit modular multiplication: (a * b) % q
/// Uses iterative doubling to avoid overflow issues
inline ulong mul_mod_128(ulong a, ulong b, ulong q) {
    // Use Russian peasant multiplication algorithm in modular arithmetic
    // This avoids any 128-bit intermediate values
    ulong result = 0;
    a = a % q;  // Ensure a < q

    while (b > 0) {
        if (b & 1) {
            // Add a to result (mod q)
            result = add_mod_lazy(result, a, q);
            if (result >= q) result -= q;
        }

        // Double a (mod q)
        a = add_mod_lazy(a, a, q);
        if (a >= q) a -= q;

        // Halve b
        b >>= 1;
    }

    return result;
}

/// RNS Exact Rescale with Proper DRLMQ
kernel void rns_exact_rescale_fixed(
    device const ulong* poly_in [[buffer(0)]],
    device ulong* poly_out [[buffer(1)]],
    constant ulong* moduli [[buffer(2)]],
    constant ulong* qtop_inv_mod_qi [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& num_primes_in [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint coeff_idx = gid;
    uint num_primes_out = num_primes_in - 1;
    uint top_idx = num_primes_in - 1;

    ulong q_last = moduli[top_idx];
    ulong r_last = poly_in[top_idx * n + coeff_idx];

    // Step 1: Add (q_last - 1)/2 for centered rounding
    ulong half_last = q_last >> 1;
    ulong r_last_rounded = add_mod_lazy(r_last, half_last, q_last);
    if (r_last_rounded >= q_last) {
        r_last_rounded -= q_last;
    }

    // Step 2: For each output prime q_i
    for (uint i = 0; i < num_primes_out; i++) {
        ulong q_i = moduli[i];
        ulong r_i = poly_in[i * n + coeff_idx];
        ulong qtop_inv = qtop_inv_mod_qi[i];

        // Map r_last_rounded to q_i domain
        ulong r_last_mod_qi = r_last_rounded % q_i;

        // Map rounding correction to q_i domain
        ulong half_mod_qi = half_last % q_i;

        // Subtract the mapped rounding correction
        ulong r_last_adjusted = sub_mod(r_last_mod_qi, half_mod_qi, q_i);

        // Compute difference
        ulong diff = sub_mod(r_i, r_last_adjusted, q_i);

        // Multiply by q_last^{-1} mod q_i using 128-bit modular multiplication
        ulong result = mul_mod_128(diff, qtop_inv, q_i);

        // Store result
        poly_out[i * n + coeff_idx] = result;
    }
}
