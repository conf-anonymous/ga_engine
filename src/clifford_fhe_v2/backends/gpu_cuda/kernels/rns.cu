/**
 * CUDA Kernels for RNS (Residue Number System) Operations
 *
 * Implements exact CKKS rescaling with centered rounding on NVIDIA GPUs.
 * Based on the DRLMQ (Divide and Round Last Modulus Quotient) algorithm.
 */

extern "C" {

/**
 * Modular addition with lazy reduction
 * Returns a + b (may be >= q, but < 2q)
 */
__device__ __forceinline__ unsigned long long add_mod_lazy(
    unsigned long long a,
    unsigned long long b,
    unsigned long long q
) {
    unsigned long long sum = a + b;
    return (sum >= q) ? (sum - q) : sum;
}

/**
 * 128-bit modular multiplication using Russian peasant algorithm
 * Returns (a * b) % q without using 128-bit integers
 *
 * This avoids overflow issues by computing the product iteratively.
 */
__device__ __forceinline__ unsigned long long mul_mod_128(
    unsigned long long a,
    unsigned long long b,
    unsigned long long q
) {
    unsigned long long result = 0;
    a = a % q;

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

/**
 * Modular subtraction: (a - b) % q
 */
__device__ unsigned long long sub_mod(
    unsigned long long a,
    unsigned long long b,
    unsigned long long q
) {
    return (a >= b) ? (a - b) : (a + q - b);
}

/**
 * RNS Exact Rescaling with Centered Rounding (DRLMQ)
 *
 * Implements: C' = ⌊(C + q_last/2) / q_last⌋ mod Q'
 *
 * Algorithm per coefficient:
 * 1. Add q_last/2 to the last RNS limb for rounding
 * 2. For each output prime q_i:
 *    diff = (r_i - r_last_rounded) mod q_i
 *    result_i = (diff * q_last^{-1}) mod q_i
 *
 * Input layout:  poly_in[prime_idx * n + coeff_idx] (flat RNS)
 * Output layout: poly_out[prime_idx * n + coeff_idx] (flat RNS)
 */
__global__ void rns_exact_rescale(
    const unsigned long long* poly_in,        // Input: [n × num_primes_in] flat layout
    unsigned long long* poly_out,             // Output: [n × num_primes_out] flat layout
    const unsigned long long* moduli,         // RNS moduli: [num_primes_in]
    const unsigned long long* qtop_inv,       // q_last^{-1} mod q_i: [num_primes_out]
    unsigned int n,                           // Ring dimension
    unsigned int num_primes_in,               // Number of input primes
    unsigned int num_primes_out               // Number of output primes (= num_primes_in - 1)
) {
    unsigned int coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (coeff_idx >= n) return;

    // Last prime is being dropped (q_top = moduli[num_primes_in - 1])
    unsigned int top_idx = num_primes_in - 1;

    // Get q_top and r_top
    unsigned long long q_top = moduli[top_idx];
    unsigned long long r_top = poly_in[top_idx * n + coeff_idx];

    // Step 1: Add (q_top-1)/2 to r_top for centered rounding (mod q_top)
    unsigned long long half_qtop = q_top >> 1;
    unsigned long long r_top_rounded = add_mod_lazy(r_top, half_qtop, q_top);
    if (r_top_rounded >= q_top) r_top_rounded -= q_top;

    // Step 2: For each output prime q_i (all primes except q_top)
    for (unsigned int i = 0; i < num_primes_out; i++) {
        unsigned long long q_i = moduli[i];
        unsigned long long r_i = poly_in[i * n + coeff_idx];
        unsigned long long q_inv = qtop_inv[i];

        // Reduce r_top_rounded mod q_i
        unsigned long long r_top_mod_qi = r_top_rounded % q_i;

        // Subtract half for centered rounding correction
        unsigned long long half_mod_qi = half_qtop % q_i;
        unsigned long long r_top_corrected = sub_mod(r_top_mod_qi, half_mod_qi, q_i);

        // diff = (r_i - r_top_corrected) mod q_i
        unsigned long long diff = sub_mod(r_i, r_top_corrected, q_i);

        // result = (diff * q_top^{-1}) mod q_i
        // Use 128-bit multiplication to avoid overflow
        unsigned long long result = mul_mod_128(diff, q_inv, q_i);

        // Store in flat layout: poly_out[prime_idx * n + coeff_idx]
        poly_out[i * n + coeff_idx] = result;
    }
}

/**
 * RNS Exact Rescaling with Centered Rounding - STRIDED LAYOUT VERSION
 *
 * Same algorithm as rns_exact_rescale, but works directly on strided layout.
 * This avoids expensive CPU layout conversions (1.3M operations per rescale).
 *
 * Input layout:  poly_in[coeff_idx * num_primes_in + prime_idx] (strided)
 * Output layout: poly_out[coeff_idx * num_primes_out + prime_idx] (strided)
 */
__global__ void rns_exact_rescale_strided(
    const unsigned long long* poly_in,        // Input: strided layout
    unsigned long long* poly_out,             // Output: strided layout
    const unsigned long long* moduli,         // RNS moduli: [num_primes_in]
    const unsigned long long* qtop_inv,       // q_last^{-1} mod q_i: [num_primes_out]
    unsigned int n,                           // Ring dimension
    unsigned int num_primes_in,               // Number of input primes
    unsigned int num_primes_out               // Number of output primes (= num_primes_in - 1)
) {
    unsigned int coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (coeff_idx >= n) return;

    // Last prime is being dropped (q_top = moduli[num_primes_in - 1])
    unsigned int top_idx = num_primes_in - 1;

    // Get q_top and r_top from strided layout
    unsigned long long q_top = moduli[top_idx];
    unsigned long long r_top = poly_in[coeff_idx * num_primes_in + top_idx];

    // Step 1: Add (q_top-1)/2 to r_top for centered rounding (mod q_top)
    unsigned long long half_qtop = q_top >> 1;
    unsigned long long r_top_rounded = add_mod_lazy(r_top, half_qtop, q_top);
    if (r_top_rounded >= q_top) r_top_rounded -= q_top;

    // Step 2: For each output prime q_i (all primes except q_top)
    for (unsigned int i = 0; i < num_primes_out; i++) {
        unsigned long long q_i = moduli[i];
        unsigned long long r_i = poly_in[coeff_idx * num_primes_in + i];
        unsigned long long q_inv = qtop_inv[i];

        // Reduce r_top_rounded mod q_i
        unsigned long long r_top_mod_qi = r_top_rounded % q_i;

        // Subtract half for centered rounding correction
        unsigned long long half_mod_qi = half_qtop % q_i;
        unsigned long long r_top_corrected = sub_mod(r_top_mod_qi, half_mod_qi, q_i);

        // diff = (r_i - r_top_corrected) mod q_i
        unsigned long long diff = sub_mod(r_i, r_top_corrected, q_i);

        // result = (diff * q_top^{-1}) mod q_i
        unsigned long long result = mul_mod_128(diff, q_inv, q_i);

        // Store in strided layout: poly_out[coeff_idx * num_primes_out + prime_idx]
        poly_out[coeff_idx * num_primes_out + i] = result;
    }
}

/**
 * Polynomial addition in RNS representation
 * c[i] = (a[i] + b[i]) % q for each RNS limb
 */
__global__ void rns_add(
    const unsigned long long* a,
    const unsigned long long* b,
    unsigned long long* c,
    const unsigned long long* moduli,
    unsigned int n,
    unsigned int num_primes
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = n * num_primes;

    if (idx >= total_elements) return;

    // Determine which prime we're working with
    unsigned int prime_idx = idx / n;
    unsigned long long q = moduli[prime_idx];

    // Add with modular reduction
    unsigned long long sum = a[idx] + b[idx];
    c[idx] = (sum >= q) ? (sum - q) : sum;
}

/**
 * Polynomial subtraction in RNS representation
 * c[i] = (a[i] - b[i]) % q for each RNS limb
 */
__global__ void rns_sub(
    const unsigned long long* a,
    const unsigned long long* b,
    unsigned long long* c,
    const unsigned long long* moduli,
    unsigned int n,
    unsigned int num_primes
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = n * num_primes;

    if (idx >= total_elements) return;

    // Determine which prime we're working with
    unsigned int prime_idx = idx / n;
    unsigned long long q = moduli[prime_idx];

    // Subtract with modular reduction
    c[idx] = (a[idx] >= b[idx]) ? (a[idx] - b[idx]) : (a[idx] + q - b[idx]);
}

/**
 * Polynomial negation in RNS representation
 * b[i] = (-a[i]) % q = (q - a[i]) % q for each RNS limb
 */
__global__ void rns_negate(
    const unsigned long long* a,
    unsigned long long* b,
    const unsigned long long* moduli,
    unsigned int n,
    unsigned int num_primes
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = n * num_primes;

    if (idx >= total_elements) return;

    // Determine which prime we're working with
    unsigned int prime_idx = idx / n;
    unsigned long long q = moduli[prime_idx];

    // Negate: -a mod q = q - a (when a != 0)
    b[idx] = (a[idx] == 0) ? 0 : (q - a[idx]);
}

/**
 * Pointwise multiplication in RNS representation (strided layout)
 * c[i] = (a[i] * b[i]) % q for each RNS limb
 *
 * Uses 128-bit modular multiplication to avoid overflow.
 * Input/output layout: poly[coeff_idx * stride + prime_idx] (strided)
 */
__global__ void rns_pointwise_multiply_strided(
    const unsigned long long* a,         // First polynomial (strided)
    const unsigned long long* b,         // Second polynomial (strided)
    unsigned long long* c,               // Result (strided)
    const unsigned long long* moduli,    // RNS moduli
    unsigned int n,                      // Ring dimension
    unsigned int stride,                 // Stride (usually num_primes_total)
    unsigned int num_primes              // Number of active primes
) {
    unsigned int coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (coeff_idx >= n) return;

    // Process all primes for this coefficient
    for (unsigned int prime_idx = 0; prime_idx < num_primes; prime_idx++) {
        unsigned int idx = coeff_idx * stride + prime_idx;
        unsigned long long q = moduli[prime_idx];

        // Multiply with 128-bit safety
        unsigned long long result = mul_mod_128(a[idx], b[idx], q);
        c[idx] = result;
    }
}

/**
 * Layout conversion: Strided → Flat
 *
 * Strided: poly_in[coeff_idx * stride + prime_idx]
 * Flat:    poly_out[prime_idx * n + coeff_idx]
 *
 * This is a memory-bound operation perfect for GPU parallelization.
 * Avoids expensive CPU loops (650k+ operations per conversion).
 */
__global__ void rns_strided_to_flat(
    const unsigned long long* poly_in,   // Strided layout
    unsigned long long* poly_out,        // Flat layout
    unsigned int n,                      // Ring dimension
    unsigned int stride,                 // Stride in input (usually num_primes_total)
    unsigned int num_primes              // Number of active primes
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = n * num_primes;

    if (idx >= total_elements) return;

    // Decompose idx = prime_idx * n + coeff_idx (flat layout)
    unsigned int prime_idx = idx / n;
    unsigned int coeff_idx = idx % n;

    // Read from strided layout and write to flat layout
    poly_out[idx] = poly_in[coeff_idx * stride + prime_idx];
}

/**
 * Layout conversion: Flat → Strided
 *
 * Flat:    poly_in[prime_idx * n + coeff_idx]
 * Strided: poly_out[coeff_idx * stride + prime_idx]
 */
__global__ void rns_flat_to_strided(
    const unsigned long long* poly_in,   // Flat layout
    unsigned long long* poly_out,        // Strided layout
    unsigned int n,                      // Ring dimension
    unsigned int stride,                 // Stride in output (usually num_primes_total)
    unsigned int num_primes              // Number of active primes
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = n * num_primes;

    if (idx >= total_elements) return;

    // Decompose idx = prime_idx * n + coeff_idx (flat layout)
    unsigned int prime_idx = idx / n;
    unsigned int coeff_idx = idx % n;

    // Read from flat layout and write to strided layout
    poly_out[coeff_idx * stride + prime_idx] = poly_in[idx];
}

/**
 * Negacyclic TWIST: Multiply polynomial by psi^i for each coefficient
 *
 * This converts the standard polynomial ring multiplication to negacyclic
 * convolution in R[X]/(X^N + 1), which is required for CKKS.
 *
 * For each coefficient a[i], we compute: a[i] = a[i] * psi^i mod q
 *
 * Layout: FLAT - poly[prime_idx * n + coeff_idx]
 *
 * psi_powers layout: psi_powers[prime_idx * n + coeff_idx] = psi[prime_idx]^coeff_idx
 */
__global__ void rns_negacyclic_twist(
    unsigned long long* poly,            // In/Out: polynomial (flat layout)
    const unsigned long long* psi_powers, // psi^i for each (prime, coeff) [num_primes * n]
    const unsigned long long* moduli,    // RNS moduli
    unsigned int n,                      // Ring dimension
    unsigned int num_primes              // Number of active primes
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = n * num_primes;

    if (idx >= total_elements) return;

    // Determine which prime we're working with
    unsigned int prime_idx = idx / n;
    unsigned long long q = moduli[prime_idx];

    // Multiply by psi^coeff_idx
    poly[idx] = mul_mod_128(poly[idx], psi_powers[idx], q);
}

/**
 * Negacyclic UNTWIST: Multiply polynomial by psi^{-i} for each coefficient
 *
 * This is the inverse of the twist operation, applied after inverse NTT
 * to get the final negacyclic convolution result.
 *
 * For each coefficient a[i], we compute: a[i] = a[i] * psi^{-i} mod q
 *
 * Layout: FLAT - poly[prime_idx * n + coeff_idx]
 *
 * psi_inv_powers layout: psi_inv_powers[prime_idx * n + coeff_idx] = psi[prime_idx]^{-coeff_idx}
 */
__global__ void rns_negacyclic_untwist(
    unsigned long long* poly,                // In/Out: polynomial (flat layout)
    const unsigned long long* psi_inv_powers, // psi^{-i} for each (prime, coeff) [num_primes * n]
    const unsigned long long* moduli,        // RNS moduli
    unsigned int n,                          // Ring dimension
    unsigned int num_primes                  // Number of active primes
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = n * num_primes;

    if (idx >= total_elements) return;

    // Determine which prime we're working with
    unsigned int prime_idx = idx / n;
    unsigned long long q = moduli[prime_idx];

    // Multiply by psi^{-coeff_idx}
    poly[idx] = mul_mod_128(poly[idx], psi_inv_powers[idx], q);
}

} // extern "C"
