/**
 * CUDA Kernels for Number Theoretic Transform (NTT)
 *
 * Implements Harvey Butterfly NTT algorithm on NVIDIA GPUs
 * Based on the Metal implementation but adapted for CUDA.
 */

extern "C" {

/**
 * Modular multiplication without 128-bit types
 * Returns (a * b) % q
 *
 * Uses Shoup's method: precompute and use floating-point for quotient estimation,
 * then do exact 128-bit arithmetic for the remainder.
 */
/**
 * Modular multiplication: (a * b) mod q
 *
 * Computes the exact 128-bit product and reduces it modulo q.
 * Uses a divide-and-conquer approach that only requires 64-bit operations.
 *
 * Key insight: For 60-bit primes, the product a*b < 2^120.
 * We split this as: a*b = hi * 2^64 + lo
 * And compute: (hi * 2^64 + lo) mod q = ((hi mod q) * (2^64 mod q) + lo) mod q
 *
 * The tricky part is computing 2^64 mod q, but since q < 2^60,
 * we have 2^64 mod q = (2^64 - k*q) for some small k.
 * We can find this by: 2^64 mod q = 2^64 - floor(2^64/q) * q
 *
 * Since 2^64 doesn't fit in u64, we use: 2^64 mod q = (2^32)^2 mod q
 */
__device__ unsigned long long mul_mod(unsigned long long a, unsigned long long b, unsigned long long q) {
    // Compute full 128-bit product: product = hi * 2^64 + lo
    unsigned long long lo = a * b;
    unsigned long long hi = __umul64hi(a, b);

    // Fast path: if hi == 0, product fits in 64 bits, use hardware modulo
    if (hi == 0) {
        return lo % q;
    }

    // Compute 2^64 mod q
    // Since q is ~60 bits, 2^32 < q typically, so 2^32 mod q = 2^32
    // But (2^32)^2 = 2^64 > q, so we need to reduce
    unsigned long long two_32 = 1ULL << 32;
    unsigned long long two_32_mod_q = two_32 % q;  // Usually = 2^32 since q > 2^32

    // (2^32 mod q)^2 might overflow 64 bits, check
    unsigned long long sq_lo = two_32_mod_q * two_32_mod_q;
    unsigned long long sq_hi = __umul64hi(two_32_mod_q, two_32_mod_q);

    unsigned long long two_64_mod_q;
    if (sq_hi == 0) {
        two_64_mod_q = sq_lo % q;
    } else {
        // Rare case: (2^32 mod q)^2 overflows. Use iterative doubling.
        // Compute 2^64 mod q by doubling 64 times
        two_64_mod_q = 1;
        for (int i = 0; i < 64; i++) {
            two_64_mod_q <<= 1;
            if (two_64_mod_q >= q) two_64_mod_q -= q;
        }
    }

    // Now compute (hi mod q) * (2^64 mod q) mod q
    unsigned long long hi_mod_q = hi % q;
    unsigned long long prod_lo = hi_mod_q * two_64_mod_q;
    unsigned long long prod_hi = __umul64hi(hi_mod_q, two_64_mod_q);

    unsigned long long hi_contribution;
    if (prod_hi == 0) {
        hi_contribution = prod_lo % q;
    } else {
        // Need to reduce 128-bit value. Use iterative method.
        // This is rare since hi_mod_q < q < 2^60 and two_64_mod_q < q < 2^60
        // so their product < 2^120, but typically < 2^64 since both are < 2^60
        // and their product < 2^120 but usually fits in 64 bits

        // Use bit-by-bit reduction as fallback
        hi_contribution = 0;
        for (int bit = 63; bit >= 0; bit--) {
            hi_contribution <<= 1;
            if (hi_contribution >= q) hi_contribution -= q;
            if ((prod_hi >> bit) & 1ULL) {
                hi_contribution += 1;
                if (hi_contribution >= q) hi_contribution -= q;
            }
        }
        for (int bit = 63; bit >= 0; bit--) {
            hi_contribution <<= 1;
            if (hi_contribution >= q) hi_contribution -= q;
            if ((prod_lo >> bit) & 1ULL) {
                hi_contribution += 1;
                if (hi_contribution >= q) hi_contribution -= q;
            }
        }
    }

    // Final: (hi_contribution + lo mod q) mod q
    unsigned long long lo_mod_q = lo % q;
    unsigned long long result = hi_contribution + lo_mod_q;
    if (result >= q) result -= q;

    return result;
}

/**
 * Modular addition: (a + b) % q
 */
__device__ unsigned long long add_mod(unsigned long long a, unsigned long long b, unsigned long long q) {
    unsigned long long sum = a + b;
    return (sum >= q) ? (sum - q) : sum;
}

/**
 * Modular subtraction: (a - b) % q
 */
__device__ unsigned long long sub_mod(unsigned long long a, unsigned long long b, unsigned long long q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

/**
 * Bit-reversal permutation
 * Required preprocessing step for NTT
 *
 * Each thread computes the bit-reversal of its index and swaps if needed.
 * IMPORTANT: Must launch with n threads (not n/2)!
 */
__global__ void bit_reverse_permutation(
    unsigned long long* coeffs,
    unsigned int n,
    unsigned int log_n
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= n) return;

    // Compute bit-reversed index
    unsigned int reversed = 0;
    unsigned int temp = gid;
    for (unsigned int i = 0; i < log_n; i++) {
        reversed = (reversed << 1) | (temp & 1);
        temp >>= 1;
    }

    // Only swap if gid < reversed to avoid double-swapping
    if (gid < reversed) {
        unsigned long long tmp = coeffs[gid];
        coeffs[gid] = coeffs[reversed];
        coeffs[reversed] = tmp;
    }
}

/**
 * Forward NTT (Cooley-Tukey butterfly)
 * Transforms polynomial from coefficient to evaluation representation
 *
 * CPU algorithm reference:
 *   for stage in 0..log_n:
 *     m2 = m * 2
 *     w_m = omega^(n/m2)  // twiddle factor for this stage
 *     for k in (0..n step m2):
 *       w = 1
 *       for j in 0..m:
 *         t = w * coeffs[k + j + m]
 *         u = coeffs[k + j]
 *         coeffs[k + j] = u + t
 *         coeffs[k + j + m] = u - t
 *         w = w * w_m
 *     m = m2
 *
 * So twiddle factor for position j in stage is: omega^((n/m2) * j) = twiddles[(n/m2) * j]
 */
__global__ void ntt_forward(
    unsigned long long* coeffs,
    const unsigned long long* twiddles,
    unsigned int n,
    unsigned long long q,
    unsigned int stage,
    unsigned int m
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_butterflies = n / 2;

    if (gid >= total_butterflies) return;

    // Butterfly indices (matches CPU Cooley-Tukey DIT)
    unsigned int m2 = m * 2;
    unsigned int k = gid / m;        // Which butterfly group
    unsigned int j = gid % m;        // Position within group

    unsigned int idx1 = k * m2 + j;
    unsigned int idx2 = idx1 + m;

    // w = omega^((n/m2) * j) = twiddles[(n/m2) * j]
    // Example: n=8, stage 0 (m=1): w_m = omega^4, w[0] = omega^0, w[1] = omega^4 (but m=1, so only j=0)
    // Example: n=8, stage 1 (m=2): w_m = omega^2, w[0] = omega^0, w[1] = omega^2
    // Example: n=8, stage 2 (m=4): w_m = omega^1, w[0] = omega^0, w[1] = omega^1, w[2] = omega^2, w[3] = omega^3
    unsigned int twiddle_stride = n / m2;
    unsigned int twiddle_idx = twiddle_stride * j;  // No modulo needed, j < m < n
    unsigned long long w = twiddles[twiddle_idx];

    // Cooley-Tukey butterfly: (u, v) -> (u + w*v, u - w*v)
    unsigned long long u = coeffs[idx1];
    unsigned long long t = mul_mod(w, coeffs[idx2], q);

    coeffs[idx1] = add_mod(u, t, q);
    coeffs[idx2] = sub_mod(u, t, q);
}

/**
 * Inverse NTT (Gentleman-Sande DIF butterfly)
 * Transforms polynomial from evaluation to coefficient representation
 *
 * This uses the same algorithm as Metal (which works correctly):
 * - Stages run in REVERSE order (log_n-1 down to 0)
 * - Different butterfly formula: (u, v) -> (u + v, (u - v) * w)
 * - Bit-reversal happens at the END (not the beginning)
 *
 * The stage parameter here represents the same stage value as Metal,
 * so the Rust code must call stages in reverse order: log_n-1, log_n-2, ..., 0
 */
__global__ void ntt_inverse(
    unsigned long long* coeffs,
    const unsigned long long* twiddles_inv,
    unsigned int n,
    unsigned long long q,
    unsigned int stage,
    unsigned int m  // m = 1 << (stage + 1), passed from Rust for consistency
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_butterflies = n / 2;

    if (gid >= total_butterflies) return;

    // Butterfly indices (same indexing as Metal's ntt_inverse_stage)
    unsigned int m_half = 1 << stage;  // m_half = m / 2
    unsigned int m2 = m_half * 2;      // m2 = m = 1 << (stage + 1)

    unsigned int block_idx = gid / m_half;
    unsigned int idx_in_block = gid % m_half;

    unsigned int i = block_idx * m2 + idx_in_block;
    unsigned int j = i + m_half;

    // Twiddle index: (n / m) * idx_in_block where m = 1 << (stage + 1)
    unsigned int twiddle_idx = (n / m2) * idx_in_block;
    unsigned long long omega_inv = twiddles_inv[twiddle_idx];

    // Gentleman-Sande (DIF) butterfly: (u, v) -> (u + v, (u - v) * w)
    unsigned long long u = coeffs[i];
    unsigned long long v = coeffs[j];

    coeffs[i] = add_mod(u, v, q);
    coeffs[j] = mul_mod(sub_mod(u, v, q), omega_inv, q);
}

/**
 * Final step of inverse NTT: bit-reversal permutation + scaling by n^(-1)
 *
 * This is performed AFTER all inverse butterfly stages complete.
 * Combines bit-reversal and scaling into one kernel for efficiency.
 */
__global__ void ntt_inverse_final(
    unsigned long long* coeffs,
    unsigned int n,
    unsigned int log_n,
    unsigned long long q,
    unsigned long long n_inv
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= n) return;

    // Compute bit-reversed index
    unsigned int reversed = 0;
    unsigned int temp = gid;
    for (unsigned int i = 0; i < log_n; i++) {
        reversed = (reversed << 1) | (temp & 1);
        temp >>= 1;
    }

    // Only swap if gid <= reversed to avoid double-swapping
    // Also scale by n_inv during the swap
    if (gid <= reversed) {
        unsigned long long val_gid = mul_mod(coeffs[gid], n_inv, q);
        unsigned long long val_rev = mul_mod(coeffs[reversed], n_inv, q);

        coeffs[gid] = val_rev;
        coeffs[reversed] = val_gid;
    }
}

/**
 * Pointwise multiplication in NTT domain
 * c[i] = (a[i] * b[i]) % q for all i
 */
__global__ void ntt_pointwise_multiply(
    const unsigned long long* a,
    const unsigned long long* b,
    unsigned long long* c,
    unsigned int n,
    unsigned long long q
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        c[gid] = mul_mod(a[gid], b[gid], q);
    }
}

/**
 * Scalar multiplication: a[i] = (a[i] * scalar) % q
 */
__global__ void ntt_scalar_multiply(
    unsigned long long* a,
    unsigned long long scalar,
    unsigned int n,
    unsigned long long q
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        a[gid] = mul_mod(a[gid], scalar, q);
    }
}

/**
 * BATCHED OPERATIONS - Process multiple primes in parallel
 *
 * These kernels use 2D grid: (butterfly_blocks, num_primes)
 * to process all RNS primes simultaneously, dramatically reducing
 * kernel launch overhead.
 */

/**
 * Batched bit-reversal permutation - Process all primes in parallel
 *
 * Input layout: data[prime_idx * n + coeff_idx] (flat RNS)
 * Grid: (n_blocks, num_primes, 1) - launch with n threads per prime
 */
__global__ void bit_reverse_permutation_batched(
    unsigned long long* data,
    unsigned int n,
    unsigned int log_n,
    unsigned int num_primes
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int prime_idx = blockIdx.y;

    if (gid >= n || prime_idx >= num_primes) return;

    // Compute bit-reversed index
    unsigned int reversed = 0;
    unsigned int temp = gid;
    for (unsigned int i = 0; i < log_n; i++) {
        reversed = (reversed << 1) | (temp & 1);
        temp >>= 1;
    }

    // Only swap if gid < reversed to avoid double-swapping
    if (gid < reversed) {
        unsigned int prime_offset = prime_idx * n;
        unsigned long long tmp = data[prime_offset + gid];
        data[prime_offset + gid] = data[prime_offset + reversed];
        data[prime_offset + reversed] = tmp;
    }
}

/**
 * Batched Forward NTT - Process all primes in parallel
 *
 * Input layout: data[prime_idx * n + coeff_idx] (flat RNS)
 * Grid: (butterfly_blocks, num_primes, 1)
 *
 * Uses Cooley-Tukey DIT algorithm, matching the single-prime ntt_forward.
 * IMPORTANT: Bit-reversal must be done BEFORE calling this kernel!
 */
__global__ void ntt_forward_batched(
    unsigned long long* data,              // All primes' data [num_primes * n]
    const unsigned long long* twiddles,    // Twiddles for ALL primes [num_primes * n]
    const unsigned long long* moduli,      // RNS moduli [num_primes]
    unsigned int n,                        // Ring dimension
    unsigned int num_primes,               // Number of RNS primes
    unsigned int stage,                    // Current NTT stage
    unsigned int m                         // Butterfly group size = 2^stage
) {
    // 2D grid: x = butterfly index, y = prime index
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int prime_idx = blockIdx.y;

    unsigned int total_butterflies = n / 2;

    if (gid >= total_butterflies || prime_idx >= num_primes) return;

    // Get modulus for this prime
    unsigned long long q = moduli[prime_idx];

    // Offset to this prime's data and twiddles
    unsigned int prime_offset = prime_idx * n;
    unsigned long long* coeffs = data + prime_offset;
    const unsigned long long* twid = twiddles + prime_offset;

    // Cooley-Tukey DIT butterfly indices (matches single-prime ntt_forward)
    unsigned int m2 = m * 2;
    unsigned int k = gid / m;        // Which butterfly group
    unsigned int j = gid % m;        // Position within group

    unsigned int idx1 = k * m2 + j;
    unsigned int idx2 = idx1 + m;

    // w = omega^((n/m2) * j) = twiddles[(n/m2) * j]
    unsigned int twiddle_stride = n / m2;
    unsigned int twiddle_idx = twiddle_stride * j;
    unsigned long long w = twid[twiddle_idx];

    // Cooley-Tukey butterfly: (u, v) -> (u + w*v, u - w*v)
    unsigned long long u = coeffs[idx1];
    unsigned long long t = mul_mod(w, coeffs[idx2], q);

    coeffs[idx1] = add_mod(u, t, q);
    coeffs[idx2] = sub_mod(u, t, q);
}

/**
 * Batched Inverse NTT - Process all primes in parallel
 *
 * Input layout: data[prime_idx * n + coeff_idx] (flat RNS)
 * Grid: (butterfly_blocks, num_primes, 1)
 *
 * Uses Gentleman-Sande DIF algorithm, matching the single-prime ntt_inverse.
 * IMPORTANT: Stages must be called in REVERSE order (log_n-1 down to 0)
 * IMPORTANT: Bit-reversal + scaling must be done AFTER all stages complete!
 */
__global__ void ntt_inverse_batched(
    unsigned long long* data,
    const unsigned long long* twiddles_inv,
    const unsigned long long* moduli,
    unsigned int n,
    unsigned int num_primes,
    unsigned int stage,
    unsigned int m  // m = 1 << (stage + 1), passed from Rust
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int prime_idx = blockIdx.y;

    unsigned int total_butterflies = n / 2;

    if (gid >= total_butterflies || prime_idx >= num_primes) return;

    unsigned long long q = moduli[prime_idx];

    unsigned int prime_offset = prime_idx * n;
    unsigned long long* coeffs = data + prime_offset;
    const unsigned long long* twid_inv = twiddles_inv + prime_offset;

    // Gentleman-Sande DIF butterfly indices (matches single-prime ntt_inverse)
    unsigned int m_half = 1 << stage;  // m_half = m / 2
    unsigned int m2 = m_half * 2;      // m2 = m = 1 << (stage + 1)

    unsigned int block_idx = gid / m_half;
    unsigned int idx_in_block = gid % m_half;

    unsigned int i = block_idx * m2 + idx_in_block;
    unsigned int j = i + m_half;

    // Twiddle index: (n / m2) * idx_in_block
    unsigned int twiddle_idx = (n / m2) * idx_in_block;
    unsigned long long omega_inv = twid_inv[twiddle_idx];

    // Gentleman-Sande (DIF) butterfly: (u, v) -> (u + v, (u - v) * w)
    unsigned long long u = coeffs[i];
    unsigned long long v = coeffs[j];

    coeffs[i] = add_mod(u, v, q);
    coeffs[j] = mul_mod(sub_mod(u, v, q), omega_inv, q);
}

/**
 * Batched final step of inverse NTT: bit-reversal + scaling by n^(-1)
 *
 * This is performed AFTER all inverse butterfly stages complete.
 * Processes all primes in parallel.
 *
 * Grid: (coeff_blocks, num_primes, 1)
 */
__global__ void ntt_inverse_final_batched(
    unsigned long long* data,
    const unsigned long long* n_inv_values,  // n_inv for each prime [num_primes]
    const unsigned long long* moduli,        // moduli for each prime [num_primes]
    unsigned int n,
    unsigned int num_primes,
    unsigned int log_n
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int prime_idx = blockIdx.y;

    if (gid >= n || prime_idx >= num_primes) return;

    unsigned long long q = moduli[prime_idx];
    unsigned long long n_inv = n_inv_values[prime_idx];

    unsigned int prime_offset = prime_idx * n;
    unsigned long long* coeffs = data + prime_offset;

    // Compute bit-reversed index
    unsigned int reversed = 0;
    unsigned int temp = gid;
    for (unsigned int i = 0; i < log_n; i++) {
        reversed = (reversed << 1) | (temp & 1);
        temp >>= 1;
    }

    // Only swap if gid <= reversed to avoid double-swapping
    // Also scale by n_inv during the swap
    if (gid <= reversed) {
        unsigned long long val_gid = mul_mod(coeffs[gid], n_inv, q);
        unsigned long long val_rev = mul_mod(coeffs[reversed], n_inv, q);

        coeffs[gid] = val_rev;
        coeffs[reversed] = val_gid;
    }
}

/**
 * Batched Pointwise Multiplication - Process all primes in parallel
 *
 * c[prime_idx * n + i] = (a[...] * b[...]) % q[prime_idx]
 * Grid: (coeff_blocks, num_primes, 1)
 */
__global__ void ntt_pointwise_multiply_batched(
    const unsigned long long* a,
    const unsigned long long* b,
    unsigned long long* c,
    const unsigned long long* moduli,
    unsigned int n,
    unsigned int num_primes
) {
    unsigned int coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int prime_idx = blockIdx.y;

    if (coeff_idx >= n || prime_idx >= num_primes) return;

    unsigned long long q = moduli[prime_idx];
    unsigned int idx = prime_idx * n + coeff_idx;

    c[idx] = mul_mod(a[idx], b[idx], q);
}

} // extern "C"
