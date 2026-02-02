/**
 * FIXED CUDA Kernels for Number Theoretic Transform (NTT)
 *
 * Implements Cooley-Tukey DIT NTT to match the working CPU implementation exactly.
 * The original kernel had wrong twiddle indexing which caused the round-trip test to fail.
 *
 * Key fix: Compute twiddle factors correctly per stage using omega^((n/m2) * j)
 */

extern "C" {

/**
 * Modular exponentiation: base^exp mod m
 * Used to compute twiddle factors dynamically
 */
__device__ unsigned long long pow_mod(unsigned long long base, unsigned long long exp, unsigned long long m) {
    unsigned long long result = 1;
    base %= m;

    while (exp > 0) {
        if (exp & 1) {
            // result = (result * base) % m
            unsigned long long lo = result * base;
            unsigned long long hi = __umul64hi(result, base);

            if (hi == 0) {
                result = lo >= m ? lo - m : lo;
            } else {
                // Full 128-bit modular reduction
                double result_d = (double)result;
                double base_d = (double)base;
                double m_d = (double)m;
                double quot_approx = (result_d * base_d) / m_d;
                unsigned long long quot = (unsigned long long)quot_approx;

                unsigned long long qprod_lo = quot * m;
                unsigned long long qprod_hi = __umul64hi(quot, m);

                unsigned long long r_lo = lo - qprod_lo;
                unsigned long long r_hi = hi - qprod_hi - (lo < qprod_lo ? 1 : 0);

                if (r_hi != 0 || r_lo >= m) {
                    if (r_hi != 0) r_lo += m;
                    while (r_lo >= m) r_lo -= m;
                }
                result = r_lo;
            }
        }

        // base = (base * base) % m
        unsigned long long lo = base * base;
        unsigned long long hi = __umul64hi(base, base);

        if (hi == 0) {
            base = lo >= m ? lo - m : lo;
        } else {
            double base_d = (double)base;
            double m_d = (double)m;
            double quot_approx = (base_d * base_d) / m_d;
            unsigned long long quot = (unsigned long long)quot_approx;

            unsigned long long qprod_lo = quot * m;
            unsigned long long qprod_hi = __umul64hi(quot, m);

            unsigned long long r_lo = lo - qprod_lo;
            unsigned long long r_hi = hi - qprod_hi - (lo < qprod_lo ? 1 : 0);

            if (r_hi != 0 || r_lo >= m) {
                if (r_hi != 0) r_lo += m;
                while (r_lo >= m) r_lo -= m;
            }
            base = r_lo;
        }

        exp >>= 1;
    }

    return result;
}

/**
 * Modular multiplication
 */
__device__ unsigned long long mul_mod(unsigned long long a, unsigned long long b, unsigned long long q) {
    unsigned long long lo = a * b;
    unsigned long long hi = __umul64hi(a, b);

    if (hi == 0) {
        return lo >= q ? lo - q : lo;
    }

    double a_d = (double)a;
    double b_d = (double)b;
    double q_d = (double)q;
    double quotient_approx = (a_d * b_d) / q_d;

    unsigned long long quotient = (unsigned long long)quotient_approx;
    unsigned long long qprod_lo = quotient * q;
    unsigned long long qprod_hi = __umul64hi(quotient, q);

    unsigned long long r_lo = lo - qprod_lo;
    unsigned long long r_hi = hi - qprod_hi - (lo < qprod_lo ? 1 : 0);

    if (r_hi != 0 || r_lo >= q) {
        if (r_hi != 0) r_lo += q;
        while (r_lo >= q) r_lo -= q;
    }

    return r_lo;
}

/**
 * Modular addition
 */
__device__ unsigned long long add_mod(unsigned long long a, unsigned long long b, unsigned long long q) {
    unsigned long long sum = a + b;
    return (sum >= q) ? (sum - q) : sum;
}

/**
 * Modular subtraction
 */
__device__ unsigned long long sub_mod(unsigned long long a, unsigned long long b, unsigned long long q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

/**
 * Forward NTT - Cooley-Tukey DIT (matches CPU implementation exactly)
 *
 * This kernel processes ONE stage of the NTT butterfly operations.
 * It must be called log_n times with m doubling each time: m = 1, 2, 4, ..., n/2
 *
 * Parameters:
 * - omega: primitive N-th root of unity
 * - m: current butterfly group size (starts at 1, doubles each stage)
 * - n: ring dimension
 */
__global__ void ntt_forward_fixed(
    unsigned long long* coeffs,
    unsigned long long omega,  // Pass omega directly instead of twiddle array
    unsigned int n,
    unsigned long long q,
    unsigned int m  // Butterfly group size for this stage
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_butterflies = n / 2;

    if (gid >= total_butterflies) return;

    unsigned int m2 = m * 2;

    // Butterfly indices (matches CPU algorithm)
    unsigned int k = gid / m;        // Which butterfly group
    unsigned int j = gid % m;        // Position within group

    unsigned int idx1 = k * m2 + j;
    unsigned int idx2 = idx1 + m;

    // Compute twiddle factor: w_m^j where w_m = omega^(n/m2)
    // This matches CPU: w_m = mod_pow(omega, n/m2, q); w = w_m^j
    unsigned long long w_m_exp = n / m2;
    unsigned long long w_m = pow_mod(omega, w_m_exp, q);
    unsigned long long w = pow_mod(w_m, j, q);

    // Cooley-Tukey butterfly
    unsigned long long u = coeffs[idx1];
    unsigned long long t = mul_mod(w, coeffs[idx2], q);

    coeffs[idx1] = add_mod(u, t, q);
    coeffs[idx2] = sub_mod(u, t, q);
}

/**
 * Inverse NTT - Same as forward but with omega_inv
 */
__global__ void ntt_inverse_fixed(
    unsigned long long* coeffs,
    unsigned long long omega_inv,  // omega^{-1} mod q
    unsigned int n,
    unsigned long long q,
    unsigned int m
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_butterflies = n / 2;

    if (gid >= total_butterflies) return;

    unsigned int m2 = m * 2;

    unsigned int k = gid / m;
    unsigned int j = gid % m;

    unsigned int idx1 = k * m2 + j;
    unsigned int idx2 = idx1 + m;

    // Compute twiddle: w_m^j where w_m = omega_inv^(n/m2)
    unsigned long long w_m_exp = n / m2;
    unsigned long long w_m = pow_mod(omega_inv, w_m_exp, q);
    unsigned long long w = pow_mod(w_m, j, q);

    // Same butterfly as forward
    unsigned long long u = coeffs[idx1];
    unsigned long long t = mul_mod(w, coeffs[idx2], q);

    coeffs[idx1] = add_mod(u, t, q);
    coeffs[idx2] = sub_mod(u, t, q);
}

/**
 * Bit-reversal permutation (unchanged from original)
 */
__global__ void bit_reverse_permutation(
    unsigned long long* coeffs,
    unsigned int n,
    unsigned int log_n
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= n / 2) return;

    unsigned int reversed = 0;
    unsigned int temp = gid;
    for (unsigned int i = 0; i < log_n; i++) {
        reversed = (reversed << 1) | (temp & 1);
        temp >>= 1;
    }

    if (gid < reversed) {
        unsigned long long tmp = coeffs[gid];
        coeffs[gid] = coeffs[reversed];
        coeffs[reversed] = tmp;
    }
}

/**
 * Scalar multiplication: a[i] = (a[i] * scalar) % q
 * Used for final n^{-1} scaling in inverse NTT
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
 * Pointwise multiplication in NTT domain
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

} // extern "C"
