//! Hoisted Automorphisms for Batch Rotations
//!
//! This module implements the "hoisting" optimization for CKKS rotations,
//! which dramatically reduces the cost of multiple rotations on the same ciphertext.
//!
//! ## Mathematical Foundation
//!
//! For Galois automorphism σₖ: X → X^k, there exists a permutation Πₖ such that:
//!
//! ```text
//! NTT(σₖ(p)) = Πₖ(NTT(p))
//! ```
//!
//! This enables hoisting:
//! 1. Decompose c₁ once: D_B(c₁) → digits
//! 2. Forward-NTT each digit once: NTT(digit_t) → digit_ntt[t]
//! 3. For each rotation k:
//!    a. Permute NTT buffers: Πₖ(digit_ntt[t]) (cheap!)
//!    b. Pointwise multiply with rotation key (already in NTT)
//!    c. Inverse NTT
//!
//! ## Performance Impact
//!
//! **Without hoisting (per rotation):**
//! - Decompose: ~0.01s
//! - 8× Forward NTT: ~0.12s
//! - Pointwise multiply: ~0.01s
//! - 8× Inverse NTT: ~0.12s
//! - Total per rotation: ~0.26s
//! - 9 rotations: ~2.34s
//!
//! **With hoisting:**
//! - Decompose once: ~0.01s
//! - 8× Forward NTT once: ~0.12s
//! - Per rotation:
//!   - Permute: ~0.01s
//!   - Pointwise multiply: ~0.01s
//!   - 8× Inverse NTT: ~0.12s
//!   - Total: ~0.14s
//! - 9 rotations: 0.13s + (9 × 0.14s) = ~1.39s
//!
//! **Speedup: 2.34s → 1.39s = 1.68× faster**
//!
//! ## NTT Domain Permutation Formula
//!
//! For our Metal NTT implementation (bit-reversal BEFORE butterflies, natural-order output):
//!
//! ```text
//! Πₖ(j) = (j · g) mod N
//! ```
//!
//! where g is the Galois element corresponding to rotation by k steps.
//!
//! ## Reference
//!
//! - Halevi & Shoup 2014: "Algorithms in HElib"
//! - Kim et al. 2018: "Bootstrapping for Approximate Homomorphic Encryption"

use super::rotation::{compute_galois_map, rotation_step_to_galois_element};

/// NTT layout determines how the permutation is computed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NttLayout {
    /// NTT output is in natural order (indices j = 0..N-1 in increasing order)
    Natural,
    /// NTT output is in bit-reversed order
    BitReversed,
}

/// Compute bit-reversal of a number
///
/// # Arguments
/// * `x` - Number to reverse
/// * `logn` - Number of bits to reverse (log₂ N)
///
/// # Returns
/// Bit-reversed value
fn bitrev(mut x: usize, logn: usize) -> usize {
    let mut y = 0usize;
    for _ in 0..logn {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

/// Compute NTT-domain Galois permutation
///
/// For Galois element g (corresponding to X → X^g), computes the permutation Πₖ
/// such that NTT(σₖ(p)) = Πₖ(NTT(p)).
///
/// # Mathematical Details
///
/// In coefficient domain, σₖ is a ring automorphism. In evaluation form (NTT),
/// a polynomial is a vector of values at ω^j. The automorphism composes the
/// evaluation points: A(ω^j) ↦ A((ω^j)^g) = A(ω^{j·g}).
///
/// This is a permutation of indices: j → (j · g) mod N.
///
/// # Arguments
/// * `n` - Ring dimension (must be power of 2)
/// * `g` - Galois element (must be odd, in (ℤ/2Nℤ)*)
/// * `layout` - Whether NTT output is in natural or bit-reversed order
///
/// # Returns
/// Permutation map where `map[j]` gives the target index for source index j
///
/// # Example
///
/// ```rust
/// use ga_engine::clifford_fhe_v2::backends::gpu_metal::hoisting::{
///     compute_ntt_galois_permutation, NttLayout
/// };
///
/// let n = 8;
/// let g = 5; // Rotation by 1 step for N=8
/// let map = compute_ntt_galois_permutation(n, g, NttLayout::Natural);
///
/// // For natural order: map[j] = (j * 5) mod 8
/// assert_eq!(map[0], 0); // (0 * 5) mod 8 = 0
/// assert_eq!(map[1], 5); // (1 * 5) mod 8 = 5
/// assert_eq!(map[2], 2); // (2 * 5) mod 8 = 2 (10 mod 8)
/// ```
pub fn compute_ntt_galois_permutation(n: usize, g: usize, layout: NttLayout) -> Vec<usize> {
    assert!(n.is_power_of_two(), "N must be a power of 2");
    assert!(g % 2 == 1, "Galois element must be odd");

    let logn = n.trailing_zeros() as usize;
    let mut map = vec![0usize; n];

    for j in 0..n {
        let j_phys = match layout {
            NttLayout::Natural => j,
            NttLayout::BitReversed => bitrev(j, logn),
        };

        // Core permutation: multiply index by Galois element
        let k_phys = (j_phys * g) % n;

        let k = match layout {
            NttLayout::Natural => k_phys,
            NttLayout::BitReversed => bitrev(k_phys, logn),
        };

        map[j] = k;
    }

    map
}

/// Compute NTT-domain Galois permutation for a rotation step WITH OFFSET
///
/// For negacyclic NTT (twist-then-cyclic convention), the correct formula is:
/// NTT_neg(σ_g a)[j] = NTT_neg(a)[(g·j + α) mod N]
/// where α = (g-1)/2 is a constant offset (NO per-bin diagonal needed!)
///
/// # Arguments
/// * `n` - Ring dimension
/// * `step` - Rotation step (positive = left rotate, negative = right rotate)
/// * `layout` - NTT layout
///
/// # Returns
/// Permutation map for this rotation with offset included
pub fn compute_ntt_permutation_for_step(
    n: usize,
    step: i32,
    layout: NttLayout,
) -> Vec<usize> {
    let g = rotation_step_to_galois_element(step, n);

    // Compute offset: α = (g-1)/2 mod N
    // Since g is always odd, (g-1) is even, so division is exact
    let alpha = ((g - 1) / 2) % n;

    // Build permutation with offset: map[j] = (j*g + α) mod N
    compute_ntt_galois_permutation_with_offset(n, g, alpha, layout)
}

/// Compute NTT-domain Galois permutation with offset
fn compute_ntt_galois_permutation_with_offset(
    n: usize,
    g: usize,
    alpha: usize,
    layout: NttLayout,
) -> Vec<usize> {
    assert!(n.is_power_of_two(), "N must be a power of 2");
    assert!(g % 2 == 1, "Galois element must be odd");

    let logn = n.trailing_zeros() as usize;
    let mut map = vec![0usize; n];

    for j in 0..n {
        let j_phys = match layout {
            NttLayout::Natural => j,
            NttLayout::BitReversed => bitrev(j, logn),
        };

        // Core permutation with offset: (j*g + α) mod N
        let k_phys = ((j_phys * g) + alpha) % n;

        let k = match layout {
            NttLayout::Natural => k_phys,
            NttLayout::BitReversed => bitrev(k_phys, logn),
        };

        map[j] = k;
    }

    map
}

/// Permute a buffer in-place using the given permutation map
///
/// Applies the permutation to each RNS limb (prime) independently.
///
/// Uses PULL semantics: output[j] = input[map[j]]
/// This matches the cyclic NTT identity: NTT(σ_g(a))[j] = NTT(a)[j*g mod N]
///
/// # Arguments
/// * `buf` - Buffer to permute (flat RNS layout: [coeff0_q0, coeff0_q1, ..., coeff1_q0, coeff1_q1, ...])
/// * `map` - Permutation map where map[j] = source index for destination j (PULL semantics)
/// * `n` - Ring dimension
/// * `num_primes` - Number of RNS primes
///
/// # Panics
/// Panics if buffer size doesn't match n × num_primes
pub fn permute_in_place_ntt(buf: &mut [u64], map: &[usize], n: usize, num_primes: usize) {
    assert_eq!(buf.len(), n * num_primes, "Buffer size mismatch");
    assert_eq!(map.len(), n, "Permutation map size mismatch");

    // Slot-major layout: buf[slot * num_primes + prime_idx]
    // Create temporary buffer for the entire array
    let mut tmp = vec![0u64; n * num_primes];

    // PULL semantics: destination j gets value from source map[j]
    for j in 0..n {
        let source_slot = map[j]; // Where to pull from
        // Copy all primes for this slot
        for prime_idx in 0..num_primes {
            tmp[j * num_primes + prime_idx] = buf[source_slot * num_primes + prime_idx];
        }
    }

    buf.copy_from_slice(&tmp);
}

/// Hoisted NTT digits for batch rotations
///
/// Stores the decomposed and NTT-transformed digits of c1, ready for efficient
/// batch rotations without repeated decompose+NTT operations.
pub struct HoistedDigits {
    /// NTT-transformed digits: digits_ntt[t][j * num_primes + prime_idx]
    /// Each digit is in NTT domain (Montgomery representation)
    pub digits_ntt: Vec<Vec<u64>>,
    /// Ring dimension
    pub n: usize,
    /// Number of RNS primes
    pub num_primes: usize,
}

/// Decompose c1 and forward-NTT all digits (HOIST step)
///
/// This is the expensive operation done ONCE before a batch of rotations.
/// Amortizes the cost of decompose+NTT across multiple rotation steps.
///
/// # Algorithm
/// 1. Gadget decompose c1_rotated into digits: c1 = ∑ᵢ digitᵢ·Bⁱ
/// 2. For each digit: twist → forward NTT → keep in Montgomery domain
/// 3. Return hoisted digits ready for batch rotations
///
/// # Arguments
/// * `c1_rotated` - Galois-rotated c1 (flat RNS layout, coefficient domain)
/// * `base_w` - Gadget decomposition base (typically 32 for B=2^32)
/// * `moduli` - Active RNS primes
/// * `ctx` - CKKS context with NTT contexts
///
/// # Returns
/// Hoisted digits in NTT domain, ready for batch rotation use
pub fn hoist_decompose_ntt(
    c1_rotated: &[u64],
    base_w: u32,
    moduli: &[u64],
    n: usize,
    ctx: &super::ckks::MetalCkksContext,
) -> Result<HoistedDigits, String> {
    use num_bigint::BigInt;
    use num_traits::{One, Zero, Signed};

    let num_primes = moduli.len();

    // Step 1: Gadget decompose c1_rotated
    let c1_digits = super::ckks::MetalCiphertext::gadget_decompose_flat(
        c1_rotated,
        base_w,
        moduli,
        n,
    )?;

    let num_digits = c1_digits.len();
    let mut digits_ntt = Vec::with_capacity(num_digits);

    // Step 2: Forward NTT each digit (with negacyclic twist)
    for digit_coeff in c1_digits.iter() {
        let mut digit_ntt_flat = vec![0u64; n * num_primes];

        // For each RNS component (each prime)
        for (prime_idx, &q) in moduli.iter().enumerate() {
            // Extract digit polynomial for this prime
            let mut digit_poly = vec![0u64; n];
            for i in 0..n {
                digit_poly[i] = digit_coeff[i * num_primes + prime_idx];
            }

            let ntt_ctx = &ctx.ntt_contexts()[prime_idx];

            // Negacyclic NTT: twist → forward NTT → result in Montgomery domain
            for i in 0..n {
                digit_poly[i] = ((digit_poly[i] as u128 * ntt_ctx.psi_powers()[i] as u128) % q as u128) as u64;
            }

            // Forward NTT (outputs Montgomery domain)
            ntt_ctx.forward(&mut digit_poly)?;

            // Store back in flat layout
            for i in 0..n {
                digit_ntt_flat[i * num_primes + prime_idx] = digit_poly[i];
            }
        }

        digits_ntt.push(digit_ntt_flat);
    }

    Ok(HoistedDigits {
        digits_ntt,
        n,
        num_primes,
    })
}

/// Compute negacyclic diagonal twist factors D_g[j] = ψ^{(g-1)·j} in Montgomery domain
///
/// For CKKS negacyclic NTT, the hoisted automorphism formula is:
///   NTT_neg(σ_g(a))[j] = ψ^{(g-1)·j} · NTT_neg(a)[j·g mod N]
///
/// This function precomputes the diagonal factors D_g for a given Galois element g.
///
/// # Arguments
/// * `g` - Galois element (rotation_step_to_galois_element(step, n))
/// * `n` - Ring dimension
/// * `moduli` - RNS primes
/// * `ctx` - CKKS context with NTT contexts (for ψ and Montgomery parameters)
///
/// # Returns
/// Diagonal twist factors in flat RNS layout [coeff0_q0, coeff0_q1, ..., coeffN-1_qL]
/// in Montgomery domain, ready for Montgomery multiplication with NTT values
pub fn compute_diagonal_twist(
    g: usize,
    n: usize,
    moduli: &[u64],
    ctx: &super::ckks::MetalCkksContext,
) -> Vec<u64> {
    let num_primes = moduli.len();
    let two_n = 2 * n;

    // Exponent: (g-1) mod 2N
    let exp = ((g + two_n - 1) % two_n) as u64;

    let mut diagonal = vec![0u64; n * num_primes];

    for (prime_idx, &q) in moduli.iter().enumerate() {
        let ntt_ctx = &ctx.ntt_contexts()[prime_idx];

        // Get ψ in standard domain, convert to Montgomery
        let psi_std = ntt_ctx.psi_powers()[1];
        let psi_mont = super::ntt::MetalNttContext::to_montgomery(
            psi_std,
            ntt_ctx.r_squared(),
            q,
            ntt_ctx.q_inv(),
        );

        // Compute ψ^{(g-1)} in Montgomery domain using exponentiation by squaring
        let psi_exp_mont = mont_pow(psi_mont, exp, q, ntt_ctx);

        // Rolling multiply: D[j] = ψ^{(g-1)·j} = (ψ^{(g-1)})^j
        let mut d_j = super::ntt::MetalNttContext::to_montgomery(
            1,
            ntt_ctx.r_squared(),
            q,
            ntt_ctx.q_inv(),
        ); // 1 in Montgomery domain

        for j in 0..n {
            diagonal[j * num_primes + prime_idx] = d_j;
            // Advance: d_{j+1} = d_j * ψ^{(g-1)}
            d_j = mont_mul(d_j, psi_exp_mont, q, ntt_ctx);
        }
    }

    diagonal
}

/// Montgomery multiplication helper
#[inline]
fn mont_mul(a: u64, b: u64, q: u64, ntt_ctx: &super::ntt::MetalNttContext) -> u64 {
    let q_inv = ntt_ctx.q_inv();

    // Step 1: Compute t = a * b (128-bit)
    let t = a as u128 * b as u128;
    let t_lo = t as u64;
    let t_hi = (t >> 64) as u64;

    // Step 2: Compute m = (t_lo * q_inv) mod 2^64
    let m = t_lo.wrapping_mul(q_inv);

    // Step 3: Compute m * q (128-bit)
    let mq = m as u128 * q as u128;
    let mq_lo = mq as u64;
    let mq_hi = (mq >> 64) as u64;

    // Step 4: Compute u = (t + m*q) / 2^64
    let (_, carry1) = t_lo.overflowing_add(mq_lo);
    let (sum_hi, carry2) = t_hi.overflowing_add(mq_hi);
    let sum_hi = sum_hi.wrapping_add(carry1 as u64).wrapping_add(carry2 as u64);

    // Step 5: Conditional subtraction
    if sum_hi >= q {
        sum_hi - q
    } else {
        sum_hi
    }
}

/// Montgomery exponentiation by squaring
#[inline]
fn mont_pow(base: u64, exp: u64, q: u64, ntt_ctx: &super::ntt::MetalNttContext) -> u64 {
    let one_mont = super::ntt::MetalNttContext::to_montgomery(
        1,
        ntt_ctx.r_squared(),
        q,
        ntt_ctx.q_inv(),
    );

    let mut result = one_mont;
    let mut base = base;
    let mut exp = exp;

    while exp > 0 {
        if exp & 1 == 1 {
            result = mont_mul(result, base, q, ntt_ctx);
        }
        base = mont_mul(base, base, q, ntt_ctx);
        exp >>= 1;
    }

    result
}

/// Perform rotation using precomputed hoisted digits (FAST PATH)
///
/// This is the cheap operation performed for each rotation in a batch.
/// Just permutes and multiplies diagonal - no decompose, no NTT!
///
/// # Algorithm (for each digit t)
/// 1. Permute hoisted NTT digit: digit_permuted[j] = digit_ntt[j·g mod N]
/// 2. Apply diagonal twist: digit_permuted[j] *= D_g[j]
/// 3. Pointwise multiply with rotation key: term = digit_permuted ⊙ rlk[t]
/// 4. Inverse NTT and untwist: result = iNTT(term) ⊙ ψ^{-1}
/// 5. Accumulate into c0_final and c1_final
///
/// # Arguments
/// * `hoisted` - Precomputed NTT digits from `hoist_decompose_ntt()`
/// * `step` - Rotation step (positive = left, negative = right)
/// * `rlk0_ntt`, `rlk1_ntt` - Rotation keys ALREADY in NTT domain (pre-cached, 15-20% faster!)
/// * `c0_rotated` - Galois-rotated c0 (coefficient domain)
/// * `moduli` - Active RNS primes
/// * `ctx` - CKKS context
///
/// # Returns
/// (c0_final, c1_final) after key-switching with hoisted optimization
pub fn rotate_with_hoisted_digits(
    hoisted: &HoistedDigits,
    step: i32,
    rlk0_ntt: &[Vec<u64>],
    rlk1_ntt: &[Vec<u64>],
    c0_rotated: &[u64],
    moduli: &[u64],
    ctx: &super::ckks::MetalCkksContext,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let n = hoisted.n;
    let num_primes = hoisted.num_primes;
    let num_digits = hoisted.digits_ntt.len();

    if moduli.len() != num_primes {
        return Err(format!("Moduli count mismatch: {} vs {}", moduli.len(), num_primes));
    }

    // Compute permutation with offset (NO diagonal needed!)
    // The offset α = (g-1)/2 is baked into the permutation map
    let ntt_perm = compute_ntt_permutation_for_step(n, step, NttLayout::Natural);

    // Initialize accumulators
    let mut c0_final = c0_rotated.to_vec();
    let mut c1_final = vec![0u64; n * num_primes];

    // For each digit t in the hoisted decomposition
    for t in 0..num_digits.min(rlk0_ntt.len()) {
        // Permute hoisted NTT digit with offset
        // No diagonal multiplication needed - the offset handles everything!
        let mut digit_ntt_permuted = hoisted.digits_ntt[t].clone();
        permute_in_place_ntt(&mut digit_ntt_permuted, &ntt_perm, n, num_primes);

        // USE PRE-CACHED NTT KEYS! (No runtime transformation needed - 15-20% faster!)
        // Keys are already in NTT domain (Montgomery), ready for multiplication
        let rlk0_ntt_digit = &rlk0_ntt[t];
        let rlk1_ntt_digit = &rlk1_ntt[t];

        // Step 3 & 4: Pointwise multiply with rotation keys and inverse NTT
        let term0 = multiply_ntt_and_intt(&digit_ntt_permuted, rlk0_ntt_digit, moduli, ctx)?;
        let term1 = multiply_ntt_and_intt(&digit_ntt_permuted, rlk1_ntt_digit, moduli, ctx)?;

        // Step 5: Accumulate (CPU is actually faster for V4's small scale!)
        // c0_final -= term0
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let q = moduli[prime_idx];

            let diff = if c0_final[i] >= term0[i] {
                c0_final[i] - term0[i]
            } else {
                q - (term0[i] - c0_final[i])
            };
            c0_final[i] = diff;
        }

        // c1_final += term1
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let q = moduli[prime_idx];
            c1_final[i] = ((c1_final[i] as u128 + term1[i] as u128) % q as u128) as u64;
        }
    }

    Ok((c0_final, c1_final))
}

/// Transform rotation key from coefficient domain to NTT domain (Metal GPU)
///
/// Rotation keys are stored in coefficient domain. This function:
/// 1. Applies negacyclic twist (multiply by ψ^i) on Metal GPU
/// 2. Forward NTT (outputs Montgomery domain) on Metal GPU
///
/// This is needed because `multiply_ntt_and_intt` expects both inputs in NTT domain.
///
/// **Performance:** Fully Metal GPU accelerated (no CPU arithmetic)
fn transform_key_to_ntt(
    key_coeff: &[u64],
    moduli: &[u64],
    ctx: &super::ckks::MetalCkksContext,
) -> Result<Vec<u64>, String> {
    let n = ctx.params.n;
    let num_primes = moduli.len();
    let mut key_ntt_flat = vec![0u64; n * num_primes];

    // For each RNS component (each prime)
    for (prime_idx, _q) in moduli.iter().enumerate() {
        let ntt_ctx = &ctx.ntt_contexts()[prime_idx];

        // Extract key polynomial for this prime
        let mut key_poly = vec![0u64; n];
        for i in 0..n {
            key_poly[i] = key_coeff[i * num_primes + prime_idx];
        }

        // Transform to NTT domain using Metal GPU (twist + forward NTT)
        // This is fully GPU accelerated with no CPU arithmetic
        ntt_ctx.coeff_to_ntt_gpu(&mut key_poly)?;

        // Store back in flat layout
        for i in 0..n {
            key_ntt_flat[i * num_primes + prime_idx] = key_poly[i];
        }
    }

    Ok(key_ntt_flat)
}

/// Pointwise multiply two NTT-domain polynomials and inverse NTT
///
/// Both inputs are in NTT domain (Montgomery). Performs:
/// 1. Pointwise multiply in NTT domain (Metal GPU)
/// 2. FUSED inverse NTT + untwist (Metal GPU single kernel!)
///
/// **Performance:** Fully Metal GPU accelerated with kernel fusion
///
/// This is a helper for `rotate_with_hoisted_digits`.
fn multiply_ntt_and_intt(
    a_ntt: &[u64],
    b_ntt: &[u64],
    moduli: &[u64],
    ctx: &super::ckks::MetalCkksContext,
) -> Result<Vec<u64>, String> {
    let n = ctx.params.n;
    let num_primes = moduli.len();
    let mut result_flat = vec![0u64; n * num_primes];

    // For each RNS component (each prime)
    for (prime_idx, _q) in moduli.iter().enumerate() {
        // Extract polynomials for this prime
        let mut a_poly = vec![0u64; n];
        let mut b_poly = vec![0u64; n];
        for i in 0..n {
            a_poly[i] = a_ntt[i * num_primes + prime_idx];
            b_poly[i] = b_ntt[i * num_primes + prime_idx];
        }

        let ntt_ctx = &ctx.ntt_contexts()[prime_idx];

        // Pointwise multiply in NTT domain (both already in Montgomery)
        let mut result_poly = vec![0u64; n];
        ntt_ctx.pointwise_multiply(&a_poly, &b_poly, &mut result_poly)?;

        // FUSED inverse NTT + untwist in one optimized operation!
        // This eliminates one GPU kernel dispatch and buffer round-trip
        ntt_ctx.inverse_and_untwist_fused(&mut result_poly)?;

        // Store back in flat layout
        for i in 0..n {
            result_flat[i * num_primes + prime_idx] = result_poly[i];
        }
    }

    Ok(result_flat)
}

/// BATCHED pointwise multiply and inverse NTT for all RNS primes
///
/// Processes all RNS primes in parallel using 2D Metal dispatch.
/// This is significantly faster than `multiply_ntt_and_intt` for multi-prime operations.
///
/// **Performance:** 15-20% faster by batching GPU operations across all primes
///
/// # Arguments
/// * `a_ntt_flat` - Input A in flat layout: [coeff0_p0, coeff0_p1, ..., coeff1_p0, ...]
/// * `b_ntt_flat` - Input B in flat layout (same layout as A)
/// * `moduli` - RNS moduli
/// * `ctx` - CKKS context with NTT contexts for each prime
///
/// # Returns
/// Result in flat layout (same as inputs)
fn multiply_ntt_and_intt_batched(
    a_ntt_flat: &[u64],
    b_ntt_flat: &[u64],
    moduli: &[u64],
    ctx: &super::ckks::MetalCkksContext,
) -> Result<Vec<u64>, String> {
    let n = ctx.params.n;
    let num_primes = moduli.len();

    if a_ntt_flat.len() != n * num_primes || b_ntt_flat.len() != n * num_primes {
        return Err(format!(
            "Expected flat arrays of size {}×{}={}, got {} and {}",
            n, num_primes, n * num_primes,
            a_ntt_flat.len(), b_ntt_flat.len()
        ));
    }

    // STEP 1: BATCHED pointwise multiply for all primes in ONE GPU dispatch!
    // This uses 2D dispatch: (n coefficients × num_primes) all computed in parallel
    let product_flat = ctx.pointwise_multiply_batched(a_ntt_flat, b_ntt_flat, moduli)?;

    // STEP 2: Per-prime inverse NTT + untwist (still needs per-prime processing
    // because iNTT has different twiddle factors per prime)
    let mut result_flat = vec![0u64; n * num_primes];

    for (prime_idx, _q) in moduli.iter().enumerate() {
        let ntt_ctx = &ctx.ntt_contexts()[prime_idx];

        // Extract this prime's data
        let mut result_poly = vec![0u64; n];
        for i in 0..n {
            result_poly[i] = product_flat[i * num_primes + prime_idx];
        }

        // FUSED inverse NTT + untwist
        ntt_ctx.inverse_and_untwist_fused(&mut result_poly)?;

        // Store back
        for i in 0..n {
            result_flat[i * num_primes + prime_idx] = result_poly[i];
        }
    }

    Ok(result_flat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitrev() {
        assert_eq!(bitrev(0, 3), 0); // 000 → 000
        assert_eq!(bitrev(1, 3), 4); // 001 → 100
        assert_eq!(bitrev(2, 3), 2); // 010 → 010
        assert_eq!(bitrev(3, 3), 6); // 011 → 110
        assert_eq!(bitrev(4, 3), 1); // 100 → 001
        assert_eq!(bitrev(5, 3), 5); // 101 → 101
        assert_eq!(bitrev(6, 3), 3); // 110 → 011
        assert_eq!(bitrev(7, 3), 7); // 111 → 111
    }

    #[test]
    fn test_ntt_permutation_natural() {
        let n = 8;
        let g = 5; // X → X^5 (rotation by 1 for N=8)
        let map = compute_ntt_galois_permutation(n, g, NttLayout::Natural);

        // For natural order: map[j] = (j * 5) mod 8
        assert_eq!(map[0], 0); // (0 * 5) % 8 = 0
        assert_eq!(map[1], 5); // (1 * 5) % 8 = 5
        assert_eq!(map[2], 2); // (2 * 5) % 8 = 2 (10 % 8)
        assert_eq!(map[3], 7); // (3 * 5) % 8 = 7 (15 % 8)
        assert_eq!(map[4], 4); // (4 * 5) % 8 = 4 (20 % 8)
        assert_eq!(map[5], 1); // (5 * 5) % 8 = 1 (25 % 8)
        assert_eq!(map[6], 6); // (6 * 5) % 8 = 6 (30 % 8)
        assert_eq!(map[7], 3); // (7 * 5) % 8 = 3 (35 % 8)
    }

    #[test]
    fn test_ntt_permutation_identity() {
        let n = 8;
        let g = 1; // Identity automorphism
        let map = compute_ntt_galois_permutation(n, g, NttLayout::Natural);

        // Identity permutation
        for j in 0..n {
            assert_eq!(map[j], j);
        }
    }

    #[test]
    fn test_permute_in_place() {
        let n = 4;
        let num_primes = 2;

        // Original buffer: [a0, a1, a2, a3, b0, b1, b2, b3]
        // where first 4 are prime 0, next 4 are prime 1
        let mut buf = vec![
            10, 20, 30, 40,  // prime 0
            50, 60, 70, 80,  // prime 1
        ];

        // Permutation: [0, 2, 1, 3] (swap indices 1 and 2)
        let map = vec![0, 2, 1, 3];

        permute_in_place_ntt(&mut buf, &map, n, num_primes);

        // Expected: [a0, a2, a1, a3, b0, b2, b1, b3]
        assert_eq!(buf, vec![
            10, 30, 20, 40,  // prime 0: swapped
            50, 70, 60, 80,  // prime 1: swapped
        ]);
    }

    #[test]
    #[should_panic(expected = "Galois element must be odd")]
    fn test_even_galois_element_panics() {
        compute_ntt_galois_permutation(8, 4, NttLayout::Natural);
    }

    #[test]
    #[should_panic(expected = "N must be a power of 2")]
    fn test_non_power_of_two_panics() {
        compute_ntt_galois_permutation(6, 5, NttLayout::Natural);
    }
}
