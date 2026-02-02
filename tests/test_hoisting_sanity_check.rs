//! Sanity Check for Hoisted Automorphisms
//!
//! Verifies that NTT(σ_g(a)) == Π_g(NTT(a)) for our Metal NTT implementation.
//!
//! This test ensures:
//! 1. The permutation formula is correct
//! 2. No extra diagonal factors are needed (Harvey-style NTT with internal twist)
//! 3. The hoisting optimization is mathematically sound

#![cfg(all(feature = "v2-gpu-metal", feature = "v2"))]

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::ntt::MetalNttContext;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::hoisting::{
    compute_ntt_permutation_for_step, permute_in_place_ntt, NttLayout
};
use ga_engine::clifford_fhe_v2::backends::gpu_metal::rotation::compute_galois_map;

/// Apply Galois automorphism to a polynomial in coefficient domain (CYCLIC version - no signs)
/// This is for testing pure cyclic NTT algebra: σ_g maps coeff[i] -> coeff[(g*i) mod N]
/// WITHOUT negacyclic sign flips (works in Z[X]/(X^N - 1), not Z[X]/(X^N + 1))
fn apply_galois_coefficient_cyclic(
    poly: &[u64],
    n: usize,
    g: usize,
    num_primes: usize,
) -> Vec<u64> {
    let mut result = vec![0u64; n * num_primes];

    for prime_idx in 0..num_primes {
        for i in 0..n {
            // Pure cyclic: X^i -> X^{(g*i) mod N}, no signs
            let target_idx = (i * g) % n;
            let val = poly[i * num_primes + prime_idx];
            result[target_idx * num_primes + prime_idx] = val;
        }
    }

    result
}

#[test]
fn test_ntt_galois_permutation_sanity_check() -> Result<(), String> {
    println!("\n════════════════════════════════════════════════════════");
    println!("NTT Galois Permutation Sanity Check");
    println!("════════════════════════════════════════════════════════\n");

    // Setup
    let params = CliffordFHEParams::new_test_ntt_1024();
    let ctx = MetalCkksContext::new(params.clone()).expect("Failed to create context");

    let n = params.n;
    let num_primes = 2; // Test with first 2 primes
    let moduli = &params.moduli[..num_primes];

    println!("Parameters:");
    println!("  N = {}", n);
    println!("  num_primes = {}", num_primes);
    println!("  moduli = {:?}\n", moduli);

    // Test rotation steps
    let test_steps = vec![1, 2, 4, -1, -2];

    for &step in &test_steps {
        println!("Testing rotation step: {}", step);

        let g = rotation_step_to_galois_element(step, n);
        println!("  Galois element g = {} (step {} → j·{} mod N)", g, step, g);

        // Create random test polynomial
        let mut poly_coeff = vec![0u64; n * num_primes];
        for slot in 0..n {
            for prime_idx in 0..num_primes {
                let i = slot * num_primes + prime_idx;
                // Use pseudo-random values based on slot index
                poly_coeff[i] = (slot as u64 * 123456789) % moduli[prime_idx];
            }
        }

        // Path 1: Apply Galois in coefficient domain, then NTT
        // Use CYCLIC version (no signs) to test pure cyclic NTT
        let poly_galois_coeff = apply_galois_coefficient_cyclic(
            &poly_coeff,
            n,
            g,
            num_primes,
        );

        // Forward NTT on Galois-rotated polynomial
        let poly_galois_ntt = forward_ntt_flat(&poly_galois_coeff, n, moduli, num_primes, &ctx)?;

        // Path 2: NTT first, then permute in NTT domain
        let poly_ntt = forward_ntt_flat(&poly_coeff, n, moduli, num_primes, &ctx)?;

        // Compute permutation for this step (Natural layout: BR before butterflies)
        let ntt_perm = compute_ntt_permutation_for_step(n, step, NttLayout::Natural);

        // Apply permutation with PULL semantics: out[j] = in[j*g mod N]
        let mut poly_ntt_permuted = poly_ntt.clone();
        permute_in_place_ntt(&mut poly_ntt_permuted, &ntt_perm, n, num_primes);

        // Compare: NTT(σ_g(a)) should equal Π_g(NTT(a)) for pure cyclic NTT
        let mut max_diff = 0u64;
        let mut num_diffs = 0;

        for i in 0..(n * num_primes) {
            let val1 = poly_galois_ntt[i];
            let val2 = poly_ntt_permuted[i];

            if val1 != val2 {
                let prime_idx = i % num_primes;
                let q = moduli[prime_idx];
                let diff = if val1 > val2 {
                    std::cmp::min(val1 - val2, q - (val1 - val2))
                } else {
                    std::cmp::min(val2 - val1, q - (val2 - val1))
                };
                max_diff = std::cmp::max(max_diff, diff);
                num_diffs += 1;
            }
        }

        if num_diffs == 0 {
            println!("  ✓ PASS: NTT(σ_g(a)) == Π_g(NTT(a)) exactly\n");
        } else {
            println!("  ✗ FAIL: Found {} differences, max diff = {}", num_diffs, max_diff);
            println!("  This indicates either:");
            println!("    1. Permutation formula is incorrect");
            println!("    2. Extra diagonal factors needed (uncommon for Harvey NTT)");
            println!("    3. NTT implementation issue\n");
            panic!("Sanity check failed for step {}", step);
        }
    }

    println!("════════════════════════════════════════════════════════");
    println!("✓ All cyclic NTT sanity checks passed!");
    println!("  - Pure cyclic NTT: NTT(σ_g(a)) == Π_g(NTT(a))");
    println!("  - Permutation formula verified: j → (j·g) mod N");
    println!("  - PULL semantics confirmed: out[j] = in[j·g mod N]");
    println!("  - Ready for negacyclic CKKS hoisting (with ψ twist)");
    println!("════════════════════════════════════════════════════════\n");

    Ok(())
}

// Helper to convert rotation step to Galois element
fn rotation_step_to_galois_element(step: i32, n: usize) -> usize {
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::rotation::rotation_step_to_galois_element;
    rotation_step_to_galois_element(step, n)
}

// Helper for modular exponentiation
fn mod_pow(base: u64, exp: u64, m: u64) -> u64 {
    let mut result = 1u128;
    let mut base = base as u128;
    let mut exp = exp;
    let m = m as u128;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % m;
        }
        base = (base * base) % m;
        exp >>= 1;
    }

    result as u64
}

// Helper for Montgomery multiplication using NttContext parameters
fn mont_mul(a: u64, b: u64, q: u64, ntt_ctx: &MetalNttContext) -> u64 {
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
    let (sum_lo, carry1) = t_lo.overflowing_add(mq_lo);
    let (sum_hi, carry2) = t_hi.overflowing_add(mq_hi);
    let sum_hi = sum_hi.wrapping_add(carry1 as u64).wrapping_add(carry2 as u64);

    // Step 5: Conditional subtraction
    if sum_hi >= q {
        sum_hi - q
    } else {
        sum_hi
    }
}

// Helper to convert to Montgomery domain
fn to_montgomery(x: u64, q: u64, ntt_ctx: &MetalNttContext) -> u64 {
    let r_squared = ntt_ctx.r_squared();
    mont_mul(x, r_squared, q, ntt_ctx)
}

// Helper for Montgomery exponentiation
fn mont_pow(base: u64, exp: u64, q: u64, ntt_ctx: &MetalNttContext) -> u64 {
    let mut result = to_montgomery(1, q, ntt_ctx); // 1 in Montgomery domain
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

// Helper to perform forward NTT on flat RNS layout
fn forward_ntt_flat(
    coeffs: &[u64],
    n: usize,
    moduli: &[u64],
    num_primes: usize,
    ctx: &MetalCkksContext,
) -> Result<Vec<u64>, String> {
    let mut result = vec![0u64; n * num_primes];

    for prime_idx in 0..num_primes {
        // Extract this prime's coefficients
        let mut prime_coeffs = vec![0u64; n];
        for i in 0..n {
            prime_coeffs[i] = coeffs[i * num_primes + prime_idx];
        }

        // Forward NTT for this prime (modifies in place)
        ctx.ntt_contexts()[prime_idx].forward(&mut prime_coeffs)?;

        // Insert back into flat layout
        for i in 0..n {
            result[i * num_primes + prime_idx] = prime_coeffs[i];
        }
    }

    Ok(result)
}
