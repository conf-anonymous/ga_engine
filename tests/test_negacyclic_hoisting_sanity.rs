//! Negacyclic NTT Hoisting Sanity Check
//!
//! Verifies that NTT_neg(σ_g(a))[j] == D_g[j] · NTT_neg(a)[j·g mod N]
//! for negacyclic NTT (the actual CKKS case).

#![cfg(all(feature = "v2-gpu-metal", feature = "v2"))]

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::hoisting::{
    compute_ntt_permutation_for_step, permute_in_place_ntt, NttLayout
};
use ga_engine::clifford_fhe_v2::backends::gpu_metal::rotation::{
    compute_galois_map, rotation_step_to_galois_element
};

/// Apply Galois automorphism in coefficient domain (NEGACYCLIC version with signs)
fn apply_galois_coefficient_negacyclic(
    poly: &[u64],
    n: usize,
    g: usize,
    num_primes: usize,
    moduli: &[u64],
) -> Vec<u64> {
    let mut result = vec![0u64; n * num_primes];
    let two_n = 2 * n;

    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];

        for i in 0..n {
            // Negacyclic: X^i -> X^{gi mod 2N}
            let gi = (i * g) % two_n;

            let val = poly[i * num_primes + prime_idx];

            if gi < n {
                // Positive: X^i -> X^{gi}
                result[gi * num_primes + prime_idx] = val;
            } else {
                // Negative: X^i -> -X^{gi - N} (because X^N = -1)
                let target_idx = gi - n;
                // Negate: q - val (since we're in mod q)
                let negated_val = if val == 0 { 0 } else { q - val };
                result[target_idx * num_primes + prime_idx] = negated_val;
            }
        }
    }

    result
}

/// Montgomery multiplication helper (copied from hoisting.rs)
#[inline]
fn mont_mul(a: u64, b: u64, q: u64, ntt_ctx: &ga_engine::clifford_fhe_v2::backends::gpu_metal::ntt::MetalNttContext) -> u64 {
    let q_inv = ntt_ctx.q_inv();

    let t = a as u128 * b as u128;
    let t_lo = t as u64;
    let t_hi = (t >> 64) as u64;

    let m = t_lo.wrapping_mul(q_inv);

    let mq = m as u128 * q as u128;
    let mq_lo = mq as u64;
    let mq_hi = (mq >> 64) as u64;

    let (_, carry1) = t_lo.overflowing_add(mq_lo);
    let (sum_hi, carry2) = t_hi.overflowing_add(mq_hi);
    let sum_hi = sum_hi.wrapping_add(carry1 as u64).wrapping_add(carry2 as u64);

    if sum_hi >= q {
        sum_hi - q
    } else {
        sum_hi
    }
}

/// Helper to perform forward NEGACYCLIC NTT on flat RNS layout
/// This applies the ψ twist before forward NTT, giving NTT_neg
fn forward_ntt_flat(
    coeffs: &[u64],
    n: usize,
    moduli: &[u64],
    num_primes: usize,
    ctx: &MetalCkksContext,
) -> Result<Vec<u64>, String> {
    let mut result = vec![0u64; n * num_primes];

    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = &ctx.ntt_contexts()[prime_idx];

        // Extract this prime's coefficients
        let mut prime_coeffs = vec![0u64; n];
        for i in 0..n {
            prime_coeffs[i] = coeffs[i * num_primes + prime_idx];
        }

        // Apply ψ twist for negacyclic NTT: coeffs[i] *= ψ^i
        for i in 0..n {
            prime_coeffs[i] = ((prime_coeffs[i] as u128 * ntt_ctx.psi_powers()[i] as u128) % q as u128) as u64;
        }

        // Forward NTT for this prime (modifies in place, outputs Montgomery)
        ntt_ctx.forward(&mut prime_coeffs)?;

        // Insert back into flat layout
        for i in 0..n {
            result[i * num_primes + prime_idx] = prime_coeffs[i];
        }
    }

    Ok(result)
}

#[test]
fn test_negacyclic_ntt_galois_permutation() -> Result<(), String> {
    println!("\n════════════════════════════════════════════════════════");
    println!("Negacyclic NTT Galois Permutation Sanity Check");
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
        println!("  Galois element g = {}", g);

        // Create random test polynomial
        let mut poly_coeff = vec![0u64; n * num_primes];
        for slot in 0..n {
            for prime_idx in 0..num_primes {
                let i = slot * num_primes + prime_idx;
                // Use pseudo-random values based on slot index
                poly_coeff[i] = (slot as u64 * 123456789) % moduli[prime_idx];
            }
        }

        // Path 1: Apply Galois in coefficient domain, then twist by DESTINATION index, then cyclic NTT
        // KEY FIX: After σ_g permutation, we must twist by ψ^{i'} (destination index) not ψ^i!
        let mut poly_galois_twisted = vec![0u64; n * num_primes];

        for prime_idx in 0..num_primes {
            let q = moduli[prime_idx];
            let ntt_ctx = &ctx.ntt_contexts()[prime_idx];

            // Step 1: Apply σ_g in coefficient domain (with negacyclic signs)
            let mut b_coeff = vec![0u64; n];
            for i in 0..n {
                let gi = i * g;
                let i_prime = gi % n;                    // Destination index
                let sign = (gi / n) & 1;                 // 0 if gi < N, 1 if gi >= N

                let val = poly_coeff[i * num_primes + prime_idx];
                let with_sign = if sign == 0 {
                    val
                } else {
                    if val == 0 { 0 } else { q - val }   // Negate in mod q
                };

                b_coeff[i_prime] = with_sign;            // Write to DESTINATION index
            }

            // Debug: Print first few b_coeff values
            if prime_idx == 0 {
                println!("  Path 1 after Galois (b_coeff, prime=0, first 4): {:?}", &b_coeff[..4]);
                println!("  Path 1 original (a, prime=0, first 4): {:?}",
                    &[poly_coeff[0], poly_coeff[2], poly_coeff[4], poly_coeff[6]]);
            }

            // Step 2: Twist by array index (not source/destination, just position in array!)
            // After Galois, b_coeff is a permuted polynomial, and we twist it like any polynomial
            let mut twisted = vec![0u64; n];
            for idx in 0..n {
                let psi_pow = ntt_ctx.psi_powers()[idx];
                // Both b_coeff and psi_pow are in standard domain
                twisted[idx] = ((b_coeff[idx] as u128 * psi_pow as u128) % q as u128) as u64;
            }

            // Debug: Print first few twisted values before NTT
            if prime_idx == 0 {
                println!("  Path 1 twisted (before NTT, prime=0, first 4): {:?}", &twisted[..4]);
            }

            // Step 3: Forward cyclic NTT (outputs Montgomery domain)
            ntt_ctx.forward(&mut twisted)?;

            // Store in flat layout
            for i in 0..n {
                poly_galois_twisted[i * num_primes + prime_idx] = twisted[i];
            }

            // Debug: Print first few NTT values
            if prime_idx == 0 {
                let ntt_vals: Vec<u64> = (0..4).map(|j| poly_galois_twisted[j * num_primes + prime_idx]).collect();
                println!("  Path 1 NTT output (prime=0, first 4): {:?}", ntt_vals);
            }
        }

        let poly_galois_ntt = poly_galois_twisted;

        // Path 2: NTT first, then permute + apply diagonal in NTT domain
        let poly_ntt = forward_ntt_flat(&poly_coeff, n, moduli, num_primes, &ctx)?;

        // Compute permutation with offset (NO diagonal needed!)
        // The correct formula is: NTT_neg(σ_g a)[j] = NTT_neg(a)[(g·j + α) mod N]
        // where α = (g-1)/2, and this offset is baked into the permutation
        let ntt_perm = compute_ntt_permutation_for_step(n, step, NttLayout::Natural);

        // Debug: Check first few permutation values (now includes offset!)
        println!("  Permutation with offset (first 8): {:?}", &ntt_perm[..8]);

        // Apply permutation (offset is included in the map, no diagonal needed!)
        let mut poly_ntt_hoisted = poly_ntt.clone();
        permute_in_place_ntt(&mut poly_ntt_hoisted, &ntt_perm, n, num_primes);

        // Debug: Print first few values from both paths
        println!("\n  Path 1 (Galois in coeff → NTT) first 4 values:");
        for j in 0..4 {
            for prime_idx in 0..num_primes {
                let idx = j * num_primes + prime_idx;
                println!("    [{}][prime={}]: {}", j, prime_idx, poly_galois_ntt[idx]);
            }
        }

        println!("\n  Path 2 (NTT → permute → diagonal) first 4 values:");
        for j in 0..4 {
            for prime_idx in 0..num_primes {
                let idx = j * num_primes + prime_idx;
                println!("    [{}][prime={}]: {}", j, prime_idx, poly_ntt_hoisted[idx]);
            }
        }

        // Compare: NTT_neg(σ_g(a))[j] should equal D_g[j] · NTT_neg(a)[j·g mod N]
        let mut max_diff = 0u64;
        let mut num_diffs = 0;
        let mut first_diff_idx = None;

        for i in 0..(n * num_primes) {
            let val1 = poly_galois_ntt[i];
            let val2 = poly_ntt_hoisted[i];

            if val1 != val2 {
                let prime_idx = i % num_primes;
                let q = moduli[prime_idx];
                let diff = if val1 > val2 {
                    std::cmp::min(val1 - val2, q - (val1 - val2))
                } else {
                    std::cmp::min(val2 - val1, q - (val2 - val1))
                };

                if diff > max_diff {
                    max_diff = diff;
                }

                if num_diffs == 0 {
                    first_diff_idx = Some(i);
                }
                num_diffs += 1;
            }
        }

        if num_diffs == 0 {
            println!("  ✓ PASS: NTT_neg(σ_g(a))[j] == D_g[j] · NTT_neg(a)[j·g] exactly\n");
        } else {
            println!("  ✗ FAIL: Found {} differences, max diff = {}", num_diffs, max_diff);
            if let Some(idx) = first_diff_idx {
                let slot = idx / num_primes;
                let prime_idx = idx % num_primes;
                println!("  First diff at slot {}, prime {}: {} vs {}",
                    slot, prime_idx, poly_galois_ntt[idx], poly_ntt_hoisted[idx]);
            }
            println!("  This indicates the hoisting formula is incorrect!\n");
            panic!("Negacyclic hoisting sanity check failed for step {}", step);
        }
    }

    println!("════════════════════════════════════════════════════════");
    println!("✓ All negacyclic NTT hoisting checks passed!");
    println!("  - Negacyclic NTT: NTT_neg(σ_g(a))[j] == D_g[j] · NTT_neg(a)[j·g]");
    println!("  - Formula verified for all rotation steps");
    println!("  - Sign cancellation working correctly");
    println!("════════════════════════════════════════════════════════\n");

    Ok(())
}
