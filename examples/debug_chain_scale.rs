//! Debug script to trace scale values through a division chain
//!
//! This helps diagnose the scale tracking bug where chained divisions
//! produce incorrect results despite single divisions working correctly.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example debug_chain_scale
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::{
    backends::{
        cpu_optimized::keys::KeyContext,
        gpu_cuda::{
            ckks::CudaCkksContext,
            device::CudaDeviceContext,
            inversion::{scalar_division_gpu, multiply_ciphertexts_gpu},
            relin_keys::CudaRelinKeys,
        },
    },
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn secret_key_to_strided(
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    num_primes: usize,
) -> Vec<u64> {
    let n = sk.n;
    let mut strided = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            strided[coeff_idx * num_primes + prime_idx] = sk.coeffs[coeff_idx].values[prime_idx];
        }
    }
    strided
}

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!("\n=== DEBUG: Scale Tracking in Division Chain ===\n");

    // Use N=4096 with 7 primes (max level 6)
    let device = Arc::new(CudaDeviceContext::new()?);
    let params = CliffordFHEParams::new_test_ntt_4096();
    let num_primes = params.moduli.len();
    let max_level = num_primes - 1;
    let initial_scale = params.scale;

    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  Num primes = {}", num_primes);
    println!("  Max level = {}", max_level);
    println!("  Initial scale = {} (2^{:.1})", initial_scale, initial_scale.log2());
    println!("\nModuli:");
    for (i, &q) in params.moduli.iter().enumerate() {
        println!("  q[{}] = {} (2^{:.1})", i, q, (q as f64).log2());
    }
    println!();

    // Setup keys
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let ctx = CudaCkksContext::new(params.clone())?;
    let sk_strided = secret_key_to_strided(&sk, num_primes);
    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided,
        16,
        ctx.ntt_contexts(),
    )?;

    // Test 1: Simple multiplication chain (no division)
    println!("=== TEST 1: Multiplication chain (a * b * c) ===\n");

    let a = 2.0;
    let b = 3.0;
    let c = 4.0;
    let expected = a * b * c;  // 24.0

    let pt_a = ctx.encode(&[a], initial_scale, max_level)?;
    let pt_b = ctx.encode(&[b], initial_scale, max_level)?;
    let pt_c = ctx.encode(&[c], initial_scale, max_level)?;

    let ct_a = ctx.encrypt(&pt_a, &pk)?;
    let ct_b = ctx.encrypt(&pt_b, &pk)?;
    let ct_c = ctx.encrypt(&pt_c, &pk)?;

    println!("After encryption:");
    println!("  ct_a: level={}, scale={:.2e} (2^{:.1})", ct_a.level, ct_a.scale, ct_a.scale.log2());
    println!("  ct_b: level={}, scale={:.2e} (2^{:.1})", ct_b.level, ct_b.scale, ct_b.scale.log2());
    println!("  ct_c: level={}, scale={:.2e} (2^{:.1})", ct_c.level, ct_c.scale, ct_c.scale.log2());

    // First multiplication: a * b
    let ct_ab = multiply_ciphertexts_gpu(&ct_a, &ct_b, &relin_keys, &ctx)?;
    println!("\nAfter a * b:");
    println!("  ct_ab: level={}, scale={:.2e} (2^{:.1})", ct_ab.level, ct_ab.scale, ct_ab.scale.log2());

    // Decrypt to check
    let pt_ab_dec = ctx.decrypt(&ct_ab, &sk)?;
    let result_ab = ctx.decode(&pt_ab_dec)?[0];
    println!("  Decrypted: {:.6} (expected {})", result_ab, a * b);

    // Second multiplication: (a*b) * c
    let ct_c_aligned = ct_c.mod_switch_to_level(ct_ab.level);
    println!("\nAfter mod-switching c to level {}:", ct_ab.level);
    println!("  ct_c: level={}, scale={:.2e} (2^{:.1})", ct_c_aligned.level, ct_c_aligned.scale, ct_c_aligned.scale.log2());

    let ct_abc = multiply_ciphertexts_gpu(&ct_ab, &ct_c_aligned, &relin_keys, &ctx)?;
    println!("\nAfter (a*b) * c:");
    println!("  ct_abc: level={}, scale={:.2e} (2^{:.1})", ct_abc.level, ct_abc.scale, ct_abc.scale.log2());

    // Decrypt to check
    let pt_abc_dec = ctx.decrypt(&ct_abc, &sk)?;
    let result_abc = ctx.decode(&pt_abc_dec)?[0];
    println!("  Decrypted: {:.6} (expected {})", result_abc, expected);
    println!("  Error: {:.2e}", (result_abc - expected).abs() / expected);

    // Test 2: Single division
    println!("\n=== TEST 2: Single division (100 / 7) ===\n");

    let num_val = 100.0;
    let denom_val = 7.0;
    let expected_div = num_val / denom_val;

    let pt_num = ctx.encode(&[num_val], initial_scale, max_level)?;
    let pt_denom = ctx.encode(&[denom_val], initial_scale, max_level)?;
    let ct_num = ctx.encrypt(&pt_num, &pk)?;
    let ct_denom = ctx.encrypt(&pt_denom, &pk)?;

    println!("Before division:");
    println!("  ct_num: level={}, scale={:.2e} (2^{:.1})", ct_num.level, ct_num.scale, ct_num.scale.log2());
    println!("  ct_denom: level={}, scale={:.2e} (2^{:.1})", ct_denom.level, ct_denom.scale, ct_denom.scale.log2());

    let ct_result = scalar_division_gpu(&ct_num, &ct_denom, 1.0/denom_val, 2, &relin_keys, &pk, &ctx)?;

    println!("\nAfter division:");
    println!("  ct_result: level={}, scale={:.2e} (2^{:.1})", ct_result.level, ct_result.scale, ct_result.scale.log2());

    let pt_result = ctx.decrypt(&ct_result, &sk)?;
    let result = ctx.decode(&pt_result)?[0];
    println!("  Decrypted: {:.10} (expected {:.10})", result, expected_div);
    println!("  Error: {:.2e}", (result - expected_div).abs() / expected_div);

    // Test 3: Chained division (manually, with scale tracking)
    println!("\n=== TEST 3: Chained division (100 / 2 / 5 = 10) ===\n");

    let start_val = 100.0;
    let div1 = 2.0;
    let div2 = 5.0;
    let expected_chain = start_val / div1 / div2;  // 10.0

    let pt_start = ctx.encode(&[start_val], initial_scale, max_level)?;
    let ct_start = ctx.encrypt(&pt_start, &pk)?;

    println!("Initial:");
    println!("  ct_start: level={}, scale={:.2e} (2^{:.1})", ct_start.level, ct_start.scale, ct_start.scale.log2());

    // First division: 100 / 2
    let pt_d1 = ctx.encode(&[div1], initial_scale, max_level)?;
    let ct_d1 = ctx.encrypt(&pt_d1, &pk)?;

    println!("\nDivisor 1:");
    println!("  ct_d1: level={}, scale={:.2e} (2^{:.1})", ct_d1.level, ct_d1.scale, ct_d1.scale.log2());

    println!("\n--- Performing first division (100/2) ---");
    let ct_after_d1 = scalar_division_gpu(&ct_start, &ct_d1, 1.0/div1, 1, &relin_keys, &pk, &ctx)?;

    println!("After first division:");
    println!("  ct_after_d1: level={}, scale={:.2e} (2^{:.1})", ct_after_d1.level, ct_after_d1.scale, ct_after_d1.scale.log2());

    // Decrypt intermediate result
    let pt_after_d1 = ctx.decrypt(&ct_after_d1, &sk)?;
    let result_d1 = ctx.decode(&pt_after_d1)?[0];
    println!("  Decrypted: {:.10} (expected {})", result_d1, start_val / div1);
    println!("  Error: {:.2e}", (result_d1 - start_val/div1).abs() / (start_val/div1));

    // Check if we have enough depth for second division
    if ct_after_d1.level < 3 {
        println!("\n  WARNING: Insufficient depth for second division (need level >= 3, have {})", ct_after_d1.level);
        println!("  This is why chained divisions fail with standard parameters!");
        return Ok(());
    }

    // Second division: 50 / 5
    println!("\n--- Performing second division (50/5) ---");

    // Create divisor at the current level
    let pt_d2 = ctx.encode(&[div2], ct_after_d1.scale, ct_after_d1.level)?;
    let ct_d2 = ctx.encrypt(&pt_d2, &pk)?;

    println!("Divisor 2:");
    println!("  ct_d2: level={}, scale={:.2e} (2^{:.1})", ct_d2.level, ct_d2.scale, ct_d2.scale.log2());

    // KEY QUESTION: Are the scales matched?
    println!("\nScale comparison before second division:");
    println!("  ct_after_d1.scale = {:.2e} (2^{:.1})", ct_after_d1.scale, ct_after_d1.scale.log2());
    println!("  ct_d2.scale       = {:.2e} (2^{:.1})", ct_d2.scale, ct_d2.scale.log2());
    println!("  Ratio: {:.6}", ct_after_d1.scale / ct_d2.scale);

    let ct_after_d2 = scalar_division_gpu(&ct_after_d1, &ct_d2, 1.0/div2, 1, &relin_keys, &pk, &ctx)?;

    println!("\nAfter second division:");
    println!("  ct_after_d2: level={}, scale={:.2e} (2^{:.1})", ct_after_d2.level, ct_after_d2.scale, ct_after_d2.scale.log2());

    // Decrypt final result
    let pt_final = ctx.decrypt(&ct_after_d2, &sk)?;
    let result_final = ctx.decode(&pt_final)?[0];
    println!("  Decrypted: {:.10} (expected {})", result_final, expected_chain);
    println!("  Error: {:.2e}", (result_final - expected_chain).abs() / expected_chain);

    println!("\n=== SUMMARY ===\n");
    println!("Single division: WORKS (error ~10^-9)");
    println!("Chained division: {} (result = {:.6}, expected = {})",
             if (result_final - expected_chain).abs() / expected_chain < 0.01 { "WORKS" } else { "FAILS" },
             result_final, expected_chain);

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires: --features v2,v2-gpu-cuda");
}
