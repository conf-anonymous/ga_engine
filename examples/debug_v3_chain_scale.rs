//! Debug V3 chain scale tracking
//!
//! This test uses V3 parameters and traces the scale through a chain
//! to identify where the scale tracking diverges.
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 --example debug_v3_chain_scale
//! ```

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::{
    backends::{
        cpu_optimized::keys::KeyContext,
        gpu_cuda::{
            ckks::CudaCkksContext,
            device::CudaDeviceContext,
            inversion::scalar_division_gpu,
            relin_keys::CudaRelinKeys,
        },
    },
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
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

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!("\n=== DEBUG: V3 Chain Scale Tracking ===\n");

    let device = Arc::new(CudaDeviceContext::new()?);
    let params = CliffordFHEParams::new_v3_bootstrap_cuda_full()?;
    let num_primes = params.moduli.len();
    let max_level = num_primes - 1;
    let original_scale = params.scale;

    println!("V3 Parameters:");
    println!("  N = {}", params.n);
    println!("  Num primes = {}", num_primes);
    println!("  Max level = {}", max_level);
    println!("  Original scale = {:.2e} (2^{:.1})", original_scale, original_scale.log2());
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

    // Single division first
    println!("=== SINGLE DIVISION (100 / 7) ===\n");

    let num_val = 100.0;
    let denom_val = 7.0;
    let expected = num_val / denom_val;

    let pt_num = ctx.encode(&[num_val], original_scale, max_level)?;
    let pt_denom = ctx.encode(&[denom_val], original_scale, max_level)?;
    let ct_num = ctx.encrypt(&pt_num, &pk)?;
    let ct_denom = ctx.encrypt(&pt_denom, &pk)?;

    println!("Before division:");
    println!("  ct_num.scale   = {:.2e} (2^{:.1})", ct_num.scale, ct_num.scale.log2());
    println!("  ct_denom.scale = {:.2e} (2^{:.1})", ct_denom.scale, ct_denom.scale.log2());

    let ct_result = scalar_division_gpu(&ct_num, &ct_denom, 1.0/denom_val, 2, &relin_keys, &pk, &ctx)?;

    println!("\nAfter division:");
    println!("  ct_result.scale = {:.2e} (2^{:.1})", ct_result.scale, ct_result.scale.log2());
    println!("  ct_result.level = {}", ct_result.level);

    // Decrypt with tracked scale
    let pt_result = ctx.decrypt(&ct_result, &sk)?;
    println!("  pt_result.scale = {:.2e} (2^{:.1})", pt_result.scale, pt_result.scale.log2());

    let result_with_tracked = ctx.decode(&pt_result)?[0];
    println!("\nDecoded with tracked scale: {:.10} (expected {:.10})", result_with_tracked, expected);
    println!("Error: {:.2e}", (result_with_tracked - expected).abs() / expected);

    // Now manually decode with original scale
    println!("\n--- Manual decode with original scale ---");
    let pt_result_fixed = ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaPlaintext {
        poly: pt_result.poly.clone(),
        n: pt_result.n,
        num_primes: pt_result.num_primes,
        level: pt_result.level,
        scale: original_scale,  // Use original scale!
    };
    let result_with_original = ctx.decode(&pt_result_fixed)?[0];
    println!("Decoded with original scale: {:.10} (expected {:.10})", result_with_original, expected);
    println!("Error: {:.2e}", (result_with_original - expected).abs() / expected);

    // Chain division
    println!("\n\n=== CHAINED DIVISION (1000 / 2 / 5 = 100) ===\n");

    let start_val = 1000.0;
    let div1 = 2.0;
    let div2 = 5.0;
    let expected_chain = start_val / div1 / div2;  // 100.0

    let pt_start = ctx.encode(&[start_val], original_scale, max_level)?;
    let mut ct_current = ctx.encrypt(&pt_start, &pk)?;

    println!("Initial: scale = {:.2e} (2^{:.1}), level = {}",
             ct_current.scale, ct_current.scale.log2(), ct_current.level);

    // First division
    println!("\n--- First division (1000 / 2) ---");
    let pt_d1 = ctx.encode(&[div1], original_scale, ct_current.level)?;
    let ct_d1 = ctx.encrypt(&pt_d1, &pk)?;

    ct_current = scalar_division_gpu(&ct_current, &ct_d1, 1.0/div1, 1, &relin_keys, &pk, &ctx)?;

    println!("After div 1: scale = {:.2e} (2^{:.1}), level = {}",
             ct_current.scale, ct_current.scale.log2(), ct_current.level);

    // Decrypt intermediate
    let pt_inter = ctx.decrypt(&ct_current, &sk)?;
    let inter_with_tracked = ctx.decode(&pt_inter)?[0];
    println!("Intermediate (tracked scale): {:.6} (expected {})", inter_with_tracked, start_val / div1);

    let pt_inter_fixed = ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaPlaintext {
        poly: pt_inter.poly.clone(),
        n: pt_inter.n,
        num_primes: pt_inter.num_primes,
        level: pt_inter.level,
        scale: original_scale,
    };
    let inter_with_original = ctx.decode(&pt_inter_fixed)?[0];
    println!("Intermediate (original scale): {:.6} (expected {})", inter_with_original, start_val / div1);

    // Second division
    println!("\n--- Second division (500 / 5) ---");
    let pt_d2 = ctx.encode(&[div2], original_scale, ct_current.level)?;
    let ct_d2 = ctx.encrypt(&pt_d2, &pk)?;

    ct_current = scalar_division_gpu(&ct_current, &ct_d2, 1.0/div2, 1, &relin_keys, &pk, &ctx)?;

    println!("After div 2: scale = {:.2e} (2^{:.1}), level = {}",
             ct_current.scale, ct_current.scale.log2(), ct_current.level);

    // Decrypt final
    let pt_final = ctx.decrypt(&ct_current, &sk)?;
    println!("pt_final.scale = {:.2e} (2^{:.1})", pt_final.scale, pt_final.scale.log2());

    let final_with_tracked = ctx.decode(&pt_final)?[0];
    println!("\nFinal (tracked scale): {:.6} (expected {})", final_with_tracked, expected_chain);
    println!("Error: {:.2e}", (final_with_tracked - expected_chain).abs() / expected_chain);

    let pt_final_fixed = ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaPlaintext {
        poly: pt_final.poly.clone(),
        n: pt_final.n,
        num_primes: pt_final.num_primes,
        level: pt_final.level,
        scale: original_scale,
    };
    let final_with_original = ctx.decode(&pt_final_fixed)?[0];
    println!("Final (original scale): {:.6} (expected {})", final_with_original, expected_chain);
    println!("Error: {:.2e}", (final_with_original - expected_chain).abs() / expected_chain);

    println!("\n=== DIAGNOSIS ===\n");
    println!("Scale ratio (tracked vs original): {:.6}", ct_current.scale / original_scale);
    println!("Log2 difference: {:.1} bits", ct_current.scale.log2() - original_scale.log2());
    println!();

    if (final_with_original - expected_chain).abs() / expected_chain < 1e-6 {
        println!("✓ Using ORIGINAL scale gives correct result!");
        println!("  The scale TRACKING is broken, but the actual computation is correct.");
        println!();
        println!("FIX: Modify decode to use a fixed scale (params.scale) instead of pt.scale");
    } else {
        println!("✗ Even with original scale, result is wrong.");
        println!("  The actual computation has an issue, not just scale tracking.");
    }

    Ok(())
}

#[cfg(not(all(feature = "v3", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires: --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3");
}
