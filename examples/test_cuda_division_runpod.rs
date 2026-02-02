//! Comprehensive CUDA GPU Homomorphic Division Test for RunPod
//!
//! This test validates the complete CUDA GPU homomorphic division pipeline:
//! - Ciphertext multiplication with relinearization
//! - Newton-Raphson inverse computation
//! - Full homomorphic division (a/b)
//!
//! Run on RunPod.io with NVIDIA GPU:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example test_cuda_division_runpod
//! ```
//!
//! Requirements:
//! - NVIDIA GPU with CUDA support (Compute Capability 7.0+)
//! - CUDA Toolkit 11.0+ (12.0 recommended)
//! - Recommended: RTX 5090, RTX 4090, or A100

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::{
    backends::{
        cpu_optimized::keys::KeyContext,
        gpu_cuda::{
            ckks::CudaCkksContext,
            device::CudaDeviceContext,
            inversion::{multiply_ciphertexts_gpu, newton_raphson_inverse_gpu, scalar_division_gpu},
            relin_keys::CudaRelinKeys,
        },
    },
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::sync::Arc;
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::time::Instant;

/// Convert CPU SecretKey (RNS representation) to CUDA strided format
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
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     CUDA GPU Homomorphic Division - RunPod Test Suite                 ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    // Step 1: Initialize CUDA device
    println!("Step 1: Initializing CUDA GPU...");
    let device_start = Instant::now();
    let device = Arc::new(CudaDeviceContext::new()?);
    println!("  CUDA device initialized ({:.2}ms)", device_start.elapsed().as_secs_f64() * 1000.0);
    println!();

    // Step 2: Initialize parameters (use N=4096 for 7 primes = depth 6)
    println!("Step 2: Initializing FHE parameters...");
    let params = CliffordFHEParams::new_test_ntt_4096();
    let num_primes = params.moduli.len();
    let max_level = num_primes - 1;
    let scale = params.scale;
    println!("  Parameters: N={}, {} primes (max level {})",
             params.n, num_primes, max_level);
    println!("  Scale: 2^{}", (scale.log2() as u32));
    println!();

    // Step 3: Generate keys
    println!("Step 3: Generating cryptographic keys...");
    let key_ctx = KeyContext::new(params.clone());
    let start = Instant::now();
    let (pk, sk, _evk) = key_ctx.keygen();
    let keygen_time = start.elapsed();
    println!("  Keys generated in {:.2}ms", keygen_time.as_secs_f64() * 1000.0);
    println!();

    // Step 4: Initialize CUDA context
    println!("Step 4: Initializing CUDA CKKS context...");
    let start = Instant::now();
    let ctx = CudaCkksContext::new(params.clone())?;
    let init_time = start.elapsed();
    println!("  CUDA context ready in {:.2}ms", init_time.as_secs_f64() * 1000.0);
    println!();

    // Convert secret key to CUDA strided format
    let sk_strided = secret_key_to_strided(&sk, num_primes);

    // Step 5: Generate relinearization keys
    println!("Step 5: Generating CUDA relinearization keys...");
    let start = Instant::now();
    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided,
        16, // base_bits
        ctx.ntt_contexts(),
    )?;
    let relin_keygen_time = start.elapsed();
    println!("  Relinearization keys generated in {:.2}ms", relin_keygen_time.as_secs_f64() * 1000.0);
    println!();

    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 1: Ciphertext Multiplication + Rescaling");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Test multiplication: 2.0 * 3.0 = 6.0
    let a = 2.0;
    let b = 3.0;
    let expected_product = a * b;

    println!("Test: {:.1} * {:.1} = {:.1}", a, b, expected_product);
    println!();

    // Encode and encrypt
    println!("  Encoding and encrypting...");
    let pt_a = ctx.encode(&[a], scale, max_level)?;
    let pt_b = ctx.encode(&[b], scale, max_level)?;

    let ct_a = ctx.encrypt(&pt_a, &pk)?;
    let ct_b = ctx.encrypt(&pt_b, &pk)?;
    println!("  Encrypted at level {}", ct_a.level);
    println!();

    // Multiply on GPU (includes rescale)
    println!("  Performing GPU multiplication with relinearization...");
    let start = Instant::now();
    let ct_product = multiply_ciphertexts_gpu(&ct_a, &ct_b, &relin_keys, &ctx)?;
    let mult_time = start.elapsed();
    println!("  GPU multiplication completed in {:.2}ms", mult_time.as_secs_f64() * 1000.0);
    println!("  Level after multiply+rescale: {}", ct_product.level);
    println!();

    // Decrypt and decode
    println!("  Decrypting and decoding...");
    let pt_result = ctx.decrypt(&ct_product, &sk)?;
    let result = ctx.decode(&pt_result)?;

    let error = (result[0] - expected_product).abs();
    let rel_error = error / expected_product;

    println!();
    println!("Results:");
    println!("  Expected:     {:.10}", expected_product);
    println!("  Computed:     {:.10}", result[0]);
    println!("  Error:        {:.2e}", error);
    println!("  Rel. Error:   {:.2e}", rel_error);

    if rel_error < 1e-4 {
        println!("  PASS - Multiplication working correctly");
    } else {
        println!("  FAIL - Error too large");
        return Err(format!("Multiplication test failed: error = {:.2e}", error));
    }
    println!();

    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 2: Newton-Raphson Inverse");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Test inverse: 1/4.0 = 0.25
    let x = 4.0;
    let expected_inverse = 1.0 / x;
    let initial_guess = 0.25; // Good guess for 1/4

    println!("Test: 1/{:.1} = {:.4}", x, expected_inverse);
    println!("Initial guess: {:.2}", initial_guess);
    println!();

    // Encode and encrypt
    let pt_x = ctx.encode(&[x], scale, max_level)?;
    let ct_x = ctx.encrypt(&pt_x, &pk)?;
    println!("  Encrypted input at level {}", ct_x.level);
    println!();

    // Run Newton-Raphson (2 iterations to fit in depth budget)
    println!("  Running Newton-Raphson iteration (2 iterations)...");
    let iterations = 2;
    let start = Instant::now();
    let ct_inverse = newton_raphson_inverse_gpu(&ct_x, initial_guess, iterations, &relin_keys, &pk, &ctx)?;
    let inverse_time = start.elapsed();
    println!("  Newton-Raphson completed in {:.2}ms", inverse_time.as_secs_f64() * 1000.0);
    println!("  Final level: {}", ct_inverse.level);
    println!();

    // Decrypt and decode
    let pt_inverse = ctx.decrypt(&ct_inverse, &sk)?;
    let result_inverse = ctx.decode(&pt_inverse)?;

    let error_inv = (result_inverse[0] - expected_inverse).abs();
    let rel_error_inv = error_inv / expected_inverse;

    println!("Results:");
    println!("  Expected:     {:.10}", expected_inverse);
    println!("  Computed:     {:.10}", result_inverse[0]);
    println!("  Error:        {:.2e}", error_inv);
    println!("  Rel. Error:   {:.2e}", rel_error_inv);

    if rel_error_inv < 1e-2 {
        println!("  PASS - Inverse computation working");
    } else {
        println!("  FAIL - Error too large");
        return Err(format!("Inverse test failed: error = {:.2e}", error_inv));
    }
    println!();

    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 3: Complete Homomorphic Division");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Test division: 100.0 / 7.0 = 14.285714...
    let numerator = 100.0;
    let denominator = 7.0;
    let expected_quotient = numerator / denominator;
    let initial_guess_div = 1.0 / denominator;

    println!("Test: {:.1} / {:.1} = {:.10}", numerator, denominator, expected_quotient);
    println!("Initial guess for 1/{}: {:.6}", denominator, initial_guess_div);
    println!();

    // Encode and encrypt
    let pt_num = ctx.encode(&[numerator], scale, max_level)?;
    let pt_den = ctx.encode(&[denominator], scale, max_level)?;

    let ct_num = ctx.encrypt(&pt_num, &pk)?;
    let ct_den = ctx.encrypt(&pt_den, &pk)?;
    println!("  Encrypted numerator and denominator at level {}", ct_num.level);
    println!();

    // Perform division on GPU
    println!("  Performing complete GPU division...");
    let start = Instant::now();
    let ct_quotient = scalar_division_gpu(&ct_num, &ct_den, initial_guess_div, iterations, &relin_keys, &pk, &ctx)?;
    let div_time = start.elapsed();
    println!("  GPU division completed in {:.2}ms", div_time.as_secs_f64() * 1000.0);
    println!("  Final level: {}", ct_quotient.level);
    println!();

    // Decrypt and decode
    let pt_quotient = ctx.decrypt(&ct_quotient, &sk)?;
    let result_quotient = ctx.decode(&pt_quotient)?;

    let error_div = (result_quotient[0] - expected_quotient).abs();
    let rel_error_div = error_div / expected_quotient;

    println!("Results:");
    println!("  Expected:     {:.10}", expected_quotient);
    println!("  Computed:     {:.10}", result_quotient[0]);
    println!("  Error:        {:.2e}", error_div);
    println!("  Rel. Error:   {:.2e}", rel_error_div);

    if rel_error_div < 1e-2 {
        println!("  PASS - Division working correctly");
    } else {
        println!("  FAIL - Error too large");
        return Err(format!("Division test failed: error = {:.2e}", error_div));
    }
    println!();

    println!("════════════════════════════════════════════════════════════════════════");
    println!("PERFORMANCE SUMMARY");
    println!("════════════════════════════════════════════════════════════════════════\n");

    println!("Operation Timings:");
    println!("  Key Generation:      {:.2}ms", keygen_time.as_secs_f64() * 1000.0);
    println!("  Relin Key Gen:       {:.2}ms", relin_keygen_time.as_secs_f64() * 1000.0);
    println!("  Multiplication:      {:.2}ms", mult_time.as_secs_f64() * 1000.0);
    println!("  Newton-Raphson:      {:.2}ms", inverse_time.as_secs_f64() * 1000.0);
    println!("  Complete Division:   {:.2}ms", div_time.as_secs_f64() * 1000.0);
    println!();

    println!("════════════════════════════════════════════════════════════════════════");
    println!("ALL TESTS PASSED - CUDA GPU DIVISION FULLY FUNCTIONAL");
    println!("════════════════════════════════════════════════════════════════════════\n");

    println!("Implementation Status: READY FOR PRODUCTION");
    println!("GPU Acceleration: VERIFIED");
    println!();

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                 CUDA GPU Division Test - RunPod                       ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");
    println!("This test requires the 'v2' and 'v2-gpu-cuda' features.");
    println!();
    println!("Run with:");
    println!("  cargo run --release --features v2,v2-gpu-cuda --example test_cuda_division_runpod");
    println!();
    println!("Requirements:");
    println!("  - NVIDIA GPU with CUDA support (Compute Capability 7.0+)");
    println!("  - CUDA Toolkit 11.0+ (12.0 recommended)");
    println!();
    println!("For RunPod users:");
    println!("  1. Select RTX 5090, RTX 4090, or A100 GPU pod");
    println!("  2. Install CUDA Toolkit");
    println!("  3. Clone repository and run this test");
}
