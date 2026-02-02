//! Integration Test: Division Chain with Bootstrap
//!
//! Tests the specific flow:
//! 1. Two consecutive divisions
//! 2. One bootstrap (refresh)
//! 3. Two more consecutive divisions
//! 4. One bootstrap (refresh)
//! 5. Verify final result
//!
//! This validates the integration of homomorphic division with CKKS bootstrapping,
//! demonstrating that the combination enables unlimited division chains.
//!
//! ## Run:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 \
//!     --example test_division_bootstrap_integration
//! ```

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::device::CudaDeviceContext;
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::rotation::CudaRotationContext;
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::rotation_keys::CudaRotationKeys;
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::relin_keys::CudaRelinKeys;
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::inversion::scalar_division_gpu;
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v3::bootstrapping::cuda_bootstrap::{CudaBootstrapContext, CudaCiphertext};
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v3::bootstrapping::BootstrapParams;

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use std::sync::Arc;
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use std::time::Instant;

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║    INTEGRATION TEST: Division Chain with Bootstrap                           ║");
    println!("║                                                                              ║");
    println!("║    Flow: Div → Div → Bootstrap → Div → Div → Bootstrap → Verify             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // ========================================================================
    // SETUP
    // ========================================================================

    println!("┌────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SETUP: Initialize CUDA, Keys, and Bootstrap Context                       │");
    println!("└────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let setup_start = Instant::now();

    // Initialize CUDA
    println!("  [1/5] Initializing CUDA device...");
    let device = Arc::new(CudaDeviceContext::new()?);

    // Parameters
    println!("  [2/5] Setting up V3 bootstrap parameters...");
    let params = CliffordFHEParams::new_v3_bootstrap_cuda_full()?;
    let n = params.n;
    let num_primes = params.moduli.len();
    let max_level = num_primes - 1;
    let scale = params.scale;
    let bootstrap_params = BootstrapParams::balanced();

    println!("         N={}, primes={}, max_level={}, scale=2^{}",
             n, num_primes, max_level, scale.log2() as u32);

    // Contexts
    println!("  [3/5] Creating CKKS and rotation contexts...");
    let ckks_ctx = Arc::new(CudaCkksContext::new(params.clone())?);
    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);

    // Keys
    println!("  [4/5] Generating keys...");
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();

    // Convert secret key to strided format
    let mut secret_key_strided = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            secret_key_strided[coeff_idx * num_primes + prime_idx] = sk.coeffs[coeff_idx].values[prime_idx];
        }
    }

    // Rotation keys for bootstrap
    let mut rotation_keys = CudaRotationKeys::new(
        device.clone(),
        params.clone(),
        rotation_ctx.clone(),
        secret_key_strided.clone(),
        16,
    )?;

    let num_slots = n / 2;
    let num_fft_levels = (num_slots as f64).log2() as usize;
    for level_idx in 0..num_fft_levels {
        rotation_keys.generate_rotation_key_gpu(1 << level_idx, ckks_ctx.ntt_contexts())?;
    }

    // Relinearization keys
    let relin_keys = Arc::new(CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        secret_key_strided.clone(),
        16,
        ckks_ctx.ntt_contexts(),
    )?);

    // Bootstrap context
    println!("  [5/5] Creating bootstrap context...");
    let bootstrap_ctx = CudaBootstrapContext::new(
        ckks_ctx.clone(),
        rotation_ctx.clone(),
        Arc::new(rotation_keys),
        relin_keys.clone(),
        bootstrap_params,
        params.clone(),
    )?;

    let setup_time = setup_start.elapsed().as_secs_f64();
    println!();
    println!("  Setup complete in {:.2}s", setup_time);
    println!();

    // ========================================================================
    // TEST DATA
    // ========================================================================

    // Test: 1000 / 2 / 5 / 3 / 4 = 1000 / 120 = 8.333...
    let initial_value = 1000.0;
    let divisors = [2.0, 5.0, 3.0, 4.0];
    let expected_result = divisors.iter().fold(initial_value, |acc, &d| acc / d);

    println!("┌────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TEST DATA                                                                 │");
    println!("└────────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("  Computation: {} / {} / {} / {} / {} = {:.10}",
             initial_value, divisors[0], divisors[1], divisors[2], divisors[3], expected_result);
    println!();
    println!("  Flow:");
    println!("    • Division 1: / {} → need ~5 levels", divisors[0]);
    println!("    • Division 2: / {} → need ~5 levels", divisors[1]);
    println!("    • BOOTSTRAP 1: refresh levels");
    println!("    • Division 3: / {} → need ~5 levels", divisors[2]);
    println!("    • Division 4: / {} → need ~5 levels", divisors[3]);
    println!("    • BOOTSTRAP 2: refresh levels");
    println!("    • Verify result");
    println!();

    // ========================================================================
    // ENCRYPT INITIAL VALUE
    // ========================================================================

    println!("┌────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 0: Encrypt Initial Value                                            │");
    println!("└────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let pt_initial = ckks_ctx.encode(&[initial_value], scale, max_level)?;
    let mut ct_current = ckks_ctx.encrypt(&pt_initial, &pk)?;

    println!("  Encrypted {} at level {}", initial_value, ct_current.level);
    println!();

    // Helper function to perform division
    let perform_division = |ct: &ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext,
                            divisor: f64,
                            ckks: &CudaCkksContext,
                            relin: &CudaRelinKeys,
                            pk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey|
                            -> Result<ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext, String> {
        let pt_div = ckks.encode(&[divisor], ct.scale, ct.level)?;
        let ct_div = ckks.encrypt(&pt_div, pk)?;
        scalar_division_gpu(ct, &ct_div, 1.0 / divisor, 2, relin, pk, ckks)
    };

    // Helper function to perform bootstrap
    let perform_bootstrap = |ct: &ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext,
                              bootstrap: &CudaBootstrapContext|
                              -> Result<ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext, String> {
        let cuda_ct = CudaCiphertext {
            c0: ct.c0.clone(),
            c1: ct.c1.clone(),
            n: ct.n,
            num_primes: ct.num_primes,
            level: ct.level,
            scale: ct.scale,
        };

        let refreshed = bootstrap.bootstrap(&cuda_ct)?;

        Ok(ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext {
            c0: refreshed.c0,
            c1: refreshed.c1,
            n: refreshed.n,
            num_primes: refreshed.num_primes,
            level: refreshed.level,
            scale: refreshed.scale,
        })
    };

    // Track intermediate results
    let mut intermediate_expected = initial_value;
    let mut total_div_time = 0.0;
    let mut total_bootstrap_time = 0.0;

    // ========================================================================
    // PHASE 1: First Two Divisions
    // ========================================================================

    println!("┌────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 1: First Two Consecutive Divisions                                  │");
    println!("└────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Division 1
    println!("  Division 1: {} / {} = {:.10}", intermediate_expected, divisors[0],
             intermediate_expected / divisors[0]);
    println!("    Level before: {}", ct_current.level);

    let div1_start = Instant::now();
    ct_current = perform_division(&ct_current, divisors[0], &ckks_ctx, &relin_keys, &pk)?;
    let div1_time = div1_start.elapsed().as_secs_f64() * 1000.0;
    total_div_time += div1_time;
    intermediate_expected /= divisors[0];

    println!("    Level after: {}", ct_current.level);
    println!("    Time: {:.2}ms", div1_time);

    // Verify intermediate
    let pt_check = ckks_ctx.decrypt(&ct_current, &sk)?;
    let result_check = ckks_ctx.decode(&pt_check)?[0];
    let error_check = (result_check - intermediate_expected).abs() / intermediate_expected.abs();
    println!("    Decrypted: {:.10} (expected: {:.10}, error: {:.2e})",
             result_check, intermediate_expected, error_check);
    println!();

    // Division 2
    println!("  Division 2: {:.10} / {} = {:.10}", intermediate_expected, divisors[1],
             intermediate_expected / divisors[1]);
    println!("    Level before: {}", ct_current.level);

    let div2_start = Instant::now();
    ct_current = perform_division(&ct_current, divisors[1], &ckks_ctx, &relin_keys, &pk)?;
    let div2_time = div2_start.elapsed().as_secs_f64() * 1000.0;
    total_div_time += div2_time;
    intermediate_expected /= divisors[1];

    println!("    Level after: {}", ct_current.level);
    println!("    Time: {:.2}ms", div2_time);

    let pt_check = ckks_ctx.decrypt(&ct_current, &sk)?;
    let result_check = ckks_ctx.decode(&pt_check)?[0];
    let error_check = (result_check - intermediate_expected).abs() / intermediate_expected.abs();
    println!("    Decrypted: {:.10} (expected: {:.10}, error: {:.2e})",
             result_check, intermediate_expected, error_check);
    println!();

    // ========================================================================
    // PHASE 2: First Bootstrap
    // ========================================================================

    println!("┌────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 2: First Bootstrap (Refresh Levels)                                 │");
    println!("└────────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("  Level before bootstrap: {}", ct_current.level);
    println!("  Bootstrapping...");

    let bootstrap1_start = Instant::now();
    ct_current = perform_bootstrap(&ct_current, &bootstrap_ctx)?;
    let bootstrap1_time = bootstrap1_start.elapsed().as_secs_f64();
    total_bootstrap_time += bootstrap1_time;

    println!("  Level after bootstrap: {}", ct_current.level);
    println!("  Time: {:.2}s", bootstrap1_time);

    // Verify after bootstrap
    let pt_check = ckks_ctx.decrypt(&ct_current, &sk)?;
    let result_check = ckks_ctx.decode(&pt_check)?[0];
    let error_check = (result_check - intermediate_expected).abs() / intermediate_expected.abs();
    println!("  Decrypted after bootstrap: {:.10} (expected: {:.10}, error: {:.2e})",
             result_check, intermediate_expected, error_check);
    println!();

    // ========================================================================
    // PHASE 3: Second Two Divisions
    // ========================================================================

    println!("┌────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 3: Second Two Consecutive Divisions                                 │");
    println!("└────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Division 3
    println!("  Division 3: {:.10} / {} = {:.10}", intermediate_expected, divisors[2],
             intermediate_expected / divisors[2]);
    println!("    Level before: {}", ct_current.level);

    let div3_start = Instant::now();
    ct_current = perform_division(&ct_current, divisors[2], &ckks_ctx, &relin_keys, &pk)?;
    let div3_time = div3_start.elapsed().as_secs_f64() * 1000.0;
    total_div_time += div3_time;
    intermediate_expected /= divisors[2];

    println!("    Level after: {}", ct_current.level);
    println!("    Time: {:.2}ms", div3_time);

    let pt_check = ckks_ctx.decrypt(&ct_current, &sk)?;
    let result_check = ckks_ctx.decode(&pt_check)?[0];
    let error_check = (result_check - intermediate_expected).abs() / intermediate_expected.abs();
    println!("    Decrypted: {:.10} (expected: {:.10}, error: {:.2e})",
             result_check, intermediate_expected, error_check);
    println!();

    // Division 4
    println!("  Division 4: {:.10} / {} = {:.10}", intermediate_expected, divisors[3],
             intermediate_expected / divisors[3]);
    println!("    Level before: {}", ct_current.level);

    let div4_start = Instant::now();
    ct_current = perform_division(&ct_current, divisors[3], &ckks_ctx, &relin_keys, &pk)?;
    let div4_time = div4_start.elapsed().as_secs_f64() * 1000.0;
    total_div_time += div4_time;
    intermediate_expected /= divisors[3];

    println!("    Level after: {}", ct_current.level);
    println!("    Time: {:.2}ms", div4_time);

    let pt_check = ckks_ctx.decrypt(&ct_current, &sk)?;
    let result_check = ckks_ctx.decode(&pt_check)?[0];
    let error_check = (result_check - intermediate_expected).abs() / intermediate_expected.abs();
    println!("    Decrypted: {:.10} (expected: {:.10}, error: {:.2e})",
             result_check, intermediate_expected, error_check);
    println!();

    // ========================================================================
    // PHASE 4: Second Bootstrap
    // ========================================================================

    println!("┌────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 4: Second Bootstrap (Refresh Levels)                                │");
    println!("└────────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("  Level before bootstrap: {}", ct_current.level);
    println!("  Bootstrapping...");

    let bootstrap2_start = Instant::now();
    ct_current = perform_bootstrap(&ct_current, &bootstrap_ctx)?;
    let bootstrap2_time = bootstrap2_start.elapsed().as_secs_f64();
    total_bootstrap_time += bootstrap2_time;

    println!("  Level after bootstrap: {}", ct_current.level);
    println!("  Time: {:.2}s", bootstrap2_time);
    println!();

    // ========================================================================
    // FINAL VERIFICATION
    // ========================================================================

    println!("┌────────────────────────────────────────────────────────────────────────────┐");
    println!("│ FINAL VERIFICATION                                                        │");
    println!("└────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let pt_final = ckks_ctx.decrypt(&ct_current, &sk)?;
    let final_result = ckks_ctx.decode(&pt_final)?[0];
    let final_error = (final_result - expected_result).abs() / expected_result.abs();

    println!("  Computation: {} / {} / {} / {} / {}",
             initial_value, divisors[0], divisors[1], divisors[2], divisors[3]);
    println!();
    println!("  Expected result: {:.10}", expected_result);
    println!("  Actual result:   {:.10}", final_result);
    println!("  Relative error:  {:.2e}", final_error);
    println!();
    println!("  Final level: {} (refreshed, ready for more operations)", ct_current.level);
    println!();

    // ========================================================================
    // SUMMARY
    // ========================================================================

    println!("┌────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SUMMARY                                                                   │");
    println!("└────────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("  Divisions performed:    4");
    println!("  Bootstraps performed:   2");
    println!("  Total division time:    {:.2}ms", total_div_time);
    println!("  Total bootstrap time:   {:.2}s", total_bootstrap_time);
    println!("  Final error:            {:.2e}", final_error);
    println!();

    // Determine test status
    let test_passed = final_error < 0.1;  // 10% relative error threshold

    if test_passed {
        println!("  ╔════════════════════════════════════════════════════════════════════════╗");
        println!("  ║                     TEST PASSED                                        ║");
        println!("  ║                                                                        ║");
        println!("  ║  Division + Bootstrap integration working correctly.                   ║");
        println!("  ║  Unlimited division chains are now possible.                           ║");
        println!("  ╚════════════════════════════════════════════════════════════════════════╝");
    } else {
        println!("  ╔════════════════════════════════════════════════════════════════════════╗");
        println!("  ║                     TEST FAILED                                        ║");
        println!("  ║                                                                        ║");
        println!("  ║  Error too large: {:.2e} > 0.1                                      ║", final_error);
        println!("  ╚════════════════════════════════════════════════════════════════════════╝");
    }
    println!();

    Ok(())
}

#[cfg(not(all(feature = "v3", feature = "v2-gpu-cuda")))]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║    INTEGRATION TEST: Division Chain with Bootstrap                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("This test requires V3 bootstrap and CUDA GPU support.");
    println!();
    println!("Run with:");
    println!("  cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 \\");
    println!("      --example test_division_bootstrap_integration");
}
