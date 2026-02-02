//! Homomorphic Division with Bootstrapping for Unlimited Depth
//!
//! This benchmark demonstrates how to chain multiple homomorphic divisions
//! using V3 bootstrapping to refresh the noise budget, enabling unlimited
//! multiplicative depth.
//!
//! ## Scenarios:
//! 1. Multiple divisions without bootstrap (limited by depth)
//! 2. Multiple divisions WITH bootstrap (unlimited depth)
//! 3. Complex computation chains with division
//!
//! ## Key Insight
//!
//! Without bootstrapping, each Newton-Raphson division consumes 2k+1 levels
//! (for k iterations). With typical parameters (~9 levels), this limits us to
//! 1-2 divisions before exhausting the noise budget.
//!
//! With bootstrapping, we can refresh the ciphertext to full depth, enabling
//! arbitrary chains of divisions.
//!
//! ## Run:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 \
//!     --example bench_division_with_bootstrap
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
use ga_engine::clifford_fhe_v3::bootstrapping::cuda_bootstrap::CudaBootstrapContext;
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v3::bootstrapping::BootstrapParams;

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use std::sync::Arc;
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use std::time::Instant;

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║          HOMOMORPHIC DIVISION WITH BOOTSTRAPPING                                             ║");
    println!("║                    Unlimited Depth via Noise Refresh                                         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // ========================================================================
    // SECTION 1: INITIALIZATION
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 1: INITIALIZATION                                                                   │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("  [1/7] Initializing CUDA device...");
    let device = Arc::new(CudaDeviceContext::new()?);

    println!("  [2/7] Setting up V3 bootstrap parameters...");
    let params = CliffordFHEParams::new_v3_bootstrap_cuda_full()?;
    let n = params.n;
    let num_primes = params.moduli.len();
    let max_level = num_primes - 1;
    let scale = params.scale;
    let bootstrap_params = BootstrapParams::balanced();

    println!("         Ring dimension (N): {}", n);
    println!("         Number of primes: {} (max level: {})", num_primes, max_level);
    println!("         Scale: 2^{}", (scale.log2() as u32));
    println!();

    println!("  [3/7] Creating CUDA CKKS context...");
    let ckks_ctx = Arc::new(CudaCkksContext::new(params.clone())?);

    println!("  [4/7] Creating rotation context...");
    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);

    println!("  [5/7] Generating keys via KeyContext...");
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();

    // Convert secret key to strided format for CUDA
    let mut secret_key_strided = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            secret_key_strided[coeff_idx * num_primes + prime_idx] = sk.coeffs[coeff_idx].values[prime_idx];
        }
    }

    println!("  [6/7] Generating rotation and relinearization keys...");
    let rot_start = Instant::now();

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
    let rot_time = rot_start.elapsed().as_secs_f64();

    // Relinearization keys
    let relin_start = Instant::now();
    let relin_keys = Arc::new(CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        secret_key_strided.clone(),
        16,
        ckks_ctx.ntt_contexts(),
    )?);
    let relin_time = relin_start.elapsed().as_secs_f64();

    println!("         Rotation keys: {:.2}s ({} keys)", rot_time, rotation_keys.num_keys());
    println!("         Relin keys: {:.2}s", relin_time);
    println!();

    println!("  [7/7] Creating bootstrap context...");
    let bootstrap_ctx = CudaBootstrapContext::new(
        ckks_ctx.clone(),
        rotation_ctx.clone(),
        Arc::new(rotation_keys),
        relin_keys.clone(),
        bootstrap_params,
        params.clone(),
    )?;
    println!();

    println!("  Initialization complete!");
    println!();

    // ========================================================================
    // SECTION 2: SINGLE DIVISION BASELINE
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 2: SINGLE DIVISION BASELINE                                                         │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let num_val = 100.0;
    let denom_val = 7.0;
    let expected = num_val / denom_val;

    println!("  Test: {}/{} = {:.10}", num_val, denom_val, expected);
    println!();

    // Encrypt values
    let pt_num = ckks_ctx.encode(&[num_val], scale, max_level)?;
    let pt_denom = ckks_ctx.encode(&[denom_val], scale, max_level)?;

    let ct_num = ckks_ctx.encrypt(&pt_num, &pk)?;
    let ct_denom = ckks_ctx.encrypt(&pt_denom, &pk)?;

    println!("  Initial ciphertext level: {}", ct_num.level);

    // Single division
    let div_start = Instant::now();
    let ct_result = scalar_division_gpu(
        &ct_num,
        &ct_denom,
        1.0 / denom_val,
        2,  // 2 iterations
        &*relin_keys,
        &pk,
        &ckks_ctx,
    )?;
    let div_time = div_start.elapsed().as_secs_f64() * 1000.0;

    println!("  After division:");
    println!("    - Time: {:.2}ms", div_time);
    println!("    - Level: {} (consumed {} levels)", ct_result.level, max_level - ct_result.level);

    // Decrypt to verify
    let pt_result = ckks_ctx.decrypt(&ct_result, &sk)?;
    let result = ckks_ctx.decode(&pt_result)?[0];
    let error = (result - expected).abs() / expected;

    println!("    - Result: {:.10}", result);
    println!("    - Error: {:.2e}", error);
    println!();

    // ========================================================================
    // SECTION 3: CHAINED DIVISIONS WITHOUT BOOTSTRAP
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 3: CHAINED DIVISIONS WITHOUT BOOTSTRAP (limited by depth)                           │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Try to chain multiple divisions until we run out of depth
    // Each division with 2 iterations needs 5 levels

    let chain_values = vec![1000.0, 2.0, 5.0, 2.0, 5.0];  // 1000 / 2 / 5 / 2 / 5 = 10
    let expected_chain: f64 = chain_values.iter().skip(1).fold(chain_values[0], |acc, x| acc / x);

    println!("  Chain: {} / {} / {} / {} / {} = {}",
             chain_values[0], chain_values[1], chain_values[2],
             chain_values[3], chain_values[4], expected_chain);
    println!();

    let pt_start = ckks_ctx.encode(&[chain_values[0]], scale, max_level)?;
    let mut ct_current = ckks_ctx.encrypt(&pt_start, &pk)?;

    println!("  Starting level: {}", ct_current.level);
    println!();

    let mut divisions_completed = 0;
    let chain_start = Instant::now();

    for (i, &divisor) in chain_values.iter().skip(1).enumerate() {
        println!("  Division {}: / {}", i + 1, divisor);
        println!("    Current level: {}", ct_current.level);

        // Check if we have enough depth
        let required_levels = 5;  // 2 iterations * 2 + 1
        if ct_current.level < required_levels {
            println!("    INSUFFICIENT DEPTH - need {} levels, have {}", required_levels, ct_current.level);
            println!("    Chain stopped after {} divisions", divisions_completed);
            break;
        }

        // CORRECT: Encode divisor at the CURRENT ciphertext's scale and level
        // The tracked scale IS correct - use it for encoding
        let pt_div = ckks_ctx.encode(&[divisor], ct_current.scale, ct_current.level)?;
        let ct_div = ckks_ctx.encrypt(&pt_div, &pk)?;

        println!("    Numerator scale: {:.2e}, Divisor scale: {:.2e}", ct_current.scale, ct_div.scale);

        let div_start = Instant::now();
        ct_current = scalar_division_gpu(
            &ct_current,
            &ct_div,
            1.0 / divisor,
            2,
            &*relin_keys,
            &pk,
            &ckks_ctx,
        )?;
        let div_time = div_start.elapsed().as_secs_f64() * 1000.0;

        divisions_completed += 1;

        println!("    After division: level = {}, scale = {:.2e}, time = {:.2}ms",
                 ct_current.level, ct_current.scale, div_time);
    }

    let chain_time = chain_start.elapsed().as_secs_f64() * 1000.0;

    // Decrypt final result
    let pt_final = ckks_ctx.decrypt(&ct_current, &sk)?;
    let final_result = ckks_ctx.decode(&pt_final)?[0];

    // Calculate expected after partial chain
    let partial_expected: f64 = chain_values.iter()
        .skip(1)
        .take(divisions_completed)
        .fold(chain_values[0], |acc, x| acc / x);
    let final_error = (final_result - partial_expected).abs() / partial_expected;

    println!();
    println!("  Summary (without bootstrap):");
    println!("    - Divisions completed: {} of {}", divisions_completed, chain_values.len() - 1);
    println!("    - Total time: {:.2}ms", chain_time);
    println!("    - Final level: {}", ct_current.level);
    println!("    - Result: {:.10}", final_result);
    println!("    - Expected: {:.10}", partial_expected);
    println!("    - Error: {:.2e}", final_error);
    println!();

    // ========================================================================
    // SECTION 4: CHAINED DIVISIONS WITH BOOTSTRAP
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 4: CHAINED DIVISIONS WITH BOOTSTRAP (unlimited depth)                               │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("  Same chain: {} / {} / {} / {} / {} = {}",
             chain_values[0], chain_values[1], chain_values[2],
             chain_values[3], chain_values[4], expected_chain);
    println!();

    let pt_start = ckks_ctx.encode(&[chain_values[0]], scale, max_level)?;
    let mut ct_current = ckks_ctx.encrypt(&pt_start, &pk)?;

    println!("  Starting level: {}", ct_current.level);
    println!();

    let mut bootstraps_performed = 0;
    let mut total_div_time = 0.0;
    let mut total_bootstrap_time = 0.0;
    let chain_with_bootstrap_start = Instant::now();

    for (i, &divisor) in chain_values.iter().skip(1).enumerate() {
        println!("  Division {}: / {}", i + 1, divisor);
        println!("    Current level: {}", ct_current.level);

        // Check if we need bootstrap
        let required_levels = 5;
        if ct_current.level < required_levels {
            println!("    -> Bootstrapping to refresh noise budget...");
            let bootstrap_start = Instant::now();

            // Convert to bootstrap format and back
            let cuda_ct = ga_engine::clifford_fhe_v3::bootstrapping::cuda_bootstrap::CudaCiphertext {
                c0: ct_current.c0.clone(),
                c1: ct_current.c1.clone(),
                n: ct_current.n,
                num_primes: ct_current.num_primes,
                level: ct_current.level,
                scale: ct_current.scale,
            };

            let refreshed_ct = bootstrap_ctx.bootstrap(&cuda_ct)?;
            let bootstrap_time = bootstrap_start.elapsed().as_secs_f64() * 1000.0;
            total_bootstrap_time += bootstrap_time;
            bootstraps_performed += 1;

            // Convert back to CKKS ciphertext format
            ct_current = ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext {
                c0: refreshed_ct.c0,
                c1: refreshed_ct.c1,
                n: refreshed_ct.n,
                num_primes: refreshed_ct.num_primes,
                level: refreshed_ct.level,
                scale: refreshed_ct.scale,
            };

            println!("    -> Bootstrap complete: {:.2}ms, new level: {}", bootstrap_time, ct_current.level);
        }

        // CORRECT: Encode divisor at the CURRENT ciphertext's scale and level
        let pt_div = ckks_ctx.encode(&[divisor], ct_current.scale, ct_current.level)?;
        let ct_div = ckks_ctx.encrypt(&pt_div, &pk)?;

        println!("    Numerator scale: {:.2e}, Divisor scale: {:.2e}", ct_current.scale, ct_div.scale);

        let div_start = Instant::now();
        ct_current = scalar_division_gpu(
            &ct_current,
            &ct_div,
            1.0 / divisor,
            2,
            &*relin_keys,
            &pk,
            &ckks_ctx,
        )?;
        let div_time = div_start.elapsed().as_secs_f64() * 1000.0;
        total_div_time += div_time;

        println!("    After division: level = {}, scale = {:.2e}, time = {:.2}ms",
                 ct_current.level, ct_current.scale, div_time);
    }

    let total_chain_time = chain_with_bootstrap_start.elapsed().as_secs_f64() * 1000.0;

    // Decrypt final result
    let pt_final = ckks_ctx.decrypt(&ct_current, &sk)?;
    let final_result = ckks_ctx.decode(&pt_final)?[0];
    let final_error = (final_result - expected_chain).abs() / expected_chain;

    println!();
    println!("  Summary (with bootstrap):");
    println!("    - All {} divisions completed!", chain_values.len() - 1);
    println!("    - Bootstraps performed: {}", bootstraps_performed);
    println!("    - Total division time: {:.2}ms", total_div_time);
    println!("    - Total bootstrap time: {:.2}ms", total_bootstrap_time);
    println!("    - Total chain time: {:.2}ms", total_chain_time);
    println!("    - Final level: {}", ct_current.level);
    println!("    - Result: {:.10}", final_result);
    println!("    - Expected: {:.10}", expected_chain);
    println!("    - Error: {:.2e}", final_error);
    println!();

    // ========================================================================
    // SECTION 5: PERFORMANCE COMPARISON
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 5: PERFORMANCE COMPARISON                                                           │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("  ┌────────────────────────────────────────┬────────────────────────────────────────┐");
    println!("  │ Metric                                 │ Without Bootstrap │ With Bootstrap    │");
    println!("  ├────────────────────────────────────────┼───────────────────┼───────────────────┤");
    println!("  │ Divisions completed                    │ {:>17} │ {:>17} │",
             divisions_completed, chain_values.len() - 1);
    println!("  │ Maximum possible divisions             │ {:>17} │ {:>17} │",
             "~1-2", "Unlimited");
    println!("  │ Total time (ms)                        │ {:>17.2} │ {:>17.2} │",
             chain_time, total_chain_time);
    println!("  │ Bootstraps required                    │ {:>17} │ {:>17} │",
             0, bootstraps_performed);
    println!("  └────────────────────────────────────────┴───────────────────┴───────────────────┘");
    println!();

    println!("  Key Insights:");
    println!("    1. Without bootstrap, depth limits us to ~1-2 divisions");
    println!("    2. Bootstrap enables unlimited division chains");
    println!("    3. Bootstrap cost (~10s) amortized over multiple divisions");
    println!("    4. For long chains, bootstrap overhead becomes negligible per-division");
    println!();

    // ========================================================================
    // SECTION 6: SUMMARY
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 6: SUMMARY                                                                          │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("  Homomorphic Division + Bootstrap enables:");
    println!("    - Unlimited chains of divisions on encrypted data");
    println!("    - Complex iterative algorithms (optimization, physics)");
    println!("    - Full geometric algebra inverse operations");
    println!("    - Real-world privacy-preserving applications");
    println!();

    println!("  Depth Budget Management:");
    println!("    - Single division (2 iter): 5 levels");
    println!("    - Bootstrap refresh: restores to max level");
    println!("    - Strategy: monitor remaining depth, bootstrap when low");
    println!();

    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              BENCHMARK COMPLETE                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

#[cfg(not(all(feature = "v3", feature = "v2-gpu-cuda")))]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║          HOMOMORPHIC DIVISION WITH BOOTSTRAPPING                                             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("This benchmark requires V3 bootstrap and CUDA GPU support.");
    println!();
    println!("Run with:");
    println!("  cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 \\");
    println!("      --example bench_division_with_bootstrap");
}
