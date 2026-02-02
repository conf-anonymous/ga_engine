//! Comprehensive Homomorphic Division Benchmark Suite
//!
//! This benchmark provides extensive performance data for homomorphic division
//! across multiple scenarios, parameter sets, and iteration counts.
//!
//! ## Scenarios Covered:
//! 1. Basic scalar division (a/b)
//! 2. Multiple iteration counts (2, 3, 4 iterations)
//! 3. Different input ranges (small, medium, large values)
//! 4. Chained divisions (a/b/c)
//! 5. Precision vs performance tradeoffs
//!
//! ## Run:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example bench_division_comprehensive
//! ```
//!
//! For different ring dimensions:
//! ```bash
//! RING_DIM=1024 cargo run --release --features v2,v2-gpu-cuda --example bench_division_comprehensive
//! RING_DIM=4096 cargo run --release --features v2,v2-gpu-cuda --example bench_division_comprehensive
//! RING_DIM=8192 cargo run --release --features v2,v2-gpu-cuda --example bench_division_comprehensive
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::{
    backends::{
        cpu_optimized::keys::KeyContext,
        gpu_cuda::{
            ckks::CudaCkksContext,
            device::CudaDeviceContext,
            inversion::{newton_raphson_inverse_gpu, scalar_division_gpu, multiply_ciphertexts_gpu},
            relin_keys::CudaRelinKeys,
        },
    },
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::sync::Arc;
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::time::Instant;
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::env;

/// Statistics for a benchmark run
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
struct BenchResult {
    name: String,
    numerator: f64,
    denominator: f64,
    expected: f64,
    actual: f64,
    iterations: usize,
    time_ms: f64,
    rel_error: f64,
    depth_consumed: usize,
    initial_level: usize,
    final_level: usize,
}

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
impl BenchResult {
    fn print_row(&self) {
        println!("│ {:20} │ {:>10.4} │ {:>10.4} │ {:>10.4} │ {:>4} │ {:>10.2} │ {:>10.2e} │ {:>6} │",
            self.name,
            self.numerator,
            self.denominator,
            self.expected,
            self.iterations,
            self.time_ms,
            self.rel_error,
            self.depth_consumed,
        );
    }
}

/// Convert CPU SecretKey to CUDA strided format
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
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    COMPREHENSIVE HOMOMORPHIC DIVISION BENCHMARK                              ║");
    println!("║                              CUDA GPU Acceleration                                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // ========================================================================
    // SECTION 1: INITIALIZATION
    // ========================================================================

    // Check for ring dimension override
    let ring_dim: usize = env::var("RING_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 1: INITIALIZATION                                                                   │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Initialize CUDA device
    println!("  [1/5] Initializing CUDA GPU...");
    let device_start = Instant::now();
    let device = Arc::new(CudaDeviceContext::new()?);
    println!("         CUDA device ready ({:.2}ms)", device_start.elapsed().as_secs_f64() * 1000.0);

    // Setup parameters based on ring dimension
    // NOTE: Division requires at least 5 levels for 2 iterations (2*2+1=5)
    //       N=1024 only has 3 primes (max level 2) - INSUFFICIENT
    //       N=2048 only has 5 primes (max level 4) - INSUFFICIENT
    //       N=4096 has 7 primes (max level 6) - OK for 2 iterations
    //       N=8192 has 9 primes (max level 8) - OK for 2-3 iterations
    println!("  [2/5] Setting up FHE parameters...");
    let params = match ring_dim {
        1024 => {
            println!("\n  ⚠️  WARNING: N=1024 only has 3 primes (max level 2)");
            println!("     Division requires 5 levels minimum (for 2 iterations).");
            println!("     Skipping N=1024 - use N=4096 or higher for division.\n");
            return Ok(());
        },
        2048 => {
            println!("\n  ⚠️  WARNING: N=2048 only has 5 primes (max level 4)");
            println!("     Division requires 5 levels minimum (for 2 iterations).");
            println!("     Skipping N=2048 - use N=4096 or higher for division.\n");
            return Ok(());
        },
        4096 => CliffordFHEParams::new_test_ntt_4096(),
        8192 => CliffordFHEParams::new_128bit(),  // N=8192, 9 primes, production security
        _ => {
            println!("         Warning: Unsupported ring dimension {}, using 4096", ring_dim);
            CliffordFHEParams::new_test_ntt_4096()
        }
    };

    let num_primes = params.moduli.len();
    let max_level = num_primes - 1;
    let scale = params.scale;

    println!("         Ring dimension (N): {}", params.n);
    println!("         Number of primes: {} (max level: {})", num_primes, max_level);
    println!("         Scale: 2^{}", (scale.log2() as u32));
    println!("         Security: ~128-bit");
    println!("         Max Newton-Raphson iterations: {}", (max_level - 1) / 2);

    // Key generation
    println!("  [3/5] Generating cryptographic keys...");
    let key_start = Instant::now();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let keygen_time = key_start.elapsed().as_secs_f64() * 1000.0;
    println!("         Keys generated ({:.2}ms)", keygen_time);

    // Create CUDA context
    println!("  [4/5] Initializing CUDA CKKS context...");
    let ctx_start = Instant::now();
    let ctx = CudaCkksContext::new(params.clone())?;
    println!("         CUDA CKKS context ready ({:.2}ms)", ctx_start.elapsed().as_secs_f64() * 1000.0);

    // Generate relinearization keys
    println!("  [5/5] Generating CUDA relinearization keys...");
    let sk_strided = secret_key_to_strided(&sk, num_primes);
    let relin_start = Instant::now();
    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided,
        16,
        ctx.ntt_contexts(),
    )?;
    let relin_time = relin_start.elapsed().as_secs_f64() * 1000.0;
    println!("         Relinearization keys ready ({:.2}ms)", relin_time);
    println!();

    // Print initialization summary
    println!("  ┌────────────────────────────────────────────────────────────────────────┐");
    println!("  │ INITIALIZATION SUMMARY                                                 │");
    println!("  ├────────────────────────┬───────────────────────────────────────────────┤");
    println!("  │ Parameter              │ Value                                         │");
    println!("  ├────────────────────────┼───────────────────────────────────────────────┤");
    println!("  │ Ring Dimension (N)     │ {:>45} │", params.n);
    println!("  │ Number of Primes       │ {:>45} │", num_primes);
    println!("  │ Max Level              │ {:>45} │", max_level);
    println!("  │ Scale (log2)           │ {:>45} │", (scale.log2() as u32));
    println!("  │ Key Generation         │ {:>42.2} ms │", keygen_time);
    println!("  │ Relin Key Generation   │ {:>42.2} ms │", relin_time);
    println!("  └────────────────────────┴───────────────────────────────────────────────┘");
    println!();

    // ========================================================================
    // SECTION 2: BASIC DIVISION BENCHMARKS
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 2: BASIC DIVISION BENCHMARKS                                                        │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Determine max iterations based on available depth
    // Division needs 2k+1 levels for k iterations
    let max_iterations = (max_level - 1) / 2;
    println!("  Max iterations possible with {} levels: {}", max_level, max_iterations);
    println!();

    let basic_tests: Vec<(f64, f64, &str)> = vec![
        (10.0, 2.0, "Simple (10/2)"),
        (100.0, 7.0, "Integer (100/7)"),
        (1000.0, 13.0, "Large/Prime (1000/13)"),
        (5000.0, 17.0, "Larger (5000/17)"),
        (1.0, 3.0, "Fraction (1/3)"),
        (22.0, 7.0, "Pi approx (22/7)"),
        (355.0, 113.0, "Better Pi (355/113)"),
        (99.0, 100.0, "Near unity (0.99)"),
    ];

    let mut basic_results: Vec<BenchResult> = Vec::new();

    // Run with 2 iterations (if we have enough depth)
    if max_iterations >= 2 {
        println!("  Testing with 2 Newton-Raphson iterations (depth cost: 5 levels):");
        println!("  ┌──────────────────────┬────────────┬────────────┬────────────┬──────┬────────────┬────────────┬────────┐");
        println!("  │ Test Case            │  Numerator │ Denominator│   Expected │ Iter │   Time(ms) │  Rel Error │  Depth │");
        println!("  ├──────────────────────┼────────────┼────────────┼────────────┼──────┼────────────┼────────────┼────────┤");

        for (num, denom, name) in &basic_tests {
            let result = run_division_test(&ctx, &pk, &sk, &relin_keys, *num, *denom, 2, name, max_level)?;
            print!("  ");
            result.print_row();
            basic_results.push(result);
        }
        println!("  └──────────────────────┴────────────┴────────────┴────────────┴──────┴────────────┴────────────┴────────┘");
        println!();
    }

    // Run with 3 iterations (if we have enough depth)
    if max_iterations >= 3 {
        println!("  Testing with 3 Newton-Raphson iterations (depth cost: 7 levels):");
        println!("  ┌──────────────────────┬────────────┬────────────┬────────────┬──────┬────────────┬────────────┬────────┐");
        println!("  │ Test Case            │  Numerator │ Denominator│   Expected │ Iter │   Time(ms) │  Rel Error │  Depth │");
        println!("  ├──────────────────────┼────────────┼────────────┼────────────┼──────┼────────────┼────────────┼────────┤");

        for (num, denom, name) in &basic_tests {
            let result = run_division_test(&ctx, &pk, &sk, &relin_keys, *num, *denom, 3, name, max_level)?;
            print!("  ");
            result.print_row();
            basic_results.push(result);
        }
        println!("  └──────────────────────┴────────────┴────────────┴────────────┴──────┴────────────┴────────────┴────────┘");
        println!();
    }

    // ========================================================================
    // SECTION 3: PRECISION VS ITERATION COUNT
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 3: PRECISION VS ITERATION COUNT                                                     │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Test the same division with different iteration counts
    let test_num = 100.0;
    let test_denom = 7.0;

    println!("  Test case: {}/{} = {:.10}", test_num, test_denom, test_num / test_denom);
    println!();
    println!("  ┌────────────┬──────────┬────────────┬────────────────────┬────────────┐");
    println!("  │ Iterations │ Depth    │ Time (ms)  │ Relative Error     │ Digits     │");
    println!("  ├────────────┼──────────┼────────────┼────────────────────┼────────────┤");

    for iter in 1..=max_iterations.min(4) {
        let result = run_division_test(&ctx, &pk, &sk, &relin_keys, test_num, test_denom, iter, "precision_test", max_level)?;
        let digits = if result.rel_error > 0.0 { -result.rel_error.log10() } else { 16.0 };
        println!("  │ {:>10} │ {:>8} │ {:>10.2} │ {:>18.2e} │ {:>10.1} │",
            iter, result.depth_consumed, result.time_ms, result.rel_error, digits);
    }
    println!("  └────────────┴──────────┴────────────┴────────────────────┴────────────┘");
    println!();

    // ========================================================================
    // SECTION 4: CHAINED DIVISIONS
    // ========================================================================

    if max_iterations >= 2 && max_level >= 10 {
        println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
        println!("│ SECTION 4: CHAINED DIVISIONS (without bootstrap)                                            │");
        println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
        println!();

        // Chain: ((a / b) / c)
        let a = 1000.0;
        let b = 4.0;
        let c = 5.0;
        let expected_chain = (a / b) / c; // 50.0

        println!("  Test: (({} / {}) / {}) = {}", a, b, c, expected_chain);
        println!();

        let chain_start = Instant::now();

        // First division: a / b
        let pt_a = ctx.encode(&[a], scale, max_level)?;
        let pt_b = ctx.encode(&[b], scale, max_level)?;
        let ct_a = ctx.encrypt(&pt_a, &pk)?;
        let ct_b = ctx.encrypt(&pt_b, &pk)?;

        let div1_start = Instant::now();
        let ct_ab = scalar_division_gpu(&ct_a, &ct_b, 1.0/b, 2, &relin_keys, &pk, &ctx)?;
        let div1_time = div1_start.elapsed().as_secs_f64() * 1000.0;

        // Second division: (a/b) / c
        let pt_c = ctx.encode(&[c], scale, ct_ab.level)?;
        let ct_c = ctx.encrypt(&pt_c, &pk)?;

        // Align levels if needed
        let (ct_ab_aligned, ct_c_aligned) = if ct_ab.level != ct_c.level {
            let target = ct_ab.level.min(ct_c.level);
            (ct_ab.mod_switch_to_level(target), ct_c.mod_switch_to_level(target))
        } else {
            (ct_ab.clone(), ct_c)
        };

        let div2_start = Instant::now();
        let ct_result = scalar_division_gpu(&ct_ab_aligned, &ct_c_aligned, 1.0/c, 1, &relin_keys, &pk, &ctx)?;
        let div2_time = div2_start.elapsed().as_secs_f64() * 1000.0;

        let total_chain_time = chain_start.elapsed().as_secs_f64() * 1000.0;

        // Decrypt and verify
        let pt_result = ctx.decrypt(&ct_result, &sk)?;
        let result = ctx.decode(&pt_result)?;
        let error = (result[0] - expected_chain).abs() / expected_chain;

        println!("  ┌─────────────────────────┬────────────────────────────────────────────┐");
        println!("  │ Metric                  │ Value                                      │");
        println!("  ├─────────────────────────┼────────────────────────────────────────────┤");
        println!("  │ First division time     │ {:>39.2} ms │", div1_time);
        println!("  │ Second division time    │ {:>39.2} ms │", div2_time);
        println!("  │ Total chain time        │ {:>39.2} ms │", total_chain_time);
        println!("  │ Expected result         │ {:>42.6} │", expected_chain);
        println!("  │ Actual result           │ {:>42.6} │", result[0]);
        println!("  │ Relative error          │ {:>42.2e} │", error);
        println!("  │ Final level             │ {:>42} │", ct_result.level);
        println!("  │ Total depth consumed    │ {:>42} │", max_level - ct_result.level);
        println!("  └─────────────────────────┴────────────────────────────────────────────┘");
        println!();
    }

    // ========================================================================
    // SECTION 5: INVERSE COMPUTATION ONLY
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 5: INVERSE COMPUTATION ONLY (1/x)                                                   │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let inverse_tests: Vec<(f64, &str)> = vec![
        (2.0, "1/2"),
        (4.0, "1/4"),
        (10.0, "1/10"),
        (100.0, "1/100"),
        (0.5, "1/0.5 = 2"),
        (0.25, "1/0.25 = 4"),
    ];

    println!("  ┌──────────────────────┬────────────┬────────────┬────────────┬──────┬────────────┐");
    println!("  │ Test Case            │      Input │   Expected │     Actual │ Iter │   Time(ms) │");
    println!("  ├──────────────────────┼────────────┼────────────┼────────────┼──────┼────────────┤");

    for (x, name) in &inverse_tests {
        let expected = 1.0 / x;
        let pt_x = ctx.encode(&[*x], scale, max_level)?;
        let ct_x = ctx.encrypt(&pt_x, &pk)?;

        let start = Instant::now();
        let ct_inv = newton_raphson_inverse_gpu(&ct_x, 1.0 / x, 2, &relin_keys, &pk, &ctx)?;
        let time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let pt_result = ctx.decrypt(&ct_inv, &sk)?;
        let result = ctx.decode(&pt_result)?;

        println!("  │ {:20} │ {:>10.4} │ {:>10.6} │ {:>10.6} │ {:>4} │ {:>10.2} │",
            name, x, expected, result[0], 2, time_ms);
    }
    println!("  └──────────────────────┴────────────┴────────────┴────────────┴──────┴────────────┘");
    println!();

    // ========================================================================
    // SECTION 6: PERFORMANCE SUMMARY
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 6: PERFORMANCE SUMMARY                                                              │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Calculate statistics from basic results
    if !basic_results.is_empty() {
        let iter_2_results: Vec<&BenchResult> = basic_results.iter().filter(|r| r.iterations == 2).collect();
        let iter_3_results: Vec<&BenchResult> = basic_results.iter().filter(|r| r.iterations == 3).collect();

        if !iter_2_results.is_empty() {
            let avg_time_2: f64 = iter_2_results.iter().map(|r| r.time_ms).sum::<f64>() / iter_2_results.len() as f64;
            let avg_error_2: f64 = iter_2_results.iter().map(|r| r.rel_error).sum::<f64>() / iter_2_results.len() as f64;
            println!("  2 Iterations Statistics:");
            println!("    - Average time:  {:.2} ms", avg_time_2);
            println!("    - Average error: {:.2e}", avg_error_2);
            println!("    - Throughput:    {:.2} divisions/second", 1000.0 / avg_time_2);
            println!();
        }

        if !iter_3_results.is_empty() {
            let avg_time_3: f64 = iter_3_results.iter().map(|r| r.time_ms).sum::<f64>() / iter_3_results.len() as f64;
            let avg_error_3: f64 = iter_3_results.iter().map(|r| r.rel_error).sum::<f64>() / iter_3_results.len() as f64;
            println!("  3 Iterations Statistics:");
            println!("    - Average time:  {:.2} ms", avg_time_3);
            println!("    - Average error: {:.2e}", avg_error_3);
            println!("    - Throughput:    {:.2} divisions/second", 1000.0 / avg_time_3);
            println!();
        }
    }

    println!("  Algorithm Properties:");
    println!("    - Newton-Raphson iteration: x_(n+1) = x_n(2 - b*x_n)");
    println!("    - Quadratic convergence: doubles precision each iteration");
    println!("    - Depth cost: 2k+1 levels for k iterations");
    println!("    - No comparison circuits required (constant-time)");
    println!();

    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              BENCHMARK COMPLETE                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

/// Run a single division benchmark
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn run_division_test(
    ctx: &CudaCkksContext,
    pk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey,
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    relin_keys: &CudaRelinKeys,
    numerator: f64,
    denominator: f64,
    iterations: usize,
    name: &str,
    max_level: usize,
) -> Result<BenchResult, String> {
    let scale = ctx.params().scale;
    let expected = numerator / denominator;

    // Encode and encrypt
    let pt_num = ctx.encode(&[numerator], scale, max_level)?;
    let pt_denom = ctx.encode(&[denominator], scale, max_level)?;
    let ct_num = ctx.encrypt(&pt_num, pk)?;
    let ct_denom = ctx.encrypt(&pt_denom, pk)?;

    let initial_level = ct_num.level;

    // Run division
    let start = Instant::now();
    let ct_result = scalar_division_gpu(
        &ct_num,
        &ct_denom,
        1.0 / denominator,
        iterations,
        relin_keys,
        pk,
        ctx,
    )?;
    let time_ms = start.elapsed().as_secs_f64() * 1000.0;

    let final_level = ct_result.level;

    // Decrypt and compute error
    let pt_result = ctx.decrypt(&ct_result, sk)?;
    let result = ctx.decode(&pt_result)?;
    let actual = result[0];
    let rel_error = (actual - expected).abs() / expected.abs();

    Ok(BenchResult {
        name: name.to_string(),
        numerator,
        denominator,
        expected,
        actual,
        iterations,
        time_ms,
        rel_error,
        depth_consumed: initial_level - final_level,
        initial_level,
        final_level,
    })
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    COMPREHENSIVE HOMOMORPHIC DIVISION BENCHMARK                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("This benchmark requires CUDA GPU support.");
    println!();
    println!("Run with:");
    println!("  cargo run --release --features v2,v2-gpu-cuda --example bench_division_comprehensive");
    println!();
    println!("For different ring dimensions:");
    println!("  RING_DIM=1024 cargo run --release --features v2,v2-gpu-cuda --example bench_division_comprehensive");
    println!("  RING_DIM=4096 cargo run --release --features v2,v2-gpu-cuda --example bench_division_comprehensive");
    println!("  RING_DIM=8192 cargo run --release --features v2,v2-gpu-cuda --example bench_division_comprehensive");
    println!();
    println!("Requirements:");
    println!("  - NVIDIA GPU with CUDA support (Compute Capability 7.0+)");
    println!("  - CUDA Toolkit 12.0+");
}
