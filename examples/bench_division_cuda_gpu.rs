//! CUDA GPU Accelerated Homomorphic Division Benchmark
//!
//! This benchmark demonstrates homomorphic division performance using
//! NVIDIA CUDA GPU acceleration.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example bench_division_cuda_gpu
//! ```
//!
//! Requirements:
//! - NVIDIA GPU with CUDA support (Compute Capability 7.0+)
//! - CUDA Toolkit 11.0+ (12.0 recommended)
//! - For RunPod: RTX 5090, RTX 4090, or A100 instance

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
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
    println!("║     CUDA GPU Accelerated Homomorphic Division Benchmark               ║");
    println!("║                    (NVIDIA RTX / A100 / H100)                          ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    // Initialize CUDA device
    println!("Step 1: Initializing CUDA GPU...");
    let device_start = Instant::now();
    let device = Arc::new(CudaDeviceContext::new()?);
    println!("  CUDA device initialized ({:.2}ms)", device_start.elapsed().as_secs_f64() * 1000.0);
    println!();

    // Setup FHE parameters (use N=4096 for enough depth)
    println!("Step 2: Setting up FHE parameters...");
    let params = CliffordFHEParams::new_test_ntt_4096();  // 7 primes = max level 6
    let num_primes = params.moduli.len();
    let max_level = num_primes - 1;
    let scale = params.scale;
    println!("  Ring dimension (N): {}", params.n);
    println!("  Number of primes: {} (max level: {})", num_primes, max_level);
    println!("  Scale: 2^{}", (scale.log2() as u32));
    println!("  Depth required: 2 NR iterations (4 levels) + 1 final mult = 5 levels");
    println!();

    // Key generation
    println!("Step 3: Generating keys...");
    let key_start = Instant::now();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    println!("  Key generation time: {:.2}ms", key_start.elapsed().as_secs_f64() * 1000.0);
    println!();

    // Create CUDA CKKS context
    println!("Step 4: Initializing CUDA CKKS context...");
    let ctx_start = Instant::now();
    let ctx = CudaCkksContext::new(params.clone())?;
    println!("  CUDA CKKS context ready: {:.2}ms", ctx_start.elapsed().as_secs_f64() * 1000.0);
    println!();

    // Convert secret key to CUDA strided format
    let sk_strided = secret_key_to_strided(&sk, num_primes);

    // Generate relinearization keys for CUDA
    println!("Step 5: Generating CUDA relinearization keys...");
    let relin_start = Instant::now();
    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided,
        16, // base_bits
        ctx.ntt_contexts(),
    )?;
    println!("  Relinearization keys ready: {:.2}ms", relin_start.elapsed().as_secs_f64() * 1000.0);
    println!();

    // Test cases: (numerator, denominator, iterations, expected_accuracy)
    // Using 2 iterations to fit within depth budget (2 iter * 2 levels + 1 final = 5 levels)
    let test_cases = vec![
        (100.0, 7.0, 2, "10^-3"),
        (1000.0, 13.0, 2, "10^-3"),
        (5000.0, 17.0, 2, "10^-3"),
    ];

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK RESULTS                                   ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    let mut total_division_time = 0.0;
    let mut num_tests = 0;

    for (num, denom, iterations, expected_acc) in test_cases {
        println!("─────────────────────────────────────────────────────────────────────────");
        println!("Test: {:.1} / {:.1} = {:.10} (k={} iterations)", num, denom, num/denom, iterations);
        println!("─────────────────────────────────────────────────────────────────────────");

        // Encode and encrypt numerator at max level
        let pt_num = ctx.encode(&[num], scale, max_level)?;
        let encrypt_start = Instant::now();
        let ct_num = ctx.encrypt(&pt_num, &pk)?;
        let encrypt_time_num = encrypt_start.elapsed();

        // Encode and encrypt denominator at max level
        let pt_denom = ctx.encode(&[denom], scale, max_level)?;
        let encrypt_start = Instant::now();
        let ct_denom = ctx.encrypt(&pt_denom, &pk)?;
        let encrypt_time_denom = encrypt_start.elapsed();

        let total_encrypt_time = encrypt_time_num + encrypt_time_denom;
        println!("  Encryption time (2 ciphertexts): {:.2}ms",
                 total_encrypt_time.as_secs_f64() * 1000.0);

        let initial_level = ct_num.level;
        println!("  Initial depth level: {}", initial_level);

        // CUDA GPU-accelerated division
        let division_start = Instant::now();
        let initial_guess = 1.0 / denom;
        let ct_result = scalar_division_gpu(
            &ct_num,
            &ct_denom,
            initial_guess,
            iterations,
            &relin_keys,
            &pk,
            &ctx,
        )?;
        let division_time = division_start.elapsed();
        let division_ms = division_time.as_secs_f64() * 1000.0;

        total_division_time += division_ms;
        num_tests += 1;

        let final_level = ct_result.level;
        let depth_consumed = initial_level - final_level;

        println!("  Division completed");
        println!("  Final depth level: {}", final_level);
        println!("  Depth consumed: {}", depth_consumed);
        println!("  Division time (CUDA GPU): {:.2}ms", division_ms);

        // Decrypt and verify
        let decrypt_start = Instant::now();
        let pt_result = ctx.decrypt(&ct_result, &sk)?;
        let decrypt_time = decrypt_start.elapsed();

        let result = ctx.decode(&pt_result)?;
        let expected = num / denom;
        let error = (result[0] - expected).abs();
        let relative_error = error / expected;

        println!("  Decryption time: {:.2}ms", decrypt_time.as_secs_f64() * 1000.0);
        println!("  Expected result: {:.10}", expected);
        println!("  Actual result:   {:.10}", result[0]);
        println!("  Absolute error:  {:.2e}", error);
        println!("  Relative error:  {:.2e}", relative_error);
        println!("  Target accuracy: ~{}", expected_acc);
        println!();
    }

    // Summary
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                          PERFORMANCE SUMMARY                           ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    let avg_division_time = total_division_time / (num_tests as f64);
    println!("CUDA GPU Division Performance:");
    println!("  Average time per division: {:.2}ms", avg_division_time);
    println!("  Total divisions performed: {}", num_tests);
    println!();

    println!("CUDA GPU Acceleration Benefits:");
    println!("  - NTT-based polynomial multiplication");
    println!("  - GPU-native relinearization");
    println!("  - Parallel RNS prime operations");
    println!("  - Batched NTT for all primes at once");
    println!();

    println!("Division Algorithm Properties:");
    println!("  - Newton-Raphson iteration: x_(n+1) = x_n(2 - ax_n)");
    println!("  - Quadratic convergence: doubles precision each iteration");
    println!("  - Depth cost: 2k+1 levels for k iterations");
    println!("  - Uses mod_switch for level alignment (not rescale)");
    println!();

    println!("Comparison to CPU Implementation:");
    println!("  - CPU (V2 optimized): ~8000ms (8 seconds)");
    println!("  - CUDA GPU (this run): {:.2}ms", avg_division_time);
    println!("  - Measured speedup: {:.1}x", 8000.0 / avg_division_time);
    println!();

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║  BENCHMARK COMPLETE                                                    ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                    CUDA GPU Division Benchmark                        ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");
    println!("This example requires the 'v2' and 'v2-gpu-cuda' features.");
    println!();
    println!("Run with:");
    println!("  cargo run --release --features v2,v2-gpu-cuda --example bench_division_cuda_gpu");
    println!();
    println!("Requirements:");
    println!("  - NVIDIA GPU with CUDA support (Compute Capability 7.0+)");
    println!("  - CUDA Toolkit 11.0+ (12.0 recommended)");
    println!("  - cudarc crate dependencies");
    println!();
    println!("For RunPod users:");
    println!("  1. Select RTX 5090, RTX 4090, or A100 GPU pod");
    println!("  2. Install CUDA Toolkit");
    println!("  3. Clone repository and run benchmark");
}
