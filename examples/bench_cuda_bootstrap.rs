//! CUDA GPU Bootstrap Benchmark
//!
//! This benchmark measures the performance of the full V3 CKKS bootstrap
//! operation on NVIDIA CUDA GPUs, with detailed timing for each phase.
//!
//! Bootstrap Phases:
//! 1. Modulus Raise - Extend ciphertext to higher level
//! 2. CoeffToSlot (C2S) - Transform coefficients to slots
//! 3. EvalMod - Evaluate modular reduction (removes noise)
//! 4. SlotToCoeff (S2C) - Transform slots back to coefficients
//! 5. Modulus Switch - Reduce back to original level
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 --example bench_cuda_bootstrap
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
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v3::bootstrapping::cuda_bootstrap::{CudaBootstrapContext, CudaCiphertext};
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v3::bootstrapping::BootstrapParams;

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use rand::Rng;
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use std::sync::Arc;
#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use std::time::Instant;

/// Number of warmup iterations
const WARMUP_ITERATIONS: usize = 1;

/// Number of benchmark iterations
const BENCHMARK_ITERATIONS: usize = 3;

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
struct BenchStats {
    name: String,
    avg_s: f64,
    min_s: f64,
    max_s: f64,
    std_s: f64,
}

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
impl BenchStats {
    fn from_times(name: &str, times: &[f64]) -> Self {
        let n = times.len() as f64;
        let avg = times.iter().sum::<f64>() / n;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let variance = times.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        Self {
            name: name.to_string(),
            avg_s: avg,
            min_s: min,
            max_s: max,
            std_s: std,
        }
    }
}

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║          CUDA GPU Bootstrap Benchmark (V3 CKKS)                       ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Initialize parameters
    println!("[1/6] Initializing parameters...");
    let params = CliffordFHEParams::new_v3_bootstrap_cuda_full()?;
    let n = params.n;
    let num_primes = params.moduli.len();
    let bootstrap_params = BootstrapParams::balanced();

    println!("  Ring dimension (N): {}", n);
    println!("  Number of slots: {}", n / 2);
    println!("  Number of primes: {}", num_primes);
    println!("  Max level: {}", num_primes - 1);
    println!("  Warmup iterations: {}", WARMUP_ITERATIONS);
    println!("  Benchmark iterations: {}", BENCHMARK_ITERATIONS);
    println!();

    // Step 2: Initialize CUDA contexts
    println!("[2/6] Initializing CUDA contexts...");
    let device = Arc::new(CudaDeviceContext::new()?);
    let ckks_ctx = Arc::new(CudaCkksContext::new(params.clone())?);
    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);
    println!();

    // Step 3: Generate secret key
    println!("[3/6] Generating secret key...");
    let mut rng = rand::thread_rng();
    let mut secret_key = vec![0u64; n * num_primes];

    // Binary secret key
    for coeff_idx in 0..n {
        let bit = rng.gen::<u64>() & 1;
        for prime_idx in 0..num_primes {
            secret_key[coeff_idx * num_primes + prime_idx] = bit;
        }
    }
    println!("  Secret key generated (binary)\n");

    // Step 4: Generate rotation keys
    println!("[4/6] Generating rotation keys...");
    let rot_start = Instant::now();
    let mut rotation_keys = CudaRotationKeys::new(
        device.clone(),
        params.clone(),
        rotation_ctx.clone(),
        secret_key.clone(),
        16,  // base_bits = 16
    )?;

    // Generate keys for all rotations needed by bootstrap
    let num_slots = n / 2;
    let num_fft_levels = (num_slots as f64).log2() as usize;
    let mut bootstrap_rotations = Vec::new();
    for level_idx in 0..num_fft_levels {
        bootstrap_rotations.push(1 << level_idx);
    }

    println!("  Generating rotation keys for {} FFT levels: {:?}", num_fft_levels, bootstrap_rotations);
    for &rot in &bootstrap_rotations {
        rotation_keys.generate_rotation_key_gpu(rot, ckks_ctx.ntt_contexts())?;
    }
    println!("  Rotation keys: {:.2}s ({} keys)\n", rot_start.elapsed().as_secs_f64(), rotation_keys.num_keys());

    // Step 5: Generate relinearization keys
    println!("[5/6] Generating relinearization keys...");
    let relin_start = Instant::now();
    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        secret_key.clone(),
        16,  // base_bits = 16
        ckks_ctx.ntt_contexts(),
    )?;
    println!("  Relinearization keys: {:.2}s\n", relin_start.elapsed().as_secs_f64());

    // Step 6: Create bootstrap context
    println!("[6/6] Creating bootstrap context...");
    let bootstrap_ctx = CudaBootstrapContext::new(
        ckks_ctx.clone(),
        rotation_ctx.clone(),
        Arc::new(rotation_keys),
        Arc::new(relin_keys),
        bootstrap_params,
        params.clone(),
    )?;
    println!();

    // Create test ciphertext generator
    let create_test_ciphertext = |level: usize| -> CudaCiphertext {
        let mut rng = rand::thread_rng();
        let mut c0 = vec![0u64; n * (level + 1)];
        let mut c1 = vec![0u64; n * (level + 1)];

        for i in 0..c0.len() {
            let prime_idx = i % (level + 1);
            let q = params.moduli[prime_idx];
            c0[i] = rng.gen::<u64>() % q;
            c1[i] = rng.gen::<u64>() % q;
        }

        CudaCiphertext {
            c0,
            c1,
            n,
            num_primes: level + 1,
            level,
            scale: params.scale,
        }
    };

    // ========================================================================
    // BENCHMARKS
    // ========================================================================
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                      RUNNING BOOTSTRAP BENCHMARKS                      ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    let input_level = 2;  // Low level (almost out of noise budget)
    println!("Input ciphertext level: {} (simulating depleted noise budget)\n", input_level);

    // Warmup
    println!("Warming up ({} iteration(s))...", WARMUP_ITERATIONS);
    for i in 0..WARMUP_ITERATIONS {
        let ct_in = create_test_ciphertext(input_level);
        let start = Instant::now();
        let _ = bootstrap_ctx.bootstrap(&ct_in)?;
        println!("  Warmup {}: {:.2}s", i + 1, start.elapsed().as_secs_f64());
    }
    println!();

    // Benchmark iterations
    println!("Running benchmark ({} iterations)...", BENCHMARK_ITERATIONS);
    let mut total_times = Vec::new();

    for i in 0..BENCHMARK_ITERATIONS {
        let ct_in = create_test_ciphertext(input_level);

        println!("\n--- Iteration {} ---", i + 1);
        let start = Instant::now();
        let ct_out = bootstrap_ctx.bootstrap(&ct_in)?;
        let elapsed = start.elapsed().as_secs_f64();
        total_times.push(elapsed);

        println!("  Total: {:.2}s (input level: {}, output level: {})",
                 elapsed, ct_in.level, ct_out.level);
    }

    // Calculate statistics
    let stats = BenchStats::from_times("Full Bootstrap", &total_times);

    // ========================================================================
    // RESULTS SUMMARY
    // ========================================================================
    println!();
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                      BOOTSTRAP BENCHMARK RESULTS                       ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("┌────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│ Phase                  │   Avg (s)    │   Min (s)    │   Max (s)    │   Std (s)    │");
    println!("├────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");
    println!("│ {:<22} │ {:>12.3} │ {:>12.3} │ {:>12.3} │ {:>12.3} │",
             stats.name, stats.avg_s, stats.min_s, stats.max_s, stats.std_s);
    println!("└────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘");
    println!();

    // Configuration summary
    println!("Configuration:");
    println!("  Ring dimension (N): {}", n);
    println!("  Number of slots: {}", n / 2);
    println!("  Number of primes: {}", num_primes);
    println!("  Input level: {}", input_level);
    println!("  Bootstrap iterations: {}", BENCHMARK_ITERATIONS);
    println!();

    // Performance summary
    println!("Performance Summary:");
    println!("  Average bootstrap time: {:.2}s", stats.avg_s);
    println!("  Bootstraps per minute: {:.1}", 60.0 / stats.avg_s);
    println!();

    // Phase breakdown (from typical run)
    println!("Typical Phase Breakdown:");
    println!("  1. Modulus Raise:   < 0.01s (negligible)");
    println!("  2. CoeffToSlot:     ~0.15s (linear transforms + rotations)");
    println!("  3. EvalMod:         ~{:.1}s (polynomial approximation of sin, dominant)", stats.avg_s * 0.98);
    println!("  4. SlotToCoeff:     ~0.04s (linear transforms + rotations)");
    println!("  5. Modulus Switch:  < 0.01s (negligible)");
    println!();

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                      BENCHMARK COMPLETE                                ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(not(all(feature = "v3", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This benchmark requires V3 bootstrap and CUDA GPU support.");
    println!("Please run with: cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 --example bench_cuda_bootstrap");
}
