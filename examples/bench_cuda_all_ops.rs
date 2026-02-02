//! Comprehensive CUDA Homomorphic Operations Benchmark
//!
//! This benchmark measures the performance of ALL core homomorphic operations
//! on NVIDIA CUDA GPUs for Clifford FHE.
//!
//! Operations benchmarked:
//! - Encode: Convert plaintext values to polynomial representation
//! - Encrypt: Encrypt a plaintext to ciphertext
//! - Decrypt: Decrypt a ciphertext to plaintext
//! - Decode: Convert polynomial back to values
//! - Add: Add two ciphertexts
//! - Subtract: Subtract two ciphertexts
//! - Multiply: Multiply two ciphertexts (includes relin + rescale)
//! - Multiply Plain: Multiply ciphertext by plaintext
//! - Rotate: Rotate slots by a given amount
//! - Rescale: Reduce scale and drop a prime
//! - Mod Switch: Drop a prime without rescaling
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda --example bench_cuda_all_ops
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::{
    backends::{
        cpu_optimized::keys::KeyContext,
        gpu_cuda::{
            ckks::{CudaCkksContext, CudaPlaintext},
            device::CudaDeviceContext,
            inversion::multiply_ciphertexts_gpu,
            relin_keys::CudaRelinKeys,
            rotation::CudaRotationContext,
            rotation_keys::CudaRotationKeys,
        },
    },
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::sync::Arc;
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::time::Instant;

/// Number of warmup iterations before timing
const WARMUP_ITERATIONS: usize = 2;

/// Number of timed iterations for averaging
const BENCHMARK_ITERATIONS: usize = 10;

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

/// Statistics for benchmark results
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
struct BenchStats {
    name: String,
    avg_ms: f64,
    min_ms: f64,
    max_ms: f64,
    std_ms: f64,
}

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
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
            avg_ms: avg,
            min_ms: min,
            max_ms: max,
            std_ms: std,
        }
    }
}

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     CUDA Comprehensive Homomorphic Operations Benchmark               ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize CUDA device
    println!("[1/5] Initializing CUDA GPU...");
    let device = Arc::new(CudaDeviceContext::new()?);
    println!();

    // Setup FHE parameters
    println!("[2/5] Setting up FHE parameters...");
    let params = CliffordFHEParams::new_test_ntt_4096();
    let num_primes = params.moduli.len();
    let max_level = num_primes - 1;
    let scale = params.scale;
    let n = params.n;

    println!("  Ring dimension (N): {}", n);
    println!("  Number of primes: {} (max level: {})", num_primes, max_level);
    println!("  Scale: 2^{}", (scale.log2() as u32));
    println!("  Warmup iterations: {}", WARMUP_ITERATIONS);
    println!("  Benchmark iterations: {}", BENCHMARK_ITERATIONS);
    println!();

    // Key generation
    println!("[3/5] Generating keys...");
    let key_start = Instant::now();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    println!("  Key generation: {:.2}ms", key_start.elapsed().as_secs_f64() * 1000.0);

    // Create CUDA CKKS context
    println!("[4/5] Initializing CUDA CKKS context...");
    let ctx_start = Instant::now();
    let ctx = CudaCkksContext::new(params.clone())?;
    println!("  CUDA CKKS context: {:.2}ms", ctx_start.elapsed().as_secs_f64() * 1000.0);

    // Convert secret key to CUDA strided format (for relin keys)
    let sk_strided = secret_key_to_strided(&sk, num_primes);

    // Generate relinearization keys
    println!("[5/5] Generating evaluation keys...");
    let relin_start = Instant::now();
    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided.clone(),
        16, // base_bits
        ctx.ntt_contexts(),
    )?;
    println!("  Relinearization keys: {:.2}ms", relin_start.elapsed().as_secs_f64() * 1000.0);

    // Generate rotation context and keys
    let rot_ctx_start = Instant::now();
    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);
    let mut rot_keys = CudaRotationKeys::new(
        device.clone(),
        params.clone(),
        rotation_ctx.clone(),
        sk_strided.clone(),
        16, // base_bits
    )?;

    // Generate rotation keys for rotations ±1 to ±8
    for rot in 1..=8 {
        rot_keys.generate_rotation_key_gpu(rot, ctx.ntt_contexts())?;
        rot_keys.generate_rotation_key_gpu(-rot, ctx.ntt_contexts())?;
    }
    println!("  Rotation keys: {:.2}ms ({} keys)", rot_ctx_start.elapsed().as_secs_f64() * 1000.0, rot_keys.num_keys());
    println!();

    // Create test data
    let slots = n / 2;
    let values_a: Vec<f64> = (0..slots).map(|i| (i as f64) * 0.001 + 1.0).collect();
    let values_b: Vec<f64> = (0..slots).map(|i| (i as f64) * 0.002 + 2.0).collect();
    let plain_const: Vec<f64> = vec![3.14159; slots];

    // ========================================================================
    // BENCHMARKS
    // ========================================================================
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                      RUNNING BENCHMARKS                                ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    let mut results: Vec<BenchStats> = Vec::new();

    // ------------------------------------------------------------------------
    // 1. ENCODE
    // ------------------------------------------------------------------------
    print!("Benchmarking ENCODE...");
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = CudaPlaintext::encode(&values_a, scale, &params);
    }

    // Benchmark
    for _ in 0..BENCHMARK_ITERATIONS {
        let start = Instant::now();
        let _ = CudaPlaintext::encode(&values_a, scale, &params);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    results.push(BenchStats::from_times("Encode", &times));
    println!(" done ({:.3}ms avg)", results.last().unwrap().avg_ms);

    // ------------------------------------------------------------------------
    // 2. ENCRYPT
    // ------------------------------------------------------------------------
    print!("Benchmarking ENCRYPT...");
    let pt_a = CudaPlaintext::encode(&values_a, scale, &params);
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = ctx.encrypt(&pt_a, &pk)?;
    }

    // Benchmark
    for _ in 0..BENCHMARK_ITERATIONS {
        let start = Instant::now();
        let _ = ctx.encrypt(&pt_a, &pk)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    results.push(BenchStats::from_times("Encrypt", &times));
    println!(" done ({:.3}ms avg)", results.last().unwrap().avg_ms);

    // ------------------------------------------------------------------------
    // 3. DECRYPT
    // ------------------------------------------------------------------------
    print!("Benchmarking DECRYPT...");
    let ct_a = ctx.encrypt(&pt_a, &pk)?;
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = ctx.decrypt(&ct_a, &sk);
    }

    // Benchmark
    for _ in 0..BENCHMARK_ITERATIONS {
        let start = Instant::now();
        let _ = ctx.decrypt(&ct_a, &sk);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    results.push(BenchStats::from_times("Decrypt", &times));
    println!(" done ({:.3}ms avg)", results.last().unwrap().avg_ms);

    // ------------------------------------------------------------------------
    // 4. DECODE
    // ------------------------------------------------------------------------
    print!("Benchmarking DECODE...");
    let pt_dec = ctx.decrypt(&ct_a, &sk)?;
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = ctx.decode(&pt_dec);
    }

    // Benchmark
    for _ in 0..BENCHMARK_ITERATIONS {
        let start = Instant::now();
        let _ = ctx.decode(&pt_dec);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    results.push(BenchStats::from_times("Decode", &times));
    println!(" done ({:.3}ms avg)", results.last().unwrap().avg_ms);

    // ------------------------------------------------------------------------
    // 5. ADD (Ciphertext + Ciphertext)
    // ------------------------------------------------------------------------
    print!("Benchmarking ADD...");
    let pt_b = CudaPlaintext::encode(&values_b, scale, &params);
    let ct_b = ctx.encrypt(&pt_b, &pk)?;
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = ct_a.add(&ct_b, &ctx)?;
    }

    // Benchmark
    for _ in 0..BENCHMARK_ITERATIONS {
        let start = Instant::now();
        let _ = ct_a.add(&ct_b, &ctx)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    results.push(BenchStats::from_times("Add (ct+ct)", &times));
    println!(" done ({:.3}ms avg)", results.last().unwrap().avg_ms);

    // ------------------------------------------------------------------------
    // 6. SUBTRACT (Ciphertext - Ciphertext)
    // ------------------------------------------------------------------------
    print!("Benchmarking SUBTRACT...");
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = ct_a.subtract(&ct_b, &ctx)?;
    }

    // Benchmark
    for _ in 0..BENCHMARK_ITERATIONS {
        let start = Instant::now();
        let _ = ct_a.subtract(&ct_b, &ctx)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    results.push(BenchStats::from_times("Subtract (ct-ct)", &times));
    println!(" done ({:.3}ms avg)", results.last().unwrap().avg_ms);

    // ------------------------------------------------------------------------
    // 7. MULTIPLY (Ciphertext * Ciphertext) with relin + rescale
    // ------------------------------------------------------------------------
    print!("Benchmarking MULTIPLY (ct*ct)...");
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = multiply_ciphertexts_gpu(&ct_a, &ct_b, &relin_keys, &ctx)?;
    }

    // Benchmark
    for _ in 0..BENCHMARK_ITERATIONS {
        let start = Instant::now();
        let _ = multiply_ciphertexts_gpu(&ct_a, &ct_b, &relin_keys, &ctx)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    results.push(BenchStats::from_times("Multiply (ct*ct)", &times));
    println!(" done ({:.3}ms avg)", results.last().unwrap().avg_ms);

    // ------------------------------------------------------------------------
    // 8. MULTIPLY PLAIN (Ciphertext * Plaintext)
    // ------------------------------------------------------------------------
    print!("Benchmarking MULTIPLY PLAIN (ct*pt)...");
    let pt_const = CudaPlaintext::encode(&plain_const, scale, &params);
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = ct_a.multiply_plain(&pt_const, &ctx)?;
    }

    // Benchmark
    for _ in 0..BENCHMARK_ITERATIONS {
        let start = Instant::now();
        let _ = ct_a.multiply_plain(&pt_const, &ctx)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    results.push(BenchStats::from_times("Multiply Plain (ct*pt)", &times));
    println!(" done ({:.3}ms avg)", results.last().unwrap().avg_ms);

    // ------------------------------------------------------------------------
    // 9. ROTATE (by 1 slot)
    // ------------------------------------------------------------------------
    print!("Benchmarking ROTATE (1 slot)...");
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = ct_a.rotate_by_steps(1, &rot_keys, &ctx)?;
    }

    // Benchmark
    for _ in 0..BENCHMARK_ITERATIONS {
        let start = Instant::now();
        let _ = ct_a.rotate_by_steps(1, &rot_keys, &ctx)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    results.push(BenchStats::from_times("Rotate (1 slot)", &times));
    println!(" done ({:.3}ms avg)", results.last().unwrap().avg_ms);

    // ------------------------------------------------------------------------
    // 10. ROTATE (by 8 slots)
    // ------------------------------------------------------------------------
    print!("Benchmarking ROTATE (8 slots)...");
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = ct_a.rotate_by_steps(8, &rot_keys, &ctx)?;
    }

    // Benchmark
    for _ in 0..BENCHMARK_ITERATIONS {
        let start = Instant::now();
        let _ = ct_a.rotate_by_steps(8, &rot_keys, &ctx)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    results.push(BenchStats::from_times("Rotate (8 slots)", &times));
    println!(" done ({:.3}ms avg)", results.last().unwrap().avg_ms);

    // ------------------------------------------------------------------------
    // 11. RESCALE
    // ------------------------------------------------------------------------
    print!("Benchmarking RESCALE...");
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = ct_a.rescale_to_next(&ctx)?;
    }

    // Benchmark
    for _ in 0..BENCHMARK_ITERATIONS {
        let start = Instant::now();
        let _ = ct_a.rescale_to_next(&ctx)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    results.push(BenchStats::from_times("Rescale", &times));
    println!(" done ({:.3}ms avg)", results.last().unwrap().avg_ms);

    // ------------------------------------------------------------------------
    // 12. MOD SWITCH
    // ------------------------------------------------------------------------
    print!("Benchmarking MOD SWITCH...");
    let mut times = Vec::new();
    let target_level = ct_a.level - 1;

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = ct_a.mod_switch_to_level(target_level);
    }

    // Benchmark
    for _ in 0..BENCHMARK_ITERATIONS {
        let start = Instant::now();
        let _ = ct_a.mod_switch_to_level(target_level);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    results.push(BenchStats::from_times("Mod Switch", &times));
    println!(" done ({:.3}ms avg)", results.last().unwrap().avg_ms);

    // ========================================================================
    // RESULTS SUMMARY
    // ========================================================================
    println!();
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                      BENCHMARK RESULTS                                 ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("┌─────────────────────────┬───────────┬───────────┬───────────┬───────────┐");
    println!("│ Operation               │  Avg (ms) │  Min (ms) │  Max (ms) │  Std (ms) │");
    println!("├─────────────────────────┼───────────┼───────────┼───────────┼───────────┤");

    for stat in &results {
        println!(
            "│ {:<23} │ {:>9.3} │ {:>9.3} │ {:>9.3} │ {:>9.3} │",
            stat.name, stat.avg_ms, stat.min_ms, stat.max_ms, stat.std_ms
        );
    }

    println!("└─────────────────────────┴───────────┴───────────┴───────────┴───────────┘");
    println!();

    // Print summary stats
    println!("Configuration:");
    println!("  Ring dimension (N): {}", n);
    println!("  Number of slots: {}", slots);
    println!("  Number of primes: {}", num_primes);
    println!("  GPU: NVIDIA (via CUDA)");
    println!();

    // Calculate throughput for key operations
    let mult_time_ms = results.iter().find(|r| r.name == "Multiply (ct*ct)").unwrap().avg_ms;
    let add_time_ms = results.iter().find(|r| r.name == "Add (ct+ct)").unwrap().avg_ms;
    let rotate_time_ms = results.iter().find(|r| r.name == "Rotate (1 slot)").unwrap().avg_ms;

    println!("Throughput (operations per second):");
    println!("  Multiplications: {:.1} ops/sec", 1000.0 / mult_time_ms);
    println!("  Additions:       {:.1} ops/sec", 1000.0 / add_time_ms);
    println!("  Rotations:       {:.1} ops/sec", 1000.0 / rotate_time_ms);
    println!();

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                      BENCHMARK COMPLETE                                ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This benchmark requires CUDA GPU support.");
    println!("Please run with: cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda --example bench_cuda_all_ops");
}
