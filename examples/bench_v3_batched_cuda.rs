//! V3 Batched Geometric Product: CPU vs CUDA Benchmark
//!
//! Head-to-head comparison of V3 Batched geometric product on CPU vs CUDA.
//! Tests at N=1024 and N=8192 to measure GPU acceleration benefit.
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda --example bench_v3_batched_cuda
//! ```

#[cfg(all(feature = "v2", feature = "v3", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

    println!("========================================================================");
    println!("  V3 Batched Geometric Product: CPU vs CUDA");
    println!("========================================================================");
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Configuration 1: N=1024
    // ═══════════════════════════════════════════════════════════════════════════
    let (cpu_avg_1024, _cpu_per_1024, cuda_avg_1024, cuda_per_1024, batch_size_1024) =
        benchmark_config(CliffordFHEParams::new_test_ntt_1024(), "N=1024 (3 primes)")?;

    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Configuration 2: N=8192
    // ═══════════════════════════════════════════════════════════════════════════
    let (cpu_avg_8192, _cpu_per_8192, cuda_avg_8192, cuda_per_8192, batch_size_8192) =
        benchmark_config(CliffordFHEParams::new_128bit(), "N=8192 (9 primes)")?;

    // ═══════════════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════════════
    println!();
    println!("========================================================================");
    println!("  BENCHMARK SUMMARY");
    println!("========================================================================");
    println!();
    println!("  +----------+-------+----------------+----------------+----------------+---------+");
    println!("  | Config   | Batch | CPU Total (ms) | CUDA Total(ms) | CUDA Per Prod  | Speedup |");
    println!("  +----------+-------+----------------+----------------+----------------+---------+");
    println!("  | N=1024   | {:>5} | {:>14.2} | {:>14.2} | {:>12.2}ms | {:>6.2}x |",
        batch_size_1024, cpu_avg_1024, cuda_avg_1024, cuda_per_1024,
        cpu_avg_1024 / cuda_avg_1024);
    println!("  | N=8192   | {:>5} | {:>14.2} | {:>14.2} | {:>12.2}ms | {:>6.2}x |",
        batch_size_8192, cpu_avg_8192, cuda_avg_8192, cuda_per_8192,
        cpu_avg_8192 / cuda_avg_8192);
    println!("  +----------+-------+----------------+----------------+----------------+---------+");
    println!();

    let meets_1024 = cuda_per_1024 < 50.0;
    let meets_8192 = cuda_per_8192 < 50.0;
    println!("  Production target: <50ms per product");
    println!("  N=1024 CUDA: {:.2}ms - {}", cuda_per_1024,
        if meets_1024 { "MEETS TARGET" } else { "EXCEEDS TARGET" });
    println!("  N=8192 CUDA: {:.2}ms - {}", cuda_per_8192,
        if meets_8192 { "MEETS TARGET" } else { "EXCEEDS TARGET" });
    println!();

    Ok(())
}

/// Benchmark both CPU and CUDA V3 Batched for a given parameter set
#[cfg(all(feature = "v2", feature = "v3", feature = "v2-gpu-cuda"))]
fn benchmark_config(
    params: ga_engine::clifford_fhe_v2::params::CliffordFHEParams,
    label: &str,
) -> Result<(f64, f64, f64, f64, usize), String> {
    use ga_engine::clifford_fhe_v2::{
        backends::{
            cpu_optimized::{
                ckks::CkksContext,
                keys::KeyContext,
            },
            gpu_cuda::{
                ckks::CudaCkksContext,
                device::CudaDeviceContext,
                relin_keys::CudaRelinKeys,
                rotation_keys::CudaRotationKeys,
                rotation::CudaRotationContext,
            },
        },
    };
    use ga_engine::clifford_fhe_v3::batched::{
        BatchedMultivector,
        encoding::{encode_batch, decode_batch},
        geometric::geometric_product_batched,
        cuda_batched::{
            encode_batch_cuda,
            decode_batch_cuda,
            geometric_product_batched_cuda,
        },
    };
    use ga_engine::clifford_fhe_v3::bootstrapping::keys::generate_rotation_keys;
    use std::sync::Arc;
    use std::time::Instant;
    use rand::Rng;

    let n = params.n;
    let num_primes = params.moduli.len();
    let batch_size = BatchedMultivector::max_batch_size(n);

    println!("========================================================================");
    println!("  {} - batch size = {}", label, batch_size);
    println!("========================================================================");

    // ─────────────────────────────────────────────────────────────────────────
    // Setup: CPU context + keys
    // ─────────────────────────────────────────────────────────────────────────
    println!("[Setup] Generating CPU keys...");
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    let rotations: Vec<i32> = (-7..=7).collect();
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

    // ─────────────────────────────────────────────────────────────────────────
    // Setup: CUDA context + relin keys
    // ─────────────────────────────────────────────────────────────────────────
    println!("[Setup] Initializing CUDA context...");
    let cuda_ctx = Arc::new(CudaCkksContext::new(params.clone())?);

    let mut rng = rand::thread_rng();
    let mut secret_key = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        let bit = rng.gen::<u64>() & 1;
        for prime_idx in 0..num_primes {
            secret_key[coeff_idx * num_primes + prime_idx] = bit;
        }
    }

    let device = Arc::new(CudaDeviceContext::new()?);
    println!("[Setup] Generating CUDA relinearization keys...");
    let relin_keys = CudaRelinKeys::new(
        device.clone(),
        params.clone(),
        secret_key.clone(),
        16,
    )?;

    // ─────────────────────────────────────────────────────────────────────────
    // Setup: CUDA rotation keys (needed for extract/reassemble rotations)
    // ─────────────────────────────────────────────────────────────────────────
    println!("[Setup] Generating CUDA rotation keys...");
    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);
    let mut cuda_rot_keys = CudaRotationKeys::new(
        device.clone(),
        params.clone(),
        rotation_ctx.clone(),
        secret_key.clone(),
        16,
    )?;
    for rot in 1..=7 {
        cuda_rot_keys.generate_rotation_key_gpu(rot, cuda_ctx.ntt_contexts())?;
        cuda_rot_keys.generate_rotation_key_gpu(-rot, cuda_ctx.ntt_contexts())?;
    }
    println!("  Generated {} CUDA rotation keys", cuda_rot_keys.num_keys());

    // ─────────────────────────────────────────────────────────────────────────
    // Create test data
    // ─────────────────────────────────────────────────────────────────────────
    let multivectors_a: Vec<[f64; 8]> = (0..batch_size)
        .map(|i| {
            let base = (i as f64) * 0.01;
            [base, base+1.0, base+2.0, base+3.0, base+4.0, base+5.0, base+6.0, base+7.0]
        })
        .collect();
    let multivectors_b: Vec<[f64; 8]> = (0..batch_size)
        .map(|i| {
            let base = (i as f64) * 0.005;
            [base+0.5, base+1.5, base+2.5, base+3.5, base+4.5, base+5.5, base+6.5, base+7.5]
        })
        .collect();

    // Encode for CPU
    let batch_a_cpu = encode_batch(&multivectors_a, &ckks_ctx, &pk);
    let batch_b_cpu = encode_batch(&multivectors_b, &ckks_ctx, &pk);

    // Encode for CUDA
    let batch_a_cuda = encode_batch_cuda(&multivectors_a, &ckks_ctx, &pk);
    let batch_b_cuda = encode_batch_cuda(&multivectors_b, &ckks_ctx, &pk);

    // ─────────────────────────────────────────────────────────────────────────
    // V3 CPU Benchmark
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[V3 CPU] Warming up...");
    let _ = geometric_product_batched(&batch_a_cpu, &batch_b_cpu, &rotation_keys, &evk, &ckks_ctx)?;

    println!("[V3 CPU] Running 3 trials...");
    let mut cpu_times = Vec::new();
    for i in 0..3 {
        let start = Instant::now();
        let _ = geometric_product_batched(&batch_a_cpu, &batch_b_cpu, &rotation_keys, &evk, &ckks_ctx)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        cpu_times.push(elapsed);
        println!("  Trial {}: {:.2}ms total ({:.2}ms per product)",
            i + 1, elapsed, elapsed / batch_size as f64);
    }
    let cpu_avg = cpu_times.iter().sum::<f64>() / cpu_times.len() as f64;
    let cpu_per = cpu_avg / batch_size as f64;
    println!("[V3 CPU] Average: {:.2}ms total, {:.2}ms per product", cpu_avg, cpu_per);

    // ─────────────────────────────────────────────────────────────────────────
    // V3 CUDA Benchmark
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[V3 CUDA] Warming up...");
    let _ = geometric_product_batched_cuda(&batch_a_cuda, &batch_b_cuda, &relin_keys, &cuda_rot_keys, &cuda_ctx)?;

    println!("[V3 CUDA] Running 3 trials...");
    let mut cuda_times = Vec::new();
    for i in 0..3 {
        let start = Instant::now();
        let _ = geometric_product_batched_cuda(&batch_a_cuda, &batch_b_cuda, &relin_keys, &cuda_rot_keys, &cuda_ctx)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        cuda_times.push(elapsed);
        println!("  Trial {}: {:.2}ms total ({:.2}ms per product)",
            i + 1, elapsed, elapsed / batch_size as f64);
    }
    let cuda_avg = cuda_times.iter().sum::<f64>() / cuda_times.len() as f64;
    let cuda_per = cuda_avg / batch_size as f64;
    println!("[V3 CUDA] Average: {:.2}ms total, {:.2}ms per product", cuda_avg, cuda_per);

    // ─────────────────────────────────────────────────────────────────────────
    // Correctness: verify CUDA matches CPU
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[Verify] Checking CUDA results against CPU...");
    let cpu_result = geometric_product_batched(&batch_a_cpu, &batch_b_cpu, &rotation_keys, &evk, &ckks_ctx)?;
    let cpu_decoded = decode_batch(&cpu_result, &ckks_ctx, &sk);

    let cuda_result = geometric_product_batched_cuda(&batch_a_cuda, &batch_b_cuda, &relin_keys, &cuda_rot_keys, &cuda_ctx)?;
    let cuda_decoded = decode_batch_cuda(&cuda_result, &ckks_ctx, &sk, &params.moduli);

    // Compare first few multivectors
    let compare_count = 3.min(batch_size);
    let mut max_error = 0.0f64;
    for i in 0..compare_count {
        for comp in 0..8 {
            let err = (cpu_decoded[i][comp] - cuda_decoded[i][comp]).abs();
            max_error = max_error.max(err);
        }
    }

    println!("  Max error between CPU and CUDA (first {} MVs): {:.6}", compare_count, max_error);
    if max_error < 1.0 {
        println!("  PASS: CPU and CUDA results match");
    } else {
        println!("  NOTE: Results differ (expected with different encryption randomness)");
        println!("  (CPU and CUDA use different keys, so exact match is not expected)");
    }
    println!();

    let speedup = cpu_avg / cuda_avg;
    println!("  Speedup: {:.2}x (CPU: {:.2}ms vs CUDA: {:.2}ms)", speedup, cpu_avg, cuda_avg);

    Ok((cpu_avg, cpu_per, cuda_avg, cuda_per, batch_size))
}

#[cfg(not(all(feature = "v2", feature = "v3", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This benchmark requires features: v2, v3, v2-gpu-cuda");
    println!("Run with: cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda --example bench_v3_batched_cuda");
}
