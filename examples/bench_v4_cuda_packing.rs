//! Benchmark V4 CUDA Packing/Unpacking Performance
//!
//! Measures timing for V4 packing operations on CUDA GPU.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda,v4 --example bench_v4_cuda_packing
//! ```

#[cfg(all(feature = "v4", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::backends::gpu_cuda::{
        ckks::CudaCkksContext,
        device::CudaDeviceContext,
        rotation::CudaRotationContext,
        rotation_keys::CudaRotationKeys,
    };
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v4::{
        pack_multivector, unpack_multivector,
    };
    use std::sync::Arc;
    use std::time::Instant;
    use rand::Rng;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     V4 CUDA Packing/Unpacking Performance Benchmark         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Setup
    println!("Setting up FHE context...");
    let params = CliffordFHEParams::default();
    let n = params.n;
    let num_primes = params.moduli.len();
    println!("  Parameters: N={}, {} primes", n, num_primes);

    let start = Instant::now();
    let device = Arc::new(CudaDeviceContext::new()?);
    let ckks_ctx = Arc::new(CudaCkksContext::new(params.clone())?);
    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);
    let ctx_time = start.elapsed();
    println!("  Context initialization: {:.3}s\n", ctx_time.as_secs_f64());

    // Generate keys
    println!("Generating rotation keys...");
    let mut rng = rand::thread_rng();
    let mut secret_key = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        let bit = rng.gen::<u64>() & 1;
        for prime_idx in 0..num_primes {
            secret_key[coeff_idx * num_primes + prime_idx] = bit;
        }
    }

    let start = Instant::now();
    let mut rotation_keys = CudaRotationKeys::new(
        device.clone(),
        params.clone(),
        rotation_ctx.clone(),
        secret_key.clone(),
        16,
    )?;

    for rot in 1..=8 {
        rotation_keys.generate_rotation_key_gpu(rot, ckks_ctx.ntt_contexts())?;
        rotation_keys.generate_rotation_key_gpu(-rot, ckks_ctx.ntt_contexts())?;
    }
    let keygen_time = start.elapsed();
    println!("  Rotation key generation: {:.3}s", keygen_time.as_secs_f64());
    println!("  Generated {} rotation keys\n", rotation_keys.num_keys());

    // Create test ciphertexts
    println!("Creating test ciphertexts...");
    let level = params.moduli.len() - 2; // Leave room for operations
    let scale = params.scale;

    let mut components = Vec::new();
    for i in 0..8 {
        let mut c0 = vec![0u64; n * (level + 1)];
        let mut c1 = vec![0u64; n * (level + 1)];

        for j in 0..c0.len() {
            let prime_idx = j % (level + 1);
            let q = params.moduli[prime_idx];
            c0[j] = rng.gen::<u64>() % q;
            c1[j] = rng.gen::<u64>() % q;
        }

        let ct = ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext {
            c0, c1, n,
            num_primes: level + 1,
            level,
            scale,
        };
        components.push(ct);
    }
    println!("  Created 8 component ciphertexts\n");

    // Benchmark packing
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    PACKING BENCHMARK                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let num_trials = 10;
    let mut pack_times = Vec::new();

    for trial in 0..num_trials {
        let start = Instant::now();
        let packed = pack_multivector(
            &components.clone().try_into().unwrap(),
            &rotation_keys,
            &rotation_ctx,
            &ckks_ctx,
        )?;
        let elapsed = start.elapsed().as_secs_f64();
        pack_times.push(elapsed);
        println!("  Trial {}: {:.4}s", trial + 1, elapsed);

        // Save packed for unpacking benchmark
        if trial == 0 {
            // Benchmark unpacking
            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║                  UNPACKING BENCHMARK                         ║");
            println!("╚══════════════════════════════════════════════════════════════╝\n");

            let mut unpack_times = Vec::new();
            for unpack_trial in 0..num_trials {
                let start = Instant::now();
                let _unpacked = unpack_multivector(
                    &packed,
                    &rotation_keys,
                    &rotation_ctx,
                    &ckks_ctx,
                )?;
                let elapsed = start.elapsed().as_secs_f64();
                unpack_times.push(elapsed);
                println!("  Trial {}: {:.4}s", unpack_trial + 1, elapsed);
            }

            let unpack_avg = unpack_times.iter().sum::<f64>() / unpack_times.len() as f64;
            let unpack_min = unpack_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let unpack_max = unpack_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            println!("\n  Unpacking Statistics:");
            println!("    Average: {:.4}s", unpack_avg);
            println!("    Min:     {:.4}s", unpack_min);
            println!("    Max:     {:.4}s", unpack_max);
        }
    }

    let pack_avg = pack_times.iter().sum::<f64>() / pack_times.len() as f64;
    let pack_min = pack_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let pack_max = pack_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("\n  Packing Statistics:");
    println!("    Average: {:.4}s", pack_avg);
    println!("    Min:     {:.4}s", pack_min);
    println!("    Max:     {:.4}s", pack_max);

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK SUMMARY                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("V4 CUDA Packing Performance:");
    println!("  Pack (8→1):   {:.4}s average ({} rotations)", pack_avg, 7);
    println!("  Memory:       8× reduction vs V3 (naive layout)");
    println!();
    println!("Hardware:");
    println!("  GPU:          NVIDIA RTX 5090");
    println!("  Parameters:   N={}, L={} levels", n, num_primes);
    println!();
    println!("✅ V4 CUDA PACKING BENCHMARK COMPLETE\n");

    Ok(())
}

#[cfg(not(all(feature = "v4", feature = "v2-gpu-cuda")))]
fn main() {
    eprintln!("This example requires features: v4,v2-gpu-cuda");
    eprintln!("Run with: cargo run --release --features v2,v2-gpu-cuda,v4 --example bench_v4_cuda_packing");
    std::process::exit(1);
}
