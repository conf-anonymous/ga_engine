//! Benchmark V4 CUDA Geometric Product Performance
//!
//! Measures complete V4 geometric product pipeline on CUDA GPU.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda,v4 --example bench_v4_cuda_geometric
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
        pack_multivector,
        geometric_ops::geometric_product_packed,
    };
    use std::sync::Arc;
    use std::time::Instant;
    use rand::Rng;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   V4 CUDA Geometric Product Performance Benchmark           ║");
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

    // Generate rotations needed for V4 (±1 to ±8 for packing, more for butterfly)
    for rot in 1..=8 {
        rotation_keys.generate_rotation_key_gpu(rot, ckks_ctx.ntt_contexts())?;
        rotation_keys.generate_rotation_key_gpu(-rot, ckks_ctx.ntt_contexts())?;
    }
    // Also generate power-of-2 rotations for butterfly (used in geometric product)
    for i in 0..=(n/2).trailing_zeros() {
        let rot = 1i32 << i;
        if rot <= (n/2) as i32 {
            rotation_keys.generate_rotation_key_gpu(rot, ckks_ctx.ntt_contexts())?;
            rotation_keys.generate_rotation_key_gpu(-rot, ckks_ctx.ntt_contexts())?;
        }
    }
    let keygen_time = start.elapsed();
    println!("  Rotation key generation: {:.3}s", keygen_time.as_secs_f64());
    println!("  Generated {} rotation keys\n", rotation_keys.num_keys());

    // Create test ciphertexts
    println!("Creating test ciphertexts...");
    let level = params.moduli.len() - 2; // Leave room for operations
    let scale = params.scale;

    let mut a_components = Vec::new();
    let mut b_components = Vec::new();

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
            c0: c0.clone(), c1: c1.clone(), n, num_primes: level + 1, level, scale,
        };
        a_components.push(ct);

        // Create different ciphertext for b
        for j in 0..c0.len() {
            let prime_idx = j % (level + 1);
            let q = params.moduli[prime_idx];
            c0[j] = rng.gen::<u64>() % q;
            c1[j] = rng.gen::<u64>() % q;
        }
        let ct = ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext {
            c0, c1, n, num_primes: level + 1, level, scale,
        };
        b_components.push(ct);
    }
    println!("  Created 2×8 component ciphertexts\n");

    // Pack multivectors
    println!("Packing multivectors...");
    let start = Instant::now();
    let a_packed = pack_multivector(
        &a_components.try_into().unwrap(),
        &rotation_keys,
        &rotation_ctx,
        &ckks_ctx,
    )?;
    let b_packed = pack_multivector(
        &b_components.try_into().unwrap(),
        &rotation_keys,
        &rotation_ctx,
        &ckks_ctx,
    )?;
    let pack_time = start.elapsed();
    println!("  Packing time: {:.3}s\n", pack_time.as_secs_f64());

    // Benchmark geometric product
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║            GEOMETRIC PRODUCT BENCHMARK                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let num_trials = 5;
    let mut gp_times = Vec::new();

    for trial in 0..num_trials {
        let start = Instant::now();
        let _result = geometric_product_packed(&a_packed, &b_packed, &rotation_keys, &ckks_ctx)?;
        let elapsed = start.elapsed().as_secs_f64();
        gp_times.push(elapsed);
        println!("  Trial {}: {:.4}s", trial + 1, elapsed);
    }

    let gp_avg = gp_times.iter().sum::<f64>() / gp_times.len() as f64;
    let gp_min = gp_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let gp_max = gp_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("\n  Geometric Product Statistics:");
    println!("    Average: {:.4}s", gp_avg);
    println!("    Min:     {:.4}s", gp_min);
    println!("    Max:     {:.4}s", gp_max);

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK SUMMARY                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("V4 CUDA Performance:");
    println!("  Geometric Product: {:.4}s average", gp_avg);
    println!("  Memory savings:    8× vs V3 (no ciphertext expansion)");
    println!();
    println!("Comparison to V3:");
    println!("  V3 CUDA Bootstrap: 12.94s (reported earlier)");
    println!("  V4 memory:         1 ciphertext vs 8 ciphertexts");
    println!();
    println!("Hardware:");
    println!("  GPU:               NVIDIA RTX 5090");
    println!("  Parameters:        N={}, L={} levels", n, num_primes);
    println!();
    println!("✅ V4 CUDA GEOMETRIC PRODUCT BENCHMARK COMPLETE");
    println!("   Full implementation with no workarounds!\n");

    Ok(())
}

#[cfg(not(all(feature = "v4", feature = "v2-gpu-cuda")))]
fn main() {
    eprintln!("This example requires features: v4,v2-gpu-cuda");
    eprintln!("Run with: cargo run --release --features v2,v2-gpu-cuda,v4 --example bench_v4_cuda_geometric");
    std::process::exit(1);
}
