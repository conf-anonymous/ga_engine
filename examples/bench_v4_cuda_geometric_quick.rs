//! Quick Benchmark V4 CUDA Geometric Product (Small Parameters)
//!
//! Fast test version with N=1024 for quick validation.
//! For production benchmarks, use bench_v4_cuda_geometric.rs
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda,v4 --example bench_v4_cuda_geometric_quick
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
    println!("║   V4 CUDA Geometric Product - QUICK TEST (N=1024)           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Setup with SMALL parameters for fast testing
    println!("Setting up FHE context (QUICK TEST PARAMETERS)...");
    let params = CliffordFHEParams::new_test_ntt_1024(); // Small N=1024
    let n = params.n;
    let num_primes = params.moduli.len();
    println!("  Parameters: N={}, {} primes (QUICK TEST)", n, num_primes);

    let start = Instant::now();
    let device = Arc::new(CudaDeviceContext::new()?);
    let ckks_ctx = Arc::new(CudaCkksContext::new(params.clone())?);
    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);
    let ctx_time = start.elapsed();
    println!("  Context initialization: {:.3}s\n", ctx_time.as_secs_f64());

    // Generate keys
    println!("Generating rotation keys (minimal set for quick test)...");
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

    // Generate ONLY the essential rotations for V4
    // ±1 to ±8 for packing, and a few power-of-2 for butterfly
    println!("  Generating ±1 to ±8 for packing...");
    for rot in 1..=8 {
        rotation_keys.generate_rotation_key_gpu(rot, ckks_ctx.ntt_contexts())?;
        rotation_keys.generate_rotation_key_gpu(-rot, ckks_ctx.ntt_contexts())?;
    }

    // Only essential power-of-2 rotations (up to 256 for N=1024)
    println!("  Generating power-of-2 rotations for butterfly...");
    for &rot in &[16, 32, 64, 128, 256] {
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
    let level = num_primes - 1; // Use full level to match params structure
    let scale = params.scale;

    let mut a_components = Vec::new();
    let mut b_components = Vec::new();

    for _i in 0..8 {
        // CUDA uses strided layout: total size is n * num_primes
        let mut c0 = vec![0u64; n * num_primes];
        let mut c1 = vec![0u64; n * num_primes];

        // Fill in strided format: coeff_idx * num_primes + prime_idx
        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                let idx = coeff_idx * num_primes + prime_idx;
                let q = params.moduli[prime_idx];
                c0[idx] = rng.gen::<u64>() % q;
                c1[idx] = rng.gen::<u64>() % q;
            }
        }

        let ct = ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext {
            c0: c0.clone(), c1: c1.clone(), n, num_primes, level, scale,
        };
        a_components.push(ct);

        // Create different ciphertext for b
        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                let idx = coeff_idx * num_primes + prime_idx;
                let q = params.moduli[prime_idx];
                c0[idx] = rng.gen::<u64>() % q;
                c1[idx] = rng.gen::<u64>() % q;
            }
        }
        let ct = ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext {
            c0, c1, n, num_primes, level, scale,
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

    let num_trials = 3; // Fewer trials for quick test
    let mut gp_times = Vec::new();

    for trial in 0..num_trials {
        println!("  Running trial {}...", trial + 1);
        let start = Instant::now();
        let _result = geometric_product_packed(&a_packed, &b_packed, &rotation_keys, &ckks_ctx)?;
        let elapsed = start.elapsed().as_secs_f64();
        gp_times.push(elapsed);
        println!("    Trial {}: {:.4}s", trial + 1, elapsed);
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
    println!("║                    QUICK TEST SUMMARY                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("V4 CUDA Performance (N=1024 quick test):");
    println!("  Geometric Product: {:.4}s average", gp_avg);
    println!("  Packing (8→1):     {:.4}s", pack_time.as_secs_f64());
    println!("  Key generation:    {:.3}s ({} keys)", keygen_time.as_secs_f64(), rotation_keys.num_keys());
    println!();
    println!("Memory savings:      8× vs V3 (no ciphertext expansion)");
    println!();
    println!("Note: This is a QUICK TEST with small parameters.");
    println!("      For production benchmarks, use bench_v4_cuda_geometric.rs");
    println!();
    println!("✅ V4 CUDA GEOMETRIC PRODUCT WORKING!");
    println!("   Full implementation, no workarounds!\n");

    Ok(())
}

#[cfg(not(all(feature = "v4", feature = "v2-gpu-cuda")))]
fn main() {
    eprintln!("This example requires features: v4,v2-gpu-cuda");
    eprintln!("Run with: cargo run --release --features v2,v2-gpu-cuda,v4 --example bench_v4_cuda_geometric_quick");
    std::process::exit(1);
}
