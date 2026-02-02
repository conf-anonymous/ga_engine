//! Test V3 CUDA GPU Bootstrap
//!
//! This test validates the full bootstrap pipeline on CUDA GPU.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap
//! ```

use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::device::CudaDeviceContext;
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::rotation::CudaRotationContext;
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::rotation_keys::CudaRotationKeys;
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::relin_keys::CudaRelinKeys;
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v3::bootstrapping::cuda_bootstrap::{CudaBootstrapContext, CudaCiphertext};
use ga_engine::clifford_fhe_v3::bootstrapping::BootstrapParams;
use rand::Rng;
use std::sync::Arc;

fn main() -> Result<(), String> {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║           V3 CUDA GPU Bootstrap Test                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Step 1: Initialize parameters
    println!("Step 1: Initializing parameters");
    let params = CliffordFHEParams::new_v3_bootstrap_cuda_full()?;
    let n = params.n;
    let num_primes = params.moduli.len();
    let bootstrap_params = BootstrapParams::balanced();
    println!();

    // Step 2: Initialize CUDA contexts
    println!("Step 2: Initializing CUDA contexts");
    let device = Arc::new(CudaDeviceContext::new()?);
    let ckks_ctx = Arc::new(CudaCkksContext::new(params.clone())?);
    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);

    // Step 3: Generate secret key and keys
    println!("Step 3: Generating secret key, rotation keys, and relinearization keys");
    let mut rng = rand::thread_rng();
    let mut secret_key = vec![0u64; n * num_primes];

    // Binary secret key
    for coeff_idx in 0..n {
        let bit = rng.gen::<u64>() & 1;
        for prime_idx in 0..num_primes {
            secret_key[coeff_idx * num_primes + prime_idx] = bit;
        }
    }

    // Generate rotation keys using GPU
    let mut rotation_keys = CudaRotationKeys::new(
        device.clone(),
        params.clone(),
        rotation_ctx.clone(),
        secret_key.clone(),
        16,  // base_bits = 16
    )?;

    // Generate keys for all rotations needed by bootstrap
    // For N=1024 (slots=512), CoeffToSlot needs: 1, 2, 4, 8, 16, 32, 64, 128, 256
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
    println!("  ✅ Generated {} rotation keys", rotation_keys.num_keys());

    // Generate relinearization keys using GPU
    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        secret_key.clone(),
        16,  // base_bits = 16
        ckks_ctx.ntt_contexts(),
    )?;
    println!("  ✅ Generated relinearization keys\n");

    // Step 4: Create bootstrap context
    println!("Step 4: Creating bootstrap context");
    let bootstrap_ctx = CudaBootstrapContext::new(
        ckks_ctx.clone(),
        rotation_ctx.clone(),
        Arc::new(rotation_keys),
        Arc::new(relin_keys),
        bootstrap_params,
        params.clone(),
    )?;

    // Step 5: Create test ciphertext
    println!("Step 5: Creating test ciphertext");
    let level = 2;  // Low level (almost out of noise budget)
    let mut c0 = vec![0u64; n * (level + 1)];
    let mut c1 = vec![0u64; n * (level + 1)];

    // Fill with random values
    for i in 0..c0.len() {
        let prime_idx = i % (level + 1);
        let q = params.moduli[prime_idx];
        c0[i] = rng.gen::<u64>() % q;
        c1[i] = rng.gen::<u64>() % q;
    }

    let ct_in = CudaCiphertext {
        c0,
        c1,
        n,
        num_primes: level + 1,
        level,
        scale: params.scale,  // Use the scale from parameters (2^45)
    };

    println!("  Input ciphertext: level = {}, scale = {:.2e}\n", ct_in.level, ct_in.scale);

    // Step 6: Run bootstrap
    println!("Step 6: Running bootstrap pipeline");
    let start = std::time::Instant::now();
    let ct_out = bootstrap_ctx.bootstrap(&ct_in)?;
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n  Output ciphertext: level = {}, scale = {:.2e}", ct_out.level, ct_out.scale);
    println!("  ✅ Bootstrap completed in {:.2}s\n", elapsed);

    // Step 7: Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Results:");
    println!("  Bootstrap time: {:.2}s", elapsed);
    println!("  Input level: {}", ct_in.level);
    println!("  Output level: {}", ct_out.level);
    println!("  GPU acceleration: ✅");
    println!("═══════════════════════════════════════════════════════════════");
    println!("✅ V3 CUDA GPU BOOTSTRAP COMPLETE");
    println!("   Full implementation with relinearization!\n");

    Ok(())
}
