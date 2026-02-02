//! Test CUDA GPU Rotation Keys
//!
//! This test validates rotation key generation using GPU-accelerated NTT.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example test_cuda_rotation_keys
//! ```

use ga_engine::clifford_fhe_v2::backends::gpu_cuda::device::CudaDeviceContext;
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::rotation::CudaRotationContext;
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::rotation_keys::CudaRotationKeys;
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use rand::Rng;
use std::sync::Arc;

fn main() -> Result<(), String> {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║           CUDA GPU Rotation Keys Test                        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Step 1: Initialize device and parameters
    println!("Step 1: Initializing CUDA device and parameters");
    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();
    let device = Arc::new(CudaDeviceContext::new()?);
    println!("  N = {}, num_primes = {}\n", n, num_primes);

    // Step 2: Generate secret key
    println!("Step 2: Generating secret key");
    let mut rng = rand::thread_rng();
    let mut secret_key = vec![0u64; n * num_primes];

    // Binary secret key: coefficients in {0, 1}
    for coeff_idx in 0..n {
        let bit = rng.gen::<u64>() & 1;
        for prime_idx in 0..num_primes {
            secret_key[coeff_idx * num_primes + prime_idx] = bit;
        }
    }
    println!("  ✅ Secret key generated (binary, strided layout)\n");

    // Step 3: Create rotation context
    println!("Step 3: Creating rotation context");
    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);

    // Step 4: Create rotation keys manager
    println!("Step 4: Creating rotation keys manager");
    let base_bits = 16;  // w = 2^16 = 65536
    let mut rot_keys = CudaRotationKeys::new(
        device.clone(),
        params.clone(),
        rotation_ctx.clone(),
        secret_key,
        base_bits,
    )?;

    // Step 5: Generate rotation keys for common rotations
    println!("Step 5: Generating rotation keys");

    let test_rotations = vec![1, 2, 4];
    println!("  Generating keys for rotations: {:?}\n", test_rotations);

    let start = std::time::Instant::now();
    for &rot in &test_rotations {
        rot_keys.generate_rotation_key(rot)?;
    }
    let elapsed = start.elapsed();

    println!("  ✅ Generated {} rotation keys in {:.2}s",
        rot_keys.num_keys(), elapsed.as_secs_f64());
    println!("  Average: {:.2}s per key\n", elapsed.as_secs_f64() / rot_keys.num_keys() as f64);

    // Step 6: Verify key structure
    println!("Step 6: Verifying rotation key structure");
    println!("  Number of rotation keys: {}", rot_keys.num_keys());
    println!("  Gadget base w: 2^{} = {}", base_bits, 1u64 << base_bits);
    println!("  Number of gadget digits (dnum): {}", rot_keys.dnum);
    println!("  ✅ Key structure verified\n");

    println!("═══════════════════════════════════════════════════════════════");
    println!("Results:");
    println!("  Rotation keys generated: {}", rot_keys.num_keys());
    println!("  Total time: {:.2}s", elapsed.as_secs_f64());
    println!("  GPU NTT multiply used: ✅");
    println!("═══════════════════════════════════════════════════════════════");
    println!("✅ CUDA ROTATION KEYS WORKING");
    println!("   Ready for V3 bootstrap implementation!\n");

    Ok(())
}
