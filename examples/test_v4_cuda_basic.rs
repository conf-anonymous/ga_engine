//! Test V4 CUDA Basic Packing/Unpacking
//!
//! This example tests basic V4 packing and unpacking on CUDA GPU.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda,v4 --example test_v4_cuda_basic
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
    use rand::Rng;

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║     V4 CUDA GPU Basic Packing/Unpacking Test                ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Step 1: Initialize parameters
    println!("Step 1: Initializing parameters");
    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();
    println!("  ✅ Ring dimension: N={}", n);
    println!("  ✅ Modulus chain: {} primes\n", num_primes);

    // Step 2: Initialize CUDA contexts
    println!("Step 2: Initializing CUDA contexts");
    let device = Arc::new(CudaDeviceContext::new()?);
    let ckks_ctx = Arc::new(CudaCkksContext::new(params.clone())?);
    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);
    println!("  ✅ CUDA device initialized");
    println!("  ✅ CKKS context ready\n");

    // Step 3: Generate secret key and rotation keys
    println!("Step 3: Generating secret key and rotation keys");
    let mut rng = rand::thread_rng();
    let mut secret_key = vec![0u64; n * num_primes];

    // Binary secret key
    for coeff_idx in 0..n {
        let bit = rng.gen::<u64>() & 1;
        for prime_idx in 0..num_primes {
            secret_key[coeff_idx * num_primes + prime_idx] = bit;
        }
    }

    // Generate rotation keys (need rotations ±1 to ±8 for packing/unpacking)
    let mut rotation_keys = CudaRotationKeys::new(
        device.clone(),
        params.clone(),
        rotation_ctx.clone(),
        secret_key.clone(),
        16,  // base_bits = 16
    )?;

    println!("  Generating rotation keys for V4 packing (rotations ±1 to ±8)");
    // Generate both positive and negative rotations
    for rot in 1..=8 {
        rotation_keys.generate_rotation_key_gpu(rot, ckks_ctx.ntt_contexts())?;
        rotation_keys.generate_rotation_key_gpu(-rot, ckks_ctx.ntt_contexts())?;
    }
    println!("  ✅ Generated {} rotation keys\n", rotation_keys.num_keys());

    // Step 4: Create test ciphertexts (8 components)
    println!("Step 4: Creating 8 test ciphertexts (one per component)");
    let level = 2;
    let scale = params.scale;

    let mut components = Vec::new();
    for i in 0..8 {
        let mut c0 = vec![0u64; n * (level + 1)];
        let mut c1 = vec![0u64; n * (level + 1)];

        // Fill with simple pattern (different for each component)
        for j in 0..c0.len() {
            let prime_idx = j % (level + 1);
            let q = params.moduli[prime_idx];
            c0[j] = ((i * 1000 + j) as u64) % q;
            c1[j] = ((i * 2000 + j) as u64) % q;
        }

        let ct = ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext {
            c0,
            c1,
            n,
            num_primes: level + 1,
            level,
            scale,
        };

        components.push(ct);
    }
    println!("  ✅ Created 8 component ciphertexts (level={}, scale={:.2e})\n", level, scale);

    // Step 5: Pack into V4 format
    println!("Step 5: Packing 8 ciphertexts into 1 packed ciphertext");
    let packed = pack_multivector(
        &components.try_into().unwrap(),
        &rotation_keys,
        &rotation_ctx,
        &ckks_ctx,
    )?;
    println!("  ✅ Packed into single ciphertext");
    println!("     Batch size: {}", packed.batch_size);
    println!("     Level: {}", packed.level);
    println!("     Scale: {:.2e}\n", packed.scale);

    // Step 6: Unpack back to 8 components
    println!("Step 6: Unpacking back to 8 component ciphertexts");
    let unpacked = unpack_multivector(
        &packed,
        &rotation_keys,
        &rotation_ctx,
        &ckks_ctx,
    )?;
    println!("  ✅ Unpacked to 8 ciphertexts\n");

    // Step 7: Verify structure
    println!("Step 7: Verifying unpacked ciphertexts");
    let mut all_ok = true;
    for (i, ct) in unpacked.iter().enumerate() {
        let ok = ct.n == n && ct.level == level && ct.num_primes == level + 1;
        println!("  Component {}: level={}, n={}, num_primes={} {}",
                 i, ct.level, ct.n, ct.num_primes,
                 if ok { "✅" } else { "❌" });
        if !ok {
            all_ok = false;
        }
    }
    println!();

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    if all_ok {
        println!("✅ ALL TESTS PASSED");
        println!("═══════════════════════════════════════════════════════════════");
        println!("\nV4 CUDA Operations Summary:");
        println!("  • Pack 8 → 1: ✅");
        println!("  • Unpack 1 → 8: ✅");
        println!("  • Structure verification: ✅");
        println!("\n🎉 V4 CUDA BASIC OPERATIONS WORKING!");
        println!("    Ready for geometric product testing\n");
        Ok(())
    } else {
        println!("❌ TESTS FAILED");
        println!("═══════════════════════════════════════════════════════════════\n");
        Err("Test failed: Structure mismatch".to_string())
    }
}

#[cfg(not(all(feature = "v4", feature = "v2-gpu-cuda")))]
fn main() {
    eprintln!("This example requires features: v4,v2-gpu-cuda");
    eprintln!("Run with: cargo run --release --features v2,v2-gpu-cuda,v4 --example test_v4_cuda_basic");
    std::process::exit(1);
}
