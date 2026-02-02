//! Test V4 Geometric Product with CUDA GPU
//!
//! This example verifies that V4 packed geometric product works correctly on CUDA
//! by testing: (1 + 2e₁) ⊗ (3e₂) = 3e₂ + 6e₁₂

#[cfg(all(feature = "v4", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::backends::gpu_cuda::{
        ckks::CudaCkksContext,
        rotation_keys::CudaRotationKeys,
        device::CudaDeviceContext,
    };
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v4::{
        pack_multivector, unpack_multivector,
        geometric_ops::geometric_product_packed,
    };
    use std::sync::Arc;

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     V4 CUDA GPU Geometric Product Test                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Step 1: Initialize parameters
    println!("Step 1: Initializing parameters (N=1024, depth=3)");
    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("  ✓ Ring dimension: N={}", params.n);
    println!("  ✓ Modulus chain: {} primes", params.moduli.len());
    println!("  ✓ Scale: 2^40\n");

    // Step 2: Generate encryption keys (using CPU for key generation)
    println!("Step 2: Generating encryption keys");
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    println!("  ✓ Secret key generated");
    println!("  ✓ Public key generated\n");

    // Step 3: Create CUDA GPU context
    println!("Step 3: Creating CUDA GPU context");
    let device = Arc::new(CudaDeviceContext::new()?);
    let ckks_ctx = CudaCkksContext::new(params.clone(), device.clone())?;
    println!("  ✓ CUDA device initialized");
    println!("  ✓ NTT contexts ready\n");

    // Step 4: Generate rotation keys (needed for packing/unpacking)
    println!("Step 4: Generating rotation keys");
    // Need both positive and negative rotations for packing/unpacking
    let mut rotation_steps: Vec<i32> = (1..=8).collect();
    rotation_steps.extend((-8..=-1).collect::<Vec<i32>>());

    let rot_keys = CudaRotationKeys::generate(
        device.clone(),
        &sk,
        &rotation_steps,
        &params,
        ckks_ctx.ntt_contexts(),
        16, // base_bits for gadget decomposition
    )?;
    println!("  ✓ Generated {} rotation keys\n", rot_keys.num_keys());

    // Step 5: Encrypt test multivectors
    println!("Step 5: Encrypting test multivectors");

    // Multivector a = 1 + 2e₁
    let batch_size = 4; // Small batch for testing
    let a_values = vec![
        vec![1.0; batch_size], // scalar: 1
        vec![2.0; batch_size], // e1: 2
        vec![0.0; batch_size], // e2: 0
        vec![0.0; batch_size], // e3: 0
        vec![0.0; batch_size], // e12: 0
        vec![0.0; batch_size], // e23: 0
        vec![0.0; batch_size], // e31: 0
        vec![0.0; batch_size], // I: 0
    ];

    // Multivector b = 3e₂
    let b_values = vec![
        vec![0.0; batch_size], // scalar: 0
        vec![0.0; batch_size], // e1: 0
        vec![3.0; batch_size], // e2: 3
        vec![0.0; batch_size], // e3: 0
        vec![0.0; batch_size], // e12: 0
        vec![0.0; batch_size], // e23: 0
        vec![0.0; batch_size], // e31: 0
        vec![0.0; batch_size], // I: 0
    ];

    // Encrypt each component
    let mut a_cts = Vec::new();
    let mut b_cts = Vec::new();

    for i in 0..8 {
        let a_ct = key_ctx.encrypt_ckks(&a_values[i], &pk);
        let b_ct = key_ctx.encrypt_ckks(&b_values[i], &pk);
        a_cts.push(a_ct);
        b_cts.push(b_ct);
    }
    println!("  ✓ Encrypted multivector a = 1 + 2e₁");
    println!("  ✓ Encrypted multivector b = 3e₂\n");

    // Step 6: Pack into V4 format
    println!("Step 6: Packing into V4 slot-interleaved format");
    let a_packed = pack_multivector(
        &a_cts.try_into().unwrap(),
        batch_size,
        &rot_keys,
        &ckks_ctx,
    )?;
    let b_packed = pack_multivector(
        &b_cts.try_into().unwrap(),
        batch_size,
        &rot_keys,
        &ckks_ctx,
    )?;
    println!("  ✓ Packed a into single ciphertext");
    println!("  ✓ Packed b into single ciphertext\n");

    // Step 7: Compute geometric product
    println!("Step 7: Computing geometric product (packed)");
    let result_packed = geometric_product_packed(&a_packed, &b_packed, &rot_keys, &ckks_ctx)?;
    println!("  ✓ Geometric product computed\n");

    // Step 8: Unpack and decrypt result
    println!("Step 8: Unpacking and decrypting result");
    let result_cts = unpack_multivector(&result_packed, &rot_keys, &ckks_ctx)?;

    let mut result_values = Vec::new();
    for ct in result_cts.iter() {
        let decrypted = key_ctx.decrypt_ckks(ct, &sk);
        result_values.push(decrypted);
    }
    println!("  ✓ Unpacked and decrypted all components\n");

    // Step 9: Verify correctness
    println!("Step 9: Verifying result");
    println!("\nExpected: (1 + 2e₁) ⊗ (3e₂) = 3e₂ + 6e₁₂");
    println!("\nActual result (first value in batch):");

    let component_names = ["scalar", "e1", "e2", "e3", "e12", "e23", "e31", "I"];
    let expected = [0.0, 0.0, 3.0, 0.0, 6.0, 0.0, 0.0, 0.0];

    let mut all_correct = true;
    for i in 0..8 {
        let actual = result_values[i][0];
        let exp = expected[i];
        let error = (actual - exp).abs();
        let is_correct = error < 0.1; // Allow small numerical error

        if !is_correct {
            all_correct = false;
        }

        let status = if is_correct { "✓" } else { "✗" };
        println!("  {} {:6}: expected {:6.2}, got {:6.2} (error: {:.2e})",
                 status, component_names[i], exp, actual, error);
    }

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    if all_correct {
        println!("║                     TEST PASSED ✓                            ║");
    } else {
        println!("║                     TEST FAILED ✗                            ║");
    }
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    if all_correct {
        Ok(())
    } else {
        Err("Test failed: Results do not match expected values".to_string())
    }
}

#[cfg(not(all(feature = "v4", feature = "v2-gpu-cuda")))]
fn main() {
    eprintln!("This example requires features: v4,v2-gpu-cuda");
    eprintln!("Run with: cargo run --release --features v2,v2-gpu-cuda,v4 --example test_v4_cuda_geometric_product");
    std::process::exit(1);
}
