//! Test V4 Geometric Product with Metal GPU
//!
//! This example verifies that geometric_product_packed() works correctly
//! by testing: (1 + 2e₁) ⊗ (3e₂) = 3e₂ + 6e₁₂

#[cfg(all(feature = "v4", feature = "v2-gpu-metal"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::{
        ckks::MetalCkksContext,
        rotation_keys::MetalRotationKeys,
    };
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v4::{
        packing::{pack_multivector, unpack_multivector},
        geometric_ops::geometric_product_packed,
    };

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     V4 Metal GPU Geometric Product Test                     ║");
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

    // Step 3: Create Metal GPU context
    println!("Step 3: Creating Metal GPU context");
    let ckks_ctx = MetalCkksContext::new(params.clone())?;
    println!("  ✓ Metal device initialized");
    println!("  ✓ NTT contexts ready\n");

    // Step 4: Generate rotation keys (needed for packing/unpacking)
    println!("Step 4: Generating rotation keys");
    // Need both positive and negative rotations for packing/unpacking
    let mut rotation_steps: Vec<i32> = (1..=8).collect();
    rotation_steps.extend((-8..=-1).collect::<Vec<i32>>());

    // Reuse the Metal device and NTT contexts from the CKKS context
    let metal_device = ckks_ctx.device().clone();
    let metal_ntt_contexts = ckks_ctx.ntt_contexts();

    let base_w = 20u32;
    let rot_keys = MetalRotationKeys::generate(
        metal_device,
        &sk,
        &rotation_steps,
        &params,
        metal_ntt_contexts,
        base_w,
    )?;
    println!("  ✓ Rotation keys generated (rotations 1-8)\n");

    // Step 4: Create test multivectors
    println!("Step 4: Creating test multivectors");
    println!("  a = 1 + 2e₁");
    println!("  b = 3e₂\n");

    let batch_size = 1; // Single multivector for this test
    let num_slots = params.n / 2;

    // Create component values for a = 1 + 2e₁
    let mut a_vals = vec![vec![0.0; num_slots]; 8];
    a_vals[0][0] = 1.0;  // scalar = 1
    a_vals[1][0] = 2.0;  // e1 = 2

    // Create component values for b = 3e₂
    let mut b_vals = vec![vec![0.0; num_slots]; 8];
    b_vals[2][0] = 3.0;  // e2 = 3

    // Step 5: Encode and encrypt components
    println!("Step 5: Encoding and encrypting");
    let mut a_components = Vec::with_capacity(8);
    let mut b_components = Vec::with_capacity(8);

    for i in 0..8 {
        let a_pt = ckks_ctx.encode(&a_vals[i])?;
        let a_ct = ckks_ctx.encrypt(&a_pt, &pk)?;
        a_components.push(a_ct);

        let b_pt = ckks_ctx.encode(&b_vals[i])?;
        let b_ct = ckks_ctx.encrypt(&b_pt, &pk)?;
        b_components.push(b_ct);
    }

    let a_array: [_; 8] = [
        a_components[0].clone(), a_components[1].clone(), a_components[2].clone(),
        a_components[3].clone(), a_components[4].clone(), a_components[5].clone(),
        a_components[6].clone(), a_components[7].clone(),
    ];
    let b_array: [_; 8] = [
        b_components[0].clone(), b_components[1].clone(), b_components[2].clone(),
        b_components[3].clone(), b_components[4].clone(), b_components[5].clone(),
        b_components[6].clone(), b_components[7].clone(),
    ];

    println!("  ✓ All components encrypted\n");

    // Step 6: Pack into PackedMultivectors
    println!("Step 6: Packing into V4 packed layout");
    let a_packed = pack_multivector(&a_array, batch_size, &rot_keys, &ckks_ctx)?;
    let b_packed = pack_multivector(&b_array, batch_size, &rot_keys, &ckks_ctx)?;
    println!("  ✓ Multivectors packed (8 ciphertexts → 1 ciphertext each)");
    println!("  ✓ Memory reduction: 8× (16 ciphertexts → 2 ciphertexts)\n");

    // Step 7: Compute geometric product
    println!("Step 7: Computing geometric product on GPU");
    println!("  Computing: (1 + 2e₁) ⊗ (3e₂) = ?");
    let start = std::time::Instant::now();
    let result_packed = geometric_product_packed(&a_packed, &b_packed, &rot_keys, &ckks_ctx)?;
    let elapsed = start.elapsed();
    println!("  ✓ Geometric product completed in {:.3}s\n", elapsed.as_secs_f64());

    // Step 8: Unpack and decrypt result
    println!("Step 8: Unpacking and decrypting result");
    let result_components = unpack_multivector(&result_packed, &rot_keys, &ckks_ctx)?;

    let mut result_vals = Vec::new();
    for i in 0..8 {
        let pt = ckks_ctx.decrypt(&result_components[i], &sk)?;
        let vals = ckks_ctx.decode(&pt)?;
        result_vals.push(vals[0]);
    }

    println!("  ✓ Result unpacked and decrypted\n");

    // Step 9: Verify result
    println!("Step 9: Verifying result");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Expected: 3e₂ + 6e₁₂");
    println!("───────────────────────────────────────────────────────────────");
    println!("Component          | Expected | Actual    | Error     ");
    println!("───────────────────────────────────────────────────────────────");

    let component_names = ["scalar", "e1", "e2", "e3", "e12", "e13", "e23", "e123"];
    let expected = [0.0, 0.0, 3.0, 0.0, 6.0, 0.0, 0.0, 0.0];

    let mut max_error: f64 = 0.0;
    for i in 0..8 {
        let error = (result_vals[i] - expected[i]).abs();
        max_error = max_error.max(error);
        println!("{:18} | {:8.4} | {:9.4} | {:9.6}",
                 component_names[i], expected[i], result_vals[i], error);
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("Maximum error: {:.6e}", max_error);

    if max_error < 1e-3 {
        println!("\n✅ V4 GEOMETRIC PRODUCT TEST PASSED!");
        println!("   All components match expected values within tolerance");
        println!("   (1 + 2e₁) ⊗ (3e₂) = 3e₂ + 6e₁₂ ✓\n");
        Ok(())
    } else {
        println!("\n❌ TEST FAILED: Maximum error exceeds tolerance\n");
        Err(format!("Error {} exceeds threshold 1e-3", max_error))
    }
}

#[cfg(not(all(feature = "v4", feature = "v2-gpu-metal")))]
fn main() {
    println!("This example requires features: v4,v2-gpu-metal");
    println!("Run with: cargo run --release --features v4,v2-gpu-metal --example test_v4_geometric_product");
}
