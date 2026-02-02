//! Coefficient-Level Rotation Debug Test
//!
//! This test shows which specific coefficients are computed correctly after rotation.
//! It helps identify whether the issue is:
//! 1. Galois automorphism (wrong coefficient indices)
//! 2. CKKS slot-to-coefficient mapping
//! 3. Flat RNS layout indexing
//!
//! **Run:**
//! ```bash
//! cargo run --release --features v2,v2-gpu-metal,v3 --example test_rotation_debug
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::rotation_keys::MetalRotationKeys;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::device::MetalDevice;
    use std::sync::Arc;

    println!("=== Coefficient-Level Rotation Debug ===\n");

    // Use small parameters for clarity
    let params = CliffordFHEParams::new_v3_bootstrap_metal()?;

    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  num_primes = {}", params.moduli.len());
    println!("  moduli = {:?}", params.moduli);
    println!();

    // Generate keys
    println!("Step 1: Generating encryption keys");
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    println!("  ✅ Keys generated\n");

    // Create Metal GPU context
    println!("Step 2: Initializing Metal GPU context");
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    println!("  ✅ Metal GPU context ready\n");

    // Generate rotation keys for k=1 (rotate left by 1)
    println!("Step 3: Generating rotation key for k=1");
    let metal_device = Arc::new(MetalDevice::new()?);
    let metal_ntt_contexts = create_metal_ntt_contexts(&params, metal_device.clone())?;

    let base_w = 30;
    let metal_rot_keys = MetalRotationKeys::generate(
        metal_device.clone(),
        &sk,
        &[1],  // Only generate key for rotation by 1
        &params,
        &metal_ntt_contexts,
        base_w,
    )?;

    // Compute log_q manually
    let log_q: u32 = params.moduli.iter().map(|&q| {
        (q as f64).log2().ceil() as u32
    }).sum();
    let num_digits = (log_q + base_w - 1) / base_w;
    println!("  base_w = {}", base_w);
    println!("  log_q ≈ {}", log_q);
    println!("  num_digits = {}", num_digits);
    println!("  ✅ Rotation key generated\n");

    // Create simple message: only first 3 slots non-zero
    let num_slots = params.n / 2;
    let mut message = vec![0.0; num_slots];
    message[0] = 100.0;
    message[1] = 200.0;
    message[2] = 300.0;

    println!("Step 4: Input message (first 10 slots):");
    for i in 0..10 {
        println!("  Slot {}: {:.1}", i, message[i]);
    }
    println!();

    // Encrypt
    println!("Step 5: Encrypting");
    let pt = metal_ctx.encode(&message)?;
    let ct = metal_ctx.encrypt(&pt, &pk)?;
    println!("  ✅ Encrypted");
    println!("  Level={}, Scale={:.2e}", ct.level, ct.scale);
    println!();

    // Perform rotation
    println!("Step 6: Performing rotation by +1");
    let rotated = ct.rotate_by_steps(1, &metal_rot_keys, &metal_ctx)?;
    println!("  ✅ Rotation complete");
    println!("  Level={}, Scale={:.2e}\n", rotated.level, rotated.scale);

    // Decrypt and decode
    println!("Step 7: Decrypting and decoding");
    let pt_decrypted = metal_ctx.decrypt(&rotated, &sk)?;
    let decrypted_message = metal_ctx.decode(&pt_decrypted)?;
    println!("  ✅ Decrypted\n");

    // Expected result after rotate left by 1: [200, 300, 0, 0, ...]
    println!("=== Decoded Slot Comparison ===\n");
    println!("After rotate left by 1, expect: [200, 300, 0, ...]");
    println!();

    let mut max_error: f64 = 0.0;
    let mut num_correct = 0;
    let mut num_incorrect = 0;

    for i in 0..10 {
        let expected = if i == 0 {
            200.0
        } else if i == 1 {
            300.0
        } else {
            0.0
        };

        let got = decrypted_message[i];
        let error = (got - expected).abs();
        max_error = max_error.max(error);

        let status = if error < 1.0 { "✓ PASS" } else { "✗ FAIL" };
        if error < 1.0 {
            num_correct += 1;
        } else {
            num_incorrect += 1;
        }

        println!(
            "Slot {:2}: expected {:8.1}, got {:12.1}, error {:10.2e}  {}",
            i, expected, got, error, status
        );
    }

    println!();
    println!("Summary:");
    println!("  Correct slots:   {}/10", num_correct);
    println!("  Incorrect slots: {}/10", num_incorrect);
    println!("  Max error:       {:.2e}", max_error);
    println!();

    // Final verdict
    if num_correct == 10 {
        println!("✅ ALL SLOTS CORRECT - Rotation working perfectly!");
    } else if num_correct > 0 {
        println!("⚠️  PARTIAL SUCCESS - {} slots correct, {} incorrect", num_correct, num_incorrect);
        println!("    → Likely an indexing or slot mapping issue");
    } else {
        println!("❌ ALL SLOTS INCORRECT - Systematic error in rotation");
        println!("    → Check Galois automorphism or key switching");
    }

    Ok(())
}

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn create_metal_ntt_contexts(
    params: &ga_engine::clifford_fhe_v2::params::CliffordFHEParams,
    device: std::sync::Arc<ga_engine::clifford_fhe_v2::backends::gpu_metal::device::MetalDevice>,
) -> Result<Vec<ga_engine::clifford_fhe_v2::backends::gpu_metal::ntt::MetalNttContext>, String> {
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::ntt::MetalNttContext;

    let mut contexts = Vec::new();
    for &q in &params.moduli {
        let psi = find_primitive_2n_root(params.n, q)?;
        let ctx = MetalNttContext::new_with_device(device.clone(), params.n, q, psi)?;
        contexts.push(ctx);
    }
    Ok(contexts)
}

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn find_primitive_2n_root(n: usize, q: u64) -> Result<u64, String> {
    let two_n = (2 * n) as u64;
    if (q - 1) % two_n != 0 {
        return Err(format!("q = {} is not NTT-friendly for n = {}", q, n));
    }

    let exp = (q - 1) / two_n;

    for g in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        let psi = pow_mod(g % q, exp, q);
        if psi != 1
            && pow_mod(psi, two_n, q) == 1
            && pow_mod(psi, n as u64, q) != 1
        {
            return Ok(psi);
        }
    }

    Err(format!("Could not find primitive 2N-th root for n = {}, q = {}", n, q))
}

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exp /= 2;
    }
    result
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3")))]
fn main() {
    println!("This example requires features: v2, v2-gpu-metal, v3");
    println!("Run with: cargo run --release --features v2,v2-gpu-metal,v3 --example test_rotation_debug");
}
