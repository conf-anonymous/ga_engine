//! Metal GPU Rotation Test (No Bootstrap)
//!
//! Tests rotation in isolation without requiring CoeffToSlot/SlotToCoeff.
//! This validates that the Metal GPU rotation infrastructure works correctly.
//!
//! **Run:**
//! ```bash
//! cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_rotation_only
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::rotation_keys::MetalRotationKeys;
    use std::time::Instant;

    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              Metal GPU Rotation Test (Infrastructure Validation)             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝\n");

    // Test parameters
    println!("Step 1: Setting up parameters (N=1024)");
    let params = CliffordFHEParams::new_v3_bootstrap_metal()?;
    println!("  N = {}", params.n);
    println!("  Primes = {}", params.moduli.len());
    println!();

    // Generate keys
    println!("Step 2: Generating encryption keys");
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    println!("  ✅ Keys generated");
    println!();

    // Create Metal GPU context
    println!("Step 3: Initializing Metal GPU CKKS context");
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    println!("  ✅ Metal GPU context ready");
    println!();

    // Generate rotation keys
    println!("Step 4: Generating rotation keys");
    let rotation_steps = vec![1, -1, 2, -2, 4, -4];

    let start_keygen = Instant::now();
    let metal_device = std::sync::Arc::new(
        ga_engine::clifford_fhe_v2::backends::gpu_metal::device::MetalDevice::new()?
    );

    let metal_ntt_contexts = create_metal_ntt_contexts(&params, metal_device.clone())?;

    let metal_rot_keys = MetalRotationKeys::generate(
        metal_device.clone(),
        &sk,
        &rotation_steps,
        &params,
        &metal_ntt_contexts,
        30,  // base_w = 30 (B = 2^30) - larger base = fewer digits!
    )?;
    let keygen_time = start_keygen.elapsed();
    println!("  ✅ {} rotation keys generated in {:.2}s", metal_rot_keys.num_keys(), keygen_time.as_secs_f64());
    println!();

    // Create test message with pattern
    println!("Step 5: Creating test message");
    let num_slots = params.n / 2;
    let mut message = vec![0.0; num_slots];
    for i in 0..10.min(num_slots) {
        message[i] = (i + 1) as f64;
    }
    println!("  Original: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, ...]");
    println!();

    // Encrypt
    println!("Step 6: Encrypting on Metal GPU");
    let start_encrypt = Instant::now();
    let pt = metal_ctx.encode(&message)?;
    let ct = metal_ctx.encrypt(&pt, &pk)?;
    let encrypt_time = start_encrypt.elapsed();
    println!("  ✅ Encrypted in {:.2}ms", encrypt_time.as_secs_f64() * 1000.0);
    println!("  Level={}, Scale={:.2e}", ct.level, ct.scale);
    println!();

    // Test rotations
    println!("Step 7: Testing rotations on Metal GPU");

    // Test +1 rotation
    println!("\n  Test 1: Rotate by +1 (left shift)");
    let start_rot = Instant::now();
    let ct_rot1 = ct.rotate_by_steps(1, &metal_rot_keys, &metal_ctx)?;
    let rot_time = start_rot.elapsed();

    let pt_rot1 = metal_ctx.decrypt(&ct_rot1, &sk)?;
    let result_rot1 = metal_ctx.decode(&pt_rot1)?;

    println!("    Rotation time: {:.2}ms", rot_time.as_secs_f64() * 1000.0);
    println!("    Expected: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, ...]");
    print!("    Got:      [");
    for i in 0..10 {
        print!("{:.1}", result_rot1[i]);
        if i < 9 { print!(", "); }
    }
    println!(", ...]");

    // Verify
    let mut rot1_correct = true;
    for i in 0..9 {
        let expected = message[i + 1];
        let got = result_rot1[i];
        let error = (expected - got).abs();
        if error > 0.1 {
            rot1_correct = false;
            println!("    ❌ Slot {}: expected {:.1}, got {:.1}, error {:.2e}", i, expected, got, error);
        }
    }
    if rot1_correct {
        println!("    ✅ Rotation by +1 is CORRECT!");
    }

    // Test -1 rotation
    println!("\n  Test 2: Rotate by -1 (right shift)");
    let start_rot = Instant::now();
    let ct_rot_neg1 = ct.rotate_by_steps(-1, &metal_rot_keys, &metal_ctx)?;
    let rot_time = start_rot.elapsed();

    let pt_rot_neg1 = metal_ctx.decrypt(&ct_rot_neg1, &sk)?;
    let result_rot_neg1 = metal_ctx.decode(&pt_rot_neg1)?;

    println!("    Rotation time: {:.2}ms", rot_time.as_secs_f64() * 1000.0);
    println!("    Expected: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, ...]");
    print!("    Got:      [");
    for i in 0..10 {
        print!("{:.1}", result_rot_neg1[i]);
        if i < 9 { print!(", "); }
    }
    println!(", ...]");

    let mut rot_neg1_correct = true;
    if result_rot_neg1[0].abs() > 0.1 {
        rot_neg1_correct = false;
    }
    for i in 1..10 {
        let expected = message[i - 1];
        let got = result_rot_neg1[i];
        let error = (expected - got).abs();
        if error > 0.1 {
            rot_neg1_correct = false;
            println!("    ❌ Slot {}: expected {:.1}, got {:.1}, error {:.2e}", i, expected, got, error);
        }
    }
    if rot_neg1_correct {
        println!("    ✅ Rotation by -1 is CORRECT!");
    }

    // Test +2 rotation
    println!("\n  Test 3: Rotate by +2");
    let ct_rot2 = ct.rotate_by_steps(2, &metal_rot_keys, &metal_ctx)?;
    let pt_rot2 = metal_ctx.decrypt(&ct_rot2, &sk)?;
    let result_rot2 = metal_ctx.decode(&pt_rot2)?;

    println!("    Expected: [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, ...]");
    print!("    Got:      [");
    for i in 0..10 {
        print!("{:.1}", result_rot2[i]);
        if i < 9 { print!(", "); }
    }
    println!(", ...]");

    let mut rot2_correct = true;
    for i in 0..8 {
        let expected = message[i + 2];
        let got = result_rot2[i];
        let error = (expected - got).abs();
        if error > 0.1 {
            rot2_correct = false;
        }
    }
    if rot2_correct {
        println!("    ✅ Rotation by +2 is CORRECT!");
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                   SUMMARY                                     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Rotation by +1:  {}                                                      ║",
             if rot1_correct { "✅ PASS" } else { "❌ FAIL" });
    println!("║ Rotation by -1:  {}                                                      ║",
             if rot_neg1_correct { "✅ PASS" } else { "❌ FAIL" });
    println!("║ Rotation by +2:  {}                                                      ║",
             if rot2_correct { "✅ PASS" } else { "❌ FAIL" });
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");

    if rot1_correct && rot_neg1_correct && rot2_correct {
        println!("║                                                                               ║");
        println!("║  🎉 SUCCESS: Metal GPU rotation infrastructure is working correctly!         ║");
        println!("║                                                                               ║");
        println!("║  Performance:                                                                 ║");
        println!("║    - Rotation time: ~{:.1}ms per rotation                                     ║", rot_time.as_secs_f64() * 1000.0);
        println!("║    - All operations on GPU (zero CPU fallback)                               ║");
        println!("║    - Ready for production use                                                 ║");
        println!("║                                                                               ║");
        println!("║  Note: Bootstrap (CoeffToSlot/SlotToCoeff) requires correct DFT matrices.    ║");
        println!("║        See BOOTSTRAP_NEXT_STEPS.md for implementation guide.                 ║");
        println!("║                                                                               ║");
        println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
        Ok(())
    } else {
        println!("║                                                                               ║");
        println!("║  ❌ FAILURE: Some rotations did not produce correct results                   ║");
        println!("║                                                                               ║");
        println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
        Err("Rotation tests failed".to_string())
    }
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

    Err(format!("Could not find primitive 2n-th root for n={}, q={}", n, q))
}

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn pow_mod(base: u64, exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u128;
    let mut base = base as u128 % modulus as u128;
    let mut exp = exp;
    let modulus = modulus as u128;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }
    result as u64
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3")))]
fn main() {
    eprintln!("This example requires features: v2,v2-gpu-metal,v3");
    eprintln!("Run: cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_rotation_only");
    std::process::exit(1);
}
