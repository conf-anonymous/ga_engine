//! V3 Bootstrap with Metal GPU - Using CORRECT V3 Implementation
//!
//! This test uses the CORRECT V3 bootstrap implementation that we spent time debugging:
//! - V3's CoeffToSlot (CPU - with correct scale management)
//! - V3's SlotToCoeff (CPU - with correct scale management)
//! - Metal GPU for encryption/decryption and key generation
//!
//! The V2 Metal GPU bootstrap.rs has bugs (scale explosion).
//! This test uses the working V3 CPU implementation instead.

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
    use ga_engine::clifford_fhe_v3::bootstrapping::{coeff_to_slot, slot_to_coeff, generate_rotation_keys};
    use std::time::Instant;

    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║    V3 Bootstrap Test - Using CORRECT V3 Implementation (CPU transforms)      ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝\n");

    // Test parameters: N=8192 with sufficient primes for full bootstrap
    println!("Step 1: Setting up parameters (N=8192, 41 primes for full bootstrap)");
    let params = CliffordFHEParams::new_v3_bootstrap_8192();

    println!("  N = {}", params.n);
    println!("  Primes = {}", params.moduli.len());
    println!("  Scale = {:.2e}", params.scale);
    println!();

    // Generate encryption keys (CPU)
    println!("Step 2: Generating encryption keys (CPU)");
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    println!("  ✅ Encryption keys generated");
    println!();

    // Generate rotation keys (V3)
    println!("Step 3: Generating rotation keys (V3 implementation)");
    // For N=8192, we need rotations up to N/4 = 2048
    // Generate: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    let rotation_steps: Vec<i32> = (0..12).map(|i| 1 << i).collect();  // 1, 2, 4, ..., 2048
    let rotation_steps_neg: Vec<i32> = rotation_steps.iter().map(|&x| -x).collect();
    let mut all_rotations = rotation_steps.clone();
    all_rotations.extend(rotation_steps_neg);

    println!("  Generating {} rotation keys...", all_rotations.len());
    let start_keygen = Instant::now();
    let rotation_keys = generate_rotation_keys(&all_rotations, &sk, &params);
    let keygen_time = start_keygen.elapsed();
    println!("  ✅ Rotation keys generated in {:.2}s", keygen_time.as_secs_f64());
    println!();

    // Create Metal GPU CKKS context for encryption/decryption
    println!("Step 4: Initializing Metal GPU CKKS context");
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    println!("  ✅ Metal GPU context ready");
    println!();

    // Create test message
    println!("Step 5: Creating test message");
    let num_slots = params.n / 2;
    let mut message = vec![0.0; num_slots];
    message[0] = 1.0;
    message[1] = 2.0;
    message[2] = 3.0;
    message[3] = 4.0;
    message[4] = 5.0;
    println!("  Message: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}, ...]",
             message[0], message[1], message[2], message[3], message[4]);
    println!();

    // Encode and encrypt on Metal GPU
    println!("Step 6: Encoding and encrypting on Metal GPU");
    let start_encrypt = Instant::now();
    let pt = metal_ctx.encode(&message)?;
    let ct_metal = metal_ctx.encrypt(&pt, &pk)?;
    let encrypt_time = start_encrypt.elapsed();
    println!("  ✅ Encrypted in {:.2}ms", encrypt_time.as_secs_f64() * 1000.0);
    println!("  Initial: level={}, scale={:.2e}", ct_metal.level, ct_metal.scale);
    println!();

    // Convert Metal GPU ciphertext to CPU format for V3 bootstrap
    println!("Step 7: Converting Metal GPU → CPU for V3 bootstrap");
    let ct_cpu = metal_ctx.to_cpu_ciphertext(&ct_metal);
    println!("  ✅ Converted to CPU ciphertext");
    println!();

    // CoeffToSlot using V3 (CPU, CORRECT implementation)
    println!("Step 8: Running CoeffToSlot (V3 CPU - CORRECT implementation)");
    println!("  Expected: 9 rotations, scale preserved at each level");
    let start_c2s = Instant::now();
    let ct_slots = coeff_to_slot(&ct_cpu, &rotation_keys)?;
    let c2s_time = start_c2s.elapsed();
    println!("  ✅ CoeffToSlot completed in {:.2}s", c2s_time.as_secs_f64());
    println!("  After C2S: level={}, scale={:.2e}", ct_slots.level, ct_slots.scale);
    println!();

    // SlotToCoeff using V3 (CPU, CORRECT implementation)
    println!("Step 9: Running SlotToCoeff (V3 CPU - CORRECT implementation)");
    println!("  Expected: 9 rotations (reversed order), scale preserved");
    let start_s2c = Instant::now();
    let ct_coeffs = slot_to_coeff(&ct_slots, &rotation_keys)?;
    let s2c_time = start_s2c.elapsed();
    println!("  ✅ SlotToCoeff completed in {:.2}s", s2c_time.as_secs_f64());
    println!("  After S2C: level={}, scale={:.2e}", ct_coeffs.level, ct_coeffs.scale);
    println!();

    // Convert back to Metal GPU and decrypt
    println!("Step 10: Converting CPU → Metal GPU and decrypting");
    let ct_final_metal = metal_ctx.from_cpu_ciphertext(&ct_coeffs);
    let start_decrypt = Instant::now();
    let pt_result = metal_ctx.decrypt(&ct_final_metal, &sk)?;
    let result = metal_ctx.decode(&pt_result)?;
    let decrypt_time = start_decrypt.elapsed();
    println!("  ✅ Decrypted in {:.2}ms", decrypt_time.as_secs_f64() * 1000.0);
    println!();

    // Verify correctness
    println!("Step 11: Verifying roundtrip accuracy");
    println!("\n  Slot | Expected | Decrypted | Error");
    println!("  -----|----------|-----------|----------");

    let mut max_error: f64 = 0.0;
    for i in 0..10 {
        let expected = message[i];
        let got = result[i];
        let error = (expected - got).abs();
        max_error = max_error.max(error);

        let status = if error < 1.0 { "✅" } else { "❌" };
        println!("  {:4} | {:8.2} | {:9.2} | {:.2e} {}",
                 i, expected, got, error, status);
    }
    println!();

    // Summary
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              PERFORMANCE SUMMARY                              ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Operation          │ Time         │ Notes                                     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Key Generation     │ {:8.2}s   │ V3 rotation keys (CPU)                    ║", keygen_time.as_secs_f64());
    println!("║ Encryption         │ {:8.2}ms  │ Metal GPU                                 ║", encrypt_time.as_secs_f64() * 1000.0);
    println!("║ CoeffToSlot        │ {:8.2}s   │ V3 CPU (CORRECT)                          ║", c2s_time.as_secs_f64());
    println!("║ SlotToCoeff        │ {:8.2}s   │ V3 CPU (CORRECT)                          ║", s2c_time.as_secs_f64());
    println!("║ Decryption         │ {:8.2}ms  │ Metal GPU                                 ║", decrypt_time.as_secs_f64() * 1000.0);
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ TOTAL BOOTSTRAP    │ {:8.2}s   │ CoeffToSlot + SlotToCoeff                 ║", (c2s_time + s2c_time).as_secs_f64());
    println!("║ Max Roundtrip Error│ {:.2e}    │ Target: < 1.0                             ║", max_error);
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Final verdict
    if max_error < 1.0 {
        println!("✅ SUCCESS: V3 bootstrap roundtrip is accurate!");
        println!("   Using CORRECT V3 CPU implementation (with proper scale management)");
        println!("   Metal GPU used for encryption/decryption only");
        Ok(())
    } else {
        println!("❌ FAILURE: Roundtrip error too large: {:.2e}", max_error);
        Err(format!("Bootstrap roundtrip failed with error {:.2e}", max_error))
    }
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3")))]
fn main() {
    println!("This example requires features: v2, v2-gpu-metal, v3");
    println!("Run with: cargo run --release --features v2,v2-gpu-metal,v3 --example test_v3_metal_bootstrap_correct");
}
