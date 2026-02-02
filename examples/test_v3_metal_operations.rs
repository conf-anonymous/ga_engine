//! V3 Metal GPU Operations Demo - Test Metal GPU CKKS Operations
//!
//! This example demonstrates Metal GPU CKKS operations:
//! - Encryption/Decryption
//! - multiply_plain (with automatic rescale)
//! - add
//! - Lossless GPU ↔ CPU conversion
//!
//! **100% METAL GPU BACKEND** for all CKKS operations!

use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use std::time::Instant;

fn main() -> Result<(), String> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     V3 Metal GPU Operations Demo - CKKS Operations Test        ║");
    println!("║     100% Metal GPU Backend with multiply_plain + rescale       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Setup parameters
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 1: Setup V3 Parameters");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let params = CliffordFHEParams::new_v3_bootstrap_fast_demo();

    println!("Parameters:");
    println!("  Ring dimension N: {}", params.n);
    println!("  Number of primes: {}", params.moduli.len());
    println!("  Scale: 2^40 = {}\n", params.scale);

    // Key Generation
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 2: Generating Encryption Keys (Metal GPU)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let start = Instant::now();
    let mut key_ctx = MetalKeyContext::new(params.clone())?;
    let (pk, sk, _evk) = key_ctx.keygen()?;
    let keygen_time = start.elapsed();
    println!("  ✓ Keys generated in {:.2}s (Metal GPU)\n", keygen_time.as_secs_f64());

    // Create CKKS context
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 3: Test Metal GPU CKKS Operations");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let ckks = MetalCkksContext::new(params.clone())?;

    // Test 1: Encryption/Decryption
    println!("Test 1: Encryption/Decryption");
    let value1 = 42.0;
    let pt1 = ckks.encode(&[value1])?;
    let ct1 = ckks.encrypt(&pt1, &pk)?;

    println!("  Original value: {}", value1);
    println!("  Encrypted level: {}", ct1.level);
    println!("  Encrypted scale: {:.2e}", ct1.scale);

    let decrypted = ckks.decrypt(&ct1, &sk)?;
    let decoded = ckks.decode(&decrypted)?;
    let error = (decoded[0] - value1).abs();

    println!("  Decrypted value: {:.10}", decoded[0]);
    println!("  Error: {:.2e}", error);
    assert!(error < 0.01, "Encryption/decryption error too large");
    println!("  ✓ PASS\n");

    // Test 2: multiply_plain with rescale
    println!("Test 2: multiply_plain (with automatic rescale)");
    let multiplier = 2.0;
    let pt_mult = ckks.encode(&[multiplier])?;

    println!("  Multiplying by {}", multiplier);
    println!("  Before: level={}, scale={:.2e}", ct1.level, ct1.scale);

    let ct2 = ct1.multiply_plain(&pt_mult, &ckks)?;

    println!("  After:  level={}, scale={:.2e}", ct2.level, ct2.scale);
    println!("  (Note: level decreased by 1 due to rescale)");

    let decrypted2 = ckks.decrypt(&ct2, &sk)?;
    let decoded2 = ckks.decode(&decrypted2)?;
    let expected2 = value1 * multiplier;
    let error2 = (decoded2[0] - expected2).abs();

    println!("  Expected: {}", expected2);
    println!("  Got:      {:.10}", decoded2[0]);
    println!("  Error:    {:.2e}", error2);
    assert!(error2 < 0.01, "Multiply error too large");
    println!("  ✓ PASS\n");

    // Test 3: add
    println!("Test 3: Ciphertext Addition");
    let value3 = 10.0;
    let pt3 = ckks.encode(&[value3])?;
    let ct3 = ckks.encrypt(&pt3, &pk)?;

    // Need to match levels for addition - multiply ct3 to drop its level
    let pt_one = ckks.encode(&[1.0])?;
    let ct3 = ct3.multiply_plain(&pt_one, &ckks)?;  // Drop to level 14

    println!("  Adding {} + {}", expected2, value3);
    println!("  ct2 level: {}, ct3 level: {}", ct2.level, ct3.level);

    let ct4 = ct2.add(&ct3, &ckks)?;

    println!("  Result level: {}", ct4.level);

    let decrypted4 = ckks.decrypt(&ct4, &sk)?;
    let decoded4 = ckks.decode(&decrypted4)?;
    let expected4 = expected2 + value3;
    let error4 = (decoded4[0] - expected4).abs();

    println!("  Expected: {}", expected4);
    println!("  Got:      {:.10}", decoded4[0]);
    println!("  Error:    {:.2e}", error4);
    assert!(error4 < 0.1, "Addition error too large");
    println!("  ✓ PASS\n");

    // Test 4: Conversion to CPU and back
    println!("Test 4: Lossless GPU ↔ CPU Conversion");
    let cpu_ct = ckks.to_cpu_ciphertext(&ct4);
    let gpu_ct_back = ckks.from_cpu_ciphertext(&cpu_ct);

    println!("  Original GPU ct: level={}, scale={:.2e}", ct4.level, ct4.scale);
    println!("  After CPU→GPU:   level={}, scale={:.2e}", gpu_ct_back.level, gpu_ct_back.scale);

    let decrypted5 = ckks.decrypt(&gpu_ct_back, &sk)?;
    let decoded5 = ckks.decode(&decrypted5)?;
    let error5 = (decoded5[0] - expected4).abs();

    println!("  Value after conversion: {:.10}", decoded5[0]);
    println!("  Conversion error: {:.2e}", error5);
    assert!(error5 < 0.1, "Conversion error too large");
    println!("  ✓ PASS\n");

    // Summary
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         ✅ ALL TESTS PASSED - Metal GPU CKKS Working!          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Summary of Metal GPU Operations:");
    println!("  ✓ Encryption/Decryption");
    println!("  ✓ multiply_plain (with automatic rescale)");
    println!("  ✓ add (ciphertext addition)");
    println!("  ✓ Lossless GPU ↔ CPU conversion");
    println!("\n🎉 Complete Metal GPU CKKS backend working perfectly!");
    println!("   All operations running on Apple M3 Max GPU.\n");

    Ok(())
}
