//! Test Metal GPU Ciphertext Conversion
//!
//! Tests the lossless conversion between MetalCiphertext and CPU Ciphertext.
//! This is critical for the full GPU bootstrap pipeline.
//!
//! **Run with:**
//! ```bash
//! cargo run --release --features v2,v2-gpu-metal --example test_metal_conversion
//! ```

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Metal GPU Ciphertext Conversion Test                   ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Use small parameters (N=4096, 5 primes)
    let params = CliffordFHEParams::new_test_ntt_4096();
    println!("Parameters: N={}, {} primes\n", params.n, params.moduli.len());

    // Step 1: Generate keys
    println!("Step 1: Generating keys with Metal GPU...");
    let mut key_ctx = MetalKeyContext::new(params.clone())
        .map_err(|e| format!("Failed to create Metal context: {}", e))?;
    let (pk, sk, _evk) = key_ctx.keygen()
        .map_err(|e| format!("Key generation failed: {}", e))?;
    println!("  ✓ Keys generated\n");

    // Step 2: Create CKKS context
    println!("Step 2: Creating Metal CKKS context...");
    let ckks_ctx = MetalCkksContext::new(params.clone())
        .map_err(|e| format!("Failed to create Metal CKKS context: {}", e))?;
    println!("  ✓ Context created\n");

    // Step 3: Encrypt a value
    println!("Step 3: Encrypting test value...");
    let test_value = 42.123456789;
    println!("  Original value: {:.10}", test_value);

    let pt = ckks_ctx.encode(&[test_value])
        .map_err(|e| format!("Encoding failed: {}", e))?;
    let ct_metal = ckks_ctx.encrypt(&pt, &pk)
        .map_err(|e| format!("Encryption failed: {}", e))?;
    println!("  ✓ Encrypted (Metal GPU ciphertext)\n");

    // Step 4: Convert to CPU format
    println!("Step 4: Converting Metal GPU → CPU ciphertext...");
    let ct_cpu = ckks_ctx.to_cpu_ciphertext(&ct_metal);
    println!("  ✓ Converted to CPU format");
    println!("    CPU ciphertext: n={}, level={}, scale={:.2e}\n", ct_cpu.n, ct_cpu.level, ct_cpu.scale);

    // Step 5: Convert back to Metal GPU format
    println!("Step 5: Converting CPU → Metal GPU ciphertext...");
    let ct_metal_roundtrip = ckks_ctx.from_cpu_ciphertext(&ct_cpu);
    println!("  ✓ Converted back to Metal GPU format");
    println!("    Metal ciphertext: n={}, level={}, scale={:.2e}\n", ct_metal_roundtrip.n, ct_metal_roundtrip.level, ct_metal_roundtrip.scale);

    // Step 6: Decrypt and verify
    println!("Step 6: Verifying conversion is lossless...");

    // Decrypt original Metal ciphertext
    let pt_dec_original = ckks_ctx.decrypt(&ct_metal, &sk)
        .map_err(|e| format!("Decryption failed: {}", e))?;
    let decoded_original = ckks_ctx.decode(&pt_dec_original)
        .map_err(|e| format!("Decoding failed: {}", e))?;

    // Decrypt roundtrip ciphertext
    let pt_dec_roundtrip = ckks_ctx.decrypt(&ct_metal_roundtrip, &sk)
        .map_err(|e| format!("Decryption failed: {}", e))?;
    let decoded_roundtrip = ckks_ctx.decode(&pt_dec_roundtrip)
        .map_err(|e| format!("Decoding failed: {}", e))?;

    let error_original = (decoded_original[0] - test_value).abs();
    let error_roundtrip = (decoded_roundtrip[0] - test_value).abs();
    let error_diff = (decoded_original[0] - decoded_roundtrip[0]).abs();

    println!("  Original ciphertext:");
    println!("    Decrypted: {:.10}", decoded_original[0]);
    println!("    Error:     {:.2e}", error_original);
    println!("\n  Roundtrip ciphertext:");
    println!("    Decrypted: {:.10}", decoded_roundtrip[0]);
    println!("    Error:     {:.2e}", error_roundtrip);
    println!("\n  Difference between original and roundtrip:");
    println!("    Diff:      {:.2e}\n", error_diff);

    // Verify conversion is lossless (should be exactly equal)
    if error_diff < 1e-10 {
        println!("✓ SUCCESS: Conversion is lossless!");
        println!("✓ Metal GPU ↔ CPU ciphertext conversion working perfectly!");
        println!("✓ Ready for full GPU bootstrap pipeline!\n");
        Ok(())
    } else {
        println!("✗ FAILED: Conversion introduced error!");
        println!("  Expected exact match, got difference: {:.2e}\n", error_diff);
        Err(format!("Conversion error too large: {:.2e}", error_diff))
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    eprintln!("This example requires the v2-gpu-metal feature.");
    eprintln!("Run with: cargo run --release --features v2,v2-gpu-metal --example test_metal_conversion");
}
