//! Debug multiply_plain 512× bug
//!
//! NTT works perfectly, so bug must be in rescale or CKKS logic

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This test requires Metal GPU.");
    println!("Run with: cargo run --release --features v2,v3,v2-gpu-metal --example test_multiply_plain_debug");
}

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         multiply_plain Debug - Hunt for 512× Bug                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("Parameters: N={}, {} primes\n", params.n, params.moduli.len());

    // Generate keys
    println!("Generating keys...");
    let mut key_ctx = MetalKeyContext::new(params.clone())?;
    let (pk, sk, _evk) = key_ctx.keygen()?;
    println!("Keys generated!\n");

    // Create CKKS context
    let ckks = MetalCkksContext::new(params.clone())?;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test: Encrypt 42.0, multiply by plain 2.0");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Encode and encrypt 42.0
    let value = 42.0;
    println!("Encoding value: {}", value);
    let pt1 = ckks.encode(&[value])?;
    println!("  Scale: {}", pt1.scale);
    println!("  Level: {}", pt1.level);

    println!("\nEncrypting...");
    let ct = ckks.encrypt(&pt1, &pk)?;
    println!("  Ciphertext level: {}", ct.level);
    println!("  Ciphertext scale: {}", ct.scale);

    // Decode to verify encoding/encryption
    println!("\nDecrypting to verify...");
    let pt_back = ckks.decrypt(&ct, &sk)?;
    let decoded_back = ckks.decode(&pt_back)?;
    println!("  Decrypted: {}", decoded_back[0]);
    println!("  Error: {:.2e}", (decoded_back[0] - value).abs());

    // Encode plaintext multiplier
    let multiplier = 2.0;
    println!("\nEncoding plaintext multiplier: {}", multiplier);
    let pt_mult = ckks.encode(&[multiplier])?;
    println!("  Scale: {}", pt_mult.scale);
    println!("  Level: {}", pt_mult.level);

    // Multiply
    println!("\nPerforming multiply_plain...");
    println!("  Before: ct.level={}, ct.scale={}", ct.level, ct.scale);
    println!("  pt_mult.level={}, pt_mult.scale={}", pt_mult.level, pt_mult.scale);

    let ct_mult = ct.multiply_plain(&pt_mult, &ckks)?;

    println!("  After:  ct_mult.level={}, ct_mult.scale={}", ct_mult.level, ct_mult.scale);

    // Decrypt result
    println!("\nDecrypting result...");
    let pt_result = ckks.decrypt(&ct_mult, &sk)?;
    let decoded_result = ckks.decode(&pt_result)?;

    let expected = value * multiplier;
    let got = decoded_result[0];
    let error = (got - expected).abs();
    let error_factor = expected / got;

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                           RESULTS                                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Expected: {}", expected);
    println!("Got:      {}", got);
    println!("Error:    {:.2e}", error);
    println!("Factor:   {:.2}×", error_factor);

    if (error_factor - 512.0).abs() < 1.0 {
        println!("\n❌ BUG CONFIRMED: 512× error in multiply_plain!");
        println!("   NTT works fine, so bug is in rescale or CKKS multiply logic\n");
    } else if error < 1.0 {
        println!("\n✅ SUCCESS: multiply_plain working correctly!\n");
    } else {
        println!("\n⚠️  Unexpected error: {:.2e}\n", error);
    }

    // Additional diagnostic: check intermediate values
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Additional Diagnostics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Scale analysis:");
    println!("  Initial ct scale: {}", ct.scale);
    println!("  Plaintext scale:  {}", pt_mult.scale);
    println!("  Product scale:    {} (should be ct.scale × pt_mult.scale)", ct.scale * pt_mult.scale);
    println!("  Result scale:     {}", ct_mult.scale);
    println!("  Dropped prime:    {}", params.moduli[ct.level]);
    println!("  Expected rescale: product_scale / dropped_prime");

    let expected_scale_after_rescale = (ct.scale * pt_mult.scale) / params.moduli[ct.level] as f64;
    println!("  Expected scale after rescale: {}", expected_scale_after_rescale);
    println!("  Ratio: result_scale / expected_scale = {:.2}", ct_mult.scale / expected_scale_after_rescale);

    Ok(())
}
