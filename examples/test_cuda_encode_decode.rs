//! Minimal CUDA encode/encrypt/decrypt/decode test
//!
//! Test JUST the encode/decode pipeline without any homomorphic operations

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

    println!("=== CUDA Encode/Encrypt/Decrypt/Decode Test ===\n");

    let params = CliffordFHEParams::new_128bit();
    let ctx = CudaCkksContext::new(params.clone())?;

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();

    // Test simple value
    let test_val = 2.0;
    println!("Test value: {}", test_val);

    let scale = params.scale;
    let level = params.moduli.len() - 1;

    println!("Encoding...");
    let pt = ctx.encode(&[test_val], scale, level)?;
    println!("  Encoded plaintext scale: {}", pt.scale);
    println!("  Plaintext level: {}", pt.level);

    println!("\nEncrypting...");
    let ct = ctx.encrypt(&pt, &pk)?;
    println!("  Ciphertext scale: {}", ct.scale);
    println!("  Ciphertext level: {}", ct.level);

    println!("\nDecrypting...");
    let pt_dec = ctx.decrypt(&ct, &sk)?;
    println!("  Decrypted plaintext scale: {}", pt_dec.scale);
    println!("  Decrypted level: {}", pt_dec.level);

    println!("\nDecoding...");
    let result = ctx.decode(&pt_dec)?;

    println!("\nResults:");
    println!("  Expected: {}", test_val);
    println!("  Got:      {}", result[0]);
    println!("  Error:    {:.2e}", (result[0] - test_val).abs());

    let error = (result[0] - test_val).abs();
    if error < 0.01 {
        println!("\n✅ PASS - Encode/Decode working!");
        Ok(())
    } else {
        println!("\n❌ FAIL - Error too large!");
        Err(format!("Error {} is too large", error))
    }
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires features: v2,v2-gpu-cuda");
}
