//! Simple test: encode → encrypt → decrypt → decode
//! This isolates the Metal GPU enc/dec path to find the bug

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::ckks::MetalCkksContext,
    backends::cpu_optimized::keys::KeyContext,
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Testing Metal GPU encode → encrypt → decrypt → decode");
    println!("====================================================\n");

    // Setup
    let params = CliffordFHEParams::new_test_ntt_4096();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ctx = MetalCkksContext::new(params.clone())?;

    // Test value
    let test_val = 42.0;
    println!("Original value: {}", test_val);

    // Encode
    let pt = ctx.encode(&[test_val])?;
    println!("Encoded with scale: {}", pt.scale);
    println!("Encoded at level: {}", pt.level);

    // Encrypt
    let ct = ctx.encrypt(&pt, &pk)?;
    println!("Encrypted at level: {}", ct.level);
    println!("Encrypted with scale: {}", ct.scale);

    // Decrypt
    let pt_dec = ctx.decrypt(&ct, &sk)?;
    println!("Decrypted at level: {}", pt_dec.level);
    println!("Decrypted with scale: {}", pt_dec.scale);

    // Decode
    let result = ctx.decode(&pt_dec)?;
    println!("Decoded value: {}", result[0]);

    let error = (result[0] - test_val).abs();
    let rel_error = error / test_val;
    println!("\nAbsolute error: {:.2e}", error);
    println!("Relative error: {:.2e}", rel_error);

    if rel_error < 1e-6 {
        println!("\n✅ PASS: Basic enc/dec works correctly!");
        Ok(())
    } else {
        println!("\n❌ FAIL: Basic enc/dec has errors!");
        Err(format!("Relative error too large: {}", rel_error))
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
    println!("Run with: cargo run --release --features v2,v2-gpu-metal --example test_metal_enc_dec");
}
