//! Test Metal GPU ciphertext subtraction

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
        inversion::subtract_ciphertexts_metal,
    },
    backends::cpu_optimized::keys::KeyContext,
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Testing Metal GPU ciphertext subtraction");
    println!("========================================\n");

    // Setup
    let params = CliffordFHEParams::new_test_ntt_4096();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ctx = MetalCkksContext::new(params.clone())?;

    // Test: 10.0 - 3.0 = 7.0
    let a = 10.0;
    let b = 3.0;
    let expected = a - b;

    println!("Computing {} - {} = {}", a, b, expected);

    // Encrypt
    let pt_a = ctx.encode(&[a])?;
    let pt_b = ctx.encode(&[b])?;
    let ct_a = ctx.encrypt(&pt_a, &pk)?;
    let ct_b = ctx.encrypt(&pt_b, &pk)?;

    println!("Level: {}", ct_a.level);
    println!("Scale: {}", ct_a.scale);

    // Subtract
    let ct_result = subtract_ciphertexts_metal(&ct_a, &ct_b, &ctx)?;

    // Decrypt and decode
    let pt_result = ctx.decrypt(&ct_result, &sk)?;
    let result = ctx.decode(&pt_result)?;

    println!("\nExpected: {}", expected);
    println!("Got:      {}", result[0]);

    let error = (result[0] - expected).abs();
    let rel_error = error / expected;
    println!("Absolute error: {:.2e}", error);
    println!("Relative error: {:.2e}", rel_error);

    if rel_error < 1e-6 {
        println!("\n✅ PASS: Subtraction works correctly!");
        Ok(())
    } else {
        println!("\n❌ FAIL: Subtraction has errors!");
        Err(format!("Relative error too large: {}", rel_error))
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
    println!("Run with: cargo run --release --features v2,v2-gpu-metal --example test_metal_subtract");
}
