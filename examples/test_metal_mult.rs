//! Test Metal GPU ciphertext multiplication with relinearization

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
        ntt::MetalNttContext,
    },
    backends::cpu_optimized::keys::KeyContext,
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Testing Metal GPU ciphertext multiplication");
    println!("==========================================\n");

    // Setup
    let params = CliffordFHEParams::new_test_ntt_4096();
    let device = Arc::new(MetalDevice::new()?);
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ctx = MetalCkksContext::new(params.clone())?;

    // Generate relin keys using NTT contexts from CKKS context
    let ntt_contexts = ctx.ntt_contexts();
    let relin_keys = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        16,
    )?;

    // Test: 6.0 × 7.0 = 42.0
    let a = 6.0;
    let b = 7.0;
    let expected = a * b;

    println!("Computing {} × {} = {}", a, b, expected);

    // Encrypt
    let pt_a = ctx.encode(&[a])?;
    let pt_b = ctx.encode(&[b])?;
    let ct_a = ctx.encrypt(&pt_a, &pk)?;
    let ct_b = ctx.encrypt(&pt_b, &pk)?;

    println!("Initial level: {}", ct_a.level);
    println!("Initial scale: {}", ct_a.scale);

    // Multiply
    let ct_result = ct_a.multiply(&ct_b, &relin_keys, &ctx)?;

    println!("After multiply level: {}", ct_result.level);
    println!("After multiply scale: {}", ct_result.scale);

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
        println!("\n✅ PASS: Multiplication works correctly!");
        Ok(())
    } else {
        println!("\n❌ FAIL: Multiplication has errors!");
        Err(format!("Relative error too large: {}", rel_error))
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
    println!("Run with: cargo run --release --features v2,v2-gpu-metal --example test_metal_mult");
}
