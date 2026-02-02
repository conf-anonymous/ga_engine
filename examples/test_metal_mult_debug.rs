//! Debug Metal GPU ciphertext multiplication step by step

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::keys::KeyContext,
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Testing Metal GPU multiplication with debug output\n");

    // Use smaller parameters for easier debugging
    let params = CliffordFHEParams::new_test_ntt_1024();
    let device = Arc::new(MetalDevice::new()?);
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ctx = MetalCkksContext::new(params.clone())?;

    // Generate relin keys
    let ntt_contexts = ctx.ntt_contexts();
    let relin_keys = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        16,
    )?;

    // Simple test: 2.0 × 3.0 = 6.0
    let a = 2.0;
    let b = 3.0;
    let expected = a * b;

    println!("Test: {} × {} = {}\n", a, b, expected);

    // Encrypt
    let pt_a = ctx.encode(&[a])?;
    let pt_b = ctx.encode(&[b])?;
    let ct_a = ctx.encrypt(&pt_a, &pk)?;
    let ct_b = ctx.encrypt(&pt_b, &pk)?;

    println!("Before multiplication:");
    println!("  ct_a: level={}, scale={}", ct_a.level, ct_a.scale);
    println!("  ct_b: level={}, scale={}", ct_b.level, ct_b.scale);

    // Decrypt inputs to verify
    let pt_a_dec = ctx.decrypt(&ct_a, &sk)?;
    let val_a_dec = ctx.decode(&pt_a_dec)?;
    println!("  Decrypted ct_a: {}", val_a_dec[0]);

    let pt_b_dec = ctx.decrypt(&ct_b, &sk)?;
    let val_b_dec = ctx.decode(&pt_b_dec)?;
    println!("  Decrypted ct_b: {}\n", val_b_dec[0]);

    // Multiply
    let ct_result = ct_a.multiply(&ct_b, &relin_keys, &ctx)?;

    println!("After multiplication:");
    println!("  result: level={}, scale={}\n", ct_result.level, ct_result.scale);

    // Decrypt and decode
    let pt_result = ctx.decrypt(&ct_result, &sk)?;
    let result = ctx.decode(&pt_result)?;

    println!("Results:");
    println!("  Expected: {}", expected);
    println!("  Got:      {}", result[0]);
    println!("  Error:    {:.2e}", (result[0] - expected).abs());
    println!("  Rel err:  {:.2e}", (result[0] - expected).abs() / expected);

    if ((result[0] - expected).abs() / expected) < 1e-6 {
        println!("\n✅ PASS");
        Ok(())
    } else {
        println!("\n❌ FAIL");
        Err(format!("Wrong result: got {} instead of {}", result[0], expected))
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
