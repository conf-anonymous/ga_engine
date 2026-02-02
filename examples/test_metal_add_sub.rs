//! Test: Metal addition and subtraction
//!
//! Verify that add/sub operations work correctly.

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
    println!("METAL ADD/SUB TEST\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _cpu_evk) = key_ctx.keygen();

    let device = Arc::new(MetalDevice::new()?);
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let ntt_contexts = metal_ctx.ntt_contexts();

    let _metal_evk = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        20,
    )?;

    // Test: 3 + 5 = 8
    println!("=== Test: 3 + 5 = 8 ===");
    let ct_a = metal_ctx.encrypt(&metal_ctx.encode(&[3.0])?, &pk)?;
    let ct_b = metal_ctx.encrypt(&metal_ctx.encode(&[5.0])?, &pk)?;
    let ct_sum = ct_a.add(&ct_b, &metal_ctx)?;
    let result = metal_ctx.decode(&metal_ctx.decrypt(&ct_sum, &sk)?)?[0];
    println!("  Result: {} (expected 8.0, error: {:.2e})\n", result, (result - 8.0).abs());

    // Test: 10 - 3 = 7
    println!("=== Test: 10 - 3 = 7 ===");
    let ct_a = metal_ctx.encrypt(&metal_ctx.encode(&[10.0])?, &pk)?;
    let ct_b = metal_ctx.encrypt(&metal_ctx.encode(&[3.0])?, &pk)?;
    let ct_diff = ct_a.sub(&ct_b, &metal_ctx)?;
    let result = metal_ctx.decode(&metal_ctx.decrypt(&ct_diff, &sk)?)?[0];
    println!("  Result: {} (expected 7.0, error: {:.2e})\n", result, (result - 7.0).abs());

    // Test: 2 - ct (where ct encrypts something)
    // This is key for Newton-Raphson: 2 - a*x_n
    println!("=== Test: 2 - 0.5 = 1.5 (plaintext - ciphertext) ===");
    let ct_half = metal_ctx.encrypt(&metal_ctx.encode(&[0.5])?, &pk)?;
    // Create trivial ciphertext for 2.0
    let pt_two = metal_ctx.encode(&[2.0])?;
    let ct_two = metal_ctx.trivial_encrypt(&pt_two)?;
    let ct_result = ct_two.sub(&ct_half, &metal_ctx)?;
    let result = metal_ctx.decode(&metal_ctx.decrypt(&ct_result, &sk)?)?[0];
    println!("  Result: {} (expected 1.5, error: {:.2e})\n", result, (result - 1.5).abs());

    // Test Newton-Raphson style: 2 - a*x where a=4, x=0.2 (so a*x=0.8)
    // Then 2 - 0.8 = 1.2
    println!("=== Test Newton-Raphson style: 2 - (4 × 0.2) = 1.2 ===");
    let ct_a = metal_ctx.encrypt(&metal_ctx.encode(&[4.0])?, &pk)?;
    let ct_x = metal_ctx.encrypt(&metal_ctx.encode(&[0.2])?, &pk)?;

    // First multiply: a * x
    let ct_ax = ct_a.multiply(&ct_x, &_metal_evk, &metal_ctx)?;
    let ax_result = metal_ctx.decode(&metal_ctx.decrypt(&ct_ax, &sk)?)?[0];
    println!("  a × x = {} (expected 0.8)", ax_result);

    // Create trivial ct for 2.0 at same level
    let pt_two = metal_ctx.encode_at_level(&[2.0], ct_ax.level)?;
    let ct_two = metal_ctx.trivial_encrypt(&pt_two)?;

    // Then subtract: 2 - a*x
    let ct_2_minus_ax = ct_two.sub(&ct_ax, &metal_ctx)?;
    let result = metal_ctx.decode(&metal_ctx.decrypt(&ct_2_minus_ax, &sk)?)?[0];
    println!("  2 - (a × x) = {} (expected 1.2, error: {:.2e})\n", result, (result - 1.2).abs());

    Ok(())
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
