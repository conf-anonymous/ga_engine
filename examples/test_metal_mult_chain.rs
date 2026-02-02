//! Test: Metal multiplication chain
//!
//! Verify that multiple multiplications work correctly.

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
    println!("METAL MULTIPLICATION CHAIN TEST\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _cpu_evk) = key_ctx.keygen();

    let device = Arc::new(MetalDevice::new()?);
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let ntt_contexts = metal_ctx.ntt_contexts();

    let metal_evk = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        20,
    )?;

    // Test: 2 × 2 = 4
    println!("=== Test: 2 × 2 = 4 ===");
    let pt = metal_ctx.encode(&[2.0])?;
    let ct = metal_ctx.encrypt(&pt, &pk)?;
    println!("  Encrypted 2.0 at level {}", ct.level);

    let ct_sq = ct.multiply(&ct, &metal_evk, &metal_ctx)?;
    println!("  After multiply, level {}", ct_sq.level);

    let result = metal_ctx.decode(&metal_ctx.decrypt(&ct_sq, &sk)?)?[0];
    println!("  Result: {} (expected 4.0, error: {:.2e})\n", result, (result - 4.0).abs());

    // Test: 2 × 2 × 2 = 8 (two multiplications)
    println!("=== Test: 2 × 2 × 2 = 8 ===");
    let pt = metal_ctx.encode(&[2.0])?;
    let ct = metal_ctx.encrypt(&pt, &pk)?;
    println!("  Encrypted 2.0 at level {}", ct.level);

    let ct_sq1 = ct.multiply(&ct, &metal_evk, &metal_ctx)?;
    println!("  After first multiply, level {}", ct_sq1.level);
    let val1 = metal_ctx.decode(&metal_ctx.decrypt(&ct_sq1, &sk)?)?[0];
    println!("  Intermediate: {} (expected 4.0)", val1);

    // For second multiply, we need a fresh ct at same level
    // Let's use ct_sq1 × a where a is also level 1
    // But ct is level 2, ct_sq1 is level 1... we need to rescale ct first

    // Actually, let's do ct_sq1 × ct_sq1 = 16
    println!("\n=== Test: (2²)² = 16 ===");
    if ct_sq1.level >= 1 {
        let ct_4th = ct_sq1.multiply(&ct_sq1, &metal_evk, &metal_ctx)?;
        println!("  After second multiply, level {}", ct_4th.level);
        let val2 = metal_ctx.decode(&metal_ctx.decrypt(&ct_4th, &sk)?)?[0];
        println!("  Result: {} (expected 16.0, error: {:.2e})", val2, (val2 - 16.0).abs());
    } else {
        println!("  Skipped - level too low");
    }

    // Test: a × b where a != b
    println!("\n=== Test: 3 × 5 = 15 ===");
    let pt_a = metal_ctx.encode(&[3.0])?;
    let pt_b = metal_ctx.encode(&[5.0])?;
    let ct_a = metal_ctx.encrypt(&pt_a, &pk)?;
    let ct_b = metal_ctx.encrypt(&pt_b, &pk)?;
    println!("  Encrypted 3.0 and 5.0 at level {}", ct_a.level);

    let ct_prod = ct_a.multiply(&ct_b, &metal_evk, &metal_ctx)?;
    println!("  After multiply, level {}", ct_prod.level);
    let result_ab = metal_ctx.decode(&metal_ctx.decrypt(&ct_prod, &sk)?)?[0];
    println!("  Result: {} (expected 15.0, error: {:.2e})", result_ab, (result_ab - 15.0).abs());

    Ok(())
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
