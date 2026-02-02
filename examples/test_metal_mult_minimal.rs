//! Minimal test to isolate Metal GPU multiplication bug

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::KeyContext,
        multiplication::multiply_ciphertexts,
        ckks::CkksContext,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Minimal Metal GPU Multiplication Test");
    println!("======================================\n");

    // Use N=1024 for faster debugging
    let params = CliffordFHEParams::new_test_ntt_1024();
    let device = Arc::new(MetalDevice::new()?);
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();

    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let cpu_ctx = CkksContext::new(params.clone());

    // Generate Metal relin keys
    let ntt_contexts = metal_ctx.ntt_contexts();
    let metal_relin_keys = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        20,  // Use same base_w as CPU (20)
    )?;

    // Simple test: 2.0 × 3.0 = 6.0
    let a = 2.0;
    let b = 3.0;
    let expected = a * b;

    println!("Test: {} × {} = {}\n", a, b, expected);

    // === CPU Version ===
    println!("--- CPU Multiplication ---");
    let cpu_pt_a = cpu_ctx.encode(&[a]);
    let cpu_pt_b = cpu_ctx.encode(&[b]);
    let cpu_ct_a = cpu_ctx.encrypt(&cpu_pt_a, &pk);
    let cpu_ct_b = cpu_ctx.encrypt(&cpu_pt_b, &pk);

    println!("Before: level={}, scale={}", cpu_ct_a.level, cpu_ct_a.scale);

    let cpu_ct_result = multiply_ciphertexts(&cpu_ct_a, &cpu_ct_b, &evk, &key_ctx);

    println!("After:  level={}, scale={}", cpu_ct_result.level, cpu_ct_result.scale);

    let cpu_pt_result = cpu_ctx.decrypt(&cpu_ct_result, &sk);

    // Debug: print CPU plaintext coeffs
    println!("CPU plaintext: level={}, scale={}", cpu_pt_result.level, cpu_pt_result.scale);
    println!("CPU coeff[0] across {} primes:", cpu_pt_result.coeffs[0].values.len());
    for (j, &val) in cpu_pt_result.coeffs[0].values.iter().enumerate() {
        println!("  prime[{}]: {}", j, val);
    }

    let cpu_result = cpu_ctx.decode(&cpu_pt_result);

    println!("CPU Result: {}", cpu_result[0]);
    println!("CPU Error:  {:.2e}\n", (cpu_result[0] - expected).abs());

    // === Metal Version ===
    println!("--- Metal GPU Multiplication ---");
    let metal_pt_a = metal_ctx.encode(&[a])?;
    let metal_pt_b = metal_ctx.encode(&[b])?;

    // Sanity check: encrypt and decrypt 'a' without any operations
    let metal_ct_a_test = metal_ctx.encrypt(&metal_pt_a, &pk)?;
    let metal_pt_a_test = metal_ctx.decrypt(&metal_ct_a_test, &sk)?;
    println!("Sanity check - coeff[0] for a={} (encoded then enc/dec):", a);
    for j in 0..metal_pt_a_test.num_primes {
        println!("  prime[{}]: {}", j, metal_pt_a_test.coeffs[0 * metal_pt_a_test.num_primes + j]);
    }
    let a_recovered = metal_ctx.decode(&metal_pt_a_test)?;
    println!("Sanity check - decoded a={}, error={:.2e}\n", a_recovered[0], (a_recovered[0] - a).abs());

    let metal_ct_a = metal_ctx.encrypt(&metal_pt_a, &pk)?;
    let metal_ct_b = metal_ctx.encrypt(&metal_pt_b, &pk)?;

    println!("Before: level={}, scale={}", metal_ct_a.level, metal_ct_a.scale);

    let metal_ct_result = metal_ct_a.multiply(&metal_ct_b, &metal_relin_keys, &metal_ctx)?;

    println!("After:  level={}, scale={}", metal_ct_result.level, metal_ct_result.scale);
    println!("After:  num_primes={}, c0.len()={}, c1.len()={}",
        metal_ct_result.num_primes, metal_ct_result.c0.len(), metal_ct_result.c1.len());

    let metal_pt_result = metal_ctx.decrypt(&metal_ct_result, &sk)?;
    println!("Decrypted plaintext: num_primes={}, coeffs.len()={}, scale={}",
        metal_pt_result.num_primes, metal_pt_result.coeffs.len(), metal_pt_result.scale);
    println!("First few coeffs (prime 0): {:?}",
        &metal_pt_result.coeffs[0..4.min(metal_pt_result.coeffs.len())]);
    println!("coeff[0] across {} primes:", metal_pt_result.num_primes);
    for j in 0..metal_pt_result.num_primes {
        println!("  prime[{}]: {}", j, metal_pt_result.coeffs[0 * metal_pt_result.num_primes + j]);
    }

    let metal_result = metal_ctx.decode(&metal_pt_result)?;

    println!("Metal Result: {}", metal_result[0]);
    println!("Metal Error:  {:.2e}\n", (metal_result[0] - expected).abs());

    // === Comparison ===
    println!("--- Comparison ---");
    println!("Expected:     {}", expected);
    println!("CPU Got:      {} (error: {:.2e})", cpu_result[0], (cpu_result[0] - expected).abs());
    println!("Metal Got:    {} (error: {:.2e})", metal_result[0], (metal_result[0] - expected).abs());

    let cpu_ok = (cpu_result[0] - expected).abs() < 1e-6;
    let metal_ok = (metal_result[0] - expected).abs() < 1e-6;

    println!("\nCPU:   {}", if cpu_ok { "✅ PASS" } else { "❌ FAIL" });
    println!("Metal: {}", if metal_ok { "✅ PASS" } else { "❌ FAIL" });

    if !metal_ok {
        Err("Metal multiplication failed".to_string())
    } else {
        Ok(())
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
