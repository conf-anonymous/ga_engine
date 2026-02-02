//! Divide and Conquer V2: Test actual Metal GPU multiply function
//!
//! Key insight: We need to test the ACTUAL MetalCiphertext::multiply function,
//! not a manual reimplementation. The bug might be in some detail we're missing.

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::KeyContext,
        ckks::CkksContext,
        multiplication::multiply_ciphertexts,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  DIVIDE AND CONQUER V2: Test Actual GPU Multiply Function    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, cpu_evk) = key_ctx.keygen();

    let device = Arc::new(MetalDevice::new()?);
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let cpu_ctx = CkksContext::new(params.clone());
    let ntt_contexts = metal_ctx.ntt_contexts();

    // Generate Metal EVK
    let metal_evk = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        20,
    )?;

    println!("Testing with value: 2.0");
    println!("Expected after squaring: 4.0\n");

    // ═══════════════════════════════════════════════════════════════
    // TEST 1: CPU encryption + CPU multiplication + CPU EVK
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 1: Full CPU Pipeline (reference)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let cpu_pt = cpu_ctx.encode(&[2.0]);
    let cpu_ct = cpu_ctx.encrypt(&cpu_pt, &pk);
    let cpu_squared = multiply_ciphertexts(&cpu_ct, &cpu_ct, &cpu_evk, &key_ctx);
    let cpu_dec = cpu_ctx.decrypt(&cpu_squared, &sk);
    let cpu_val = cpu_ctx.decode(&cpu_dec)[0];

    println!("  CPU result: {} (error: {:.2e})", cpu_val, (cpu_val - 4.0).abs());
    let cpu_ok = (cpu_val - 4.0).abs() < 0.1;
    println!("  {}\n", if cpu_ok { "✅ PASS" } else { "❌ FAIL" });

    // ═══════════════════════════════════════════════════════════════
    // TEST 2: Metal encryption + Metal multiplication + Metal EVK
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 2: Full Metal Pipeline with Metal EVK");
    println!("═══════════════════════════════════════════════════════════════\n");

    let metal_pt = metal_ctx.encode(&[2.0])?;
    let metal_ct = metal_ctx.encrypt(&metal_pt, &pk)?;

    // Verify encryption
    let metal_dec_before = metal_ctx.decrypt(&metal_ct, &sk)?;
    let metal_val_before = metal_ctx.decode(&metal_dec_before)?[0];
    println!("  Before multiply: {} (should be ~2.0)", metal_val_before);

    // Multiply using Metal EVK
    let metal_squared = metal_ct.multiply(&metal_ct, &metal_evk, &metal_ctx)?;

    // Decrypt
    let metal_dec = metal_ctx.decrypt(&metal_squared, &sk)?;
    let metal_val = metal_ctx.decode(&metal_dec)?[0];

    println!("  Metal EVK result: {} (error: {:.2e})", metal_val, (metal_val - 4.0).abs());
    let metal_evk_ok = (metal_val - 4.0).abs() < 0.1;
    println!("  {}\n", if metal_evk_ok { "✅ PASS" } else { "❌ FAIL" });

    // ═══════════════════════════════════════════════════════════════
    // TEST 3: Metal encryption + Metal multiplication + CPU EVK (converted)
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 3: Metal Pipeline with CPU EVK (converted to flat format)");
    println!("═══════════════════════════════════════════════════════════════\n");

    // We need to create a custom multiplication that uses CPU EVK
    // but this isn't directly supported by the API...
    // Let's test differently: use the same ciphertext for both

    println!("  (Skipping - would need custom multiply implementation)\n");

    // ═══════════════════════════════════════════════════════════════
    // TEST 4: Compare c0/c1 values before/after multiply
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 4: Detailed comparison of ciphertext values");
    println!("═══════════════════════════════════════════════════════════════\n");

    let level = cpu_ct.level;
    let num_primes = level + 1;

    println!("  Input ciphertext level: {}", level);
    println!("  Input ciphertext scale: {}", cpu_ct.scale);

    // CPU multiply result
    println!("\n  CPU multiply result:");
    println!("    level: {}", cpu_squared.level);
    println!("    scale: {}", cpu_squared.scale);
    println!("    c0[0] first 3 primes: {:?}",
        (0..num_primes-1).map(|j| cpu_squared.c0[0].values[j]).collect::<Vec<_>>());

    // Metal multiply result
    println!("\n  Metal multiply result:");
    println!("    level: {}", metal_squared.level);
    println!("    scale: {}", metal_squared.scale);
    let metal_num_primes = metal_squared.num_primes;
    println!("    c0[0] first 3 primes: {:?}",
        (0..metal_num_primes).map(|j| metal_squared.c0[0 * metal_num_primes + j]).collect::<Vec<_>>());

    // ═══════════════════════════════════════════════════════════════
    // TEST 5: Different input values
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("TEST 5: Different input values");
    println!("═══════════════════════════════════════════════════════════════\n");

    for val in [1.0, 3.0, 5.0, 10.0] {
        let expected = val * val;

        // CPU
        let cpu_pt = cpu_ctx.encode(&[val]);
        let cpu_ct = cpu_ctx.encrypt(&cpu_pt, &pk);
        let cpu_sq = multiply_ciphertexts(&cpu_ct, &cpu_ct, &cpu_evk, &key_ctx);
        let cpu_result = cpu_ctx.decode(&cpu_ctx.decrypt(&cpu_sq, &sk))[0];

        // Metal
        let metal_pt = metal_ctx.encode(&[val])?;
        let metal_ct = metal_ctx.encrypt(&metal_pt, &pk)?;
        let metal_sq = metal_ct.multiply(&metal_ct, &metal_evk, &metal_ctx)?;
        let metal_result = metal_ctx.decode(&metal_ctx.decrypt(&metal_sq, &sk)?)?[0];

        let cpu_err = (cpu_result - expected).abs();
        let metal_err = (metal_result - expected).abs();

        println!("  {}² = {}: CPU={:.6} (err={:.2e}), Metal={:.6} (err={:.2e}) {}",
            val, expected, cpu_result, cpu_err, metal_result, metal_err,
            if metal_err < 1.0 { "✅" } else { "❌" });
    }

    // ═══════════════════════════════════════════════════════════════
    // TEST 6: Multiply different ciphertexts (not squaring)
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("TEST 6: Multiply different ciphertexts (a × b)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let a = 2.0;
    let b = 3.0;
    let expected = a * b;

    // CPU
    let cpu_pt_a = cpu_ctx.encode(&[a]);
    let cpu_pt_b = cpu_ctx.encode(&[b]);
    let cpu_ct_a = cpu_ctx.encrypt(&cpu_pt_a, &pk);
    let cpu_ct_b = cpu_ctx.encrypt(&cpu_pt_b, &pk);
    let cpu_prod = multiply_ciphertexts(&cpu_ct_a, &cpu_ct_b, &cpu_evk, &key_ctx);
    let cpu_result = cpu_ctx.decode(&cpu_ctx.decrypt(&cpu_prod, &sk))[0];

    // Metal
    let metal_pt_a = metal_ctx.encode(&[a])?;
    let metal_pt_b = metal_ctx.encode(&[b])?;
    let metal_ct_a = metal_ctx.encrypt(&metal_pt_a, &pk)?;
    let metal_ct_b = metal_ctx.encrypt(&metal_pt_b, &pk)?;
    let metal_prod = metal_ct_a.multiply(&metal_ct_b, &metal_evk, &metal_ctx)?;
    let metal_result = metal_ctx.decode(&metal_ctx.decrypt(&metal_prod, &sk)?)?[0];

    println!("  {} × {} = {}", a, b, expected);
    println!("  CPU result: {} (error: {:.2e})", cpu_result, (cpu_result - expected).abs());
    println!("  Metal result: {} (error: {:.2e})", metal_result, (metal_result - expected).abs());

    let test6_ok = (metal_result - expected).abs() < 1.0;
    println!("  {}\n", if test6_ok { "✅ PASS" } else { "❌ FAIL" });

    // ═══════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("  TEST 1 (CPU reference): {}", if cpu_ok { "✅ PASS" } else { "❌ FAIL" });
    println!("  TEST 2 (Metal EVK): {}", if metal_evk_ok { "✅ PASS" } else { "❌ FAIL" });
    println!("  TEST 6 (a × b): {}", if test6_ok { "✅ PASS" } else { "❌ FAIL" });

    if metal_evk_ok && test6_ok {
        println!("\n✅ ALL TESTS PASSED!");
        Ok(())
    } else {
        println!("\n❌ SOME TESTS FAILED");
        Err("Metal EVK multiplication failed".to_string())
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
