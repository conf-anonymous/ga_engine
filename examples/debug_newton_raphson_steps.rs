//! Debug: Test each step of Newton-Raphson separately
//!
//! Newton-Raphson: x_{n+1} = x_n * (2 - a * x_n)
//!
//! For a=7, x_0 ≈ 1/7 ≈ 0.142857:
//!   - a * x_0 should be ≈ 1
//!   - 2 - a*x_0 should be ≈ 1
//!   - x_1 = x_0 * 1 ≈ x_0 ≈ 0.142857

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::{MetalCkksContext, MetalCiphertext, MetalPlaintext},
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
        inversion::subtract_ciphertexts_metal,
    },
    backends::cpu_optimized::keys::KeyContext,
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  DEBUG: Newton-Raphson Step-by-Step                           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();

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

    // Test with a = 7.0
    let a = 7.0;
    let initial_guess = 1.0 / a;  // ≈ 0.142857

    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 1: Basic encryption/decryption");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Encrypt a
    let pt_a = metal_ctx.encode(&[a])?;
    let ct_a = metal_ctx.encrypt(&pt_a, &pk)?;
    let dec_a = metal_ctx.decode(&metal_ctx.decrypt(&ct_a, &sk)?)?[0];
    println!("  Encrypted a = {}, decrypted = {} (error: {:.2e})", a, dec_a, (dec_a - a).abs());

    // Encrypt x_0
    let pt_x0 = metal_ctx.encode(&[initial_guess])?;
    let ct_x0 = metal_ctx.encrypt(&pt_x0, &pk)?;
    let dec_x0 = metal_ctx.decode(&metal_ctx.decrypt(&ct_x0, &sk)?)?[0];
    println!("  Encrypted x_0 = {}, decrypted = {} (error: {:.2e})", initial_guess, dec_x0, (dec_x0 - initial_guess).abs());

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("TEST 2: First Newton-Raphson step breakdown");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Step 1: Compute a * x_0 (should be ≈ 1)
    println!("Step 1: a * x_0 (expected: ~1.0)");
    let ct_ax0 = ct_a.multiply(&ct_x0, &metal_evk, &metal_ctx)?;
    let dec_ax0 = metal_ctx.decode(&metal_ctx.decrypt(&ct_ax0, &sk)?)?[0];
    println!("  Level after mult: {}", ct_ax0.level);
    println!("  Scale: {}", ct_ax0.scale);
    println!("  a * x_0 = {} (expected: {} error: {:.2e})", dec_ax0, a * initial_guess, (dec_ax0 - 1.0).abs());

    if (dec_ax0 - 1.0).abs() > 1.0 {
        println!("  ❌ PROBLEM: a * x_0 is way off from 1.0!");
        return Err("Multiplication result incorrect".to_string());
    }
    println!("  ✅ a * x_0 looks good");

    // Step 2: Create trivial ciphertext for 2.0
    println!("\nStep 2: Trivial ciphertext for 2.0");
    let n = params.n;
    let num_slots = n / 2;
    let mut two_vec = vec![0.0; num_slots];
    two_vec[0] = 2.0;
    let pt_two = MetalPlaintext::encode_at_level(&two_vec, ct_ax0.scale, &params, ct_ax0.level);

    // Create trivial: c0 = pt, c1 = 0
    let ct_two = MetalCiphertext {
        c0: pt_two.coeffs.clone(),
        c1: vec![0u64; n * pt_two.num_primes],
        n,
        num_primes: pt_two.num_primes,
        level: pt_two.level,
        scale: pt_two.scale,
    };

    let dec_two = metal_ctx.decode(&metal_ctx.decrypt(&ct_two, &sk)?)?[0];
    println!("  Level: {}", ct_two.level);
    println!("  Scale: {}", ct_two.scale);
    println!("  Decrypted: {} (expected: 2.0, error: {:.2e})", dec_two, (dec_two - 2.0).abs());

    if (dec_two - 2.0).abs() > 0.1 {
        println!("  ❌ PROBLEM: Trivial ciphertext for 2.0 decrypts incorrectly!");

        // Debug: check first few coefficients
        println!("\n  Debug: pt_two.coeffs[0..{}]:", pt_two.num_primes.min(5));
        for j in 0..pt_two.num_primes.min(5) {
            println!("    prime {}: {}", j, pt_two.coeffs[0 * pt_two.num_primes + j]);
        }
    } else {
        println!("  ✅ Trivial ciphertext for 2.0 looks good");
    }

    // Step 3: Compute 2 - a*x_0 (should be ≈ 1)
    println!("\nStep 3: 2 - a*x_0 (expected: ~1.0)");

    // Verify levels match
    println!("  ct_two level: {}, ct_ax0 level: {}", ct_two.level, ct_ax0.level);
    println!("  ct_two scale: {}, ct_ax0 scale: {}", ct_two.scale, ct_ax0.scale);

    let ct_diff = subtract_ciphertexts_metal(&ct_two, &ct_ax0, &metal_ctx)?;
    let dec_diff = metal_ctx.decode(&metal_ctx.decrypt(&ct_diff, &sk)?)?[0];
    let expected_diff = 2.0 - (a * initial_guess);
    println!("  2 - a*x_0 = {} (expected: {}, error: {:.2e})", dec_diff, expected_diff, (dec_diff - expected_diff).abs());

    if (dec_diff - expected_diff).abs() > 1.0 {
        println!("  ❌ PROBLEM: Subtraction result is way off!");

        // Debug the subtraction
        println!("\n  Debug subtraction components:");
        println!("    ct_two.c0[0..{}]: ", ct_two.num_primes.min(3));
        for j in 0..ct_two.num_primes.min(3) {
            println!("      prime {}: {}", j, ct_two.c0[0 * ct_two.num_primes + j]);
        }
        println!("    ct_ax0.c0[0..{}]: ", ct_ax0.num_primes.min(3));
        for j in 0..ct_ax0.num_primes.min(3) {
            println!("      prime {}: {}", j, ct_ax0.c0[0 * ct_ax0.num_primes + j]);
        }
        println!("    ct_diff.c0[0..{}]: ", ct_diff.num_primes.min(3));
        for j in 0..ct_diff.num_primes.min(3) {
            println!("      prime {}: {}", j, ct_diff.c0[0 * ct_diff.num_primes + j]);
        }
    } else {
        println!("  ✅ Subtraction looks good");
    }

    // Step 4: Need to mod_switch x_0 to match level of (2 - a*x_0) before final mult
    // Note: mod_switch just drops primes without dividing - keeps same scale and value!
    println!("\nStep 4: Mod-switch x_0 to match level");
    let ct_x0_switched = if ct_x0.level > ct_diff.level {
        println!("  Mod-switching from level {} to {}", ct_x0.level, ct_diff.level);
        ct_x0.mod_switch_to_level(ct_diff.level)
    } else {
        ct_x0.clone()
    };

    let dec_x0_switched = metal_ctx.decode(&metal_ctx.decrypt(&ct_x0_switched, &sk)?)?[0];
    println!("  Mod-switched x_0 level: {}", ct_x0_switched.level);
    println!("  Mod-switched x_0 value: {} (expected: {}, error: {:.2e})",
        dec_x0_switched, initial_guess, (dec_x0_switched - initial_guess).abs());

    if (dec_x0_switched - initial_guess).abs() > 0.001 {
        println!("  ❌ Mod-switch changed the value (should preserve it!)");
    } else {
        println!("  ✅ Mod-switch preserved the value correctly");
    }

    // Step 5: Final multiplication x_1 = x_0 * (2 - a*x_0) (should be ≈ x_0)
    println!("\nStep 5: x_1 = x_0 * (2 - a*x_0) (expected: ~{:.6})", initial_guess);

    let ct_x1 = ct_x0_switched.multiply(&ct_diff, &metal_evk, &metal_ctx)?;
    let dec_x1 = metal_ctx.decode(&metal_ctx.decrypt(&ct_x1, &sk)?)?[0];

    // With perfect initial guess, x_1 should be close to x_0 (maybe slightly better)
    // Actually x_1 should converge toward 1/a = 0.142857...
    println!("  x_1 = {} (expected: ~{}, error: {:.2e})", dec_x1, initial_guess, (dec_x1 - initial_guess).abs());

    if (dec_x1 - initial_guess).abs() > 0.01 {
        println!("  ❌ Result diverging significantly from initial guess");
    } else {
        println!("  ✅ First iteration looks reasonable");
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("TEST 3: Full inverse comparison");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Run 3 iterations
    let expected_inverse = 1.0 / a;
    println!("  Computing 1/{} with 3 iterations...", a);
    println!("  Expected: {}", expected_inverse);

    let ct_result = ga_engine::clifford_fhe_v2::backends::gpu_metal::inversion::newton_raphson_inverse_metal(
        &ct_a,
        initial_guess,
        3,
        &metal_evk,
        &pk,
        &metal_ctx,
    )?;

    let dec_result = metal_ctx.decode(&metal_ctx.decrypt(&ct_result, &sk)?)?[0];
    let error = (dec_result - expected_inverse).abs();
    println!("  Result: {} (error: {:.2e})", dec_result, error);

    if error < 0.01 {
        println!("\n✅ Newton-Raphson inverse working correctly!");
        Ok(())
    } else {
        println!("\n❌ Newton-Raphson producing incorrect results");
        Err("Newton-Raphson failed".to_string())
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
