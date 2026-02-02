//! Debug: Check if Metal EVK extraction for level=2 is correct

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::KeyContext,
        ntt::NttContext,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Debug: Metal EVK Extraction for Level 2");
    println!("=======================================\n");

    // Use params with exactly 3 primes
    let params = CliffordFHEParams::new_test_ntt_1024();
    assert_eq!(params.moduli.len(), 3, "Expected 3 primes");

    // Generate keys ONCE - pk, sk, and EVK are all tied together
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, cpu_evk) = key_ctx.keygen();

    let device = Arc::new(MetalDevice::new()?);
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let ntt_contexts = metal_ctx.ntt_contexts();

    // Generate Metal EVK using the SAME sk
    let metal_evk = MetalRelinKeys::generate(
        device.clone(),
        &sk,  // Same sk as used for pk
        &params,
        &ntt_contexts,
        20,
    )?;

    let n = params.n;
    let level = 2;
    let num_primes = level + 1;  // = 3
    let moduli = &params.moduli[..num_primes];

    println!("Params: n={}, level={}, num_primes={}", n, level, num_primes);
    println!("Moduli: {:?}", moduli);

    // Get the EVK for level 2
    let (evk0, evk1) = metal_evk.get_coeff_keys(level)?;

    println!("\nMetal EVK at level {}:", level);
    println!("  Number of digits: {}", evk0.len());
    println!("  evk0[0].len() = {} (expected n * num_primes = {})", evk0[0].len(), n * num_primes);

    // Verify the identity for this extracted EVK
    println!("\nVerifying EVK identity for extracted level {} keys...", level);

    for digit_idx in 0..evk0.len() {
        let max_noise = check_evk_identity(
            &evk0[digit_idx],
            &evk1[digit_idx],
            &sk,
            moduli,
            n,
            digit_idx,
            20,
        )?;

        let ok = max_noise < 1_000_000;
        if digit_idx < 3 || !ok {
            println!("  Digit {}: max_noise = {} {}", digit_idx, max_noise, if ok { "✅" } else { "❌" });
        }
    }

    // Now test multiplication using the same pk, sk, and Metal EVK
    println!("\n=== Testing multiplication with Metal EVK ===");

    // Encrypt value 2 using the SAME pk
    let pt = metal_ctx.encode(&[2.0])?;
    let ct = metal_ctx.encrypt(&pt, &pk)?;

    // Verify encryption worked
    let dec_test = metal_ctx.decrypt(&ct, &sk)?;
    let val_test = metal_ctx.decode(&dec_test)?[0];
    println!("  Encrypted 2, decrypted: {} (error: {:.2e})", val_test, (val_test - 2.0).abs());

    // Square it using Metal EVK
    let ct_squared = ct.multiply(&ct, &metal_evk, &metal_ctx)?;

    // Decrypt
    let dec = metal_ctx.decrypt(&ct_squared, &sk)?;
    let val = metal_ctx.decode(&dec)?[0];

    println!("  After squaring with Metal EVK: {} (expected 4)", val);
    println!("  Error: {:.2e}", (val - 4.0).abs());

    // Also test with CPU EVK for comparison
    println!("\n=== Testing multiplication with CPU EVK (for comparison) ===");

    // Convert CPU EVK to flat format
    let (cpu_evk0_flat, cpu_evk1_flat) = cpu_evk_to_flat(&cpu_evk, num_primes);

    // We need to use the same ciphertext, multiply manually using CPU EVK
    // Actually let's use the CPU multiplication function
    let cpu_ctx = ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext::new(params.clone());
    let cpu_pt = cpu_ctx.encode(&[2.0]);
    let cpu_ct = cpu_ctx.encrypt(&cpu_pt, &pk);
    let cpu_squared = ga_engine::clifford_fhe_v2::backends::cpu_optimized::multiplication::multiply_ciphertexts(
        &cpu_ct, &cpu_ct, &cpu_evk, &key_ctx
    );
    let cpu_dec = cpu_ctx.decrypt(&cpu_squared, &sk);
    let cpu_val = cpu_ctx.decode(&cpu_dec)[0];
    println!("  After squaring with CPU EVK: {} (expected 4)", cpu_val);
    println!("  Error: {:.2e}", (cpu_val - 4.0).abs());

    // Final verdict
    println!("\n=== Summary ===");
    let metal_ok = (val - 4.0).abs() < 1.0;
    let cpu_ok = (cpu_val - 4.0).abs() < 1.0;
    println!("  Metal EVK: {} (result: {})", if metal_ok { "✅" } else { "❌" }, val);
    println!("  CPU EVK:   {} (result: {})", if cpu_ok { "✅" } else { "❌" }, cpu_val);

    if metal_ok {
        Ok(())
    } else {
        Err("Metal EVK multiplication failed".to_string())
    }
}

#[cfg(feature = "v2-gpu-metal")]
fn check_evk_identity(
    evk0: &[u64],
    evk1: &[u64],
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    moduli: &[u64],
    n: usize,
    digit_idx: usize,
    base_w: u32,
) -> Result<i64, String> {
    let num_primes = moduli.len();

    // Compute B^t mod q
    let base = 1u64 << base_w;
    let mut bt_mod_q = vec![0u64; num_primes];
    for (j, &q) in moduli.iter().enumerate() {
        let mut p = 1u128;
        for _ in 0..digit_idx {
            p = (p * (base as u128)) % (q as u128);
        }
        bt_mod_q[j] = p as u64;
    }

    // Compute evk1 * s
    let mut evk1_times_s = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);

        let mut evk1_poly = vec![0u64; n];
        let mut s_poly = vec![0u64; n];
        for i in 0..n {
            evk1_poly[i] = evk1[i * num_primes + prime_idx];
            s_poly[i] = sk.coeffs[i].values[prime_idx];
        }

        let product = ntt_ctx.multiply_polynomials(&evk1_poly, &s_poly);
        for i in 0..n {
            evk1_times_s[i * num_primes + prime_idx] = product[i];
        }
    }

    // Compute s²
    let mut s_sq = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);

        let mut s_poly = vec![0u64; n];
        for i in 0..n {
            s_poly[i] = sk.coeffs[i].values[prime_idx];
        }

        let sq = ntt_ctx.multiply_polynomials(&s_poly, &s_poly);
        for i in 0..n {
            s_sq[i * num_primes + prime_idx] = sq[i];
        }
    }

    // Compute -B^t * s²
    let mut neg_bt_s_sq = vec![0u64; n * num_primes];
    for i in 0..n {
        for (j, &q) in moduli.iter().enumerate() {
            let s2 = s_sq[i * num_primes + j];
            let bt = bt_mod_q[j];
            let bt_s2 = ((bt as u128 * s2 as u128) % q as u128) as u64;
            neg_bt_s_sq[i * num_primes + j] = if bt_s2 == 0 { 0 } else { q - bt_s2 };
        }
    }

    // Compute evk0 - evk1*s
    let mut diff = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        diff[i] = if evk0[i] >= evk1_times_s[i] {
            evk0[i] - evk1_times_s[i]
        } else {
            q - (evk1_times_s[i] - evk0[i])
        };
    }

    // Check max noise
    let mut max_noise: i64 = 0;
    for i in 0..n {
        let d = diff[i * num_primes + 0];
        let expected = neg_bt_s_sq[i * num_primes + 0];
        let q = moduli[0];
        let noise = if d >= expected { d - expected } else { q - (expected - d) };
        let noise_centered = if noise > q/2 { noise as i64 - q as i64 } else { noise as i64 };
        max_noise = max_noise.max(noise_centered.abs());
    }

    Ok(max_noise)
}

#[cfg(feature = "v2-gpu-metal")]
fn cpu_evk_to_flat(
    evk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::EvaluationKey,
    num_primes: usize
) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
    let num_digits = evk.evk0.len();
    let n = evk.evk0[0].len();

    let mut evk0 = Vec::with_capacity(num_digits);
    let mut evk1 = Vec::with_capacity(num_digits);

    for t in 0..num_digits {
        let mut e0 = vec![0u64; n * num_primes];
        let mut e1 = vec![0u64; n * num_primes];
        for i in 0..n {
            for j in 0..num_primes {
                e0[i * num_primes + j] = evk.evk0[t][i].values[j];
                e1[i * num_primes + j] = evk.evk1[t][i].values[j];
            }
        }
        evk0.push(e0);
        evk1.push(e1);
    }

    (evk0, evk1)
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
