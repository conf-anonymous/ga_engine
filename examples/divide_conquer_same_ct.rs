//! Divide and Conquer: Use EXACT SAME Metal ciphertext, apply CPU vs Metal EVK
//!
//! This isolates whether the bug is in EVK usage vs ciphertext generation.

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::{MetalCkksContext, MetalCiphertext},
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::{KeyContext, EvaluationKey, SecretKey},
        ntt::NttContext,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  SAME CIPHERTEXT: CPU EVK vs Metal EVK                        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, cpu_evk) = key_ctx.keygen();

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

    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];

    // Create ONE Metal ciphertext (we'll use the same one for both tests)
    let metal_pt = metal_ctx.encode(&[2.0])?;
    let metal_ct = metal_ctx.encrypt(&metal_pt, &pk)?;

    println!("Created Metal ciphertext for value 2.0");
    println!("  c0[0]: {:?}", extract_coeff(&metal_ct.c0, 0, num_primes));
    println!("  c1[0]: {:?}", extract_coeff(&metal_ct.c1, 0, num_primes));

    // Verify it decrypts correctly before multiply
    let dec_before = metal_ctx.decrypt(&metal_ct, &sk)?;
    let val_before = metal_ctx.decode(&dec_before)?[0];
    println!("  Decrypts to: {} (expected ~2.0)\n", val_before);

    // Step 1: Compute tensor product (same for both)
    println!("Step 1: Tensor product");
    let c0c0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&metal_ct.c0, &metal_ct.c0, moduli)?;
    let c0c1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&metal_ct.c0, &metal_ct.c1, moduli)?;
    let c1c0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&metal_ct.c1, &metal_ct.c0, moduli)?;
    let c1c1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&metal_ct.c1, &metal_ct.c1, moduli)?;

    let d0 = c0c0;
    let mut d1 = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        d1[i] = ((c0c1[i] as u128 + c1c0[i] as u128) % q as u128) as u64;
    }
    let d2 = c1c1;

    println!("  d0[0]: {:?}", extract_coeff(&d0, 0, num_primes));
    println!("  d1[0]: {:?}", extract_coeff(&d1, 0, num_primes));
    println!("  d2[0]: {:?}", extract_coeff(&d2, 0, num_primes));

    // Step 2: Gadget decompose d2 (same for both)
    println!("\nStep 2: Gadget decomposition");
    let base_w = 20u32;
    let d2_digits = MetalCiphertext::gadget_decompose_flat(&d2, base_w, moduli, n)?;
    println!("  Number of digits: {}", d2_digits.len());
    println!("  digit[0][0]: {:?}", extract_coeff(&d2_digits[0], 0, num_primes));

    // Get EVK keys
    let (metal_evk0, metal_evk1) = metal_evk.get_coeff_keys(level)?;
    let (cpu_evk0, cpu_evk1) = evk_to_flat(&cpu_evk, num_primes);

    println!("\nEVK values at digit 0, coeff 0:");
    println!("  CPU evk0:   {:?}", extract_coeff(&cpu_evk0[0], 0, num_primes));
    println!("  Metal evk0: {:?}", extract_coeff(&metal_evk0[0], 0, num_primes));
    println!("  CPU evk1:   {:?}", extract_coeff(&cpu_evk1[0], 0, num_primes));
    println!("  Metal evk1: {:?}", extract_coeff(&metal_evk1[0], 0, num_primes));

    // Step 3: Apply relinearization with CPU EVK
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Relinearization with CPU EVK");
    println!("═══════════════════════════════════════════════════════════════");

    let (cpu_c0_relin, cpu_c1_relin) = apply_relin(
        &d0, &d1, &d2_digits, &cpu_evk0, &cpu_evk1, &metal_ctx, moduli, n
    )?;
    println!("  c0[0] after relin: {:?}", extract_coeff(&cpu_c0_relin, 0, num_primes));
    println!("  c1[0] after relin: {:?}", extract_coeff(&cpu_c1_relin, 0, num_primes));

    // Rescale
    let cpu_c0_rescaled = metal_ctx.exact_rescale_gpu(&cpu_c0_relin, level)?;
    let cpu_c1_rescaled = metal_ctx.exact_rescale_gpu(&cpu_c1_relin, level)?;

    // Decrypt
    let cpu_result = decrypt_result(&cpu_c0_rescaled, &cpu_c1_rescaled, &sk, &params, level, &metal_ctx)?;
    println!("  Result: {} (expected 4.0, error: {:.2e})", cpu_result, (cpu_result - 4.0).abs());

    // Step 4: Apply relinearization with Metal EVK
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Relinearization with Metal EVK");
    println!("═══════════════════════════════════════════════════════════════");

    let (metal_c0_relin, metal_c1_relin) = apply_relin(
        &d0, &d1, &d2_digits, metal_evk0, metal_evk1, &metal_ctx, moduli, n
    )?;
    println!("  c0[0] after relin: {:?}", extract_coeff(&metal_c0_relin, 0, num_primes));
    println!("  c1[0] after relin: {:?}", extract_coeff(&metal_c1_relin, 0, num_primes));

    // Rescale
    let metal_c0_rescaled = metal_ctx.exact_rescale_gpu(&metal_c0_relin, level)?;
    let metal_c1_rescaled = metal_ctx.exact_rescale_gpu(&metal_c1_relin, level)?;

    // Decrypt
    let metal_result = decrypt_result(&metal_c0_rescaled, &metal_c1_rescaled, &sk, &params, level, &metal_ctx)?;
    println!("  Result: {} (expected 4.0, error: {:.2e})", metal_result, (metal_result - 4.0).abs());

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("SUMMARY (using EXACT SAME ciphertext)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  CPU EVK result:   {} (error: {:.2e})", cpu_result, (cpu_result - 4.0).abs());
    println!("  Metal EVK result: {} (error: {:.2e})", metal_result, (metal_result - 4.0).abs());

    let cpu_ok = (cpu_result - 4.0).abs() < 1.0;
    let metal_ok = (metal_result - 4.0).abs() < 1.0;

    println!("\n  CPU EVK:   {}", if cpu_ok { "✅ PASS" } else { "❌ FAIL" });
    println!("  Metal EVK: {}", if metal_ok { "✅ PASS" } else { "❌ FAIL" });

    if cpu_ok && metal_ok {
        println!("\n✅ BOTH PASS - Bug is NOT in EVK!");
        Ok(())
    } else if cpu_ok && !metal_ok {
        println!("\n❌ Metal EVK fails with same ciphertext - BUG IS IN METAL EVK!");
        Err("Metal EVK fails".to_string())
    } else {
        println!("\n❌ Both fail");
        Err("Both fail".to_string())
    }
}

#[cfg(feature = "v2-gpu-metal")]
fn apply_relin(
    d0: &[u64], d1: &[u64], d2_digits: &[Vec<u64>],
    evk0: &[Vec<u64>], evk1: &[Vec<u64>],
    ctx: &MetalCkksContext, moduli: &[u64], n: usize
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let num_primes = moduli.len();
    let mut c0 = d0.to_vec();
    let mut c1 = d1.to_vec();

    for (t, digit) in d2_digits.iter().enumerate() {
        if t >= evk0.len() { break; }

        // term0 = digit × evk0[t]
        let term0 = ctx.multiply_polys_flat_ntt_negacyclic(digit, &evk0[t], moduli)?;
        // term1 = digit × evk1[t]
        let term1 = ctx.multiply_polys_flat_ntt_negacyclic(digit, &evk1[t], moduli)?;

        // c0 -= term0, c1 += term1
        for i in 0..(n * num_primes) {
            let q = moduli[i % num_primes];
            c0[i] = if c0[i] >= term0[i] { c0[i] - term0[i] } else { q - (term0[i] - c0[i]) };
            c1[i] = ((c1[i] as u128 + term1[i] as u128) % q as u128) as u64;
        }
    }

    Ok((c0, c1))
}

#[cfg(feature = "v2-gpu-metal")]
fn decrypt_result(c0: &[u64], c1: &[u64], sk: &SecretKey, params: &CliffordFHEParams, orig_level: usize, ctx: &MetalCkksContext) -> Result<f64, String> {
    // Create a MetalCiphertext from the rescaled values
    let new_level = orig_level - 1;
    let num_primes = new_level + 1;
    let n = params.n;

    // Scale = scale² / q_level
    let new_scale = (params.scale * params.scale) / params.moduli[orig_level] as f64;

    let result_ct = MetalCiphertext {
        c0: c0.to_vec(),
        c1: c1.to_vec(),
        n,
        num_primes,
        level: new_level,
        scale: new_scale,
    };

    // Use the actual Metal decrypt and decode
    let dec = ctx.decrypt(&result_ct, sk)?;
    let val = ctx.decode(&dec)?[0];

    Ok(val)
}

#[cfg(feature = "v2-gpu-metal")]
fn evk_to_flat(evk: &EvaluationKey, num_primes: usize) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
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

#[cfg(feature = "v2-gpu-metal")]
fn extract_coeff(poly: &[u64], coeff_idx: usize, num_primes: usize) -> Vec<u64> {
    (0..num_primes).map(|j| poly[coeff_idx * num_primes + j]).collect()
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
