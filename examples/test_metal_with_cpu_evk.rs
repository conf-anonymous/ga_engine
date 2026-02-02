//! Test Metal relinearization using CPU EVK
//! This isolates whether the issue is in Metal EVK generation or Metal relin algorithm

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::{MetalCkksContext, MetalCiphertext},
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::{KeyContext, EvaluationKey},
        multiplication::multiply_ciphertexts,
        ckks::{CkksContext, Ciphertext},
        rns::RnsRepresentation,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Test Metal Relin with CPU EVK");
    println!("==============================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();

    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let cpu_ctx = CkksContext::new(params.clone());

    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];

    // Create CPU ciphertext
    let cpu_pt_a = cpu_ctx.encode(&[2.0]);
    let cpu_pt_b = cpu_ctx.encode(&[3.0]);
    let cpu_ct_a = cpu_ctx.encrypt(&cpu_pt_a, &pk);
    let cpu_ct_b = cpu_ctx.encrypt(&cpu_pt_b, &pk);

    // CPU multiplication (reference)
    println!("=== CPU Multiplication (reference) ===");
    let cpu_result = multiply_ciphertexts(&cpu_ct_a, &cpu_ct_b, &evk, &key_ctx);
    let cpu_dec = cpu_ctx.decrypt(&cpu_result, &sk);
    let cpu_val = cpu_ctx.decode(&cpu_dec)[0];
    println!("CPU result: {} (error: {:.2e})\n", cpu_val, (cpu_val - 6.0).abs());

    // Convert CPU ciphertext to Metal format
    let metal_ct_a = cpu_to_metal(&cpu_ct_a, &params)?;
    let metal_ct_b = cpu_to_metal(&cpu_ct_b, &params)?;

    // Convert CPU EVK to flat format for Metal
    let (evk0_flat, evk1_flat) = cpu_evk_to_flat(&evk, num_primes)?;

    println!("=== Metal Relin with CPU EVK ===");
    println!("CPU EVK converted to Metal format");
    println!("  evk0_flat[0] size: {}", evk0_flat[0].len());
    println!("  evk0_flat[0][coeff=0]: {:?}",
        (0..num_primes).map(|j| evk0_flat[0][0 * num_primes + j]).collect::<Vec<_>>());

    // Verify conversion matches
    println!("  CPU evk0[0][coeff=0]: {:?}",
        (0..num_primes).map(|j| evk.evk0[0][0].values[j]).collect::<Vec<_>>());

    // Do Metal multiplication using CPU EVK
    let metal_result = metal_multiply_with_evk(
        &metal_ct_a, &metal_ct_b,
        &evk0_flat, &evk1_flat,
        evk.base_w,
        &metal_ctx, &params
    )?;

    // Decrypt with CPU
    let result_cpu_format = metal_to_cpu(&metal_result, &params)?;
    let metal_dec = cpu_ctx.decrypt(&result_cpu_format, &sk);
    let metal_val = cpu_ctx.decode(&metal_dec)[0];

    println!("\nMetal result (using CPU EVK): {} (error: {:.2e})", metal_val, (metal_val - 6.0).abs());

    println!("\n=== Comparison ===");
    println!("Expected:     6.0");
    println!("CPU Got:      {} (error: {:.2e})", cpu_val, (cpu_val - 6.0).abs());
    println!("Metal Got:    {} (error: {:.2e})", metal_val, (metal_val - 6.0).abs());

    let cpu_evk_ok = (metal_val - 6.0).abs() < 1.0;
    if cpu_evk_ok {
        println!("\n✅ Metal relin with CPU EVK PASSED!");
    } else {
        println!("\n❌ Metal relin with CPU EVK FAILED!");
    }

    // Now test with Metal-generated EVK
    println!("\n=== Metal Relin with Metal EVK ===");
    let device = Arc::new(MetalDevice::new()?);
    let ntt_contexts = metal_ctx.ntt_contexts();
    let metal_relin_keys = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        20,
    )?;

    let (metal_evk0, metal_evk1) = metal_relin_keys.get_coeff_keys(level)?;
    println!("Metal EVK retrieved");
    println!("  metal_evk0[0] size: {}", metal_evk0[0].len());
    println!("  metal_evk0[0][coeff=0]: {:?}",
        (0..num_primes).map(|j| metal_evk0[0][0 * num_primes + j]).collect::<Vec<_>>());

    // Do Metal multiplication using Metal EVK
    let metal_result2 = metal_multiply_with_evk(
        &metal_ct_a, &metal_ct_b,
        metal_evk0, metal_evk1,
        20,  // base_w
        &metal_ctx, &params
    )?;

    let result2_cpu = metal_to_cpu(&metal_result2, &params)?;
    let metal_dec2 = cpu_ctx.decrypt(&result2_cpu, &sk);
    let metal_val2 = cpu_ctx.decode(&metal_dec2)[0];

    println!("\nMetal result (using Metal EVK): {} (error: {:.2e})", metal_val2, (metal_val2 - 6.0).abs());

    let metal_evk_ok = (metal_val2 - 6.0).abs() < 1.0;
    if metal_evk_ok {
        println!("\n✅ Metal relin with Metal EVK PASSED!");
    } else {
        println!("\n❌ Metal relin with Metal EVK FAILED!");
    }

    println!("\n=== Summary ===");
    println!("CPU EVK test: {}", if cpu_evk_ok { "✅" } else { "❌" });
    println!("Metal EVK test: {}", if metal_evk_ok { "✅" } else { "❌" });

    if cpu_evk_ok && metal_evk_ok {
        Ok(())
    } else if cpu_evk_ok && !metal_evk_ok {
        Err("Metal EVK generation is broken".to_string())
    } else {
        Err("Metal relin algorithm is broken".to_string())
    }
}

#[cfg(feature = "v2-gpu-metal")]
fn cpu_to_metal(ct: &Ciphertext, params: &CliffordFHEParams) -> Result<MetalCiphertext, String> {
    let n = params.n;
    let num_primes = ct.level + 1;

    let mut c0 = vec![0u64; n * num_primes];
    let mut c1 = vec![0u64; n * num_primes];

    for i in 0..n {
        for j in 0..num_primes {
            c0[i * num_primes + j] = ct.c0[i].values[j];
            c1[i * num_primes + j] = ct.c1[i].values[j];
        }
    }

    Ok(MetalCiphertext {
        c0, c1, n, num_primes,
        level: ct.level, scale: ct.scale,
    })
}

#[cfg(feature = "v2-gpu-metal")]
fn metal_to_cpu(ct: &MetalCiphertext, params: &CliffordFHEParams) -> Result<Ciphertext, String> {
    let n = ct.n;
    let num_primes = ct.num_primes;
    let moduli = &params.moduli[..num_primes];

    let mut c0 = Vec::with_capacity(n);
    let mut c1 = Vec::with_capacity(n);

    for i in 0..n {
        let mut c0_vals = vec![0u64; num_primes];
        let mut c1_vals = vec![0u64; num_primes];
        for j in 0..num_primes {
            c0_vals[j] = ct.c0[i * num_primes + j];
            c1_vals[j] = ct.c1[i * num_primes + j];
        }
        c0.push(RnsRepresentation::new(c0_vals, moduli.to_vec()));
        c1.push(RnsRepresentation::new(c1_vals, moduli.to_vec()));
    }

    Ok(Ciphertext { c0, c1, level: ct.level, scale: ct.scale, n })
}

#[cfg(feature = "v2-gpu-metal")]
fn cpu_evk_to_flat(evk: &EvaluationKey, num_primes: usize) -> Result<(Vec<Vec<u64>>, Vec<Vec<u64>>), String> {
    let num_digits = evk.evk0.len();
    let n = evk.evk0[0].len();

    let mut evk0_flat = Vec::with_capacity(num_digits);
    let mut evk1_flat = Vec::with_capacity(num_digits);

    for t in 0..num_digits {
        let mut e0 = vec![0u64; n * num_primes];
        let mut e1 = vec![0u64; n * num_primes];

        for i in 0..n {
            for j in 0..num_primes {
                e0[i * num_primes + j] = evk.evk0[t][i].values[j];
                e1[i * num_primes + j] = evk.evk1[t][i].values[j];
            }
        }

        evk0_flat.push(e0);
        evk1_flat.push(e1);
    }

    Ok((evk0_flat, evk1_flat))
}

#[cfg(feature = "v2-gpu-metal")]
fn metal_multiply_with_evk(
    ct_a: &MetalCiphertext,
    ct_b: &MetalCiphertext,
    evk0: &[Vec<u64>],
    evk1: &[Vec<u64>],
    base_w: u32,
    ctx: &MetalCkksContext,
    params: &CliffordFHEParams,
) -> Result<MetalCiphertext, String> {
    let n = ct_a.n;
    let level = ct_a.level;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];

    // Step 1: Tensor product
    let ct0_ct0 = ctx.multiply_polys_flat_ntt_negacyclic(&ct_a.c0, &ct_b.c0, moduli)?;
    let ct0_ct1 = ctx.multiply_polys_flat_ntt_negacyclic(&ct_a.c0, &ct_b.c1, moduli)?;
    let ct1_ct0 = ctx.multiply_polys_flat_ntt_negacyclic(&ct_a.c1, &ct_b.c0, moduli)?;
    let ct1_ct1 = ctx.multiply_polys_flat_ntt_negacyclic(&ct_a.c1, &ct_b.c1, moduli)?;

    // d0 = ct0_a * ct0_b
    let d0 = ct0_ct0;

    // d1 = ct0_a * ct1_b + ct1_a * ct0_b
    let mut d1 = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        d1[i] = ((ct0_ct1[i] as u128 + ct1_ct0[i] as u128) % q as u128) as u64;
    }

    // d2 = ct1_a * ct1_b (to be relinearized)
    let d2 = ct1_ct1;

    // Step 2: Relinearization
    let mut c0 = d0.clone();
    let mut c1 = d1.clone();

    // Gadget decompose d2
    let d2_digits = MetalCiphertext::gadget_decompose_flat(&d2, base_w, moduli, n)?;

    println!("  Metal relinearization:");
    println!("    Num digits: {}", d2_digits.len());
    println!("    Using {} EVK components", evk0.len());

    // Accumulate
    for (t, d2_digit) in d2_digits.iter().enumerate() {
        if t >= evk0.len() {
            break;
        }

        let term0 = ctx.multiply_polys_flat_ntt_negacyclic(d2_digit, &evk0[t], moduli)?;
        let term1 = ctx.multiply_polys_flat_ntt_negacyclic(d2_digit, &evk1[t], moduli)?;

        // c0 -= term0, c1 += term1
        for i in 0..(n * num_primes) {
            let q = moduli[i % num_primes];
            c0[i] = if c0[i] >= term0[i] { c0[i] - term0[i] } else { q - (term0[i] - c0[i]) };
            c1[i] = ((c1[i] as u128 + term1[i] as u128) % q as u128) as u64;
        }
    }

    // Step 3: Rescale
    let c0_rescaled = ctx.exact_rescale_gpu(&c0, level)?;
    let c1_rescaled = ctx.exact_rescale_gpu(&c1, level)?;

    let new_level = level - 1;
    let new_scale = (ct_a.scale * ct_b.scale) / params.moduli[level] as f64;

    Ok(MetalCiphertext {
        c0: c0_rescaled,
        c1: c1_rescaled,
        n,
        num_primes: new_level + 1,
        level: new_level,
        scale: new_scale,
    })
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
