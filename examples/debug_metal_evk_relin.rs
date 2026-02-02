//! Debug Metal EVK - trace relinearization step by step with Metal EVK
//! Compare with what we'd expect if the EVK was correct

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::{MetalCkksContext, MetalCiphertext},
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::KeyContext,
        ckks::CkksContext,
        ntt::NttContext,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Debug Metal EVK Relinearization");
    println!("================================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, cpu_evk) = key_ctx.keygen();

    let device = Arc::new(MetalDevice::new()?);
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let cpu_ctx = CkksContext::new(params.clone());
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
    let base_w = 20u32;

    // Create ciphertext
    let cpu_pt = cpu_ctx.encode(&[2.0]);
    let cpu_ct = cpu_ctx.encrypt(&cpu_pt, &pk);

    // Convert to Metal format
    let metal_ct = cpu_to_metal(&cpu_ct, &params)?;

    // Tensor product: ct * ct gives c2 = c1^2
    let c2 = metal_ctx.multiply_polys_flat_ntt_negacyclic(
        &metal_ct.c1, &metal_ct.c1, moduli
    )?;

    println!("c2[0] (c1^2 at coeff 0): {:?}",
        (0..num_primes).map(|j| c2[0 * num_primes + j]).collect::<Vec<_>>());

    // Gadget decompose c2
    let c2_digits = MetalCiphertext::gadget_decompose_flat(&c2, base_w, moduli, n)?;
    println!("\nGadget decomposition: {} digits", c2_digits.len());
    println!("Digit 0, coeff 0: {:?}",
        (0..num_primes).map(|j| c2_digits[0][0 * num_primes + j]).collect::<Vec<_>>());

    // Get both CPU and Metal EVK
    let (metal_evk0, metal_evk1) = metal_evk.get_coeff_keys(level)?;
    let cpu_evk0_flat = cpu_evk_to_flat_single(&cpu_evk.evk0[0], num_primes);
    let cpu_evk1_flat = cpu_evk_to_flat_single(&cpu_evk.evk1[0], num_primes);

    println!("\n=== EVK Comparison (digit 0, coeff 0) ===");
    println!("CPU EVK0: {:?}",
        (0..num_primes).map(|j| cpu_evk0_flat[0 * num_primes + j]).collect::<Vec<_>>());
    println!("Metal EVK0: {:?}",
        (0..num_primes).map(|j| metal_evk0[0][0 * num_primes + j]).collect::<Vec<_>>());
    println!("CPU EVK1: {:?}",
        (0..num_primes).map(|j| cpu_evk1_flat[0 * num_primes + j]).collect::<Vec<_>>());
    println!("Metal EVK1: {:?}",
        (0..num_primes).map(|j| metal_evk1[0][0 * num_primes + j]).collect::<Vec<_>>());

    // Multiply first digit by EVK (using both CPU and Metal EVK)
    println!("\n=== Digit 0 × EVK0 ===");
    let cpu_term0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(
        &c2_digits[0], &cpu_evk0_flat, moduli
    )?;
    let metal_term0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(
        &c2_digits[0], &metal_evk0[0], moduli
    )?;

    println!("CPU term0[0]: {:?}",
        (0..num_primes).map(|j| cpu_term0[0 * num_primes + j]).collect::<Vec<_>>());
    println!("Metal term0[0]: {:?}",
        (0..num_primes).map(|j| metal_term0[0 * num_primes + j]).collect::<Vec<_>>());

    // Do FULL relinearization with BOTH EVKs
    println!("\n=== Full Relinearization ===");

    // Prepare starting values
    let d0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&metal_ct.c0, &metal_ct.c0, moduli)?;
    let ct0_ct1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&metal_ct.c0, &metal_ct.c1, moduli)?;
    let ct1_ct0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&metal_ct.c1, &metal_ct.c0, moduli)?;
    let mut d1 = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        d1[i] = ((ct0_ct1[i] as u128 + ct1_ct0[i] as u128) % q as u128) as u64;
    }

    // CPU EVK relin
    let (cpu_c0, cpu_c1) = do_relin(&d0, &d1, &c2, &c2_digits,
        &cpu_evk0_to_flat(&cpu_evk, num_primes).0,
        &cpu_evk0_to_flat(&cpu_evk, num_primes).1,
        &metal_ctx, moduli, n)?;

    // Metal EVK relin
    let (metal_c0, metal_c1) = do_relin(&d0, &d1, &c2, &c2_digits,
        metal_evk0, metal_evk1,
        &metal_ctx, moduli, n)?;

    println!("After relinearization (coeff 0):");
    println!("  CPU EVK:   c0={:?}",
        (0..num_primes).map(|j| cpu_c0[0 * num_primes + j]).collect::<Vec<_>>());
    println!("  Metal EVK: c0={:?}",
        (0..num_primes).map(|j| metal_c0[0 * num_primes + j]).collect::<Vec<_>>());
    println!("  CPU EVK:   c1={:?}",
        (0..num_primes).map(|j| cpu_c1[0 * num_primes + j]).collect::<Vec<_>>());
    println!("  Metal EVK: c1={:?}",
        (0..num_primes).map(|j| metal_c1[0 * num_primes + j]).collect::<Vec<_>>());

    // Rescale and decrypt
    let cpu_c0_rs = metal_ctx.exact_rescale_gpu(&cpu_c0, level)?;
    let cpu_c1_rs = metal_ctx.exact_rescale_gpu(&cpu_c1, level)?;
    let metal_c0_rs = metal_ctx.exact_rescale_gpu(&metal_c0, level)?;
    let metal_c1_rs = metal_ctx.exact_rescale_gpu(&metal_c1, level)?;

    // Decrypt
    let new_level = level - 1;
    let new_num_primes = new_level + 1;
    let new_scale = (metal_ct.scale * metal_ct.scale) / moduli[level] as f64;

    let cpu_result = metal_to_cpu_ct(&cpu_c0_rs, &cpu_c1_rs, n, new_level, new_scale, &params)?;
    let metal_result = metal_to_cpu_ct(&metal_c0_rs, &metal_c1_rs, n, new_level, new_scale, &params)?;

    let cpu_dec = cpu_ctx.decrypt(&cpu_result, &sk);
    let metal_dec = cpu_ctx.decrypt(&metal_result, &sk);

    let cpu_val = cpu_ctx.decode(&cpu_dec)[0];
    let metal_val = cpu_ctx.decode(&metal_dec)[0];

    println!("\n=== Final Results ===");
    println!("Expected: 4.0 (2^2)");
    println!("With CPU EVK: {} (error: {:.2e})", cpu_val, (cpu_val - 4.0).abs());
    println!("With Metal EVK: {} (error: {:.2e})", metal_val, (metal_val - 4.0).abs());

    // Now verify EVK identity for the METAL EVK that we actually used
    println!("\n=== Verify Metal EVK Identity (for digit 0) ===");
    verify_evk_identity(&metal_evk0[0], &metal_evk1[0], &sk, moduli, n)?;

    Ok(())
}

#[cfg(feature = "v2-gpu-metal")]
fn do_relin(
    d0: &[u64], d1: &[u64], _d2: &[u64], d2_digits: &[Vec<u64>],
    evk0: &[Vec<u64>], evk1: &[Vec<u64>],
    ctx: &MetalCkksContext, moduli: &[u64], n: usize
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let num_primes = moduli.len();
    let mut c0 = d0.to_vec();
    let mut c1 = d1.to_vec();

    for (t, digit) in d2_digits.iter().enumerate() {
        if t >= evk0.len() { break; }

        let term0 = ctx.multiply_polys_flat_ntt_negacyclic(digit, &evk0[t], moduli)?;
        let term1 = ctx.multiply_polys_flat_ntt_negacyclic(digit, &evk1[t], moduli)?;

        for i in 0..(n * num_primes) {
            let q = moduli[i % num_primes];
            c0[i] = if c0[i] >= term0[i] { c0[i] - term0[i] } else { q - (term0[i] - c0[i]) };
            c1[i] = ((c1[i] as u128 + term1[i] as u128) % q as u128) as u64;
        }
    }

    Ok((c0, c1))
}

#[cfg(feature = "v2-gpu-metal")]
fn verify_evk_identity(
    evk0: &[u64], evk1: &[u64],
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    moduli: &[u64], n: usize
) -> Result<(), String> {
    let num_primes = moduli.len();

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

    // Compute -s^2
    let mut neg_s_sq = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);

        let mut s_poly = vec![0u64; n];
        for i in 0..n {
            s_poly[i] = sk.coeffs[i].values[prime_idx];
        }

        let s_sq = ntt_ctx.multiply_polynomials(&s_poly, &s_poly);
        for i in 0..n {
            neg_s_sq[i * num_primes + prime_idx] = if s_sq[i] == 0 { 0 } else { q - s_sq[i] };
        }
    }

    // Check noise
    let mut max_noise: i64 = 0;
    for i in 0..n {
        let d = diff[i * num_primes + 0];
        let expected = neg_s_sq[i * num_primes + 0];
        let q = moduli[0];
        let noise = if d >= expected { d - expected } else { q - (expected - d) };
        let noise_centered = if noise > q/2 { noise as i64 - q as i64 } else { noise as i64 };
        max_noise = max_noise.max(noise_centered.abs());
    }

    println!("  diff[0] = {}", diff[0]);
    println!("  -s^2[0] = {}", neg_s_sq[0]);
    println!("  max_noise = {} (should be small)", max_noise);

    if max_noise < 1_000_000 {
        println!("  ✅ Identity verified");
    } else {
        println!("  ❌ Identity FAILED");
    }

    Ok(())
}

#[cfg(feature = "v2-gpu-metal")]
fn cpu_to_metal(
    ct: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext,
    params: &CliffordFHEParams
) -> Result<MetalCiphertext, String> {
    let n = ct.c0.len();
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
fn metal_to_cpu_ct(
    c0: &[u64], c1: &[u64], n: usize, level: usize, scale: f64,
    params: &CliffordFHEParams
) -> Result<ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext, String> {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;

    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];

    let mut cpu_c0 = Vec::with_capacity(n);
    let mut cpu_c1 = Vec::with_capacity(n);

    for i in 0..n {
        let mut c0_vals = vec![0u64; num_primes];
        let mut c1_vals = vec![0u64; num_primes];
        for j in 0..num_primes {
            c0_vals[j] = c0[i * num_primes + j];
            c1_vals[j] = c1[i * num_primes + j];
        }
        cpu_c0.push(RnsRepresentation::new(c0_vals, moduli.to_vec()));
        cpu_c1.push(RnsRepresentation::new(c1_vals, moduli.to_vec()));
    }

    Ok(Ciphertext { c0: cpu_c0, c1: cpu_c1, level, scale, n })
}

#[cfg(feature = "v2-gpu-metal")]
fn cpu_evk0_to_flat(
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

#[cfg(feature = "v2-gpu-metal")]
fn cpu_evk_to_flat_single(
    evk_t: &[ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation],
    num_primes: usize
) -> Vec<u64> {
    let n = evk_t.len();
    let mut flat = vec![0u64; n * num_primes];
    for i in 0..n {
        for j in 0..num_primes {
            flat[i * num_primes + j] = evk_t[i].values[j];
        }
    }
    flat
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
