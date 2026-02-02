//! Test EVK identity for ALL digits, not just t=0
//! evk0[t] - evk1[t]*s = -B^t*s² + noise

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
    println!("Test: EVK Identity for ALL Digits");
    println!("==================================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let device = Arc::new(MetalDevice::new()?);
    let key_ctx = KeyContext::new(params.clone());
    let (_, sk, cpu_evk) = key_ctx.keygen();

    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let ntt_contexts = metal_ctx.ntt_contexts();
    let metal_evk = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        20,  // base_w = 20
    )?;

    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];
    let base_w = 20u32;

    // Compute s² (needed to check identity)
    let mut s_squared = vec![vec![0u64; num_primes]; n];
    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);

        let s_poly: Vec<u64> = sk.coeffs.iter().map(|rns| rns.values[prime_idx]).collect();
        let s_sq = ntt_ctx.multiply_polynomials(&s_poly, &s_poly);

        for coeff_idx in 0..n {
            s_squared[coeff_idx][prime_idx] = s_sq[coeff_idx];
        }
    }

    // Precompute B^t mod q for each prime
    let base = 1u64 << base_w;
    let num_digits = cpu_evk.evk0.len();
    let mut b_pow_t_mod_q = vec![vec![0u64; num_primes]; num_digits];
    for (j, &q) in moduli.iter().enumerate() {
        let q_u128 = q as u128;
        let mut p = 1u128;
        for t in 0..num_digits {
            b_pow_t_mod_q[t][j] = (p % q_u128) as u64;
            p = (p * (base as u128)) % q_u128;
        }
    }

    println!("Checking EVK identity for all {} digits...\n", num_digits);

    // Get Metal EVK
    let (metal_evk0, metal_evk1) = metal_evk.get_coeff_keys(level)?;

    // Test each digit
    let mut cpu_failed = false;
    let mut metal_failed = false;

    for t in 0..num_digits {
        // CPU EVK check
        let cpu_max_noise = check_evk_identity_cpu_digit(
            t, &cpu_evk, &sk, &s_squared, &b_pow_t_mod_q[t], moduli, n
        )?;

        // Metal EVK check
        let metal_max_noise = check_evk_identity_metal_digit(
            t, &metal_evk0[t], &metal_evk1[t], &sk, &s_squared, &b_pow_t_mod_q[t], moduli, n
        )?;

        let cpu_ok = cpu_max_noise < 1_000_000;
        let metal_ok = metal_max_noise < 1_000_000;

        println!("Digit {}: CPU max_noise={:>12}, Metal max_noise={:>12}  [CPU: {}, Metal: {}]",
            t, cpu_max_noise, metal_max_noise,
            if cpu_ok { "✅" } else { "❌" },
            if metal_ok { "✅" } else { "❌" }
        );

        if !cpu_ok { cpu_failed = true; }
        if !metal_ok { metal_failed = true; }
    }

    println!();
    if !cpu_failed {
        println!("CPU EVK: ✅ All digits PASSED");
    } else {
        println!("CPU EVK: ❌ Some digits FAILED");
    }
    if !metal_failed {
        println!("Metal EVK: ✅ All digits PASSED");
        Ok(())
    } else {
        println!("Metal EVK: ❌ Some digits FAILED");
        Err("Metal EVK failed identity check".to_string())
    }
}

#[cfg(feature = "v2-gpu-metal")]
fn check_evk_identity_cpu_digit(
    t: usize,
    evk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::EvaluationKey,
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    s_squared: &[Vec<u64>],
    b_pow_t_mod_q: &[u64],
    moduli: &[u64],
    n: usize,
) -> Result<i64, String> {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

    let evk0 = &evk.evk0[t];
    let evk1 = &evk.evk1[t];

    // Compute evk1*s using NTT multiplication
    let mut evk1_times_s = Vec::with_capacity(n);
    for prime_idx in 0..moduli.len() {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);

        let evk1_poly: Vec<u64> = evk1.iter().map(|rns| rns.values[prime_idx]).collect();
        let s_poly: Vec<u64> = sk.coeffs.iter().map(|rns| rns.values[prime_idx]).collect();
        let product = ntt_ctx.multiply_polynomials(&evk1_poly, &s_poly);

        if prime_idx == 0 {
            for coeff_idx in 0..n {
                evk1_times_s.push(RnsRepresentation::new(vec![product[coeff_idx]; moduli.len()], moduli.to_vec()));
            }
        } else {
            for coeff_idx in 0..n {
                evk1_times_s[coeff_idx].values[prime_idx] = product[coeff_idx];
            }
        }
    }

    // Compute evk0 - evk1*s
    let mut diff = Vec::with_capacity(n);
    for coeff_idx in 0..n {
        let mut d = vec![0u64; moduli.len()];
        for (prime_idx, &q) in moduli.iter().enumerate() {
            let e0 = evk0[coeff_idx].values[prime_idx];
            let e1s = evk1_times_s[coeff_idx].values[prime_idx];
            d[prime_idx] = if e0 >= e1s { e0 - e1s } else { q - (e1s - e0) };
        }
        diff.push(d);
    }

    // Compute -B^t * s²
    let mut neg_bt_s_squared = Vec::with_capacity(n);
    for coeff_idx in 0..n {
        let mut vals = vec![0u64; moduli.len()];
        for (prime_idx, &q) in moduli.iter().enumerate() {
            let s2 = s_squared[coeff_idx][prime_idx];
            let bt = b_pow_t_mod_q[prime_idx];
            let bt_s2 = ((bt as u128 * s2 as u128) % q as u128) as u64;
            vals[prime_idx] = if bt_s2 == 0 { 0 } else { q - bt_s2 };
        }
        neg_bt_s_squared.push(vals);
    }

    // Check noise bound
    let mut max_noise: i64 = 0;
    for coeff_idx in 0..n {
        let d0 = diff[coeff_idx][0];
        let neg_bt_s2_0 = neg_bt_s_squared[coeff_idx][0];
        let q = moduli[0];

        let noise = if d0 >= neg_bt_s2_0 { d0 - neg_bt_s2_0 } else { q - (neg_bt_s2_0 - d0) };
        let noise_centered = if noise > q/2 { noise as i64 - q as i64 } else { noise as i64 };
        max_noise = max_noise.max(noise_centered.abs());
    }

    Ok(max_noise)
}

#[cfg(feature = "v2-gpu-metal")]
fn check_evk_identity_metal_digit(
    t: usize,
    evk0: &[u64],
    evk1: &[u64],
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    s_squared: &[Vec<u64>],
    b_pow_t_mod_q: &[u64],
    moduli: &[u64],
    n: usize,
) -> Result<i64, String> {
    let num_primes = moduli.len();

    // Compute evk1*s using NTT multiplication
    let mut evk1_times_s = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);

        // Extract polynomials
        let mut evk1_poly = vec![0u64; n];
        let mut s_poly = vec![0u64; n];
        for coeff_idx in 0..n {
            evk1_poly[coeff_idx] = evk1[coeff_idx * num_primes + prime_idx];
            s_poly[coeff_idx] = sk.coeffs[coeff_idx].values[prime_idx];
        }

        let product = ntt_ctx.multiply_polynomials(&evk1_poly, &s_poly);

        for coeff_idx in 0..n {
            evk1_times_s[coeff_idx * num_primes + prime_idx] = product[coeff_idx];
        }
    }

    // Compute evk0 - evk1*s
    let mut diff = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        let e0 = evk0[i];
        let e1s = evk1_times_s[i];
        diff[i] = if e0 >= e1s { e0 - e1s } else { q - (e1s - e0) };
    }

    // Compute -B^t * s²
    let mut neg_bt_s_squared = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for (prime_idx, &q) in moduli.iter().enumerate() {
            let s2 = s_squared[coeff_idx][prime_idx];
            let bt = b_pow_t_mod_q[prime_idx];
            let bt_s2 = ((bt as u128 * s2 as u128) % q as u128) as u64;
            neg_bt_s_squared[coeff_idx * num_primes + prime_idx] = if bt_s2 == 0 { 0 } else { q - bt_s2 };
        }
    }

    // Check noise bound
    let mut max_noise: i64 = 0;
    for coeff_idx in 0..n {
        let d0 = diff[coeff_idx * num_primes + 0];
        let neg_bt_s2_0 = neg_bt_s_squared[coeff_idx * num_primes + 0];
        let q = moduli[0];

        let noise = if d0 >= neg_bt_s2_0 { d0 - neg_bt_s2_0 } else { q - (neg_bt_s2_0 - d0) };
        let noise_centered = if noise > q/2 { noise as i64 - q as i64 } else { noise as i64 };
        max_noise = max_noise.max(noise_centered.abs());
    }

    Ok(max_noise)
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
