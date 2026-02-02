//! Debug: Test with REAL d2_digits from actual multiplication
//!
//! The simple d2=[1,0,0,...] test passes. Let's test with real d2 values.

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::{MetalCkksContext, MetalCiphertext},
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::{KeyContext, EvaluationKey},
        ntt::NttContext,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  REAL d2_digits TEST                                          ║");
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
    let base_w = 20u32;

    // Create real ciphertext and compute c2 (d2)
    let metal_pt = metal_ctx.encode(&[2.0])?;
    let metal_ct = metal_ctx.encrypt(&metal_pt, &pk)?;

    println!("Encrypted value 2.0");
    println!("Decrypted before multiply: {}\n", metal_ctx.decode(&metal_ctx.decrypt(&metal_ct, &sk)?)?[0]);

    // Compute c2 = c1 × c1 (squaring)
    let c2 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&metal_ct.c1, &metal_ct.c1, moduli)?;

    println!("c2 = c1 × c1 (polynomial product)");
    println!("c2[coeff=0]: {:?}\n",
        (0..num_primes).map(|j| c2[0 * num_primes + j]).collect::<Vec<_>>());

    // Gadget decompose c2
    let d2_digits = MetalCiphertext::gadget_decompose_flat(&c2, base_w, moduli, n)?;
    println!("Gadget decomposition of c2:");
    println!("  Number of digits: {}", d2_digits.len());
    for t in 0..d2_digits.len().min(4) {
        print!("  digit[{}][coeff=0]: ", t);
        for j in 0..num_primes {
            print!("{} ", d2_digits[t][0 * num_primes + j]);
        }
        println!();
    }

    // Get EVK keys
    let (metal_evk0, metal_evk1) = metal_evk.get_coeff_keys(level)?;
    let (cpu_evk0, cpu_evk1) = evk_to_flat(&cpu_evk, num_primes);

    // Compute s² using CPU NTT
    let mut s_sq = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);
        let s_poly: Vec<u64> = sk.coeffs.iter().map(|rns| rns.values[prime_idx]).collect();
        let sq = ntt_ctx.multiply_polynomials(&s_poly, &s_poly);
        for i in 0..n {
            s_sq[i * num_primes + prime_idx] = sq[i];
        }
    }

    // For this test, we expect:
    // Σ digit[t] × (evk0[t] - evk1[t] × s) = -c2 × s² + noise
    //
    // So: delta_c0 - delta_c1*s should equal -c2 × s²

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Computing expected: -c2 × s²");
    println!("═══════════════════════════════════════════════════════════════");

    // c2 × s²
    let c2_times_s2 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&c2, &s_sq, moduli)?;
    println!("c2 × s² [coeff=0]: {:?}",
        (0..num_primes).map(|j| c2_times_s2[0 * num_primes + j]).collect::<Vec<_>>());

    // -c2 × s²
    let mut neg_c2_times_s2 = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        neg_c2_times_s2[i] = if c2_times_s2[i] == 0 { 0 } else { q - c2_times_s2[i] };
    }
    println!("-c2 × s² [coeff=0]: {:?}",
        (0..num_primes).map(|j| neg_c2_times_s2[0 * num_primes + j]).collect::<Vec<_>>());

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Testing CPU EVK");
    println!("═══════════════════════════════════════════════════════════════");

    let (cpu_delta_c0, cpu_delta_c1) = compute_relin_delta(
        &d2_digits, &cpu_evk0, &cpu_evk1, &metal_ctx, moduli, n
    )?;

    let cpu_check = decrypt_delta_minus(&cpu_delta_c0, &cpu_delta_c1, &sk, moduli, n);
    println!("CPU: delta_c0 - delta_c1*s [coeff=0]: {:?}",
        (0..num_primes).map(|j| cpu_check[0 * num_primes + j]).collect::<Vec<_>>());
    println!("Expected -c2×s² [coeff=0]: {:?}",
        (0..num_primes).map(|j| neg_c2_times_s2[0 * num_primes + j]).collect::<Vec<_>>());

    let cpu_errors: Vec<i64> = (0..num_primes).map(|j| {
        diff_centered(cpu_check[0 * num_primes + j], neg_c2_times_s2[0 * num_primes + j], moduli[j])
    }).collect();
    println!("CPU errors: {:?} {}", cpu_errors, if cpu_errors.iter().all(|&e| e.abs() < 1_000_000) { "✅" } else { "❌" });

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Testing Metal EVK");
    println!("═══════════════════════════════════════════════════════════════");

    let (metal_delta_c0, metal_delta_c1) = compute_relin_delta(
        &d2_digits, metal_evk0, metal_evk1, &metal_ctx, moduli, n
    )?;

    let metal_check = decrypt_delta_minus(&metal_delta_c0, &metal_delta_c1, &sk, moduli, n);
    println!("Metal: delta_c0 - delta_c1*s [coeff=0]: {:?}",
        (0..num_primes).map(|j| metal_check[0 * num_primes + j]).collect::<Vec<_>>());
    println!("Expected -c2×s² [coeff=0]: {:?}",
        (0..num_primes).map(|j| neg_c2_times_s2[0 * num_primes + j]).collect::<Vec<_>>());

    let metal_errors: Vec<i64> = (0..num_primes).map(|j| {
        diff_centered(metal_check[0 * num_primes + j], neg_c2_times_s2[0 * num_primes + j], moduli[j])
    }).collect();
    println!("Metal errors: {:?} {}", metal_errors, if metal_errors.iter().all(|&e| e.abs() < 1_000_000) { "✅" } else { "❌" });

    // Check all coefficients
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Checking all {} coefficients, all {} primes", n, num_primes);
    println!("═══════════════════════════════════════════════════════════════");

    let mut cpu_max_error: i64 = 0;
    let mut metal_max_error: i64 = 0;
    let mut cpu_fail_count = 0;
    let mut metal_fail_count = 0;

    for i in 0..n {
        for j in 0..num_primes {
            let idx = i * num_primes + j;
            let q = moduli[j];
            let expected = neg_c2_times_s2[idx];

            let cpu_err = diff_centered(cpu_check[idx], expected, q);
            let metal_err = diff_centered(metal_check[idx], expected, q);

            cpu_max_error = cpu_max_error.max(cpu_err.abs());
            metal_max_error = metal_max_error.max(metal_err.abs());

            if cpu_err.abs() > 1_000_000 { cpu_fail_count += 1; }
            if metal_err.abs() > 1_000_000 { metal_fail_count += 1; }
        }
    }

    println!("CPU   max_error: {}, failures: {}/{} {}",
        cpu_max_error, cpu_fail_count, n * num_primes,
        if cpu_fail_count == 0 { "✅" } else { "❌" });
    println!("Metal max_error: {}, failures: {}/{} {}",
        metal_max_error, metal_fail_count, n * num_primes,
        if metal_fail_count == 0 { "✅" } else { "❌" });

    if metal_fail_count == 0 {
        println!("\n✅ Both EVKs pass the relin identity with real c2!");
        Ok(())
    } else {
        println!("\n❌ Metal EVK fails with real c2");
        Err("Metal EVK fails".to_string())
    }
}

#[cfg(feature = "v2-gpu-metal")]
fn compute_relin_delta(
    d2_digits: &[Vec<u64>],
    evk0: &[Vec<u64>], evk1: &[Vec<u64>],
    ctx: &MetalCkksContext, moduli: &[u64], n: usize
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let num_primes = moduli.len();
    let mut delta_c0 = vec![0u64; n * num_primes];
    let mut delta_c1 = vec![0u64; n * num_primes];

    for (t, digit) in d2_digits.iter().enumerate() {
        if t >= evk0.len() { break; }

        let term0 = ctx.multiply_polys_flat_ntt_negacyclic(digit, &evk0[t], moduli)?;
        let term1 = ctx.multiply_polys_flat_ntt_negacyclic(digit, &evk1[t], moduli)?;

        for i in 0..(n * num_primes) {
            let q = moduli[i % num_primes];
            delta_c0[i] = ((delta_c0[i] as u128 + term0[i] as u128) % q as u128) as u64;
            delta_c1[i] = ((delta_c1[i] as u128 + term1[i] as u128) % q as u128) as u64;
        }
    }

    Ok((delta_c0, delta_c1))
}

#[cfg(feature = "v2-gpu-metal")]
fn decrypt_delta_minus(c0: &[u64], c1: &[u64], sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey, moduli: &[u64], n: usize) -> Vec<u64> {
    let num_primes = moduli.len();

    // c1 * s
    let mut c1_times_s = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);

        let mut c1_poly = vec![0u64; n];
        let mut s_poly = vec![0u64; n];
        for i in 0..n {
            c1_poly[i] = c1[i * num_primes + prime_idx];
            s_poly[i] = sk.coeffs[i].values[prime_idx];
        }

        let product = ntt_ctx.multiply_polynomials(&c1_poly, &s_poly);
        for i in 0..n {
            c1_times_s[i * num_primes + prime_idx] = product[i];
        }
    }

    // c0 - c1*s
    let mut result = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        result[i] = if c0[i] >= c1_times_s[i] {
            c0[i] - c1_times_s[i]
        } else {
            q - (c1_times_s[i] - c0[i])
        };
    }

    result
}

#[cfg(feature = "v2-gpu-metal")]
fn diff_centered(a: u64, b: u64, q: u64) -> i64 {
    let diff = if a >= b { a - b } else { q - (b - a) };
    if diff > q/2 { diff as i64 - q as i64 } else { diff as i64 }
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

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
