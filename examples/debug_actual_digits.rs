//! Debug: Use ACTUAL digits from a real multiply operation
//!
//! Test with the exact same digits that fail in divide_conquer_same_ct.

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
    println!("ACTUAL DIGITS FROM REAL MULTIPLY\n");

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

    // Get EVK keys
    let (metal_evk0, metal_evk1) = metal_evk.get_coeff_keys(level)?;
    let (cpu_evk0, cpu_evk1) = evk_to_flat(&cpu_evk, num_primes);

    // Create ciphertext and get ACTUAL c2
    let metal_pt = metal_ctx.encode(&[2.0])?;
    let metal_ct = metal_ctx.encrypt(&metal_pt, &pk)?;

    // c2 = c1 × c1
    let c2 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&metal_ct.c1, &metal_ct.c1, moduli)?;

    // Gadget decompose c2
    let d2_digits = MetalCiphertext::gadget_decompose_flat(&c2, base_w, moduli, n)?;

    println!("Number of digits: {}", d2_digits.len());
    println!("First 3 digit values at coeff 0:");
    for t in 0..3.min(d2_digits.len()) {
        print!("  digit[{}]: ", t);
        for j in 0..num_primes {
            print!("{} ", d2_digits[t][j]);
        }
        println!();
    }

    // Now compute the relinearization contribution for JUST digit[0]
    println!("\n--- Testing with digit[0] only ---\n");

    let digit0 = &d2_digits[0];

    // CPU path
    let cpu_term0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(digit0, &cpu_evk0[0], moduli)?;
    let cpu_term1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(digit0, &cpu_evk1[0], moduli)?;
    let cpu_term1_s = multiply_by_s(&cpu_term1, &sk, moduli, n);

    // Metal path
    let metal_term0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(digit0, &metal_evk0[0], moduli)?;
    let metal_term1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(digit0, &metal_evk1[0], moduli)?;
    let metal_term1_s = multiply_by_s(&metal_term1, &sk, moduli, n);

    // The contribution to c0 is -term0, to c1 is +term1
    // After decryption: c0 + c1*s changes by -term0 + term1*s
    // For correct relin: -term0 + term1*s = -digit × evk0 + digit × evk1 × s
    //                                     = -digit × (evk0 - evk1×s)
    //                                     = -digit × (-s² + noise)
    //                                     = digit × s²

    // So the net decryption change should be: digit[0] × s² (for t=0, B^0=1)

    // Compute expected: digit[0] × s²
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

    let expected = metal_ctx.multiply_polys_flat_ntt_negacyclic(digit0, &s_sq, moduli)?;

    println!("Expected digit[0]×s² [coeff 0]: {:?}",
        (0..num_primes).map(|j| expected[j]).collect::<Vec<_>>());

    // CPU: -term0 + term1_s = digit × s²
    let cpu_decrypt_change: Vec<u64> = (0..n*num_primes).map(|i| {
        let q = moduli[i % num_primes];
        // -term0 + term1_s
        let neg_term0 = if cpu_term0[i] == 0 { 0 } else { q - cpu_term0[i] };
        ((neg_term0 as u128 + cpu_term1_s[i] as u128) % q as u128) as u64
    }).collect();

    println!("\nCPU: -term0 + term1×s [coeff 0]: {:?}",
        (0..num_primes).map(|j| cpu_decrypt_change[j]).collect::<Vec<_>>());

    let cpu_errors: Vec<i64> = (0..num_primes).map(|j| {
        diff_centered(cpu_decrypt_change[j], expected[j], moduli[j])
    }).collect();
    println!("CPU errors [coeff 0]: {:?} {}",
        cpu_errors, if cpu_errors.iter().all(|&e| e.abs() < 1_000_000) { "✅" } else { "❌" });

    // Metal
    let metal_decrypt_change: Vec<u64> = (0..n*num_primes).map(|i| {
        let q = moduli[i % num_primes];
        let neg_term0 = if metal_term0[i] == 0 { 0 } else { q - metal_term0[i] };
        ((neg_term0 as u128 + metal_term1_s[i] as u128) % q as u128) as u64
    }).collect();

    println!("\nMetal: -term0 + term1×s [coeff 0]: {:?}",
        (0..num_primes).map(|j| metal_decrypt_change[j]).collect::<Vec<_>>());

    let metal_errors: Vec<i64> = (0..num_primes).map(|j| {
        diff_centered(metal_decrypt_change[j], expected[j], moduli[j])
    }).collect();
    println!("Metal errors [coeff 0]: {:?} {}",
        metal_errors, if metal_errors.iter().all(|&e| e.abs() < 1_000_000) { "✅" } else { "❌" });

    // Check ALL coefficients for digit[0]
    println!("\n--- Full coefficient check for digit[0] ---");

    let mut cpu_max: i64 = 0;
    let mut metal_max: i64 = 0;
    let mut cpu_fail = 0;
    let mut metal_fail = 0;

    for i in 0..n {
        for j in 0..num_primes {
            let idx = i * num_primes + j;
            let q = moduli[j];

            let cpu_e = diff_centered(cpu_decrypt_change[idx], expected[idx], q);
            let metal_e = diff_centered(metal_decrypt_change[idx], expected[idx], q);

            cpu_max = cpu_max.max(cpu_e.abs());
            metal_max = metal_max.max(metal_e.abs());

            if cpu_e.abs() > 1_000_000 { cpu_fail += 1; }
            if metal_e.abs() > 1_000_000 { metal_fail += 1; }
        }
    }

    println!("CPU   max error: {}, failures: {}/{} {}",
        cpu_max, cpu_fail, n * num_primes, if cpu_fail == 0 { "✅" } else { "❌" });
    println!("Metal max error: {}, failures: {}/{} {}",
        metal_max, metal_fail, n * num_primes, if metal_fail == 0 { "✅" } else { "❌" });

    // Now test ALL digits accumulated
    println!("\n--- Testing ALL digits accumulated ---\n");

    let mut cpu_total_c0 = vec![0u64; n * num_primes];
    let mut cpu_total_c1 = vec![0u64; n * num_primes];
    let mut metal_total_c0 = vec![0u64; n * num_primes];
    let mut metal_total_c1 = vec![0u64; n * num_primes];

    for (t, digit) in d2_digits.iter().enumerate() {
        if t >= cpu_evk0.len() { break; }

        let cpu_t0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(digit, &cpu_evk0[t], moduli)?;
        let cpu_t1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(digit, &cpu_evk1[t], moduli)?;
        let metal_t0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(digit, &metal_evk0[t], moduli)?;
        let metal_t1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(digit, &metal_evk1[t], moduli)?;

        for i in 0..(n * num_primes) {
            let q = moduli[i % num_primes];
            cpu_total_c0[i] = ((cpu_total_c0[i] as u128 + cpu_t0[i] as u128) % q as u128) as u64;
            cpu_total_c1[i] = ((cpu_total_c1[i] as u128 + cpu_t1[i] as u128) % q as u128) as u64;
            metal_total_c0[i] = ((metal_total_c0[i] as u128 + metal_t0[i] as u128) % q as u128) as u64;
            metal_total_c1[i] = ((metal_total_c1[i] as u128 + metal_t1[i] as u128) % q as u128) as u64;
        }
    }

    // The total should give: -sum(term0) + sum(term1×s) = c2 × s²
    let c2_times_s2 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&c2, &s_sq, moduli)?;

    let cpu_total_change: Vec<u64> = (0..n*num_primes).map(|i| {
        let q = moduli[i % num_primes];
        let neg_c0 = if cpu_total_c0[i] == 0 { 0 } else { q - cpu_total_c0[i] };
        let c1_s = multiply_by_s_single(&cpu_total_c1, &sk, moduli, n, i);
        ((neg_c0 as u128 + c1_s as u128) % q as u128) as u64
    }).collect();

    let metal_total_change: Vec<u64> = (0..n*num_primes).map(|i| {
        let q = moduli[i % num_primes];
        let neg_c0 = if metal_total_c0[i] == 0 { 0 } else { q - metal_total_c0[i] };
        let c1_s = multiply_by_s_single(&metal_total_c1, &sk, moduli, n, i);
        ((neg_c0 as u128 + c1_s as u128) % q as u128) as u64
    }).collect();

    println!("Expected c2×s² [coeff 0]: {:?}",
        (0..num_primes).map(|j| c2_times_s2[j]).collect::<Vec<_>>());
    println!("CPU   total change [coeff 0]: {:?}",
        (0..num_primes).map(|j| cpu_total_change[j]).collect::<Vec<_>>());
    println!("Metal total change [coeff 0]: {:?}",
        (0..num_primes).map(|j| metal_total_change[j]).collect::<Vec<_>>());

    let cpu_total_err: Vec<i64> = (0..num_primes).map(|j| {
        diff_centered(cpu_total_change[j], c2_times_s2[j], moduli[j])
    }).collect();
    let metal_total_err: Vec<i64> = (0..num_primes).map(|j| {
        diff_centered(metal_total_change[j], c2_times_s2[j], moduli[j])
    }).collect();

    println!("\nCPU   total errors [coeff 0]: {:?}", cpu_total_err);
    println!("Metal total errors [coeff 0]: {:?}", metal_total_err);

    Ok(())
}

#[cfg(feature = "v2-gpu-metal")]
fn multiply_by_s(poly: &[u64], sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey, moduli: &[u64], n: usize) -> Vec<u64> {
    let num_primes = moduli.len();
    let mut result = vec![0u64; n * num_primes];

    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);

        let mut poly_prime = vec![0u64; n];
        let mut s_poly = vec![0u64; n];
        for i in 0..n {
            poly_prime[i] = poly[i * num_primes + prime_idx];
            s_poly[i] = sk.coeffs[i].values[prime_idx];
        }

        let product = ntt_ctx.multiply_polynomials(&poly_prime, &s_poly);
        for i in 0..n {
            result[i * num_primes + prime_idx] = product[i];
        }
    }

    result
}

#[cfg(feature = "v2-gpu-metal")]
fn multiply_by_s_single(poly: &[u64], sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey, moduli: &[u64], n: usize, target_idx: usize) -> u64 {
    let num_primes = moduli.len();
    let target_prime = target_idx % num_primes;
    let target_coeff = target_idx / num_primes;
    let q = moduli[target_prime];
    let ntt_ctx = NttContext::new(n, q);

    let mut poly_prime = vec![0u64; n];
    let mut s_poly = vec![0u64; n];
    for i in 0..n {
        poly_prime[i] = poly[i * num_primes + target_prime];
        s_poly[i] = sk.coeffs[i].values[target_prime];
    }

    let product = ntt_ctx.multiply_polynomials(&poly_prime, &s_poly);
    product[target_coeff]
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
