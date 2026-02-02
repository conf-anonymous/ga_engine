//! Debug: Check each digit's EVK identity separately
//!
//! Test if the issue is with specific digits.

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
    println!("PER-DIGIT EVK TEST\n");

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

    // Get actual c2 from a real multiply
    let metal_pt = metal_ctx.encode(&[2.0])?;
    let metal_ct = metal_ctx.encrypt(&metal_pt, &pk)?;
    let c2 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&metal_ct.c1, &metal_ct.c1, moduli)?;
    let d2_digits = MetalCiphertext::gadget_decompose_flat(&c2, base_w, moduli, n)?;

    // Compute s² for each prime
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

    let base = 1u64 << base_w;

    println!("Testing each digit separately:\n");

    for t in 0..d2_digits.len().min(8) {
        // Compute B^t mod each prime
        let bt: Vec<u64> = moduli.iter().map(|&q| {
            let mut p = 1u128;
            for _ in 0..t {
                p = (p * base as u128) % q as u128;
            }
            p as u64
        }).collect();

        let digit = &d2_digits[t];

        // Expected: digit[t] × B^t × s²
        let digit_bt_s2 = compute_digit_bt_s2(digit, &bt, &s_sq, moduli, n, &metal_ctx)?;

        // CPU: digit × evk0[t] - digit × evk1[t] × s should = -digit × B^t × s² + noise
        let cpu_term0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(digit, &cpu_evk0[t], moduli)?;
        let cpu_term1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(digit, &cpu_evk1[t], moduli)?;
        let cpu_term1_s = multiply_by_s(&cpu_term1, &sk, moduli, n);

        let cpu_result: Vec<u64> = (0..n*num_primes).map(|i| {
            let q = moduli[i % num_primes];
            if cpu_term0[i] >= cpu_term1_s[i] {
                cpu_term0[i] - cpu_term1_s[i]
            } else {
                q - (cpu_term1_s[i] - cpu_term0[i])
            }
        }).collect();

        // Metal
        let metal_term0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(digit, &metal_evk0[t], moduli)?;
        let metal_term1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(digit, &metal_evk1[t], moduli)?;
        let metal_term1_s = multiply_by_s(&metal_term1, &sk, moduli, n);

        let metal_result: Vec<u64> = (0..n*num_primes).map(|i| {
            let q = moduli[i % num_primes];
            if metal_term0[i] >= metal_term1_s[i] {
                metal_term0[i] - metal_term1_s[i]
            } else {
                q - (metal_term1_s[i] - metal_term0[i])
            }
        }).collect();

        // Expected: -digit × B^t × s²
        let neg_digit_bt_s2: Vec<u64> = (0..n*num_primes).map(|i| {
            let q = moduli[i % num_primes];
            if digit_bt_s2[i] == 0 { 0 } else { q - digit_bt_s2[i] }
        }).collect();

        // Check coeff 0 errors
        let cpu_errors: Vec<i64> = (0..num_primes).map(|j| {
            diff_centered(cpu_result[j], neg_digit_bt_s2[j], moduli[j])
        }).collect();
        let metal_errors: Vec<i64> = (0..num_primes).map(|j| {
            diff_centered(metal_result[j], neg_digit_bt_s2[j], moduli[j])
        }).collect();

        let cpu_consistent = cpu_errors.iter().all(|&e| (e - cpu_errors[0]).abs() < 1000);
        let metal_consistent = metal_errors.iter().all(|&e| (e - metal_errors[0]).abs() < 1000);

        println!("Digit {}:", t);
        println!("  CPU   errors: {:?} {}", cpu_errors,
            if cpu_consistent { "✅ consistent" } else { "❌ inconsistent" });
        println!("  Metal errors: {:?} {}", metal_errors,
            if metal_consistent { "✅ consistent" } else { "❌ inconsistent" });
        println!();
    }

    Ok(())
}

#[cfg(feature = "v2-gpu-metal")]
fn compute_digit_bt_s2(
    digit: &[u64], bt: &[u64], s_sq: &[u64],
    moduli: &[u64], n: usize, ctx: &MetalCkksContext
) -> Result<Vec<u64>, String> {
    let num_primes = moduli.len();

    // First compute digit × B^t
    let mut digit_bt = vec![0u64; n * num_primes];
    for i in 0..n {
        for j in 0..num_primes {
            let idx = i * num_primes + j;
            let q = moduli[j];
            digit_bt[idx] = ((digit[idx] as u128 * bt[j] as u128) % q as u128) as u64;
        }
    }

    // Then compute digit_bt × s²
    ctx.multiply_polys_flat_ntt_negacyclic(&digit_bt, s_sq, moduli)
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
