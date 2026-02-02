//! Debug: Large digit × EVK component
//!
//! Test with digit values similar to actual gadget decomposition.

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
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
    println!("LARGE DIGIT × EVK COMPONENT TEST\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (_pk, sk, cpu_evk) = key_ctx.keygen();

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

    // Get EVK keys
    let (metal_evk0, metal_evk1) = metal_evk.get_coeff_keys(level)?;
    let (cpu_evk0, cpu_evk1) = evk_to_flat(&cpu_evk, num_primes);

    // Create a digit that represents a NEGATIVE value (like from gadget decomposition)
    // digit = -100000 represented mod each prime
    let digit_val: i64 = -100000;
    let mut digit = vec![0u64; n * num_primes];
    for j in 0..num_primes {
        let q = moduli[j];
        digit[0 * num_primes + j] = if digit_val >= 0 {
            digit_val as u64 % q
        } else {
            q - ((-digit_val) as u64 % q)
        };
    }

    println!("Test digit: {} (constant)", digit_val);
    println!("As RNS: {:?}\n", (0..num_primes).map(|j| digit[j]).collect::<Vec<_>>());

    // Compute s²
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

    // For digit × evk identity:
    // digit × evk0 - digit × evk1 × s = digit × (-s²) + noise = -digit × s² + noise
    // Since digit = -100000, we expect: -(-100000) × s² = 100000 × s²

    println!("Expected: digit × (-s²) = {} × s²", -digit_val);

    // Compute digit × (-s²) directly
    let mut expected = vec![0u64; n * num_primes];
    for i in 0..n {
        for j in 0..num_primes {
            let q = moduli[j];
            let idx = i * num_primes + j;
            // digit × (-s²) = digit × (q - s²)
            let neg_s2 = if s_sq[idx] == 0 { 0 } else { q - s_sq[idx] };
            let d = digit[j]; // digit is constant, same for all coeffs
            expected[idx] = ((d as u128 * neg_s2 as u128) % q as u128) as u64;
        }
    }

    println!("digit × (-s²) [coeff 0]: {:?}\n",
        (0..num_primes).map(|j| expected[j]).collect::<Vec<_>>());

    // CPU path
    println!("--- CPU EVK ---");
    let cpu_term0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&digit, &cpu_evk0[0], moduli)?;
    let cpu_term1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&digit, &cpu_evk1[0], moduli)?;
    let cpu_term1_s = multiply_by_s(&cpu_term1, &sk, moduli, n);

    let cpu_result: Vec<u64> = (0..n*num_primes).map(|i| {
        let q = moduli[i % num_primes];
        if cpu_term0[i] >= cpu_term1_s[i] {
            cpu_term0[i] - cpu_term1_s[i]
        } else {
            q - (cpu_term1_s[i] - cpu_term0[i])
        }
    }).collect();

    println!("digit×evk0 - digit×evk1×s [coeff 0]: {:?}",
        (0..num_primes).map(|j| cpu_result[j]).collect::<Vec<_>>());

    let cpu_errors: Vec<i64> = (0..num_primes).map(|j| {
        diff_centered(cpu_result[j], expected[j], moduli[j])
    }).collect();
    println!("Errors: {:?} {}\n", cpu_errors,
        if cpu_errors.iter().all(|&e| e.abs() < 1_000_000) { "✅" } else { "❌" });

    // Metal path
    println!("--- Metal EVK ---");
    let metal_term0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&digit, &metal_evk0[0], moduli)?;
    let metal_term1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&digit, &metal_evk1[0], moduli)?;
    let metal_term1_s = multiply_by_s(&metal_term1, &sk, moduli, n);

    let metal_result: Vec<u64> = (0..n*num_primes).map(|i| {
        let q = moduli[i % num_primes];
        if metal_term0[i] >= metal_term1_s[i] {
            metal_term0[i] - metal_term1_s[i]
        } else {
            q - (metal_term1_s[i] - metal_term0[i])
        }
    }).collect();

    println!("digit×evk0 - digit×evk1×s [coeff 0]: {:?}",
        (0..num_primes).map(|j| metal_result[j]).collect::<Vec<_>>());

    let metal_errors: Vec<i64> = (0..num_primes).map(|j| {
        diff_centered(metal_result[j], expected[j], moduli[j])
    }).collect();
    println!("Errors: {:?} {}\n", metal_errors,
        if metal_errors.iter().all(|&e| e.abs() < 1_000_000) { "✅" } else { "❌" });

    // Check all coefficients
    println!("--- Full coefficient check ---");
    let mut cpu_max_err: i64 = 0;
    let mut metal_max_err: i64 = 0;

    for i in 0..n {
        for j in 0..num_primes {
            let idx = i * num_primes + j;
            let cpu_e = diff_centered(cpu_result[idx], expected[idx], moduli[j]);
            let metal_e = diff_centered(metal_result[idx], expected[idx], moduli[j]);
            cpu_max_err = cpu_max_err.max(cpu_e.abs());
            metal_max_err = metal_max_err.max(metal_e.abs());
        }
    }

    println!("CPU   max error across all coeffs: {} {}", cpu_max_err,
        if cpu_max_err < 1_000_000 { "✅" } else { "❌" });
    println!("Metal max error across all coeffs: {} {}", metal_max_err,
        if metal_max_err < 1_000_000 { "✅" } else { "❌" });

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
