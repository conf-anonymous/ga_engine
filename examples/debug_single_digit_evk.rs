//! Debug: Single digit × single EVK component
//!
//! Isolate the exact multiplication that differs.

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
    println!("SINGLE DIGIT × SINGLE EVK COMPONENT TEST\n");

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

    // Create a simple test digit: [1, 0, 0, ...] for each prime
    let mut digit = vec![0u64; n * num_primes];
    for j in 0..num_primes {
        digit[0 * num_primes + j] = 1;
    }

    println!("Test digit: [1, 0, 0, ...] (constant 1)\n");

    // Multiply digit × evk0[0] using Metal NTT
    let cpu_term0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&digit, &cpu_evk0[0], moduli)?;
    let metal_term0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&digit, &metal_evk0[0], moduli)?;

    println!("digit × evk0[0] (coeff 0):");
    println!("  CPU:   {:?}", (0..num_primes).map(|j| cpu_term0[j]).collect::<Vec<_>>());
    println!("  Metal: {:?}", (0..num_primes).map(|j| metal_term0[j]).collect::<Vec<_>>());

    // For digit=[1,0,0,...], the result should just be evk0[0] (since 1 × poly = poly for constant 1)
    println!("\nExpected (evk0[0] coeff 0):");
    println!("  CPU:   {:?}", (0..num_primes).map(|j| cpu_evk0[0][j]).collect::<Vec<_>>());
    println!("  Metal: {:?}", (0..num_primes).map(|j| metal_evk0[0][j]).collect::<Vec<_>>());

    // Check if multiplication by 1 gives expected result
    let cpu_match = (0..num_primes).all(|j| cpu_term0[j] == cpu_evk0[0][j]);
    let metal_match = (0..num_primes).all(|j| metal_term0[j] == metal_evk0[0][j]);

    println!("\nMultiply by 1 check:");
    println!("  CPU:   {} (term0[0] == evk0[0][0])", if cpu_match { "✅" } else { "❌" });
    println!("  Metal: {} (term0[0] == evk0[0][0])", if metal_match { "✅" } else { "❌" });

    // Now verify the EVK identity for digit=1
    // evk0 - evk1*s should = -s² + noise
    println!("\n--- EVK Identity Check ---");

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

    // CPU: evk0[0] - evk1[0]*s
    let cpu_evk1_times_s = multiply_by_s(&cpu_evk1[0], &sk, moduli, n);
    let cpu_identity: Vec<u64> = (0..n*num_primes).map(|i| {
        let q = moduli[i % num_primes];
        if cpu_evk0[0][i] >= cpu_evk1_times_s[i] {
            cpu_evk0[0][i] - cpu_evk1_times_s[i]
        } else {
            q - (cpu_evk1_times_s[i] - cpu_evk0[0][i])
        }
    }).collect();

    // Metal: evk0[0] - evk1[0]*s
    let metal_evk1_times_s = multiply_by_s(&metal_evk1[0], &sk, moduli, n);
    let metal_identity: Vec<u64> = (0..n*num_primes).map(|i| {
        let q = moduli[i % num_primes];
        if metal_evk0[0][i] >= metal_evk1_times_s[i] {
            metal_evk0[0][i] - metal_evk1_times_s[i]
        } else {
            q - (metal_evk1_times_s[i] - metal_evk0[0][i])
        }
    }).collect();

    // Expected: -s²
    let neg_s_sq: Vec<u64> = (0..n*num_primes).map(|i| {
        let q = moduli[i % num_primes];
        if s_sq[i] == 0 { 0 } else { q - s_sq[i] }
    }).collect();

    println!("evk0[0] - evk1[0]*s (coeff 0):");
    println!("  CPU:      {:?}", (0..num_primes).map(|j| cpu_identity[j]).collect::<Vec<_>>());
    println!("  Metal:    {:?}", (0..num_primes).map(|j| metal_identity[j]).collect::<Vec<_>>());
    println!("  Expected: {:?}", (0..num_primes).map(|j| neg_s_sq[j]).collect::<Vec<_>>());

    // Check error
    let cpu_err: i64 = diff_centered(cpu_identity[0], neg_s_sq[0], moduli[0]);
    let metal_err: i64 = diff_centered(metal_identity[0], neg_s_sq[0], moduli[0]);
    println!("\nIdentity error (coeff 0, prime 0):");
    println!("  CPU:   {} {}", cpu_err, if cpu_err.abs() < 100 { "✅" } else { "❌" });
    println!("  Metal: {} {}", metal_err, if metal_err.abs() < 100 { "✅" } else { "❌" });

    // Now the KEY difference: what happens when we use NTT multiplication?
    println!("\n--- NTT Multiplication Check ---");

    // digit × evk0 using NTT should give same as direct multiplication for digit=1
    // But let's check digit × evk1 × s

    let cpu_digit_evk1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&digit, &cpu_evk1[0], moduli)?;
    let cpu_digit_evk1_s = multiply_by_s(&cpu_digit_evk1, &sk, moduli, n);

    let metal_digit_evk1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&digit, &metal_evk1[0], moduli)?;
    let metal_digit_evk1_s = multiply_by_s(&metal_digit_evk1, &sk, moduli, n);

    println!("digit × evk1[0] × s (coeff 0):");
    println!("  CPU:   {:?}", (0..num_primes).map(|j| cpu_digit_evk1_s[j]).collect::<Vec<_>>());
    println!("  Metal: {:?}", (0..num_primes).map(|j| metal_digit_evk1_s[j]).collect::<Vec<_>>());

    // Final: digit×evk0 - digit×evk1×s
    let cpu_final: Vec<u64> = (0..n*num_primes).map(|i| {
        let q = moduli[i % num_primes];
        if cpu_term0[i] >= cpu_digit_evk1_s[i] {
            cpu_term0[i] - cpu_digit_evk1_s[i]
        } else {
            q - (cpu_digit_evk1_s[i] - cpu_term0[i])
        }
    }).collect();

    let metal_final: Vec<u64> = (0..n*num_primes).map(|i| {
        let q = moduli[i % num_primes];
        if metal_term0[i] >= metal_digit_evk1_s[i] {
            metal_term0[i] - metal_digit_evk1_s[i]
        } else {
            q - (metal_digit_evk1_s[i] - metal_term0[i])
        }
    }).collect();

    println!("\ndigit×evk0 - digit×evk1×s (coeff 0):");
    println!("  CPU:      {:?}", (0..num_primes).map(|j| cpu_final[j]).collect::<Vec<_>>());
    println!("  Metal:    {:?}", (0..num_primes).map(|j| metal_final[j]).collect::<Vec<_>>());
    println!("  Expected: {:?}", (0..num_primes).map(|j| neg_s_sq[j]).collect::<Vec<_>>());

    let cpu_final_err: i64 = diff_centered(cpu_final[0], neg_s_sq[0], moduli[0]);
    let metal_final_err: i64 = diff_centered(metal_final[0], neg_s_sq[0], moduli[0]);
    println!("\nFinal error (coeff 0, prime 0):");
    println!("  CPU:   {} {}", cpu_final_err, if cpu_final_err.abs() < 100 { "✅" } else { "❌" });
    println!("  Metal: {} {}", metal_final_err, if metal_final_err.abs() < 100 { "✅" } else { "❌" });

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
