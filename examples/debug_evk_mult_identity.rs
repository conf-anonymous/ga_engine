//! Debug: Test EVK multiplication identity
//!
//! For relinearization to work:
//! Σ_t digit[t] × (evk0[t] - evk1[t] × s) = Σ_t digit[t] × (-B^t × s² + error)
//!                                        = -Σ_t digit[t] × B^t × s² + noise
//!                                        = -d2 × s² + noise
//!
//! So: c0 - Σ digit[t] × evk0[t] + (c1 + Σ digit[t] × evk1[t]) × s
//!   = c0 + c1*s - d2*s² + noise
//!   = d0 + d1*s (original message without s² term)

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
    println!("║  EVK MULTIPLICATION IDENTITY TEST                             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

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
    let base_w = 20u32;

    // Get EVK keys
    let (metal_evk0, metal_evk1) = metal_evk.get_coeff_keys(level)?;
    let (cpu_evk0, cpu_evk1) = evk_to_flat(&cpu_evk, num_primes);

    println!("Test: Using d2 = [1, 0, 0, ...] (constant 1)");
    println!("After gadget decomposition: digit[0] = [1, 0, ...], rest = [0, 0, ...]\n");

    // Create d2 = constant 1
    let mut d2 = vec![0u64; n * num_primes];
    for j in 0..num_primes {
        d2[0 * num_primes + j] = 1;
    }

    // Gadget decompose: for d2=1, we get digit[0] = [1,1,1], rest = [0,0,0]
    let d2_digits = MetalCiphertext::gadget_decompose_flat(&d2, base_w, moduli, n)?;
    println!("Gadget decomposition result:");
    for t in 0..d2_digits.len().min(3) {
        print!("  digit[{}][coeff=0]: ", t);
        for j in 0..num_primes {
            print!("{} ", d2_digits[t][0 * num_primes + j]);
        }
        println!();
    }

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
    println!("\ns²[coeff=0]: {:?}",
        (0..num_primes).map(|j| s_sq[0 * num_primes + j]).collect::<Vec<_>>());

    // For d2 = 1, the relin contribution should be:
    // delta_c0 = Σ digit[t] × evk0[t] (we subtract this)
    // delta_c1 = Σ digit[t] × evk1[t] (we add this)
    //
    // And after decrypting: delta_c0 + delta_c1 × s = -d2 × s² + noise = -s² + noise

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Testing CPU EVK");
    println!("═══════════════════════════════════════════════════════════════");

    let (cpu_delta_c0, cpu_delta_c1) = compute_relin_delta(
        &d2_digits, &cpu_evk0, &cpu_evk1, &metal_ctx, moduli, n
    )?;

    // Decrypt: delta_c0 + delta_c1 × s
    let cpu_decrypted = decrypt_delta(&cpu_delta_c0, &cpu_delta_c1, &sk, moduli, n);
    println!("delta_c0 + delta_c1*s [coeff=0, prime=0] = {}", cpu_decrypted[0]);

    // This should equal -s² (negated, since we subtract)
    // But wait - in relin we do c0 -= delta, so we're adding -delta to c0
    // So delta should be +s² (since evk encodes -s²)
    // Actually: evk0 - evk1*s = -B^t*s² + e
    // So: digit × evk0 - digit × evk1 × s = -digit × B^t × s² + noise
    // For digit[0] = 1: delta = -s² + noise
    //
    // In relin: c0 -= digit × evk0 = c0 - delta_evk0
    //           c1 += digit × evk1
    // So: c0_new + c1_new*s = c0 - delta_evk0 + (c1 + delta_evk1)*s
    //                       = c0 + c1*s - delta_evk0 + delta_evk1*s
    //                       = c0 + c1*s - (delta_evk0 - delta_evk1*s)
    //                       = c0 + c1*s - (-digit*B^t*s² + e)
    //                       = c0 + c1*s + digit*B^t*s² - e
    //
    // This cancels the s² term: (d0 + d1*s + d2*s²) -> (d0 + d1*s + d2*s² - d2*s²) = d0 + d1*s
    //
    // So: delta_evk0 - delta_evk1*s = -digit*B^t*s² + noise

    // For digit[0]=1, B^0=1: delta_c0 - delta_c1*s should be -s²
    let cpu_check = decrypt_delta_minus(&cpu_delta_c0, &cpu_delta_c1, &sk, moduli, n);
    println!("delta_c0 - delta_c1*s [coeff=0, prime=0] = {}", cpu_check[0]);

    // Expected: -s²[0]
    let q0 = moduli[0];
    let expected_neg_s2 = if s_sq[0] == 0 { 0 } else { q0 - s_sq[0] };
    println!("Expected -s²[0, prime=0] = {}", expected_neg_s2);

    let cpu_error = diff_centered(cpu_check[0], expected_neg_s2, q0);
    println!("CPU error: {} {}", cpu_error, if cpu_error.abs() < 1_000_000 { "✅" } else { "❌" });

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Testing Metal EVK");
    println!("═══════════════════════════════════════════════════════════════");

    let (metal_delta_c0, metal_delta_c1) = compute_relin_delta(
        &d2_digits, metal_evk0, metal_evk1, &metal_ctx, moduli, n
    )?;

    let metal_check = decrypt_delta_minus(&metal_delta_c0, &metal_delta_c1, &sk, moduli, n);
    println!("delta_c0 - delta_c1*s [coeff=0, prime=0] = {}", metal_check[0]);
    println!("Expected -s²[0, prime=0] = {}", expected_neg_s2);

    let metal_error = diff_centered(metal_check[0], expected_neg_s2, q0);
    println!("Metal error: {} {}", metal_error, if metal_error.abs() < 1_000_000 { "✅" } else { "❌" });

    // Check all coefficients
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Checking all {} coefficients", n);
    println!("═══════════════════════════════════════════════════════════════");

    let mut cpu_max_error: i64 = 0;
    let mut metal_max_error: i64 = 0;
    let mut cpu_fail_count = 0;
    let mut metal_fail_count = 0;

    for i in 0..n {
        let expected = if s_sq[i * num_primes] == 0 { 0 } else { q0 - s_sq[i * num_primes] };

        let cpu_err = diff_centered(cpu_check[i * num_primes], expected, q0);
        let metal_err = diff_centered(metal_check[i * num_primes], expected, q0);

        cpu_max_error = cpu_max_error.max(cpu_err.abs());
        metal_max_error = metal_max_error.max(metal_err.abs());

        if cpu_err.abs() > 1_000_000 { cpu_fail_count += 1; }
        if metal_err.abs() > 1_000_000 { metal_fail_count += 1; }
    }

    println!("CPU   max_error: {}, failures: {}/{} {}",
        cpu_max_error, cpu_fail_count, n, if cpu_fail_count == 0 { "✅" } else { "❌" });
    println!("Metal max_error: {}, failures: {}/{} {}",
        metal_max_error, metal_fail_count, n, if metal_fail_count == 0 { "✅" } else { "❌" });

    if metal_fail_count == 0 {
        println!("\n✅ Both pass - bug might be elsewhere");
        Ok(())
    } else {
        println!("\n❌ Metal fails - bug confirmed in EVK multiplication identity");
        Err("Metal EVK fails multiplication identity".to_string())
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
fn decrypt_delta(c0: &[u64], c1: &[u64], sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey, moduli: &[u64], n: usize) -> Vec<u64> {
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

    // c0 + c1*s
    let mut result = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        result[i] = ((c0[i] as u128 + c1_times_s[i] as u128) % q as u128) as u64;
    }

    result
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
