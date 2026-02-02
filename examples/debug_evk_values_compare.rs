//! Debug: Compare EXACT CPU and Metal EVK values coefficient by coefficient
//!
//! The divide_conquer_same_ct test proved the bug is in Metal EVK.
//! This test compares the actual values to find where they differ.

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::{KeyContext, EvaluationKey},
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  EXACT EVK VALUE COMPARISON: CPU vs Metal                     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (_pk, sk, cpu_evk) = key_ctx.keygen();

    let device = Arc::new(MetalDevice::new()?);
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let ntt_contexts = metal_ctx.ntt_contexts();

    // Generate Metal EVK with SAME secret key
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

    println!("Parameters:");
    println!("  n = {}", n);
    println!("  level = {}", level);
    println!("  num_primes = {}", num_primes);
    println!("  moduli = {:?}\n", moduli);

    // Get EVK keys
    let (metal_evk0, metal_evk1) = metal_evk.get_coeff_keys(level)?;
    let (cpu_evk0, cpu_evk1) = evk_to_flat(&cpu_evk, num_primes);

    println!("Number of digits: CPU={}, Metal={}", cpu_evk0.len(), metal_evk0.len());

    // Compare digit by digit
    for digit_idx in 0..cpu_evk0.len().min(metal_evk0.len()).min(3) {
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("DIGIT {}", digit_idx);
        println!("═══════════════════════════════════════════════════════════════");

        // Check dimensions
        let cpu_len = cpu_evk0[digit_idx].len();
        let metal_len = metal_evk0[digit_idx].len();
        println!("  CPU evk0[{}].len() = {}, Metal evk0[{}].len() = {}",
            digit_idx, cpu_len, digit_idx, metal_len);

        if cpu_len != metal_len {
            println!("  ❌ LENGTH MISMATCH!");
            continue;
        }

        // Compare first few coefficients
        println!("\n  Comparing first 5 coefficients (coeff 0-4):");
        let mut any_diff = false;
        for coeff_idx in 0..5 {
            let cpu_vals: Vec<u64> = (0..num_primes)
                .map(|j| cpu_evk0[digit_idx][coeff_idx * num_primes + j])
                .collect();
            let metal_vals: Vec<u64> = (0..num_primes)
                .map(|j| metal_evk0[digit_idx][coeff_idx * num_primes + j])
                .collect();

            let match_status = if cpu_vals == metal_vals { "✅" } else { "❌" };
            if cpu_vals != metal_vals {
                any_diff = true;
            }
            println!("    coeff[{}] evk0: {} CPU={:?}, Metal={:?}",
                coeff_idx, match_status, cpu_vals, metal_vals);
        }

        // evk1 comparison
        println!("\n  evk1 comparison:");
        for coeff_idx in 0..5 {
            let cpu_vals: Vec<u64> = (0..num_primes)
                .map(|j| cpu_evk1[digit_idx][coeff_idx * num_primes + j])
                .collect();
            let metal_vals: Vec<u64> = (0..num_primes)
                .map(|j| metal_evk1[digit_idx][coeff_idx * num_primes + j])
                .collect();

            let match_status = if cpu_vals == metal_vals { "✅" } else { "❌" };
            if cpu_vals != metal_vals {
                any_diff = true;
            }
            println!("    coeff[{}] evk1: {} CPU={:?}, Metal={:?}",
                coeff_idx, match_status, cpu_vals, metal_vals);
        }

        if any_diff {
            println!("\n  ⚠️ EVK values differ - but this is EXPECTED since different random a_i");
            println!("     What matters is whether the IDENTITY holds:");
            println!("     evk0[t] - evk1[t] * s = -B^t * s² + error");
        }
    }

    // Now the KEY test: verify identity holds for BOTH and compare the noise
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("VERIFYING EVK IDENTITY: evk0 - evk1*s = -B^t * s² + e");
    println!("═══════════════════════════════════════════════════════════════\n");

    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;

    let base_w = 20u32;
    let base = 1u64 << base_w;

    for digit_idx in 0..cpu_evk0.len().min(metal_evk0.len()).min(3) {
        println!("Digit {}:", digit_idx);

        // Compute B^t mod q
        let mut bt_mod_q = vec![0u64; num_primes];
        for (j, &q) in moduli.iter().enumerate() {
            let mut p = 1u128;
            for _ in 0..digit_idx {
                p = (p * (base as u128)) % (q as u128);
            }
            bt_mod_q[j] = p as u64;
        }

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

        // CPU: evk0 - evk1*s should = -B^t * s² + noise
        let cpu_noise = check_identity(
            &cpu_evk0[digit_idx], &cpu_evk1[digit_idx],
            &sk, &s_sq, &bt_mod_q, moduli, n
        );

        // Metal: same check
        let metal_noise = check_identity(
            &metal_evk0[digit_idx], &metal_evk1[digit_idx],
            &sk, &s_sq, &bt_mod_q, moduli, n
        );

        println!("  CPU   max_noise: {} {}", cpu_noise, if cpu_noise < 1_000_000 { "✅" } else { "❌" });
        println!("  Metal max_noise: {} {}", metal_noise, if metal_noise < 1_000_000 { "✅" } else { "❌" });
    }

    // Finally, let's manually apply relinearization and track intermediate values
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("MANUAL RELINEARIZATION WITH d2 = [1, 0, 0, ...]");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create a simple d2 that is just [1, 0, 0, ...] (constant 1 polynomial)
    // After gadget decomposition, digit[0] = [1, 0, ...], other digits = [0, 0, ...]
    let mut d2_simple = vec![0u64; n * num_primes];
    for j in 0..num_primes {
        d2_simple[0 * num_primes + j] = 1;  // coeff[0] = 1 for all primes
    }

    println!("d2 = [1, 0, 0, ...] (constant polynomial = 1)");

    // Gadget decompose (for d2 = 1, digit[0] = 1, rest = 0)
    println!("\nGadget decomposition of d2 = 1:");
    for digit_idx in 0..3 {
        let expected = if digit_idx == 0 { 1 } else { 0 };
        println!("  digit[{}] = {} (all coeffs should be {})", digit_idx, expected, expected);
    }

    // For d2 = 1 (constant), relinearization should add:
    // c0 += digit[0] * evk0[0] = 1 * evk0[0] = evk0[0]
    // c1 += digit[0] * evk1[0] = 1 * evk1[0] = evk1[0]
    //
    // When we decrypt: c0 + c1*s = evk0[0] + evk1[0]*s = -B^0*s² + e = -s² + e
    //
    // So if d2 = 1, after relin we add -s² to the result (plus noise)

    println!("\nIf d2=1, relin adds -s² to result. Let's verify:");
    println!("  -s²[0] should be large negative (centered): ");
    let s2_0_prime0 = s_sq_val(&sk, moduli, n, 0, 0);
    let neg_s2_centered = if s2_0_prime0 == 0 { 0i64 } else { -(s2_0_prime0 as i64) };
    println!("  s²[coeff=0, prime=0] = {}, -s² centered = {}", s2_0_prime0, neg_s2_centered);

    // Now apply CPU EVK's evk0[0] + evk1[0]*s for coeff 0
    let cpu_relin_result = compute_evk_decrypt(
        &cpu_evk0[0], &cpu_evk1[0], &sk, moduli, n
    );
    println!("\n  CPU evk0[0] + evk1[0]*s (coeff 0, prime 0) = {}", cpu_relin_result[0]);
    let cpu_centered = if cpu_relin_result[0] > moduli[0]/2 {
        cpu_relin_result[0] as i64 - moduli[0] as i64
    } else {
        cpu_relin_result[0] as i64
    };
    println!("  Centered: {} (should be close to {})", cpu_centered, neg_s2_centered);

    let metal_relin_result = compute_evk_decrypt(
        &metal_evk0[0], &metal_evk1[0], &sk, moduli, n
    );
    println!("\n  Metal evk0[0] + evk1[0]*s (coeff 0, prime 0) = {}", metal_relin_result[0]);
    let metal_centered = if metal_relin_result[0] > moduli[0]/2 {
        metal_relin_result[0] as i64 - moduli[0] as i64
    } else {
        metal_relin_result[0] as i64
    };
    println!("  Centered: {} (should be close to {})", metal_centered, neg_s2_centered);

    let cpu_error = (cpu_centered - neg_s2_centered).abs();
    let metal_error = (metal_centered - neg_s2_centered).abs();
    println!("\n  CPU   error from expected: {} {}", cpu_error, if cpu_error < 100000 { "✅" } else { "❌" });
    println!("  Metal error from expected: {} {}", metal_error, if metal_error < 100000 { "✅" } else { "❌" });

    Ok(())
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
fn check_identity(
    evk0: &[u64], evk1: &[u64],
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    s_sq: &[u64], bt_mod_q: &[u64], moduli: &[u64], n: usize
) -> i64 {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;
    let num_primes = moduli.len();

    // evk1 * s
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

    // evk0 - evk1*s
    let mut diff = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        diff[i] = if evk0[i] >= evk1_times_s[i] {
            evk0[i] - evk1_times_s[i]
        } else {
            q - (evk1_times_s[i] - evk0[i])
        };
    }

    // Expected: -B^t * s²
    let mut expected = vec![0u64; n * num_primes];
    for i in 0..n {
        for (j, &q) in moduli.iter().enumerate() {
            let bt_s2 = ((bt_mod_q[j] as u128 * s_sq[i * num_primes + j] as u128) % q as u128) as u64;
            expected[i * num_primes + j] = if bt_s2 == 0 { 0 } else { q - bt_s2 };
        }
    }

    // Max noise
    let mut max_noise: i64 = 0;
    for i in 0..n {
        let d = diff[i * num_primes + 0];
        let exp = expected[i * num_primes + 0];
        let q = moduli[0];
        let noise = if d >= exp { d - exp } else { q - (exp - d) };
        let noise_centered = if noise > q/2 { noise as i64 - q as i64 } else { noise as i64 };
        max_noise = max_noise.max(noise_centered.abs());
    }

    max_noise
}

#[cfg(feature = "v2-gpu-metal")]
fn s_sq_val(
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    moduli: &[u64], n: usize, coeff_idx: usize, prime_idx: usize
) -> u64 {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;
    let q = moduli[prime_idx];
    let ntt_ctx = NttContext::new(n, q);
    let s_poly: Vec<u64> = sk.coeffs.iter().map(|rns| rns.values[prime_idx]).collect();
    let sq = ntt_ctx.multiply_polynomials(&s_poly, &s_poly);
    sq[coeff_idx]
}

#[cfg(feature = "v2-gpu-metal")]
fn compute_evk_decrypt(
    evk0: &[u64], evk1: &[u64],
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    moduli: &[u64], n: usize
) -> Vec<u64> {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;
    let num_primes = moduli.len();

    // evk1 * s
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

    // evk0 + evk1*s (not minus! we want the decryption)
    let mut result = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        result[i] = ((evk0[i] as u128 + evk1_times_s[i] as u128) % q as u128) as u64;
    }

    result
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
