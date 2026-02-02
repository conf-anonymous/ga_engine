//! Debug: Trace Metal EVK generation step by step

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::cpu_optimized::{
        keys::KeyContext,
        ntt::NttContext,
        rns::RnsRepresentation,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Debug: Metal EVK Generation Step by Step");
    println!("=========================================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (_, sk, _cpu_evk) = key_ctx.keygen();

    let n = params.n;
    let num_primes = params.moduli.len();
    let moduli = &params.moduli;
    let base_w = 20u32;

    // Step 1: Compute s²
    println!("Step 1: Computing s²...");
    let mut s_squared = Vec::with_capacity(n);
    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);
        let s_poly: Vec<u64> = sk.coeffs.iter().map(|rns| rns.values[prime_idx]).collect();
        let s_sq = ntt_ctx.multiply_polynomials(&s_poly, &s_poly);

        if prime_idx == 0 {
            for coeff_idx in 0..n {
                s_squared.push(RnsRepresentation::new(vec![s_sq[coeff_idx]; num_primes], moduli.to_vec()));
            }
        } else {
            for coeff_idx in 0..n {
                s_squared[coeff_idx].values[prime_idx] = s_sq[coeff_idx];
            }
        }
    }
    println!("  s²[0] = {:?}", s_squared[0].values);

    // Step 2: Compute B^t mod q for t=0
    println!("\nStep 2: Computing B^0 mod q (should all be 1)...");
    let base = 1u64 << base_w;
    let mut bpow_0_mod_q = vec![0u64; num_primes];
    for (j, &q) in moduli.iter().enumerate() {
        bpow_0_mod_q[j] = 1;  // B^0 = 1
        println!("  B^0 mod q[{}] = {} (q={})", j, bpow_0_mod_q[j], q);
    }

    // Step 3: Sample a_0 (random uniform polynomial)
    println!("\nStep 3: Sampling a_0...");
    let mut rng = rand::thread_rng();
    let mut a_0 = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for (prime_idx, &q) in moduli.iter().enumerate() {
            a_0[coeff_idx * num_primes + prime_idx] = rng.gen::<u64>() % q;
        }
    }
    println!("  a_0[0] = {:?}",
        (0..num_primes).map(|j| a_0[0 * num_primes + j]).collect::<Vec<_>>());

    // Step 4: Compute a_0 * s using NTT
    println!("\nStep 4: Computing a_0 * s...");
    let mut a_times_s = vec![0u64; n * num_primes];
    for (prime_idx, &q) in moduli.iter().enumerate() {
        let ntt_ctx = NttContext::new(n, q);

        let mut a_poly = vec![0u64; n];
        let mut s_poly = vec![0u64; n];
        for coeff_idx in 0..n {
            a_poly[coeff_idx] = a_0[coeff_idx * num_primes + prime_idx];
            s_poly[coeff_idx] = sk.coeffs[coeff_idx].values[prime_idx];
        }

        let product = ntt_ctx.multiply_polynomials(&a_poly, &s_poly);

        for coeff_idx in 0..n {
            a_times_s[coeff_idx * num_primes + prime_idx] = product[coeff_idx];
        }
    }
    println!("  (a_0 * s)[0] = {:?}",
        (0..num_primes).map(|j| a_times_s[0 * num_primes + j]).collect::<Vec<_>>());

    // Step 5: Sample error e_0
    println!("\nStep 5: Sampling error e_0...");
    let normal = Normal::new(0.0, params.error_std).unwrap();
    let mut e_0 = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        let e_float: f64 = normal.sample(&mut rng);
        for (prime_idx, &q) in moduli.iter().enumerate() {
            let e = if e_float >= 0.0 {
                (e_float.round() as u64) % q
            } else {
                let abs_e = ((-e_float).round() as u64) % q;
                if abs_e == 0 { 0 } else { q - abs_e }
            };
            e_0[coeff_idx * num_primes + prime_idx] = e;
        }
    }
    println!("  e_0[0] = {:?}",
        (0..num_primes).map(|j| e_0[0 * num_primes + j]).collect::<Vec<_>>());

    // Step 6: Compute -B^0 * s² = -s²
    println!("\nStep 6: Computing -B^0 * s² = -s²...");
    let mut neg_bt_s2 = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for (prime_idx, &q) in moduli.iter().enumerate() {
            let s2_val = s_squared[coeff_idx].values[prime_idx];
            let bt_mod_q = bpow_0_mod_q[prime_idx]; // = 1
            let power_s2_pos = ((bt_mod_q as u128 * s2_val as u128) % q as u128) as u64;
            neg_bt_s2[coeff_idx * num_primes + prime_idx] = if power_s2_pos == 0 { 0 } else { q - power_s2_pos };
        }
    }
    println!("  -s²[0] = {:?}",
        (0..num_primes).map(|j| neg_bt_s2[0 * num_primes + j]).collect::<Vec<_>>());

    // Step 7: Compute b_0 = -B^0 * s² + a_0 * s + e_0
    println!("\nStep 7: Computing b_0 = -s² + a*s + e...");
    let mut b_0 = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for (prime_idx, &q) in moduli.iter().enumerate() {
            let idx = coeff_idx * num_primes + prime_idx;
            let sum = (neg_bt_s2[idx] as u128 + a_times_s[idx] as u128 + e_0[idx] as u128) % q as u128;
            b_0[idx] = sum as u64;
        }
    }
    println!("  b_0[0] = {:?}",
        (0..num_primes).map(|j| b_0[0 * num_primes + j]).collect::<Vec<_>>());

    // Step 8: Verify the identity: b_0 - a_0 * s = -s² + e
    println!("\nStep 8: Verifying identity b_0 - a_0*s = -s² + e...");
    let mut diff = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        diff[i] = if b_0[i] >= a_times_s[i] {
            b_0[i] - a_times_s[i]
        } else {
            q - (a_times_s[i] - b_0[i])
        };
    }
    println!("  (b_0 - a*s)[0] = {:?}",
        (0..num_primes).map(|j| diff[0 * num_primes + j]).collect::<Vec<_>>());
    println!("  Expected (-s² + e)[0] = {:?}",
        (0..num_primes).map(|j| {
            let q = moduli[j];
            ((neg_bt_s2[0 * num_primes + j] as u128 + e_0[0 * num_primes + j] as u128) % q as u128) as u64
        }).collect::<Vec<_>>());

    // Now the key insight: let's verify this works for multiplication
    println!("\n=== Now let's trace through multiplication ===");

    // Create a simple test: encrypt a known value, square it
    let test_val = 2.0f64;
    let scale = params.scale;

    // Encode as polynomial (just constant term)
    let scaled_val = (test_val * scale).round() as u64;
    println!("\nEncoding {} * {} = {}", test_val, scale, scaled_val);

    // Create "ciphertext" with c0 = m, c1 = 0 for simplicity
    let mut c0 = vec![0u64; n * num_primes];
    let mut c1 = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        c0[0 * num_primes + prime_idx] = scaled_val % moduli[prime_idx];
    }
    println!("c0[0] = {:?}",
        (0..num_primes).map(|j| c0[0 * num_primes + j]).collect::<Vec<_>>());

    // For a REAL ciphertext, c1 would be random, let's use a simple one
    // Actually, let's use the secret key approach:
    // encrypt: c0 = pk0 * u + e + m, c1 = pk1 * u
    // decrypt: c0 + c1 * s = m + error

    // For now, just verify that the EVK structure is correct by checking:
    // digit × evk0 - digit × evk1 × s = digit × (-s² + e)

    // Let's create a simple digit (constant = 1)
    let mut digit = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        digit[0 * num_primes + prime_idx] = 1;
    }

    // Compute digit × evk0 = 1 × b_0 = b_0
    println!("\nComputing digit × evk0 (digit = 1)...");
    let digit_times_evk0 = mul_polys_ntt(&digit, &b_0, moduli, n)?;
    println!("  (digit × evk0)[0] = {:?}",
        (0..num_primes).map(|j| digit_times_evk0[0 * num_primes + j]).collect::<Vec<_>>());

    // Compute digit × evk1 = 1 × a_0 = a_0
    let digit_times_evk1 = mul_polys_ntt(&digit, &a_0, moduli, n)?;
    println!("  (digit × evk1)[0] = {:?}",
        (0..num_primes).map(|j| digit_times_evk1[0 * num_primes + j]).collect::<Vec<_>>());

    // Compute digit × evk1 × s
    let digit_evk1_times_s = mul_polys_ntt(&digit_times_evk1, &secret_to_flat(&sk, num_primes), moduli, n)?;
    println!("  (digit × evk1 × s)[0] = {:?}",
        (0..num_primes).map(|j| digit_evk1_times_s[0 * num_primes + j]).collect::<Vec<_>>());

    // Compute difference: digit × evk0 - digit × evk1 × s
    let mut final_diff = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        final_diff[i] = if digit_times_evk0[i] >= digit_evk1_times_s[i] {
            digit_times_evk0[i] - digit_evk1_times_s[i]
        } else {
            q - (digit_evk1_times_s[i] - digit_times_evk0[i])
        };
    }
    println!("  (digit × evk0 - digit × evk1 × s)[0] = {:?}",
        (0..num_primes).map(|j| final_diff[0 * num_primes + j]).collect::<Vec<_>>());

    // This should equal digit × (-s² + e)
    // Since digit = 1 (constant), digit × x = x for polynomial x
    // So we expect final_diff = -s² + e
    let mut expected = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        expected[i] = ((neg_bt_s2[i] as u128 + e_0[i] as u128) % q as u128) as u64;
    }
    println!("  Expected (-s² + e)[0] = {:?}",
        (0..num_primes).map(|j| expected[0 * num_primes + j]).collect::<Vec<_>>());

    // Check max error
    let mut max_error: i64 = 0;
    for coeff_idx in 0..n {
        let actual = final_diff[coeff_idx * num_primes + 0];
        let exp = expected[coeff_idx * num_primes + 0];
        let q = moduli[0];
        let diff = if actual >= exp { actual - exp } else { q - (exp - actual) };
        let diff_centered = if diff > q/2 { diff as i64 - q as i64 } else { diff as i64 };
        max_error = max_error.max(diff_centered.abs());
    }
    println!("\n  Max error: {} (should be 0 or very small)", max_error);

    if max_error == 0 {
        println!("  ✅ Perfect match!");
    } else if max_error < 100 {
        println!("  ✅ Close match (possible rounding)");
    } else {
        println!("  ❌ Mismatch!");
    }

    Ok(())
}

#[cfg(feature = "v2-gpu-metal")]
fn mul_polys_ntt(a: &[u64], b: &[u64], moduli: &[u64], n: usize) -> Result<Vec<u64>, String> {
    let num_primes = moduli.len();
    let mut result = vec![0u64; n * num_primes];

    for (prime_idx, &q) in moduli.iter().enumerate() {
        let ntt_ctx = NttContext::new(n, q);

        let mut a_poly = vec![0u64; n];
        let mut b_poly = vec![0u64; n];
        for coeff_idx in 0..n {
            a_poly[coeff_idx] = a[coeff_idx * num_primes + prime_idx];
            b_poly[coeff_idx] = b[coeff_idx * num_primes + prime_idx];
        }

        let product = ntt_ctx.multiply_polynomials(&a_poly, &b_poly);

        for coeff_idx in 0..n {
            result[coeff_idx * num_primes + prime_idx] = product[coeff_idx];
        }
    }

    Ok(result)
}

#[cfg(feature = "v2-gpu-metal")]
fn secret_to_flat(
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    num_primes: usize
) -> Vec<u64> {
    let n = sk.coeffs.len();
    let mut flat = vec![0u64; n * num_primes];
    for i in 0..n {
        for j in 0..num_primes {
            flat[i * num_primes + j] = sk.coeffs[i].values[j];
        }
    }
    flat
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
