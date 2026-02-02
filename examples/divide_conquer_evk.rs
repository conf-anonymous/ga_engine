//! Divide and Conquer: Isolate exactly where Metal EVK multiplication fails
//!
//! We test EVERY atomic step to find the exact point of failure.

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::{MetalCkksContext, MetalCiphertext},
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::{KeyContext, EvaluationKey, SecretKey},
        ckks::CkksContext,
        ntt::NttContext,
        rns::RnsRepresentation,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;
#[cfg(feature = "v2-gpu-metal")]
use rand::Rng;
#[cfg(feature = "v2-gpu-metal")]
use rand_distr::{Distribution, Normal};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     DIVIDE AND CONQUER: Metal EVK Multiplication Debug       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, cpu_evk) = key_ctx.keygen();

    let device = Arc::new(MetalDevice::new()?);
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let cpu_ctx = CkksContext::new(params.clone());

    let n = params.n;
    let num_primes = params.moduli.len();  // 3
    let moduli = &params.moduli;
    let base_w = 20u32;

    println!("Parameters: n={}, num_primes={}, base_w={}", n, num_primes, base_w);
    println!("Moduli: {:?}\n", moduli);

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 1: EVK GENERATION - Test each step
    // ═══════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("PHASE 1: EVK GENERATION STEPS");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Step 1a: Compute s²
    println!("Step 1a: Compute s² using NTT");
    let s_squared = compute_s_squared(&sk, moduli, n)?;
    println!("  s²[0] = {:?}", &s_squared[0].values[..num_primes]);
    println!("  ✅ Step 1a complete\n");

    // Step 1b: Sample random polynomial a_0
    println!("Step 1b: Sample random polynomial a_0");
    let a_0 = sample_uniform_poly(n, moduli);
    println!("  a_0[0] = {:?}", extract_coeff(&a_0, 0, num_primes));
    println!("  ✅ Step 1b complete\n");

    // Step 1c: Compute a_0 × s using NTT
    println!("Step 1c: Compute a_0 × s using NTT");
    let a_times_s = multiply_poly_by_secret(&a_0, &sk, moduli, n)?;
    println!("  (a_0 × s)[0] = {:?}", extract_coeff(&a_times_s, 0, num_primes));
    println!("  ✅ Step 1c complete\n");

    // Step 1d: Sample error e_0
    println!("Step 1d: Sample error e_0");
    let e_0 = sample_error_poly(n, moduli, params.error_std);
    println!("  e_0[0] = {:?}", extract_coeff(&e_0, 0, num_primes));
    println!("  ✅ Step 1d complete\n");

    // Step 1e: Compute B^0 mod q (for digit 0, B^0 = 1)
    println!("Step 1e: Compute B^0 mod q for each prime");
    let b_pow_0 = compute_b_power_mod_q(0, base_w, moduli);
    println!("  B^0 mod q = {:?} (should all be 1)", b_pow_0);
    assert!(b_pow_0.iter().all(|&x| x == 1), "B^0 should be 1!");
    println!("  ✅ Step 1e complete\n");

    // Step 1f: Compute B^0 × s² mod q
    println!("Step 1f: Compute B^0 × s² mod q");
    let b_pow_times_s_sq = multiply_scalar_by_poly(&b_pow_0, &s_squared, moduli, n);
    println!("  (B^0 × s²)[0] = {:?}", extract_coeff(&b_pow_times_s_sq, 0, num_primes));
    // Since B^0 = 1, this should equal s²
    let s_sq_0 = &s_squared[0].values[..num_primes];
    let b_s_sq_0 = extract_coeff(&b_pow_times_s_sq, 0, num_primes);
    assert_eq!(s_sq_0, &b_s_sq_0[..], "B^0 × s² should equal s²!");
    println!("  ✅ Step 1f complete (matches s²)\n");

    // Step 1g: Negate to get -B^0 × s²
    println!("Step 1g: Negate to get -B^0 × s²");
    let neg_b_pow_times_s_sq = negate_poly(&b_pow_times_s_sq, moduli, n);
    println!("  (-B^0 × s²)[0] = {:?}", extract_coeff(&neg_b_pow_times_s_sq, 0, num_primes));
    println!("  ✅ Step 1g complete\n");

    // Step 1h: Sum evk0[0] = -B^0×s² + a_0×s + e_0
    println!("Step 1h: Compute evk0[0] = -B^0×s² + a_0×s + e_0");
    let evk0_manual = add_three_polys(&neg_b_pow_times_s_sq, &a_times_s, &e_0, moduli, n);
    println!("  evk0[0][coeff=0] = {:?}", extract_coeff(&evk0_manual, 0, num_primes));
    println!("  ✅ Step 1h complete\n");

    // Step 1i: evk1[0] = a_0
    println!("Step 1i: Store evk1[0] = a_0");
    let evk1_manual = a_0.clone();
    println!("  evk1[0][coeff=0] = {:?}", extract_coeff(&evk1_manual, 0, num_primes));
    println!("  ✅ Step 1i complete\n");

    // ═══════════════════════════════════════════════════════════════════════
    // VERIFY: Check that manually computed EVK satisfies identity
    // ═══════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("VERIFICATION: Manual EVK Identity Check");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Checking: evk0 - evk1×s = -B^0×s² + e");
    let evk1_times_s = multiply_flat_by_secret(&evk1_manual, &sk, moduli, n)?;
    let evk0_minus_evk1s = subtract_polys(&evk0_manual, &evk1_times_s, moduli, n);
    let expected = add_polys(&neg_b_pow_times_s_sq, &e_0, moduli, n);

    println!("  (evk0 - evk1×s)[0] = {:?}", extract_coeff(&evk0_minus_evk1s, 0, num_primes));
    println!("  (-s² + e)[0] = {:?}", extract_coeff(&expected, 0, num_primes));

    let identity_ok = compare_polys(&evk0_minus_evk1s, &expected, n, num_primes);
    if identity_ok {
        println!("  ✅ Manual EVK identity VERIFIED\n");
    } else {
        println!("  ❌ Manual EVK identity FAILED\n");
        return Err("Manual EVK identity failed".to_string());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // COMPARE: Manual EVK vs Metal-generated EVK
    // ═══════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("COMPARE: Manual EVK vs Metal-generated EVK");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Generate Metal EVK
    let ntt_contexts = metal_ctx.ntt_contexts();
    let metal_evk = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        base_w,
    )?;

    let level = 2;
    let level_num_primes = level + 1;
    let (metal_evk0, metal_evk1) = metal_evk.get_coeff_keys(level)?;

    println!("Metal EVK generated for level {}", level);
    println!("  metal_evk0[0][coeff=0] = {:?}", extract_coeff(&metal_evk0[0], 0, level_num_primes));
    println!("  metal_evk1[0][coeff=0] = {:?}", extract_coeff(&metal_evk1[0], 0, level_num_primes));

    // Verify Metal EVK identity
    println!("\nVerifying Metal EVK identity for digit 0...");
    let metal_identity_noise = check_evk_identity_noise(
        &metal_evk0[0], &metal_evk1[0], &sk, &params.moduli[..level_num_primes], n, 0, base_w
    )?;
    println!("  Metal EVK identity max_noise = {}", metal_identity_noise);
    if metal_identity_noise < 100 {
        println!("  ✅ Metal EVK identity VERIFIED\n");
    } else {
        println!("  ❌ Metal EVK identity has HIGH NOISE\n");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 2: RELINEARIZATION - Test each step
    // ═══════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("PHASE 2: RELINEARIZATION STEPS");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create a ciphertext
    let cpu_pt = cpu_ctx.encode(&[2.0]);
    let cpu_ct = cpu_ctx.encrypt(&cpu_pt, &pk);

    // Step 3a: Tensor product
    println!("Step 3a: Compute tensor product for ct × ct");
    let ct_flat = ct_to_flat(&cpu_ct, level_num_primes);
    let c0c0 = cpu_multiply_flat(&ct_flat.0, &ct_flat.0, &params.moduli[..level_num_primes], n)?;
    let c0c1 = cpu_multiply_flat(&ct_flat.0, &ct_flat.1, &params.moduli[..level_num_primes], n)?;
    let c1c0 = cpu_multiply_flat(&ct_flat.1, &ct_flat.0, &params.moduli[..level_num_primes], n)?;
    let c1c1 = cpu_multiply_flat(&ct_flat.1, &ct_flat.1, &params.moduli[..level_num_primes], n)?;
    println!("  c0×c0[0] = {:?}", extract_coeff(&c0c0, 0, level_num_primes));
    println!("  c1×c1[0] = {:?}", extract_coeff(&c1c1, 0, level_num_primes));
    println!("  ✅ Step 3a complete\n");

    // Step 3b: Form d0, d1, d2
    println!("Step 3b: Form d0, d1, d2");
    let d0 = c0c0;
    let d1 = add_polys_flat(&c0c1, &c1c0, &params.moduli[..level_num_primes], n);
    let d2 = c1c1;
    println!("  d0[0] = {:?}", extract_coeff(&d0, 0, level_num_primes));
    println!("  d1[0] = {:?}", extract_coeff(&d1, 0, level_num_primes));
    println!("  d2[0] = {:?}", extract_coeff(&d2, 0, level_num_primes));
    println!("  ✅ Step 3b complete\n");

    // Step 3c: Gadget decompose d2
    println!("Step 3c: Gadget decompose d2");
    let d2_digits = MetalCiphertext::gadget_decompose_flat(&d2, base_w, &params.moduli[..level_num_primes], n)?;
    println!("  Number of digits: {}", d2_digits.len());
    println!("  digit[0][coeff=0] = {:?}", extract_coeff(&d2_digits[0], 0, level_num_primes));
    println!("  digit[1][coeff=0] = {:?}", extract_coeff(&d2_digits[1], 0, level_num_primes));
    println!("  ✅ Step 3c complete\n");

    // Now test relinearization with BOTH CPU EVK and Metal EVK
    println!("═══════════════════════════════════════════════════════════════");
    println!("PHASE 3: RELIN ACCUMULATION - Comparing CPU vs Metal EVK");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Convert CPU EVK to flat format
    let (cpu_evk0_flat, cpu_evk1_flat) = evk_to_flat(&cpu_evk, level_num_primes);

    // Initialize accumulators
    let mut cpu_c0 = d0.clone();
    let mut cpu_c1 = d1.clone();
    let mut metal_c0 = d0.clone();
    let mut metal_c1 = d1.clone();

    let num_test_digits = 3.min(d2_digits.len()); // Test first 3 digits

    for t in 0..num_test_digits {
        println!("--- Digit {} ---", t);

        // Step 3d: digit[t] × evk0[t]
        let cpu_term0 = cpu_multiply_flat(&d2_digits[t], &cpu_evk0_flat[t], &params.moduli[..level_num_primes], n)?;
        let metal_term0 = cpu_multiply_flat(&d2_digits[t], &metal_evk0[t], &params.moduli[..level_num_primes], n)?;

        println!("  CPU  term0[0] = {:?}", extract_coeff(&cpu_term0, 0, level_num_primes));
        println!("  Metal term0[0] = {:?}", extract_coeff(&metal_term0, 0, level_num_primes));

        // Step 3e: digit[t] × evk1[t]
        let cpu_term1 = cpu_multiply_flat(&d2_digits[t], &cpu_evk1_flat[t], &params.moduli[..level_num_primes], n)?;
        let metal_term1 = cpu_multiply_flat(&d2_digits[t], &metal_evk1[t], &params.moduli[..level_num_primes], n)?;

        println!("  CPU  term1[0] = {:?}", extract_coeff(&cpu_term1, 0, level_num_primes));
        println!("  Metal term1[0] = {:?}", extract_coeff(&metal_term1, 0, level_num_primes));

        // Step 3f: Accumulate
        cpu_c0 = subtract_polys_flat(&cpu_c0, &cpu_term0, &params.moduli[..level_num_primes], n);
        cpu_c1 = add_polys_flat(&cpu_c1, &cpu_term1, &params.moduli[..level_num_primes], n);
        metal_c0 = subtract_polys_flat(&metal_c0, &metal_term0, &params.moduli[..level_num_primes], n);
        metal_c1 = add_polys_flat(&metal_c1, &metal_term1, &params.moduli[..level_num_primes], n);

        println!("  CPU  c0[0] after = {:?}", extract_coeff(&cpu_c0, 0, level_num_primes));
        println!("  Metal c0[0] after = {:?}", extract_coeff(&metal_c0, 0, level_num_primes));
        println!();
    }

    // Complete remaining digits
    for t in num_test_digits..d2_digits.len() {
        if t >= cpu_evk0_flat.len() || t >= metal_evk0.len() { break; }

        let cpu_term0 = cpu_multiply_flat(&d2_digits[t], &cpu_evk0_flat[t], &params.moduli[..level_num_primes], n)?;
        let metal_term0 = cpu_multiply_flat(&d2_digits[t], &metal_evk0[t], &params.moduli[..level_num_primes], n)?;
        let cpu_term1 = cpu_multiply_flat(&d2_digits[t], &cpu_evk1_flat[t], &params.moduli[..level_num_primes], n)?;
        let metal_term1 = cpu_multiply_flat(&d2_digits[t], &metal_evk1[t], &params.moduli[..level_num_primes], n)?;

        cpu_c0 = subtract_polys_flat(&cpu_c0, &cpu_term0, &params.moduli[..level_num_primes], n);
        cpu_c1 = add_polys_flat(&cpu_c1, &cpu_term1, &params.moduli[..level_num_primes], n);
        metal_c0 = subtract_polys_flat(&metal_c0, &metal_term0, &params.moduli[..level_num_primes], n);
        metal_c1 = add_polys_flat(&metal_c1, &metal_term1, &params.moduli[..level_num_primes], n);
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("PHASE 4: DECRYPT AND COMPARE (before rescale)");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Decrypt without rescaling to see raw values
    let scale_squared = cpu_ct.scale * cpu_ct.scale;
    let cpu_decrypted = decrypt_flat(&cpu_c0, &cpu_c1, &sk, &params.moduli[..level_num_primes], n, scale_squared)?;
    let metal_decrypted = decrypt_flat(&metal_c0, &metal_c1, &sk, &params.moduli[..level_num_primes], n, scale_squared)?;

    println!("Decrypted values (before rescale):");
    println!("  Expected: 4.0 (2² = 4)");
    println!("  CPU EVK result:   {} (error: {:.2e})", cpu_decrypted, (cpu_decrypted - 4.0).abs());
    println!("  Metal EVK result: {} (error: {:.2e})", metal_decrypted, (metal_decrypted - 4.0).abs());

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("FINAL VERDICT");
    println!("═══════════════════════════════════════════════════════════════\n");

    let cpu_ok = (cpu_decrypted - 4.0).abs() < 1e6;  // Before rescale, values are large
    let metal_ok = (metal_decrypted - 4.0).abs() < 1e6;

    if cpu_ok && metal_ok {
        println!("✅ BOTH CPU and Metal EVK produce correct results!");
        Ok(())
    } else if cpu_ok && !metal_ok {
        println!("❌ CPU EVK works, Metal EVK FAILS");
        println!("\nThe bug is in Metal EVK GENERATION or EXTRACTION");
        Err("Metal EVK multiplication fails".to_string())
    } else {
        println!("❌ BOTH fail - something else is wrong");
        Err("Both EVK types fail".to_string())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "v2-gpu-metal")]
fn compute_s_squared(sk: &SecretKey, moduli: &[u64], n: usize) -> Result<Vec<RnsRepresentation>, String> {
    let num_primes = moduli.len();
    let mut result = Vec::with_capacity(n);

    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);
        let s_poly: Vec<u64> = sk.coeffs.iter().map(|rns| rns.values[prime_idx]).collect();
        let s_sq = ntt_ctx.multiply_polynomials(&s_poly, &s_poly);

        if prime_idx == 0 {
            for i in 0..n {
                result.push(RnsRepresentation::new(vec![s_sq[i]; num_primes], moduli.to_vec()));
            }
        } else {
            for i in 0..n {
                result[i].values[prime_idx] = s_sq[i];
            }
        }
    }
    Ok(result)
}

#[cfg(feature = "v2-gpu-metal")]
fn sample_uniform_poly(n: usize, moduli: &[u64]) -> Vec<u64> {
    let num_primes = moduli.len();
    let mut rng = rand::thread_rng();
    let mut poly = vec![0u64; n * num_primes];
    for i in 0..n {
        for (j, &q) in moduli.iter().enumerate() {
            poly[i * num_primes + j] = rng.gen::<u64>() % q;
        }
    }
    poly
}

#[cfg(feature = "v2-gpu-metal")]
fn sample_error_poly(n: usize, moduli: &[u64], std_dev: f64) -> Vec<u64> {
    let num_primes = moduli.len();
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, std_dev).unwrap();
    let mut poly = vec![0u64; n * num_primes];
    for i in 0..n {
        let e: f64 = normal.sample(&mut rng);
        for (j, &q) in moduli.iter().enumerate() {
            poly[i * num_primes + j] = if e >= 0.0 {
                (e.round() as u64) % q
            } else {
                let abs = ((-e).round() as u64) % q;
                if abs == 0 { 0 } else { q - abs }
            };
        }
    }
    poly
}

#[cfg(feature = "v2-gpu-metal")]
fn multiply_poly_by_secret(poly: &[u64], sk: &SecretKey, moduli: &[u64], n: usize) -> Result<Vec<u64>, String> {
    let num_primes = moduli.len();
    let mut result = vec![0u64; n * num_primes];
    for (prime_idx, &q) in moduli.iter().enumerate() {
        let ntt_ctx = NttContext::new(n, q);
        let mut p = vec![0u64; n];
        let mut s = vec![0u64; n];
        for i in 0..n {
            p[i] = poly[i * num_primes + prime_idx];
            s[i] = sk.coeffs[i].values[prime_idx];
        }
        let prod = ntt_ctx.multiply_polynomials(&p, &s);
        for i in 0..n {
            result[i * num_primes + prime_idx] = prod[i];
        }
    }
    Ok(result)
}

#[cfg(feature = "v2-gpu-metal")]
fn multiply_flat_by_secret(poly: &[u64], sk: &SecretKey, moduli: &[u64], n: usize) -> Result<Vec<u64>, String> {
    multiply_poly_by_secret(poly, sk, moduli, n)
}

#[cfg(feature = "v2-gpu-metal")]
fn compute_b_power_mod_q(digit: usize, base_w: u32, moduli: &[u64]) -> Vec<u64> {
    let base = 1u64 << base_w;
    moduli.iter().map(|&q| {
        let mut p = 1u128;
        for _ in 0..digit {
            p = (p * (base as u128)) % (q as u128);
        }
        p as u64
    }).collect()
}

#[cfg(feature = "v2-gpu-metal")]
fn multiply_scalar_by_poly(scalar: &[u64], poly: &[RnsRepresentation], moduli: &[u64], n: usize) -> Vec<u64> {
    let num_primes = moduli.len();
    let mut result = vec![0u64; n * num_primes];
    for i in 0..n {
        for (j, &q) in moduli.iter().enumerate() {
            result[i * num_primes + j] = ((scalar[j] as u128 * poly[i].values[j] as u128) % q as u128) as u64;
        }
    }
    result
}

#[cfg(feature = "v2-gpu-metal")]
fn negate_poly(poly: &[u64], moduli: &[u64], n: usize) -> Vec<u64> {
    let num_primes = moduli.len();
    let mut result = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        result[i] = if poly[i] == 0 { 0 } else { q - poly[i] };
    }
    result
}

#[cfg(feature = "v2-gpu-metal")]
fn add_three_polys(a: &[u64], b: &[u64], c: &[u64], moduli: &[u64], n: usize) -> Vec<u64> {
    let num_primes = moduli.len();
    let mut result = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        result[i] = ((a[i] as u128 + b[i] as u128 + c[i] as u128) % q as u128) as u64;
    }
    result
}

#[cfg(feature = "v2-gpu-metal")]
fn add_polys(a: &[u64], b: &[u64], moduli: &[u64], n: usize) -> Vec<u64> {
    let num_primes = moduli.len();
    let mut result = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        result[i] = ((a[i] as u128 + b[i] as u128) % q as u128) as u64;
    }
    result
}

#[cfg(feature = "v2-gpu-metal")]
fn add_polys_flat(a: &[u64], b: &[u64], moduli: &[u64], n: usize) -> Vec<u64> {
    add_polys(a, b, moduli, n)
}

#[cfg(feature = "v2-gpu-metal")]
fn subtract_polys(a: &[u64], b: &[u64], moduli: &[u64], n: usize) -> Vec<u64> {
    let num_primes = moduli.len();
    let mut result = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        result[i] = if a[i] >= b[i] { a[i] - b[i] } else { q - (b[i] - a[i]) };
    }
    result
}

#[cfg(feature = "v2-gpu-metal")]
fn subtract_polys_flat(a: &[u64], b: &[u64], moduli: &[u64], n: usize) -> Vec<u64> {
    subtract_polys(a, b, moduli, n)
}

#[cfg(feature = "v2-gpu-metal")]
fn compare_polys(a: &[u64], b: &[u64], n: usize, num_primes: usize) -> bool {
    for i in 0..(n * num_primes) {
        if a[i] != b[i] {
            return false;
        }
    }
    true
}

#[cfg(feature = "v2-gpu-metal")]
fn extract_coeff(poly: &[u64], coeff_idx: usize, num_primes: usize) -> Vec<u64> {
    (0..num_primes).map(|j| poly[coeff_idx * num_primes + j]).collect()
}

#[cfg(feature = "v2-gpu-metal")]
fn check_evk_identity_noise(
    evk0: &[u64], evk1: &[u64], sk: &SecretKey, moduli: &[u64], n: usize, digit: usize, base_w: u32
) -> Result<i64, String> {
    let num_primes = moduli.len();
    let evk1_s = multiply_flat_by_secret(evk1, sk, moduli, n)?;
    let diff = subtract_polys(evk0, &evk1_s, moduli, n);

    let b_pow = compute_b_power_mod_q(digit, base_w, moduli);
    let s_sq = compute_s_squared(sk, moduli, n)?;
    let b_s_sq = multiply_scalar_by_poly(&b_pow, &s_sq, moduli, n);
    let neg_b_s_sq = negate_poly(&b_s_sq, moduli, n);

    let mut max_noise: i64 = 0;
    for i in 0..n {
        let d = diff[i * num_primes];
        let exp = neg_b_s_sq[i * num_primes];
        let q = moduli[0];
        let noise = if d >= exp { d - exp } else { q - (exp - d) };
        let nc = if noise > q/2 { noise as i64 - q as i64 } else { noise as i64 };
        max_noise = max_noise.max(nc.abs());
    }
    Ok(max_noise)
}

#[cfg(feature = "v2-gpu-metal")]
fn ct_to_flat(ct: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext, num_primes: usize) -> (Vec<u64>, Vec<u64>) {
    let n = ct.c0.len();
    let mut c0 = vec![0u64; n * num_primes];
    let mut c1 = vec![0u64; n * num_primes];
    for i in 0..n {
        for j in 0..num_primes {
            c0[i * num_primes + j] = ct.c0[i].values[j];
            c1[i * num_primes + j] = ct.c1[i].values[j];
        }
    }
    (c0, c1)
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
fn cpu_multiply_flat(a: &[u64], b: &[u64], moduli: &[u64], n: usize) -> Result<Vec<u64>, String> {
    let num_primes = moduli.len();
    let mut result = vec![0u64; n * num_primes];
    for (prime_idx, &q) in moduli.iter().enumerate() {
        let ntt_ctx = NttContext::new(n, q);
        let mut ap = vec![0u64; n];
        let mut bp = vec![0u64; n];
        for i in 0..n {
            ap[i] = a[i * num_primes + prime_idx];
            bp[i] = b[i * num_primes + prime_idx];
        }
        let prod = ntt_ctx.multiply_polynomials(&ap, &bp);
        for i in 0..n {
            result[i * num_primes + prime_idx] = prod[i];
        }
    }
    Ok(result)
}

#[cfg(feature = "v2-gpu-metal")]
fn decrypt_flat(c0: &[u64], c1: &[u64], sk: &SecretKey, moduli: &[u64], n: usize, scale: f64) -> Result<f64, String> {
    let num_primes = moduli.len();

    // Compute c1 × s
    let c1_s = multiply_flat_by_secret(c1, sk, moduli, n)?;

    // Compute m = c0 + c1×s
    let m = add_polys(c0, &c1_s, moduli, n);

    // Get first coefficient, convert to centered value
    let m0 = m[0];  // First prime
    let q0 = moduli[0];
    let m0_centered = if m0 > q0 / 2 { m0 as i64 - q0 as i64 } else { m0 as i64 };

    Ok(m0_centered as f64 / scale)
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
