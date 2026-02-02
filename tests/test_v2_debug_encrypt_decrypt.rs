//! Micro-tests to isolate V2 encryption/decryption issues
//!
//! Strategy: Test one operation at a time, starting from basics
//!
//! Run with: cargo test --test test_v2_debug_encrypt_decrypt --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

/// Helper to decrypt a single value
fn decrypt_value(
    ct: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext,
    ckks_ctx: &CkksContext,
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
) -> f64 {
    let pt = ckks_ctx.decrypt(ct, sk);
    let val = pt.coeffs[0].values[0] as i64;
    let q = pt.coeffs[0].moduli[0] as i64;

    // Centered lift: convert from [0, q) to (-q/2, q/2]
    let centered = if val > q / 2 {
        val - q
    } else {
        val
    };

    (centered as f64) / ct.scale
}

/// Helper to encrypt a single value
fn encrypt_value(
    value: f64,
    ckks_ctx: &CkksContext,
    pk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey,
) -> ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;

    let params = &ckks_ctx.params;
    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let n = params.n;

    // Create plaintext with value in first coefficient
    let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); n];

    // Convert to scaled integer, handling negatives properly
    let scaled_val = (value * params.scale).round() as i64;

    // Convert to RNS representation (handles negative by taking mod q)
    let values: Vec<u64> = moduli.iter().map(|&q| {
        let q_i64 = q as i64;
        let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
        normalized as u64
    }).collect();

    coeffs[0] = RnsRepresentation::new(values, moduli.clone());

    let pt = Plaintext::new(coeffs, params.scale, level);
    ckks_ctx.encrypt(&pt, pk)
}

#[test]
fn test_1_basic_encrypt_decrypt() {
    println!("\n========== TEST 1: Basic Encrypt/Decrypt ==========");

    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  Number of primes = {}", params.moduli.len());
    println!("  Scale = 2^{}", (params.scale as f64).log2() as u32);
    println!("  Primes: {:?}", params.moduli);

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Test multiple values
    let test_values = vec![1.0, 2.0, 3.5, -1.5, 0.0, 100.0];

    for value in test_values {
        let ct = encrypt_value(value, &ckks_ctx, &pk);
        let decrypted = decrypt_value(&ct, &ckks_ctx, &sk);
        let error = (decrypted - value).abs();

        println!("  Encrypt({}) -> Decrypt = {} (error: {:.2e})",
                 value, decrypted, error);

        assert!(error < 1e-6,
                "Basic encrypt/decrypt failed! value={}, decrypted={}, error={:.2e}",
                value, decrypted, error);
    }

    println!("✓ TEST 1 PASSED: Basic encrypt/decrypt works!\n");
}

#[test]
fn test_2_ciphertext_addition() {
    println!("\n========== TEST 2: Ciphertext Addition ==========");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    let a_val = 3.0;
    let b_val = 5.0;
    let expected = a_val + b_val;

    println!("  Computing {} + {} = {} (expected)", a_val, b_val, expected);

    let ct_a = encrypt_value(a_val, &ckks_ctx, &pk);
    let ct_b = encrypt_value(b_val, &ckks_ctx, &pk);

    // Add ciphertexts
    let ct_sum = add_ciphertexts(&ct_a, &ct_b);

    let result = decrypt_value(&ct_sum, &ckks_ctx, &sk);
    let error = (result - expected).abs();

    println!("  Decrypted result: {} (error: {:.2e})", result, error);

    assert!(error < 1e-6,
            "Addition failed! expected={}, got={}, error={:.2e}",
            expected, result, error);

    println!("✓ TEST 2 PASSED: Ciphertext addition works!\n");
}

#[test]
fn test_3_scalar_multiplication() {
    println!("\n========== TEST 3: Scalar Multiplication ==========");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    let value = 4.0;
    let scalar = 3.5;
    let expected = value * scalar;

    println!("  Computing {} * {} = {} (expected)", value, scalar, expected);

    let ct = encrypt_value(value, &ckks_ctx, &pk);
    let ct_scaled = mul_scalar(&ct, scalar);

    let result = decrypt_value(&ct_scaled, &ckks_ctx, &sk);
    let error = (result - expected).abs();

    println!("  Decrypted result: {} (error: {:.2e})", result, error);

    assert!(error < 1e-6,
            "Scalar multiplication failed! expected={}, got={}, error={:.2e}",
            expected, result, error);

    println!("✓ TEST 3 PASSED: Scalar multiplication works!\n");
}

// Helper: Add two ciphertexts component-wise
fn add_ciphertexts(
    a: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext,
    b: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext,
) -> ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;

    let c0: Vec<_> = a.c0.iter().zip(&b.c0).map(|(a_i, b_i)| a_i.add(b_i)).collect();
    let c1: Vec<_> = a.c1.iter().zip(&b.c1).map(|(a_i, b_i)| a_i.add(b_i)).collect();

    Ciphertext::new(c0, c1, a.level, a.scale)
}

// Helper: Multiply ciphertext by scalar
fn mul_scalar(
    ct: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext,
    scalar: f64,
) -> ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;

    // Scalar multiplication in CKKS: multiply coefficients by scalar (as integer)
    // For fractional scalars, scale stays the same
    // The encrypted value is already scaled by 'scale', so we just multiply by scalar

    let c0: Vec<_> = ct.c0.iter().map(|rns| {
        let new_values: Vec<u64> = rns.values.iter()
            .zip(&rns.moduli)
            .map(|(&val, &q)| {
                let val_i64 = if val > q / 2 {
                    (val as i64) - (q as i64)
                } else {
                    val as i64
                };

                let result_i64 = (val_i64 as f64 * scalar).round() as i64;

                let q_i64 = q as i64;
                let normalized = ((result_i64 % q_i64) + q_i64) % q_i64;
                normalized as u64
            })
            .collect();
        RnsRepresentation::new(new_values, rns.moduli.clone())
    }).collect();

    let c1: Vec<_> = ct.c1.iter().map(|rns| {
        let new_values: Vec<u64> = rns.values.iter()
            .zip(&rns.moduli)
            .map(|(&val, &q)| {
                let val_i64 = if val > q / 2 {
                    (val as i64) - (q as i64)
                } else {
                    val as i64
                };

                let result_i64 = (val_i64 as f64 * scalar).round() as i64;

                let q_i64 = q as i64;
                let normalized = ((result_i64 % q_i64) + q_i64) % q_i64;
                normalized as u64
            })
            .collect();
        RnsRepresentation::new(new_values, rns.moduli.clone())
    }).collect();

    // Scale stays the same for scalar multiplication
    Ciphertext::new(c0, c1, ct.level, ct.scale)
}

#[test]
fn test_4_ciphertext_multiplication_no_relin() {
    println!("\n========== TEST 4: Ciphertext Multiplication (No Relinearization) ==========");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    let a_val = 3.0;
    let b_val = 4.0;
    let expected = a_val * b_val;

    println!("  Computing {} * {} = {} (expected)", a_val, b_val, expected);

    let ct_a = encrypt_value(a_val, &ckks_ctx, &pk);
    let ct_b = encrypt_value(b_val, &ckks_ctx, &pk);

    println!("  Before multiplication:");
    println!("    ct_a.level = {}, ct_a.scale = {:.2e}", ct_a.level, ct_a.scale);
    println!("    ct_b.level = {}, ct_b.scale = {:.2e}", ct_b.level, ct_b.scale);

    // Multiply ciphertexts (produces degree-2 ciphertext)
    // This is just tensor product, no relinearization yet
    let ct_mult_deg2 = multiply_ciphertexts_no_relin(&ct_a, &ct_b, &params);

    println!("  After multiplication (degree-2):");
    println!("    ct_mult.c0.len() = {}", ct_mult_deg2.c0.len());
    println!("    ct_mult.c1.len() = {}", ct_mult_deg2.c1.len());
    println!("    ct_mult.c2.len() = {}", ct_mult_deg2.c2.as_ref().map_or(0, |c| c.len()));
    println!("    ct_mult.level = {}, ct_mult.scale = {:.2e}", ct_mult_deg2.level, ct_mult_deg2.scale);

    // For degree-2 ciphertext, decrypt formula is: m = c0 + c1*s + c2*s^2
    println!("  DEBUG: First coefficient of c0: {}", ct_mult_deg2.c0[0].values[0]);
    println!("  DEBUG: Expected scaled value: {:.2e}", expected * ct_mult_deg2.scale);

    let result = decrypt_degree2_value(&ct_mult_deg2, &ckks_ctx, &sk);
    let error = (result - expected).abs();

    println!("  Decrypted result: {} (error: {:.2e})", result, error);

    assert!(error < 1e-3,
            "Multiplication (no relin) failed! expected={}, got={}, error={:.2e}",
            expected, result, error);

    println!("✓ TEST 4 PASSED: Ciphertext multiplication (no relin) works!\n");
}

// Helper: Multiply two ciphertexts to produce degree-2 ciphertext (c0, c1, c2)
fn multiply_ciphertexts_no_relin(
    a: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext,
    b: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext,
    params: &CliffordFHEParams,
) -> Ciphertext2 {
    

    let n = params.n;
    let level = a.level.min(b.level);
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();

    // Tensor product: (a0 + a1*s) * (b0 + b1*s) = a0*b0 + (a0*b1 + a1*b0)*s + a1*b1*s^2

    // c0 = a0 * b0
    let c0 = multiply_polys(&a.c0, &b.c0, n, &moduli);

    // c1 = a0*b1 + a1*b0
    let a0_b1 = multiply_polys(&a.c0, &b.c1, n, &moduli);
    let a1_b0 = multiply_polys(&a.c1, &b.c0, n, &moduli);
    let c1 = add_polys(&a0_b1, &a1_b0);

    // c2 = a1 * b1
    let c2 = multiply_polys(&a.c1, &b.c1, n, &moduli);

    Ciphertext2 {
        c0,
        c1,
        c2: Some(c2),
        level,
        scale: a.scale * b.scale,
    }
}

// Degree-2 ciphertext (c0, c1, c2)
struct Ciphertext2 {
    c0: Vec<RnsRepresentation>,
    c1: Vec<RnsRepresentation>,
    c2: Option<Vec<RnsRepresentation>>,
    level: usize,
    scale: f64,
}

// Helper: Decrypt degree-2 ciphertext
fn decrypt_degree2_value(
    ct: &Ciphertext2,
    ckks_ctx: &CkksContext,
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
) -> f64 {
    // m = c0 + c1*s + c2*s^2

    // Extract secret key at correct level
    let sk_at_level: Vec<RnsRepresentation> = sk.coeffs.iter()
        .map(|rns| {
            let values = rns.values[..=ct.level].to_vec();
            let moduli = rns.moduli[..=ct.level].to_vec();
            RnsRepresentation::new(values, moduli)
        })
        .collect();

    let params = &ckks_ctx.params;
    let n = params.n;
    let moduli: Vec<u64> = params.moduli[..=ct.level].to_vec();

    // Compute c1*s
    let c1_s = multiply_polys(&ct.c1, &sk_at_level, n, &moduli);

    // Compute s^2
    let s_squared = multiply_polys(&sk_at_level, &sk_at_level, n, &moduli);

    // Compute c2*s^2
    let c2_s2 = multiply_polys(ct.c2.as_ref().unwrap(), &s_squared, n, &moduli);

    // m = c0 + c1*s + c2*s^2
    let temp = add_polys(&ct.c0, &c1_s);
    let m = add_polys(&temp, &c2_s2);

    println!("      DEBUG decrypt_degree2:");
    println!("        c0[0] = {}", ct.c0[0].values[0]);
    println!("        c1_s[0] = {}", c1_s[0].values[0]);
    println!("        c2_s2[0] = {}", c2_s2[0].values[0]);
    println!("        m[0] (before centering) = {}", m[0].values[0]);

    // Extract first coefficient and decode
    let val = m[0].values[0] as i64;
    let q = m[0].moduli[0] as i64;

    let centered = if val > q / 2 {
        val - q
    } else {
        val
    };

    println!("        m[0] (after centering) = {}", centered);
    println!("        scale = {:.2e}", ct.scale);

    (centered as f64) / ct.scale
}

// Helper: Multiply two polynomials using NTT
fn multiply_polys(
    a: &[RnsRepresentation],
    b: &[RnsRepresentation],
    n: usize,
    moduli: &[u64],
) -> Vec<RnsRepresentation> {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;

    let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.to_vec()); n];

    for (prime_idx, &q) in moduli.iter().enumerate() {
        let ntt_ctx = NttContext::new(n, q);

        let a_mod_q: Vec<u64> = a.iter().map(|rns| rns.values[prime_idx]).collect();
        let b_mod_q: Vec<u64> = b.iter().map(|rns| rns.values[prime_idx]).collect();

        let product_mod_q = ntt_ctx.multiply_polynomials(&a_mod_q, &b_mod_q);

        for i in 0..n {
            result[i].values[prime_idx] = product_mod_q[i];
        }
    }

    result
}

// Helper: Add two polynomials
fn add_polys(
    a: &[RnsRepresentation],
    b: &[RnsRepresentation],
) -> Vec<RnsRepresentation> {
    a.iter().zip(b).map(|(a_i, b_i)| a_i.add(b_i)).collect()
}

#[test]
fn test_5_ciphertext_multiplication_with_relin_and_rescale() {
    println!("
========== TEST 5: Ciphertext Multiplication (With Relin + Rescale) ==========");

    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::multiplication::multiply_ciphertexts;

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    let a_val = 3.0;
    let b_val = 4.0;
    let expected = a_val * b_val;

    println!("  Computing {} * {} = {} (expected)", a_val, b_val, expected);

    let ct_a = encrypt_value(a_val, &ckks_ctx, &pk);
    let ct_b = encrypt_value(b_val, &ckks_ctx, &pk);

    println!("  Before multiplication:");
    println!("    ct_a.level = {}, ct_a.scale = {:.2e}", ct_a.level, ct_a.scale);
    println!("    ct_b.level = {}, ct_b.scale = {:.2e}", ct_b.level, ct_b.scale);

    // Use V2s multiply_ciphertexts (includes tensor + relin + rescale)
    let ct_result = multiply_ciphertexts(&ct_a, &ct_b, &evk, &key_ctx);

    println!("  After multiplication (with relin + rescale):");
    println!("    ct_result.level = {}, ct_result.scale = {:.2e}", ct_result.level, ct_result.scale);

    let result = decrypt_value(&ct_result, &ckks_ctx, &sk);
    let error = (result - expected).abs();

    println!("  Decrypted result: {} (error: {:.2e})", result, error);

    assert!(error < 1e-3,
            "Multiplication with relin+rescale failed! expected={}, got={}, error={:.2e}",
            expected, result, error);

    println!("✓ TEST 5 PASSED: Full ciphertext multiplication works!
");
}


#[test]
fn test_6_debug_multiply_ciphertexts_output() {
    println!("\n========== TEST 6: Debug multiply_ciphertexts Output ==========");

    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::multiplication::multiply_ciphertexts;

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    let a_val = 3.0;
    let b_val = 4.0;

    let ct_a = encrypt_value(a_val, &ckks_ctx, &pk);
    let ct_b = encrypt_value(b_val, &ckks_ctx, &pk);

    println!("  Before multiply:");
    println!("    ct_a.c0[0]: {:?}", ct_a.c0[0].values.iter().collect::<Vec<_>>());
    println!("    ct_b.c0[0]: {:?}", ct_b.c0[0].values.iter().collect::<Vec<_>>());

    let ct_result = multiply_ciphertexts(&ct_a, &ct_b, &evk, &key_ctx);

    println!("  After multiply:");
    println!("    ct_result.level = {}", ct_result.level);
    println!("    ct_result.scale = {:.2e}", ct_result.scale);
    println!("    ct_result.c0[0]: {:?}", ct_result.c0[0].values.iter().collect::<Vec<_>>());
    println!("    ct_result.c1[0]: {:?}", ct_result.c1[0].values.iter().collect::<Vec<_>>());

    // Expected: 3 * 4 * scale = 12 * 1.1e12 = 1.32e13
    println!("    Expected value (scaled): {:.2e}", 12.0 * ct_result.scale);

    let result = decrypt_value(&ct_result, &ckks_ctx, &sk);
    println!("    Decrypted: {}", result);
    println!("    Expected: 12.0");
}
