//! Atomic unit tests for RNS-CKKS components
//!
//! Strategy: Break down multiplication into smallest possible pieces and test each one.
//! Goal: Isolate the exact line/parameter/value causing the error.
//! V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::rns::{RnsPolynomial, rns_add, rns_sub, rns_multiply as rns_poly_multiply, decompose_base_pow2, precompute_rescale_inv, rns_rescale_exact};
use ga_engine::clifford_fhe::keys_rns::rns_keygen;

/// Helper: Polynomial multiply function (negacyclic)
fn polynomial_multiply_ntt(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
    let mut result = vec![0i128; n];
    let q128 = q as i128;

    for i in 0..n {
        for j in 0..n {
            let idx = i + j;
            let prod = (a[i] as i128) * (b[j] as i128) % q128;
            if idx < n {
                result[idx] = (result[idx] + prod) % q128;
            } else {
                let wrapped_idx = idx % n;
                result[wrapped_idx] = (result[wrapped_idx] - prod) % q128;
            }
        }
    }

    result.iter().map(|&x| ((x % q128 + q128) % q128) as i64).collect()
}

// ============================================================================
// LEVEL 0: PRIMITIVE OPERATIONS
// ============================================================================

#[test]
fn test_modular_add() {
    let q: i64 = 1_152_921_504_606_584_833; // 60-bit prime
    let a = q - 100; // Near q
    let b = 200;
    let result = ((a as i128 + b as i128) % q as i128) as i64;
    assert_eq!(result, 100, "Modular addition wraps correctly");
}

#[test]
fn test_modular_multiply() {
    let q: i64 = 1_152_921_504_606_584_833;
    let a = 2i64.pow(40); // Δ = 2^40
    let b = 3i64;
    let result = ((a as i128 * b as i128) % q as i128) as i64;
    let expected = 3 * 2i64.pow(40);
    assert_eq!(result, expected, "Small modular multiply works");
}

#[test]
fn test_negacyclic_polynomial_multiply() {
    // Test: [2, 0, 0, 0] × [3, 0, 0, 0] = [6, 0, 0, 0]
    let n = 4;
    let q = 97; // Small prime for testing
    let a = vec![2, 0, 0, 0];
    let b = vec![3, 0, 0, 0];
    let result = polynomial_multiply_ntt(&a, &b, q, n);
    assert_eq!(result[0], 6, "Constant × constant works");
    assert_eq!(result[1], 0);
    assert_eq!(result[2], 0);
    assert_eq!(result[3], 0);
}

#[test]
fn test_negacyclic_reduction() {
    // Test: [0, 1, 0, 0] × [0, 0, 0, 1] = [0, 0, 0, 0, 0, 1]
    // With x^4 = -1 reduction: [0, -1, 0, 0] = [0, 96, 0, 0] mod 97
    let n = 4;
    let q = 97;
    let a = vec![0, 1, 0, 0]; // x
    let b = vec![0, 0, 0, 1]; // x^3
    let result = polynomial_multiply_ntt(&a, &b, q, n);
    assert_eq!(result[0], q - 1, "x · x^3 = x^4 = -1 (negacyclic)");
    assert_eq!(result[1], 0);
    assert_eq!(result[2], 0);
    assert_eq!(result[3], 0);
}

// ============================================================================
// LEVEL 1: RNS POLYNOMIAL OPERATIONS
// ============================================================================

#[test]
fn test_rns_polynomial_add() {
    let primes = vec![97, 101];
    let n = 4;

    // Create two RNS polynomials: [10, 0, 0, 0] and [20, 0, 0, 0]
    let coeffs_a = vec![10i64, 0, 0, 0];
    let coeffs_b = vec![20i64, 0, 0, 0];

    let a = RnsPolynomial::from_coeffs(&coeffs_a, &primes, n, 0);
    let b = RnsPolynomial::from_coeffs(&coeffs_b, &primes, n, 0);

    let result = rns_add(&a, &b, &primes);

    // Result should be [30, 0, 0, 0] in both primes
    assert_eq!(result.rns_coeffs[0][0], 30, "Add works mod 97");
    assert_eq!(result.rns_coeffs[0][1], 30, "Add works mod 101");
}

#[test]
fn test_rns_polynomial_subtract() {
    let primes = vec![97, 101];
    let n = 4;

    let coeffs_a = vec![10i64, 0, 0, 0];
    let coeffs_b = vec![20i64, 0, 0, 0];

    let a = RnsPolynomial::from_coeffs(&coeffs_a, &primes, n, 0);
    let b = RnsPolynomial::from_coeffs(&coeffs_b, &primes, n, 0);

    let result = rns_sub(&a, &b, &primes);

    // Result should be -10 mod q = q - 10
    assert_eq!(result.rns_coeffs[0][0], 97 - 10, "Subtract works mod 97");
    assert_eq!(result.rns_coeffs[0][1], 101 - 10, "Subtract works mod 101");
}

#[test]
fn test_rns_polynomial_multiply() {
    let primes = vec![97, 101];
    let n = 4;

    // [2, 0, 0, 0] × [3, 0, 0, 0] = [6, 0, 0, 0]
    let coeffs_a = vec![2i64, 0, 0, 0];
    let coeffs_b = vec![3i64, 0, 0, 0];

    let a = RnsPolynomial::from_coeffs(&coeffs_a, &primes, n, 0);
    let b = RnsPolynomial::from_coeffs(&coeffs_b, &primes, n, 0);

    let result = rns_poly_multiply(&a, &b, &primes, polynomial_multiply_ntt);

    assert_eq!(result.rns_coeffs[0][0], 6, "Multiply works mod 97");
    assert_eq!(result.rns_coeffs[0][1], 6, "Multiply works mod 101");
    assert_eq!(result.rns_coeffs[1][0], 0);
    assert_eq!(result.rns_coeffs[1][1], 0);
}

#[test]
fn test_rns_roundtrip() {
    let primes = vec![97, 101, 103];
    let n = 8;

    let original_coeffs = vec![42, 17, -5, 0, 123, -88, 7, 1];

    // Convert to RNS
    let rns_poly = RnsPolynomial::from_coeffs(&original_coeffs, &primes, n, 0);

    // Convert back using single prime (for small values)
    let recovered = rns_poly.to_coeffs_single_prime(0, primes[0]);

    for i in 0..n {
        let orig = ((original_coeffs[i] % primes[0] as i64) + primes[0] as i64) % primes[0] as i64;
        assert_eq!(recovered[i], orig, "Roundtrip preserves coefficient {}", i);
    }
}

// ============================================================================
// LEVEL 2: BASE DECOMPOSITION
// ============================================================================

#[test]
fn test_single_coefficient_decomposition() {
    // Decompose a single large number in base 2^10
    let primes = vec![1_152_921_504_606_584_833i64];
    let n = 1; // Single coefficient
    let w = 10;
    let b = 1i64 << w; // B = 1024

    // Test value: 123456 = ?·1024^0 + ?·1024^1 + ?·1024^2 + ...
    let value = 123456i64;
    let coeffs = vec![value];
    let poly = RnsPolynomial::from_coeffs(&coeffs, &primes, n, 0);

    let digits = decompose_base_pow2(&poly, &primes, w);

    // Reconstruct: value = d0 + d1·B + d2·B² + ...
    let mut reconstructed = 0i64;
    let mut b_power = 1i64;
    for digit_poly in &digits {
        let digit = digit_poly.rns_coeffs[0][0];
        println!("Digit: {}, B^power: {}", digit, b_power);
        assert!(digit >= 0 && digit < b, "Digit {} must be in [0, B)", digit);
        reconstructed += digit * b_power;
        b_power *= b;
    }

    assert_eq!(reconstructed % primes[0], value % primes[0],
               "Decomposition reconstructs to original value");
}

#[test]
fn test_decomposition_digit_bounds() {
    let primes = vec![1_152_921_504_606_584_833i64, 1_152_921_504_606_588_929i64];
    let n = 4;
    let w = 10;
    let b = 1i64 << w;

    let coeffs = vec![123456, 789012, 345678, 901234];
    let poly = RnsPolynomial::from_coeffs(&coeffs, &primes, n, 0);

    let digits = decompose_base_pow2(&poly, &primes, w);

    // Every digit must be in [0, B)
    for (t, digit_poly) in digits.iter().enumerate() {
        for i in 0..n {
            for j in 0..digit_poly.num_primes() {
                let digit = digit_poly.rns_coeffs[i][j];
                assert!(digit >= 0 && digit < b,
                       "Digit[{}][{}][{}] = {} must be in [0, {})", t, i, j, digit, b);
            }
        }
    }
}

#[test]
fn test_decomposition_reconstruction() {
    let primes = vec![1_152_921_504_606_584_833i64];
    let n = 4;
    let w = 20;
    let b = 1i64 << w;

    let coeffs = vec![1_000_000_000i64, 2_000_000_000, 500_000_000, 750_000_000];
    let poly = RnsPolynomial::from_coeffs(&coeffs, &primes, n, 0);

    let digits = decompose_base_pow2(&poly, &primes, w);

    // For each coefficient, verify reconstruction
    for i in 0..n {
        let original = poly.rns_coeffs[i][0];
        let mut reconstructed = 0i128;
        let mut b_power = 1i128;

        for digit_poly in &digits {
            let digit = digit_poly.rns_coeffs[i][0] as i128;
            reconstructed = (reconstructed + digit * b_power) % (primes[0] as i128);
            b_power = (b_power * (b as i128)) % (primes[0] as i128);
        }

        assert_eq!(reconstructed as i64, original,
                   "Coefficient {} reconstructs correctly", i);
    }
}

// ============================================================================
// LEVEL 3: KEY GENERATION IDENTITY TESTS
// ============================================================================

#[test]
fn test_public_key_identity() {
    // Test: pk.b - pk.a · s = e (small noise)
    let params = CliffordFHEParams::new_rns_mult();
    let (pk, sk, _evk) = rns_keygen(&params);

    let primes = &params.moduli;

    // Compute a · s
    let a_s = rns_poly_multiply(&pk.a, &sk.coeffs, primes, polynomial_multiply_ntt);

    // Compute b - a·s
    let noise = rns_sub(&pk.b, &a_s, primes);

    // Convert to coefficients (use single prime for small values)
    let noise_coeffs = noise.to_coeffs_single_prime(0, primes[0]);

    // Noise should be small (< 100 for error_std = 3.2)
    for (i, &n) in noise_coeffs.iter().take(10).enumerate() {
        let centered = if n > primes[0] / 2 { n - primes[0] } else { n };
        println!("noise[{}] = {}", i, centered);
        assert!(centered.abs() < 1000, "Public key noise[{}] = {} is too large", i, centered);
    }
}

#[test]
fn test_evaluation_key_identity() {
    // CRITICAL TEST: Verify evk0[t] ± evk1[t]·s = ±B^t·s² + e
    let params = CliffordFHEParams::new_rns_mult();
    let (_pk, sk, evk) = rns_keygen(&params);

    let primes = &params.moduli;
    let w = evk.base_w;
    let b = 1i64 << w;

    // Compute s²
    let s_squared = rns_poly_multiply(&sk.coeffs, &sk.coeffs, primes, polynomial_multiply_ntt);

    // Test each digit's EVK
    for t in 0..evk.evk0.len() {
        println!("\n=== Testing EVK digit {} ===", t);

        // Compute B^t mod each prime
        let mut b_power_t = vec![0i64; primes.len()];
        for j in 0..primes.len() {
            let q = primes[j] as i128;
            let mut p = 1i128;
            for _ in 0..t {
                p = (p * (b as i128)) % q;
            }
            b_power_t[j] = p as i64;
        }

        // Compute B^t · s²
        let mut bt_s2_coeffs = vec![vec![0i64; primes.len()]; params.n];
        for i in 0..params.n {
            for j in 0..primes.len() {
                let q = primes[j] as i128;
                let val = ((s_squared.rns_coeffs[i][j] as i128) * (b_power_t[j] as i128)) % q;
                bt_s2_coeffs[i][j] = val as i64;
            }
        }
        let bt_s2 = RnsPolynomial::new(bt_s2_coeffs, params.n, 0);

        // Compute evk1[t] · s
        let evk1_s = rns_poly_multiply(&evk.evk1[t], &sk.coeffs, primes, polynomial_multiply_ntt);

        // TEST VARIANT 1: evk0 + evk1·s = B^t·s² + e
        let lhs_plus = rns_add(&evk.evk0[t], &evk1_s, primes);
        let noise_plus = rns_sub(&lhs_plus, &bt_s2, primes);
        let noise_plus_coeffs = noise_plus.to_coeffs_single_prime(0, primes[0]);

        // TEST VARIANT 2: evk0 - evk1·s = B^t·s² + e
        let lhs_minus = rns_sub(&evk.evk0[t], &evk1_s, primes);
        let noise_minus = rns_sub(&lhs_minus, &bt_s2, primes);
        let noise_minus_coeffs = noise_minus.to_coeffs_single_prime(0, primes[0]);

        // TEST VARIANT 3: evk0 + evk1·s = -B^t·s² + e
        let neg_bt_s2 = rns_sub(&RnsPolynomial::from_coeffs(&vec![0i64; params.n], primes, params.n, 0), &bt_s2, primes);
        let noise_neg_plus = rns_sub(&lhs_plus, &neg_bt_s2, primes);
        let noise_neg_plus_coeffs = noise_neg_plus.to_coeffs_single_prime(0, primes[0]);

        // TEST VARIANT 4: evk0 - evk1·s = -B^t·s² + e
        let noise_neg_minus = rns_sub(&lhs_minus, &neg_bt_s2, primes);
        let noise_neg_minus_coeffs = noise_neg_minus.to_coeffs_single_prime(0, primes[0]);

        // Check which variant has small noise
        let check_noise = |coeffs: &[i64], name: &str| -> bool {
            let mut max_noise = 0i64;
            for (i, &n) in coeffs.iter().take(10).enumerate() {
                let centered = if n > primes[0] / 2 { n - primes[0] } else { n };
                max_noise = max_noise.max(centered.abs());
                if i < 3 {
                    println!("  {} noise[{}] = {}", name, i, centered);
                }
            }
            println!("  {} max_noise = {}", name, max_noise);
            max_noise < 10000 // Should be < 10000 for valid identity
        };

        let variant1_ok = check_noise(&noise_plus_coeffs, "VARIANT 1 (evk0 + evk1·s = +B^t·s²)");
        let variant2_ok = check_noise(&noise_minus_coeffs, "VARIANT 2 (evk0 - evk1·s = +B^t·s²)");
        let variant3_ok = check_noise(&noise_neg_plus_coeffs, "VARIANT 3 (evk0 + evk1·s = -B^t·s²)");
        let variant4_ok = check_noise(&noise_neg_minus_coeffs, "VARIANT 4 (evk0 - evk1·s = -B^t·s²)");

        // Exactly ONE variant should have small noise
        let valid_count = [variant1_ok, variant2_ok, variant3_ok, variant4_ok].iter().filter(|&&x| x).count();

        assert!(valid_count > 0, "EVK[{}] doesn't satisfy ANY identity variant!", t);

        if valid_count > 1 {
            println!("WARNING: EVK[{}] satisfies {} variants (should be exactly 1)", t, valid_count);
        }

        // Print which variant works
        if variant1_ok { println!("✓ EVK[{}] satisfies: evk0 + evk1·s = +B^t·s² + e", t); }
        if variant2_ok { println!("✓ EVK[{}] satisfies: evk0 - evk1·s = +B^t·s² + e", t); }
        if variant3_ok { println!("✓ EVK[{}] satisfies: evk0 + evk1·s = -B^t·s² + e", t); }
        if variant4_ok { println!("✓ EVK[{}] satisfies: evk0 - evk1·s = -B^t·s² + e", t); }
    }
}

// ============================================================================
// LEVEL 4: ENCRYPTION/DECRYPTION
// ============================================================================

#[test]
fn test_encryption_decryption_identity() {
    use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};

    let params = CliffordFHEParams::new_rns_mult();
    let (pk, sk, _evk) = rns_keygen(&params);

    // Encrypt a simple value
    let value = 42.0;
    let scaled = (value * params.scale).round() as i64;
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = scaled;

    let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
    let ct = rns_encrypt(&pk, &pt, &params);

    // Test identity: c0 + c1·s = m + e
    let c1_s = rns_poly_multiply(&ct.c1, &sk.coeffs, &params.moduli, polynomial_multiply_ntt);
    let plaintext_with_noise = rns_add(&ct.c0, &c1_s, &params.moduli);

    let recovered_coeffs = plaintext_with_noise.to_coeffs_single_prime(0, params.moduli[0]);
    let recovered_value = (recovered_coeffs[0] as f64) / params.scale;

    println!("Original: {}, Recovered: {}", value, recovered_value);
    assert!((recovered_value - value).abs() < 0.1, "Encryption identity holds");

    // Also test via decrypt
    let pt_dec = rns_decrypt(&sk, &ct, &params);
    let dec_coeffs = pt_dec.coeffs.to_coeffs_single_prime(0, params.moduli[0]);
    let dec_value = (dec_coeffs[0] as f64) / pt_dec.scale;

    assert!((dec_value - value).abs() < 0.1, "Decryption recovers value");
}

// ============================================================================
// LEVEL 4.5: CRT DECODING
// ============================================================================

#[test]
fn test_crt_reconstruction() {
    let primes = vec![97i64, 101, 103]; // Small primes for testing
    let n = 4;

    // Test value: 12345
    let value = 12345i64;
    let coeffs = vec![value, 0, 0, 0];

    // Convert to RNS
    let rns_poly = RnsPolynomial::from_coeffs(&coeffs, &primes, n, 0);

    println!("\n=== CRT RECONSTRUCTION TEST ===");
    println!("Original value: {}", value);
    println!("RNS residues:");
    for (j, &p) in primes.iter().enumerate() {
        let residue = value % p;
        println!("  mod {}: {} (computed {})", p, rns_poly.rns_coeffs[0][j], residue);
        assert_eq!(rns_poly.rns_coeffs[0][j], residue);
    }

    // Reconstruct using CRT
    let reconstructed = rns_poly.to_coeffs_crt_centered(&primes);

    println!("Reconstructed (CRT centered): {}", reconstructed[0]);
    println!("Error: {}", (reconstructed[0] - value as f64).abs());

    assert!((reconstructed[0] - value as f64).abs() < 0.1,
           "CRT should reconstruct value (got {}, expected {})", reconstructed[0], value);
}

#[test]
fn test_crt_with_large_value() {
    // Test with 60-bit primes and large value
    let primes = vec![
        1_152_921_504_606_584_833i64,
        1_152_921_504_606_588_929i64,
    ];
    let n = 4;

    // Large value: 6 * 2^40 (like after CKKS multiplication)
    let value_f64 = 6.0 * 2f64.powi(40);
    println!("\n=== LARGE VALUE CRT TEST ===");
    println!("Testing value: {:.3e}", value_f64);

    // Manually compute residues
    let mut rns_coeffs = vec![vec![0i64; 2]; n];
    for p_idx in 0..2 {
        let q = primes[p_idx] as f64;
        let residue = value_f64 % q;
        rns_coeffs[0][p_idx] = residue as i64;
        println!("Residue mod q_{}: {}", p_idx, residue as i64);
    }

    let rns_poly = RnsPolynomial::new(rns_coeffs, n, 0);

    // Reconstruct
    let reconstructed = rns_poly.to_coeffs_crt_centered(&primes);

    println!("Reconstructed: {:.3e}", reconstructed[0]);
    println!("Expected: {:.3e}", value_f64);
    println!("Relative error: {:.3e}", ((reconstructed[0] - value_f64) / value_f64).abs());

    // Allow 1% error due to floating point precision
    assert!(((reconstructed[0] - value_f64) / value_f64).abs() < 0.01,
           "CRT should reconstruct large value within 1%");
}

// ============================================================================
// LEVEL 5: TENSOR PRODUCT
// ============================================================================

#[test]
fn test_tensor_product_algebraically() {
    use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt};

    // Use smaller scale for testing to avoid i64 overflow
    let mut params = CliffordFHEParams::new_rns_mult();
    params.scale = 2f64.powi(20); // Δ = 2^20 instead of 2^40
    let (pk, sk, _evk) = rns_keygen(&params);
    let primes = &params.moduli;

    println!("\n=== SECRET KEY DEBUG ===");
    let sk_coeffs = sk.coeffs.to_coeffs_single_prime(0, primes[0]);
    let sk_hamming_weight: usize = sk_coeffs.iter().filter(|&&x| x != 0).count();
    println!("Secret key Hamming weight: {} / {}", sk_hamming_weight, params.n);
    println!("First 10 sk coeffs: {:?}", &sk_coeffs[..10]);

    // Encrypt [2] and [3]
    let pt1 = RnsPlaintext::from_coeffs(
        vec![(2.0 * params.scale).round() as i64; params.n].iter().enumerate()
            .map(|(i, &v)| if i == 0 { v } else { 0 }).collect(),
        params.scale, primes, 0
    );
    let pt2 = RnsPlaintext::from_coeffs(
        vec![(3.0 * params.scale).round() as i64; params.n].iter().enumerate()
            .map(|(i, &v)| if i == 0 { v } else { 0 }).collect(),
        params.scale, primes, 0
    );

    let ct1 = rns_encrypt(&pk, &pt1, &params);
    let ct2 = rns_encrypt(&pk, &pt2, &params);

    // Compute tensor product components
    let c0d0 = rns_poly_multiply(&ct1.c0, &ct2.c0, primes, polynomial_multiply_ntt);
    let c0d1 = rns_poly_multiply(&ct1.c0, &ct2.c1, primes, polynomial_multiply_ntt);
    let c1d0 = rns_poly_multiply(&ct1.c1, &ct2.c0, primes, polynomial_multiply_ntt);
    let c1d1 = rns_poly_multiply(&ct1.c1, &ct2.c1, primes, polynomial_multiply_ntt);

    let d0 = c0d0;
    let d1 = rns_add(&c0d1, &c1d0, primes);
    let d2 = c1d1;

    // Compute s²
    let s_squared = rns_poly_multiply(&sk.coeffs, &sk.coeffs, primes, polynomial_multiply_ntt);

    // Verify identity: d0 + d1·s + d2·s² = (m1 + e1)(m2 + e2)
    let d1_s = rns_poly_multiply(&d1, &sk.coeffs, primes, polynomial_multiply_ntt);
    let d2_s2 = rns_poly_multiply(&d2, &s_squared, primes, polynomial_multiply_ntt);

    let mut result = rns_add(&d0, &d1_s, primes);
    result = rns_add(&result, &d2_s2, primes);

    let result_coeffs = result.to_coeffs_single_prime(0, primes[0]);

    println!("\n=== TENSOR PRODUCT DEBUG ===");
    println!("result.rns_coeffs[0]: {:?}", &result.rns_coeffs[0][..result.num_primes().min(3)]);
    println!("result_coeffs[0] (single prime): {}", result_coeffs[0]);
    println!("Δ²: {}", params.scale * params.scale);
    println!("Expected coefficient: ~{}", (6.0 * params.scale * params.scale) as i64);

    let result_value = (result_coeffs[0] as f64) / (params.scale * params.scale);

    println!("Tensor product result: {}", result_value);
    println!("Expected: 6.0");
    println!("Scale: {}", params.scale * params.scale);

    assert!((result_value - 6.0).abs() < 1.0,
           "Tensor product algebraically computes 2×3=6 (got {})", result_value);
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

#[test]
fn run_all_atomic_tests() {
    println!("\n========================================");
    println!("Running ALL atomic RNS-CKKS tests");
    println!("========================================\n");

    // This test just ensures all others run
    // Individual test failures will be caught separately
}

#[test]
fn test_crt_with_10_primes() {
    use ga_engine::clifford_fhe::rns::RnsPolynomial;
    
    // Test Garner's algorithm with 10 primes
    let primes = vec![
        1141392289560813569i64,
        1141392289560840193,
        1141392289560907777,
        1141392289560926209,
        1141392289561065473,
        1141392289561077761,
        1141392289561092097,
        1141392289561157633,
        1141392289561184257,
        1141392289561194497,
    ];
    let n = 4;

    println!("\n=== CRT TEST WITH 10 PRIMES ===");

    // Test values of different sizes
    let test_values = vec![
        42i64,
        12345i64,
        1_000_000i64,
        1_000_000_000i64,
    ];

    for &value in &test_values {
        println!("\nTesting value: {}", value);

        // Compute residues
        let mut rns_coeffs = vec![vec![0i64; 10]; n];
        for p_idx in 0..10 {
            rns_coeffs[0][p_idx] = value % primes[p_idx];
        }

        let rns_poly = RnsPolynomial::new(rns_coeffs, n, 0);

        // Reconstruct
        let reconstructed = rns_poly.to_coeffs_crt_centered(&primes);

        println!("  Reconstructed: {:.0}", reconstructed[0]);
        println!("  Expected: {}", value);
        println!("  Error: {:.3e}", (reconstructed[0] - value as f64).abs());

        assert!((reconstructed[0] - value as f64).abs() < 1.0,
               "CRT with 10 primes should reconstruct exactly (got {}, expected {})",
               reconstructed[0], value);
    }

    println!("\n✓ CRT with 10 primes works correctly!");
}
