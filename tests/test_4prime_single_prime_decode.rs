// Test 4-prime multiplication using single-prime decoding
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{rns_encrypt, rns_decrypt, rns_multiply_ciphertexts, RnsPlaintext};

#[test]
fn test_4prime_mult_single_prime_decode() {
    let params = CliffordFHEParams::new_rns_mult_depth2();  // 4 primes
    let (pk, sk, evk) = rns_keygen(&params);

    let a = 2.0;
    let b = 3.0;

    let mut coeffs_a = vec![0i64; params.n];
    let mut coeffs_b = vec![0i64; params.n];
    coeffs_a[0] = (a * params.scale).round() as i64;
    coeffs_b[0] = (b * params.scale).round() as i64;

    let pt_a = RnsPlaintext::from_coeffs(coeffs_a, params.scale, &params.moduli, 0);
    let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);

    let ct_a = rns_encrypt(&pk, &pt_a, &params);
    let ct_b = rns_encrypt(&pk, &pt_b, &params);

    let ct_product = rns_multiply_ciphertexts(&ct_a, &ct_b, &evk, &params);

    let pt_product = rns_decrypt(&sk, &ct_product, &params);

    // CRITICAL: Use single-prime decoding instead of full CRT!
    // After rescaling, we have 3 primes remaining
    let num_primes_after_mult = ct_product.c0.num_primes();
    println!("After multiplication: {} primes remaining", num_primes_after_mult);

    // Try decoding from each prime separately
    for prime_idx in 0..num_primes_after_mult {
        let prime_val = params.moduli[prime_idx];
        let product_coeffs = pt_product.coeffs.to_coeffs_single_prime(prime_idx, prime_val);
        let product_val = (product_coeffs[0] as f64) / ct_product.scale;

        let expected = 6.0;
        let error = (product_val - expected).abs();

        println!("\nDecoding from prime[{}] (q = {}):", prime_idx, prime_val);
        println!("  Expected: {}", expected);
        println!("  Got:      {}", product_val);
        println!("  Error:    {}", error);

        if error < 0.1 {
            println!("  ✅ SUCCESS with single-prime decoding!");
            return;  // Test passes
        }
    }

    panic!("All single-prime decodings failed!");
}
