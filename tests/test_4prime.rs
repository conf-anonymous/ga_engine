// Test with 4 primes
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{rns_encrypt, rns_decrypt, rns_multiply_ciphertexts, RnsPlaintext};

#[test]
fn test_4prime_mult() {
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
    let product_coeffs = pt_product.to_coeffs(&params.moduli);
    let product_val = (product_coeffs[0] as f64) / ct_product.scale;

    let expected = 6.0;
    let error = (product_val - expected).abs();

    println!("\nExpected: {}", expected);
    println!("Got:      {}", product_val);
    println!("Error:    {}", error);

    assert!(error < 0.1, "Error too large: {}", error);
}
