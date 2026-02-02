// Minimal test: does multiplication work with 5 primes?
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]


use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{rns_encrypt, rns_decrypt, RnsPlaintext, rns_multiply_ciphertexts};

fn decode_value(pt: &RnsPlaintext, scale: f64, primes: &[i64], level: usize) -> f64 {
    let num_primes = primes.len() - level;
    let active_primes = &primes[..num_primes];

    let val = pt.coeffs.rns_coeffs[0][0];
    let q = active_primes[0];
    let centered = if val > q / 2 { val - q } else { val };

    (centered as f64) / scale
}

#[test]
fn test_mult_with_5_primes() {
    let params = CliffordFHEParams::new_rns_mult_depth2_safe(); // 5 primes

    let (pk, sk, evk) = rns_keygen(&params);

    let a = 2.0;
    let b = 3.0;
    let expected = a * b;  // 6.0

    let mut coeffs_a = vec![0i64; params.n];
    let mut coeffs_b = vec![0i64; params.n];
    coeffs_a[0] = (a * params.scale).round() as i64;
    coeffs_b[0] = (b * params.scale).round() as i64;

    let pt_a = RnsPlaintext::from_coeffs(coeffs_a, params.scale, &params.moduli, 0);
    let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);

    let ct_a = rns_encrypt(&pk, &pt_a, &params);
    let ct_b = rns_encrypt(&pk, &pt_b, &params);

    println!("Fresh ct_a: level={}, num_primes={}, scale={}", ct_a.level, ct_a.c0.num_primes(), ct_a.scale);
    println!("Fresh ct_b: level={}, num_primes={}, scale={}", ct_b.level, ct_b.c0.num_primes(), ct_b.scale);

    // Decrypt and check fresh ciphertexts are correct
    let pt_a_dec = rns_decrypt(&sk, &ct_a, &params);
    let pt_b_dec = rns_decrypt(&sk, &ct_b, &params);
    let a_dec = decode_value(&pt_a_dec, ct_a.scale, &params.moduli, ct_a.level);
    let b_dec = decode_value(&pt_b_dec, ct_b.scale, &params.moduli, ct_b.level);
    println!("Decrypted a: {} (expected {})", a_dec, a);
    println!("Decrypted b: {} (expected {})", b_dec, b);

    let ct_prod = rns_multiply_ciphertexts(&ct_a, &ct_b, &evk, &params);

    println!("After mult: level={}, num_primes={}, scale={}", ct_prod.level, ct_prod.c0.num_primes(), ct_prod.scale);

    let pt_prod = rns_decrypt(&sk, &ct_prod, &params);

    let result = decode_value(&pt_prod, ct_prod.scale, &params.moduli, ct_prod.level);
    let error = (result - expected).abs();

    println!("Expected: {}, Got: {}, Error: {}", expected, result, error);

    assert!(error < 0.1, "Error too large: {}", error);
}
