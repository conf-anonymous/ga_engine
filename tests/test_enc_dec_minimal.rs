//! Minimal encryption/decryption test to isolate the bug
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]


use ga_engine::clifford_fhe_v1::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v1::keys_rns::rns_keygen;
use ga_engine::clifford_fhe_v1::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};

#[test]
fn test_minimal_enc_dec() {
    let mut params = CliffordFHEParams::new_rns_mult();
    params.scale = 1024.0; // Small scale for testing

    let (pk, sk, _evk) = rns_keygen(&params);

    // Encrypt simple value
    let value = 5.0;
    let scaled = (value * params.scale).round() as i64;
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = scaled;

    println!("\n=== INPUT ===");
    println!("Value: {}", value);
    println!("Scale: {}", params.scale);
    println!("Scaled coefficient: {}", scaled);

    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);

    println!("\n=== PLAINTEXT RNS ===");
    println!("pt.coeffs.rns_coeffs[0]: {:?}", &pt.coeffs.rns_coeffs[0][..3]);

    let ct = rns_encrypt(&pk, &pt, &params);

    println!("\n=== CIPHERTEXT ===");
    println!("ct.c0.rns_coeffs[0]: {:?}", &ct.c0.rns_coeffs[0][..3]);
    println!("ct.c1.rns_coeffs[0]: {:?}", &ct.c1.rns_coeffs[0][..3]);

    // Use built-in decryption
    let decrypted_pt = rns_decrypt(&sk, &ct, &params);

    // Reconstruct coefficients using single prime (as per comment in code)
    let recovered_coeffs = decrypted_pt.coeffs.to_coeffs_single_prime(0, params.moduli[0]);
    let recovered_value = (recovered_coeffs[0] as f64) / params.scale;

    println!("\n=== RESULT ===");
    println!("Recovered coefficient: {}", recovered_coeffs[0]);
    println!("Recovered value: {}", recovered_value);
    println!("Expected value: {}", value);
    println!("Error: {}", (recovered_value - value).abs());

    assert!((recovered_value - value).abs() < 1.0,
           "Encryption/decryption should recover value (got {}, expected {})", recovered_value, value);
}
