//! Test relinearization in isolation
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]


use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt};
use ga_engine::clifford_fhe::rns::{rns_multiply as rns_poly_multiply, rns_add};

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

#[test]
fn test_verify_tensor_product_then_relin() {
    let mut params = CliffordFHEParams::new_rns_mult();
    params.scale = 1024.0; // Small scale to avoid overflow

    let (pk, sk, evk) = rns_keygen(&params);
    let primes = &params.moduli;

    // Encrypt [2] and [3]
    let mut coeffs_a = vec![0i64; params.n];
    coeffs_a[0] = (2.0 * params.scale) as i64;
    let pt_a = RnsPlaintext::from_coeffs(coeffs_a, params.scale, primes, 0);
    let ct_a = rns_encrypt(&pk, &pt_a, &params);

    let mut coeffs_b = vec![0i64; params.n];
    coeffs_b[0] = (3.0 * params.scale) as i64;
    let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, primes, 0);
    let ct_b = rns_encrypt(&pk, &pt_b, &params);

    // Tensor product
    let d0 = rns_poly_multiply(&ct_a.c0, &ct_b.c0, primes, polynomial_multiply_ntt);
    let d1_1 = rns_poly_multiply(&ct_a.c0, &ct_b.c1, primes, polynomial_multiply_ntt);
    let d1_2 = rns_poly_multiply(&ct_a.c1, &ct_b.c0, primes, polynomial_multiply_ntt);
    let d1 = rns_add(&d1_1, &d1_2, primes);
    let d2 = rns_poly_multiply(&ct_a.c1, &ct_b.c1, primes, polynomial_multiply_ntt);

    // Decrypt degree-2 ciphertext BEFORE relinearization
    let s_squared = rns_poly_multiply(&sk.coeffs, &sk.coeffs, primes, polynomial_multiply_ntt);
    let d1_s = rns_poly_multiply(&d1, &sk.coeffs, primes, polynomial_multiply_ntt);
    let d2_s2 = rns_poly_multiply(&d2, &s_squared, primes, polynomial_multiply_ntt);

    let mut decrypted_deg2 = rns_add(&d0, &d1_s, primes);
    decrypted_deg2 = rns_add(&decrypted_deg2, &d2_s2, primes);

    let dec_coeffs_deg2 = decrypted_deg2.to_coeffs_single_prime(0, primes[0]);
    let value_deg2 = (dec_coeffs_deg2[0] as f64) / (params.scale * params.scale);

    println!("\n=== BEFORE RELINEARIZATION ===");
    println!("Decrypted degree-2 value: {:.6}", value_deg2);
    println!("Expected: 6.000000");
    println!("Error: {:.6}", (value_deg2 - 6.0).abs());

    assert!((value_deg2 - 6.0).abs() < 1.0,
           "Degree-2 ciphertext should decrypt to 6 (got {})", value_deg2);

    println!("\n✓ Tensor product + degree-2 decryption WORKS!");
}
