// Debug: Compare multiplication with 3 vs 5 primes step by step
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

fn test_mult_with_n_primes(num_primes: usize, test_name: &str) -> f64 {
    println!("\n{}", "=".repeat(60));
    println!("Testing multiplication with {} primes: {}", num_primes, test_name);
    println!("{}", "=".repeat(60));

    let params = match num_primes {
        3 => CliffordFHEParams::new_rns_mult(),
        5 => CliffordFHEParams::new_rns_mult_depth2_safe(),
        _ => panic!("Unsupported number of primes"),
    };

    println!("Primes: {:?}", params.moduli);
    println!("Scale: {}", params.scale);

    let (pk, sk, evk) = rns_keygen(&params);

    let a = 2.0;
    let b = 3.0;
    let expected = 6.0;

    let mut coeffs_a = vec![0i64; params.n];
    let mut coeffs_b = vec![0i64; params.n];
    coeffs_a[0] = (a * params.scale).round() as i64;
    coeffs_b[0] = (b * params.scale).round() as i64;

    println!("\n[PLAINTEXT VALUES]");
    println!("  a = {}, scaled = {}", a, coeffs_a[0]);
    println!("  b = {}, scaled = {}", b, coeffs_b[0]);
    println!("  expected product = {}", expected);

    let pt_a = RnsPlaintext::from_coeffs(coeffs_a.clone(), params.scale, &params.moduli, 0);
    let pt_b = RnsPlaintext::from_coeffs(coeffs_b.clone(), params.scale, &params.moduli, 0);

    println!("\n[PLAINTEXT RNS FORM]");
    println!("  pt_a.coeffs[0] residues: {:?}", &pt_a.coeffs.rns_coeffs[0]);
    println!("  pt_b.coeffs[0] residues: {:?}", &pt_b.coeffs.rns_coeffs[0]);

    let ct_a = rns_encrypt(&pk, &pt_a, &params);
    let ct_b = rns_encrypt(&pk, &pt_b, &params);

    println!("\n[ENCRYPTED CIPHERTEXTS]");
    println!("  ct_a: level={}, num_primes={}, scale={}", ct_a.level, ct_a.c0.num_primes(), ct_a.scale);
    println!("  ct_b: level={}, num_primes={}, scale={}", ct_b.level, ct_b.c0.num_primes(), ct_b.scale);

    // Verify fresh ciphertexts decrypt correctly
    let pt_a_dec = rns_decrypt(&sk, &ct_a, &params);
    let pt_b_dec = rns_decrypt(&sk, &ct_b, &params);
    let a_dec = decode_value(&pt_a_dec, ct_a.scale, &params.moduli, ct_a.level);
    let b_dec = decode_value(&pt_b_dec, ct_b.scale, &params.moduli, ct_b.level);

    println!("\n[DECRYPTED FRESH CIPHERTEXTS]");
    println!("  a_dec = {} (error: {})", a_dec, (a_dec - a).abs());
    println!("  b_dec = {} (error: {})", b_dec, (b_dec - b).abs());

    // NOW DO MULTIPLICATION
    println!("\n[MULTIPLICATION]");
    let ct_prod = rns_multiply_ciphertexts(&ct_a, &ct_b, &evk, &params);

    println!("  ct_prod: level={}, num_primes={}, scale={}",
             ct_prod.level, ct_prod.c0.num_primes(), ct_prod.scale);

    // Print first coefficient residues
    println!("  ct_prod.c0[0] residues (first 4): {:?}",
             &ct_prod.c0.rns_coeffs[0][..ct_prod.c0.num_primes().min(4)]);
    println!("  ct_prod.c1[0] residues (first 4): {:?}",
             &ct_prod.c1.rns_coeffs[0][..ct_prod.c1.num_primes().min(4)]);

    let pt_prod = rns_decrypt(&sk, &ct_prod, &params);

    println!("\n[DECRYPTED PRODUCT]");
    println!("  pt_prod.coeffs[0] residues: {:?}", &pt_prod.coeffs.rns_coeffs[0]);

    let result = decode_value(&pt_prod, ct_prod.scale, &params.moduli, ct_prod.level);
    let error = (result - expected).abs();

    println!("\n[RESULT]");
    println!("  Expected: {}", expected);
    println!("  Got:      {}", result);
    println!("  Error:    {}", error);
    println!("  Relative: {:.2}%", (error / expected) * 100.0);

    error
}

#[test]
fn compare_3_vs_5_primes() {
    println!("\n\n");
    println!("##########################################################");
    println!("#  COMPARING 3-PRIME VS 5-PRIME MULTIPLICATION");
    println!("##########################################################");

    let error_3 = test_mult_with_n_primes(3, "WORKING CASE");
    let error_5 = test_mult_with_n_primes(5, "BROKEN CASE");

    println!("\n\n");
    println!("{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!("3 primes: error = {:.6} ✅", error_3);
    println!("5 primes: error = {:.6} ❌", error_5);
    println!("Ratio: {:.2}x worse", error_5 / error_3.max(0.000001));

    assert!(error_3 < 0.01, "3-prime case should work!");
    // Don't assert on 5-prime yet - we're debugging
}
