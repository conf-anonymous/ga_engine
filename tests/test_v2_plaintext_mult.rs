//! Test plaintext polynomial multiplication
//!
//! Run with: cargo test --test test_v2_plaintext_mult --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

#[test]
fn test_plaintext_polynomial_mult() {
    println!("\n========== PLAINTEXT POLYNOMIAL MULTIPLICATION ==========");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let moduli = &params.moduli;

    println!("Testing: (2 * 2^40) * (3 * 2^40) = 6 * 2^80");

    // Create two constant polynomials
    let val_a = (2.0 * params.scale).round() as i64;  // 2 * 2^40
    let val_b = (3.0 * params.scale).round() as i64;  // 3 * 2^40

    println!("val_a = {} (2 * 2^40)", val_a);
    println!("val_b = {} (3 * 2^40)", val_b);

    // Convert to RNS
    let mut poly_a = vec![RnsRepresentation::from_u64(0, moduli); n];
    let mut poly_b = vec![RnsRepresentation::from_u64(0, moduli); n];

    // Put values in first coefficient only
    poly_a[0] = RnsRepresentation::new(
        moduli.iter().map(|&q| {
            let q_i64 = q as i64;
            ((val_a % q_i64 + q_i64) % q_i64) as u64
        }).collect(),
        moduli.to_vec()
    );

    poly_b[0] = RnsRepresentation::new(
        moduli.iter().map(|&q| {
            let q_i64 = q as i64;
            ((val_b % q_i64 + q_i64) % q_i64) as u64
        }).collect(),
        moduli.to_vec()
    );

    println!("poly_a[0] = {:?}", poly_a[0].values);
    println!("poly_b[0] = {:?}", poly_b[0].values);

    // Multiply using NTT
    let key_ctx = KeyContext::new(params.clone());
    let product = multiply_polys(&poly_a, &poly_b, n, moduli, &key_ctx);

    println!("\nResult:");
    println!("product[0] = {:?}", product[0].values);

    // Decode
    let val_prod = product[0].values[0];
    let q0 = product[0].moduli[0];
    let centered = if val_prod > q0 / 2 {
        val_prod as i64 - q0 as i64
    } else {
        val_prod as i64
    };

    println!("product[0] (centered) = {}", centered);

    // Expected
    let expected = (val_a as i128) * (val_b as i128);
    println!("\nExpected: {} * {} = {}", val_a, val_b, expected);
    println!("Actual (centered): {}", centered);

    // For comparison with modular arithmetic
    let expected_mod_q0 = (expected % (q0 as i128)) as i64;
    let expected_centered = if expected_mod_q0 > (q0/2) as i64 {
        expected_mod_q0 - q0 as i64
    } else {
        expected_mod_q0
    };

    println!("Expected (mod q0, centered): {}", expected_centered);

    if centered == expected_centered {
        println!("✓ PASS: Plaintext multiplication works correctly!");
    } else {
        println!("✗ FAIL: Plaintext multiplication is wrong!");
        println!("  Difference: {}", (centered - expected_centered).abs());
    }
}

fn multiply_polys(
    a: &[RnsRepresentation],
    b: &[RnsRepresentation],
    n: usize,
    moduli: &[u64],
    key_ctx: &KeyContext,
) -> Vec<RnsRepresentation> {
    let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.to_vec()); n];

    for (prime_idx, &q) in moduli.iter().enumerate() {
        let ntt_ctx = key_ctx.ntt_contexts.iter().find(|ctx| ctx.q == q).unwrap();

        let a_mod_q: Vec<u64> = a.iter().map(|rns| rns.values[prime_idx]).collect();
        let b_mod_q: Vec<u64> = b.iter().map(|rns| rns.values[prime_idx]).collect();

        let product_mod_q = ntt_ctx.multiply_polynomials(&a_mod_q, &b_mod_q);

        for i in 0..n {
            result[i].values[prime_idx] = product_mod_q[i];
        }
    }

    result
}
