//! Check if V2 encryption produces reasonable values
//! Run with: cargo test --test test_v2_encryption_values --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;

#[test]
fn test_encryption_values() {
    println!("\n=== V2 ENCRYPTION VALUES TEST ===\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  Scale = {:.2e}", params.scale);
    println!("  Moduli: {:?}", params.moduli);

    // Encrypt 2.0
    let pt_a = ckks_ctx.encode(&[2.0]);
    println!("\nPlaintext for 2.0:");
    println!("  pt_a.coeffs[0]: {:?}", pt_a.coeffs[0].values);

    let ct_a = ckks_ctx.encrypt(&pt_a, &pk);
    println!("\nCiphertext for 2.0:");
    println!("  ct_a.c0[0]: {:?}", ct_a.c0[0].values);
    println!("  ct_a.c1[0]: {:?}", ct_a.c1[0].values);
    println!("  ct_a.level = {}", ct_a.level);
    println!("  ct_a.scale = {:.2e}", ct_a.scale);

    // Decrypt to verify
    let dec_a = ckks_ctx.decrypt(&ct_a, &sk);
    let val_a = decode(&dec_a, ct_a.scale);
    println!("\nDecrypt(ct_a) = {:.10} (expected 2.0)", val_a);
    println!("Error: {:.2e}", (val_a - 2.0).abs());

    // Encrypt 3.0
    let pt_b = ckks_ctx.encode(&[3.0]);
    let ct_b = ckks_ctx.encrypt(&pt_b, &pk);

    println!("\nCiphertext for 3.0:");
    println!("  ct_b.c0[0]: {:?}", ct_b.c0[0].values);
    println!("  ct_b.c1[0]: {:?}", ct_b.c1[0].values);

    let dec_b = ckks_ctx.decrypt(&ct_b, &sk);
    let val_b = decode(&dec_b, ct_b.scale);
    println!("\nDecrypt(ct_b) = {:.10} (expected 3.0)", val_b);
    println!("Error: {:.2e}", (val_b - 3.0).abs());

    // Now let's manually compute what d0 SHOULD be
    // d0 = c0_a * c0_b (polynomial multiplication mod x^n + 1)
    println!("\n=== MANUAL TENSOR PRODUCT CHECK ===");
    println!("Computing d0 = c0_a * c0_b using NTT...");

    let moduli: Vec<u64> = params.moduli[..=params.max_level()].to_vec();
    let d0 = mult_polys(&ct_a.c0, &ct_b.c0, &key_ctx, &moduli);

    println!("d0[0]: {:?}", d0[0].values);

    // What value SHOULD d0[0] have?
    // If Enc(m) = (c0, c1) where c0 ≈ m + noise, c1 ≈ noise
    // Then d0 = c0_a * c0_b ≈ (m_a + e_a) * (m_b + e_b) = m_a*m_b + cross terms
    //         ≈ (2Δ) * (3Δ) = 6Δ² mod q
    let expected_d0 = 2.0 * params.scale * 3.0 * params.scale;
    println!("\nExpected d0[0] ≈ 2Δ * 3Δ = {:.2e}", expected_d0);

    // Check if d0[0] mod q0 is close to expected
    let q0 = moduli[0];
    let d0_val = d0[0].values[0];
    let d0_centered = if d0_val > q0 / 2 { d0_val as i64 - q0 as i64 } else { d0_val as i64 };

    println!("Actual d0[0] (centered) = {:.2e}", d0_centered as f64);
    println!("Ratio: actual / expected = {:.2}", d0_centered as f64 / expected_d0);

    if ((d0_centered as f64 / expected_d0) - 1.0).abs() < 0.1 {
        println!("\n✓ Tensor product d0 looks reasonable!");
    } else {
        println!("\n✗ Tensor product d0 is WAY off!");
    }
}

fn mult_polys(
    a: &[ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation],
    b: &[ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation],
    key_ctx: &KeyContext,
    moduli: &[u64],
) -> Vec<ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation> {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

    let n = a.len();
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

fn decode(pt: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext, scale: f64) -> f64 {
    let val = pt.coeffs[0].values[0];
    let q = pt.coeffs[0].moduli[0];
    let centered = if val > q / 2 { val as i64 - q as i64 } else { val as i64 };
    (centered as f64) / scale
}
