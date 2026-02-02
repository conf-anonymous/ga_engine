//! Manually decrypt multiplication result to see raw values
//! Run with: cargo test --test test_v2_mult_manual_decrypt --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::multiplication::multiply_ciphertexts;

#[test]
fn test_mult_manual_decrypt() {
    println!("\n=== MANUAL DECRYPT TEST ===\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Encrypt 2.0 and 3.0
    let pt_a = ckks_ctx.encode(&[2.0]);
    let ct_a = ckks_ctx.encrypt(&pt_a, &pk);
    let pt_b = ckks_ctx.encode(&[3.0]);
    let ct_b = ckks_ctx.encrypt(&pt_b, &pk);

    println!("Encrypted 2.0 and 3.0");
    println!("ct_a.scale = {:.2e}", ct_a.scale);
    println!("ct_b.scale = {:.2e}", ct_b.scale);

    // Multiply
    let ct_prod = multiply_ciphertexts(&ct_a, &ct_b, &evk, &key_ctx);

    println!("\nAfter multiplication:");
    println!("ct_prod.level = {}", ct_prod.level);
    println!("ct_prod.scale = {:.2e}", ct_prod.scale);
    println!("ct_prod.c0[0]: {:?}", ct_prod.c0[0].values);
    println!("ct_prod.c1[0]: {:?}", ct_prod.c1[0].values);

    // Manual decrypt: m' = c0 + c1*s
    let moduli = &ct_prod.c0[0].moduli;
    println!("\nManual decryption (using moduli at result level):");
    println!("Moduli: {:?}", moduli);

    // Get secret key at the correct level
    let level = ct_prod.level;
    let sk_at_level: Vec<_> = sk.coeffs.iter()
        .map(|rns| {
            let vals = rns.values[..=level].to_vec();
            let mods = rns.moduli[..=level].to_vec();
            ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation::new(vals, mods)
        })
        .collect();

    // Multiply c1 * s using NTT
    let c1_times_s = multiply_polys(&ct_prod.c1, &sk_at_level, &key_ctx, moduli);

    // Add c0 + c1*s
    let m_prime_0 = ct_prod.c0[0].add(&c1_times_s[0]);

    println!("\nDecryption formula: m' = c0[0] + (c1*s)[0]");
    println!("c0[0]: {:?}", ct_prod.c0[0].values);
    println!("(c1*s)[0]: {:?}", c1_times_s[0].values);
    println!("m'[0] = c0[0] + (c1*s)[0]: {:?}", m_prime_0.values);

    // Center and decode
    let q0 = moduli[0];
    let val_raw = m_prime_0.values[0];
    let val_centered = if val_raw > q0 / 2 { val_raw as i64 - q0 as i64 } else { val_raw as i64 };
    let result = (val_centered as f64) / ct_prod.scale;

    println!("\nFinal decoding:");
    println!("val_raw = {}", val_raw);
    println!("val_centered = {}", val_centered);
    println!("scale = {:.2e}", ct_prod.scale);
    println!("result = val_centered / scale = {:.10}", result);
    println!("\nExpected: 6.0");
    println!("Error: {:.2e}", (result - 6.0).abs());

    if (result - 6.0).abs() < 0.1 {
        println!("\n✓ SUCCESS!");
    } else {
        println!("\n✗ FAILED!");
    }
}

fn multiply_polys(
    a: &[ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation],
    b: &[ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation],
    key_ctx: &KeyContext,
    moduli: &[u64],
) -> Vec<ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation> {
    let n = a.len();
    let num_primes = moduli.len();
    let mut result = vec![ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation::new(vec![0; num_primes], moduli.to_vec()); n];

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
