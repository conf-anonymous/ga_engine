//! Test to check if relinearization is working correctly
//!
//! Run with: MULT_DEBUG=1 cargo test --test test_v2_relin_debug --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::multiplication::multiply_ciphertexts;

#[test]
fn test_check_relin_before_rescale() {
    println!("\n========== RELINEARIZATION DEBUG TEST ==========");

    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  Primes: {:?}", params.moduli);
    println!("  Scale = {:.2e}", params.scale);

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Simple test: multiply 2 * 3 = 6
    let a_val = 2.0;
    let b_val = 3.0;
    println!("\nEncrypting {} and {}", a_val, b_val);

    let ct_a = encrypt_value(a_val, &ckks_ctx, &pk);
    let ct_b = encrypt_value(b_val, &ckks_ctx, &pk);

    // Verify encryption
    let pt_a = ckks_ctx.decrypt(&ct_a, &sk);
    let dec_a = decode_first_coeff(&pt_a, ct_a.scale);
    println!("Decrypted ct_a: {} (expected {})", dec_a, a_val);

    let pt_b = ckks_ctx.decrypt(&ct_b, &sk);
    let dec_b = decode_first_coeff(&pt_b, ct_b.scale);
    println!("Decrypted ct_b: {} (expected {})", dec_b, b_val);

    // We need to access intermediate values, but multiply_ciphertexts doesn't expose them.
    // So let's manually do the tensor product and relinearization.

    println!("\n--- Manual Tensor Product ---");
    let level = ct_a.level;
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let n = params.n;

    // d0 = c0_a * c0_b
    let d0 = multiply_polys(&ct_a.c0, &ct_b.c0, n, &moduli, &key_ctx);
    println!("d0[0]: {:?}", d0[0].values);

    // d1 = c0_a * c1_b + c1_a * c0_b
    let c0a_c1b = multiply_polys(&ct_a.c0, &ct_b.c1, n, &moduli, &key_ctx);
    let c1a_c0b = multiply_polys(&ct_a.c1, &ct_b.c0, n, &moduli, &key_ctx);
    let d1: Vec<_> = c0a_c1b.iter().zip(&c1a_c0b).map(|(x, y)| x.add(y)).collect();
    println!("d1[0]: {:?}", d1[0].values);

    // d2 = c1_a * c1_b
    let d2 = multiply_polys(&ct_a.c1, &ct_b.c1, n, &moduli, &key_ctx);
    println!("d2[0]: {:?}", d2[0].values);

    // Now manually decrypt the degree-2 ciphertext: m = d0 + d1*s + d2*s^2
    println!("\n--- Decrypting Degree-2 Ciphertext (Before Relin) ---");

    // Extract sk at correct level
    let sk_at_level: Vec<RnsRepresentation> = sk.coeffs.iter()
        .map(|rns| {
            let values = rns.values[..=level].to_vec();
            let moduli_level = rns.moduli[..=level].to_vec();
            RnsRepresentation::new(values, moduli_level)
        })
        .collect();

    println!("Secret key (first 5 coefficients):");
    for i in 0..5.min(sk_at_level.len()) {
        println!("  sk[{}]: {:?}", i, sk_at_level[i].values);
    }

    let d1_s = multiply_polys(&d1, &sk_at_level, n, &moduli, &key_ctx);
    let s_squared = multiply_polys(&sk_at_level, &sk_at_level, n, &moduli, &key_ctx);
    let d2_s2 = multiply_polys(&d2, &s_squared, n, &moduli, &key_ctx);

    let temp: Vec<_> = d0.iter().zip(&d1_s).map(|(x, y)| x.add(y)).collect();
    let m_deg2: Vec<_> = temp.iter().zip(&d2_s2).map(|(x, y)| x.add(y)).collect();

    println!("m_deg2[0]: {:?}", m_deg2[0].values);

    let val = m_deg2[0].values[0];
    let q = m_deg2[0].moduli[0];
    let centered = if val > q / 2 { val as i64 - q as i64 } else { val as i64 };
    let scale_squared = ct_a.scale * ct_b.scale;
    let result_deg2 = (centered as f64) / scale_squared;

    println!("Decrypted (degree-2, no relin): {}", result_deg2);
    println!("Expected: {}", a_val * b_val);
    println!("Error: {:.2e}", (result_deg2 - a_val * b_val).abs());

    // Now use multiply_ciphertexts which does relin+rescale
    println!("\n--- Using multiply_ciphertexts (with relin+rescale) ---");
    let ct_result = multiply_ciphertexts(&ct_a, &ct_b, &evk, &key_ctx);

    let pt_result = ckks_ctx.decrypt(&ct_result, &sk);
    let dec_result = decode_first_coeff(&pt_result, ct_result.scale);

    println!("Decrypted (after relin+rescale): {}", dec_result);
    println!("Expected: {}", a_val * b_val);
    println!("Error: {:.2e}", (dec_result - a_val * b_val).abs());
}

fn encrypt_value(
    value: f64,
    ckks_ctx: &CkksContext,
    pk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey,
) -> ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext {
    // Use the proper encode function!
    let pt = ckks_ctx.encode(&[value]);

    if std::env::var("ENCODE_DEBUG").is_ok() {
        println!("[ENCODE] value={}, scale={:.2e}", value, pt.scale);
        println!("  pt.coeffs[0]: {:?}", pt.coeffs[0].values.iter().take(3).collect::<Vec<_>>());
        println!("  pt.coeffs[1]: {:?}", pt.coeffs[1].values.iter().take(3).collect::<Vec<_>>());
    }

    ckks_ctx.encrypt(&pt, pk)
}

fn decode_first_coeff(pt: &Plaintext, scale: f64) -> f64 {
    let val = pt.coeffs[0].values[0];
    let q = pt.coeffs[0].moduli[0];
    let centered = if val > q / 2 { val as i64 - q as i64 } else { val as i64 };
    (centered as f64) / scale
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
