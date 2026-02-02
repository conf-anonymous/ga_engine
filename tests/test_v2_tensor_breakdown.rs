//! ULTIMATE TENSOR PRODUCT BREAKDOWN
//! Run with: cargo test --test test_v2_tensor_breakdown --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

#[test]
fn test_tensor_product_breakdown() {
    println!("\n=== TENSOR PRODUCT BREAKDOWN ===");
    println!("Goal: Find EXACT step where things go wrong\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    let n = params.n;
    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let q0 = moduli[0];

    // Encrypt 2 and 3
    let pt_a = ckks_ctx.encode(&[2.0]);
    let pt_b = ckks_ctx.encode(&[3.0]);
    let ct_a = ckks_ctx.encrypt(&pt_a, &pk);
    let ct_b = ckks_ctx.encrypt(&pt_b, &pk);

    println!("Plaintext A = 2 * scale = {}", 2.0 * params.scale);
    println!("Plaintext B = 3 * scale = {}", 3.0 * params.scale);
    println!("A * B = {:.2e}", (2.0 * params.scale) * (3.0 * params.scale));
    let expected_mod = ((2.0 * params.scale) * (3.0 * params.scale)) as u128 % q0 as u128;
    println!("(A * B) mod q0 = {}\n", expected_mod);

    // Verify decryption works
    let dec_a = ckks_ctx.decrypt(&ct_a, &sk);
    let val_a = decode_first(&dec_a, ct_a.scale);
    let dec_b = ckks_ctx.decrypt(&ct_b, &sk);
    let val_b = decode_first(&dec_b, ct_b.scale);
    println!("Decrypt(ct_a) = {:.10} (expected 2.0)", val_a);
    println!("Decrypt(ct_b) = {:.10} (expected 3.0)\n", val_b);

    // Get secret key
    let sk_at_level: Vec<RnsRepresentation> = sk.coeffs.iter()
        .map(|rns| {
            let values = rns.values[..=level].to_vec();
            let moduli_level = rns.moduli[..=level].to_vec();
            RnsRepresentation::new(values, moduli_level)
        })
        .collect();

    println!("=== STEP 1: Verify c0_a + c1_a*s = m_a ===");
    let c1a_s = multiply_polys(&ct_a.c1, &sk_at_level, n, &moduli, &key_ctx);
    let m_a_reconstructed: Vec<_> = ct_a.c0.iter().zip(&c1a_s).map(|(x, y)| x.add(y)).collect();

    println!("(c0_a + c1_a*s)[0] (centered) = {}", center(m_a_reconstructed[0].values[0], q0));
    println!("Expected m_a = {}", 2.0 * params.scale);

    let m_a_val = (center(m_a_reconstructed[0].values[0], q0) as f64) / ct_a.scale;
    println!("Decoded: {:.10}", m_a_val);
    assert!((m_a_val - 2.0).abs() < 0.01, "Step 1 failed!");
    println!("PASS\n");

    println!("=== STEP 2-4: Compute d0, d1, d2 ===");
    let d0 = multiply_polys(&ct_a.c0, &ct_b.c0, n, &moduli, &key_ctx);
    let c0a_c1b = multiply_polys(&ct_a.c0, &ct_b.c1, n, &moduli, &key_ctx);
    let c1a_c0b = multiply_polys(&ct_a.c1, &ct_b.c0, n, &moduli, &key_ctx);
    let d1: Vec<_> = c0a_c1b.iter().zip(&c1a_c0b).map(|(x, y)| x.add(y)).collect();
    let d2 = multiply_polys(&ct_a.c1, &ct_b.c1, n, &moduli, &key_ctx);

    println!("d0[0] (centered) = {:.2e}", center(d0[0].values[0], q0) as f64);
    println!("d1[0] (centered) = {:.2e}", center(d1[0].values[0], q0) as f64);
    println!("d2[0] (centered) = {:.2e}\n", center(d2[0].values[0], q0) as f64);

    println!("=== STEP 5-7: Compute d1*s, s^2, d2*s^2 ===");
    let d1_s = multiply_polys(&d1, &sk_at_level, n, &moduli, &key_ctx);
    let s_squared = multiply_polys(&sk_at_level, &sk_at_level, n, &moduli, &key_ctx);
    let d2_s2 = multiply_polys(&d2, &s_squared, n, &moduli, &key_ctx);

    println!("(d1*s)[0] (centered) = {:.2e}", center(d1_s[0].values[0], q0) as f64);
    println!("(s^2)[0] (centered) = {}", center(s_squared[0].values[0], q0));
    println!("(d2*s^2)[0] (centered) = {:.2e}\n", center(d2_s2[0].values[0], q0) as f64);

    println!("=== STEP 8: Compute m = d0 + d1*s + d2*s^2 ===");
    let temp: Vec<_> = d0.iter().zip(&d1_s).map(|(x, y)| x.add(y)).collect();
    let m: Vec<_> = temp.iter().zip(&d2_s2).map(|(x, y)| x.add(y)).collect();

    println!("d0[0]      = {:>20}", center(d0[0].values[0], q0));
    println!("(d1*s)[0]  = {:>20}", center(d1_s[0].values[0], q0));
    println!("(d2*s^2)[0] = {:>20}", center(d2_s2[0].values[0], q0));
    println!("----------------------------------------");
    println!("m[0]       = {:>20}\n", center(m[0].values[0], q0));

    let m_centered = center(m[0].values[0], q0);
    let scale_squared = ct_a.scale * ct_b.scale;
    let result = (m_centered as f64) / scale_squared;

    println!("=== FINAL RESULT ===");
    println!("m[0] (centered) = {:.2e}", m_centered as f64);
    println!("scale^2 = {:.2e}", scale_squared);
    println!("result = m / scale^2 = {:.10}", result);
    println!("\nEXPECTED: 6.0");
    println!("ACTUAL:   {:.10}", result);
    println!("ERROR:    {:.2e}\n", (result - 6.0).abs());

    // The key diagnostic
    let expected_m_centered = expected_mod as i64;
    if expected_m_centered > (q0 as i64)/2 {
        panic!("Expected value wraps - fix calculation");
    }

    println!("=== DIAGNOSTIC ===");
    println!("Expected m (mod q0) = {:.2e}", expected_m_centered as f64);
    println!("Actual m (centered) = {:.2e}", m_centered as f64);
    println!("Ratio (actual/expected) = {:.2e}", (m_centered as f64) / (expected_m_centered as f64));

    if (result - 6.0).abs() < 0.1 {
        println!("\nSUCCESS!");
    } else {
        println!("\nFAILURE - bug is in how the three terms combine!");
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

fn center(val: u64, q: u64) -> i64 {
    if val > q / 2 {
        val as i64 - q as i64
    } else {
        val as i64
    }
}

fn decode_first(pt: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext, scale: f64) -> f64 {
    let val = pt.coeffs[0].values[0];
    let q = pt.coeffs[0].moduli[0];
    let centered = center(val, q);
    (centered as f64) / scale
}
