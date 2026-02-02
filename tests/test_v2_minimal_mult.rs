//! Ultra-minimal multiplication test - divide and conquer
//!
//! Run with: cargo test --test test_v2_minimal_mult --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;

#[test]
fn test_minimal_multiply_step_by_step() {
    println!("\n========== MINIMAL MULTIPLICATION TEST ==========");
    println!("Goal: Encrypt 2 and 3, multiply to get 6");

    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("\nParameters:");
    println!("  N = {}", params.n);
    println!("  Primes: {:?}", params.moduli);
    println!("  Scale = {:.2e} = 2^40", params.scale);

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Step 1: Encrypt two constants
    println!("\n========== STEP 1: ENCRYPT ==========");
    let pt_a = ckks_ctx.encode(&[2.0]);
    let pt_b = ckks_ctx.encode(&[3.0]);

    println!("Plaintext A (value=2):");
    println!("  pt_a.coeffs[0] = {:?}", pt_a.coeffs[0].values);
    println!("  Expected: 2 * scale = 2 * 2^40 = {}", 2.0 * params.scale);

    println!("Plaintext B (value=3):");
    println!("  pt_b.coeffs[0] = {:?}", pt_b.coeffs[0].values);
    println!("  Expected: 3 * scale = 3 * 2^40 = {}", 3.0 * params.scale);

    let ct_a = ckks_ctx.encrypt(&pt_a, &pk);
    let ct_b = ckks_ctx.encrypt(&pt_b, &pk);

    // Get common variables for decryption
    let n = ct_a.c0.len();
    let level = ct_a.level;
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();

    // Verify encryption works - MANUAL DECRYPTION
    println!("\nManual decryption of ct_a:");
    let sk_at_level_enc: Vec<RnsRepresentation> = sk.coeffs.iter()
        .map(|rns| {
            let values = rns.values[..=level].to_vec();
            let moduli_level = rns.moduli[..=level].to_vec();
            RnsRepresentation::new(values, moduli_level)
        })
        .collect();

    // m = c0 + c1*s
    let c1_s_a = multiply_polys(&ct_a.c1, &sk_at_level_enc, n, &moduli, &key_ctx);
    let m_a: Vec<_> = ct_a.c0.iter().zip(&c1_s_a).map(|(x, y)| x.add(y)).collect();

    let val_m_a = m_a[0].values[0];
    let q0 = m_a[0].moduli[0];
    let centered_a = if val_m_a > q0 / 2 {
        val_m_a as i64 - q0 as i64
    } else {
        val_m_a as i64
    };
    let decoded_a = (centered_a as f64) / ct_a.scale;

    println!("  m[0] (raw) = {}", val_m_a);
    println!("  m[0] (centered) = {}", centered_a);
    println!("  decoded = {} (expected 2.0)", decoded_a);

    // Now use built-in decrypt for comparison
    let dec_a = ckks_ctx.decrypt(&ct_a, &sk);
    let val_a = decode_first(&dec_a, ct_a.scale);
    println!("  Built-in decrypt = {}", val_a);

    assert!((decoded_a - 2.0).abs() < 1e-6, "Manual decryption of 2.0 failed!");
    assert!((val_a - 2.0).abs() < 1e-6, "Built-in decryption of 2.0 failed!");
    println!("  ✓ Both manual and built-in decryption work!");

    // Step 1.5: Analyze ciphertext components
    println!("\n========== STEP 1.5: ANALYZE CIPHERTEXT COMPONENTS ==========");
    println!("Let's see what c0_a[0] and c1_a[0] actually are:");

    println!("\nct_a.c0[0] = {:?}", ct_a.c0[0].values);
    let c0a_centered = if ct_a.c0[0].values[0] > moduli[0]/2 {
        ct_a.c0[0].values[0] as i64 - moduli[0] as i64
    } else {
        ct_a.c0[0].values[0] as i64
    };
    println!("  centered: {:.2e}", c0a_centered as f64);

    println!("\nct_a.c1[0] = {:?}", ct_a.c1[0].values);
    let c1a_centered = if ct_a.c1[0].values[0] > moduli[0]/2 {
        ct_a.c1[0].values[0] as i64 - moduli[0] as i64
    } else {
        ct_a.c1[0].values[0] as i64
    };
    println!("  centered: {:.2e}", c1a_centered as f64);

    // Step 2: Tensor product (WITHOUT relin, WITHOUT rescale)
    println!("\n========== STEP 2: TENSOR PRODUCT ==========");
    println!("Computing (c0_a, c1_a) ⊗ (c0_b, c1_b) = (d0, d1, d2)");
    println!("Formula:");
    println!("  d0 = c0_a * c0_b");
    println!("  d1 = c0_a * c1_b + c1_a * c0_b  ");
    println!("  d2 = c1_a * c1_b");

    // d0 = c0_a * c0_b
    let d0 = multiply_polys(&ct_a.c0, &ct_b.c0, n, &moduli, &key_ctx);
    println!("\nd0 = c0_a * c0_b:");
    println!("  d0[0] = {:?}", d0[0].values);

    // d1 = c0_a * c1_b + c1_a * c0_b
    let c0a_c1b = multiply_polys(&ct_a.c0, &ct_b.c1, n, &moduli, &key_ctx);
    let c1a_c0b = multiply_polys(&ct_a.c1, &ct_b.c0, n, &moduli, &key_ctx);
    let d1: Vec<_> = c0a_c1b.iter().zip(&c1a_c0b).map(|(x, y)| x.add(y)).collect();
    println!("\nd1 = c0_a*c1_b + c1_a*c0_b:");
    println!("  d1[0] = {:?}", d1[0].values);

    // d2 = c1_a * c1_b
    let d2 = multiply_polys(&ct_a.c1, &ct_b.c1, n, &moduli, &key_ctx);
    println!("\nd2 = c1_a * c1_b:");
    println!("  d2[0] = {:?}", d2[0].values);

    // Step 3: Decrypt degree-2 ciphertext
    println!("\n========== STEP 3: DECRYPT DEGREE-2 ==========");
    println!("Computing m = d0 + d1*s + d2*s²");

    // Get secret key at correct level
    let sk_at_level: Vec<RnsRepresentation> = sk.coeffs.iter()
        .map(|rns| {
            let values = rns.values[..=level].to_vec();
            let moduli_level = rns.moduli[..=level].to_vec();
            RnsRepresentation::new(values, moduli_level)
        })
        .collect();

    // Compute d1 * s
    let d1_s = multiply_polys(&d1, &sk_at_level, n, &moduli, &key_ctx);
    println!("\nd1 * s:");
    println!("  (d1*s)[0] = {:?}", d1_s[0].values);

    // First, let's see what the secret key looks like
    println!("\nSecret key (first 10 coefficients):");
    for i in 0..10.min(n) {
        let val0 = sk_at_level[i].values[0];
        let q0 = sk_at_level[i].moduli[0];
        // Convert to centered representation (-q/2, q/2]
        let centered = if val0 > q0 / 2 {
            val0 as i64 - q0 as i64
        } else {
            val0 as i64
        };
        if centered != 0 {  // Only print non-zero coefficients
            println!("  sk[{}] = {} (mod q0 representation: {})", i, centered, val0);
        }
    }

    // Compute s²
    let s_squared = multiply_polys(&sk_at_level, &sk_at_level, n, &moduli, &key_ctx);
    println!("\ns²:");
    println!("  (s²)[0] = {:?}", s_squared[0].values);

    // Also decode s²[0]
    let s2_val0 = s_squared[0].values[0];
    let s2_centered = if s2_val0 > moduli[0] / 2 {
        s2_val0 as i64 - moduli[0] as i64
    } else {
        s2_val0 as i64
    };
    println!("  (s²)[0] centered = {}", s2_centered);

    // Compute d2 * s²
    let d2_s2 = multiply_polys(&d2, &s_squared, n, &moduli, &key_ctx);
    println!("\nd2 * s²:");
    println!("  (d2*s²)[0] = {:?}", d2_s2[0].values);

    // m = d0 + d1*s + d2*s²
    let temp: Vec<_> = d0.iter().zip(&d1_s).map(|(x, y)| x.add(y)).collect();
    let m: Vec<_> = temp.iter().zip(&d2_s2).map(|(x, y)| x.add(y)).collect();

    println!("\nm = d0 + d1*s + d2*s²:");
    println!("  d0[0] = {:?}", d0[0].values);
    println!("  (d1*s)[0] = {:?}", d1_s[0].values);
    println!("  (d2*s²)[0] = {:?}", d2_s2[0].values);
    println!("  m[0] = d0[0] + (d1*s)[0] + (d2*s²)[0]");
    println!("  m[0] = {:?}", m[0].values);

    // Decode
    let val_deg2 = m[0].values[0];
    let q = m[0].moduli[0];
    let centered = if val_deg2 > q / 2 {
        val_deg2 as i64 - q as i64
    } else {
        val_deg2 as i64
    };

    let scale_squared = ct_a.scale * ct_b.scale;
    let result_deg2 = (centered as f64) / scale_squared;

    println!("\nDecoded result (degree-2, no relin/rescale):");
    println!("  centered value: {}", centered);
    println!("  scale² = {:.2e}", scale_squared);
    println!("  result = centered / scale² = {}", result_deg2);
    println!("  EXPECTED: 2 * 3 = 6.0");
    println!("  ERROR: {:.2e}", (result_deg2 - 6.0).abs());

    // Let's also compute what the theoretical value SHOULD be
    println!("\n  THEORY CHECK:");
    println!("    Plaintext A = 2 * scale = {}", 2.0 * params.scale);
    println!("    Plaintext B = 3 * scale = {}", 3.0 * params.scale);
    println!("    A * B = {:.2e}", (2.0 * params.scale) * (3.0 * params.scale));
    println!("    Expected m[0] ≈ A*B (ignoring noise) = {:.2e}", 6.0 * scale_squared);
    println!("    Actual m[0] (centered) = {:.2e}", centered as f64);
    println!("    Ratio = {:.2e}", centered as f64 / (6.0 * scale_squared));

    if (result_deg2 - 6.0).abs() < 0.1 {
        println!("  ✓ PASS: Degree-2 multiplication works!");
    } else {
        println!("  ✗ FAIL: Degree-2 multiplication is broken!");
        println!("\n  DIAGNOSIS:");
        println!("    Expected centered value: {:.2e}", 6.0 * scale_squared);
        println!("    Actual centered value: {:.2e}", centered as f64);
        println!("    Ratio: {:.2e}", (6.0 * scale_squared) / (centered as f64));
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

fn decode_first(pt: &Plaintext, scale: f64) -> f64 {
    let val = pt.coeffs[0].values[0];
    let q = pt.coeffs[0].moduli[0];
    let centered = if val > q / 2 { val as i64 - q as i64 } else { val as i64 };
    (centered as f64) / scale
}
