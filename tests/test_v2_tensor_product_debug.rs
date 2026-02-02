//! Minimal test to isolate tensor product bug
//!
//! Run with: cargo test --test test_v2_tensor_product_debug --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;

#[test]
fn test_manual_tensor_product() {
    println!("\n========== MANUAL TENSOR PRODUCT TEST ==========");

    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  Primes: {:?}", params.moduli);
    println!("  Scale = {:.2e}", params.scale);

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Encrypt two simple values
    let a_val = 3.0;
    let b_val = 4.0;
    println!("\nEncrypting {} and {}", a_val, b_val);

    let ct_a = encrypt_value(a_val, &ckks_ctx, &pk);
    let ct_b = encrypt_value(b_val, &ckks_ctx, &pk);

    println!("\nCiphertext A:");
    println!("  level = {}, scale = {:.2e}", ct_a.level, ct_a.scale);
    println!("  c0[0] (first coeff, all primes): {:?}",
             ct_a.c0[0].values.iter().map(|&v| v).collect::<Vec<_>>());
    println!("  c1[0] (first coeff, all primes): {:?}",
             ct_a.c1[0].values.iter().map(|&v| v).collect::<Vec<_>>());

    println!("\nCiphertext B:");
    println!("  level = {}, scale = {:.2e}", ct_b.level, ct_b.scale);
    println!("  c0[0]: {:?}", ct_b.c0[0].values.iter().map(|&v| v).collect::<Vec<_>>());
    println!("  c1[0]: {:?}", ct_b.c1[0].values.iter().map(|&v| v).collect::<Vec<_>>());

    // Manually compute tensor product: (a0 + a1*s) * (b0 + b1*s) = a0*b0 + (a0*b1 + a1*b0)*s + a1*b1*s^2
    let level = ct_a.level;
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let n = params.n;

    println!("\n--- Computing d0 = a0 * b0 ---");
    let d0 = multiply_polys_debug(&ct_a.c0, &ct_b.c0, n, &moduli);
    println!("  d0[0]: {:?}", d0[0].values.iter().map(|&v| v).collect::<Vec<_>>());

    println!("\n--- Computing a0 * b1 ---");
    let a0_b1 = multiply_polys_debug(&ct_a.c0, &ct_b.c1, n, &moduli);
    println!("  a0*b1[0]: {:?}", a0_b1[0].values.iter().map(|&v| v).collect::<Vec<_>>());

    println!("\n--- Computing a1 * b0 ---");
    let a1_b0 = multiply_polys_debug(&ct_a.c1, &ct_b.c0, n, &moduli);
    println!("  a1*b0[0]: {:?}", a1_b0[0].values.iter().map(|&v| v).collect::<Vec<_>>());

    println!("\n--- Computing d1 = a0*b1 + a1*b0 ---");
    let d1: Vec<_> = a0_b1.iter().zip(&a1_b0).map(|(x, y)| x.add(y)).collect();
    println!("  d1[0]: {:?}", d1[0].values.iter().map(|&v| v).collect::<Vec<_>>());

    println!("\n--- Computing d2 = a1 * b1 ---");
    let d2 = multiply_polys_debug(&ct_a.c1, &ct_b.c1, n, &moduli);
    println!("  d2[0]: {:?}", d2[0].values.iter().map(|&v| v).collect::<Vec<_>>());

    // Now decrypt the degree-2 ciphertext: m = d0 + d1*s + d2*s^2
    println!("\n--- Decrypting degree-2 ciphertext ---");

    // Extract sk at correct level
    let sk_at_level: Vec<RnsRepresentation> = sk.coeffs.iter()
        .map(|rns| {
            let values = rns.values[..=level].to_vec();
            let moduli_level = rns.moduli[..=level].to_vec();
            RnsRepresentation::new(values, moduli_level)
        })
        .collect();

    println!("  sk[0]: {:?}", sk_at_level[0].values.iter().map(|&v| v).collect::<Vec<_>>());

    // Compute d1*s
    println!("\n  Computing d1*s...");
    let d1_s = multiply_polys_debug(&d1, &sk_at_level, n, &moduli);
    println!("    d1*s[0]: {:?}", d1_s[0].values.iter().map(|&v| v).collect::<Vec<_>>());

    // Compute s^2
    println!("\n  Computing s^2...");
    let s_squared = multiply_polys_debug(&sk_at_level, &sk_at_level, n, &moduli);
    println!("    s^2[0]: {:?}", s_squared[0].values.iter().map(|&v| v).collect::<Vec<_>>());

    // Compute d2*s^2
    println!("\n  Computing d2*s^2...");
    let d2_s2 = multiply_polys_debug(&d2, &s_squared, n, &moduli);
    println!("    d2*s^2[0]: {:?}", d2_s2[0].values.iter().map(|&v| v).collect::<Vec<_>>());

    // m = d0 + d1*s + d2*s^2
    println!("\n  Computing m = d0 + d1*s + d2*s^2...");
    let temp: Vec<_> = d0.iter().zip(&d1_s).map(|(x, y)| x.add(y)).collect();
    let m: Vec<_> = temp.iter().zip(&d2_s2).map(|(x, y)| x.add(y)).collect();

    println!("    m[0]: {:?}", m[0].values.iter().map(|&v| v).collect::<Vec<_>>());

    // Decode
    let val = m[0].values[0] as i64;
    let q = m[0].moduli[0] as i64;
    let centered = if val > q / 2 { val - q } else { val };

    let scale = ct_a.scale * ct_b.scale;
    let result = (centered as f64) / scale;

    println!("\n  Raw value (mod q): {}", val);
    println!("  Centered value: {}", centered);
    println!("  Scale: {:.2e}", scale);
    println!("  Decoded result: {}", result);
    println!("  Expected: 12.0");
    println!("  Error: {:.2e}", (result - 12.0).abs());
}

fn encrypt_value(
    value: f64,
    ckks_ctx: &CkksContext,
    pk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey,
) -> ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext {
    let params = &ckks_ctx.params;
    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let n = params.n;

    let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); n];

    let scaled_val = (value * params.scale).round() as i64;
    let values: Vec<u64> = moduli.iter().map(|&q| {
        let q_i64 = q as i64;
        let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
        normalized as u64
    }).collect();

    coeffs[0] = RnsRepresentation::new(values, moduli.clone());

    let pt = Plaintext::new(coeffs, params.scale, level);
    ckks_ctx.encrypt(&pt, pk)
}

fn multiply_polys_debug(
    a: &[RnsRepresentation],
    b: &[RnsRepresentation],
    n: usize,
    moduli: &[u64],
) -> Vec<RnsRepresentation> {
    let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.to_vec()); n];

    for (prime_idx, &q) in moduli.iter().enumerate() {
        let ntt_ctx = NttContext::new(n, q);

        let a_mod_q: Vec<u64> = a.iter().map(|rns| rns.values[prime_idx]).collect();
        let b_mod_q: Vec<u64> = b.iter().map(|rns| rns.values[prime_idx]).collect();

        if prime_idx == 0 {
            println!("    Prime {}: q = {}", prime_idx, q);
            println!("      a[0] mod q = {}", a_mod_q[0]);
            println!("      b[0] mod q = {}", b_mod_q[0]);
        }

        let product_mod_q = ntt_ctx.multiply_polynomials(&a_mod_q, &b_mod_q);

        if prime_idx == 0 {
            println!("      product[0] mod q = {}", product_mod_q[0]);
        }

        for i in 0..n {
            result[i].values[prime_idx] = product_mod_q[i];
        }
    }

    result
}
