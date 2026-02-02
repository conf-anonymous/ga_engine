//! Verify that the public key satisfies b + a*s ≈ e
//!
//! Run with: cargo test --test test_v2_verify_pk --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

#[test]
fn test_verify_public_key() {
    println!("\n========== VERIFY PUBLIC KEY ==========");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();

    let n = params.n;
    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();

    println!("Public key should satisfy: b = -a*s + e");
    println!("Therefore: b + a*s = e (small error)");

    // Compute a*s
    let a_times_s = multiply_polys(&pk.a, &sk.coeffs, n, &moduli, &key_ctx);

    // Compute b + a*s
    let b_plus_as: Vec<_> = pk.b.iter().zip(&a_times_s).map(|(b_i, as_i)| b_i.add(as_i)).collect();

    println!("\nChecking first coefficient:");
    println!("  (b + a*s)[0] = {:?}", b_plus_as[0].values);

    let val = b_plus_as[0].values[0];
    let q0 = b_plus_as[0].moduli[0];
    let centered = if val > q0/2 {
        val as i64 - q0 as i64
    } else {
        val as i64
    };

    println!("  Centered: {}", centered);

    if centered.abs() < 1000 {
        println!("  ✓ PASS: Public key is correct! Error = {}", centered);
    } else {
        println!("  ✗ FAIL: Public key has huge error = {}", centered);
        println!("        Expected small error (~3-10), got {}", centered);
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
