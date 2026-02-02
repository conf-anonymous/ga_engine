//! Test to verify NTT multiplication with extracted polynomials on CPU

use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

fn main() {
    println!("=== CPU NTT Polynomial Extraction Test ===\n");

    let params = CliffordFHEParams::new_128bit();
    let ctx = CkksContext::new(params.clone());
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();

    // Encode and encrypt a simple value
    let test_val = 2.0;
    let pt = ctx.encode(&[test_val]);
    let ct = ctx.encrypt(&pt, &pk);

    println!("Test value: {}", test_val);
    println!("Plaintext scale: {}", pt.scale);
    println!("Ciphertext level: {}, scale: {}", ct.level, ct.scale);
    println!();

    // Extract c1 and sk at level for first prime
    let level = ct.level;
    let num_primes = level + 1;
    let q0 = params.moduli[0];

    println!("First prime: q0 = {}", q0);
    println!("Number of primes at level {}: {}", level, num_primes);
    println!();

    // Extract c1 polynomial for first prime
    let c1_prime0: Vec<u64> = ct.c1.iter().map(|rns| rns.values[0]).collect();

    // Extract sk polynomial for first prime (truncated to level)
    let sk_prime0: Vec<u64> = sk.coeffs.iter().map(|rns| rns.values[0]).collect();

    println!("c1[0] (prime 0): {}", c1_prime0[0]);
    println!("c1[1] (prime 0): {}", c1_prime0[1]);
    println!("sk[0] (prime 0): {}", sk_prime0[0]);
    println!("sk[1] (prime 0): {}", sk_prime0[1]);
    println!();

    // Now multiply using NTT
    let ntt_ctx = NttContext::new(params.n, q0);
    let product = ntt_ctx.multiply_polynomials(&c1_prime0, &sk_prime0);

    println!("NTT multiply result:");
    println!("product[0]: {}", product[0]);
    println!("product[1]: {}", product[1]);
    println!();

    // Also compute using the CPU decrypt for comparison
    let pt_dec = ctx.decrypt(&ct, &sk);
    println!("Full decrypt (m = c0 + c1*sk):");
    println!("m[0] (prime 0): {}", pt_dec.coeffs[0].values[0]);
    println!();

    // Also show c0 for reference
    println!("c0[0] (prime 0): {}", ct.c0[0].values[0]);
    println!();

    // Verify: m[0] should = (c0[0] + product[0]) mod q0
    let expected_m0 = ((ct.c0[0].values[0] as u128 + product[0] as u128) % q0 as u128) as u64;
    println!("Expected m[0] = (c0[0] + product[0]) mod q0:");
    println!("  = ({} + {}) mod {}", ct.c0[0].values[0], product[0], q0);
    println!("  = {}", expected_m0);
    println!();

    if expected_m0 == pt_dec.coeffs[0].values[0] {
        println!("✅ NTT multiplication matches!");
    } else {
        println!("❌ NTT multiplication MISMATCH!");
        println!("   Expected: {}", expected_m0);
        println!("   Got:      {}", pt_dec.coeffs[0].values[0]);
    }
}
