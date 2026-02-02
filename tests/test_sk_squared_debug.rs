// Debug secret key squared computation
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::rns::rns_multiply;
use ga_engine::clifford_fhe::ckks_rns::polynomial_multiply_ntt;

#[test]
fn debug_sk_squared_4_primes() {
    let params = CliffordFHEParams::new_rns_mult_depth2();  // 4 primes
    println!("Primes: {:?}", params.moduli);

    let (_, sk, _) = rns_keygen(&params);

    println!("\nSecret key s (first 5 coeffs in standard form):");
    let s_coeffs = sk.coeffs.to_coeffs(&params.moduli);
    for i in 0..5 {
        println!("  s[{}] = {}", i, s_coeffs[i]);
    }

    println!("\nSecret key s (first coeff RNS residues):");
    for j in 0..params.moduli.len() {
        println!("  s[0] mod q[{}] = {}", j, sk.coeffs.rns_coeffs[0][j]);
    }

    // Manually compute s[0]² in regular arithmetic
    let s0_val = s_coeffs[0];
    let s0_squared = s0_val * s0_val;
    println!("\ns[0]² in regular arithmetic: {} × {} = {}", s0_val, s0_val, s0_squared);

    // Manually compute what residues SHOULD be
    println!("\nExpected residues for s[0]²:");
    for j in 0..params.moduli.len() {
        let qj = params.moduli[j];
        let expected = ((s0_squared % qj) + qj) % qj;
        println!("  s[0]² mod q[{}] = {}", j, expected);
    }

    // Now compute s² using RNS multiplication
    let s_squared = rns_multiply(&sk.coeffs, &sk.coeffs, &params.moduli, polynomial_multiply_ntt);

    println!("\nActual s² from rns_multiply (first coeff RNS residues):");
    for j in 0..params.moduli.len() {
        println!("  s²[0] mod q[{}] = {}", j, s_squared.rns_coeffs[0][j]);
    }

    // CRT reconstruct
    let s_squared_coeffs = s_squared.to_coeffs(&params.moduli);
    println!("\ns² CRT reconstruction:");
    println!("  s²[0] = {}", s_squared_coeffs[0]);

    // For polynomial multiplication in X^N + 1, we need to account for wraparound
    // s² might not equal s[0]² due to other term contributions
    println!("\nNote: Polynomial multiplication in R = Z[X]/(X^N + 1) includes wraparound.");
    println!("So s²[0] is not just s[0]² but includes contributions from other coefficients.");

    // Let's manually verify NTT multiplication for first few coefficients
    println!("\n=== Verifying NTT multiplication per prime ===");

    for j in 0..params.moduli.len() {
        let qj = params.moduli[j];
        let n = params.n;

        println!("\nPrime[{}]: q = {}", j, qj);

        // Extract s's residues for this prime
        let s_mod_qj: Vec<i64> = (0..n).map(|i| sk.coeffs.rns_coeffs[i][j]).collect();

        println!("  s mod q (first 5 coeffs): {:?}", &s_mod_qj[..5]);

        // Call NTT multiply for this prime
        let result_mod_qj = polynomial_multiply_ntt(&s_mod_qj, &s_mod_qj, qj, n);

        println!("  s² mod q (first 5 coeffs): {:?}", &result_mod_qj[..5]);

        // Compare with what rns_multiply gave us
        println!("  rns_multiply gave (first 5 coeffs):");
        for i in 0..5 {
            println!("    [{}]: expected={}, got={}, match={}",
                     i, result_mod_qj[i], s_squared.rns_coeffs[i][j],
                     result_mod_qj[i] == s_squared.rns_coeffs[i][j]);
        }

        // Check if they match
        let all_match = (0..n).all(|i| result_mod_qj[i] == s_squared.rns_coeffs[i][j]);
        if all_match {
            println!("  ✅ All coefficients match for prime[{}]", j);
        } else {
            println!("  ❌ MISMATCH for prime[{}]!", j);
            // Find first mismatch
            for i in 0..n {
                if result_mod_qj[i] != s_squared.rns_coeffs[i][j] {
                    println!("     First mismatch at coeff[{}]: expected={}, got={}",
                             i, result_mod_qj[i], s_squared.rns_coeffs[i][j]);
                    break;
                }
            }
        }
    }
}
