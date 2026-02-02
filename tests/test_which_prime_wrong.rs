// Identify which prime has wrong NTT results
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::rns::rns_multiply;
use ga_engine::clifford_fhe::ckks_rns::polynomial_multiply_ntt;

#[test]
fn test_which_prime_breaks_crt() {
    let params = CliffordFHEParams::new_rns_mult_depth2();  // 4 primes
    println!("Primes: {:?}", params.moduli);

    let (_, sk, _) = rns_keygen(&params);

    let s_squared = rns_multiply(&sk.coeffs, &sk.coeffs, &params.moduli, polynomial_multiply_ntt);

    println!("\nChecking which prime causes CRT inconsistency...\n");

    // For each prime j, check if removing it makes things consistent
    for exclude_j in 0..4 {
        println!("=== Testing WITHOUT prime[{}] (q = {}) ===", exclude_j, params.moduli[exclude_j]);

        let mut consistent_count = 0;

        for coeff_idx in 0..params.n {
            let residues = &s_squared.rns_coeffs[coeff_idx];

            // Use first non-excluded prime as reference
            let ref_idx = if exclude_j == 0 { 1 } else { 0 };
            let r_ref = residues[ref_idx] as i128;
            let q_ref = params.moduli[ref_idx] as i128;

            let mut is_consistent = true;

            // Check against all other primes (except excluded one)
            for j in 0..4 {
                if j == exclude_j || j == ref_idx {
                    continue;
                }

                let qj = params.moduli[j] as i128;
                let rj = residues[j] as i128;
                let r_ref_mod_qj = ((r_ref % qj) + qj) % qj;

                if r_ref_mod_qj != rj {
                    is_consistent = false;
                    break;
                }
            }

            if is_consistent {
                consistent_count += 1;
            }
        }

        let percentage = 100.0 * consistent_count as f64 / params.n as f64;
        println!("  Consistent coefficients: {} / {} ({:.1}%)", consistent_count, params.n, percentage);

        if consistent_count == params.n {
            println!("  ✅ PERFECT! Prime[{}] is the problematic one!", exclude_j);
        }
    }
}
