// Check CRT consistency for all coefficients of s²
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]


use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::rns::rns_multiply;
use ga_engine::clifford_fhe::ckks_rns::polynomial_multiply_ntt;

fn check_coeff_crt(poly: &ga_engine::clifford_fhe::rns::RnsPolynomial, primes: &[i64], coeff_idx: usize) -> bool {
    if poly.num_primes() < 2 {
        return true;
    }

    let residues = &poly.rns_coeffs[coeff_idx];
    let r0 = residues[0] as i128;

    for j in 1..poly.num_primes() {
        let qj = primes[j] as i128;
        let rj = residues[j] as i128;
        let r0_mod_qj = ((r0 % qj) + qj) % qj;

        if r0_mod_qj != rj {
            return false;
        }
    }

    true
}

#[test]
fn test_all_coefficients_crt_consistency() {
    let params = CliffordFHEParams::new_rns_mult_depth2();  // 4 primes
    let (_, sk, _) = rns_keygen(&params);

    let s_squared = rns_multiply(&sk.coeffs, &sk.coeffs, &params.moduli, polynomial_multiply_ntt);

    println!("\nChecking CRT consistency for all {} coefficients:", params.n);

    let mut consistent_count = 0;
    let mut inconsistent_count = 0;

    for i in 0..params.n {
        let is_consistent = check_coeff_crt(&s_squared, &params.moduli, i);

        if is_consistent {
            consistent_count += 1;
        } else {
            inconsistent_count += 1;
            if inconsistent_count <= 10 {  // Print first 10 inconsistencies
                println!("  coeff[{}]: ❌ INCONSISTENT", i);
                println!("    residues: {:?}", &s_squared.rns_coeffs[i]);
            }
        }
    }

    println!("\n=== Summary ===");
    println!("Consistent:   {} / {}", consistent_count, params.n);
    println!("Inconsistent: {} / {}", inconsistent_count, params.n);
    println!("Percentage consistent: {:.1}%",
             100.0 * consistent_count as f64 / params.n as f64);

    if inconsistent_count > 0 {
        panic!("{} coefficients are CRT-inconsistent!", inconsistent_count);
    }
}
