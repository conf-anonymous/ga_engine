// Check if secret key is CRT-consistent
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]


use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;

#[test]
fn test_secret_key_crt_consistency() {
    let params = CliffordFHEParams::new_rns_mult_depth2();  // 4 primes
    let (_, sk, _) = rns_keygen(&params);

    println!("Secret key (first 10 coeffs):");
    let s_coeffs = sk.coeffs.to_coeffs(&params.moduli);
    for i in 0..10 {
        println!("  s[{}] = {}", i, s_coeffs[i]);
    }

    println!("\nVerifying CRT consistency for each coefficient:");

    for i in 0..10 {
        let value = s_coeffs[i];
        let residues = &sk.coeffs.rns_coeffs[i];

        let mut consistent = true;
        for (j, &qj) in params.moduli.iter().enumerate() {
            let expected = ((value % qj) + qj) % qj;
            let actual = residues[j];

            if expected != actual {
                println!("  s[{}]: ❌ INCONSISTENT at prime[{}]: expected {}, got {}",
                         i, j, expected, actual);
                consistent = false;
            }
        }

        if consistent {
            println!("  s[{}]: ✅ CRT consistent", i);
        }
    }
}
