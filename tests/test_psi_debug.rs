//! Debug test for psi computation for prime[3]
//! V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]

use ga_engine::clifford_fhe::ckks_rns::polynomial_multiply_ntt;

/// Modular exponentiation: base^exp mod m
fn mod_pow_u64(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % m as u128) as u64;
        }
        base = ((base as u128 * base as u128) % m as u128) as u64;
        exp >>= 1;
    }
    result
}

/// Find a primitive root of Z_q^* (q prime)
fn primitive_root(q: u64) -> u64 {
    let phi = q - 1;
    let prime_factors = prime_factors_of(phi);

    eprintln!("Finding primitive root for q = {}", q);
    eprintln!("phi = q-1 = {}", phi);
    eprintln!("Prime factors of phi: {:?}", prime_factors);

    for g in 2..q {
        let mut is_primitive = true;
        for &p in &prime_factors {
            let test_exp = phi / p;
            let test_val = mod_pow_u64(g, test_exp, q);
            if test_val == 1 {
                is_primitive = false;
                break;
            }
        }
        if is_primitive {
            eprintln!("Found primitive root: g = {}", g);
            // Verify it's correct
            for &p in &prime_factors {
                let test_exp = phi / p;
                let test_val = mod_pow_u64(g, test_exp, q);
                eprintln!("  g^(phi/{}) = {} (should NOT be 1)", p, test_val);
            }
            let g_to_phi = mod_pow_u64(g, phi, q);
            eprintln!("  g^phi = {} (should be 1)", g_to_phi);
            return g;
        }
    }
    unreachable!()
}

/// Find prime factors of n
fn prime_factors_of(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    if n % 2 == 0 {
        factors.push(2);
        while n % 2 == 0 { n /= 2; }
    }
    let mut p = 3u64;
    while p * p <= n {
        if n % p == 0 {
            factors.push(p);
            while n % p == 0 { n /= p; }
        }
        p += 2;
    }
    if n > 1 { factors.push(n); }
    factors
}

#[test]
fn test_psi_computation() {
    let q = 1099511693313u64;  // prime[3]
    let n = 1024usize;

    // Find primitive root
    let g = primitive_root(q);

    // Compute psi = g^((q-1)/(2N))
    let two_n = 2 * n as u64;
    eprintln!("\n=== Computing psi ===");
    eprintln!("N = {}", n);
    eprintln!("2N = {}", two_n);
    eprintln!("(q-1) mod (2N) = {}", (q - 1) % two_n);

    let exp = (q - 1) / two_n;
    eprintln!("exp = (q-1)/(2N) = {}", exp);

    let psi = mod_pow_u64(g, exp, q);
    eprintln!("psi = g^exp = {}", psi);

    // Check psi properties
    eprintln!("\n=== Checking psi properties ===");
    let psi_to_2n = mod_pow_u64(psi, two_n, q);
    eprintln!("psi^(2N) = {} (should be 1)", psi_to_2n);

    let psi_to_n = mod_pow_u64(psi, n as u64, q);
    eprintln!("psi^N = {} (should be {} = -1 mod q)", psi_to_n, q - 1);

    let omega = mod_pow_u64(psi, 2, q);
    eprintln!("\nomega = psi^2 = {}", omega);
    let omega_to_n = mod_pow_u64(omega, n as u64, q);
    eprintln!("omega^N = {} (should be 1)", omega_to_n);

    let omega_to_n_half = mod_pow_u64(omega, (n / 2) as u64, q);
    eprintln!("omega^(N/2) = {} (should NOT be 1, should be {})", omega_to_n_half, q - 1);

    // Check if assertions would pass
    assert_eq!(psi_to_2n, 1, "psi^(2N) must be 1");
    assert_eq!(psi_to_n, q - 1, "psi^N must be -1");
    assert_eq!(omega_to_n, 1, "omega^N must be 1");
    assert_ne!(omega_to_n_half, 1, "omega^(N/2) must not be 1");
}
