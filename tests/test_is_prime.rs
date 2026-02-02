//! Test if our "primes" are actually prime
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]


/// Simple trial division primality test
fn is_prime_trial(n: u64) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }

    let mut d = 3u64;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// Miller-Rabin primality test
fn is_prime_miller_rabin(n: u64) -> bool {
    if n < 2 { return false; }
    if n == 2 || n == 3 { return true; }
    if n % 2 == 0 { return false; }

    // Write n-1 as 2^r * d
    let mut d = n - 1;
    let mut r = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }

    // Test with witnesses [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    let witnesses = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    'witness: for &a in &witnesses {
        if a >= n { continue; }

        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue 'witness;
        }

        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }

        return false; // Composite
    }

    true // Probably prime
}

fn mod_pow(base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u128;
    let mut b = (base % m) as u128;
    let m128 = m as u128;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * b) % m128;
        }
        b = (b * b) % m128;
        exp >>= 1;
    }

    (result % m128) as u64
}

fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Find smallest prime factor
fn smallest_factor(n: u64) -> Option<u64> {
    if n % 2 == 0 { return Some(2); }

    let mut d = 3u64;
    let limit = (n as f64).sqrt() as u64 + 1;
    while d <= limit {
        if n % d == 0 {
            return Some(d);
        }
        d += 2;
        if d > 1_000_000 { break; } // Give up after 1M
    }
    None
}

#[test]
fn test_our_primes() {
    // Use the actual primes from params.rs
    use ga_engine::clifford_fhe::params::CliffordFHEParams;

    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let primes: Vec<i64> = params.moduli.clone();

    for (i, &p) in primes.iter().enumerate() {
        let p_u64 = p as u64;
        println!("\n=== Testing q[{}] = {} ===", i, p);

        // Check divisibility by 2N (required for NTT)
        let n = 1024i64;
        let div_2n = (p - 1) % (2 * n);
        println!("(q-1) mod 2N = {} (should be 0)", div_2n);

        // Miller-Rabin test
        let is_prime_mr = is_prime_miller_rabin(p_u64);
        println!("Miller-Rabin: {}", if is_prime_mr { "PRIME" } else { "COMPOSITE" });

        // Try to find factor
        if !is_prime_mr {
            println!("Searching for factors...");
            if let Some(factor) = smallest_factor(p_u64) {
                println!("  Found factor: {}", factor);
                println!("  {} / {} = {}", p, factor, p / factor as i64);
            }
        }

        // Fermat test
        let base = 2u64;
        let fermat = mod_pow(base, (p - 1) as u64, p_u64);
        println!("Fermat test (2^(q-1) mod q): {} (should be 1)", fermat);

        assert_eq!(div_2n, 0, "q[{}] does not satisfy (q-1) divisible by 2N", i);
        assert!(is_prime_mr, "q[{}] = {} is NOT PRIME!", i, p);
        assert_eq!(fermat, 1, "q[{}] fails Fermat test", i);
    }
}
