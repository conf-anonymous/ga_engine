// Test factorization of phi for prime[3]

#[test]
fn test_factor_phi_prime3() {
    let q = 1099511693313u64;
    let phi = q - 1; // 1099511693312

    println!("phi = q - 1 = {}", phi);

    let factors = prime_factors_of(phi);

    println!("\nPrime factors of phi:");
    for &f in &factors {
        println!("  {}", f);
    }

    // Verify
    let product: u64 = factors.iter().map(|&f| {
        let mut count = 0u32;
        let mut n = phi;
        while n % f == 0 {
            count += 1;
            n /= f;
        }
        f.pow(count)
    }).product();

    println!("\nProduct of factors: {}", product);
    println!("Original phi:       {}", phi);
    println!("Match: {}", product == phi);

    // Manual factorization: phi = 2^18 × 4097 × 4099
    println!("\nExpected: 2^18 × ? = {}", phi);
    let after_2s = phi / (1u64 << 18);
    println!("phi / 2^18 = {}", after_2s);

    // Factor this
    println!("\nFactoring {}...", after_2s);
    for p in 2..100000u64 {
        if after_2s % p == 0 {
            println!("  Divisible by {}", p);
            let other = after_2s / p;
            println!("  {} = {} × {}", after_2s, p, other);
            break;
        }
    }
}

fn prime_factors_of(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();

    // Factor out 2
    if n % 2 == 0 {
        factors.push(2);
        while n % 2 == 0 {
            n /= 2;
        }
    }

    // Factor out odd primes
    let mut p = 3u64;
    while p * p <= n {
        if n % p == 0 {
            factors.push(p);
            while n % p == 0 {
                n /= p;
            }
        }
        p += 2;
    }

    // If n > 1, then it's a prime factor
    if n > 1 {
        factors.push(n);
    }

    factors
}
