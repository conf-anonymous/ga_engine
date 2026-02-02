//! Find a replacement prime for q[3]

/// Miller-Rabin primality test
fn is_prime(n: u64) -> bool {
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

    // Test with several witnesses
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

        return false;
    }

    true
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

#[test]
fn find_ntt_friendly_prime_near_2_40() {
    let target = 1099511627776u64;  // 2^40
    let n = 1024u64;
    let required_div = 2 * n;  // (q-1) must be divisible by 2N = 2048

    println!("Looking for NTT-friendly primes near 2^40 = {}", target);
    println!("Requirement: (q-1) divisible by {} and q is prime", required_div);
    println!();

    // Search around 2^40
    let mut candidates = Vec::new();

    // Search upward from 2^40
    // We need candidates where (q-1) ≡ 0 (mod 2048), i.e., q ≡ 1 (mod 2048)
    let start = ((target / required_div) * required_div) + 1;
    println!("Start search from: {}", start);
    println!("Checking every {} values...\n", required_div);

    for i in 0..200_000 {
        let candidate = start + (i * required_div);

        if candidate < target {
            continue;
        }

        // Debug first few
        if i < 10 {
            let prime_check = is_prime(candidate);
            let mod_check = (candidate - 1) % required_div;
            println!("Checking {}: (q-1) mod {} = {} (should be 0), prime: {}",
                     candidate, required_div, mod_check, prime_check);
        }

        // Verify divisibility
        assert_eq!((candidate - 1) % required_div, 0, "Bug in search logic!");

        if is_prime(candidate) {
            candidates.push(candidate);
            let diff = (candidate as i128) - (target as i128);
            println!("✅ Found #{}: {} (2^40 + {})", candidates.len(), candidate, diff);

            if candidates.len() >= 10 {
                break;
            }
        }
    }

    println!("\n=== Summary: {} NTT-friendly primes near 2^40 ===", candidates.len());
    for (i, &q) in candidates.iter().enumerate() {
        let bits = 64 - q.leading_zeros();
        let diff = (q as i128) - (target as i128);
        println!("{}: {} ({} bits, 2^40 + {})", i, q, bits, diff);

        // Verify
        assert_eq!((q - 1) % required_div, 0);
        assert!(is_prime(q));
    }

    println!("\n=== Recommended primes to replace q[3] = 1099511693313 (COMPOSITE) ===");
    println!("Any of these primes will work. Choose one that doesn't conflict with q[1], q[2], q[4]:");
    println!();
    println!("Current (WRONG): q[3] = 1099511693313  # COMPOSITE! (3 × 366503897771)");
    println!();
    for (i, &q) in candidates.iter().take(5).enumerate() {
        println!("Option {}: q[3] = {}  # {} bits, prime, NTT-friendly", i + 1, q, 64 - q.leading_zeros());
    }
}

#[test]
fn verify_existing_primes() {
    let primes = vec![
        (0, 1141392289560813569u64),
        (1, 1099511678977),
        (2, 1099511683073),
        (3, 1099511693313), // COMPOSITE!
        (4, 1099511697409),
    ];

    println!("=== Verifying existing primes ===\n");

    for (i, q) in primes {
        let n = 1024u64;
        let div_ok = (q - 1) % (2 * n) == 0;
        let prime_ok = is_prime(q);
        let bits = 64 - q.leading_zeros();

        let status = if prime_ok && div_ok { "✅" } else { "❌" };

        println!("q[{}] = {} ({} bits)", i, q, bits);
        println!("  (q-1) mod 2N: {} {}", if div_ok { "✅ 0" } else { "❌ non-zero" }, if !div_ok { format!("({} mod {})", q-1, 2*n) } else { String::new() });
        println!("  Primality: {}", if prime_ok { "✅ PRIME" } else { "❌ COMPOSITE" });
        println!("  Overall: {}\n", status);
    }
}
