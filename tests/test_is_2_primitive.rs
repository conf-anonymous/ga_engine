// Check if g=2 is a primitive root for prime[3]

#[test]
fn test_is_2_primitive_root() {
    let q = 1099511693313u64;
    let phi = q - 1;
    let g = 2u64;

    println!("Testing if g={} is a primitive root mod q={}", g, q);
    println!("phi = {}", phi);

    let prime_factors = vec![2u64, 97, 257, 673];
    println!("\nPrime factors of phi: {:?}", prime_factors);

    println!("\nChecking g^(phi/p) mod q for each prime factor p:");

    for &p in &prime_factors {
        let exp = phi / p;
        let result = mod_pow(g, exp, q);

        println!("  g^(phi/{}) = {}^{} mod {} = {}", p, g, exp, q, result);

        if result == 1 {
            println!("    ❌ EQUALS 1, so g is NOT primitive!");
        } else {
            println!("    ✅ Not 1");
        }
    }

    // Conclusion
    println!("\nConclusion:");
    let is_primitive = prime_factors.iter().all(|&p| {
        mod_pow(g, phi / p, q) != 1
    });

    if is_primitive {
        println!("✅ g={} IS a primitive root", g);
    } else {
        println!("❌ g={} is NOT a primitive root", g);
    }

    // Try g=3
    println!("\n\n=== Trying g=3 ===");
    let g = 3u64;

    for &p in &prime_factors {
        let exp = phi / p;
        let result = mod_pow(g, exp, q);

        println!("  g^(phi/{}) = {}^{} mod {} = {}", p, g, exp, q, result);

        if result == 1 {
            println!("    ❌ EQUALS 1");
        } else {
            println!("    ✅ Not 1");
        }
    }

    let is_primitive = prime_factors.iter().all(|&p| {
        mod_pow(g, phi / p, q) != 1
    });

    if is_primitive {
        println!("✅ g=3 IS a primitive root!");
    } else {
        println!("❌ g=3 is NOT a primitive root");
    }
}

fn mod_pow(mut base: u64, mut exp: u64, q: u64) -> u64 {
    let mut result = 1u64;
    base = base % q;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % q as u128) as u64;
        }
        base = ((base as u128 * base as u128) % q as u128) as u64;
        exp >>= 1;
    }
    result
}
