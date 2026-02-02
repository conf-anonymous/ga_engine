//! Test to find the bug in mod_pow_u64

/// Modular exponentiation using u128 intermediate (CURRENT VERSION)
fn mod_pow_u64_v1(mut base: u64, mut exp: u64, m: u64) -> u64 {
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

/// Modular exponentiation with EXPLICIT modular reduction
fn mod_pow_u64_v2(base: u64, exp: u64, m: u64) -> u64 {
    if m == 1 { return 0; }

    let mut result = 1u128;
    let mut b = (base % m) as u128;
    let mut e = exp;
    let m128 = m as u128;

    while e > 0 {
        if e & 1 == 1 {
            result = (result * b) % m128;
        }
        b = (b * b) % m128;
        e >>= 1;
    }

    (result % m128) as u64
}

#[test]
fn test_fermats_little_theorem() {
    // Use an actual prime from V2 params (41-bit NTT-friendly prime)
    let q = 1099511678977u64;  // This is actually prime
    let phi = q - 1;

    println!("Testing Fermat's Little Theorem for q = {}", q);
    println!("phi = q-1 = {}", phi);
    println!("Note: Using a 41-bit prime from V2 parameters");

    // Test for several small bases
    for base in [2u64, 3, 5, 7, 11, 13] {
        let result_v1 = mod_pow_u64_v1(base, phi, q);
        let result_v2 = mod_pow_u64_v2(base, phi, q);

        println!("\nBase = {}:", base);
        println!("  v1 (current): {}^{} mod {} = {}", base, phi, q, result_v1);
        println!("  v2 (fixed):   {}^{} mod {} = {}", base, phi, q, result_v2);

        assert_eq!(result_v1, 1, "Fermat's Little Theorem: {}^(q-1) must be 1 (mod q) [v1]", base);
        assert_eq!(result_v2, 1, "Fermat's Little Theorem: {}^(q-1) must be 1 (mod q) [v2]", base);
    }
}

#[test]
fn test_primitive_root_check() {
    // Use an actual prime from V2 params
    let q = 1099511678977u64;  // This is actually prime
    let phi = q - 1;
    // phi = 1099511678976 = 2^11 × 536870351
    // Prime factors: 2 and 536870351
    let prime_factors = vec![2u64, 536870351];

    println!("Testing if g=2 is primitive root of q = {}", q);
    println!("phi = q-1 = {} = 2^11 × 536870351", phi);

    for g in [2u64, 3, 5, 7] {
        println!("\n=== Testing g = {} ===", g);

        // Check g^(phi/p) != 1 for all prime factors p
        let mut is_primitive = true;
        for &p in &prime_factors {
            let exp = phi / p;
            let val_v1 = mod_pow_u64_v1(g, exp, q);
            let val_v2 = mod_pow_u64_v2(g, exp, q);
            println!("  g^(phi/{}) mod q:", p);
            println!("    v1: {}", val_v1);
            println!("    v2: {}", val_v2);
            if val_v2 == 1 {
                is_primitive = false;
            }
        }

        // Check g^phi == 1 (Fermat)
        let g_phi_v1 = mod_pow_u64_v1(g, phi, q);
        let g_phi_v2 = mod_pow_u64_v2(g, phi, q);
        println!("  g^phi mod q:");
        println!("    v1: {}", g_phi_v1);
        println!("    v2: {} (should be 1)", g_phi_v2);

        if is_primitive && g_phi_v2 == 1 {
            println!("  ✅ g={} IS a primitive root", g);
        } else {
            println!("  ❌ g={} is NOT a primitive root", g);
        }
    }
}
