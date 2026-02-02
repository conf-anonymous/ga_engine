// Check if NTT roots of unity exist for all primes

#[test]
fn test_ntt_roots_exist() {
    let primes_3 = vec![1141392289560813569i64, 1099511678977, 1099511683073];
    let primes_4 = vec![1141392289560813569i64, 1099511678977, 1099511683073, 1099511693313];
    let primes_5 = vec![1141392289560813569i64, 1099511678977, 1099511683073, 1099511693313, 1099511697409];

    let n = 1024;
    let two_n = 2048;

    println!("\nChecking if 2N-th roots of unity exist for N={}:", n);

    for (set_name, primes) in [("3-prime", primes_3), ("4-prime", primes_4), ("5-prime", primes_5)] {
        println!("\n=== {} set ===", set_name);

        for (i, &q) in primes.iter().enumerate() {
            // For NTT to work mod q with polynomial ring Z[X]/(X^N + 1),
            // we need a primitive 2N-th root of unity mod q
            // This exists iff q ≡ 1 (mod 2N)

            let remainder = q % (two_n as i64);

            if remainder == 1 {
                println!("  Prime[{}]: q = {} → q ≡ 1 (mod {}) ✅", i, q, two_n);
            } else {
                println!("  Prime[{}]: q = {} → q ≡ {} (mod {}) ❌ NTT WON'T WORK!",
                         i, q, remainder, two_n);
            }
        }
    }
}
