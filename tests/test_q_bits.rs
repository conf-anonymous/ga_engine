// Check actual bit length of Q

use num_bigint::BigInt;

#[test]
fn test_q_product_bits() {
    let primes_3 = vec![1141392289560813569i64, 1099511678977, 1099511683073];
    let primes_5 = vec![1141392289560813569i64, 1099511678977, 1099511683073, 1099511693313, 1099511697409];

    // 3 primes
    let q3: BigInt = primes_3.iter().map(|&p| BigInt::from(p)).product();
    let bits_3 = q3.bits();
    println!("\n3 primes:");
    println!("  Q = {}", q3);
    println!("  bits() = {}", bits_3);
    println!("  Expected digits (w=20): {}", (bits_3 + 19) / 20);

    // 5 primes
    let q5: BigInt = primes_5.iter().map(|&p| BigInt::from(p)).product();
    let bits_5 = q5.bits();
    println!("\n5 primes:");
    println!("  Q = {}", q5);
    println!("  bits() = {}", bits_5);
    println!("  Expected digits (w=20): {}", (bits_5 + 19) / 20);

    // Manual calculation
    let manual_bits_3: u32 = primes_3.iter()
        .map(|&q| {
            let mut bits = 0u32;
            let mut val = q;
            while val > 0 {
                bits += 1;
                val >>= 1;
            }
            bits
        })
        .sum();

    let manual_bits_5: u32 = primes_5.iter()
        .map(|&q| {
            let mut bits = 0u32;
            let mut val = q;
            while val > 0 {
                bits += 1;
                val >>= 1;
            }
            bits
        })
        .sum();

    println!("\nManual sum of individual prime bits:");
    println!("  3 primes: {}", manual_bits_3);
    println!("  5 primes: {}", manual_bits_5);
    println!("  Expected digits (3 primes, w=20): {}", (manual_bits_3 + 19) / 20);
    println!("  Expected digits (5 primes, w=20): {}", (manual_bits_5 + 19) / 20);
}
