// Test gadget decomposition with different numbers of primes
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::rns::{RnsPolynomial, decompose_base_pow2};

#[test]
fn test_decomposition_5_primes() {
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let primes = &params.moduli;
    let n = params.n;

    println!("\n=== Testing Gadget Decomposition with 5 Primes ===");
    println!("Primes: {:?}", primes);

    // Create a simple test polynomial with known value
    // Let's use a polynomial where first coeff = 2*scale^2 ≈ 2.42e24
    let scale = params.scale;
    let value = ((2.0 * scale * scale) as i64);

    let mut coeffs = vec![0i64; n];
    coeffs[0] = value;

    println!("\nOriginal value: {}", value);

    // Convert to RNS
    let poly = RnsPolynomial::from_coeffs(&coeffs, primes, n, 0);

    println!("RNS residues[0]: {:?}", &poly.rns_coeffs[0]);

    // Decompose in base 2^20
    let w = 20;
    let digits = decompose_base_pow2(&poly, primes, w);

    println!("\nNumber of digits: {}", digits.len());

    // Print all digits
    for (t, digit) in digits.iter().enumerate() {
        println!("  digit[{}][0] residues: {:?}", t, &digit.rns_coeffs[0]);
    }

    // Verify: reconstruct value from digits
    // value should = sum_t (digit_t * 2^(20*t))
    use num_bigint::BigInt;
    use num_traits::{Zero, One};

    let b = BigInt::from(1i64 << 20);  // 2^20
    let mut reconstructed = BigInt::zero();
    let mut b_power = BigInt::one();

    for digit in &digits {
        // Get the digit value by CRT reconstruction
        let digit_coeffs = digit.to_coeffs(primes);
        let digit_val_i64 = digit_coeffs[0];

        // Handle signed digits (balanced representation) - use BigInt for large primes
        let q0_big = BigInt::from(primes[0]);
        let digit_val_big = BigInt::from(digit_val_i64);
        let q0_half = &q0_big / 2;

        let digit_centered = if digit_val_big > q0_half {
            digit_val_big - q0_big
        } else {
            digit_val_big
        };

        println!("  digit value (centered): {}", digit_centered);

        reconstructed += &digit_centered * &b_power;
        b_power *= &b;
    }

    let reconstructed_i64 = reconstructed.to_string().parse::<i128>().unwrap();

    println!("\nOriginal:      {}", value);
    println!("Reconstructed: {}", reconstructed_i64);
    let error = (reconstructed_i64 - value as i128).abs();
    println!("Error:         {}", error);

    // Decomposition should be exact or very close
    assert!(error < 1000, "Decomposition/reconstruction error too large: {}", error);
}
