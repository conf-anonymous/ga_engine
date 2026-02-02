// Test CRT reconstruction with increasing number of primes
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]

use ga_engine::clifford_fhe::rns::RnsPolynomial;

fn test_crt_with_n_primes(primes: &[i64], test_value: i64) {
    println!("\n=== Testing CRT with {} primes ===", primes.len());
    println!("Primes: {:?}", primes);
    println!("Test value: {}", test_value);

    // Convert test value to RNS form
    let n = 4; // Small polynomial for testing
    let coeffs = vec![test_value, test_value * 2, test_value * 3, 0];

    println!("Original coeffs: {:?}", coeffs);

    // Create RNS polynomial
    let rns_poly = RnsPolynomial::from_coeffs(&coeffs, primes, n, 0);

    println!("RNS coeffs[0] (residues): {:?}", rns_poly.rns_coeffs[0]);
    println!("RNS coeffs[1] (residues): {:?}", rns_poly.rns_coeffs[1]);

    // Reconstruct using CRT
    let reconstructed = rns_poly.to_coeffs(primes);

    println!("Reconstructed coeffs: {:?}", reconstructed);

    // Check correctness
    for i in 0..n {
        let error = (reconstructed[i] - coeffs[i]).abs();
        println!("  coeff[{}]: original={}, reconstructed={}, error={}",
                 i, coeffs[i], reconstructed[i], error);
        assert_eq!(reconstructed[i], coeffs[i],
                   "CRT reconstruction failed for coeff[{}] with {} primes", i, primes.len());
    }

    println!("✅ CRT works correctly with {} primes!", primes.len());
}

#[test]
fn test_crt_3_primes() {
    let primes = vec![
        1141392289560813569,  // q₀ (60-bit)
        1099511678977,        // q₁ (41-bit)
        1099511683073,        // q₂ (41-bit)
    ];

    test_crt_with_n_primes(&primes, 12345678);
}

#[test]
fn test_crt_4_primes() {
    let primes = vec![
        1141392289560813569,  // q₀ (60-bit)
        1099511678977,        // q₁ (41-bit)
        1099511683073,        // q₂ (41-bit)
        1099511693313,        // q₃ (41-bit)
    ];

    test_crt_with_n_primes(&primes, 12345678);
}

#[test]
fn test_crt_5_primes() {
    let primes = vec![
        1141392289560813569,  // q₀ (60-bit)
        1099511678977,        // q₁ (41-bit)
        1099511683073,        // q₂ (41-bit)
        1099511693313,        // q₃ (41-bit)
        1099511697409,        // q₄ (41-bit)
    ];

    test_crt_with_n_primes(&primes, 12345678);
}

#[test]
fn test_crt_6_primes() {
    let primes = vec![
        1141392289560813569,  // q₀ (60-bit)
        1099511678977,        // q₁ (41-bit)
        1099511683073,        // q₂ (41-bit)
        1099511693313,        // q₃ (41-bit)
        1099511697409,        // q₄ (41-bit)
        1099511701505,        // q₅ (41-bit)
    ];

    // Verify q₅ is NTT-friendly
    let q5 = 1099511701505_i64;
    assert_eq!((q5 - 1) % 2048, 0, "q₅ must be NTT-friendly");

    test_crt_with_n_primes(&primes, 12345678);
}

#[test]
fn test_crt_large_value() {
    // Test with a value close to the scaling factor
    let primes = vec![
        1141392289560813569,
        1099511678977,
        1099511683073,
        1099511693313,
        1099511697409,
    ];

    let scale = 2_f64.powi(40); // ≈ 1.1 × 10^12
    let test_value = (scale * 3.5).round() as i64;

    test_crt_with_n_primes(&primes, test_value);
}

#[test]
fn test_crt_negative_value() {
    let primes = vec![
        1141392289560813569,
        1099511678977,
        1099511683073,
        1099511693313,
    ];

    let test_value = -12345678_i64;

    println!("\n=== Testing CRT with negative value ===");
    println!("Test value: {}", test_value);

    let n = 4;
    let coeffs = vec![test_value, 0, 0, 0];

    let rns_poly = RnsPolynomial::from_coeffs(&coeffs, &primes, n, 0);
    let reconstructed = rns_poly.to_coeffs(&primes);

    println!("Original: {}, Reconstructed: {}", coeffs[0], reconstructed[0]);

    assert_eq!(reconstructed[0], coeffs[0],
               "CRT failed for negative value");
}
