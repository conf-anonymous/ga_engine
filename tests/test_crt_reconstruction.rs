// Test CRT reconstruction directly
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]

use ga_engine::clifford_fhe::rns::RnsPolynomial;
use num_bigint::BigInt;
use num_traits::Zero;

#[test]
fn test_crt_reconstruction_4_primes() {
    let primes = vec![1141392289560813569i64, 1099511678977, 1099511683073, 1099511693313];
    let n = 4;

    // Test case: residues from s²[0]
    let residues = vec![
        1141392289560813543,  // s²[0] mod q[0]
        1099511678951,         // s²[0] mod q[1]
        1099511683047,         // s²[0] mod q[2]
        837136114724,          // s²[0] mod q[3]
    ];

    println!("Input residues: {:?}", residues);

    // Create RNS polynomial with these residues for coeff[0]
    let mut rns_coeffs = vec![vec![0i64; 4]; n];
    rns_coeffs[0] = residues.clone();

    let poly = RnsPolynomial::new(rns_coeffs, n, 0);

    // Reconstruct using CRT
    let reconstructed = poly.to_coeffs(&primes);
    let value = reconstructed[0];

    println!("CRT reconstructed value: {}", value);

    // Verify: does this value give back the correct residues?
    println!("\nVerification - computing value mod each prime:");
    for (j, &qj) in primes.iter().enumerate() {
        let residue_from_value = ((value % qj) + qj) % qj;
        let original_residue = residues[j];

        println!("  Prime[{}]: q = {}", j, qj);
        println!("    Original residue:     {}", original_residue);
        println!("    value mod q:          {}", residue_from_value);
        println!("    Match: {}", residue_from_value == original_residue);

        if residue_from_value != original_residue {
            println!("    ❌ MISMATCH!");
        }
    }

    // The real test: manually compute CRT using the formula
    println!("\n=== Manual CRT Computation ===");

    // Q = product of all primes
    let q_prod: BigInt = primes.iter().map(|&q| BigInt::from(q)).product();
    println!("Q = {}", q_prod);

    let mut x = BigInt::zero();
    for j in 0..primes.len() {
        let qj = BigInt::from(primes[j]);
        let rj = BigInt::from(residues[j]);

        // Compute Q/qj
        let q_div_qj = &q_prod / &qj;

        // Compute (Q/qj)^{-1} mod qj
        let inv = mod_inverse_bigint(&q_div_qj, &qj);

        // Add rj * (Q/qj) * inv to x
        let term = (&rj * &q_div_qj * &inv) % &q_prod;
        x = (x + term) % &q_prod;

        println!("Step {}: rj={}, Q/qj={}, inv={}", j, rj, q_div_qj, inv);
    }

    println!("\nManual CRT result: {}", x);

    // Center-lift
    let q_half = &q_prod / 2;
    let x_centered = if x > q_half {
        x - q_prod
    } else {
        x.clone()
    };

    println!("Centered result: {}", x_centered);

    // Convert to i64
    let x_i64: i64 = x_centered.to_string().parse().expect("Failed to parse");
    println!("As i64: {}", x_i64);

    // Compare with to_coeffs result
    println!("\nComparison:");
    println!("  Manual CRT:     {}", x_i64);
    println!("  to_coeffs():    {}", value);
    println!("  Match: {}", x_i64 == value);

    if x_i64 != value {
        println!("❌ CRT RECONSTRUCTION BUG FOUND!");
        println!("The to_coeffs() function is giving wrong results!");
    }
}

fn mod_inverse_bigint(a: &BigInt, m: &BigInt) -> BigInt {
    use num_bigint::Sign;

    let (gcd, x, _) = extended_gcd_bigint(a, m);

    if gcd != BigInt::from(1) {
        panic!("Modular inverse does not exist");
    }

    let mut result = x % m;
    if result.sign() == Sign::Minus {
        result += m;
    }

    result
}

fn extended_gcd_bigint(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if b == &BigInt::from(0) {
        return (a.clone(), BigInt::from(1), BigInt::from(0));
    }

    let (gcd, x1, y1) = extended_gcd_bigint(b, &(a % b));
    let x = y1.clone();
    let y = x1 - (a / b) * &y1;

    (gcd, x, y)
}
