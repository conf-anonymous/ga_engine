// Divide-and-conquer tests for relinearization steps
//
// Break down relinearization into atomic testable operations to isolate
// where CRT consistency breaks with 4+ primes.
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::rns::{RnsPolynomial, decompose_base_pow2, rns_add, rns_sub, rns_multiply};
use ga_engine::clifford_fhe::ckks_rns::polynomial_multiply_ntt;

/// Check if an RNS polynomial has CRT-consistent residues
///
/// For residues [r₀, r₁, ..., rₖ] to be CRT-consistent, they must satisfy:
/// r₀ ≡ rⱼ (mod qⱼ) for all j
///
/// This avoids full CRT reconstruction which can overflow i64.
fn check_crt_consistency(poly: &RnsPolynomial, primes: &[i64], name: &str) -> bool {
    if poly.num_primes() < 2 {
        return true;
    }

    let residues = &poly.rns_coeffs[0];
    let r0 = residues[0] as i128;  // Use first residue as reference
    let _q0 = primes[0] as i128;

    // Check if r0 (mod qj) == rj for all j
    for j in 1..poly.num_primes() {
        let qj = primes[j] as i128;
        let rj = residues[j] as i128;

        // Compute r0 mod qj
        let r0_mod_qj = ((r0 % qj) + qj) % qj;

        if r0_mod_qj != rj {
            println!("❌ {} CRT INCONSISTENT:", name);
            println!("   r[0] = {} (mod q[0])", r0);
            println!("   r[0] mod q[{}] = {}, but r[{}] = {} (expected equal)",
                     j, r0_mod_qj, j, rj);
            println!("   Difference: {}", (r0_mod_qj - rj).abs());
            return false;
        }
    }

    println!("✅ {} CRT consistent", name);
    true
}

/// Test 1: Check if s² (EVK base) is CRT-consistent
#[test]
fn test_step1_secret_key_squared() {
    println!("\n=== TEST 1: Secret Key Squared ===");

    for num_primes in [3, 4, 5] {
        println!("\n--- Testing with {} primes ---", num_primes);

        let params = match num_primes {
            3 => CliffordFHEParams::new_rns_mult(),
            4 => CliffordFHEParams::new_rns_mult_depth2(),
            5 => CliffordFHEParams::new_rns_mult_depth2_safe(),
            _ => panic!("Invalid"),
        };

        let (_, sk, _) = rns_keygen(&params);

        // Compute s²
        let s_squared = rns_multiply(&sk.coeffs, &sk.coeffs, &params.moduli, polynomial_multiply_ntt);

        let consistent = check_crt_consistency(&s_squared, &params.moduli, "s²");
        assert!(consistent, "s² must be CRT-consistent with {} primes", num_primes);
    }
}

/// Test 2: Check if EVK entries are CRT-consistent
#[test]
fn test_step2_evk_consistency() {
    println!("\n=== TEST 2: EVK Entry Consistency ===");

    for num_primes in [3, 4, 5] {
        println!("\n--- Testing with {} primes ---", num_primes);

        let params = match num_primes {
            3 => CliffordFHEParams::new_rns_mult(),
            4 => CliffordFHEParams::new_rns_mult_depth2(),
            5 => CliffordFHEParams::new_rns_mult_depth2_safe(),
            _ => panic!("Invalid"),
        };

        let (_, _, evk) = rns_keygen(&params);

        println!("EVK has {} digit pairs", evk.evk0.len());

        // Check each EVK entry
        for t in 0..evk.evk0.len() {
            let consistent_0 = check_crt_consistency(&evk.evk0[t], &params.moduli,
                                                      &format!("evk0[{}]", t));
            let consistent_1 = check_crt_consistency(&evk.evk1[t], &params.moduli,
                                                      &format!("evk1[{}]", t));

            assert!(consistent_0, "evk0[{}] inconsistent with {} primes", t, num_primes);
            assert!(consistent_1, "evk1[{}] inconsistent with {} primes", t, num_primes);
        }
    }
}

/// Test 3: Check if gadget decomposition produces CRT-consistent digits
#[test]
fn test_step3_gadget_decomposition() {
    println!("\n=== TEST 3: Gadget Decomposition Consistency ===");

    for num_primes in [3, 4, 5] {
        println!("\n--- Testing with {} primes ---", num_primes);

        let params = match num_primes {
            3 => CliffordFHEParams::new_rns_mult(),
            4 => CliffordFHEParams::new_rns_mult_depth2(),
            5 => CliffordFHEParams::new_rns_mult_depth2_safe(),
            _ => panic!("Invalid"),
        };

        // Create a test polynomial (simulate d2 = c1 × c1)
        let scale_squared = params.scale * params.scale;
        let mut coeffs = vec![0i64; params.n];
        coeffs[0] = (scale_squared / (params.moduli[params.moduli.len() - 1] as f64)).round() as i64;

        let d2 = RnsPolynomial::from_coeffs(&coeffs, &params.moduli, params.n, 0);

        println!("Input d2:");
        check_crt_consistency(&d2, &params.moduli, "d2");

        // Decompose
        let digits = decompose_base_pow2(&d2, &params.moduli, 20);

        println!("Decomposed into {} digits", digits.len());

        // Check each digit
        for (t, digit) in digits.iter().enumerate() {
            let consistent = check_crt_consistency(digit, &params.moduli,
                                                   &format!("digit[{}]", t));
            assert!(consistent, "digit[{}] inconsistent with {} primes", t, num_primes);
        }
    }
}

/// Test 4: Check if digit × EVK multiplication preserves CRT consistency
#[test]
fn test_step4_digit_times_evk() {
    println!("\n=== TEST 4: Digit × EVK Multiplication ===");

    for num_primes in [3, 4, 5] {
        println!("\n--- Testing with {} primes ---", num_primes);

        let params = match num_primes {
            3 => CliffordFHEParams::new_rns_mult(),
            4 => CliffordFHEParams::new_rns_mult_depth2(),
            5 => CliffordFHEParams::new_rns_mult_depth2_safe(),
            _ => panic!("Invalid"),
        };

        let (_, _, evk) = rns_keygen(&params);

        // Create a small test digit (simulating a balanced digit from decomposition)
        let mut digit_coeffs = vec![0i64; params.n];
        digit_coeffs[0] = 12345;  // Small value typical of balanced decomposition
        let digit = RnsPolynomial::from_coeffs(&digit_coeffs, &params.moduli, params.n, 0);

        check_crt_consistency(&digit, &params.moduli, "test digit");

        // Multiply digit by each EVK entry
        for t in 0..evk.evk0.len().min(3) {  // Test first 3 to keep output manageable
            println!("\n  Testing EVK pair {}", t);

            check_crt_consistency(&evk.evk0[t], &params.moduli, &format!("evk0[{}]", t));
            check_crt_consistency(&evk.evk1[t], &params.moduli, &format!("evk1[{}]", t));

            let u0 = rns_multiply(&digit, &evk.evk0[t], &params.moduli, polynomial_multiply_ntt);
            let u1 = rns_multiply(&digit, &evk.evk1[t], &params.moduli, polynomial_multiply_ntt);

            let consistent_u0 = check_crt_consistency(&u0, &params.moduli, &format!("u0 (digit×evk0[{}])", t));
            let consistent_u1 = check_crt_consistency(&u1, &params.moduli, &format!("u1 (digit×evk1[{}])", t));

            assert!(consistent_u0, "u0 from digit×evk0[{}] inconsistent with {} primes", t, num_primes);
            assert!(consistent_u1, "u1 from digit×evk1[{}] inconsistent with {} primes", t, num_primes);
        }
    }
}

/// Test 5: Check if accumulation (add/sub) preserves CRT consistency
#[test]
fn test_step5_accumulation() {
    println!("\n=== TEST 5: Accumulation (Add/Sub) ===");

    for num_primes in [3, 4, 5] {
        println!("\n--- Testing with {} primes ---", num_primes);

        let params = match num_primes {
            3 => CliffordFHEParams::new_rns_mult(),
            4 => CliffordFHEParams::new_rns_mult_depth2(),
            5 => CliffordFHEParams::new_rns_mult_depth2_safe(),
            _ => panic!("Invalid"),
        };

        // Create two consistent test polynomials
        let mut coeffs_a = vec![0i64; params.n];
        let mut coeffs_b = vec![0i64; params.n];
        coeffs_a[0] = 1000000;
        coeffs_b[0] = 500000;

        let poly_a = RnsPolynomial::from_coeffs(&coeffs_a, &params.moduli, params.n, 0);
        let poly_b = RnsPolynomial::from_coeffs(&coeffs_b, &params.moduli, params.n, 0);

        check_crt_consistency(&poly_a, &params.moduli, "poly_a");
        check_crt_consistency(&poly_b, &params.moduli, "poly_b");

        // Test addition
        let sum = rns_add(&poly_a, &poly_b, &params.moduli);
        let consistent_sum = check_crt_consistency(&sum, &params.moduli, "poly_a + poly_b");
        assert!(consistent_sum, "Addition broke CRT with {} primes", num_primes);

        // Test subtraction
        let diff = rns_sub(&poly_a, &poly_b, &params.moduli);
        let consistent_diff = check_crt_consistency(&diff, &params.moduli, "poly_a - poly_b");
        assert!(consistent_diff, "Subtraction broke CRT with {} primes", num_primes);
    }
}

/// Test 6: Full relinearization simulation with detailed step tracking
#[test]
fn test_step6_full_relinearization_simulation() {
    println!("\n=== TEST 6: Full Relinearization Simulation ===");

    for num_primes in [3, 4, 5] {
        println!("\n--- Testing with {} primes ---", num_primes);

        let params = match num_primes {
            3 => CliffordFHEParams::new_rns_mult(),
            4 => CliffordFHEParams::new_rns_mult_depth2(),
            5 => CliffordFHEParams::new_rns_mult_depth2_safe(),
            _ => panic!("Invalid"),
        };

        let (_, _, evk) = rns_keygen(&params);

        // Create realistic d0, d1, d2 (simulating tensor product output)
        let scale_squared = params.scale * params.scale;
        let mut d0_coeffs = vec![0i64; params.n];
        let mut d1_coeffs = vec![0i64; params.n];
        let mut d2_coeffs = vec![0i64; params.n];

        // d0 ≈ constant term, d1 ≈ mixed term, d2 ≈ c1²
        d0_coeffs[0] = (scale_squared / (params.moduli[params.moduli.len() - 1] as f64) * 0.5).round() as i64;
        d1_coeffs[0] = (scale_squared / (params.moduli[params.moduli.len() - 1] as f64) * 0.3).round() as i64;
        d2_coeffs[0] = (scale_squared / (params.moduli[params.moduli.len() - 1] as f64) * 0.2).round() as i64;

        let d0 = RnsPolynomial::from_coeffs(&d0_coeffs, &params.moduli, params.n, 0);
        let d1 = RnsPolynomial::from_coeffs(&d1_coeffs, &params.moduli, params.n, 0);
        let d2 = RnsPolynomial::from_coeffs(&d2_coeffs, &params.moduli, params.n, 0);

        println!("\nInputs:");
        check_crt_consistency(&d0, &params.moduli, "d0");
        check_crt_consistency(&d1, &params.moduli, "d1");
        check_crt_consistency(&d2, &params.moduli, "d2");

        // Decompose d2
        let digits = decompose_base_pow2(&d2, &params.moduli, 20);
        println!("\nDecomposed d2 into {} digits", digits.len());

        // Initialize accumulators
        let mut c0 = d0.clone();
        let mut c1 = d1.clone();

        // Accumulate digit by digit
        for t in 0..digits.len() {
            println!("\n  --- Processing digit {} ---", t);

            let digit = &digits[t];
            check_crt_consistency(digit, &params.moduli, &format!("digit[{}]", t));

            // Multiply by EVK
            let u0 = rns_multiply(digit, &evk.evk0[t], &params.moduli, polynomial_multiply_ntt);
            let u1 = rns_multiply(digit, &evk.evk1[t], &params.moduli, polynomial_multiply_ntt);

            let u0_ok = check_crt_consistency(&u0, &params.moduli, &format!("u0[{}]", t));
            let u1_ok = check_crt_consistency(&u1, &params.moduli, &format!("u1[{}]", t));

            if !u0_ok || !u1_ok {
                println!("❌ FOUND INCONSISTENCY at digit {} with {} primes!", t, num_primes);
                println!("   This is the first operation that breaks CRT consistency.");
                assert!(false, "CRT consistency broken at digit {} with {} primes", t, num_primes);
            }

            // Accumulate
            c0 = rns_sub(&c0, &u0, &params.moduli);
            c1 = rns_add(&c1, &u1, &params.moduli);

            let c0_ok = check_crt_consistency(&c0, &params.moduli, &format!("c0 after digit {}", t));
            let c1_ok = check_crt_consistency(&c1, &params.moduli, &format!("c1 after digit {}", t));

            if !c0_ok || !c1_ok {
                println!("❌ FOUND INCONSISTENCY in accumulation at digit {} with {} primes!", t, num_primes);
                assert!(false, "Accumulation broke CRT at digit {} with {} primes", t, num_primes);
            }
        }

        println!("\n✅ Full relinearization simulation completed successfully with {} primes", num_primes);
    }
}
