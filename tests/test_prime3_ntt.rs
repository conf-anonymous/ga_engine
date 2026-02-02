// Test NTT specifically for prime[3]
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]


use ga_engine::clifford_fhe::ckks_rns::polynomial_multiply_ntt;
use ga_engine::clifford_fhe::params::CliffordFHEParams;

#[test]
fn test_prime3_ntt_simple() {
    // Use the actual q[3] from params (now fixed to be a real prime!)
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let q = params.moduli[3];  // q[3] = 1099511795713 (was 1099511693313 which was composite!)
    let n = params.n;

    println!("Testing NTT for q = {} with N = {}", q, n);

    // Verify q ≡ 1 (mod 2N)
    let two_n = (2 * n) as i64;
    println!("q mod 2N = {} (should be 1)", q % two_n);

    if q % two_n != 1 {
        panic!("Prime is not NTT-friendly!");
    }

    // Test simple multiplication: [1, 0, 0, ...] × [1, 0, 0, ...] = [1, 0, 0, ...]
    let mut a = vec![0i64; n];
    let mut b = vec![0i64; n];
    a[0] = 1;
    b[0] = 1;

    let result = polynomial_multiply_ntt(&a, &b, q, n);

    println!("\nTest 1: 1 × 1");
    println!("  result[0] = {} (expected 1)", result[0]);
    assert_eq!(result[0], 1, "1 × 1 should give 1");

    // Test: [2, 0, ...] × [3, 0, ...] = [6, 0, ...]
    a[0] = 2;
    b[0] = 3;

    let result = polynomial_multiply_ntt(&a, &b, q, n);

    println!("\nTest 2: 2 × 3");
    println!("  result[0] = {} (expected 6)", result[0]);
    assert_eq!(result[0], 6, "2 × 3 should give 6");

    // Test: [0, 1, 0, ...] × [0, 1, 0, ...] = [0, 0, -1, 0, ...] (since X² × X² = X⁴, and for large enough N this is X⁴)
    // Actually for X^N + 1, we have: X^N = -1
    // So X^(N-1) × X^(N-1) = X^(2N-2), and we need to reduce mod X^N + 1

    let mut a = vec![0i64; n];
    let mut b = vec![0i64; n];
    a[n-1] = 1;  // X^{N-1}
    b[n-1] = 1;  // X^{N-1}

    let result = polynomial_multiply_ntt(&a, &b, q, n);

    println!("\nTest 3: X^{{N-1}} × X^{{N-1}} = X^{{2N-2}}");
    println!("  In ring Z[X]/(X^N + 1): X^{{2N-2}} = X^{{2N-2 mod N}} × (-1)^{{(2N-2)/N}}");
    println!("  = X^{{N-2}} × (-1)^1 = -X^{{N-2}}");
    println!("  result[N-2] = {} (expected {})", result[n-2], q - 1);  // -1 mod q = q-1

    assert_eq!(result[n-2], q - 1, "X^(N-1) × X^(N-1) should give -X^(N-2)");

    println!("\n✅ All basic NTT tests passed for prime[3]!");
}
