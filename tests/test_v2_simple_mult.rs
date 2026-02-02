//! Ultra-simple test: multiply constant polynomials
//!
//! Run with: cargo test --test test_v2_simple_mult --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;

#[test]
fn test_multiply_constants() {
    println!("\n========== SIMPLE POLYNOMIAL MULTIPLICATION TEST ==========");

    let n = 1024;
    let q = 1152921504606584833u64; // First prime from params

    let ntt_ctx = NttContext::new(n, q);

    // Test 1: Multiply constants 3 and 4
    println!("\nTest 1: (3) * (4) = (12)");
    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    a[0] = 3;
    b[0] = 4;

    let result = ntt_ctx.multiply_polynomials(&a, &b);

    println!("  a[0] = {}", a[0]);
    println!("  b[0] = {}", b[0]);
    println!("  result[0] = {}", result[0]);
    println!("  result[1] = {}", result[1]);
    println!("  result[2] = {}", result[2]);

    assert_eq!(result[0], 12, "Constant multiplication failed!");
    println!("  ✓ PASS");

    // Test 2: Multiply (1 + x) * (1 + x)
    println!("\nTest 2: (1 + x) * (1 + x) = (1 + 2x + x^2)");
    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    a[0] = 1;
    a[1] = 1;
    b[0] = 1;
    b[1] = 1;

    let result = ntt_ctx.multiply_polynomials(&a, &b);

    println!("  result[0] = {} (expected 1)", result[0]);
    println!("  result[1] = {} (expected 2)", result[1]);
    println!("  result[2] = {} (expected 1)", result[2]);

    assert_eq!(result[0], 1, "Coeff 0 wrong!");
    assert_eq!(result[1], 2, "Coeff 1 wrong!");
    assert_eq!(result[2], 1, "Coeff 2 wrong!");
    println!("  ✓ PASS");

    // Test 3: Check negacyclic property: x^n = -1
    // So x^(n-1) * x = -1
    println!("\nTest 3: x^(n-1) * x = -1 (mod x^n + 1)");
    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    a[n-1] = 1; // x^(n-1)
    b[1] = 1;    // x

    let result = ntt_ctx.multiply_polynomials(&a, &b);

    let expected = q - 1; // -1 mod q
    println!("  result[0] = {} (expected {} for -1 mod q)", result[0], expected);

    assert_eq!(result[0], expected, "Negacyclic property failed!");
    println!("  ✓ PASS");

    println!("\n All tests passed! NTT polynomial multiplication is correct.");
}
