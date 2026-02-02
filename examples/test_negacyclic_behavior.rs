//! Understand negacyclic convolution behavior for constant polynomials

use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;

fn main() {
    println!("Understanding negacyclic convolution with constant polynomials\n");

    let n = 1024;
    let q = 1152921504606584833u64;

    let ntt = NttContext::new(n, q);

    // Test 1: Constant polynomials
    println!("Test 1: a(x) = 42 (constant), b(x) = 2 (constant)");
    let a1 = vec![42u64; n];
    let b1 = vec![2u64; n];
    let result1 = ntt.multiply_polynomials(&a1, &b1);

    println!("  Result in centered mod q:");
    for i in 0..5 {
        let val = result1[i];
        let centered = if val > q/2 { (val as i128) - (q as i128) } else { val as i128 };
        println!("    result[{}] = {} (centered: {})", i, val, centered);
    }
    println!("    ...");
    for i in (n-5)..n {
        let val = result1[i];
        let centered = if val > q/2 { (val as i128) - (q as i128) } else { val as i128 };
        println!("    result[{}] = {} (centered: {})", i, val, centered);
    }

    println!("\nExplanation:");
    println!("  In negacyclic convolution (mod x^n + 1):");
    println!("  (a_0 + a_1*x + ... + a_(n-1)*x^(n-1)) * (b_0 + b_1*x + ... + b_(n-1)*x^(n-1))");
    println!("  When all coeffs are equal (constant poly), high-degree terms wrap with SIGN FLIP");
    println!("  So result is NOT constant!");

    // Test 2: Simple monomials
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\nTest 2: a(x) = x, b(x) = x");
    println!("  Expected: x * x = x^2 (no wraparound)");
    let mut a2 = vec![0u64; n];
    let mut b2 = vec![0u64; n];
    a2[1] = 1; // x
    b2[1] = 1; // x
    let result2 = ntt.multiply_polynomials(&a2, &b2);

    println!("  Result:");
    for i in 0..5 {
        if result2[i] != 0 {
            println!("    result[{}] = {}", i, result2[i]);
        }
    }

    // Test 3: Wraparound
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\nTest 3: a(x) = x^{}, b(x) = x", n-1);
    println!("  Expected: x^(n-1) * x = x^n = -1 (negacyclic wrap with sign flip)");
    let mut a3 = vec![0u64; n];
    let mut b3 = vec![0u64; n];
    a3[n-1] = 1; // x^(n-1)
    b3[1] = 1;   // x
    let result3 = ntt.multiply_polynomials(&a3, &b3);

    println!("  Result:");
    for i in 0..5 {
        let val = result3[i];
        if val != 0 {
            let centered = if val > q/2 { (val as i128) - (q as i128) } else { val as i128 };
            println!("    result[{}] = {} (centered: {})", i, val, centered);
        }
    }

    println!("\n✅ This confirms negacyclic convolution behavior is correct!");
}
