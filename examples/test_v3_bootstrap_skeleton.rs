//! Test V3 Bootstrap Skeleton
//!
//! Verifies that V3 module structure compiles and basic operations work.

#[cfg(feature = "v3")]
use ga_engine::clifford_fhe_v3::bootstrapping::{
    BootstrapParams,
    taylor_sin_coeffs,
    eval_polynomial,
    chebyshev_sin_coeffs,
};

fn main() {
    #[cfg(not(feature = "v3"))]
    {
        println!("V3 feature not enabled. Run with: cargo run --example test_v3_bootstrap_skeleton --features v3");
        return;
    }

    #[cfg(feature = "v3")]
    {
        println!("=== V3 Bootstrap Skeleton Test ===\n");

        // Test 1: Bootstrap Parameters
        println!("Phase 1: Testing Bootstrap Parameters...");
        test_bootstrap_params();
        println!("  ✓ Bootstrap parameters working\n");

        // Test 2: Sine Polynomial Approximation
        println!("Phase 2: Testing Sine Approximation...");
        test_sine_approximation();
        println!("  ✓ Sine approximation working\n");

        // Test 3: Polynomial Evaluation
        println!("Phase 3: Testing Polynomial Evaluation...");
        test_polynomial_evaluation();
        println!("  ✓ Polynomial evaluation working\n");

        println!("=== V3 Bootstrap Skeleton Test Complete ===");
        println!("Status: Module structure working, ready for component implementation");
    }
}

#[cfg(feature = "v3")]
fn test_bootstrap_params() {
    // Test preset configurations
    let balanced = BootstrapParams::balanced();
    assert_eq!(balanced.sin_degree, 23);
    assert_eq!(balanced.bootstrap_levels, 12);
    assert_eq!(balanced.target_precision, 1e-4);
    println!("  Balanced params: degree={}, levels={}, precision={}",
             balanced.sin_degree, balanced.bootstrap_levels, balanced.target_precision);

    let conservative = BootstrapParams::conservative();
    assert_eq!(conservative.sin_degree, 31);
    println!("  Conservative params: degree={}", conservative.sin_degree);

    let fast = BootstrapParams::fast();
    assert_eq!(fast.sin_degree, 15);
    println!("  Fast params: degree={}", fast.sin_degree);

    // Test validation
    assert!(balanced.validate().is_ok());
    println!("  Validation: passed");
}

#[cfg(feature = "v3")]
fn test_sine_approximation() {
    use std::f64::consts::PI;

    // Test Taylor series coefficients
    let coeffs = taylor_sin_coeffs(15);
    assert_eq!(coeffs.len(), 16);
    println!("  Taylor coeffs (degree 15): {} terms", coeffs.len());

    // Verify odd function (even powers are zero)
    for i in 0..coeffs.len() {
        if i % 2 == 0 && i > 0 {
            assert_eq!(coeffs[i], 0.0);
        }
    }
    println!("  Odd function property: verified");

    // Test accuracy
    let test_points = vec![0.0, PI / 4.0, PI / 2.0];
    let mut max_error: f64 = 0.0;

    for x in test_points {
        let approx = eval_polynomial(&coeffs, x);
        let exact = x.sin();
        let error = (approx - exact).abs();
        max_error = max_error.max(error);
    }

    println!("  Accuracy: max error = {:.6e}", max_error);
    assert!(max_error < 1e-6, "Taylor approximation error too large");

    // Test Chebyshev
    let coeffs = chebyshev_sin_coeffs(15);
    assert_eq!(coeffs.len(), 16);
    println!("  Chebyshev coeffs (degree 15): {} terms", coeffs.len());
}

#[cfg(feature = "v3")]
fn test_polynomial_evaluation() {
    // Test p(x) = 1 + 2x + 3x²
    let coeffs = vec![1.0, 2.0, 3.0];
    let result = eval_polynomial(&coeffs, 2.0);
    assert_eq!(result, 17.0);  // 1 + 4 + 12
    println!("  p(2) = 1 + 2*2 + 3*4 = {}", result);

    // Test p(x) = x - x³/6
    let coeffs = vec![0.0, 1.0, 0.0, -1.0/6.0];
    let result = eval_polynomial(&coeffs, 1.0);
    let expected = 1.0 - 1.0/6.0;
    assert!((result - expected).abs() < 1e-10);
    println!("  sin(1) approx = 1 - 1/6 = {:.6}", result);
}
