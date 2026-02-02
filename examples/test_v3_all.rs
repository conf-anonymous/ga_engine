//! Comprehensive V3 Test Suite
//!
//! Validates all V3 components implemented so far:
//! - Phase 1: Bootstrap foundation (sin approx, mod raise, bootstrap context)
//! - Phase 2: V3 parameters + rotation keys

#[cfg(feature = "v3")]
use ga_engine::clifford_fhe_v3::bootstrapping::{
    BootstrapParams,
    taylor_sin_coeffs,
    eval_polynomial,
    chebyshev_sin_coeffs,
    galois_element_for_rotation,
    required_rotations_for_bootstrap,
    generate_rotation_keys,
};

#[cfg(feature = "v3")]
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

#[cfg(feature = "v3")]
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

fn main() {
    #[cfg(not(feature = "v3"))]
    {
        println!("V3 feature not enabled. Run with: cargo run --example test_v3_all --features v3");
        return;
    }

    #[cfg(feature = "v3")]
    {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║         V3 Comprehensive Test Suite - Phase 1 & 2             ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();

        let mut passed = 0;
        let mut total = 0;

        // Phase 1 Tests
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Phase 1: Bootstrap Foundation");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        total += 1;
        if test_bootstrap_params() {
            println!("  ✓ Test 1: Bootstrap Parameters\n");
            passed += 1;
        } else {
            println!("  ✗ Test 1: Bootstrap Parameters FAILED\n");
        }

        total += 1;
        if test_sine_approximation() {
            println!("  ✓ Test 2: Sine Approximation\n");
            passed += 1;
        } else {
            println!("  ✗ Test 2: Sine Approximation FAILED\n");
        }

        total += 1;
        if test_polynomial_evaluation() {
            println!("  ✓ Test 3: Polynomial Evaluation\n");
            passed += 1;
        } else {
            println!("  ✗ Test 3: Polynomial Evaluation FAILED\n");
        }

        // Phase 2 Tests
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Phase 2: V3 Parameters & Rotation Keys");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        total += 1;
        if test_v3_parameters() {
            println!("  ✓ Test 4: V3 Parameter Sets\n");
            passed += 1;
        } else {
            println!("  ✗ Test 4: V3 Parameter Sets FAILED\n");
        }

        total += 1;
        if test_galois_elements() {
            println!("  ✓ Test 5: Galois Elements\n");
            passed += 1;
        } else {
            println!("  ✗ Test 5: Galois Elements FAILED\n");
        }

        total += 1;
        if test_required_rotations() {
            println!("  ✓ Test 6: Required Rotations\n");
            passed += 1;
        } else {
            println!("  ✗ Test 6: Required Rotations FAILED\n");
        }

        total += 1;
        if test_rotation_keys() {
            println!("  ✓ Test 7: Rotation Key Structure\n");
            passed += 1;
        } else {
            println!("  ✗ Test 7: Rotation Key Structure FAILED\n");
        }

        // Summary
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Test Summary");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();
        println!("  Total Tests:  {}", total);
        println!("  Passed:       {} ✓", passed);
        println!("  Failed:       {}", total - passed);
        println!();

        if passed == total {
            println!("╔════════════════════════════════════════════════════════════════╗");
            println!("║              ✓ ALL TESTS PASSED - V3 VALIDATED ✓              ║");
            println!("╚════════════════════════════════════════════════════════════════╝");
        } else {
            println!("╔════════════════════════════════════════════════════════════════╗");
            println!("║                  ✗ SOME TESTS FAILED ✗                        ║");
            println!("╚════════════════════════════════════════════════════════════════╝");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "v3")]
fn test_bootstrap_params() -> bool {
    println!("Test 1: Bootstrap Parameters");

    // Test presets
    let balanced = BootstrapParams::balanced();
    if balanced.sin_degree != 23 { return false; }
    if balanced.bootstrap_levels != 12 { return false; }
    println!("    Balanced preset: degree={}, levels={}", balanced.sin_degree, balanced.bootstrap_levels);

    let conservative = BootstrapParams::conservative();
    if conservative.sin_degree != 31 { return false; }
    println!("    Conservative preset: degree={}", conservative.sin_degree);

    let fast = BootstrapParams::fast();
    if fast.sin_degree != 15 { return false; }
    println!("    Fast preset: degree={}", fast.sin_degree);

    // Test validation
    if balanced.validate().is_err() { return false; }
    println!("    Validation: passed");

    true
}

#[cfg(feature = "v3")]
fn test_sine_approximation() -> bool {
    use std::f64::consts::PI;

    println!("Test 2: Sine Approximation");

    // Test Taylor coefficients
    let coeffs = taylor_sin_coeffs(15);
    if coeffs.len() != 16 { return false; }
    println!("    Taylor coefficients: {} terms", coeffs.len());

    // Verify odd function
    for i in 0..coeffs.len() {
        if i % 2 == 0 && i > 0 {
            if coeffs[i] != 0.0 { return false; }
        }
    }
    println!("    Odd function property: verified");

    // Test accuracy
    let test_points = vec![0.0, PI / 4.0, PI / 2.0];
    let mut max_error: f64 = 0.0;

    for x in test_points {
        let approx = eval_polynomial(&coeffs, x);
        let exact = x.sin();
        let error = (approx - exact).abs();
        max_error = max_error.max(error);
    }

    println!("    Accuracy: max error = {:.6e}", max_error);
    if max_error >= 1e-6 { return false; }

    // Test Chebyshev
    let coeffs = chebyshev_sin_coeffs(15);
    if coeffs.len() != 16 { return false; }
    println!("    Chebyshev coefficients: {} terms", coeffs.len());

    true
}

#[cfg(feature = "v3")]
fn test_polynomial_evaluation() -> bool {
    println!("Test 3: Polynomial Evaluation");

    // Test p(x) = 1 + 2x + 3x²
    let coeffs = vec![1.0, 2.0, 3.0];
    let result = eval_polynomial(&coeffs, 2.0);
    if result != 17.0 { return false; }
    println!("    p(2) = 1 + 2*2 + 3*4 = {}", result);

    // Test p(x) = x - x³/6
    let coeffs = vec![0.0, 1.0, 0.0, -1.0/6.0];
    let result = eval_polynomial(&coeffs, 1.0);
    let expected = 1.0 - 1.0/6.0;
    if (result - expected).abs() >= 1e-10 { return false; }
    println!("    sin(1) approx = 1 - 1/6 = {:.6}", result);

    true
}

#[cfg(feature = "v3")]
fn test_v3_parameters() -> bool {
    println!("Test 4: V3 Parameter Sets");

    // Test V3 Bootstrap 8192
    let params = CliffordFHEParams::new_v3_bootstrap_8192();
    if params.n != 8192 { return false; }
    if params.moduli.len() != 22 { return false; }
    println!("    V3 Bootstrap 8192: N={}, {} primes", params.n, params.moduli.len());

    // Verify NTT-friendly
    let two_n = 2 * params.n as u64;
    for &q in &params.moduli {
        if (q - 1) % two_n != 0 { return false; }
    }
    println!("    All primes NTT-friendly: ✓");

    // Test computation levels
    let comp_levels = params.computation_levels(12);
    if comp_levels != 9 { return false; }
    println!("    Computation levels (12 bootstrap): {}", comp_levels);

    // Test V3 Bootstrap 16384
    let params = CliffordFHEParams::new_v3_bootstrap_16384();
    if params.n != 16384 { return false; }
    if params.moduli.len() != 25 { return false; }
    println!("    V3 Bootstrap 16384: N={}, {} primes", params.n, params.moduli.len());

    // Test Minimal
    let params = CliffordFHEParams::new_v3_bootstrap_minimal();
    if params.n != 8192 { return false; }
    if params.moduli.len() != 20 { return false; }
    if !params.supports_bootstrap(12) { return false; }
    println!("    V3 Minimal: N={}, {} primes", params.n, params.moduli.len());

    true
}

#[cfg(feature = "v3")]
fn test_galois_elements() -> bool {
    println!("Test 5: Galois Elements");

    let n = 8192;

    // Test specific values
    let g0 = galois_element_for_rotation(0, n);
    if g0 != 1 { return false; }
    println!("    Rotation 0: g = {} (identity)", g0);

    let g1 = galois_element_for_rotation(1, n);
    if g1 != 5 { return false; }
    println!("    Rotation 1: g = {} (5^1 mod {})", g1, 2*n);

    let g2 = galois_element_for_rotation(2, n);
    if g2 != 25 { return false; }
    println!("    Rotation 2: g = {} (5^2 mod {})", g2, 2*n);

    // Test negative rotation
    let g_neg1 = galois_element_for_rotation(-1, n);
    println!("    Rotation -1: g = {} (inverse)", g_neg1);

    true
}

#[cfg(feature = "v3")]
fn test_required_rotations() -> bool {
    println!("Test 6: Required Rotations for Bootstrap");

    let n = 8192;
    let rotations = required_rotations_for_bootstrap(n);

    let expected = 2 * ((n as f64).log2() as usize);
    if rotations.len() != expected { return false; }
    println!("    N = {}, rotations = {} (2 * log2(N))", n, rotations.len());

    // Verify specific rotations
    if !rotations.contains(&1) { return false; }
    if !rotations.contains(&-1) { return false; }
    if !rotations.contains(&(n as i32 / 2)) { return false; }
    if !rotations.contains(&(-(n as i32 / 2))) { return false; }
    println!("    Contains: ±1, ±2, ±4, ..., ±{}", n/2);

    true
}

#[cfg(feature = "v3")]
fn test_rotation_keys() -> bool {
    println!("Test 7: Rotation Key Structure");

    // Use smaller parameters for faster test
    let params = CliffordFHEParams::new_128bit();
    let key_ctx = KeyContext::new(params.clone());
    let (_, secret_key, _) = key_ctx.keygen();
    println!("    Generated keys: N={}, {} primes", params.n, params.moduli.len());

    // Generate rotation keys for small set
    let rotations = vec![1, 2, 4];
    let rotation_keys = generate_rotation_keys(&rotations, &secret_key, &params);

    if rotation_keys.num_keys() != rotations.len() { return false; }
    println!("    Generated {} rotation keys (placeholder)", rotation_keys.num_keys());

    // Verify keys exist
    for &k in &rotations {
        let g = galois_element_for_rotation(k, params.n);
        if !rotation_keys.has_key(g) { return false; }
    }
    println!("    All keys accessible by Galois element: ✓");

    true
}
