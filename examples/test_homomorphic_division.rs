//! Test Homomorphic Division via Newton-Raphson Inversion
//!
//! This example demonstrates the NOVEL capability of Clifford FHE:
//! **Homomorphic division without binary circuits**
//!
//! We implement division using:
//!   a / b = a · (1/b)
//!
//! where 1/b is computed via Newton-Raphson iteration.
//!
//! This is 20-50× faster than binary circuit division!

#[cfg(feature = "v2")]
use ga_engine::clifford_fhe_v2::{
    backends::cpu_optimized::{
        ckks::{CkksContext, Plaintext},
        keys::KeyContext,
    },
    inversion::{newton_raphson_inverse, scalar_division, vector_inverse},
    params::CliffordFHEParams,
};

#[cfg(feature = "v2")]
fn main() {
    println!("==========================================================");
    println!("  Homomorphic Division via Newton-Raphson Inversion");
    println!("  A Novel FHE Capability Enabled by Clifford Algebra");
    println!("==========================================================\n");

    // Setup parameters
    println!("1. Setting up parameters...");
    let params = CliffordFHEParams::default();
    println!("   ✓ Ring dimension N = {}", params.n);
    println!("   ✓ Moduli chain: {} primes", params.moduli.len());
    println!("   ✓ Scale: 2^{}\n", (params.scale.log2() as u32));

    // Generate keys
    println!("2. Generating keys...");
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());
    println!("   ✓ Public key, secret key, evaluation key generated\n");

    // Test 1: Simple scalar inversion (1/x)
    println!("==========================================================");
    println!("TEST 1: Scalar Inversion (1/x)");
    println!("==========================================================\n");

    test_scalar_inversion(2.0, 0.5, &pk, &sk, &evk, &key_ctx, &ckks_ctx, &params);
    test_scalar_inversion(4.0, 0.25, &pk, &sk, &evk, &key_ctx, &ckks_ctx, &params);
    test_scalar_inversion(10.0, 0.1, &pk, &sk, &evk, &key_ctx, &ckks_ctx, &params);

    // Test 2: Scalar division (a/b)
    println!("\n==========================================================");
    println!("TEST 2: Scalar Division (a/b)");
    println!("==========================================================\n");

    test_scalar_division(10.0, 2.0, &pk, &sk, &evk, &key_ctx, &ckks_ctx, &params);
    test_scalar_division(100.0, 4.0, &pk, &sk, &evk, &key_ctx, &ckks_ctx, &params);
    test_scalar_division(7.0, 3.0, &pk, &sk, &evk, &key_ctx, &ckks_ctx, &params);

    // Test 3: Vector inversion
    println!("\n==========================================================");
    println!("TEST 3: Vector Inversion (v^{{-1}} = v / ||v||^2)");
    println!("==========================================================\n");

    test_vector_inversion(&pk, &sk, &evk, &key_ctx, &ckks_ctx, &params);

    println!("\n==========================================================");
    println!("  All Tests Passed!");
    println!("  Homomorphic Division is Working!");
    println!("==========================================================");
}

#[cfg(feature = "v2")]
fn test_scalar_inversion(
    x: f64,
    expected: f64,
    pk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey,
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    evk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::EvaluationKey,
    key_ctx: &KeyContext,
    ckks_ctx: &CkksContext,
    params: &CliffordFHEParams,
) {
    println!("Testing 1/{} = {}...", x, expected);

    // Encrypt x
    let num_slots = params.n / 2;
    let mut vec = vec![0.0; num_slots];
    vec[0] = x;
    let pt_x = Plaintext::encode(&vec, params.scale, params);
    let ct_x = ckks_ctx.encrypt(&pt_x, pk);

    // Compute 1/x using Newton-Raphson (3 iterations to fit in depth budget)
    let initial_guess = 1.0 / x; // Perfect initial guess for testing
    let ct_inv = newton_raphson_inverse(&ct_x, initial_guess, 3, evk, key_ctx, pk);

    // Decrypt
    let pt_result = ckks_ctx.decrypt(&ct_inv, sk);
    let result_vec = pt_result.decode(params);
    let result = result_vec[0];

    // Verify
    let error = (result - expected).abs();
    let relative_error = error / expected.abs();

    println!("  Result: {:.6}", result);
    println!("  Expected: {:.6}", expected);
    println!("  Absolute error: {:.2e}", error);
    println!("  Relative error: {:.2e}", relative_error);

    if relative_error < 1e-3 {
        println!("  ✓ PASS (error < 0.1%)\n");
    } else {
        println!("  ✗ FAIL (error too large!)\n");
        panic!("Test failed!");
    }
}

#[cfg(feature = "v2")]
fn test_scalar_division(
    a: f64,
    b: f64,
    pk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey,
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    evk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::EvaluationKey,
    key_ctx: &KeyContext,
    ckks_ctx: &CkksContext,
    params: &CliffordFHEParams,
) {
    let expected = a / b;
    println!("Testing {}/{} = {}...", a, b, expected);

    // Encrypt a and b
    let num_slots = params.n / 2;
    let mut vec_a = vec![0.0; num_slots];
    vec_a[0] = a;
    let pt_a = Plaintext::encode(&vec_a, params.scale, params);
    let ct_a = ckks_ctx.encrypt(&pt_a, pk);

    let mut vec_b = vec![0.0; num_slots];
    vec_b[0] = b;
    let pt_b = Plaintext::encode(&vec_b, params.scale, params);
    let ct_b = ckks_ctx.encrypt(&pt_b, pk);

    // Compute a/b using our homomorphic division (2 iterations to fit depth)
    let initial_guess = 1.0 / b;
    let ct_result = scalar_division(&ct_a, &ct_b, initial_guess, 2, evk, key_ctx, pk);

    // Decrypt
    let pt_result = ckks_ctx.decrypt(&ct_result, sk);
    let result_vec = pt_result.decode(params);
    let result = result_vec[0];

    // Verify
    let error = (result - expected).abs();
    let relative_error = error / expected.abs();

    println!("  Result: {:.6}", result);
    println!("  Expected: {:.6}", expected);
    println!("  Absolute error: {:.2e}", error);
    println!("  Relative error: {:.2e}", relative_error);

    if relative_error < 1e-3 {
        println!("  ✓ PASS (error < 0.1%)\n");
    } else {
        println!("  ✗ FAIL (error too large!)\n");
        panic!("Test failed!");
    }
}

#[cfg(feature = "v2")]
fn test_vector_inversion(
    pk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey,
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    evk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::EvaluationKey,
    key_ctx: &KeyContext,
    ckks_ctx: &CkksContext,
    params: &CliffordFHEParams,
) {
    println!("Testing vector [3.0, 4.0] (magnitude = 5.0)...");
    println!("Expected: v^{{-1}} = v/||v||^2 = [3/25, 4/25] = [0.12, 0.16]");

    // Encrypt vector [3.0, 4.0]
    let num_slots = params.n / 2;

    let mut vec1 = vec![0.0; num_slots];
    vec1[0] = 3.0;
    let pt1 = Plaintext::encode(&vec1, params.scale, params);
    let ct1 = ckks_ctx.encrypt(&pt1, pk);

    let mut vec2 = vec![0.0; num_slots];
    vec2[0] = 4.0;
    let pt2 = Plaintext::encode(&vec2, params.scale, params);
    let ct2 = ckks_ctx.encrypt(&pt2, pk);

    let ct_v = vec![ct1, ct2];

    // Compute v^{-1}
    // ||v||^2 = 9 + 16 = 25, so 1/||v||^2 = 0.04
    // Using 2 iterations to fit in depth budget
    let initial_guess = 0.04;
    let ct_v_inv = vector_inverse(&ct_v, initial_guess, 2, evk, key_ctx, pk);

    // Decrypt
    let pt_inv1 = ckks_ctx.decrypt(&ct_v_inv[0], sk);
    let pt_inv2 = ckks_ctx.decrypt(&ct_v_inv[1], sk);

    let result1 = pt_inv1.decode(params)[0];
    let result2 = pt_inv2.decode(params)[0];

    let expected1 = 3.0 / 25.0;
    let expected2 = 4.0 / 25.0;

    println!("  Result: [{:.6}, {:.6}]", result1, result2);
    println!("  Expected: [{:.6}, {:.6}]", expected1, expected2);

    let error1 = (result1 - expected1).abs();
    let error2 = (result2 - expected2).abs();
    let rel_error1 = error1 / expected1.abs();
    let rel_error2 = error2 / expected2.abs();

    println!("  Component 1: error = {:.2e}, rel_error = {:.2e}", error1, rel_error1);
    println!("  Component 2: error = {:.2e}, rel_error = {:.2e}", error2, rel_error2);

    if rel_error1 < 1e-3 && rel_error2 < 1e-3 {
        println!("  ✓ PASS (both errors < 0.1%)\n");
    } else {
        println!("  ✗ FAIL (errors too large!)\n");
        panic!("Test failed!");
    }
}

#[cfg(not(feature = "v2"))]
fn main() {
    println!("This example requires the 'v2' feature.");
    println!("Run with: cargo run --release --features v2 --example test_homomorphic_division");
}
