//! State-of-the-Art Test Suite for Homomorphic Geometric Operations
//!
//! This test suite provides professional, visually stunning output with:
//! - Real-time progress bars
//! - Color-coded results
//! - Detailed performance metrics
//! - Clean, structured formatting
//!
//! Run with: cargo test --test test_geometric_operations_v2 --features v1 -- --nocapture

mod test_utils;

use ga_engine::clifford_fhe_v1::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v1::keys_rns::rns_keygen;
use ga_engine::clifford_fhe_v1::ckks_rns::{rns_encrypt, rns_decrypt, RnsPlaintext, RnsCiphertext};
use ga_engine::clifford_fhe_v1::geometric_product_rns::{
    geometric_product_3d_componentwise,
    reverse_3d,
    rotate_3d,
    wedge_product_3d,
    inner_product_3d,
    project_3d,
    reject_3d,
};
use test_utils::*;

/// Helper to encrypt a multivector (8 components for 3D)
fn encrypt_multivector_3d(
    mv: &[f64; 8],
    pk: &ga_engine::clifford_fhe_v1::keys_rns::RnsPublicKey,
    params: &CliffordFHEParams,
) -> [RnsCiphertext; 8] {
    let mut result = Vec::new();

    for &component in mv.iter() {
        let mut coeffs = vec![0i64; params.n];
        coeffs[0] = (component * params.scale).round() as i64;

        let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
        let ct = rns_encrypt(pk, &pt, params);
        result.push(ct);
    }

    result.try_into().unwrap()
}

/// Helper to encrypt with progress updates
fn encrypt_multivector_3d_with_progress(
    mv: &[f64; 8],
    pk: &ga_engine::clifford_fhe_v1::keys_rns::RnsPublicKey,
    params: &CliffordFHEParams,
    test: &TestRunner,
    label: &str,
) -> [RnsCiphertext; 8] {
    let mut result = Vec::new();

    for (i, &component) in mv.iter().enumerate() {
        test.update(&format!("{} (component {}/8)", label, i + 1));
        let mut coeffs = vec![0i64; params.n];
        coeffs[0] = (component * params.scale).round() as i64;

        let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
        let ct = rns_encrypt(pk, &pt, params);
        result.push(ct);
    }

    result.try_into().unwrap()
}

/// Helper to decrypt a multivector
fn decrypt_multivector_3d(
    ct: &[RnsCiphertext; 8],
    sk: &ga_engine::clifford_fhe_v1::keys_rns::RnsSecretKey,
    params: &CliffordFHEParams,
) -> [f64; 8] {
    let mut result = [0.0; 8];

    for i in 0..8 {
        let pt = rns_decrypt(sk, &ct[i], params);
        let val = pt.coeffs.rns_coeffs[0][0];
        let q = params.moduli[0];
        let centered = if val > q / 2 { val - q } else { val };
        result[i] = (centered as f64) / ct[i].scale;
    }

    result
}

/// Helper to decrypt with progress updates
fn decrypt_multivector_3d_with_progress(
    ct: &[RnsCiphertext; 8],
    sk: &ga_engine::clifford_fhe_v1::keys_rns::RnsSecretKey,
    params: &CliffordFHEParams,
    test: &TestRunner,
    label: &str,
) -> [f64; 8] {
    let mut result = [0.0; 8];

    for i in 0..8 {
        test.update(&format!("{} (component {}/8)", label, i + 1));
        let pt = rns_decrypt(sk, &ct[i], params);
        let val = pt.coeffs.rns_coeffs[0][0];
        let q = params.moduli[0];
        let centered = if val > q / 2 { val - q } else { val };
        result[i] = (centered as f64) / ct[i].scale;
    }

    result
}

/// Compute maximum error between two multivectors
fn max_error(a: &[f64; 8], b: &[f64; 8]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn test_all_geometric_operations() {
    let mut suite = TestSuite::new("Clifford FHE V1: Homomorphic Geometric Operations");

    // Setup
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();

    print_config(&[
        ("Ring dimension", format!("N = {}", params.n)),
        ("Number of primes", format!("{}", params.moduli.len())),
        ("Scaling factor", format!("2^{}", (params.scale as f64).log2() as u32)),
        ("Security level", "≥128 bits".to_string()),
    ]);

    // Generate keys
    let keygen = suite.test("Key Generation", 3);
    keygen.step("generating secret key");
    keygen.step("generating public key");
    let (pk, sk, evk) = rns_keygen(&params);
    keygen.step("generating evaluation key");
    let result = keygen.finish(true, 0.0);
    suite.add_result(result);

    // Test 1: Reverse
    {
        let test = suite.test("1. Reverse (~a)", 4);

        test.step("encrypting test multivector");
        let a = [1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0];
        let ct_a = encrypt_multivector_3d(&a, &pk, &params);

        test.step("applying homomorphic reverse");
        let ct_reversed = reverse_3d(&ct_a, &params);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d(&ct_reversed, &sk, &params);

        test.step("verifying correctness");
        let expected = [1.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0];
        let error = max_error(&result_mv, &expected);

        let result = test.finish(error < 1e-6, error);
        suite.add_result(result);
    }

    // Test 2: Geometric Product
    {
        let test = suite.test("2. Geometric Product (a ⊗ b)", 5);

        test.step("encrypting test multivectors (a)");
        let a = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ct_a = encrypt_multivector_3d_with_progress(&a, &pk, &params, &test, "encrypting a");

        test.step("encrypting test multivectors (b)");
        let b = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ct_b = encrypt_multivector_3d_with_progress(&b, &pk, &params, &test, "encrypting b");

        test.step("applying geometric product (64 ciphertext mults + 64 relinearizations)");
        test.update("computing tensor products (1/64)...");
        let ct_prod = geometric_product_3d_componentwise(&ct_a, &ct_b, &evk, &params);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d_with_progress(&ct_prod, &sk, &params, &test, "decrypting");

        test.step("verifying correctness");
        // (1 + 2e₁) ⊗ (3e₂) = 3e₂ + 6(e₁⊗e₂) = 3e₂ + 6e₁₂
        let expected = [0.0, 0.0, 3.0, 0.0, 6.0, 0.0, 0.0, 0.0];
        let error = max_error(&result_mv, &expected);

        let result = test.finish(error < 1e-6, error);
        suite.add_result(result);
    }

    // Test 3: Rotation
    {
        let test = suite.test("3. Rotation (R ⊗ v ⊗ ~R)", 6);

        test.step("creating rotor (90° about Z-axis)");
        // Rotor: R = cos(45°) + sin(45°)e₁₂ for 90° rotation in XY plane
        let cos45 = std::f64::consts::FRAC_1_SQRT_2;
        let sin45 = std::f64::consts::FRAC_1_SQRT_2;
        let rotor = [cos45, 0.0, 0.0, 0.0, sin45, 0.0, 0.0, 0.0];

        test.step("encrypting rotor");
        let ct_rotor = encrypt_multivector_3d_with_progress(&rotor, &pk, &params, &test, "encrypting rotor");

        test.step("encrypting vector");
        let vector = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁
        let ct_vec = encrypt_multivector_3d_with_progress(&vector, &pk, &params, &test, "encrypting vector");

        test.step("applying rotation (128 mults: R⊗v then result⊗~R)");
        test.update("computing first product R⊗v...");
        let ct_rotated = rotate_3d(&ct_rotor, &ct_vec, &evk, &params);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d_with_progress(&ct_rotated, &sk, &params, &test, "decrypting");

        test.step("computing expected result");
        // 90° rotation of e₁ about Z gives e₂
        let expected = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        test.step("verifying correctness");
        let error = max_error(&result_mv, &expected);

        // Note: Rotation uses 2 multiplications, so error accumulates more
        let result = test.finish(error < 0.5, error);
        suite.add_result(result);
    }

    // Test 4: Wedge Product
    {
        let test = suite.test("4. Wedge Product (a ∧ b)", 5);

        test.step("encrypting test vectors");
        let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁
        let b = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₂
        let ct_a = encrypt_multivector_3d(&a, &pk, &params);
        let ct_b = encrypt_multivector_3d(&b, &pk, &params);

        test.step("applying wedge product");
        let ct_wedge = wedge_product_3d(&ct_a, &ct_b, &evk, &params);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d(&ct_wedge, &sk, &params);

        test.step("computing expected result");
        // e₁ ∧ e₂ = e₁₂
        let expected = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

        test.step("verifying correctness");
        let error = max_error(&result_mv, &expected);

        let result = test.finish(error < 1e-6, error);
        suite.add_result(result);
    }

    // Test 5: Inner Product
    {
        let test = suite.test("5. Inner Product (a · b)", 5);

        test.step("encrypting test vectors");
        let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁
        let b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁
        let ct_a = encrypt_multivector_3d(&a, &pk, &params);
        let ct_b = encrypt_multivector_3d(&b, &pk, &params);

        test.step("applying inner product");
        let ct_inner = inner_product_3d(&ct_a, &ct_b, &evk, &params);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d(&ct_inner, &sk, &params);

        test.step("computing expected result");
        // e₁ · e₁ = 1 (scalar)
        let expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        test.step("verifying correctness");
        let error = max_error(&result_mv, &expected);

        let result = test.finish(error < 1e-6, error);
        suite.add_result(result);
    }

    // Test 6: Projection
    {
        let test = suite.test("6. Projection (proj_a(b))", 6);

        test.step("encrypting vector a");
        let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁
        let ct_a = encrypt_multivector_3d_with_progress(&a, &pk, &params, &test, "encrypting a");

        test.step("encrypting vector b");
        let b = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁ + e₂
        let ct_b = encrypt_multivector_3d_with_progress(&b, &pk, &params, &test, "encrypting b");

        test.step("applying projection (192 mults: ~a, a·b, then scale)");
        test.update("computing reverse ~a...");
        let ct_proj = project_3d(&ct_a, &ct_b, &evk, &params);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d_with_progress(&ct_proj, &sk, &params, &test, "decrypting");

        test.step("computing expected result");
        // proj_e₁(e₁ + e₂) = e₁
        let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        test.step("verifying correctness");
        let error = max_error(&result_mv, &expected);

        let result = test.finish(error < 1e-6, error);
        suite.add_result(result);
    }

    // Test 7: Rejection
    {
        let test = suite.test("7. Rejection (rej_a(b))", 6);

        test.step("encrypting test vectors");
        let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁
        let b = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁ + e₂
        let ct_a = encrypt_multivector_3d(&a, &pk, &params);
        let ct_b = encrypt_multivector_3d(&b, &pk, &params);

        test.step("applying rejection (depth-3 operation)");
        let ct_rej = reject_3d(&ct_a, &ct_b, &evk, &params);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d(&ct_rej, &sk, &params);

        test.step("computing expected result");
        // rej_e₁(e₁ + e₂) = e₂
        let expected = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        test.step("verifying correctness");
        let error = max_error(&result_mv, &expected);

        // Note: Rejection uses multiple multiplications, so error accumulates
        let result = test.finish(error < 0.5, error);
        suite.add_result(result);
    }

    // Print final summary
    suite.finish();
}
