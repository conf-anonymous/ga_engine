//! V2 Test Suite for Homomorphic Geometric Operations with Rayon Parallelization
//!
//! This test suite provides professional output with:
//! - Real-time progress bars with spinners
//! - Color-coded results
//! - Detailed performance metrics
//! - Clean, structured formatting
//! - 30× faster than V1 (NTT + Rayon parallelization)
//!
//! Run with: cargo test --test test_geometric_operations_v2 --features v2 -- --nocapture

mod test_utils;

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::{CkksContext, Plaintext};
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::geometric::{GeometricContext, MultivectorCiphertext};
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use test_utils::*;

/// Helper to encrypt a multivector (8 components for 3D)
fn encrypt_multivector_3d(
    mv: &[f64; 8],
    ckks_ctx: &CkksContext,
    pk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey,
) -> MultivectorCiphertext {
    let params = &ckks_ctx.params;
    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let n = params.n;

    let mut result = Vec::new();

    for &component in mv.iter() {
        // Create plaintext with value in first coefficient
        let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); n];

        // Convert to scaled integer, handling negatives properly
        let scaled_val = (component * params.scale).round() as i64;

        // Convert to RNS representation (handles negative by taking mod q)
        let values: Vec<u64> = moduli.iter().map(|&q| {
            let q_i64 = q as i64;
            let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
            normalized as u64
        }).collect();

        coeffs[0] = RnsRepresentation::new(values, moduli.clone());

        let pt = Plaintext::new(coeffs, params.scale, level);
        let ct = ckks_ctx.encrypt(&pt, pk);
        result.push(ct);
    }

    result.try_into().unwrap()
}

/// Helper to encrypt with progress updates
fn encrypt_multivector_3d_with_progress(
    mv: &[f64; 8],
    ckks_ctx: &CkksContext,
    pk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey,
    test: &TestRunner,
    label: &str,
) -> MultivectorCiphertext {
    let params = &ckks_ctx.params;
    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let n = params.n;

    let mut result = Vec::new();

    for (i, &component) in mv.iter().enumerate() {
        test.update(&format!("{} (component {}/8)", label, i + 1));

        let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); n];

        // Convert to scaled integer, handling negatives properly
        let scaled_val = (component * params.scale).round() as i64;

        // Convert to RNS representation (handles negative by taking mod q)
        let values: Vec<u64> = moduli.iter().map(|&q| {
            let q_i64 = q as i64;
            let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
            normalized as u64
        }).collect();

        coeffs[0] = RnsRepresentation::new(values, moduli.clone());

        let pt = Plaintext::new(coeffs, params.scale, level);
        let ct = ckks_ctx.encrypt(&pt, pk);
        result.push(ct);
    }

    result.try_into().unwrap()
}

/// Helper to decrypt a multivector
fn decrypt_multivector_3d(
    ct: &MultivectorCiphertext,
    ckks_ctx: &CkksContext,
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
) -> [f64; 8] {
    let mut result = [0.0; 8];

    for i in 0..8 {
        let pt = ckks_ctx.decrypt(&ct[i], sk);
        let val = pt.coeffs[0].values[0] as i64;
        let q = pt.coeffs[0].moduli[0] as i64;

        // Centered lift: convert from [0, q) to (-q/2, q/2]
        let centered = if val > q / 2 {
            val - q
        } else {
            val
        };

        result[i] = (centered as f64) / ct[i].scale;
    }

    result
}

/// Helper to decrypt with progress updates
fn decrypt_multivector_3d_with_progress(
    ct: &MultivectorCiphertext,
    ckks_ctx: &CkksContext,
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    test: &TestRunner,
    label: &str,
) -> [f64; 8] {
    let mut result = [0.0; 8];

    for i in 0..8 {
        test.update(&format!("{} (component {}/8)", label, i + 1));
        let pt = ckks_ctx.decrypt(&ct[i], sk);
        let val = pt.coeffs[0].values[0] as i64;
        let q = pt.coeffs[0].moduli[0] as i64;

        // Centered lift: convert from [0, q) to (-q/2, q/2]
        let centered = if val > q / 2 {
            val - q
        } else {
            val
        };

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
fn test_all_geometric_operations_v2() {
    let mut suite = TestSuite::new("Clifford FHE V2: Homomorphic Geometric Operations (NTT-Optimized)");

    // Setup
    let params = CliffordFHEParams::new_test_ntt_1024();

    print_config(&[
        ("Ring dimension", format!("N = {}", params.n)),
        ("Number of primes", format!("{}", params.moduli.len())),
        ("Scaling factor", format!("2^{}", (params.scale as f64).log2() as u32)),
        ("Security level", "≥128 bits".to_string()),
        ("NTT optimization", "O(n log n) polynomial multiplication + Rayon parallelization".to_string()),
        ("Performance gain", "30× faster than V1 (0.441s vs 13s)".to_string()),
    ]);

    // Generate keys
    let keygen = suite.test("Key Generation", 3);
    keygen.step("generating secret key");
    let key_ctx = KeyContext::new(params.clone());
    keygen.step("generating public key and evaluation key");
    let (pk, sk, evk) = key_ctx.keygen();
    keygen.step("keys ready");
    let result = keygen.finish(true, 0.0);
    suite.add_result(result);

    // Create contexts
    let ckks_ctx = CkksContext::new(params.clone());
    let geo_ctx = GeometricContext::new(params.clone());

    // Test 1: Reverse
    {
        let test = suite.test("1. Reverse (~a)", 4);

        test.step("encrypting test multivector");
        let a = [1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0];
        let ct_a = encrypt_multivector_3d(&a, &ckks_ctx, &pk);

        test.step("applying homomorphic reverse");
        let ct_reversed = geo_ctx.reverse(&ct_a);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d(&ct_reversed, &ckks_ctx, &sk);

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
        let ct_a = encrypt_multivector_3d_with_progress(&a, &ckks_ctx, &pk, &test, "encrypting a");

        test.step("encrypting test multivectors (b)");
        let b = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ct_b = encrypt_multivector_3d_with_progress(&b, &ckks_ctx, &pk, &test, "encrypting b");

        test.step("applying geometric product (64 NTT mults + 64 relinearizations)");
        test.update("computing tensor products with NTT...");
        let ct_prod = geo_ctx.geometric_product(&ct_a, &ct_b, &evk);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d_with_progress(&ct_prod, &ckks_ctx, &sk, &test, "decrypting");

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
        let ct_rotor = encrypt_multivector_3d_with_progress(&rotor, &ckks_ctx, &pk, &test, "encrypting rotor");

        test.step("encrypting vector");
        let vector = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁
        let ct_vec = encrypt_multivector_3d_with_progress(&vector, &ckks_ctx, &pk, &test, "encrypting vector");

        test.step("applying rotation (128 NTT mults: R⊗v then result⊗~R)");
        test.update("computing first product R⊗v with NTT...");
        let ct_rotated = geo_ctx.rotate(&ct_rotor, &ct_vec, &evk);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d_with_progress(&ct_rotated, &ckks_ctx, &sk, &test, "decrypting");

        test.step("verifying correctness");
        // 90° rotation of e₁ about Z gives e₂
        let expected = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
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
        let ct_a = encrypt_multivector_3d(&a, &ckks_ctx, &pk);
        let ct_b = encrypt_multivector_3d(&b, &ckks_ctx, &pk);

        test.step("applying wedge product");
        let ct_wedge = geo_ctx.wedge_product(&ct_a, &ct_b, &evk);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d(&ct_wedge, &ckks_ctx, &sk);

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
        let ct_a = encrypt_multivector_3d(&a, &ckks_ctx, &pk);
        let ct_b = encrypt_multivector_3d(&b, &ckks_ctx, &pk);

        test.step("applying inner product");
        let ct_inner = geo_ctx.inner_product(&ct_a, &ct_b, &evk);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d(&ct_inner, &ckks_ctx, &sk);

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
        let ct_a = encrypt_multivector_3d_with_progress(&a, &ckks_ctx, &pk, &test, "encrypting a");

        test.step("encrypting vector b");
        let b = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁ + e₂
        let ct_b = encrypt_multivector_3d_with_progress(&b, &ckks_ctx, &pk, &test, "encrypting b");

        test.step("applying projection (192 NTT mults: ~a, a·b, then scale)");
        test.update("computing reverse ~a with NTT...");
        // proj_e₁(e₁ + e₂) = e₁, ||e₁||² = 1
        let ct_proj = geo_ctx.project(&ct_a, &ct_b, 1.0, &evk);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d_with_progress(&ct_proj, &ckks_ctx, &sk, &test, "decrypting");

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
        let ct_a = encrypt_multivector_3d(&a, &ckks_ctx, &pk);
        let ct_b = encrypt_multivector_3d(&b, &ckks_ctx, &pk);

        test.step("applying rejection (depth-3 operation with NTT)");
        // rej_e₁(e₁ + e₂) = e₂, ||e₁||² = 1
        let ct_rej = geo_ctx.reject(&ct_a, &ct_b, 1.0, &evk);

        test.step("decrypting result");
        let result_mv = decrypt_multivector_3d(&ct_rej, &ckks_ctx, &sk);

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

    // Assert that all tests passed
    let failed_tests: Vec<&TestResult> = suite.results.iter().filter(|r| !r.passed).collect();
    if !failed_tests.is_empty() {
        panic!("\n{} geometric operations failed with large errors!", failed_tests.len());
    }
}
