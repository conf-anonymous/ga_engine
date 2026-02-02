//! Comprehensive tests for all homomorphic geometric operations
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]

//!
//! This test suite verifies that ALL 7 fundamental geometric algebra
//! operations work correctly with the fixed RNS-CKKS implementation.

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{rns_encrypt, rns_decrypt, RnsPlaintext, RnsCiphertext};
use ga_engine::clifford_fhe::geometric_product_rns::{
    geometric_product_3d_componentwise,
    reverse_3d,
    rotate_3d,
    wedge_product_3d,
    inner_product_3d,
    project_3d,
    reject_3d,
};

/// Helper to encrypt a multivector (8 components for 3D)
fn encrypt_multivector_3d(
    mv: &[f64; 8],
    pk: &ga_engine::clifford_fhe::keys_rns::RnsPublicKey,
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

/// Helper to decrypt a multivector
fn decrypt_multivector_3d(
    ct: &[RnsCiphertext; 8],
    sk: &ga_engine::clifford_fhe::keys_rns::RnsSecretKey,
    params: &CliffordFHEParams,
) -> [f64; 8] {
    let mut result = [0.0; 8];

    for i in 0..8 {
        let pt = rns_decrypt(sk, &ct[i], params);

        // Decode from first prime only (single-prime decoding)
        let val = pt.coeffs.rns_coeffs[0][0];
        let q = params.moduli[0];

        // Center-lift
        let centered = if val > q / 2 { val - q } else { val };

        // Decode with appropriate scale
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
fn test_homomorphic_reverse() {
    println!("\n=== Testing Homomorphic Reverse ===");

    let params = CliffordFHEParams::new_rns_mult();
    let (pk, sk, _evk) = rns_keygen(&params);

    // Test multivector: 1 + 2e₁ + 3e₁₂ + 4e₁₂₃
    let a = [1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0];
    println!("Input: a = {:?}", a);

    // Encrypt
    let ct_a = encrypt_multivector_3d(&a, &pk, &params);

    // Apply reverse
    let ct_reversed = reverse_3d(&ct_a, &params);

    // Decrypt
    let result = decrypt_multivector_3d(&ct_reversed, &sk, &params);
    println!("Result: ~a = {:?}", result);

    // Expected: ~(1 + 2e₁ + 3e₁₂ + 4e₁₂₃) = 1 + 2e₁ - 3e₁₂ + 4e₁₂₃
    // Reverse flips sign of grade-2 (bivectors) ONLY
    // Grade-0 (scalar), grade-1 (vectors), and grade-3 (trivector) unchanged
    let expected = [1.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0];
    println!("Expected: {:?}", expected);

    let error = max_error(&result, &expected);
    println!("Max error: {:.6}", error);

    assert!(error < 0.01, "Reverse error too large: {}", error);
    println!("✅ PASS: Reverse operation works!\n");
}

#[test]
fn test_homomorphic_geometric_product() {
    println!("\n=== Testing Homomorphic Geometric Product ===");

    let params = CliffordFHEParams::new_rns_mult();  // Revert to 3 primes
    let (pk, sk, evk) = rns_keygen(&params);

    // Test: e₁ ⊗ e₂ = e₁₂ (simpler test - only product of basis vectors)
    let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₁
    let b = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₂
    println!("a = e₁ = {:?}", a);
    println!("b = e₂ = {:?}", b);

    // Encrypt
    let ct_a = encrypt_multivector_3d(&a, &pk, &params);
    let ct_b = encrypt_multivector_3d(&b, &pk, &params);

    // Geometric product
    let ct_product = geometric_product_3d_componentwise(&ct_a, &ct_b, &evk, &params);

    // Decrypt
    let result = decrypt_multivector_3d(&ct_product, &sk, &params);
    println!("Result: e₁ ⊗ e₂ = {:?}", result);

    // Expected: e₁₂ (bivector)
    let expected = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    println!("Expected: e₁₂ = {:?}", expected);

    let error = max_error(&result, &expected);
    println!("Max error: {:.6}", error);

    assert!(error < 0.1, "Geometric product error too large: {}", error);
    println!("✅ PASS: Geometric product works!\n");
}

#[test]
fn test_homomorphic_wedge_product() {
    println!("\n=== Testing Homomorphic Wedge Product ===");

    let params = CliffordFHEParams::new_rns_mult_depth2_safe();  // Use depth-2 params (5 primes for better headroom)
    let (pk, sk, evk) = rns_keygen(&params);

    // Test: e₁ ∧ e₂ = e₁₂ (wedge product of orthogonal vectors)
    let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₁
    let b = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₂
    println!("a = e₁ = {:?}", a);
    println!("b = e₂ = {:?}", b);

    // Encrypt
    let ct_a = encrypt_multivector_3d(&a, &pk, &params);
    let ct_b = encrypt_multivector_3d(&b, &pk, &params);

    // Wedge product
    let ct_wedge = wedge_product_3d(&ct_a, &ct_b, &evk, &params);

    // Decrypt
    let result = decrypt_multivector_3d(&ct_wedge, &sk, &params);
    println!("Result: a ∧ b = {:?}", result);

    // Expected: e₁₂ (bivector)
    let expected = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    println!("Expected: {:?}", expected);

    let error = max_error(&result, &expected);
    println!("Max error: {:.6}", error);

    assert!(error < 0.1, "Wedge product error too large: {}", error);
    println!("✅ PASS: Wedge product works!\n");
}

#[test]
fn test_homomorphic_inner_product() {
    println!("\n=== Testing Homomorphic Inner Product ===");

    // Use 5-prime params for depth-2 support
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    println!("Using {} primes for inner product", params.moduli.len());
    let (pk, sk, evk) = rns_keygen(&params);

    // Test: e₁ · e₁ = 1 (inner product of vector with itself)
    let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₁
    let b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₁
    println!("a = e₁ = {:?}", a);
    println!("b = e₁ = {:?}", b);

    // Encrypt
    let ct_a = encrypt_multivector_3d(&a, &pk, &params);
    let ct_b = encrypt_multivector_3d(&b, &pk, &params);

    // Inner product
    let ct_inner = inner_product_3d(&ct_a, &ct_b, &evk, &params);

    // Decrypt
    let result = decrypt_multivector_3d(&ct_inner, &sk, &params);
    println!("Result: a · b = {:?}", result);

    // Expected: 1 (scalar)
    let expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    println!("Expected: {:?}", expected);

    let error = max_error(&result, &expected);
    println!("Max error: {:.6}", error);

    assert!(error < 0.1, "Inner product error too large: {}", error);
    println!("✅ PASS: Inner product works!\n");
}

#[test]
fn test_homomorphic_rotation() {
    println!("\n=== Testing Homomorphic Rotation ===");

    // Use 5-prime params for depth-2 support (rotation needs 2 multiplications)
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    println!("Using {} primes: {:?}", params.moduli.len(), params.moduli);
    let (pk, sk, evk) = rns_keygen(&params);

    // Create a 90° rotation about Z-axis
    // Rotor: R = cos(45°) + sin(45°)e₁₂ ≈ 0.707 + 0.707e₁₂
    let cos45 = std::f64::consts::FRAC_1_SQRT_2;
    let sin45 = std::f64::consts::FRAC_1_SQRT_2;
    let rotor = [cos45, 0.0, 0.0, 0.0, sin45, 0.0, 0.0, 0.0];

    // Vector to rotate: e₁ (pointing in X direction)
    let vec = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    println!("Rotor R = {:.3} + {:.3}e₁₂", cos45, sin45);
    println!("Vector v = e₁ = {:?}", vec);

    // Encrypt
    let ct_rotor = encrypt_multivector_3d(&rotor, &pk, &params);
    let ct_vec = encrypt_multivector_3d(&vec, &pk, &params);

    // Rotate: v' = R ⊗ v ⊗ ~R
    let ct_rotated = rotate_3d(&ct_rotor, &ct_vec, &evk, &params);

    // Decrypt
    let result = decrypt_multivector_3d(&ct_rotated, &sk, &params);
    println!("Result: R ⊗ v ⊗ ~R = {:?}", result);

    // Expected: e₂ (rotated 90° from X to Y)
    let expected = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    println!("Expected: e₂ = {:?}", expected);

    let error = max_error(&result, &expected);
    println!("Max error: {:.6}", error);

    // Note: Rotation uses 2 multiplications, so error accumulates more
    assert!(error < 0.5, "Rotation error too large: {}", error);
    println!("✅ PASS: Rotation works!\n");
}

#[test]
fn test_homomorphic_projection() {
    println!("\n=== Testing Homomorphic Projection ===");

    // Use 5-prime params for depth-3 support
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    println!("Using {} primes for projection", params.moduli.len());
    let (pk, sk, evk) = rns_keygen(&params);

    // Project (e₁ + e₂) onto e₁
    // Expected: e₁ (the e₁ component)
    let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₁ (project onto this)
    let b = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₁ + e₂

    println!("Project onto: a = e₁ = {:?}", a);
    println!("Vector: b = e₁ + e₂ = {:?}", b);

    // Encrypt
    let ct_a = encrypt_multivector_3d(&a, &pk, &params);
    let ct_b = encrypt_multivector_3d(&b, &pk, &params);

    // Projection
    let ct_proj = project_3d(&ct_a, &ct_b, &evk, &params);

    // Decrypt
    let result = decrypt_multivector_3d(&ct_proj, &sk, &params);
    println!("Result: proj_a(b) = {:?}", result);

    // Expected: e₁ (the component of b along a)
    let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    println!("Expected: e₁ = {:?}", expected);

    let error = max_error(&result, &expected);
    println!("Max error: {:.6}", error);

    // Projection uses multiple multiplications
    assert!(error < 0.5, "Projection error too large: {}", error);
    println!("✅ PASS: Projection works!\n");
}

#[test]
fn test_homomorphic_rejection() {
    println!("\n=== Testing Homomorphic Rejection ===");

    // Use 5-prime params for depth-3 support
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    println!("Using {} primes for rejection", params.moduli.len());
    let (pk, sk, evk) = rns_keygen(&params);

    // Reject (e₁ + e₂) from e₁
    // Expected: e₂ (the perpendicular component)
    let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₁ (reject from this)
    let b = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₁ + e₂

    println!("Reject from: a = e₁ = {:?}", a);
    println!("Vector: b = e₁ + e₂ = {:?}", b);

    // Encrypt
    let ct_a = encrypt_multivector_3d(&a, &pk, &params);
    let ct_b = encrypt_multivector_3d(&b, &pk, &params);

    // Rejection
    let ct_rej = reject_3d(&ct_a, &ct_b, &evk, &params);

    // Decrypt
    let result = decrypt_multivector_3d(&ct_rej, &sk, &params);
    println!("Result: rej_a(b) = {:?}", result);

    // Expected: e₂ (the component of b perpendicular to a)
    let expected = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    println!("Expected: e₂ = {:?}", expected);

    let error = max_error(&result, &expected);
    println!("Max error: {:.6}", error);

    // Rejection uses multiple multiplications
    assert!(error < 0.5, "Rejection error too large: {}", error);
    println!("✅ PASS: Rejection works!\n");
}
