//! Debug projection operation to find the exact issue

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::{CkksContext, Plaintext};
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::geometric::{GeometricContext, MultivectorCiphertext};
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

#[test]
fn debug_projection() {
    println!("\n=== DEBUGGING PROJECTION ===\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let ckks_ctx = CkksContext::new(params.clone());
    let geo_ctx = GeometricContext::new(params.clone());

    let (pk, sk, evk) = key_ctx.keygen();

    println!("Initial scale: {}", params.scale);
    println!("Initial level: {}", params.max_level());

    // Test vectors
    let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁
    let b = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁ + e₂

    println!("\nTest: proj_e₁(e₁ + e₂) should = e₁");
    println!("a = e₁ = {:?}", a);
    println!("b = e₁ + e₂ = {:?}", b);
    println!("Expected result: [0, 1, 0, 0, 0, 0, 0, 0]");

    // Encrypt vectors
    let ct_a = encrypt_mv(&a, &ckks_ctx, &pk, &params);
    let ct_b = encrypt_mv(&b, &ckks_ctx, &pk, &params);

    println!("\n--- After encryption ---");
    println!("ct_a[1] scale: {}", ct_a[1].scale);
    println!("ct_a[1] level: {}", ct_a[1].level);

    // Step 1: Compute b̃ (reverse)
    let b_reverse = geo_ctx.reverse(&ct_b);
    println!("\n--- Step 1: Reverse b ---");
    println!("b_reverse[1] scale: {}", b_reverse[1].scale);
    println!("b_reverse[1] level: {}", b_reverse[1].level);

    // Step 2: Compute a · b̃ (inner product)
    println!("\n--- Step 2: Inner product a · b̃ ---");
    let a_dot_b_rev = geo_ctx.inner_product(&ct_a, &b_reverse, &evk);
    println!("a_dot_b_rev[0] scale: {}", a_dot_b_rev[0].scale);
    println!("a_dot_b_rev[0] level: {}", a_dot_b_rev[0].level);

    // Decrypt intermediate result
    let a_dot_b_rev_decrypted = decrypt_mv(&a_dot_b_rev, &ckks_ctx, &sk);
    println!("a · b̃ decrypted: {:?}", a_dot_b_rev_decrypted);
    println!("Expected: [1, 0, 0, 0, 0, 0, 0, 0] (scalar 1)");

    // Step 3: Compute (a · b̃) ⊗ b
    println!("\n--- Step 3: Geometric product (a · b̃) ⊗ b ---");
    let numerator = geo_ctx.geometric_product(&a_dot_b_rev, &ct_b, &evk);
    println!("numerator[1] scale: {}", numerator[1].scale);
    println!("numerator[1] level: {}", numerator[1].level);

    // Decrypt numerator
    let numerator_decrypted = decrypt_mv(&numerator, &ckks_ctx, &sk);
    println!("Numerator decrypted: {:?}", numerator_decrypted);
    println!("Expected: [0, 1, 1, 0, 0, 0, 0, 0] (since 1 ⊗ (e₁ + e₂) = e₁ + e₂)");

    // Step 4: Divide by b·b̃ = 1
    println!("\n--- Step 4: Divide by b·b̃ = 1.0 ---");
    let b_norm_sq = 1.0;
    let scale_factor = 1.0 / b_norm_sq;
    println!("scale_factor = 1.0 / {} = {}", b_norm_sq, scale_factor);

    let result = geo_ctx.mul_multivector_scalar(&numerator, scale_factor);
    println!("result[1] scale: {}", result[1].scale);
    println!("result[1] level: {}", result[1].level);

    // Decrypt final result
    let result_decrypted = decrypt_mv(&result, &ckks_ctx, &sk);
    println!("\n--- FINAL RESULT ---");
    println!("Result decrypted: {:?}", result_decrypted);
    println!("Expected:         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]");

    // Check error
    let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let error: f64 = result_decrypted.iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    println!("\nMax error: {:.2e}", error);

    if error > 1e-6 {
        println!("\n❌ PROJECTION FAILED!");
        println!("Component errors:");
        for i in 0..8 {
            let err = (result_decrypted[i] - expected[i]).abs();
            if err > 1e-6 {
                println!("  [{}]: got {:.6}, expected {:.6}, error {:.2e}",
                         i, result_decrypted[i], expected[i], err);
            }
        }
    } else {
        println!("\n✓ Projection works!");
    }
}

fn encrypt_mv(
    mv: &[f64; 8],
    ckks_ctx: &CkksContext,
    pk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey,
    params: &CliffordFHEParams,
) -> MultivectorCiphertext {
    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let n = params.n;

    let mut result = Vec::new();

    for &component in mv.iter() {
        let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); n];
        let scaled_val = (component * params.scale).round() as i64;
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

fn decrypt_mv(
    ct: &MultivectorCiphertext,
    ckks_ctx: &CkksContext,
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
) -> [f64; 8] {
    let mut result = [0.0; 8];

    for i in 0..8 {
        let pt = ckks_ctx.decrypt(&ct[i], sk);
        let val = pt.coeffs[0].values[0] as i64;
        let q = pt.coeffs[0].moduli[0] as i64;
        let centered = if val > q / 2 { val - q } else { val };
        result[i] = (centered as f64) / ct[i].scale;
    }

    result
}
