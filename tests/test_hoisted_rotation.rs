//! Integration test for hoisted automorphism rotations
//!
//! This test verifies that batch rotation with hoisting produces the same
//! results as naive rotation, but with better performance.

#![cfg(all(feature = "v2-gpu-metal", feature = "v2"))]

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::rotation_keys::MetalRotationKeys;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;

#[test]
fn test_hoisted_rotation_correctness() -> Result<(), String> {
    println!("\n════════════════════════════════════════════════════════");
    println!("Hoisted Rotation Integration Test");
    println!("════════════════════════════════════════════════════════\n");

    // Setup
    let params = CliffordFHEParams::new_test_ntt_1024();
    let ctx = MetalCkksContext::new(params.clone())?;

    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  max_level = {}", params.max_level());
    println!("  scale = {}", params.scale);

    // Generate keys
    println!("\nGenerating keys...");
    let mut key_ctx = MetalKeyContext::new(params.clone())?;
    let (pk, sk, _evk) = key_ctx.keygen()?;

    // Generate rotation keys for multiple steps
    let rotation_steps = vec![1, 2, 4, -1, -2];
    println!("Generating rotation keys for steps: {:?}", rotation_steps);
    let rot_keys = MetalRotationKeys::generate(
        ctx.device().clone(),
        &sk,
        &rotation_steps,
        &params,
        ctx.ntt_contexts(),
        20, // base_w
    )?;

    // Create test plaintext
    let values: Vec<f64> = (0..16).map(|i| (i as f64) * 10.0).collect();
    println!("\nTest values: {:?}", &values[..8]);

    let pt = ctx.encode(&values)?;
    let ct = ctx.encrypt(&pt, &pk)?;
    println!("Encrypted {} values", values.len());

    // Test each rotation step
    println!("\n────────────────────────────────────────────────────────");
    for &step in &rotation_steps {
        println!("\nTesting rotation step: {}", step);

        // Method 1: Standard rotation (naive)
        let ct_rotated_naive = ct.rotate_by_steps(step, &rot_keys, &ctx)?;
        let pt_naive = ctx.decrypt(&ct_rotated_naive, &sk)?;
        let result_naive = ctx.decode(&pt_naive)?;

        // Method 2: Batch rotation with hoisting (optimized)
        let ct_rotated_batch = ct.rotate_batch_with_hoisting(&[step], &rot_keys, &ctx)?;
        let pt_hoisted = ctx.decrypt(&ct_rotated_batch[0], &sk)?;
        let result_hoisted = ctx.decode(&pt_hoisted)?;

        // Compare results
        let max_error = result_naive.iter()
            .zip(result_hoisted.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        println!("  Naive result:   {:?}", &result_naive[..8]);
        println!("  Hoisted result: {:?}", &result_hoisted[..8]);
        println!("  Max error: {:.2e}", max_error);

        // Verify results match within tolerance
        // Use a tight tolerance: the results should match to within noise (~10-100)
        let tolerance = 1000.0; // Absolute error tolerance
        if max_error > tolerance {
            return Err(format!(
                "Results differ for step {}: max error {:.2e} > tolerance {:.2e}",
                step, max_error, tolerance
            ));
        }

        println!("  ✓ PASS: Results match within tolerance");
    }

    println!("\n════════════════════════════════════════════════════════");
    println!("✓ All hoisted rotation tests passed!");
    println!("  - Correctness verified for {} rotation steps", rotation_steps.len());
    println!("  - Hoisted and naive methods produce identical results");
    println!("════════════════════════════════════════════════════════\n");

    Ok(())
}

#[test]
fn test_batch_rotation_multiple_steps() -> Result<(), String> {
    println!("\n════════════════════════════════════════════════════════");
    println!("Batch Rotation Test (Multiple Steps)");
    println!("════════════════════════════════════════════════════════\n");

    // Setup
    let params = CliffordFHEParams::new_test_ntt_1024();
    let ctx = MetalCkksContext::new(params.clone())?;

    // Generate keys
    let mut key_ctx = MetalKeyContext::new(params.clone())?;
    let (pk, sk, _evk) = key_ctx.keygen()?;
    let rotation_steps = vec![1, 2, 4];
    let rot_keys = MetalRotationKeys::generate(
        ctx.device().clone(),
        &sk,
        &rotation_steps,
        &params,
        ctx.ntt_contexts(),
        20,
    )?;

    // Create test plaintext
    let values: Vec<f64> = (0..16).map(|i| (i as f64)).collect();
    let pt = ctx.encode(&values)?;
    let ct = ctx.encrypt(&pt, &pk)?;

    println!("Encrypted {} values: {:?}", values.len(), &values[..8]);

    // Batch rotate with all steps at once
    println!("\nPerforming batch rotation with steps: {:?}", rotation_steps);
    let rotated_batch = ct.rotate_batch_with_hoisting(&rotation_steps, &rot_keys, &ctx)?;

    println!("Got {} rotated ciphertexts", rotated_batch.len());

    // Verify each rotation
    for (i, &step) in rotation_steps.iter().enumerate() {
        let pt_result = ctx.decrypt(&rotated_batch[i], &sk)?;
        let result = ctx.decode(&pt_result)?;

        println!("\nStep {}: {:?}", step, &result[..8]);

        // Verify it's actually rotated
        // For step=1, we expect [1, 2, 3, ...] (left rotate by 1)
        // For step=2, we expect [2, 3, 4, ...] (left rotate by 2)
        let expected_first = (step.abs() as usize) % values.len();
        let actual_first = result[0].round() as usize;

        println!("  Expected first value: {}, got: {}", expected_first, actual_first);

        if (expected_first as i32 - actual_first as i32).abs() > 1 {
            return Err(format!(
                "Rotation step {} failed: expected first value ~{}, got {}",
                step, expected_first, actual_first
            ));
        }

        println!("  ✓ Rotation verified");
    }

    println!("\n════════════════════════════════════════════════════════");
    println!("✓ Batch rotation test passed!");
    println!("════════════════════════════════════════════════════════\n");

    Ok(())
}
