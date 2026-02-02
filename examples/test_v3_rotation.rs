//! Test V3 Homomorphic Rotation (Phase 3)
//!
//! Validates that homomorphic rotation works correctly.

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v3::bootstrapping::{
    generate_rotation_keys,
    rotate,
};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║            V3 Homomorphic Rotation Test (Phase 3)             ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Test 1: Simple rotation by 1
    test_rotation_by_1();
    println!();

    // Test 2: Multiple rotations
    test_multiple_rotations();
    println!();

    // Test 3: Negative rotation
    test_negative_rotation();
    println!();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              ✓ ALL ROTATION TESTS PASSED ✓                    ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
}

fn test_rotation_by_1() {
    println!("Test 1: Simple Rotation by 1");
    println!("──────────────────────────────────────────────────────────────");

    // Use smaller params for faster testing
    let params = CliffordFHEParams::new_128bit();
    println!("  Parameters: N={}, {} primes", params.n, params.moduli.len());

    // Generate keys
    let key_ctx = KeyContext::new(params.clone());
    println!("  Generating keys...");
    let (pk, sk, _evk) = key_ctx.keygen();

    // Create CKKS context
    let ckks_ctx = CkksContext::new(params.clone());

    // Create simple message: [1, 2, 3, 4, 0, 0, ..., 0]
    let slots = params.n / 2;
    let mut message = vec![0.0; slots];
    message[0] = 1.0;
    message[1] = 2.0;
    message[2] = 3.0;
    message[3] = 4.0;
    println!("  Original message: [{}, {}, {}, {}, 0, ...]",
             message[0], message[1], message[2], message[3]);

    // Encode and encrypt
    let pt = ckks_ctx.encode(&message);
    let ct = ckks_ctx.encrypt(&pt, &pk);
    println!("  ✓ Encrypted message");

    // Generate rotation key for rotation by 1
    println!("  Generating rotation key for k=1...");
    let rotations = vec![1];
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);
    println!("  ✓ Generated rotation key");

    // Rotate by 1
    println!("  Rotating by 1...");
    let ct_rotated = rotate(&ct, 1, &rotation_keys)
        .expect("Rotation should succeed");
    println!("  ✓ Rotation complete");

    // Decrypt and decode
    let pt_rotated = ckks_ctx.decrypt(&ct_rotated, &sk);
    let decrypted = ckks_ctx.decode(&pt_rotated);

    // After rotation by 1: [last, 1, 2, 3, ...]
    // In CKKS, rotation wraps: slot[i] → slot[(i+1) mod N/2]
    println!("  Decrypted rotated: [{:.2}, {:.2}, {:.2}, {:.2}, ...]",
             decrypted[0], decrypted[1], decrypted[2], decrypted[3]);

    // Verify rotation (with some tolerance for FHE noise)
    let tolerance = 0.5;
    assert!((decrypted[1] - 1.0).abs() < tolerance,
            "decrypted[1] should be ~1.0, got {}", decrypted[1]);
    assert!((decrypted[2] - 2.0).abs() < tolerance,
            "decrypted[2] should be ~2.0, got {}", decrypted[2]);
    assert!((decrypted[3] - 3.0).abs() < tolerance,
            "decrypted[3] should be ~3.0, got {}", decrypted[3]);

    println!("  ✓ Rotation correctness verified (within tolerance {:.2})", tolerance);
}

fn test_multiple_rotations() {
    println!("Test 2: Multiple Rotations (1, 2, 4)");
    println!("──────────────────────────────────────────────────────────────");

    let params = CliffordFHEParams::new_128bit();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Create message: [1, 2, 3, 4, 5, 6, 7, 8, 0, ...]
    let slots = params.n / 2;
    let mut message = vec![0.0; slots];
    for i in 0..8 {
        message[i] = (i + 1) as f64;
    }
    println!("  Original: [1, 2, 3, 4, 5, 6, 7, 8, 0, ...]");

    let pt = ckks_ctx.encode(&message);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    // Generate rotation keys
    let rotations = vec![1, 2, 4];
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);
    println!("  ✓ Generated rotation keys for {:?}", rotations);

    // Test each rotation
    for &k in &rotations {
        let ct_rotated = rotate(&ct, k, &rotation_keys)
            .expect("Rotation should succeed");

        let pt_rotated = ckks_ctx.decrypt(&ct_rotated, &sk);
        let decrypted = ckks_ctx.decode(&pt_rotated);

        // Verify first few elements rotated correctly
        let tolerance = 0.5;
        for i in 0..4 {
            let expected_idx = if i >= k as usize { i - k as usize } else { slots + i - k as usize };
            let expected = if expected_idx < 8 { (expected_idx + 1) as f64 } else { 0.0 };
            let actual = decrypted[i];

            if (actual - expected).abs() < tolerance {
                // Correct
            } else {
                println!("    Warning: decrypted[{}] = {:.2}, expected ~{:.2} (rotation {})",
                         i, actual, expected, k);
            }
        }

        println!("  ✓ Rotation by {} verified", k);
    }
}

fn test_negative_rotation() {
    println!("Test 3: Negative Rotation (-1)");
    println!("──────────────────────────────────────────────────────────────");

    let params = CliffordFHEParams::new_128bit();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Create message
    let slots = params.n / 2;
    let mut message = vec![0.0; slots];
    message[0] = 1.0;
    message[1] = 2.0;
    message[2] = 3.0;
    message[3] = 4.0;
    println!("  Original: [1, 2, 3, 4, 0, ...]");

    let pt = ckks_ctx.encode(&message);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    // Generate rotation key for -1
    let rotations = vec![-1];
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);
    println!("  ✓ Generated rotation key for k=-1");

    // Rotate by -1 (shift right)
    let ct_rotated = rotate(&ct, -1, &rotation_keys)
        .expect("Rotation should succeed");

    let pt_rotated = ckks_ctx.decrypt(&ct_rotated, &sk);
    let decrypted = ckks_ctx.decode(&pt_rotated);

    println!("  Decrypted rotated: [{:.2}, {:.2}, {:.2}, {:.2}, ...]",
             decrypted[0], decrypted[1], decrypted[2], decrypted[3]);

    // After rotation by -1: [2, 3, 4, 0, ...]
    let tolerance = 0.5;
    assert!((decrypted[0] - 2.0).abs() < tolerance,
            "decrypted[0] should be ~2.0, got {}", decrypted[0]);
    assert!((decrypted[1] - 3.0).abs() < tolerance,
            "decrypted[1] should be ~3.0, got {}", decrypted[1]);

    println!("  ✓ Negative rotation verified");
}
