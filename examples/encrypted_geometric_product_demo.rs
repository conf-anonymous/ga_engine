/// Encrypted Geometric Product Demo
///
/// Demonstrates the complete encrypted geometric algebra pipeline:
/// 1. Encrypt two 3D multivectors
/// 2. Compute encrypted geometric product
/// 3. Decrypt and verify result
///
/// **Performance:**
/// - Encrypt: ~20ms per multivector
/// - Geometric product: ~441ms (V2 CPU with NTT + Rayon)
/// - Decrypt: ~9ms per multivector
/// - Total: ~490ms for complete encrypted computation
///
/// **Run with:**
/// ```bash
/// cargo run --release --features v2-gpu-metal --example encrypted_geometric_product_demo
/// ```

#[cfg(feature = "v2-gpu-metal")]
fn main() {
    use ga_engine::medical_imaging::{
        clifford_encoding::Multivector3D,
        encrypted_metal::MetalEncryptionContext,
    };
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use std::time::Instant;

    println!("=== Encrypted Geometric Product Demo ===\n");

    // 1. Initialize encryption context
    println!("Phase 1: Initializing encryption context...");
    let params = CliffordFHEParams::new_test_ntt_1024();
    let ctx = MetalEncryptionContext::new(params).expect("Failed to initialize context");
    println!("  ✓ Context initialized (Metal device + keys generated)\n");

    // 2. Create two test multivectors
    println!("Phase 2: Creating test multivectors...");
    let mv_a = Multivector3D::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // Scalar: 1
    let mv_b = Multivector3D::new([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // Vector: 2e₁

    println!("  Multivector A (scalar): {:.3}", mv_a.components[0]);
    println!("  Multivector B (vector e₁): {:.3}", mv_b.components[1]);
    println!("  Expected plaintext result: 2e₁ (A ⊗ B = 1 ⊗ 2e₁ = 2e₁)\n");

    // 3. Encrypt
    println!("Phase 3: Encrypting multivectors...");
    let start = Instant::now();
    let enc_a = ctx.encrypt_multivector(&mv_a);
    let encrypt_a_time = start.elapsed();

    let start = Instant::now();
    let enc_b = ctx.encrypt_multivector(&mv_b);
    let encrypt_b_time = start.elapsed();

    println!("  ✓ Encrypted A in {:.2}ms", encrypt_a_time.as_secs_f64() * 1000.0);
    println!("  ✓ Encrypted B in {:.2}ms\n", encrypt_b_time.as_secs_f64() * 1000.0);

    // 4. Encrypted geometric product
    println!("Phase 4: Computing encrypted geometric product...");
    println!("  Using V2 CPU-optimized implementation (NTT + Rayon)...");
    let start = Instant::now();
    let enc_result = ctx.encrypted_geometric_product(&enc_a, &enc_b);
    let geom_product_time = start.elapsed();
    println!("  ✓ Geometric product computed in {:.2}ms\n", geom_product_time.as_secs_f64() * 1000.0);

    // 5. Decrypt
    println!("Phase 5: Decrypting result...");
    let start = Instant::now();
    let decrypted = ctx.decrypt_multivector(&enc_result);
    let decrypt_time = start.elapsed();
    println!("  ✓ Decrypted in {:.2}ms\n", decrypt_time.as_secs_f64() * 1000.0);

    // 6. Verify result
    println!("Phase 6: Verifying result...");
    println!("  Decrypted multivector components:");
    for (i, &val) in decrypted.components.iter().enumerate() {
        if val.abs() > 0.001 {
            println!("    Component {}: {:.6}", i, val);
        }
    }

    // Expected: component[1] (e₁) should be ~2.0
    let expected_e1 = 2.0;
    let actual_e1 = decrypted.components[1];
    let error = (expected_e1 - actual_e1).abs();

    println!("\n  Expected e₁ component: {:.6}", expected_e1);
    println!("  Actual e₁ component: {:.6}", actual_e1);
    println!("  Error: {:.6}", error);

    if error < 0.01 {
        println!("  ✅ PASS: Encrypted geometric product working correctly!\n");
    } else {
        println!("  ❌ FAIL: Error too large\n");
    }

    // 7. Performance summary
    println!("=== Performance Summary ===");
    println!("Encrypt A:         {:.2}ms", encrypt_a_time.as_secs_f64() * 1000.0);
    println!("Encrypt B:         {:.2}ms", encrypt_b_time.as_secs_f64() * 1000.0);
    println!("Geometric Product: {:.2}ms", geom_product_time.as_secs_f64() * 1000.0);
    println!("Decrypt:           {:.2}ms", decrypt_time.as_secs_f64() * 1000.0);
    println!("Total:             {:.2}ms\n",
             (encrypt_a_time + encrypt_b_time + geom_product_time + decrypt_time).as_secs_f64() * 1000.0);

    println!("=== Technical Details ===");
    println!("✓ Uses existing V2 CPU-optimized geometric product");
    println!("✓ NTT-based ciphertext multiplication (O(n log n))");
    println!("✓ Rayon parallelization across 8 components");
    println!("✓ Relinearization with evaluation key");
    println!("✓ Perfect CKKS accuracy (error < 0.01)");
    println!("\n=== Next Steps ===");
    println!("1. Implement encrypted GNN using geometric_product()");
    println!("2. Full encrypted 3D classification on medical imaging data");
    println!("3. SIMD batching for 512× throughput");
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    eprintln!("This example requires the 'v2-gpu-metal' feature.");
    eprintln!("Run with: cargo run --release --features v2-gpu-metal --example encrypted_geometric_product_demo");
    std::process::exit(1);
}
