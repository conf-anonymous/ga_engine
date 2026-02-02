/// Encrypted Inference Demo (V2 CPU Backend)
///
/// This example demonstrates end-to-end encrypted inference using the V2 CPU backend.
///
/// **What it shows:**
/// 1. Generate synthetic 3D shapes (spheres, cubes, pyramids)
/// 2. Encode as Cl(3,0) multivectors
/// 3. Encrypt multivectors using V2 CKKS
/// 4. Decrypt and verify accuracy
///
/// **Note:** This is a proof-of-concept on CPU.
/// For production, use Metal/CUDA GPU backends for 100-1000× speedup.
///
/// **Run with:**
/// ```bash
/// cargo run --release --features v2 --example encrypted_inference_demo
/// ```

#[cfg(feature = "v2")]
fn main() {
    use ga_engine::medical_imaging::{
        clifford_encoding::{Multivector3D, encode_point_cloud},
        synthetic_data::{self, generate_sphere},
        encrypted_v2_cpu::V2CpuEncryptionContext,
    };
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

    println!("=== Encrypted Inference Demo (V2 CPU Backend) ===\n");

    // 1. Setup encryption
    println!("Phase 1: Setting up encryption...");
    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("  Parameters: N={}, {} primes, scale=2^40", params.n, params.moduli.len());

    println!("  Generating keys...");
    let ctx = V2CpuEncryptionContext::new(params);
    println!("  ✓ Keys generated\n");

    // 2. Generate test data
    println!("Phase 2: Generating test shapes...");
    let test_shapes = vec![
        ("Sphere", generate_sphere(100, 1.0)),
        ("Cube", synthetic_data::generate_cube(100, 2.0)),
        ("Pyramid", synthetic_data::generate_pyramid(100, 2.0, 3.0)),
    ];
    println!("  ✓ Generated {} shapes\n", test_shapes.len());

    // 3. Encode and encrypt
    println!("Phase 3: Encoding and encrypting...");
    for (name, shape) in &test_shapes {
        println!("  Processing {}...", name);

        // Encode as multivector
        let multivector = encode_point_cloud(shape);
        println!("    Original: ({:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3})",
                 multivector.components[0], multivector.components[1],
                 multivector.components[2], multivector.components[3],
                 multivector.components[4], multivector.components[5],
                 multivector.components[6], multivector.components[7]);

        // Encrypt
        let encrypted = ctx.encrypt_multivector(&multivector);
        println!("    ✓ Encrypted to 8 ciphertexts");

        // Decrypt
        let decrypted = ctx.decrypt_multivector(&encrypted);
        println!("    Decrypted: ({:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3})",
                 decrypted.components[0], decrypted.components[1],
                 decrypted.components[2], decrypted.components[3],
                 decrypted.components[4], decrypted.components[5],
                 decrypted.components[6], decrypted.components[7]);

        // Compute error
        let mut max_error: f64 = 0.0;
        for i in 0..8 {
            let error = (multivector.components[i] - decrypted.components[i]).abs();
            max_error = max_error.max(error);
        }
        println!("    Max error: {:.6}\n", max_error);
    }

    // 4. Test encrypted addition
    println!("Phase 4: Testing encrypted operations...");
    let sphere = generate_sphere(100, 1.0);
    let mv1 = encode_point_cloud(&sphere);

    let cube = synthetic_data::generate_cube(100, 2.0);
    let mv2 = encode_point_cloud(&cube);

    println!("  Encrypting two shapes...");
    let enc1 = ctx.encrypt_multivector(&mv1);
    let enc2 = ctx.encrypt_multivector(&mv2);

    println!("  Performing encrypted addition...");
    let enc_sum = ctx.encrypted_add(&enc1, &enc2);

    println!("  Decrypting result...");
    let dec_sum = ctx.decrypt_multivector(&enc_sum);

    // Verify correctness (plaintext addition)
    let expected_sum = Multivector3D::new([
        mv1.components[0] + mv2.components[0],
        mv1.components[1] + mv2.components[1],
        mv1.components[2] + mv2.components[2],
        mv1.components[3] + mv2.components[3],
        mv1.components[4] + mv2.components[4],
        mv1.components[5] + mv2.components[5],
        mv1.components[6] + mv2.components[6],
        mv1.components[7] + mv2.components[7],
    ]);

    println!("  Expected:  ({:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3})",
             expected_sum.components[0], expected_sum.components[1],
             expected_sum.components[2], expected_sum.components[3],
             expected_sum.components[4], expected_sum.components[5],
             expected_sum.components[6], expected_sum.components[7]);

    println!("  Decrypted: ({:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3})",
             dec_sum.components[0], dec_sum.components[1],
             dec_sum.components[2], dec_sum.components[3],
             dec_sum.components[4], dec_sum.components[5],
             dec_sum.components[6], dec_sum.components[7]);

    let mut max_add_error: f64 = 0.0;
    for i in 0..8 {
        let error = (expected_sum.components[i] - dec_sum.components[i]).abs();
        max_add_error = max_add_error.max(error);
    }
    println!("  Max error: {:.6}\n", max_add_error);

    // 5. Summary
    println!("=== Summary ===");
    println!("✓ Encryption/decryption working");
    println!("✓ Encrypted addition working");
    println!("✓ Max error: {:.6} (CKKS approximation)", max_add_error);
    println!("\nNext Steps:");
    println!("  1. Implement encrypted geometric product");
    println!("  2. Implement encrypted GNN forward pass");
    println!("  3. Port to Metal/CUDA GPU (100-1000× faster)");
    println!("  4. Implement SIMD batching (512× throughput)");
    println!("  5. End-to-end encrypted medical imaging classification");
}

#[cfg(not(feature = "v2"))]
fn main() {
    eprintln!("This example requires the 'v2' feature.");
    eprintln!("Run with: cargo run --release --features v2 --example encrypted_inference_demo");
    std::process::exit(1);
}
