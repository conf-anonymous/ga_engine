/// Simple V4 Metal pack/unpack API demo
///
/// This demonstrates the V4 packing API without full end-to-end encryption test.
/// For a complete test, see V3 bootstrap examples for key generation patterns.

#[cfg(all(feature = "v4", feature = "v2-gpu-metal"))]
fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("      V4 Metal Pack/Unpack API Demonstration");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("V4 Packing Overview:");
    println!("  Input:  8 component ciphertexts (one per Clifford basis)");
    println!("  Output: 1 packed ciphertext with interleaved slots\n");

    println!("Memory Savings:");
    println!("  V2/V3:  8 ciphertexts × batch_size multivectors");
    println!("  V4:     1 ciphertext for batch_size multivectors");
    println!("  Reduction: 8× memory savings!\n");

    println!("Packing Algorithm:");
    println!("  1. Start with component[0] (scalar)");
    println!("  2. For i = 1 to 7:");
    println!("       - Rotate component[i] left by i steps");
    println!("       - Add to accumulator");
    println!("  3. Result: [s₀, e1₀, e2₀, ..., I₀, s₁, e1₁, ...]\n");

    println!("Unpacking Algorithm:");
    println!("  1. Create extraction mask:");
    println!("       [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ...]");
    println!("       (1.0 at every 8th position)");
    println!("  2. For each component i:");
    println!("       - Rotate packed ciphertext right by i steps");
    println!("       - Multiply by mask to extract component");
    println!("  3. Result: 8 clean component ciphertexts\n");

    println!("API Usage:");
    println!("  // Pack");
    println!("  let packed = pack_multivector(");
    println!("      &components,  // [Ciphertext; 8]");
    println!("      batch_size,   // Number of multivectors");
    println!("      &rot_keys,    // Rotation keys for steps ±1 to ±7");
    println!("      &ckks_ctx,    // Metal CKKS context");
    println!("  )?;\n");

    println!("  // Unpack");
    println!("  let components = unpack_multivector(");
    println!("      &packed,      // PackedMultivector");
    println!("      &rot_keys,    // Same rotation keys");
    println!("      &ckks_ctx,    // Same context");
    println!("  )?;\n");

    println!("Required Rotation Keys:");
    println!("  Steps: ±1, ±2, ±3, ±4, ±5, ±6, ±7");
    println!("  Total: 14 rotation keys\n");

    println!("Implementation Status:");
    println!("  1. ✓ Packing/unpacking with rotation");
    println!("  2. ✓ Extraction masking (plaintext multiplication)");
    println!("  3. ⏭ Implement geometric operations");
    println!("  4. ⏭ Add integration test with full encryption\n");

    println!("Masking Details:");
    println!("  Mask pattern: [1, 0⁷, 1, 0⁷, ...] repeating");
    println!("  Purpose: Zero out unwanted components after rotation");
    println!("  Cost: 1 plaintext multiplication per component\n");

    println!("For full end-to-end test:");
    println!("  See examples/test_metal_rotation_only.rs");
    println!("  for key generation patterns\n");
}

#[cfg(not(all(feature = "v4", feature = "v2-gpu-metal")))]
fn main() {
    println!("This example requires features: v4,v2-gpu-metal");
    println!("Run with: cargo run --features v4,v2-gpu-metal --example test_v4_metal_pack_simple");
}
