//! Test multiply_plain with CPU → Metal conversion

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This test requires Metal GPU.");
}

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Test multiply_plain with CPU ↔ Metal conversion             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();

    // Generate keys on CPU
    println!("Generating keys on CPU...");
    let cpu_key_ctx = KeyContext::new(params.clone());
    let (cpu_pk, cpu_sk, _evk) = cpu_key_ctx.keygen();

    // Create contexts
    let cpu_ckks = CkksContext::new(params.clone());
    let metal_ckks = MetalCkksContext::new(params.clone())?;

    println!("Keys generated!\n");

    // ==================== CPU PATH ====================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("CPU Path (reference)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let pt1_cpu = cpu_ckks.encode(&[42.0]);
    let ct_cpu = cpu_ckks.encrypt(&pt1_cpu, &cpu_pk);

    println!("Encrypted 42.0");
    let pt_back_cpu = cpu_ckks.decrypt(&ct_cpu, &cpu_sk);
    let decoded_cpu = cpu_ckks.decode(&pt_back_cpu);
    println!("  Decrypted: {}", decoded_cpu[0]);

    let pt_mult_cpu = cpu_ckks.encode(&[2.0]);
    let ct_mult_cpu = ct_cpu.multiply_plain(&pt_mult_cpu, &cpu_ckks);

    let pt_result_cpu = cpu_ckks.decrypt(&ct_mult_cpu, &cpu_sk);
    let result_cpu = cpu_ckks.decode(&pt_result_cpu);

    println!("\nAfter multiply_plain by 2.0:");
    println!("  Expected: 84.0");
    println!("  Got: {}", result_cpu[0]);
    println!("  Error: {:.2e}", (result_cpu[0] - 84.0).abs());

    // ==================== METAL PATH ====================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Metal Path (test)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Convert CPU ciphertext to Metal
    println!("Converting CPU ciphertext to Metal...");
    let ct_metal = metal_ckks.from_cpu_ciphertext(&ct_cpu);

    // Convert plaintext to Metal
    let pt_mult_metal = metal_ckks.from_cpu_plaintext(&pt_mult_cpu);

    // Do multiply_plain on Metal
    println!("Performing multiply_plain on Metal...");
    let ct_mult_metal = ct_metal.multiply_plain(&pt_mult_metal, &metal_ckks)?;

    // Convert back to CPU
    println!("Converting result back to CPU...");
    let ct_mult_back_cpu = metal_ckks.to_cpu_ciphertext(&ct_mult_metal);

    // Decrypt on CPU
    let pt_result_metal = cpu_ckks.decrypt(&ct_mult_back_cpu, &cpu_sk);
    let result_metal = cpu_ckks.decode(&pt_result_metal);

    println!("\nAfter Metal multiply_plain by 2.0:");
    println!("  Expected: 84.0");
    println!("  Got: {}", result_metal[0]);
    println!("  Error: {:.2e}", (result_metal[0] - 84.0).abs());

    // ==================== COMPARISON ====================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Comparison");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let cpu_error = (result_cpu[0] - 84.0).abs();
    let metal_error = (result_metal[0] - 84.0).abs();

    println!("CPU error:   {:.2e}", cpu_error);
    println!("Metal error: {:.2e}", metal_error);

    if metal_error < 1.0 {
        println!("\n✅ SUCCESS: Metal multiply_plain works!");
    } else if cpu_error < 1.0 {
        println!("\n❌ FAILED: CPU works but Metal broken");
        println!("   Error ratio: {:.2}×", metal_error / cpu_error);
    } else {
        println!("\n⚠️  Both CPU and Metal have errors");
    }

    Ok(())
}
