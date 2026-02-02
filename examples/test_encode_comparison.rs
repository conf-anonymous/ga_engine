//! Compare CPU vs Metal encoding

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::{CkksContext, Plaintext};

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This test requires Metal GPU.");
}

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║               CPU vs Metal Encoding Comparison                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();

    let cpu_ckks = CkksContext::new(params.clone());
    let metal_ckks = MetalCkksContext::new(params.clone())?;

    // Encode the same value
    let value = 42.0;
    println!("Encoding value: {}\n", value);

    let pt_cpu = cpu_ckks.encode(&[value]);
    let pt_metal = metal_ckks.encode(&[value])?;

    println!("CPU Plaintext:");
    println!("  scale: {}", pt_cpu.scale);
    println!("  level: {}", pt_cpu.level);
    println!("  n: {}", pt_cpu.n);
    println!("  First few coeffs (prime 0):");
    for i in 0..5 {
        println!("    coeff[{}] = {}", i, pt_cpu.coeffs[i].values[0]);
    }

    println!("\nMetal Plaintext:");
    println!("  scale: {}", pt_metal.scale);
    println!("  level: {}", pt_metal.level);
    println!("  n: {}", pt_metal.n);
    println!("  First few coeffs (prime 0, from flat layout):");
    for i in 0..5 {
        // Flat layout: [coeff0_q0, coeff0_q1, coeff0_q2, coeff1_q0, ...]
        println!("    coeff[{}] = {}", i, pt_metal.coeffs[i * pt_metal.num_primes]);
    }

    // Compare
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Comparison");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut all_match = true;
    for i in 0..params.n {
        let cpu_val = pt_cpu.coeffs[i].values[0];
        let metal_val = pt_metal.coeffs[i * pt_metal.num_primes];
        if cpu_val != metal_val {
            if all_match {
                println!("❌ MISMATCH at coeff {}:", i);
                println!("   CPU:   {}", cpu_val);
                println!("   Metal: {}", metal_val);
            }
            all_match = false;
        }
    }

    if all_match {
        println!("✅ SUCCESS: CPU and Metal encoding match perfectly!");
    } else {
        println!("\n❌ FAILED: Encoding differs!");
    }

    Ok(())
}
