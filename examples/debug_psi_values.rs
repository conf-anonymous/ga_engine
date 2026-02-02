//! Debug psi values used in negacyclic convolution

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::ckks::MetalCkksContext,
    backends::cpu_optimized::ntt::NttContext,
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Debug PSI Values for Negacyclic NTT");
    println!("====================================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    
    let n = params.n;
    let num_primes = 3;
    let moduli = &params.moduli[..num_primes];
    
    // Get Metal NTT contexts
    let metal_ntts = metal_ctx.ntt_contexts();
    
    // Get CPU NTT contexts
    let cpu_ntts: Vec<NttContext> = moduli.iter()
        .map(|&q| NttContext::new(n, q))
        .collect();
    
    println!("Comparing PSI powers for prime 0:");
    println!("  n = {}", n);
    println!("  q = {}\n", moduli[0]);
    
    // Get first few psi powers from Metal
    println!("Metal PSI powers (first 8):");
    for i in 0..8.min(n) {
        print!("  psi[{}] = {}", i, metal_ntts[0].psi_powers()[i]);
        if i > 0 {
            let expected = metal_ntts[0].psi_powers()[0] * i as u64;
            print!("  (expected linear: {})", expected);
        }
        println!();
    }
    
    println!("\nCPU PSI powers (first 8):");
    for i in 0..8.min(n) {
        print!("  psi[{}] = {}", i, cpu_ntts[0].psi_powers()[i]);
        if i > 0 {
            let expected = cpu_ntts[0].psi_powers()[0] * i as u64;
            print!("  (expected linear: {})", expected);
        }
        println!();
    }
    
    // Compare first 8 values
    println!("\nComparison (Metal == CPU):");
    let mut matches = 0;
    let mut mismatches = 0;
    for i in 0..8 {
        let metal_val = metal_ntts[0].psi_powers()[i];
        let cpu_val = cpu_ntts[0].psi_powers()[i];
        if metal_val == cpu_val {
            matches += 1;
            println!("  psi[{}]: ✅ {} == {}", i, metal_val, cpu_val);
        } else {
            mismatches += 1;
            println!("  psi[{}]: ❌ {} != {}", i, metal_val, cpu_val);
        }
    }
    
    println!("\n{} matches, {} mismatches", matches, mismatches);
    
    if mismatches > 0 {
        Err("PSI power mismatch".to_string())
    } else {
        Ok(())
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
