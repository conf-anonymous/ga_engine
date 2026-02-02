//! Verify EVK is in coefficient form (not NTT form)

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::keys::KeyContext,
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Test: Verify EVK is in Coefficient Form");
    println!("========================================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let device = Arc::new(MetalDevice::new()?);
    let key_ctx = KeyContext::new(params.clone());
    let (_, sk, cpu_evk) = key_ctx.keygen();
    
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let ntt_contexts = metal_ctx.ntt_contexts();
    let metal_evk = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        20,
    )?;
    
    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    
    // Get Metal coefficient keys
    let (metal_evk0_coeff, metal_evk1_coeff) = metal_evk.get_coeff_keys(level)?;
    
    // Get Metal NTT keys
    let (metal_evk0_ntt, metal_evk1_ntt) = metal_evk.get_ntt_keys(level)?;
    
    // Check: coefficient form should be different from NTT form
    // (unless by coincidence, which is extremely unlikely)
    
    let t = 0;  // First digit
    println!("Comparing coefficient vs NTT form for digit 0:");
    
    println!("\nCoefficient form (evk0[0], first 4 coeffs × 3 primes):");
    for coeff_idx in 0..4.min(n) {
        print!("  coeff[{}]: ", coeff_idx);
        for prime_idx in 0..num_primes {
            print!("{} ", metal_evk0_coeff[t][coeff_idx * num_primes + prime_idx]);
        }
        println!();
    }
    
    println!("\nNTT form (evk0_ntt[0], first 4 coeffs × 3 primes):");
    for coeff_idx in 0..4.min(n) {
        print!("  coeff[{}]: ", coeff_idx);
        for prime_idx in 0..num_primes {
            print!("{} ", metal_evk0_ntt[t][coeff_idx * num_primes + prime_idx]);
        }
        println!();
    }
    
    // Check if they're different
    let same = (0..n*num_primes).all(|i| 
        metal_evk0_coeff[t][i] == metal_evk0_ntt[t][i]
    );
    
    if same {
        println!("\n❌ PROBLEM: Coefficient and NTT forms are IDENTICAL!");
        println!("   This means either NTT wasn't applied, or coefficient keys are actually NTT form.");
        Err("Forms should be different".to_string())
    } else {
        println!("\n✅ Coefficient and NTT forms are different (as expected).");
        Ok(())
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
