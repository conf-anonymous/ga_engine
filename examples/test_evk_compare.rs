//! Compare CPU and Metal EVK generation

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
    println!("EVK Comparison Test");
    println!("===================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let device = Arc::new(MetalDevice::new()?);
    let key_ctx = KeyContext::new(params.clone());
    let (_, sk, evk) = key_ctx.keygen();
    
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let ntt_contexts = metal_ctx.ntt_contexts();
    
    // Generate Metal EVK with same base_w as CPU
    let metal_evk = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        20,  // Same base_w as CPU
    )?;
    
    let n = params.n;
    let level = 2;  // Test at level 2
    let num_primes = level + 1;
    
    println!("Comparing EVK at level {} ({} primes)", level, num_primes);
    println!();
    
    // Get CPU EVK (it has all primes, we need to extract for level)
    println!("CPU EVK (first digit):");
    print!("  evk0[0][coeff=0] across {} primes: ", num_primes);
    for j in 0..num_primes {
        print!("{} ", evk.evk0[0][0].values[j]);
    }
    println!();
    print!("  evk1[0][coeff=0] across {} primes: ", num_primes);
    for j in 0..num_primes {
        print!("{} ", evk.evk1[0][0].values[j]);
    }
    println!();
    
    // Get Metal EVK
    let (metal_evk0, metal_evk1) = metal_evk.get_coeff_keys(level)?;
    println!("\nMetal EVK (first digit):");
    print!("  evk0[0][coeff=0] across {} primes: ", num_primes);
    for j in 0..num_primes {
        print!("{} ", metal_evk0[0][0 * num_primes + j]);
    }
    println!();
    print!("  evk1[0][coeff=0] across {} primes: ", num_primes);
    for j in 0..num_primes {
        print!("{} ", metal_evk1[0][0 * num_primes + j]);
    }
    println!();
    
    // Check if they're the same (they won't be due to different randomness, but structure should match)
    println!("\nNote: Values differ due to independent random sampling.");
    println!("Both should produce correct relinearization independently.");
    
    // Now test that we can use the CPU EVK structure with Metal:
    // The key insight is that the EVK formula must be consistent.
    
    // Check: evk0[t] - evk1[t]*s = -B^t*s^2 + noise
    // If we use the same sk, both CPU and Metal EVK should satisfy this relation.
    
    println!("\nGadget parameters:");
    println!("  CPU base_w: {}", evk.base_w);
    let (metal_base_w, metal_num_digits) = metal_evk.gadget_params();
    println!("  Metal base_w: {}, num_digits: {}", metal_base_w, metal_num_digits);
    println!("  CPU num_digits: {}", evk.evk0.len());
    
    if evk.base_w != metal_base_w {
        println!("\n❌ MISMATCH: base_w differs!");
        return Err("base_w mismatch".to_string());
    }
    if evk.evk0.len() != metal_num_digits {
        println!("\n⚠️  Note: num_digits differs (CPU: {}, Metal: {})", evk.evk0.len(), metal_num_digits);
    }
    
    println!("\n✅ EVK structure looks consistent");
    Ok(())
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
