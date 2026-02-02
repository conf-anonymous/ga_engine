// Minimal test to verify Metal rescaling matches CPU rescaling 
// Run with: cargo run --release --features v2,v2-gpu-metal --example test_metal_rescale

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
    },
    backends::cpu_optimized::{
        keys::KeyContext,
        ckks::CkksContext,
    },
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Metal Rescale Test");
    println!("==================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();

    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let cpu_ctx = CkksContext::new(params.clone());

    // Encrypt a simple value
    let value = 42.0;
    
    // CPU path
    let cpu_pt = cpu_ctx.encode(&[value]);
    let cpu_ct = cpu_ctx.encrypt(&cpu_pt, &pk);
    println!("CPU: Encrypted at level={}, scale={}", cpu_ct.level, cpu_ct.scale);
    
    // Decrypt immediately (no operations)
    let cpu_decrypted = cpu_ctx.decrypt(&cpu_ct, &sk);
    let cpu_result = cpu_ctx.decode(&cpu_decrypted);
    println!("CPU: Decrypted (no ops): {} (expected {})", cpu_result[0], value);
    println!("CPU error: {:.2e}\n", (cpu_result[0] - value).abs());

    // Metal path  
    let metal_pt = metal_ctx.encode(&[value])?;
    let metal_ct = metal_ctx.encrypt(&metal_pt, &pk)?;
    println!("Metal: Encrypted at level={}, scale={}", metal_ct.level, metal_ct.scale);
    
    // Decrypt immediately (no operations)
    let metal_decrypted = metal_ctx.decrypt(&metal_ct, &sk)?;
    let metal_result = metal_ctx.decode(&metal_decrypted)?;
    println!("Metal: Decrypted (no ops): {} (expected {})", metal_result[0], value);
    println!("Metal error: {:.2e}\n", (metal_result[0] - value).abs());

    // Test rescale directly on the ciphertext
    println!("Testing direct rescale...");
    let n = params.n;
    let level = metal_ct.level;
    let num_primes = level + 1;
    
    // Rescale c0 
    let c0_rescaled = metal_ctx.exact_rescale_gpu(&metal_ct.c0, level)?;
    println!("Rescaled c0: len={} (was {})", c0_rescaled.len(), metal_ct.c0.len());
    
    // Show first few coefficients before and after
    println!("\nBefore rescale (c0[0] across {} primes):", num_primes);
    for j in 0..num_primes {
        println!("  prime[{}]: {}", j, metal_ct.c0[0 * num_primes + j]);
    }
    println!("\nAfter rescale (c0[0] across {} primes):", num_primes - 1);
    for j in 0..(num_primes - 1) {
        println!("  prime[{}]: {}", j, c0_rescaled[0 * (num_primes - 1) + j]);
    }
    
    Ok(())
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
