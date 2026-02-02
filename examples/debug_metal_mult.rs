//! Deep debug of Metal multiplication flow

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::{MetalCkksContext, MetalCiphertext},
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
    println!("Deep Debug: Metal Multiplication Flow");
    println!("======================================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let device = Arc::new(MetalDevice::new()?);
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let ntt_contexts = metal_ctx.ntt_contexts();
    let metal_relin_keys = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        20,
    )?;
    
    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];
    
    // Test 1: Verify that mul(ct, 1) gives back the same ciphertext (identity test)
    println!("Test 1: Identity test - encrypt(x) * encrypt(1) should give x");
    let x = 42.0;
    let one = 1.0;
    
    let pt_x = metal_ctx.encode(&[x])?;
    let pt_one = metal_ctx.encode(&[one])?;
    let ct_x = metal_ctx.encrypt(&pt_x, &pk)?;
    let ct_one = metal_ctx.encrypt(&pt_one, &pk)?;
    
    // Verify initial values
    let dec_x = metal_ctx.decrypt(&ct_x, &sk)?;
    let dec_one = metal_ctx.decrypt(&ct_one, &sk)?;
    println!("  Encrypted x={}: decrypted={}", x, metal_ctx.decode(&dec_x)?[0]);
    println!("  Encrypted 1={}: decrypted={}", one, metal_ctx.decode(&dec_one)?[0]);
    
    // Multiply
    let ct_product = ct_x.multiply(&ct_one, &metal_relin_keys, &metal_ctx)?;
    let dec_product = metal_ctx.decrypt(&ct_product, &sk)?;
    let product_val = metal_ctx.decode(&dec_product)?[0];
    
    println!("  Product (x * 1): {}", product_val);
    println!("  Expected: {}", x * one);
    println!("  Error: {:.2e}", (product_val - x * one).abs());
    
    if (product_val - x * one).abs() < 1.0 {
        println!("  ✅ Identity test PASSED\n");
    } else {
        println!("  ❌ Identity test FAILED\n");
    }
    
    // Test 2: Simple squaring - encrypt(2) * encrypt(2) = 4
    println!("Test 2: Squaring test - encrypt(2) * encrypt(2) should give 4");
    let two = 2.0;
    
    let pt_two = metal_ctx.encode(&[two])?;
    let ct_two = metal_ctx.encrypt(&pt_two, &pk)?;
    
    let dec_two = metal_ctx.decrypt(&ct_two, &sk)?;
    println!("  Encrypted 2: decrypted={}", metal_ctx.decode(&dec_two)?[0]);
    
    // Multiply
    let ct_squared = ct_two.multiply(&ct_two, &metal_relin_keys, &metal_ctx)?;
    
    // Debug: check intermediate c0/c1 values
    println!("  After mult: level={}, num_primes={}", ct_squared.level, ct_squared.num_primes);
    println!("  c0[0] across {} primes:", ct_squared.num_primes);
    for j in 0..ct_squared.num_primes {
        println!("    prime[{}]: {}", j, ct_squared.c0[0 * ct_squared.num_primes + j]);
    }
    
    let dec_squared = metal_ctx.decrypt(&ct_squared, &sk)?;
    let squared_val = metal_ctx.decode(&dec_squared)?[0];
    
    println!("  Result (2 * 2): {}", squared_val);
    println!("  Expected: {}", two * two);
    println!("  Error: {:.2e}", (squared_val - two * two).abs());
    
    if (squared_val - two * two).abs() < 1.0 {
        println!("  ✅ Squaring test PASSED\n");
    } else {
        println!("  ❌ Squaring test FAILED\n");
    }
    
    Ok(())
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
