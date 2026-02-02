//! Debug tensor product without relinearization

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::ckks::MetalCkksContext,
    backends::cpu_optimized::keys::KeyContext,
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Debug: Tensor Product Without Relinearization");
    println!("==============================================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    
    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];
    
    // Encrypt 2 twice
    let pt_a = metal_ctx.encode(&[2.0])?;
    let pt_b = metal_ctx.encode(&[2.0])?;
    let ct_a = metal_ctx.encrypt(&pt_a, &pk)?;
    let ct_b = metal_ctx.encrypt(&pt_b, &pk)?;
    
    println!("Encrypted a=2, b=2");
    
    // Verify decryption
    let dec_a = metal_ctx.decrypt(&ct_a, &sk)?;
    let dec_b = metal_ctx.decrypt(&ct_b, &sk)?;
    println!("Decrypted a: {}", metal_ctx.decode(&dec_a)?[0]);
    println!("Decrypted b: {}", metal_ctx.decode(&dec_b)?[0]);
    
    // MANUAL tensor product (skip relinearization)
    println!("\nComputing tensor product manually...");
    
    // c0_a * c0_b
    let c00 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&ct_a.c0, &ct_b.c0, moduli)?;
    // c0_a * c1_b + c1_a * c0_b
    let c01 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&ct_a.c0, &ct_b.c1, moduli)?;
    let c10 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&ct_a.c1, &ct_b.c0, moduli)?;
    // c1_a * c1_b
    let c11 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&ct_a.c1, &ct_b.c1, moduli)?;
    
    // c1 = c01 + c10
    let mut c1 = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        c1[i] = ((c01[i] as u128 + c10[i] as u128) % q as u128) as u64;
    }
    
    println!("Tensor product components:");
    println!("  c0[0] (c0_a*c0_b): {:?}", (0..num_primes).map(|j| c00[j]).collect::<Vec<_>>());
    println!("  c1[0] (cross terms): {:?}", (0..num_primes).map(|j| c1[j]).collect::<Vec<_>>());
    println!("  c2[0] (c1_a*c1_b): {:?}", (0..num_primes).map(|j| c11[j]).collect::<Vec<_>>());
    
    // Now decrypt the degree-2 ciphertext: m = c0 + c1*s + c2*s²
    let mut sk_flat = vec![0u64; n * num_primes];
    for i in 0..n {
        for j in 0..num_primes {
            sk_flat[i * num_primes + j] = sk.coeffs[i].values[j];
        }
    }
    
    // c1 * s
    let c1s = metal_ctx.multiply_polys_flat_ntt_negacyclic(&c1, &sk_flat, moduli)?;
    
    // s²
    let s2 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&sk_flat, &sk_flat, moduli)?;
    
    // c2 * s²
    let c2s2 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&c11, &s2, moduli)?;
    
    // m = c0 + c1*s + c2*s² (at scale²)
    let mut m_flat = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        m_flat[i] = ((c00[i] as u128 + c1s[i] as u128 + c2s2[i] as u128) % q as u128) as u64;
    }
    
    // Create plaintext with correct scale
    let new_scale = ct_a.scale * ct_b.scale;
    let pt_result = ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalPlaintext {
        coeffs: m_flat,
        n,
        num_primes,
        level,
        scale: new_scale,
    };
    
    println!("\nDecoding result (scale = {:.2e}):", new_scale);
    let result = metal_ctx.decode(&pt_result)?;
    let expected = 4.0;  // 2 * 2
    
    println!("  Got: {}", result[0]);
    println!("  Expected: {}", expected);
    println!("  Error: {:.2e}", (result[0] - expected).abs());
    
    if (result[0] - expected).abs() < 1.0 {
        println!("\n✅ Tensor product (degree-2 decrypt) WORKS!");
        Ok(())
    } else {
        println!("\n❌ Tensor product FAILED!");
        Err("Tensor product failed".to_string())
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
