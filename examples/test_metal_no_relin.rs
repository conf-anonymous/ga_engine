//! Test Metal multiplication WITHOUT relinearization
//! This isolates whether the issue is in tensor product or relin/rescale

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::{MetalCkksContext, MetalCiphertext},
    },
    backends::cpu_optimized::{
        keys::KeyContext,
        ckks::CkksContext,
    },
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Metal Tensor Product Test (no relin)");
    println!("=====================================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    
    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];
    
    // Encrypt 2.0 and 3.0
    let a = 2.0;
    let b = 3.0;
    
    let metal_pt_a = metal_ctx.encode(&[a])?;
    let metal_pt_b = metal_ctx.encode(&[b])?;
    let metal_ct_a = metal_ctx.encrypt(&metal_pt_a, &pk)?;
    let metal_ct_b = metal_ctx.encrypt(&metal_pt_b, &pk)?;
    
    println!("Encrypted a={}, b={}", a, b);
    println!("Level: {}, num_primes: {}\n", level, num_primes);
    
    // Decrypt to verify encryption works
    let dec_a = metal_ctx.decrypt(&metal_ct_a, &sk)?;
    let dec_b = metal_ctx.decrypt(&metal_ct_b, &sk)?;
    let val_a = metal_ctx.decode(&dec_a)?[0];
    let val_b = metal_ctx.decode(&dec_b)?[0];
    println!("Decrypted: a={}, b={}", val_a, val_b);
    
    // Tensor product (WITHOUT relinearization)
    println!("\nComputing tensor product...");
    
    // c0 × d0
    let ct0_ct0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(
        &metal_ct_a.c0,
        &metal_ct_b.c0,
        moduli,
    )?;
    
    // c0 × d1
    let ct0_ct1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(
        &metal_ct_a.c0,
        &metal_ct_b.c1,
        moduli,
    )?;
    
    // c1 × d0
    let ct1_ct0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(
        &metal_ct_a.c1,
        &metal_ct_b.c0,
        moduli,
    )?;
    
    // c1 × d1 (this is c2 component)
    let c2 = metal_ctx.multiply_polys_flat_ntt_negacyclic(
        &metal_ct_a.c1,
        &metal_ct_b.c1,
        moduli,
    )?;
    
    // ct0_ct1 + ct1_ct0 (middle term)
    let mut ct1_tensor = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = params.moduli[i % num_primes];
        ct1_tensor[i] = ((ct0_ct1[i] as u128 + ct1_ct0[i] as u128) % q as u128) as u64;
    }
    
    println!("Tensor product complete.");
    println!("  c0[0]: {:?}", (0..num_primes).map(|j| ct0_ct0[0*num_primes+j]).collect::<Vec<_>>());
    println!("  c1[0]: {:?}", (0..num_primes).map(|j| ct1_tensor[0*num_primes+j]).collect::<Vec<_>>());
    println!("  c2[0]: {:?}", (0..num_primes).map(|j| c2[0*num_primes+j]).collect::<Vec<_>>());
    
    // Now decrypt the degree-2 ciphertext manually:
    // m = c0 + c1*s + c2*s^2
    
    // Get secret key in flat form
    let mut sk_flat = vec![0u64; n * num_primes];
    for i in 0..n {
        for j in 0..num_primes {
            sk_flat[i * num_primes + j] = sk.coeffs[i].values[j];
        }
    }
    
    // Compute c1*s
    let c1_times_s = metal_ctx.multiply_polys_flat_ntt_negacyclic(&ct1_tensor, &sk_flat, moduli)?;
    
    // Compute s^2
    let s_squared = metal_ctx.multiply_polys_flat_ntt_negacyclic(&sk_flat, &sk_flat, moduli)?;
    
    // Compute c2*s^2
    let c2_times_s2 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&c2, &s_squared, moduli)?;
    
    // m = c0 + c1*s + c2*s^2
    let mut m_flat = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = params.moduli[i % num_primes];
        let sum = (ct0_ct0[i] as u128 + c1_times_s[i] as u128 + c2_times_s2[i] as u128) % q as u128;
        m_flat[i] = sum as u64;
    }
    
    // Create plaintext and decode
    let pt_result = ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalPlaintext {
        coeffs: m_flat,
        n,
        num_primes,
        level,
        scale: metal_ct_a.scale * metal_ct_b.scale,  // Scale doubles after mult
    };
    
    let result = metal_ctx.decode(&pt_result)?;
    let expected = a * b;
    
    println!("\nDegree-2 decryption result:");
    println!("  Expected: {}", expected);
    println!("  Got:      {}", result[0]);
    println!("  Error:    {:.2e}", (result[0] - expected).abs());
    
    // The result should be correct modulo the larger scale
    // We expect: scale^2 / scale = scale
    // So result should be: value * scale^2 decoded with scale^2 = value * scale
    println!("  Scale after mult: {:.2e}", pt_result.scale);
    
    // The scale is squared, so the decoded value is scaled up by scale
    // We need to account for this
    let scale = metal_ct_a.scale;
    let actual_result = result[0] / scale;  // Unscale
    println!("\n  After unscaling by Δ:");
    println!("  Expected: {}", expected / scale);  
    println!("  Got:      {}", actual_result);
    
    // Actually the tensor product gives us value*scale^2, and we decode with scale^2
    // So the result should just be the value directly
    
    if (result[0] - expected).abs() < 1.0 {
        println!("\n✅ Tensor product works correctly!");
        Ok(())
    } else {
        println!("\n❌ Tensor product failed!");
        Err("Tensor product failed".to_string())
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
