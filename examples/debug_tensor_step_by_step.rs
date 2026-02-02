//! Debug tensor product step by step

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::ckks::MetalCkksContext,
    backends::cpu_optimized::keys::KeyContext,
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Debug: Tensor Product Step by Step");
    println!("===================================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    
    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];
    
    // Encrypt 2
    let pt = metal_ctx.encode(&[2.0])?;
    let ct = metal_ctx.encrypt(&pt, &pk)?;
    
    println!("Encrypted value: 2");
    println!("Scale: {}\n", ct.scale);
    
    // Step 1: Decrypt ct directly (no operations)
    let dec_direct = metal_ctx.decrypt(&ct, &sk)?;
    let val_direct = metal_ctx.decode(&dec_direct)?[0];
    println!("Step 1 - Direct decrypt: {} (expected: 2)", val_direct);
    
    // Step 2: Manual decrypt: m = c0 + c1*s
    println!("\nStep 2 - Manual decrypt (c0 + c1*s):");
    
    // Secret key in flat form
    let mut sk_flat = vec![0u64; n * num_primes];
    for i in 0..n {
        for j in 0..num_primes {
            sk_flat[i * num_primes + j] = sk.coeffs[i].values[j];
        }
    }
    
    // c1 * s
    let c1s = metal_ctx.multiply_polys_flat_ntt_negacyclic(&ct.c1, &sk_flat, moduli)?;
    
    // m = c0 + c1*s
    let mut m_manual = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        m_manual[i] = ((ct.c0[i] as u128 + c1s[i] as u128) % q as u128) as u64;
    }
    
    // Decode manually
    let pt_manual = ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalPlaintext {
        coeffs: m_manual.clone(),
        n,
        num_primes,
        level,
        scale: ct.scale,
    };
    let val_manual = metal_ctx.decode(&pt_manual)?[0];
    println!("  Manual decrypt: {} (expected: 2)", val_manual);
    
    // Compare m coefficients
    println!("  m[0] across primes: {:?}", (0..num_primes).map(|j| m_manual[j]).collect::<Vec<_>>());
    println!("  dec_direct[0] across primes: {:?}", (0..num_primes).map(|j| dec_direct.coeffs[j]).collect::<Vec<_>>());
    
    // Step 3: Squared ciphertext components  
    println!("\nStep 3 - Tensor product for ct²:");
    
    // c0_sq = c0 * c0
    let c0_sq = metal_ctx.multiply_polys_flat_ntt_negacyclic(&ct.c0, &ct.c0, moduli)?;
    // c1_part = c0*c1 + c1*c0 = 2*c0*c1
    let c0c1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&ct.c0, &ct.c1, moduli)?;
    let mut c1_part = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        c1_part[i] = ((c0c1[i] as u128 * 2) % q as u128) as u64;
    }
    // c2 = c1 * c1
    let c2 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&ct.c1, &ct.c1, moduli)?;
    
    println!("  c0² [0]: {:?}", (0..num_primes).map(|j| c0_sq[j]).collect::<Vec<_>>());
    println!("  c1_part [0]: {:?}", (0..num_primes).map(|j| c1_part[j]).collect::<Vec<_>>());
    println!("  c2 [0]: {:?}", (0..num_primes).map(|j| c2[j]).collect::<Vec<_>>());
    
    // Step 4: Degree-2 decrypt: m = c0² + c1_part*s + c2*s²
    println!("\nStep 4 - Degree-2 decrypt (c0² + c1_part*s + c2*s²):");
    
    // c1_part * s
    let c1ps = metal_ctx.multiply_polys_flat_ntt_negacyclic(&c1_part, &sk_flat, moduli)?;
    
    // s²
    let s2 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&sk_flat, &sk_flat, moduli)?;
    
    // c2 * s²
    let c2s2 = metal_ctx.multiply_polys_flat_ntt_negacyclic(&c2, &s2, moduli)?;
    
    // m = c0² + c1_part*s + c2*s²
    let mut m_deg2 = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        let sum = (c0_sq[i] as u128 + c1ps[i] as u128 + c2s2[i] as u128) % q as u128;
        m_deg2[i] = sum as u64;
    }
    
    // The scale is ct.scale² after multiplication
    let new_scale = ct.scale * ct.scale;
    println!("  New scale: {:.2e}", new_scale);
    println!("  m_deg2[0] across primes: {:?}", (0..num_primes).map(|j| m_deg2[j]).collect::<Vec<_>>());
    
    // To decode correctly, we need to understand what the plaintext represents
    // After multiplication: m_encoded = round(4 * scale * scale / n)
    // But the decryption gives us: m_raw = m_encoded (approximately)
    // To get back 4, we need: decoded = m_raw * n / (scale²)
    
    // The decode function should handle this if we pass scale = scale²
    let pt_deg2 = ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalPlaintext {
        coeffs: m_deg2.clone(),
        n,
        num_primes,
        level,
        scale: new_scale,
    };
    let val_deg2 = metal_ctx.decode(&pt_deg2)?[0];
    println!("  Decoded (scale=scale²): {} (expected: 4)", val_deg2);
    
    // What if we just use original scale?
    let pt_deg2_orig_scale = ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalPlaintext {
        coeffs: m_deg2.clone(),
        n,
        num_primes,
        level,
        scale: ct.scale,
    };
    let val_deg2_orig = metal_ctx.decode(&pt_deg2_orig_scale)?[0];
    println!("  Decoded (scale=scale): {} (expected: 4*scale)", val_deg2_orig);
    
    // Print the expected value if encoding was: coeff[0] = round(value * scale / n)
    // For value=4, scale=ct.scale, n=1024: coeff[0] should be ~round(4 * 1099511627776 / 1024) = ~4294967295
    let expected_coeff = (4.0_f64 * ct.scale / n as f64).round() as u64;
    println!("\n  For comparison, encode(4) would give coeff[0] ≈ {}", expected_coeff);
    
    Ok(())
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
