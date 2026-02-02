//! Test that EVK satisfies the relinearization identity:
//! evk0 - evk1*s = -B^t*s² + noise

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::KeyContext,
        ntt::NttContext,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Test: EVK Relinearization Identity");
    println!("===================================\n");

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
        20,  // base_w = 20
    )?;
    
    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];
    let base_w = 20u32;
    
    println!("Testing identity: evk0[t] - evk1[t]*s = -B^t*s² + noise");
    println!("For t=0, B=2^{}, B^0=1\n", base_w);
    
    // Test CPU EVK
    println!("=== CPU EVK ===");
    test_evk_identity_cpu(&cpu_evk, &sk, moduli, n)?;
    
    // Test Metal EVK
    println!("\n=== Metal EVK ===");
    let (metal_evk0, metal_evk1) = metal_evk.get_coeff_keys(level)?;
    test_evk_identity_metal(metal_evk0, metal_evk1, &sk, moduli, n)?;
    
    Ok(())
}

#[cfg(feature = "v2-gpu-metal")]
fn test_evk_identity_cpu(
    evk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::EvaluationKey,
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    moduli: &[u64],
    n: usize,
) -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
    
    // For t=0, B^0=1, so we expect evk0[0] - evk1[0]*s = -s² + noise
    let t = 0;
    let evk0 = &evk.evk0[t];
    let evk1 = &evk.evk1[t];
    
    // Compute evk1*s using NTT multiplication
    let mut evk1_times_s = Vec::with_capacity(n);
    for prime_idx in 0..moduli.len() {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);
        
        let evk1_poly: Vec<u64> = evk1.iter().map(|rns| rns.values[prime_idx]).collect();
        let s_poly: Vec<u64> = sk.coeffs.iter().map(|rns| rns.values[prime_idx]).collect();
        let product = ntt_ctx.multiply_polynomials(&evk1_poly, &s_poly);
        
        if prime_idx == 0 {
            for coeff_idx in 0..n {
                evk1_times_s.push(RnsRepresentation::new(vec![product[coeff_idx]; moduli.len()], moduli.to_vec()));
            }
        } else {
            for coeff_idx in 0..n {
                evk1_times_s[coeff_idx].values[prime_idx] = product[coeff_idx];
            }
        }
    }
    
    // Compute evk0 - evk1*s
    let mut diff = Vec::with_capacity(n);
    for coeff_idx in 0..n {
        let mut d = vec![0u64; moduli.len()];
        for (prime_idx, &q) in moduli.iter().enumerate() {
            let e0 = evk0[coeff_idx].values[prime_idx];
            let e1s = evk1_times_s[coeff_idx].values[prime_idx];
            d[prime_idx] = if e0 >= e1s { e0 - e1s } else { q - (e1s - e0) };
        }
        diff.push(d);
    }
    
    // Compute -s² = -(s*s)
    let mut neg_s_squared = Vec::with_capacity(n);
    for prime_idx in 0..moduli.len() {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);
        
        let s_poly: Vec<u64> = sk.coeffs.iter().map(|rns| rns.values[prime_idx]).collect();
        let s_sq = ntt_ctx.multiply_polynomials(&s_poly, &s_poly);
        
        // Negate
        let neg_s_sq: Vec<u64> = s_sq.iter().map(|&v| if v == 0 { 0 } else { q - v }).collect();
        
        if prime_idx == 0 {
            for coeff_idx in 0..n {
                neg_s_squared.push(vec![neg_s_sq[coeff_idx]; moduli.len()]);
            }
        } else {
            for coeff_idx in 0..n {
                neg_s_squared[coeff_idx][prime_idx] = neg_s_sq[coeff_idx];
            }
        }
    }
    
    // Check if diff ≈ -s² (modulo noise)
    println!("  First 3 coefficients comparison:");
    for coeff_idx in 0..3.min(n) {
        let d0 = diff[coeff_idx][0];
        let neg_s2_0 = neg_s_squared[coeff_idx][0];
        let q = moduli[0];
        
        // Compute difference (should be small noise)
        let noise = if d0 >= neg_s2_0 { d0 - neg_s2_0 } else { q - (neg_s2_0 - d0) };
        let noise_centered = if noise > q/2 { noise as i64 - q as i64 } else { noise as i64 };
        
        println!("    coeff[{}]: diff={}, -s²={}, noise={}", 
            coeff_idx, d0, neg_s2_0, noise_centered);
    }
    
    // Check noise bound
    let mut max_noise: i64 = 0;
    for coeff_idx in 0..n {
        let d0 = diff[coeff_idx][0];
        let neg_s2_0 = neg_s_squared[coeff_idx][0];
        let q = moduli[0];
        
        let noise = if d0 >= neg_s2_0 { d0 - neg_s2_0 } else { q - (neg_s2_0 - d0) };
        let noise_centered = if noise > q/2 { noise as i64 - q as i64 } else { noise as i64 };
        max_noise = max_noise.max(noise_centered.abs());
    }
    
    println!("  Max noise: {} (should be small, ~O(sqrt(n)*std_dev))", max_noise);
    
    if max_noise < 1_000_000 {
        println!("  ✅ CPU EVK identity verified!\n");
        Ok(())
    } else {
        println!("  ❌ CPU EVK identity FAILED!\n");
        Err("CPU EVK identity failed".to_string())
    }
}

#[cfg(feature = "v2-gpu-metal")]
fn test_evk_identity_metal(
    evk0: &[Vec<u64>],
    evk1: &[Vec<u64>],
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    moduli: &[u64],
    n: usize,
) -> Result<(), String> {
    let num_primes = moduli.len();
    
    // For t=0
    let t = 0;
    let evk0_flat = &evk0[t];
    let evk1_flat = &evk1[t];
    
    // Compute evk1*s using NTT multiplication
    let mut evk1_times_s = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);
        
        // Extract polynomials
        let mut evk1_poly = vec![0u64; n];
        let mut s_poly = vec![0u64; n];
        for coeff_idx in 0..n {
            evk1_poly[coeff_idx] = evk1_flat[coeff_idx * num_primes + prime_idx];
            s_poly[coeff_idx] = sk.coeffs[coeff_idx].values[prime_idx];
        }
        
        let product = ntt_ctx.multiply_polynomials(&evk1_poly, &s_poly);
        
        for coeff_idx in 0..n {
            evk1_times_s[coeff_idx * num_primes + prime_idx] = product[coeff_idx];
        }
    }
    
    // Compute evk0 - evk1*s
    let mut diff = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        let e0 = evk0_flat[i];
        let e1s = evk1_times_s[i];
        diff[i] = if e0 >= e1s { e0 - e1s } else { q - (e1s - e0) };
    }
    
    // Compute -s²
    let mut neg_s_squared = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);
        
        let mut s_poly = vec![0u64; n];
        for coeff_idx in 0..n {
            s_poly[coeff_idx] = sk.coeffs[coeff_idx].values[prime_idx];
        }
        
        let s_sq = ntt_ctx.multiply_polynomials(&s_poly, &s_poly);
        
        for coeff_idx in 0..n {
            let v = s_sq[coeff_idx];
            neg_s_squared[coeff_idx * num_primes + prime_idx] = if v == 0 { 0 } else { q - v };
        }
    }
    
    // Check if diff ≈ -s² (modulo noise)
    println!("  First 3 coefficients comparison:");
    for coeff_idx in 0..3.min(n) {
        let d0 = diff[coeff_idx * num_primes + 0];
        let neg_s2_0 = neg_s_squared[coeff_idx * num_primes + 0];
        let q = moduli[0];
        
        let noise = if d0 >= neg_s2_0 { d0 - neg_s2_0 } else { q - (neg_s2_0 - d0) };
        let noise_centered = if noise > q/2 { noise as i64 - q as i64 } else { noise as i64 };
        
        println!("    coeff[{}]: diff={}, -s²={}, noise={}", 
            coeff_idx, d0, neg_s2_0, noise_centered);
    }
    
    // Check noise bound
    let mut max_noise: i64 = 0;
    for coeff_idx in 0..n {
        let d0 = diff[coeff_idx * num_primes + 0];
        let neg_s2_0 = neg_s_squared[coeff_idx * num_primes + 0];
        let q = moduli[0];
        
        let noise = if d0 >= neg_s2_0 { d0 - neg_s2_0 } else { q - (neg_s2_0 - d0) };
        let noise_centered = if noise > q/2 { noise as i64 - q as i64 } else { noise as i64 };
        max_noise = max_noise.max(noise_centered.abs());
    }
    
    println!("  Max noise: {} (should be small, ~O(sqrt(n)*std_dev))", max_noise);
    
    if max_noise < 1_000_000 {
        println!("  ✅ Metal EVK identity verified!\n");
        Ok(())
    } else {
        println!("  ❌ Metal EVK identity FAILED!\n");
        Err("Metal EVK identity failed".to_string())
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
