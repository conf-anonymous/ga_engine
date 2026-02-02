//! Debug NTT multiplication in detail

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::ckks::MetalCkksContext,
    backends::cpu_optimized::{
        keys::KeyContext,
        ntt::NttContext,
    },
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Debug: NTT Multiplication Detail");
    println!("=================================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    
    let n = params.n;
    let num_primes = 3;
    let moduli = &params.moduli[..num_primes];
    
    // Create simple test polynomials: p(x) = 1 (constant)
    // When we multiply 1 * 1, we should get 1
    let mut p_flat = vec![0u64; n * num_primes];
    for j in 0..num_primes {
        p_flat[0 * num_primes + j] = 1;  // Constant term = 1
    }
    
    println!("Test: Multiply constant polynomial 1 * 1 = 1\n");
    
    // Metal NTT multiply
    let metal_result = metal_ctx.multiply_polys_flat_ntt_negacyclic(&p_flat, &p_flat, moduli)?;
    
    println!("Metal result[0] across primes: {:?}",
        (0..num_primes).map(|j| metal_result[0 * num_primes + j]).collect::<Vec<_>>());
    
    // CPU NTT multiply
    let mut cpu_results = vec![0u64; n * num_primes];
    for (j, &q) in moduli.iter().enumerate() {
        // Extract polynomials for this prime
        let mut a_poly = vec![0u64; n];
        for i in 0..n {
            a_poly[i] = p_flat[i * num_primes + j];
        }
        
        // CPU NTT multiply (negacyclic)
        let ntt_ctx = NttContext::new(n, q);
        let prod = ntt_ctx.multiply_polynomials(&a_poly, &a_poly);
        
        // Store back
        for i in 0..n {
            cpu_results[i * num_primes + j] = prod[i];
        }
    }
    
    println!("CPU result[0] across primes: {:?}",
        (0..num_primes).map(|j| cpu_results[0 * num_primes + j]).collect::<Vec<_>>());
    
    // Compare
    let all_match = (0..num_primes).all(|j| 
        metal_result[0 * num_primes + j] == cpu_results[0 * num_primes + j]
    );
    
    if all_match {
        println!("\n✅ Metal and CPU match for constant * constant\n");
    } else {
        println!("\n❌ Metal and CPU MISMATCH for constant * constant\n");
    }
    
    // Now test with actual ciphertext-like values
    println!("Test: Multiply actual ciphertext c0 * c0\n");
    
    // Encrypt a value to get realistic ciphertext
    let pt = metal_ctx.encode(&[42.0])?;
    let ct = metal_ctx.encrypt(&pt, &pk)?;
    
    // Metal NTT multiply on c0 * c0
    let metal_c0_sq = metal_ctx.multiply_polys_flat_ntt_negacyclic(&ct.c0, &ct.c0, moduli)?;
    
    println!("Metal c0²[0] across primes: {:?}",
        (0..num_primes).map(|j| metal_c0_sq[0 * num_primes + j]).collect::<Vec<_>>());
    
    // CPU NTT multiply on same c0
    let mut cpu_c0_sq = vec![0u64; n * num_primes];
    for (j, &q) in moduli.iter().enumerate() {
        let mut c0_poly = vec![0u64; n];
        for i in 0..n {
            c0_poly[i] = ct.c0[i * num_primes + j];
        }
        
        let ntt_ctx = NttContext::new(n, q);
        let prod = ntt_ctx.multiply_polynomials(&c0_poly, &c0_poly);
        
        for i in 0..n {
            cpu_c0_sq[i * num_primes + j] = prod[i];
        }
    }
    
    println!("CPU c0²[0] across primes: {:?}",
        (0..num_primes).map(|j| cpu_c0_sq[0 * num_primes + j]).collect::<Vec<_>>());
    
    // Compare first coefficient
    let c0_sq_match = (0..num_primes).all(|j|
        metal_c0_sq[0 * num_primes + j] == cpu_c0_sq[0 * num_primes + j]
    );
    
    if c0_sq_match {
        println!("\n✅ Metal and CPU match for c0² (first coeff)\n");
    } else {
        println!("\n❌ Metal and CPU MISMATCH for c0² (first coeff)\n");
        // Show diffs
        for j in 0..num_primes {
            let metal_val = metal_c0_sq[0 * num_primes + j];
            let cpu_val = cpu_c0_sq[0 * num_primes + j];
            if metal_val != cpu_val {
                println!("  prime[{}]: Metal={} vs CPU={}", j, metal_val, cpu_val);
            }
        }
    }
    
    // Check all n coefficients
    let mut mismatch_count = 0;
    for i in 0..n {
        for j in 0..num_primes {
            if metal_c0_sq[i * num_primes + j] != cpu_c0_sq[i * num_primes + j] {
                mismatch_count += 1;
                if mismatch_count <= 3 {
                    println!("Mismatch at coeff[{}] prime[{}]: Metal={} vs CPU={}",
                        i, j, metal_c0_sq[i * num_primes + j], cpu_c0_sq[i * num_primes + j]);
                }
            }
        }
    }
    
    println!("\nTotal mismatches: {} out of {} values", mismatch_count, n * num_primes);
    
    Ok(())
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
