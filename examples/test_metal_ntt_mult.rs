//! Test Metal NTT polynomial multiplication matches CPU

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::ckks::MetalCkksContext,
    backends::cpu_optimized::ntt::NttContext,
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Metal NTT Multiplication Test");
    println!("==============================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    
    let n = params.n;
    let num_primes = 3; // Use 3 primes for testing
    let moduli = &params.moduli[..num_primes];
    
    // Create simple test polynomials: a = [1, 2, 0, 0, ...], b = [3, 4, 0, 0, ...]
    // In negacyclic ring: (1 + 2x)(3 + 4x) = 3 + 4x + 6x + 8x^2 = 3 + 10x + 8x^2
    // But in x^n + 1 ring, x^n = -1, so we need to be careful
    
    // Actually let's just do constant polynomials for simplicity
    // a = 5 (constant), b = 7 (constant) => result = 35 (constant)
    
    // Create flat RNS polynomials (coefficient-major layout)
    let mut a_flat = vec![0u64; n * num_primes];
    let mut b_flat = vec![0u64; n * num_primes];
    
    // Set coeff[0] = 5 for a, coeff[0] = 7 for b (across all primes)
    for (j, &q) in moduli.iter().enumerate() {
        a_flat[0 * num_primes + j] = 5 % q;
        b_flat[0 * num_primes + j] = 7 % q;
    }
    
    println!("Input polynomials:");
    println!("  a[0] across {} primes: {:?}", num_primes, 
        (0..num_primes).map(|j| a_flat[0 * num_primes + j]).collect::<Vec<_>>());
    println!("  b[0] across {} primes: {:?}", num_primes,
        (0..num_primes).map(|j| b_flat[0 * num_primes + j]).collect::<Vec<_>>());
    
    // Metal NTT multiply (negacyclic)
    let result_metal = metal_ctx.multiply_polys_flat_ntt_negacyclic(&a_flat, &b_flat, moduli)?;
    
    println!("\nMetal result:");
    println!("  result[0] across {} primes: {:?}", num_primes,
        (0..num_primes).map(|j| result_metal[0 * num_primes + j]).collect::<Vec<_>>());
    
    // CPU NTT multiply (for comparison)
    let mut cpu_results = vec![0u64; n * num_primes];
    for (j, &q) in moduli.iter().enumerate() {
        // Extract polynomials for this prime
        let mut a_poly = vec![0u64; n];
        let mut b_poly = vec![0u64; n];
        for i in 0..n {
            a_poly[i] = a_flat[i * num_primes + j];
            b_poly[i] = b_flat[i * num_primes + j];
        }
        
        // CPU NTT multiply
        let ntt_ctx = NttContext::new(n, q);
        let prod = ntt_ctx.multiply_polynomials(&a_poly, &b_poly);
        
        // Store back
        for i in 0..n {
            cpu_results[i * num_primes + j] = prod[i];
        }
    }
    
    println!("\nCPU result:");
    println!("  result[0] across {} primes: {:?}", num_primes,
        (0..num_primes).map(|j| cpu_results[0 * num_primes + j]).collect::<Vec<_>>());
    
    // Check if they match
    let match_result = (0..num_primes).all(|j| result_metal[0 * num_primes + j] == cpu_results[0 * num_primes + j]);
    
    if match_result {
        println!("\n✅ Metal NTT matches CPU NTT!");
    } else {
        println!("\n❌ Metal NTT does NOT match CPU NTT!");
        return Err("NTT mismatch".to_string());
    }
    
    // Expected: 5 * 7 = 35 for constant polynomials
    println!("\nExpected: 35 (since 5 × 7 = 35)");
    let first_val = result_metal[0];
    if first_val == 35 {
        println!("✅ Got expected value 35");
        Ok(())
    } else {
        println!("Got: {} (expected 35)", first_val);
        Err(format!("Expected 35, got {}", first_val))
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
