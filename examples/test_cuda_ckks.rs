//! Test CUDA CKKS basic operations
//!
//! This example tests the CUDA CKKS context initialization and basic rescaling operation.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example test_cuda_ckks
//! ```

use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

fn main() -> Result<(), String> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║              CUDA CKKS Basic Operations Test                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Create parameters
    println!("Step 1: Creating FHE parameters...");
    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("  ✅ Parameters: N={}, {} primes\n", params.n, params.moduli.len());

    // Create CUDA CKKS context
    println!("Step 2: Initializing CUDA CKKS context...");
    let ctx = CudaCkksContext::new(params.clone())?;
    println!("  ✅ CUDA CKKS context ready!\n");

    // Test encoding
    println!("Step 3: Testing encoding...");
    let values = vec![1.5, 2.5, 3.5, 4.5];
    let scale = 1e10;
    let level = 2;

    let pt = ctx.encode(&values, scale, level)?;
    println!("  ✅ Encoded {} values at level {}", values.len(), level);
    println!("     Plaintext size: {} coefficients", pt.poly.len() / pt.num_primes);
    println!("     Number of RNS primes: {}\n", pt.num_primes);

    // Test GPU rescaling
    println!("Step 4: Testing GPU rescaling...");
    let n = params.n;
    let test_level = 2;
    let num_primes_in = test_level + 1;

    // Create test polynomial
    let mut test_poly = vec![0u64; n * num_primes_in];
    for i in 0..n {
        for j in 0..num_primes_in {
            test_poly[i * num_primes_in + j] = (i * 1000 + j * 100) as u64 % params.moduli[j];
        }
    }

    println!("  Input:  {} coefficients × {} primes = {} elements",
             n, num_primes_in, test_poly.len());

    let result = ctx.exact_rescale_gpu(&test_poly, test_level)?;

    let num_primes_out = num_primes_in - 1;
    println!("  Output: {} coefficients × {} primes = {} elements",
             n, num_primes_out, result.len());

    assert_eq!(result.len(), n * num_primes_out);
    println!("  ✅ GPU rescaling successful!\n");

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("✅ ALL TESTS PASSED");
    println!("═══════════════════════════════════════════════════════════════");
    println!("\nCUDA CKKS Operations Summary:");
    println!("  • Context initialization: ✅");
    println!("  • Encoding: ✅");
    println!("  • GPU Rescaling: ✅");
    println!("\nNext Steps:");
    println!("  • Test on RunPod with NVIDIA GPU");
    println!("  • Implement rotation operations");
    println!("  • Implement full bootstrap");

    Ok(())
}
