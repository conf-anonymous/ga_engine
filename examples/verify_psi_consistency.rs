//! Verify PSI consistency between CudaCkksContext and CudaRelinKeys
//!
//! This test checks that the psi values used in EVK generation match
//! the psi values stored in the CKKS context, ensuring omega = psi².
//!
//! Run with:
//! ```bash
//! PSI_DEBUG=1 cargo run --release --features v2,v2-gpu-cuda --example verify_psi_consistency
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::{
    backends::{
        cpu_optimized::keys::KeyContext,
        gpu_cuda::{
            ckks::CudaCkksContext,
            device::CudaDeviceContext,
            relin_keys::CudaRelinKeys,
        },
    },
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn secret_key_to_strided(
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    num_primes: usize,
) -> Vec<u64> {
    let n = sk.n;
    let mut strided = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            strided[coeff_idx * num_primes + prime_idx] = sk.coeffs[coeff_idx].values[prime_idx];
        }
    }
    strided
}

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     VERIFY: PSI Consistency Between CKKS Context and Relin Keys       ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    // Set PSI_DEBUG to enable debug output
    std::env::set_var("PSI_DEBUG", "1");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();

    println!("Parameters: N={}, num_primes={}\n", n, num_primes);

    // Generate keys
    let key_ctx = KeyContext::new(params.clone());
    let (_pk, sk, _evk) = key_ctx.keygen();
    let sk_strided = secret_key_to_strided(&sk, num_primes);

    println!("════════════════════════════════════════════════════════════════════════");
    println!("Step 1: Create CudaCkksContext (prints psi values)");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let device = Arc::new(CudaDeviceContext::new()?);
    let ctx = CudaCkksContext::new(params.clone())?;

    // Get psi values from context
    let psi_from_ctx = ctx.psi_per_prime();
    println!("\nCKKS Context psi values:");
    for (i, &psi) in psi_from_ctx.iter().enumerate().take(3) {
        let q = params.moduli[i];
        let omega = ctx.ntt_contexts()[i].root;
        let psi_squared = ((psi as u128 * psi as u128) % q as u128) as u64;
        println!("  Prime {}: psi={}, psi²={}, omega={}, match={}",
            i, psi, psi_squared, omega,
            if psi_squared == omega { "✓" } else { "✗ MISMATCH!" });
    }

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("Step 2: Generate EVK (prints psi values from relin_keys)");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Generate EVK - this will print PSI_DEBUG messages
    let _relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided,
        16,  // base_bits
        ctx.ntt_contexts(),
    )?;

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("DIAGNOSIS");
    println!("════════════════════════════════════════════════════════════════════════\n");

    println!("If all psi² = omega matches show ✓, the psi values are consistent.");
    println!("If any show WARNING about psi² != omega, that's the bug!");

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires the 'v2' and 'v2-gpu-cuda' features.");
    println!("Run with: PSI_DEBUG=1 cargo run --release --features v2,v2-gpu-cuda --example verify_psi_consistency");
}
