//! Debug EVK s² computation - CUDA version
//!
//! This test directly compares s² from:
//! 1. compute_secret_key_squared_gpu (used in EVK generation)
//! 2. test_multiply_polys_ntt (used in verification tests)
//!
//! If these differ, the bug is in compute_secret_key_squared_gpu.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example debug_cuda_evk_s2
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
fn strided_to_flat(strided: &[u64], n: usize, num_primes: usize) -> Vec<u64> {
    let mut flat = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            let strided_idx = coeff_idx * num_primes + prime_idx;
            let flat_idx = prime_idx * n + coeff_idx;
            flat[flat_idx] = strided[strided_idx];
        }
    }
    flat
}

/// Convert flat layout to strided layout
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn flat_to_strided_conversion(flat: &[u64], n: usize, num_primes: usize) -> Vec<u64> {
    let mut strided = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        for coeff_idx in 0..n {
            let flat_idx = prime_idx * n + coeff_idx;
            let strided_idx = coeff_idx * num_primes + prime_idx;
            strided[strided_idx] = flat[flat_idx];
        }
    }
    strided
}

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     DEBUG: Compare s² from EVK generation vs test function             ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    // Enable PSI debugging
    std::env::set_var("PSI_DEBUG", "1");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();

    println!("Parameters: N={}, num_primes={}\n", n, num_primes);

    // Generate keys
    let key_ctx = KeyContext::new(params.clone());
    let (_pk, sk, _evk) = key_ctx.keygen();
    let sk_strided = secret_key_to_strided(&sk, num_primes);

    // Create CUDA context (this will print psi values)
    println!("Creating CudaCkksContext (prints psi values):");
    let device = Arc::new(CudaDeviceContext::new()?);
    let ctx = CudaCkksContext::new(params.clone())?;

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("Step 1: Compute s² using test_multiply_polys_ntt");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let s2_ntt_strided = ctx.test_multiply_polys_ntt(&sk_strided, &sk_strided, num_primes)?;
    let s2_ntt_flat = strided_to_flat(&s2_ntt_strided, n, num_primes);

    println!("  s² from test_multiply_polys_ntt (flat layout):");
    println!("    [0] = {} (prime 0, coeff 0)", s2_ntt_flat[0]);
    println!("    [1] = {} (prime 0, coeff 1)", s2_ntt_flat[1]);
    println!("    [n] = {} (prime 1, coeff 0)", s2_ntt_flat[n]);

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("Step 2: Generate EVK (prints psi values from EVK generation)");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Generate EVK - this will call compute_secret_key_squared_gpu internally
    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided.clone(),
        16,  // base_bits
        ctx.ntt_contexts(),
    )?;

    // Now we need to compare the s² embedded in EVK with s2_ntt_flat
    // We can infer s² from EVK[0]: b_0 = a_0·s + e - s², so s² ≈ a_0·s - b_0
    let relin_key = relin_keys.get_relin_key();
    let (evk0, evk1) = &relin_key.ks_components[0];

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("Step 3: Infer s² from EVK and compare");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Compute evk1 * s using the SAME multiply function that test used
    let evk1_strided = flat_to_strided_conversion(evk1, n, num_primes);
    let evk1_s_strided = ctx.test_multiply_polys_ntt(&evk1_strided, &sk_strided, num_primes)?;
    let evk1_s_flat = strided_to_flat(&evk1_s_strided, n, num_primes);

    let evk0_flat = evk0; // Already in flat layout

    // For each coefficient, compute inferred s² = evk1·s - evk0
    let mut s2_from_evk = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        let q = params.moduli[prime_idx];
        for coeff_idx in 0..n {
            let flat_idx = prime_idx * n + coeff_idx;
            let a = evk1_s_flat[flat_idx];
            let b = evk0_flat[flat_idx];
            s2_from_evk[flat_idx] = if a >= b { a - b } else { q - (b - a) };
        }
    }

    // Compare s2_ntt_flat with s2_from_evk
    let mut match_count = 0;
    let mut small_diff = 0;  // Noise
    let mut large_diff = 0;  // Bug
    let mut large_diffs = Vec::new();

    for prime_idx in 0..num_primes {
        let q = params.moduli[prime_idx];
        for coeff_idx in 0..n {
            let flat_idx = prime_idx * n + coeff_idx;
            let expected = s2_ntt_flat[flat_idx];
            let actual = s2_from_evk[flat_idx];

            let diff = if expected >= actual { expected - actual } else { actual - expected };

            if diff == 0 {
                match_count += 1;
            } else if diff < 10000 {
                small_diff += 1;
            } else if diff > q - 10000 {
                // Near-q diff (sign error)
                large_diff += 1;
                if large_diffs.len() < 10 {
                    large_diffs.push((prime_idx, coeff_idx, expected, actual, diff));
                }
            } else {
                // Large diff
                large_diff += 1;
                if large_diffs.len() < 10 {
                    large_diffs.push((prime_idx, coeff_idx, expected, actual, diff));
                }
            }
        }
    }

    println!("Comparison results:");
    println!("  Exact matches: {}", match_count);
    println!("  Small diffs (noise <10000): {}", small_diff);
    println!("  Large diffs (potential bug): {}", large_diff);

    if !large_diffs.is_empty() {
        println!("\n  First {} large differences:", large_diffs.len());
        for (prime_idx, coeff_idx, expected, actual, diff) in &large_diffs {
            let q = params.moduli[*prime_idx];
            let diff_type = if *diff > q / 2 { "~q" } else { "large" };
            println!("    [prime={}, coeff={}]: expected={}, actual={}, diff={} ({})",
                prime_idx, coeff_idx, expected, actual, diff, diff_type);
        }
    }

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("DIAGNOSIS");
    println!("════════════════════════════════════════════════════════════════════════\n");

    if large_diff == 0 {
        println!("  ✓ EVK s² matches test_multiply_polys_ntt s² perfectly!");
        println!("  The bug must be elsewhere (relinearization application?).");
    } else {
        println!("  ✗ EVK has {} coefficients with wrong s² values!", large_diff);
        println!("  Bug is in compute_secret_key_squared_gpu or how s² is embedded in EVK.");
    }

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires the 'v2' and 'v2-gpu-cuda' features.");
    println!("Run with: cargo run --release --features v2,v2-gpu-cuda --example debug_cuda_evk_s2");
}
