//! Verify EVK fundamental property: evk0[t] - evk1[t]·s ≈ -B^t·s²
//!
//! This test verifies that the evaluation key correctly encrypts -B^t·s².
//! If this property doesn't hold, relinearization cannot work.
//!
//! CRITICAL INSIGHT: This test verifies:
//! 1. First, that CPU and GPU s² computation match
//! 2. Then, that the EVK property holds using the correct s²
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example verify_evk_property
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

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn flat_to_strided(flat: &[u64], n: usize, num_primes: usize) -> Vec<u64> {
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

/// Compute polynomial product a*b mod (X^N + 1) in coefficient domain (CPU reference)
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn negacyclic_multiply_cpu(a: &[u64], b: &[u64], q: u64, n: usize) -> Vec<u64> {
    let mut result = vec![0u64; n];
    for i in 0..n {
        for j in 0..n {
            let k = i + j;
            let prod = ((a[i] as u128 * b[j] as u128) % q as u128) as u64;
            if k < n {
                result[k] = (result[k] + prod) % q;
            } else {
                // Negacyclic: X^N = -1
                let wrap_idx = k - n;
                result[wrap_idx] = if result[wrap_idx] >= prod {
                    result[wrap_idx] - prod
                } else {
                    q - (prod - result[wrap_idx])
                };
            }
        }
    }
    result
}

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     VERIFY EVK PROPERTY: evk0[t] - evk1[t]·s ≈ -B^t·s²                 ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();
    let base_bits = 16u32;
    let base_w = 1u64 << base_bits;

    println!("Parameters: N={}, num_primes={}, base_w=2^{}={}\n", n, num_primes, base_bits, base_w);

    // Generate keys
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let sk_strided = secret_key_to_strided(&sk, num_primes);
    let sk_flat = strided_to_flat(&sk_strided, n, num_primes);

    // Create CUDA context
    let device = Arc::new(CudaDeviceContext::new()?);
    let ctx = CudaCkksContext::new(params.clone())?;

    // STEP 0: Verify CPU s² vs GPU s² to isolate multiplication bugs
    println!("════════════════════════════════════════════════════════════════════════");
    println!("STEP 0: Compare CPU s² vs GPU s² (using test_multiply_polys_ntt)");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Compute s² on CPU for reference (flat layout)
    println!("  Computing s² on CPU (coefficient domain O(N²))...");
    let mut s_squared_cpu_flat = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        let q = params.moduli[prime_idx];
        let offset = prime_idx * n;
        let s_prime: Vec<u64> = sk_flat[offset..offset+n].to_vec();
        let s2_prime = negacyclic_multiply_cpu(&s_prime, &s_prime, q, n);
        s_squared_cpu_flat[offset..offset+n].copy_from_slice(&s2_prime);
    }
    println!("  CPU s²[coeff=0, prime=0]: {}", s_squared_cpu_flat[0]);

    // Compute s² on GPU via test_multiply_polys_ntt (strided layout input/output)
    println!("  Computing s² on GPU (NTT-based)...");
    let s_squared_gpu_strided = ctx.test_multiply_polys_ntt(&sk_strided, &sk_strided, num_primes)?;
    let s_squared_gpu_flat = strided_to_flat(&s_squared_gpu_strided, n, num_primes);
    println!("  GPU s²[coeff=0, prime=0]: {}", s_squared_gpu_flat[0]);

    // Compare CPU vs GPU s² for prime 0
    let mut s2_match = 0;
    let mut s2_diff = 0;
    let mut s2_max_error = 0u64;
    for i in 0..n {
        let cpu_val = s_squared_cpu_flat[i];
        let gpu_val = s_squared_gpu_flat[i];
        if cpu_val == gpu_val {
            s2_match += 1;
        } else {
            s2_diff += 1;
            let err = if cpu_val > gpu_val { cpu_val - gpu_val } else { gpu_val - cpu_val };
            if err > s2_max_error {
                s2_max_error = err;
            }
            if s2_diff <= 5 {
                println!("  S² DIFF at coeff[{}]: CPU={}, GPU={}", i, cpu_val, gpu_val);
            }
        }
    }

    println!("\n  CPU vs GPU s²: {} matches, {} differences, max_error={}", s2_match, s2_diff, s2_max_error);

    if s2_diff == 0 {
        println!("  ✓ CPU and GPU s² MATCH - multiplication is correct\n");
    } else {
        println!("  ✗ CPU and GPU s² DIFFER - BUG IN MULTIPLICATION!");
        println!("    The bug is in either CPU reference or GPU NTT multiply.\n");
    }

    // Generate CUDA relin keys
    println!("════════════════════════════════════════════════════════════════════════");
    println!("STEP 1: Generate CUDA relinearization keys");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided.clone(),
        base_bits as usize,
        ctx.ntt_contexts(),
    )?;

    let relin_key = relin_keys.get_relin_key();
    println!("EVK has {} components\n", relin_key.ks_components.len());

    // Note: EVK generation uses gpu_multiply_flat_ntt internally.
    // The s² used in EVK is compute_secret_key_squared_gpu which also uses gpu_multiply_flat_ntt.
    // We need to verify using the SAME s² that EVK generation used.

    // Let's also compute the "EVK internal s²" by extracting it from the pattern.
    // EVK formula: evk0[0] = a[0]·s + e - B^0·s² = a[0]·s + e - s²
    // So: evk0[0] - a[0]·s - e ≈ -s² (modulo noise)

    // Verify property for digit 0
    println!("════════════════════════════════════════════════════════════════════════");
    println!("VERIFY: evk0[0] - evk1[0]·s = -B^0·s² = -s² (digit 0, B^0 = 1)");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let (evk0_0, evk1_0) = &relin_key.ks_components[0];

    println!("  evk0[0] len: {} (expected {})", evk0_0.len(), n * num_primes);
    println!("  evk1[0] len: {} (expected {})", evk1_0.len(), n * num_primes);
    println!("  evk0[0][coeff=0, prime=0]: {}", evk0_0[0]);
    println!("  evk1[0][coeff=0, prime=0]: {}", evk1_0[0]);

    // Compute evk1[0] · s for prime 0 using CPU reference
    println!("\n  Computing evk1[0] · s using CPU reference...");
    let q0 = params.moduli[0];
    let evk1_0_prime0: Vec<u64> = evk1_0[0..n].to_vec();
    let s_prime0: Vec<u64> = sk_flat[0..n].to_vec();
    let evk1_s_cpu = negacyclic_multiply_cpu(&evk1_0_prime0, &s_prime0, q0, n);
    println!("  (evk1[0] · s)[coeff=0, prime=0] via CPU: {}", evk1_s_cpu[0]);

    // Also compute evk1[0] · s using GPU NTT for comparison
    // Need to convert to strided for test_multiply_polys_ntt
    let evk1_0_strided = flat_to_strided(evk1_0, n, num_primes);
    let evk1_s_gpu_strided = ctx.test_multiply_polys_ntt(&evk1_0_strided, &sk_strided, num_primes)?;
    let evk1_s_gpu_flat = strided_to_flat(&evk1_s_gpu_strided, n, num_primes);
    println!("  (evk1[0] · s)[coeff=0, prime=0] via GPU: {}", evk1_s_gpu_flat[0]);

    // Check if CPU and GPU multiplication of evk1*s match
    let mut evk1s_match = 0;
    let mut evk1s_diff = 0;
    for i in 0..n {
        if evk1_s_cpu[i] == evk1_s_gpu_flat[i] {
            evk1s_match += 1;
        } else {
            evk1s_diff += 1;
        }
    }
    println!("  CPU vs GPU (evk1·s): {} matches, {} differences", evk1s_match, evk1s_diff);

    // Compute evk0[0] - evk1[0]·s using CPU multiply result
    println!("\n  Computing evk0[0] - evk1[0]·s (using CPU multiply)...");
    let mut diff_cpu = vec![0u64; n];
    for i in 0..n {
        diff_cpu[i] = if evk0_0[i] >= evk1_s_cpu[i] {
            evk0_0[i] - evk1_s_cpu[i]
        } else {
            q0 - (evk1_s_cpu[i] - evk0_0[i])
        };
    }
    println!("  (evk0[0] - evk1[0]·s)[coeff=0] via CPU: {}", diff_cpu[0]);

    // Also compute using GPU multiply
    let mut diff_gpu = vec![0u64; n];
    for i in 0..n {
        diff_gpu[i] = if evk0_0[i] >= evk1_s_gpu_flat[i] {
            evk0_0[i] - evk1_s_gpu_flat[i]
        } else {
            q0 - (evk1_s_gpu_flat[i] - evk0_0[i])
        };
    }
    println!("  (evk0[0] - evk1[0]·s)[coeff=0] via GPU: {}", diff_gpu[0]);

    // Expected: -s² mod q0 (using CPU s²)
    let neg_s2_cpu: Vec<u64> = s_squared_cpu_flat[0..n].iter()
        .map(|&x| if x == 0 { 0 } else { q0 - x })
        .collect();
    println!("  Expected -s²[coeff=0] (CPU): {}", neg_s2_cpu[0]);

    // Expected: -s² mod q0 (using GPU s²)
    let neg_s2_gpu: Vec<u64> = s_squared_gpu_flat[0..n].iter()
        .map(|&x| if x == 0 { 0 } else { q0 - x })
        .collect();
    println!("  Expected -s²[coeff=0] (GPU): {}", neg_s2_gpu[0]);

    // Compare diff_cpu vs neg_s2_cpu (CPU reference all the way)
    println!("\n  COMPARISON: (evk0 - evk1·s) vs -s²");
    println!("  ──────────────────────────────────────────────────────────────────────");

    // Using CPU multiply for evk1·s, CPU s² for reference
    let mut match_cpu_cpu = 0;
    let mut diff_cpu_cpu = 0;
    let mut max_err_cpu_cpu = 0u64;
    for i in 0..n {
        let err = if diff_cpu[i] >= neg_s2_cpu[i] {
            diff_cpu[i] - neg_s2_cpu[i]
        } else {
            neg_s2_cpu[i] - diff_cpu[i]
        };
        if err < 1_000_000 {  // Allow small noise
            match_cpu_cpu += 1;
        } else {
            diff_cpu_cpu += 1;
            if err > max_err_cpu_cpu { max_err_cpu_cpu = err; }
            if diff_cpu_cpu <= 3 {
                println!("    [CPU/CPU] DIFF at [{}]: actual={}, expected={}, err={}",
                    i, diff_cpu[i], neg_s2_cpu[i], err);
            }
        }
    }
    println!("  CPU evk1·s, CPU s²: {} match, {} diff, max_err={}", match_cpu_cpu, diff_cpu_cpu, max_err_cpu_cpu);

    // Using CPU multiply for evk1·s, GPU s² for reference
    let mut match_cpu_gpu = 0;
    let mut diff_cpu_gpu = 0;
    let mut max_err_cpu_gpu = 0u64;
    for i in 0..n {
        let err = if diff_cpu[i] >= neg_s2_gpu[i] {
            diff_cpu[i] - neg_s2_gpu[i]
        } else {
            neg_s2_gpu[i] - diff_cpu[i]
        };
        if err < 1_000_000 {
            match_cpu_gpu += 1;
        } else {
            diff_cpu_gpu += 1;
            if err > max_err_cpu_gpu { max_err_cpu_gpu = err; }
        }
    }
    println!("  CPU evk1·s, GPU s²: {} match, {} diff, max_err={}", match_cpu_gpu, diff_cpu_gpu, max_err_cpu_gpu);

    // Using GPU multiply for evk1·s, GPU s² for reference (should match best if EVK uses GPU internally)
    let mut match_gpu_gpu = 0;
    let mut diff_gpu_gpu = 0;
    let mut max_err_gpu_gpu = 0u64;
    for i in 0..n {
        let err = if diff_gpu[i] >= neg_s2_gpu[i] {
            diff_gpu[i] - neg_s2_gpu[i]
        } else {
            neg_s2_gpu[i] - diff_gpu[i]
        };
        if err < 1_000_000 {
            match_gpu_gpu += 1;
        } else {
            diff_gpu_gpu += 1;
            if err > max_err_gpu_gpu { max_err_gpu_gpu = err; }
            if diff_gpu_gpu <= 3 {
                println!("    [GPU/GPU] DIFF at [{}]: actual={}, expected={}, err={}",
                    i, diff_gpu[i], neg_s2_gpu[i], err);
            }
        }
    }
    println!("  GPU evk1·s, GPU s²: {} match, {} diff, max_err={}", match_gpu_gpu, diff_gpu_gpu, max_err_gpu_gpu);

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("DIAGNOSIS");
    println!("════════════════════════════════════════════════════════════════════════\n");

    if s2_diff == 0 {
        println!("  ✓ CPU and GPU s² match - NTT multiplication is correct");
    } else {
        println!("  ✗ CPU and GPU s² differ - NTT multiplication has a bug");
    }

    if diff_cpu_cpu == 0 {
        println!("  ✓ EVK property verified with CPU-only reference");
    } else if diff_gpu_gpu == 0 {
        println!("  ⚠ EVK property verified only with GPU reference (CPU has different s²)");
    } else {
        println!("  ✗ EVK property FAILED - bug in EVK generation or multiplication");

        // Additional diagnostics
        if max_err_cpu_cpu > params.moduli[0] / 2 {
            println!("    Large errors (~q) suggest sign/wrap bugs in multiplication");
        }
        if evk1s_diff > 0 {
            println!("    CPU vs GPU (evk1·s) differ - bug in one of the multiplies");
        }
    }

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("DONE");
    println!("════════════════════════════════════════════════════════════════════════\n");

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires the 'v2' and 'v2-gpu-cuda' features.");
    println!("Run with: cargo run --release --features v2,v2-gpu-cuda --example verify_evk_property");
}
