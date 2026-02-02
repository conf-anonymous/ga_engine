//! Test EVK multiply function (gpu_multiply_flat_ntt) vs CPU reference
//!
//! This isolates the multiplication used during EVK generation to see if
//! it matches the CPU reference and test_multiply_polys_ntt.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example test_evk_multiply
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

/// CPU negacyclic multiplication (coefficient domain) - O(N^2) reference
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
    println!("║     TEST EVK MULTIPLY: Compare EVK's internal s² computation           ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();

    println!("Parameters: N={}, num_primes={}\n", n, num_primes);

    // Generate keys
    let key_ctx = KeyContext::new(params.clone());
    let (_pk, sk, _evk) = key_ctx.keygen();
    let sk_strided = secret_key_to_strided(&sk, num_primes);
    let sk_flat = strided_to_flat(&sk_strided, n, num_primes);

    // Create CUDA context
    let device = Arc::new(CudaDeviceContext::new()?);
    let ctx = CudaCkksContext::new(params.clone())?;

    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST: Compare 3 ways of computing s²");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Method 1: CPU O(N²) negacyclic multiply
    println!("  Method 1: CPU O(N²) negacyclic multiply...");
    let mut s2_cpu_flat = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        let q = params.moduli[prime_idx];
        let offset = prime_idx * n;
        let s_prime: Vec<u64> = sk_flat[offset..offset+n].to_vec();
        let s2_prime = negacyclic_multiply_cpu(&s_prime, &s_prime, q, n);
        s2_cpu_flat[offset..offset+n].copy_from_slice(&s2_prime);
    }
    println!("    s²[coeff=0, prime=0]: {}", s2_cpu_flat[0]);

    // Method 2: test_multiply_polys_ntt (strided in/out)
    println!("  Method 2: test_multiply_polys_ntt (strided layout)...");
    let s2_ntt_strided = ctx.test_multiply_polys_ntt(&sk_strided, &sk_strided, num_primes)?;
    let s2_ntt_flat = strided_to_flat(&s2_ntt_strided, n, num_primes);
    println!("    s²[coeff=0, prime=0]: {}", s2_ntt_flat[0]);

    // Method 3: EVK generation's compute_secret_key_squared_gpu
    // This is called internally when we create relin keys. Let's extract it.
    // Since we can't call it directly, we'll generate keys and extract the s² from the EVK property.
    println!("  Method 3: EVK generation's internal s² (via compute_secret_key_squared_gpu)...");
    // We'll verify this by checking the EVK property:
    // evk0[0] - evk1[0]·s = e - s²  (approximately)
    // So s² ≈ evk1[0]·s - evk0[0] + e
    // Since we're testing at digit 0 where B^0=1

    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided.clone(),
        16,  // base_bits
        ctx.ntt_contexts(),
    )?;

    let relin_key = relin_keys.get_relin_key();
    let (evk0, evk1) = &relin_key.ks_components[0];

    // Compute evk1[0]·s using CPU reference
    let evk1_prime0: Vec<u64> = evk1[0..n].to_vec();
    let s_prime0: Vec<u64> = sk_flat[0..n].to_vec();
    let q0 = params.moduli[0];
    let evk1_times_s = negacyclic_multiply_cpu(&evk1_prime0, &s_prime0, q0, n);

    // evk0 - evk1·s ≈ -s² + noise
    // So evk1·s - evk0 ≈ s² - noise
    let mut s2_from_evk = vec![0u64; n];
    for i in 0..n {
        s2_from_evk[i] = if evk1_times_s[i] >= evk0[i] {
            evk1_times_s[i] - evk0[i]
        } else {
            q0 - (evk0[i] - evk1_times_s[i])
        };
    }
    println!("    s² inferred from EVK[coeff=0]: {}", s2_from_evk[0]);
    println!("    Note: This includes noise 'e', so won't be exact");

    // Compare the three methods
    println!("\n  ──────────────────────────────────────────────────────────────────────");
    println!("  COMPARISON (prime 0 only):");

    // CPU vs NTT
    let mut cpu_ntt_match = 0;
    let mut cpu_ntt_diff = 0;
    for i in 0..n {
        if s2_cpu_flat[i] == s2_ntt_flat[i] {
            cpu_ntt_match += 1;
        } else {
            cpu_ntt_diff += 1;
        }
    }
    println!("    CPU vs NTT: {} match, {} diff", cpu_ntt_match, cpu_ntt_diff);

    // CPU vs EVK inferred (allowing for noise)
    let mut cpu_evk_close = 0;
    let mut cpu_evk_far = 0;
    let mut max_evk_err = 0u64;
    for i in 0..n {
        let diff = if s2_cpu_flat[i] >= s2_from_evk[i] {
            s2_cpu_flat[i] - s2_from_evk[i]
        } else {
            s2_from_evk[i] - s2_cpu_flat[i]
        };
        if diff < 100 {  // Small noise tolerance
            cpu_evk_close += 1;
        } else {
            cpu_evk_far += 1;
            if diff > max_evk_err {
                max_evk_err = diff;
            }
            if cpu_evk_far <= 5 {
                println!("    EVK diff at [{}]: CPU={}, EVK-inferred={}, diff={}",
                    i, s2_cpu_flat[i], s2_from_evk[i], diff);
            }
        }
    }
    println!("    CPU vs EVK-inferred: {} close (<100), {} far, max_err={}", cpu_evk_close, cpu_evk_far, max_evk_err);

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("DIAGNOSIS");
    println!("════════════════════════════════════════════════════════════════════════\n");

    if cpu_ntt_diff == 0 {
        println!("  ✓ CPU and NTT s² match exactly");
    } else {
        println!("  ✗ CPU and NTT s² differ - multiplication bug!");
    }

    if max_evk_err > q0 / 2 {
        println!("  ✗ EVK-inferred s² has LARGE errors (~q) - this is the bug!");
        println!("    The s² used inside EVK generation differs from the s² we compute.");
        println!("    BUG IS IN: compute_secret_key_squared_gpu or how it's used");
    } else if cpu_evk_far > 100 {
        println!("  ⚠ EVK-inferred s² has many small errors - noise is high");
    } else {
        println!("  ✓ EVK s² matches CPU reference (within noise tolerance)");
    }

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires the 'v2' and 'v2-gpu-cuda' features.");
    println!("Run with: cargo run --release --features v2,v2-gpu-cuda --example test_evk_multiply");
}
