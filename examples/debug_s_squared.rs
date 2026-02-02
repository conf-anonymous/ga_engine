//! Debug s² computation in EVK generation
//!
//! This test compares:
//! 1. CPU O(N²) negacyclic s²
//! 2. gpu_multiply_flat_ntt s² (via compute_secret_key_squared_gpu)
//! 3. test_multiply_polys_ntt s² (via CudaCkksContext)
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example debug_s_squared
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
    println!("║     DEBUG: Compare s² from different sources                           ║");
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
    println!("Step 1: Compute s² using CPU O(N²) reference");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let mut s2_cpu_flat = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        let q = params.moduli[prime_idx];
        let offset = prime_idx * n;
        let s_prime: Vec<u64> = sk_flat[offset..offset+n].to_vec();
        let s2_prime = negacyclic_multiply_cpu(&s_prime, &s_prime, q, n);
        s2_cpu_flat[offset..offset+n].copy_from_slice(&s2_prime);
    }
    println!("  CPU s²[flat_idx=0] = {}", s2_cpu_flat[0]);
    println!("  CPU s²[flat_idx=1] = {}", s2_cpu_flat[1]);
    println!("  CPU s²[flat_idx=n] = {} (prime 1, coeff 0)", s2_cpu_flat[n]);

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("Step 2: Compute s² using test_multiply_polys_ntt (strided in/out)");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let s2_ntt_strided = ctx.test_multiply_polys_ntt(&sk_strided, &sk_strided, num_primes)?;
    let s2_ntt_flat = strided_to_flat(&s2_ntt_strided, n, num_primes);
    println!("  NTT s²[flat_idx=0] = {}", s2_ntt_flat[0]);
    println!("  NTT s²[flat_idx=1] = {}", s2_ntt_flat[1]);
    println!("  NTT s²[flat_idx=n] = {} (prime 1, coeff 0)", s2_ntt_flat[n]);

    // Compare CPU vs NTT
    let mut cpu_ntt_diff = 0;
    for i in 0..(n * num_primes) {
        if s2_cpu_flat[i] != s2_ntt_flat[i] {
            cpu_ntt_diff += 1;
            if cpu_ntt_diff <= 5 {
                println!("  DIFF at flat_idx={}: CPU={}, NTT={}", i, s2_cpu_flat[i], s2_ntt_flat[i]);
            }
        }
    }
    println!("\n  CPU vs NTT: {} differences out of {}", cpu_ntt_diff, n * num_primes);
    if cpu_ntt_diff == 0 {
        println!("  ✓ CPU and NTT s² match exactly");
    }

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("Step 3: Generate EVK and extract the s² embedded in it");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Generate EVK with debug output
    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided.clone(),
        16,  // base_bits
        ctx.ntt_contexts(),
    )?;

    let relin_key = relin_keys.get_relin_key();
    let num_digits = relin_key.ks_components.len();
    println!("  EVK has {} digits", num_digits);

    // For digit 0 (B^0 = 1), we can extract s² directly:
    // b_0 = a_0·s + e - s²
    // => s² = a_0·s + e - b_0 ≈ a_0·s - b_0 (ignoring noise)
    let (evk0, evk1) = &relin_key.ks_components[0];

    println!("\n  Extracting s² from EVK[0] for prime 0:");
    println!("    evk0[0..3] = [{}, {}, {}]", evk0[0], evk0[1], evk0[2]);
    println!("    evk1[0..3] = [{}, {}, {}]", evk1[0], evk1[1], evk1[2]);

    // Compute evk1 * s using CPU
    let evk1_prime0: Vec<u64> = evk1[0..n].to_vec();
    let s_prime0: Vec<u64> = sk_flat[0..n].to_vec();
    let q0 = params.moduli[0];
    let evk1_s = negacyclic_multiply_cpu(&evk1_prime0, &s_prime0, q0, n);

    // s² ≈ evk1·s - evk0
    let mut s2_from_evk = vec![0u64; n];
    for i in 0..n {
        s2_from_evk[i] = if evk1_s[i] >= evk0[i] {
            evk1_s[i] - evk0[i]
        } else {
            q0 - (evk0[i] - evk1_s[i])
        };
    }

    println!("    s² inferred: [{}, {}, {}]", s2_from_evk[0], s2_from_evk[1], s2_from_evk[2]);
    println!("    CPU s²:      [{}, {}, {}]", s2_cpu_flat[0], s2_cpu_flat[1], s2_cpu_flat[2]);

    // Compare
    let mut evk_diff_count = 0;
    let mut evk_diff_positions = Vec::new();
    for i in 0..n {
        let expected = s2_cpu_flat[i];
        let actual = s2_from_evk[i];
        let diff = if expected >= actual { expected - actual } else { actual - expected };
        // Allow for noise (up to ~1000 or so)
        if diff > 10000 && diff < q0 - 10000 {
            // Medium sized diff - suspicious
            evk_diff_count += 1;
            if evk_diff_positions.len() < 10 {
                evk_diff_positions.push((i, expected, actual, diff));
            }
        } else if diff >= q0 - 10000 {
            // Near-q diff - this is the ~q error we're hunting
            evk_diff_count += 1;
            if evk_diff_positions.len() < 10 {
                evk_diff_positions.push((i, expected, actual, diff));
            }
        }
    }

    println!("\n  Comparison results:");
    println!("    Significant differences: {}", evk_diff_count);
    if !evk_diff_positions.is_empty() {
        println!("    First few differences:");
        for (idx, expected, actual, diff) in &evk_diff_positions {
            let diff_type = if *diff > q0 / 2 { "~q (sign error)" } else { "medium" };
            println!("      [{}]: expected={}, actual={}, diff={} ({})",
                idx, expected, actual, diff, diff_type);
        }
    }

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("Step 4: Check psi values between ckks context and relin_keys");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let psi_from_ctx = ctx.psi_per_prime();
    println!("  psi from CudaCkksContext (prime 0): {}", psi_from_ctx[0]);

    // The relin_keys compute psi internally using find_primitive_root
    // We can't directly access it, but we can compute it the same way
    fn pow_mod(mut a: u64, mut e: u64, p: u64) -> u64 {
        let mut result = 1u64;
        a = a % p;
        while e > 0 {
            if e & 1 == 1 {
                result = ((result as u128 * a as u128) % p as u128) as u64;
            }
            e >>= 1;
            a = ((a as u128 * a as u128) % p as u128) as u64;
        }
        result
    }

    fn find_primitive_root_test(n: usize, q: u64) -> Option<u64> {
        let two_n = (2 * n) as u64;
        let candidates: [u64; 11] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
        for &candidate in &candidates {
            if pow_mod(candidate, (q - 1) / 2, q) == 1 { continue; }
            let exp = (q - 1) / two_n;
            let psi = pow_mod(candidate, exp, q);
            let psi_n = pow_mod(psi, n as u64, q);
            if psi_n != q - 1 { continue; }
            let psi_2n = pow_mod(psi, two_n, q);
            if psi_2n == 1 { return Some(psi); }
        }
        None
    }

    let psi_computed = find_primitive_root_test(n, q0).unwrap();
    println!("  psi from find_primitive_root (prime 0): {}", psi_computed);

    if psi_from_ctx[0] == psi_computed {
        println!("  ✓ psi values MATCH");
    } else {
        println!("  ✗ psi values DIFFER - THIS IS THE BUG!");
    }

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("DIAGNOSIS");
    println!("════════════════════════════════════════════════════════════════════════\n");

    if cpu_ntt_diff == 0 && evk_diff_count == 0 {
        println!("  ✓ All s² computations match. EVK generation is correct.");
    } else if cpu_ntt_diff == 0 && evk_diff_count > 0 {
        println!("  ✗ CPU/NTT s² match, but EVK has wrong s² embedded!");
        println!("    The bug is in EVK generation (how s² is computed or stored).");
    } else {
        println!("  ✗ Multiple mismatches detected.");
    }

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires the 'v2' and 'v2-gpu-cuda' features.");
    println!("Run with: cargo run --release --features v2,v2-gpu-cuda --example debug_s_squared");
}
