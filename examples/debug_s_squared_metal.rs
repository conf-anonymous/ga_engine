//! Debug s² computation in EVK generation (Metal version)
//!
//! This test compares:
//! 1. CPU O(N²) negacyclic s²
//! 2. test_multiply_polys_ntt s² (via MetalCkksContext)
//! 3. s² inferred from EVK property
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-metal --example debug_s_squared_metal
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-metal"))]
use ga_engine::clifford_fhe_v2::{
    backends::{
        cpu_optimized::keys::KeyContext,
        gpu_metal::{
            ckks::MetalCkksContext,
            relin_keys::MetalRelinKeys,
        },
    },
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v2", feature = "v2-gpu-metal"))]
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

#[cfg(all(feature = "v2", feature = "v2-gpu-metal"))]
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
#[cfg(all(feature = "v2", feature = "v2-gpu-metal"))]
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

#[cfg(all(feature = "v2", feature = "v2-gpu-metal"))]
fn main() -> Result<(), String> {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     DEBUG: Compare s² from different sources (Metal)                   ║");
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

    // Create Metal context
    let ctx = MetalCkksContext::new(params.clone())?;

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

    // Generate EVK
    let relin_keys = MetalRelinKeys::new(
        params.clone(),
        sk_strided.clone(),
        16,  // base_bits
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

    // Compare with noise tolerance
    let mut evk_small_diff = 0;  // Small diff (noise)
    let mut evk_medium_diff = 0; // Medium diff (suspicious)
    let mut evk_large_diff = 0;  // Large diff (~q, sign error)
    let mut evk_diff_positions = Vec::new();

    for i in 0..n {
        let expected = s2_cpu_flat[i];
        let actual = s2_from_evk[i];
        let diff = if expected >= actual { expected - actual } else { actual - expected };

        if diff == 0 {
            // Exact match
        } else if diff < 10000 {
            evk_small_diff += 1;  // Noise
        } else if diff < q0 / 2 {
            evk_medium_diff += 1;
            if evk_diff_positions.len() < 5 {
                evk_diff_positions.push((i, expected, actual, diff, "medium"));
            }
        } else {
            evk_large_diff += 1;
            if evk_diff_positions.len() < 10 {
                evk_diff_positions.push((i, expected, actual, diff, "~q (sign error)"));
            }
        }
    }

    println!("\n  Comparison results (prime 0 only, {} coefficients):", n);
    println!("    Small diffs (noise <10000): {}", evk_small_diff);
    println!("    Medium diffs (>10000, <q/2): {}", evk_medium_diff);
    println!("    Large diffs (~q, sign error): {}", evk_large_diff);

    if !evk_diff_positions.is_empty() {
        println!("    First few significant differences:");
        for (idx, expected, actual, diff, diff_type) in &evk_diff_positions {
            println!("      [{}]: expected={}, actual={}, diff={} ({})",
                idx, expected, actual, diff, diff_type);
        }
    }

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("Step 4: Check psi values");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let psi_from_ctx = ctx.psi_per_prime();
    println!("  psi from MetalCkksContext (prime 0): {}", psi_from_ctx[0]);
    println!("  psi from MetalCkksContext (prime 1): {}", psi_from_ctx[1]);
    println!("  psi from MetalCkksContext (prime 2): {}", psi_from_ctx[2]);

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("DIAGNOSIS");
    println!("════════════════════════════════════════════════════════════════════════\n");

    if cpu_ntt_diff == 0 && evk_large_diff == 0 && evk_medium_diff == 0 {
        println!("  ✓ All s² computations match. EVK generation is correct.");
        println!("  (Small noise differences are expected and acceptable)");
    } else if cpu_ntt_diff == 0 && (evk_large_diff > 0 || evk_medium_diff > 0) {
        println!("  ✗ CPU/NTT s² match, but EVK has wrong s² embedded!");
        println!("    {} large (~q) errors, {} medium errors", evk_large_diff, evk_medium_diff);
        println!("    The bug is in EVK generation (how s² is computed or stored).");
    } else {
        println!("  ✗ Multiple mismatches detected.");
        println!("    CPU vs NTT: {} diffs", cpu_ntt_diff);
        println!("    EVK large: {}, medium: {}", evk_large_diff, evk_medium_diff);
    }

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-metal")))]
fn main() {
    println!("This example requires the 'v2' and 'v2-gpu-metal' features.");
    println!("Run with: cargo run --release --features v2,v2-gpu-metal --example debug_s_squared_metal");
}
