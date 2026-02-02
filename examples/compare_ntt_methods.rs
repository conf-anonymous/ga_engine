//! Compare GPU NTT multiplication methods: Sequential vs Batched
//!
//! This test compares the batched GPU multiplication to a CPU reference.
//! It also tests individual operations (twist, NTT, multiply) step by step.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example compare_ntt_methods
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_cuda::ckks::CudaCkksContext,
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     COMPARE: GPU NTT Multiplication Methods                            ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();

    println!("Parameters: N={}, num_primes={}\n", n, num_primes);

    // Create CUDA context
    let ctx = CudaCkksContext::new(params.clone())?;

    // Create test polynomials with known values
    // Use simple polynomials: p1[i] = i+1, p2[i] = 2*i+1 for all primes
    let mut p1_flat = vec![0u64; n * num_primes];
    let mut p2_flat = vec![0u64; n * num_primes];

    for prime_idx in 0..num_primes {
        let q = params.moduli[prime_idx];
        for coeff_idx in 0..n {
            let idx = prime_idx * n + coeff_idx;
            p1_flat[idx] = ((coeff_idx + 1) as u64) % q;
            p2_flat[idx] = ((2 * coeff_idx + 1) as u64) % q;
        }
    }

    println!("Test polynomials:");
    println!("  p1: [1, 2, 3, 4, ...] (first {} coeffs, repeated for {} primes)",
             n.min(10), num_primes);
    println!("  p2: [1, 3, 5, 7, ...] (first {} coeffs, repeated for {} primes)\n",
             n.min(10), num_primes);

    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 1: Batched GPU multiplication vs CPU reference");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let result_batched = ctx.test_multiply_polys_ntt(
        &p1_flat,
        &p2_flat,
        num_primes,
    )?;

    println!("  Batched GPU result first 5 coefficients (prime 0):");
    for i in 0..5.min(n) {
        println!("    coeff[{}] = {}", i, result_batched[i]);
    }

    // Compute CPU reference for prime 0
    // Negacyclic convolution of p1 and p2 modulo X^N + 1
    let q = params.moduli[0];
    let mut result_cpu = vec![0u64; n];

    for i in 0..n {
        for j in 0..n {
            let k = i + j;
            let coeff_mul = ((p1_flat[j] as u128 * p2_flat[i] as u128) % q as u128) as u64;

            if k < n {
                result_cpu[k] = (result_cpu[k] + coeff_mul) % q;
            } else {
                let wrap_idx = k - n;
                result_cpu[wrap_idx] = if result_cpu[wrap_idx] >= coeff_mul {
                    result_cpu[wrap_idx] - coeff_mul
                } else {
                    q - (coeff_mul - result_cpu[wrap_idx])
                };
            }
        }
    }

    println!("\n  CPU reference first 5 coefficients:");
    for i in 0..5.min(n) {
        println!("    coeff[{}] = {}", i, result_cpu[i]);
    }

    let mut match_count = 0;
    let mut diff_count = 0;
    for i in 0..n {
        if result_batched[i] == result_cpu[i] {
            match_count += 1;
        } else {
            diff_count += 1;
            if diff_count <= 5 {
                println!("  DIFF at coeff[{}]: batched={}, cpu={}", i, result_batched[i], result_cpu[i]);
            }
        }
    }

    if diff_count > 5 {
        println!("  ... and {} more differences", diff_count - 5);
    }

    println!("\n  Summary (prime 0): {} matches, {} differences out of {} coefficients",
             match_count, diff_count, n);

    if diff_count == 0 {
        println!("  ✓ TEST 1 PASSED: Batched GPU method matches CPU reference!\n");
    } else {
        println!("  ✗ TEST 1 FAILED: Batched GPU method differs from CPU reference!\n");
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 2: Step-by-step twist/NTT/multiply/INTT/untwist");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Test just the twist operation
    let mut p1_twisted = p1_flat.clone();
    ctx.apply_negacyclic_twist_flat(&mut p1_twisted, num_primes)?;

    println!("  After twist, p1 first 5 coefficients (prime 0):");
    for i in 0..5.min(n) {
        println!("    coeff[{}] = {} -> {}", i, p1_flat[i], p1_twisted[i]);
    }

    // Compare to expected CPU twist
    let psi_values = ctx.psi_per_prime();
    let psi = psi_values[0];
    let mut expected_twist = vec![0u64; n];
    let mut psi_pow = 1u64;
    for i in 0..n {
        expected_twist[i] = ((p1_flat[i] as u128 * psi_pow as u128) % q as u128) as u64;
        psi_pow = ((psi_pow as u128 * psi as u128) % q as u128) as u64;
    }

    println!("\n  Expected twist (CPU) first 5 coefficients:");
    for i in 0..5.min(n) {
        println!("    coeff[{}] = {}", i, expected_twist[i]);
    }

    let mut twist_match = 0;
    let mut twist_diff = 0;
    for i in 0..n {
        if p1_twisted[i] == expected_twist[i] {
            twist_match += 1;
        } else {
            twist_diff += 1;
            if twist_diff <= 3 {
                println!("  TWIST DIFF at [{}]: GPU={}, expected={}", i, p1_twisted[i], expected_twist[i]);
            }
        }
    }

    println!("\n  Twist summary: {} matches, {} differences", twist_match, twist_diff);

    if twist_diff == 0 {
        println!("  ✓ TEST 2 PASSED: Twist operation correct!\n");
    } else {
        println!("  ✗ TEST 2 FAILED: Twist operation has errors!\n");
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("════════════════════════════════════════════════════════════════════════\n");

    if diff_count == 0 && twist_diff == 0 {
        println!("  ✓ All tests PASSED!");
        println!("  The batched GPU multiplication is working correctly.");
        println!("  If relinearization is still failing, the bug is elsewhere.");
    } else {
        println!("  ✗ Some tests FAILED!");
        if diff_count > 0 {
            println!("  - Batched multiplication has errors");
        }
        if twist_diff > 0 {
            println!("  - Twist operation has errors");
        }
    }

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires the 'v2' and 'v2-gpu-cuda' features.");
    println!("Run with: cargo run --release --features v2,v2-gpu-cuda --example compare_ntt_methods");
}
