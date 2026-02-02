//! Test to verify multiply_plain_metal_native_rescale produces correct output layout

use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::{MetalCiphertext, MetalCkksContext};
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

fn main() -> Result<(), String> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║    Multiply-Rescale Layout Test                              ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Create params and context
    let params = CliffordFHEParams::new_v3_bootstrap_metal()?;
    let ctx = MetalCkksContext::new(params.clone())?;
    let n = params.n;

    println!("N = {}, num_primes = {}", n, params.moduli.len());

    // Create a simple ciphertext at level 2 (3 primes active)
    let level = 2;
    let num_primes = level + 1; // 3 primes
    let ct_stride = params.moduli.len(); // Full size (20 primes)

    // Initialize ciphertext in strided layout: c[coeff_idx * stride + prime_idx]
    let mut c0 = vec![0u64; n * ct_stride];
    let mut c1 = vec![0u64; n * ct_stride];

    // Fill first 3 primes with test data
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            let q = params.moduli[prime_idx];
            c0[coeff_idx * ct_stride + prime_idx] = (coeff_idx as u64 * 1000 + prime_idx as u64) % q;
            c1[coeff_idx * ct_stride + prime_idx] = (coeff_idx as u64 * 2000 + prime_idx as u64) % q;
        }
    }

    let ct = MetalCiphertext {
        c0,
        c1,
        n,
        num_primes: ct_stride,
        level,
        scale: 1e10,
    };

    // Create plaintext (all ones) in strided layout
    let mut pt = vec![1u64; n * num_primes];
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            pt[coeff_idx * num_primes + prime_idx] = 1u64;
        }
    }

    println!("Before multiply_plain_metal_native_rescale:");
    println!("  ct.level = {}", ct.level);
    println!("  ct.num_primes = {}", ct.num_primes);
    println!("  ct.c0.len() = {}", ct.c0.len());
    println!("  ct.c1.len() = {}", ct.c1.len());
    println!("  pt.len() = {}", pt.len());

    // Multiply and rescale
    let ct_rescaled = ct.multiply_plain_metal_native_rescale(&pt, &ctx)?;

    println!("\nAfter multiply_plain_metal_native_rescale:");
    println!("  ct_rescaled.level = {}", ct_rescaled.level);
    println!("  ct_rescaled.num_primes = {}", ct_rescaled.num_primes);
    println!("  ct_rescaled.c0.len() = {}", ct_rescaled.c0.len());
    println!("  ct_rescaled.c1.len() = {}", ct_rescaled.c1.len());

    // Expected: level should decrease by 1, so level=1, num_primes=2
    let expected_level = 1;
    let expected_active_primes = 2;

    if ct_rescaled.level != expected_level {
        println!("❌ FAIL: Expected level={}, got level={}", expected_level, ct_rescaled.level);
        return Err("Level mismatch".to_string());
    }

    // Check that c0 and c1 have the correct size
    // With the fix to match hybrid version, output is now COMPACT:
    // c0/c1 have size n × num_primes_out (not padded to full stride)
    let expected_len = n * expected_active_primes;
    if ct_rescaled.c0.len() != expected_len {
        println!("❌ FAIL: Expected c0.len()={}, got {}", expected_len, ct_rescaled.c0.len());
        return Err("Output size mismatch".to_string());
    }

    // Output is in STRIDED layout with stride = expected_active_primes
    let output_stride = expected_active_primes;

    // Verify that the first 2 primes have non-zero values
    let mut non_zero_count_c0 = 0;
    let mut non_zero_count_c1 = 0;

    for coeff_idx in 0..n {
        for prime_idx in 0..expected_active_primes {
            if ct_rescaled.c0[coeff_idx * output_stride + prime_idx] != 0 {
                non_zero_count_c0 += 1;
            }
            if ct_rescaled.c1[coeff_idx * output_stride + prime_idx] != 0 {
                non_zero_count_c1 += 1;
            }
        }
    }

    println!("\nOutput verification:");
    println!("  Non-zero values in first {} primes:", expected_active_primes);
    println!("    c0: {}/{} coefficients", non_zero_count_c0, n * expected_active_primes);
    println!("    c1: {}/{} coefficients", non_zero_count_c1, n * expected_active_primes);

    // Check a few sample values
    println!("\nSample output values (first 5 coefficients, first 2 primes):");
    for coeff_idx in 0..5.min(n) {
        println!("  Coeff {}:", coeff_idx);
        for prime_idx in 0..expected_active_primes {
            let val_c0 = ct_rescaled.c0[coeff_idx * output_stride + prime_idx];
            let val_c1 = ct_rescaled.c1[coeff_idx * output_stride + prime_idx];
            println!("    Prime {}: c0={}, c1={}", prime_idx, val_c0, val_c1);
        }
    }

    println!("\n✅ SUCCESS: Layout test passed!");
    Ok(())
}
