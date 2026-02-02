//! Golden Compare Test: CUDA GPU Rescaling vs CPU Reference
//!
//! This test validates that the CUDA GPU rescaling kernel produces
//! bit-exact results compared to the CPU reference implementation.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example test_cuda_rescale_golden_compare
//! ```

use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use num_bigint::BigUint;
use rand::Rng;

fn main() -> Result<(), String> {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║  CUDA GPU Rescaling Golden Compare Test                      ║");
    println!("║  Validates bit-exact correctness vs CPU reference            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Step 1: Create FHE parameters
    println!("Step 1: Creating FHE parameters");
    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();
    println!("  N = {}, num_primes = {}", n, num_primes);
    println!("  Moduli: {:?}\n", params.moduli);

    // Step 2: Initialize CUDA context
    println!("Step 2: Initializing CUDA context");
    let cuda_ctx = CudaCkksContext::new(params.clone())?;
    println!("  ✅ CUDA context initialized\n");

    // Step 3: Test rescaling at different levels with random data
    println!("Step 3: Testing rescaling at different levels\n");

    let mut rng = rand::thread_rng();
    let num_tests = 5;
    let num_test_coeffs = 100;  // Test first 100 coefficients

    for test_idx in 0..num_tests {
        println!("  Test {}/{}: Random polynomial rescaling", test_idx + 1, num_tests);

        // Test each rescaleable level (1 to num_primes-1)
        for level in 1..num_primes {
            let num_primes_in = level + 1;
            let num_primes_out = level;
            let moduli_slice = &params.moduli[..=level];

            // Generate random polynomial in flat RNS layout
            let mut poly_in = vec![0u64; n * num_primes_in];
            for prime_idx in 0..num_primes_in {
                let q = moduli_slice[prime_idx];
                for coeff_idx in 0..num_test_coeffs {
                    poly_in[prime_idx * n + coeff_idx] = rng.gen::<u64>() % q;
                }
            }

            // CPU reference (using flat RNS layout)
            let cpu_result = cpu_rescale_reference(&poly_in, moduli_slice, n)?;

            // GPU rescaling (CUDA kernel)
            let gpu_result = cuda_ctx.exact_rescale_gpu_flat(&poly_in, level)?;

            // Compare results
            let expected_len = n * num_primes_out;
            assert_eq!(cpu_result.len(), expected_len,
                "CPU result length mismatch at level {}", level);
            assert_eq!(gpu_result.len(), expected_len,
                "GPU result length mismatch at level {}", level);

            let mut mismatches = 0;
            let mut first_mismatch_idx = None;

            for coeff_idx in 0..num_test_coeffs {
                for prime_idx in 0..num_primes_out {
                    let idx = prime_idx * n + coeff_idx;
                    if cpu_result[idx] != gpu_result[idx] {
                        mismatches += 1;
                        if first_mismatch_idx.is_none() {
                            first_mismatch_idx = Some((coeff_idx, prime_idx));
                        }
                    }
                }
            }

            if mismatches > 0 {
                let (coeff_idx, prime_idx) = first_mismatch_idx.unwrap();
                let idx = prime_idx * n + coeff_idx;
                println!("    ❌ Level {}: {} mismatches", level, mismatches);
                println!("       First mismatch at poly[{}][{}]:", coeff_idx, prime_idx);
                println!("         CPU:  {}", cpu_result[idx]);
                println!("         CUDA: {}", gpu_result[idx]);

                // Debug info
                let q_last = moduli_slice[level];
                let r_last = poly_in[level * n + coeff_idx];
                let r_i = poly_in[prime_idx * n + coeff_idx];
                let q_i = moduli_slice[prime_idx];

                println!("       Debug:");
                println!("         q_last = {}, r_last = {}", q_last, r_last);
                println!("         q_i = {}, r_i = {}", q_i, r_i);

                return Err(format!("GPU rescaling mismatch at level {}", level));
            } else {
                println!("    ✅ Level {}: 0 mismatches (bit-exact)", level);
            }
        }

        println!();
    }

    // Step 4: Test with specific edge cases
    println!("Step 4: Testing edge cases\n");

    // Test case 1: All zeros
    println!("  Edge case 1: All zeros");
    for level in 1..num_primes {
        let num_primes_in = level + 1;
        let moduli_slice = &params.moduli[..=level];
        let poly_zeros = vec![0u64; n * num_primes_in];

        let cpu_result = cpu_rescale_reference(&poly_zeros, moduli_slice, n)?;
        let gpu_result = cuda_ctx.exact_rescale_gpu_flat(&poly_zeros, level)?;

        let mismatches = cpu_result.iter().zip(&gpu_result)
            .filter(|(a, b)| a != b)
            .count();

        if mismatches > 0 {
            println!("    ❌ Level {}: {} mismatches with zeros", level, mismatches);
            return Err(format!("GPU rescaling failed on zeros at level {}", level));
        }
    }
    println!("    ✅ All zeros: bit-exact for all levels");

    // Test case 2: Maximum values (near modulus)
    println!("  Edge case 2: Maximum values (near modulus)");
    for level in 1..num_primes {
        let num_primes_in = level + 1;
        let moduli_slice = &params.moduli[..=level];
        let mut poly_max = vec![0u64; n * num_primes_in];

        for prime_idx in 0..num_primes_in {
            for coeff_idx in 0..num_test_coeffs {
                poly_max[prime_idx * n + coeff_idx] = moduli_slice[prime_idx] - 1;
            }
        }

        let cpu_result = cpu_rescale_reference(&poly_max, moduli_slice, n)?;
        let gpu_result = cuda_ctx.exact_rescale_gpu_flat(&poly_max, level)?;

        let mismatches = cpu_result.iter().zip(&gpu_result)
            .take(num_test_coeffs * level)  // Only check test coefficients
            .filter(|(a, b)| a != b)
            .count();

        if mismatches > 0 {
            println!("    ❌ Level {}: {} mismatches with max values", level, mismatches);
            return Err(format!("GPU rescaling failed on max values at level {}", level));
        }
    }
    println!("    ✅ Maximum values: bit-exact for all levels");

    // Test case 3: Boundary values (around q_last/2)
    println!("  Edge case 3: Boundary values (around q_last/2)");
    for level in 1..num_primes {
        let num_primes_in = level + 1;
        let moduli_slice = &params.moduli[..=level];
        let q_last = moduli_slice[level];
        let q_half = q_last / 2;

        let mut poly_boundary = vec![0u64; n * num_primes_in];
        for coeff_idx in 0..num_test_coeffs {
            for prime_idx in 0..num_primes_in {
                // Alternate between q_last/2 - 1, q_last/2, q_last/2 + 1
                let offset = (coeff_idx % 3) as i64 - 1;
                let val = if prime_idx == level {
                    ((q_half as i64 + offset).rem_euclid(q_last as i64)) as u64
                } else {
                    q_half % moduli_slice[prime_idx]
                };
                poly_boundary[prime_idx * n + coeff_idx] = val;
            }
        }

        let cpu_result = cpu_rescale_reference(&poly_boundary, moduli_slice, n)?;
        let gpu_result = cuda_ctx.exact_rescale_gpu_flat(&poly_boundary, level)?;

        let mismatches = cpu_result.iter().zip(&gpu_result)
            .take(num_test_coeffs * level)  // Only check test coefficients
            .filter(|(a, b)| a != b)
            .count();

        if mismatches > 0 {
            println!("    ❌ Level {}: {} mismatches with boundary values", level, mismatches);
            return Err(format!("GPU rescaling failed on boundary values at level {}", level));
        }
    }
    println!("    ✅ Boundary values: bit-exact for all levels\n");

    // Final summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Results:");
    println!("  Random tests: {} × {} levels", num_tests, num_primes - 1);
    println!("  Edge cases: 3 categories × {} levels", num_primes - 1);
    println!("  Total mismatches: 0");
    println!("═══════════════════════════════════════════════════════════════");
    println!("✅ CUDA GPU RESCALING IS BIT-EXACT");
    println!("   Ready to proceed with rotation operations!\n");

    Ok(())
}

/// CPU reference rescaling using flat RNS layout (matching GPU)
/// Input: poly_in[prime_idx * n + coeff_idx]
/// Output: result[prime_idx * n + coeff_idx]
fn cpu_rescale_reference(poly_in: &[u64], moduli: &[u64], n: usize) -> Result<Vec<u64>, String> {
    let num_primes_in = moduli.len();
    let num_primes_out = num_primes_in - 1;
    let q_last = moduli[num_primes_in - 1];
    let mut result = vec![0u64; n * num_primes_out];

    for coeff_idx in 0..n {
        // Reconstruct coefficient using CRT
        let mut c = BigUint::from(0u64);
        let mut q_product = BigUint::from(1u64);
        for &q in moduli.iter() {
            q_product *= BigUint::from(q);
        }

        // Read from flat RNS layout
        for (prime_idx, &q) in moduli.iter().enumerate() {
            let r = poly_in[prime_idx * n + coeff_idx];  // Flat RNS layout
            let q_i = BigUint::from(q);
            let q_hat = &q_product / &q_i;
            let q_hat_inv = mod_inverse_bigint(&q_hat, q).ok_or("Inverse failed")?;
            let term = BigUint::from(r) * &q_hat * q_hat_inv;
            c = (c + term) % &q_product;
        }

        // DRLMQ rescaling: c_rounded = ⌊(c + q_last/2) / q_last⌋
        let q_last_big = BigUint::from(q_last);
        let c_rounded = (c + &q_last_big / 2u64) / q_last_big;

        // Write to flat RNS layout
        for (out_idx, &q) in moduli[..num_primes_out].iter().enumerate() {
            let r = (&c_rounded % BigUint::from(q)).to_u64_digits().first().copied().unwrap_or(0);
            result[out_idx * n + coeff_idx] = r;  // Flat RNS layout
        }
    }
    Ok(result)
}

fn mod_inverse_bigint(a: &BigUint, m: u64) -> Option<BigUint> {
    use num_bigint::BigInt;
    let m_big = BigUint::from(m);
    let (g, x, _) = extended_gcd_bigint(a.clone(), m_big.clone());
    if g != BigUint::from(1u64) {
        return None;
    }
    let m_bigint = BigInt::from(m);
    let x_mod = ((x % &m_bigint) + &m_bigint) % &m_bigint;
    x_mod.to_biguint()
}

fn extended_gcd_bigint(a: BigUint, b: BigUint) -> (BigUint, num_bigint::BigInt, num_bigint::BigInt) {
    use num_bigint::BigInt;
    if b == BigUint::from(0u64) {
        return (a, BigInt::from(1), BigInt::from(0));
    }
    let (g, x1, y1) = extended_gcd_bigint(b.clone(), &a % &b);
    let x = y1.clone();
    let y = x1 - BigInt::from(&a / &b) * y1;
    (g, x, y)
}
