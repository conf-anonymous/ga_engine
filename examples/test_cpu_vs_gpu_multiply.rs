//! Test CPU negacyclic multiply vs GPU NTT multiply
//!
//! This test determines if the CPU reference negacyclic multiply is correct
//! by comparing it against the GPU NTT-based multiply.
//!
//! CRITICAL: test_multiply_polys_ntt expects STRIDED layout:
//!   strided[coeff_idx * num_primes + prime_idx]
//! NOT flat layout:
//!   flat[prime_idx * n + coeff_idx]
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example test_cpu_vs_gpu_multiply
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_cuda::ckks::CudaCkksContext,
    params::CliffordFHEParams,
};

/// CPU negacyclic multiplication (coefficient domain) - O(N^2)
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

/// Extract coefficients for a single prime from strided layout
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn extract_prime_from_strided(strided: &[u64], n: usize, num_primes: usize, prime_idx: usize) -> Vec<u64> {
    let mut result = vec![0u64; n];
    for coeff_idx in 0..n {
        result[coeff_idx] = strided[coeff_idx * num_primes + prime_idx];
    }
    result
}

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     TEST: CPU Negacyclic Multiply vs GPU NTT Multiply                  ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();

    println!("Parameters: N={}, num_primes={}", n, num_primes);
    println!("Using STRIDED layout: data[coeff_idx * num_primes + prime_idx]\n");

    // Create CUDA context
    let ctx = CudaCkksContext::new(params.clone())?;

    // Test with simple known polynomials first
    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 1: Simple polynomials p(x) = 1, q(x) = 1");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // p(x) = 1 (constant), q(x) = 1 (constant)
    // p * q = 1 (should be exact)
    {
        // Create in STRIDED layout: data[coeff_idx * num_primes + prime_idx]
        let mut p1_strided = vec![0u64; n * num_primes];
        let mut p2_strided = vec![0u64; n * num_primes];

        // coeff[0] = 1 for all primes
        for prime_idx in 0..num_primes {
            p1_strided[0 * num_primes + prime_idx] = 1;  // coefficient of x^0
            p2_strided[0 * num_primes + prime_idx] = 1;  // coefficient of x^0
        }

        let gpu_result = ctx.test_multiply_polys_ntt(&p1_strided, &p2_strided, num_primes)?;

        // Extract prime 0 for comparison
        let gpu_prime0 = extract_prime_from_strided(&gpu_result, n, num_primes, 0);

        // CPU reference for prime 0
        let q0 = params.moduli[0];
        let p1_prime0 = extract_prime_from_strided(&p1_strided, n, num_primes, 0);
        let p2_prime0 = extract_prime_from_strided(&p2_strided, n, num_primes, 0);
        let cpu_result = negacyclic_multiply_cpu(&p1_prime0, &p2_prime0, q0, n);

        println!("  Expected: p*q = 1 (coeff[0]=1, rest=0)");
        println!("  GPU result first 5 coeffs (prime 0):");
        for i in 0..5 {
            println!("    coeff[{}] = {} (expected {})", i, gpu_prime0[i], if i == 0 { 1 } else { 0 });
        }
        println!("  CPU result first 5 coeffs:");
        for i in 0..5 {
            println!("    coeff[{}] = {} (expected {})", i, cpu_result[i], if i == 0 { 1 } else { 0 });
        }

        let gpu_matches = gpu_prime0[0] == 1 && gpu_prime0[1..n].iter().all(|&x| x == 0);
        let cpu_matches = cpu_result[0] == 1 && cpu_result[1..n].iter().all(|&x| x == 0);

        println!("\n  GPU: {} | CPU: {}\n",
            if gpu_matches { "✓ PASS" } else { "✗ FAIL" },
            if cpu_matches { "✓ PASS" } else { "✗ FAIL" });
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 2: p(x) = x, q(x) = x => x^2");
    println!("════════════════════════════════════════════════════════════════════════\n");

    {
        let mut p1_strided = vec![0u64; n * num_primes];
        let mut p2_strided = vec![0u64; n * num_primes];

        // coeff[1] = 1 for all primes
        for prime_idx in 0..num_primes {
            p1_strided[1 * num_primes + prime_idx] = 1;
            p2_strided[1 * num_primes + prime_idx] = 1;
        }

        let gpu_result = ctx.test_multiply_polys_ntt(&p1_strided, &p2_strided, num_primes)?;
        let gpu_prime0 = extract_prime_from_strided(&gpu_result, n, num_primes, 0);

        let q0 = params.moduli[0];
        let p1_prime0 = extract_prime_from_strided(&p1_strided, n, num_primes, 0);
        let p2_prime0 = extract_prime_from_strided(&p2_strided, n, num_primes, 0);
        let cpu_result = negacyclic_multiply_cpu(&p1_prime0, &p2_prime0, q0, n);

        println!("  Expected: x * x = x^2 (coeff[2]=1, rest=0)");
        println!("  GPU: coeff[0]={}, coeff[1]={}, coeff[2]={} (expected 0,0,1)",
            gpu_prime0[0], gpu_prime0[1], gpu_prime0[2]);
        println!("  CPU: coeff[0]={}, coeff[1]={}, coeff[2]={} (expected 0,0,1)",
            cpu_result[0], cpu_result[1], cpu_result[2]);

        let gpu_matches = gpu_prime0[2] == 1 && gpu_prime0[0] == 0 && gpu_prime0[1] == 0;
        let cpu_matches = cpu_result[2] == 1 && cpu_result[0] == 0 && cpu_result[1] == 0;

        println!("\n  GPU: {} | CPU: {}\n",
            if gpu_matches { "✓ PASS" } else { "✗ FAIL" },
            if cpu_matches { "✓ PASS" } else { "✗ FAIL" });
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 3: x^(N-1) * x = x^N = -1 (CRITICAL NEGACYCLIC TEST)");
    println!("════════════════════════════════════════════════════════════════════════\n");

    {
        let mut p1_strided = vec![0u64; n * num_primes];
        let mut p2_strided = vec![0u64; n * num_primes];

        // p1 = x^(N-1), p2 = x
        for prime_idx in 0..num_primes {
            p1_strided[(n - 1) * num_primes + prime_idx] = 1;  // x^(N-1)
            p2_strided[1 * num_primes + prime_idx] = 1;         // x
        }

        let gpu_result = ctx.test_multiply_polys_ntt(&p1_strided, &p2_strided, num_primes)?;
        let gpu_prime0 = extract_prime_from_strided(&gpu_result, n, num_primes, 0);

        let q0 = params.moduli[0];
        let p1_prime0 = extract_prime_from_strided(&p1_strided, n, num_primes, 0);
        let p2_prime0 = extract_prime_from_strided(&p2_strided, n, num_primes, 0);
        let cpu_result = negacyclic_multiply_cpu(&p1_prime0, &p2_prime0, q0, n);

        let expected = q0 - 1;  // -1 mod q0
        println!("  Expected: x^N = -1 = {} mod q0", expected);
        println!("  GPU coeff[0] = {} (expected {})", gpu_prime0[0], expected);
        println!("  CPU coeff[0] = {} (expected {})", cpu_result[0], expected);

        // Check non-zero count
        let gpu_nonzero: usize = gpu_prime0.iter().filter(|&&x| x != 0).count();
        let cpu_nonzero: usize = cpu_result.iter().filter(|&&x| x != 0).count();
        println!("  GPU non-zero coeffs: {} (expected 1)", gpu_nonzero);
        println!("  CPU non-zero coeffs: {} (expected 1)", cpu_nonzero);

        let gpu_matches = gpu_prime0[0] == expected && gpu_prime0[1..].iter().all(|&x| x == 0);
        let cpu_matches = cpu_result[0] == expected && cpu_result[1..].iter().all(|&x| x == 0);

        println!("\n  GPU: {} | CPU: {}\n",
            if gpu_matches { "✓ PASS" } else { "✗ FAIL" },
            if cpu_matches { "✓ PASS" } else { "✗ FAIL" });
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 4: x^(N/2) * x^(N/2) = x^N = -1");
    println!("════════════════════════════════════════════════════════════════════════\n");

    {
        let half_n = n / 2;
        let mut p1_strided = vec![0u64; n * num_primes];
        let mut p2_strided = vec![0u64; n * num_primes];

        for prime_idx in 0..num_primes {
            p1_strided[half_n * num_primes + prime_idx] = 1;
            p2_strided[half_n * num_primes + prime_idx] = 1;
        }

        let gpu_result = ctx.test_multiply_polys_ntt(&p1_strided, &p2_strided, num_primes)?;
        let gpu_prime0 = extract_prime_from_strided(&gpu_result, n, num_primes, 0);

        let q0 = params.moduli[0];
        let p1_prime0 = extract_prime_from_strided(&p1_strided, n, num_primes, 0);
        let p2_prime0 = extract_prime_from_strided(&p2_strided, n, num_primes, 0);
        let cpu_result = negacyclic_multiply_cpu(&p1_prime0, &p2_prime0, q0, n);

        let expected = q0 - 1;
        println!("  GPU coeff[0] = {} (expected {})", gpu_prime0[0], expected);
        println!("  CPU coeff[0] = {} (expected {})", cpu_result[0], expected);

        let gpu_matches = gpu_prime0[0] == expected && gpu_prime0[1..].iter().all(|&x| x == 0);
        let cpu_matches = cpu_result[0] == expected && cpu_result[1..].iter().all(|&x| x == 0);

        println!("\n  GPU: {} | CPU: {}\n",
            if gpu_matches { "✓ PASS" } else { "✗ FAIL" },
            if cpu_matches { "✓ PASS" } else { "✗ FAIL" });
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 5: Random polynomials - GPU vs CPU comparison");
    println!("════════════════════════════════════════════════════════════════════════\n");

    {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut p1_strided = vec![0u64; n * num_primes];
        let mut p2_strided = vec![0u64; n * num_primes];

        // Use small random values (0-100) - same values for all primes
        let p1_coeffs: Vec<u64> = (0..n).map(|_| rng.gen_range(0..100)).collect();
        let p2_coeffs: Vec<u64> = (0..n).map(|_| rng.gen_range(0..100)).collect();

        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                p1_strided[coeff_idx * num_primes + prime_idx] = p1_coeffs[coeff_idx];
                p2_strided[coeff_idx * num_primes + prime_idx] = p2_coeffs[coeff_idx];
            }
        }

        let gpu_result = ctx.test_multiply_polys_ntt(&p1_strided, &p2_strided, num_primes)?;
        let gpu_prime0 = extract_prime_from_strided(&gpu_result, n, num_primes, 0);

        let q0 = params.moduli[0];
        let cpu_result = negacyclic_multiply_cpu(&p1_coeffs, &p2_coeffs, q0, n);

        let mut match_count = 0;
        let mut diff_count = 0;
        let mut max_diff = 0u64;

        for i in 0..n {
            if gpu_prime0[i] == cpu_result[i] {
                match_count += 1;
            } else {
                diff_count += 1;
                let diff = if gpu_prime0[i] > cpu_result[i] {
                    gpu_prime0[i] - cpu_result[i]
                } else {
                    cpu_result[i] - gpu_prime0[i]
                };
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff_count <= 10 {
                    println!("  DIFF at coeff[{}]: GPU={}, CPU={}, diff={}",
                        i, gpu_prime0[i], cpu_result[i], diff);
                }
            }
        }

        println!("\n  Summary: {} matches, {} differences", match_count, diff_count);

        if diff_count == 0 {
            println!("  ✓ PASS - GPU and CPU produce IDENTICAL results!\n");
        } else {
            println!("  ✗ FAIL - max_diff = {}", max_diff);
            if max_diff > q0 / 2 {
                println!("  NOTE: max_diff is close to q0={} - indicates sign/wrap error", q0);
            }
            println!();
        }
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("CONCLUSION");
    println!("════════════════════════════════════════════════════════════════════════\n");

    println!("If tests 3,4 PASS: negacyclic convolution works correctly");
    println!("If test 5 PASSES: CPU reference matches GPU (my test code is correct)");
    println!("If test 5 FAILS: Either GPU or CPU has a bug in multiplication");

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires the 'v2' and 'v2-gpu-cuda' features.");
    println!("Run with: cargo run --release --features v2,v2-gpu-cuda --example test_cpu_vs_gpu_multiply");
}
