//! Debug: Check NTT multiplication consistency across primes
//!
//! The bug shows Metal has inconsistent errors across primes.
//! Let's check if NTT multiplication itself is consistent.

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::ckks::MetalCkksContext,
    backends::cpu_optimized::{
        keys::KeyContext,
        ntt::NttContext,
    },
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("NTT MULTIPLICATION CONSISTENCY CHECK\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let metal_ctx = MetalCkksContext::new(params.clone())?;

    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];

    // Create two polynomials with KNOWN integer values
    // a = [5, 0, 0, ...] (constant 5)
    // b = [7, 0, 0, ...] (constant 7)
    let mut a = vec![0u64; n * num_primes];
    let mut b = vec![0u64; n * num_primes];
    for j in 0..num_primes {
        a[0 * num_primes + j] = 5;
        b[0 * num_primes + j] = 7;
    }

    println!("a = [5, 0, 0, ...] (constant 5)");
    println!("b = [7, 0, 0, ...] (constant 7)");
    println!("Expected a × b = [35, 0, 0, ...] (constant 35)\n");

    // Multiply using Metal GPU NTT
    let result = metal_ctx.multiply_polys_flat_ntt_negacyclic(&a, &b, moduli)?;

    println!("Metal result (coeff 0): {:?}",
        (0..num_primes).map(|j| result[j]).collect::<Vec<_>>());

    // Expected: 35 for all primes
    let expected = 35u64;
    let errors: Vec<i64> = (0..num_primes).map(|j| result[j] as i64 - expected as i64).collect();
    println!("Errors: {:?}", errors);
    let consistent = errors.iter().all(|&e| e == 0);
    println!("Consistent: {}\n", if consistent { "✅" } else { "❌" });

    // Now test with a polynomial that's different per prime (like from RNS)
    // Create a = [x, 0, 0, ...] where x is different mod each prime
    // This simulates what happens with encrypted data

    // Use large value: 10^18 represented mod each prime
    let large_val: u128 = 1_000_000_000_000_000_000;
    let mut a_large = vec![0u64; n * num_primes];
    for (j, &q) in moduli.iter().enumerate() {
        a_large[0 * num_primes + j] = (large_val % q as u128) as u64;
    }

    println!("Testing with large value: {}", large_val);
    println!("a as RNS: {:?}",
        (0..num_primes).map(|j| a_large[j]).collect::<Vec<_>>());

    // b = [3, 0, 0, ...]
    let mut b_small = vec![0u64; n * num_primes];
    for j in 0..num_primes {
        b_small[0 * num_primes + j] = 3;
    }

    // Expected: 3 * 10^18 mod each prime
    let expected_large: Vec<u64> = moduli.iter().map(|&q| ((3 * large_val) % q as u128) as u64).collect();

    let result_large = metal_ctx.multiply_polys_flat_ntt_negacyclic(&a_large, &b_small, moduli)?;

    println!("\nMetal result: {:?}",
        (0..num_primes).map(|j| result_large[j]).collect::<Vec<_>>());
    println!("Expected:     {:?}", expected_large);

    let errors_large: Vec<i64> = (0..num_primes).map(|j| {
        let diff = if result_large[j] >= expected_large[j] {
            result_large[j] - expected_large[j]
        } else {
            moduli[j] - (expected_large[j] - result_large[j])
        };
        if diff > moduli[j]/2 { diff as i64 - moduli[j] as i64 } else { diff as i64 }
    }).collect();
    println!("Errors: {:?}", errors_large);
    let consistent2 = errors_large.iter().all(|&e| e == 0);
    println!("Consistent: {}\n", if consistent2 { "✅" } else { "❌" });

    // Compare with CPU NTT multiplication
    println!("=== CPU NTT Comparison ===\n");

    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);

        let mut a_prime = vec![0u64; n];
        let mut b_prime = vec![0u64; n];
        a_prime[0] = a_large[prime_idx];
        b_prime[0] = 3;

        let cpu_result = ntt_ctx.multiply_polynomials(&a_prime, &b_prime);
        let metal_result_prime = result_large[prime_idx];

        println!("Prime {}: CPU={}, Metal={}, Match={}",
            prime_idx, cpu_result[0], metal_result_prime,
            if cpu_result[0] == metal_result_prime { "✅" } else { "❌" });
    }

    Ok(())
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
