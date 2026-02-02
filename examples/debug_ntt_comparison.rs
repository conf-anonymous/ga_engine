//! Debug: Compare CPU NTT vs Metal NTT for polynomial multiplication

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::KeyContext,
        ntt::NttContext,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Debug: CPU NTT vs Metal NTT Comparison");
    println!("======================================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let metal_ctx = MetalCkksContext::new(params.clone())?;

    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];

    // Create two known test polynomials
    println!("Creating test polynomials...");
    let mut poly_a = vec![0u64; n * num_primes];
    let mut poly_b = vec![0u64; n * num_primes];

    // poly_a = [1, 2, 3, 0, ..., 0]
    // poly_b = [4, 5, 6, 0, ..., 0]
    for (j, &q) in moduli.iter().enumerate() {
        poly_a[0 * num_primes + j] = 1;
        poly_a[1 * num_primes + j] = 2;
        poly_a[2 * num_primes + j] = 3;

        poly_b[0 * num_primes + j] = 4;
        poly_b[1 * num_primes + j] = 5;
        poly_b[2 * num_primes + j] = 6;
    }

    println!("poly_a = [1, 2, 3, 0, ...]");
    println!("poly_b = [4, 5, 6, 0, ...]");

    // Expected result (standard polynomial multiplication):
    // (1 + 2x + 3x²) × (4 + 5x + 6x²)
    // = 4 + 5x + 6x² + 8x + 10x² + 12x³ + 12x² + 15x³ + 18x⁴
    // = 4 + 13x + 28x² + 27x³ + 18x⁴
    // For negacyclic (mod x^n + 1), with n=1024, no wrapping for these low degrees
    println!("Expected (standard poly): [4, 13, 28, 27, 18, 0, ...]");

    // CPU multiplication
    println!("\n=== CPU NTT Multiplication ===");
    let cpu_result = cpu_multiply_flat(&poly_a, &poly_b, moduli, n)?;
    println!("CPU result first 6 coeffs:");
    for i in 0..6 {
        print!("  [{}]: ", i);
        for j in 0..num_primes {
            print!("{} ", cpu_result[i * num_primes + j]);
        }
        println!();
    }

    // Metal multiplication
    println!("\n=== Metal NTT Multiplication ===");
    let metal_result = metal_ctx.multiply_polys_flat_ntt_negacyclic(&poly_a, &poly_b, moduli)?;
    println!("Metal result first 6 coeffs:");
    for i in 0..6 {
        print!("  [{}]: ", i);
        for j in 0..num_primes {
            print!("{} ", metal_result[i * num_primes + j]);
        }
        println!();
    }

    // Compare
    println!("\n=== Comparison ===");
    let mut mismatches = 0;
    for i in 0..(n * num_primes) {
        if cpu_result[i] != metal_result[i] {
            mismatches += 1;
            if mismatches <= 5 {
                let coeff_idx = i / num_primes;
                let prime_idx = i % num_primes;
                println!("  Mismatch at coeff={}, prime={}: CPU={}, Metal={}",
                    coeff_idx, prime_idx, cpu_result[i], metal_result[i]);
            }
        }
    }
    if mismatches > 5 {
        println!("  ... and {} more mismatches", mismatches - 5);
    }
    println!("Total mismatches: {} / {}", mismatches, n * num_primes);

    if mismatches == 0 {
        println!("\n✅ CPU and Metal NTT produce IDENTICAL results");
    } else {
        println!("\n❌ CPU and Metal NTT produce DIFFERENT results!");
        return Err("NTT mismatch".to_string());
    }

    // Now test with actual EVK-like polynomials (larger random values)
    println!("\n\n=== Testing with EVK-like random polynomials ===");
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut evk_like_a = vec![0u64; n * num_primes];
    let mut evk_like_b = vec![0u64; n * num_primes];

    for i in 0..n {
        for (j, &q) in moduli.iter().enumerate() {
            evk_like_a[i * num_primes + j] = rng.gen::<u64>() % q;
            evk_like_b[i * num_primes + j] = rng.gen::<u64>() % q;
        }
    }

    let cpu_evk_result = cpu_multiply_flat(&evk_like_a, &evk_like_b, moduli, n)?;
    let metal_evk_result = metal_ctx.multiply_polys_flat_ntt_negacyclic(&evk_like_a, &evk_like_b, moduli)?;

    let mut evk_mismatches = 0;
    for i in 0..(n * num_primes) {
        if cpu_evk_result[i] != metal_evk_result[i] {
            evk_mismatches += 1;
        }
    }
    println!("EVK-like multiplication mismatches: {} / {}", evk_mismatches, n * num_primes);

    if evk_mismatches == 0 {
        println!("✅ CPU and Metal NTT agree on EVK-like polynomials");
        Ok(())
    } else {
        println!("❌ CPU and Metal NTT disagree on EVK-like polynomials!");
        Err("EVK multiplication mismatch".to_string())
    }
}

#[cfg(feature = "v2-gpu-metal")]
fn cpu_multiply_flat(a: &[u64], b: &[u64], moduli: &[u64], n: usize) -> Result<Vec<u64>, String> {
    let num_primes = moduli.len();
    let mut result = vec![0u64; n * num_primes];

    for (prime_idx, &q) in moduli.iter().enumerate() {
        let ntt_ctx = NttContext::new(n, q);

        let mut a_poly = vec![0u64; n];
        let mut b_poly = vec![0u64; n];
        for i in 0..n {
            a_poly[i] = a[i * num_primes + prime_idx];
            b_poly[i] = b[i * num_primes + prime_idx];
        }

        let product = ntt_ctx.multiply_polynomials(&a_poly, &b_poly);

        for i in 0..n {
            result[i * num_primes + prime_idx] = product[i];
        }
    }

    Ok(result)
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
