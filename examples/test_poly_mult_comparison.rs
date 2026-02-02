//! Test: Compare CPU and Metal polynomial multiplication with IDENTICAL inputs

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
    println!("Testing polynomial multiplication: CPU vs Metal\n");

    let params = CliffordFHEParams::new_test_ntt_4096();
    let n = params.n;
    let moduli = &params.moduli[..3];  // Use 3 primes
    let num_primes = moduli.len();

    println!("N = {}, primes = {:?}\n", n, moduli);

    // Create simple test polynomials
    let mut a_flat = vec![0u64; n * num_primes];
    let mut b_flat = vec![0u64; n * num_primes];

    // Set a[0] = 123, a[1] = 456 (for all primes)
    // Set b[0] = 789, b[1] = 101 (for all primes)
    for j in 0..num_primes {
        a_flat[0 * num_primes + j] = 123;
        a_flat[1 * num_primes + j] = 456;
        b_flat[0 * num_primes + j] = 789;
        b_flat[1 * num_primes + j] = 101;
    }

    // === CPU Version ===
    println!("--- CPU Polynomial Multiplication ---");
    let key_ctx = KeyContext::new(params.clone());
    let ntt_ctx = key_ctx.ntt_contexts.iter().find(|ctx| ctx.q == moduli[0]).unwrap();

    // Extract first prime
    let mut a_cpu = vec![0u64; n];
    let mut b_cpu = vec![0u64; n];
    for i in 0..n {
        a_cpu[i] = a_flat[i * num_primes + 0];
        b_cpu[i] = b_flat[i * num_primes + 0];
    }

    println!("Input a[0:2]: {} {}", a_cpu[0], a_cpu[1]);
    println!("Input b[0:2]: {} {}", b_cpu[0], b_cpu[1]);

    let product_cpu = ntt_ctx.multiply_polynomials(&a_cpu, &b_cpu);

    println!("Output product[0:2]: {} {}", product_cpu[0], product_cpu[1]);
    println!();

    // === Metal Version ===
    println!("--- Metal Polynomial Multiplication ---");
    let metal_ctx = MetalCkksContext::new(params.clone())?;

    println!("Input a[0:2]: {} {}", a_flat[0 * num_primes + 0], a_flat[1 * num_primes + 0]);
    println!("Input b[0:2]: {} {}", b_flat[0 * num_primes + 0], b_flat[1 * num_primes + 0]);

    let product_metal = metal_ctx.multiply_polys_flat_ntt_negacyclic(&a_flat, &b_flat, moduli)?;

    println!("Output product[0:2]: {} {}", product_metal[0 * num_primes + 0], product_metal[1 * num_primes + 0]);
    println!();

    // === Comparison ===
    println!("--- Comparison ---");
    let mut all_match = true;
    for i in 0..n.min(10) {  // Check first 10 coefficients
        let cpu_val = product_cpu[i];
        let metal_val = product_metal[i * num_primes + 0];
        if cpu_val != metal_val {
            println!("Mismatch at coeff[{}]: CPU={}, Metal={}", i, cpu_val, metal_val);
            all_match = false;
        }
    }

    if all_match {
        println!("✅ All coefficients match!");
        Ok(())
    } else {
        Err("❌ Polynomial multiplication results differ!".to_string())
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
