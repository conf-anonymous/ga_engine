//! Compare CPU vs Metal NTT polynomial multiplication step-by-step
//!
//! This will help identify where the twist/untwist diverges

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::backends::gpu_metal::ntt::MetalNttContext;

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This test requires Metal GPU.");
    println!("Run with: cargo run --release --features v2,v3,v2-gpu-metal --example test_ntt_comparison");
}

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          CPU vs Metal NTT Comparison (with twist)               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let q = params.moduli[0];

    println!("Parameters: N={}, q={}\n", n, q);

    // Test polynomials: simple non-zero values
    let a = vec![42u64; n];
    let b = vec![2u64; n];

    println!("Test: multiply constant polynomials a(x)=42, b(x)=2");
    println!("Expected result: polynomial with all coefficients = 84\n");

    // ===================== CPU VERSION =====================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("CPU NTT (with twist/untwist for negacyclic)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let cpu_ntt = NttContext::new(n, q);
    let cpu_result = cpu_ntt.multiply_polynomials(&a, &b);

    println!("CPU result[0..5]: {:?}", &cpu_result[0..5]);
    println!("CPU result[n-5..n]: {:?}", &cpu_result[n-5..n]);

    // ===================== METAL VERSION =====================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Metal NTT (with twist/untwist for negacyclic)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Find psi (same algorithm as CPU)
    let psi = find_primitive_2n_root(n, q)?;
    println!("Found psi = {}\n", psi);

    let metal_ntt = MetalNttContext::new(n, q, psi)?;

    // Manual implementation matching multiply_polys_flat_ntt_negacyclic
    let mut a_metal = a.clone();
    let mut b_metal = b.clone();

    println!("Step 1: TWIST - multiply by psi^i");
    let psi_powers = metal_ntt.psi_powers();
    for i in 0..n {
        a_metal[i] = mul_mod(a_metal[i], psi_powers[i], q);
        b_metal[i] = mul_mod(b_metal[i], psi_powers[i], q);
    }
    println!("  a_metal[0] after twist: {}", a_metal[0]);
    println!("  b_metal[0] after twist: {}", b_metal[0]);

    println!("\nStep 2: Forward NTT");
    metal_ntt.forward(&mut a_metal)?;
    metal_ntt.forward(&mut b_metal)?;
    println!("  a_metal[0] after NTT: {}", a_metal[0]);
    println!("  b_metal[0] after NTT: {}", b_metal[0]);

    println!("\nStep 3: Pointwise multiply");
    let mut result_metal = vec![0u64; n];
    metal_ntt.pointwise_multiply(&a_metal, &b_metal, &mut result_metal)?;
    println!("  result[0] after multiply: {}", result_metal[0]);

    println!("\nStep 4: Inverse NTT");
    metal_ntt.inverse(&mut result_metal)?;
    println!("  result[0] after INTT: {}", result_metal[0]);

    println!("\nStep 5: UNTWIST - multiply by psi^{{-i}}");
    let psi_inv_powers = metal_ntt.psi_inv_powers();
    for i in 0..n {
        result_metal[i] = mul_mod(result_metal[i], psi_inv_powers[i], q);
    }
    println!("  result[0] after untwist: {}", result_metal[0]);

    println!("\nMetal result[0..5]: {:?}", &result_metal[0..5]);
    println!("Metal result[n-5..n]: {:?}", &result_metal[n-5..n]);

    // ===================== COMPARISON =====================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Comparison");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut all_match = true;
    let mut first_mismatch = None;

    for i in 0..n {
        if cpu_result[i] != result_metal[i] {
            if first_mismatch.is_none() {
                first_mismatch = Some(i);
            }
            all_match = false;
        }
    }

    if all_match {
        println!("✅ SUCCESS: CPU and Metal results match perfectly!");
    } else {
        println!("❌ MISMATCH: Results differ!");
        if let Some(idx) = first_mismatch {
            println!("\nFirst mismatch at index {}:", idx);
            println!("  CPU:   {}", cpu_result[idx]);
            println!("  Metal: {}", result_metal[idx]);
        }

        // Show a few more mismatches
        println!("\nSample of differences:");
        for i in 0..std::cmp::min(10, n) {
            if cpu_result[i] != result_metal[i] {
                println!("  [{}] CPU: {}, Metal: {}", i, cpu_result[i], result_metal[i]);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "v2-gpu-metal")]
fn find_primitive_2n_root(n: usize, q: u64) -> Result<u64, String> {
    let two_n = (2 * n) as u64;
    if (q - 1) % two_n != 0 {
        return Err(format!("q is not NTT-friendly for n = {}", n));
    }

    for candidate in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        if is_primitive_root_candidate(candidate, q) {
            let exponent = (q - 1) / two_n;
            return Ok(pow_mod(candidate, exponent, q));
        }
    }

    for candidate in 32..20000u64 {
        if is_primitive_root_candidate(candidate, q) {
            let exponent = (q - 1) / two_n;
            return Ok(pow_mod(candidate, exponent, q));
        }
    }

    Err(format!("Failed to find primitive root for q = {}", q))
}

#[cfg(feature = "v2-gpu-metal")]
fn is_primitive_root_candidate(g: u64, q: u64) -> bool {
    let exp = (q - 1) / 2;
    pow_mod(g, exp, q) != 1
}

#[cfg(feature = "v2-gpu-metal")]
fn pow_mod(mut base: u64, mut exp: u64, q: u64) -> u64 {
    let mut result = 1u64;
    base %= q;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod(result, base, q);
        }
        base = mul_mod(base, base, q);
        exp >>= 1;
    }
    result
}

#[cfg(feature = "v2-gpu-metal")]
fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}
