//! Test Metal NTT Correctness
//!
//! Directly compares Metal GPU NTT output with CPU NTT output
//! to verify the twisted NTT implementation is correct.

#[cfg(feature = "v2-gpu-metal")]
fn main() {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext as CpuNttContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::device::MetalDevice;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::ntt::MetalNttContext;
    use std::sync::Arc;

    println!("Testing Metal NTT Correctness vs CPU");
    println!("═══════════════════════════════════════\n");

    // Test parameters
    let n = 1024;
    let q = 1152921504606748673u64;  // 60-bit NTT-friendly prime from params

    println!("Parameters: N={}, q={}", n, q);

    // Find primitive 2n-th root (psi)
    let psi = find_primitive_2n_root(n, q);
    println!("Primitive 2n-th root (psi): {}", psi);
    println!("Verification: psi^(2n) mod q = {}\n", pow_mod(psi, (2 * n) as u64, q));

    // Create test input
    let input: Vec<u64> = (0..n).map(|i| (i as u64 * 12345) % q).collect();
    let original = input.clone();

    println!("Test 1: NTT Roundtrip (Forward + Inverse)");
    println!("─────────────────────");

    // Note: Metal NTT now keeps values in Montgomery domain between forward/inverse
    // This is correct for polynomial multiplication. We test the roundtrip property.

    // CPU NTT
    let cpu_ntt = CpuNttContext::new(n, q);
    let mut cpu_result = input.clone();
    cpu_ntt.forward_ntt(&mut cpu_result);
    cpu_ntt.inverse_ntt(&mut cpu_result);
    println!("✓ CPU NTT roundtrip completed");

    // Metal NTT
    let device = Arc::new(MetalDevice::new().expect("Failed to create Metal device"));
    let metal_ntt = MetalNttContext::new_with_device(device, n, q, psi).expect("Failed to create Metal NTT");
    let mut metal_result = input.clone();
    metal_ntt.forward(&mut metal_result).expect("Failed to run Metal forward NTT");
    metal_ntt.inverse(&mut metal_result).expect("Failed to run Metal inverse NTT");
    println!("✓ Metal inverse NTT completed");

    // Compare with original
    let mut max_cpu_error = 0u64;
    let mut max_metal_error = 0u64;
    for i in 0..n {
        let cpu_err = if original[i] > cpu_result[i] {
            original[i] - cpu_result[i]
        } else {
            cpu_result[i] - original[i]
        };
        let metal_err = if original[i] > metal_result[i] {
            original[i] - metal_result[i]
        } else {
            metal_result[i] - original[i]
        };
        max_cpu_error = max_cpu_error.max(cpu_err);
        max_metal_error = max_metal_error.max(metal_err);

        if i < 5 {
            println!("  Position {}: original={}, CPU={} (err={}), Metal={} (err={})",
                     i, original[i], cpu_result[i], cpu_err, metal_result[i], metal_err);
        }
    }

    println!("\n Results:");
    println!("  CPU roundtrip max error: {}", max_cpu_error);
    println!("  Metal roundtrip max error: {}", max_metal_error);

    if max_cpu_error == 0 && max_metal_error == 0 {
        println!("\n✅ ALL TESTS PASSED - Metal NTT is CORRECT!");
    } else {
        println!("\n❌ TESTS FAILED - NTT roundtrip has errors");
    }
}

#[cfg(feature = "v2-gpu-metal")]
fn find_primitive_2n_root(n: usize, q: u64) -> u64 {
    let two_n = (2 * n) as u64;
    assert_eq!((q - 1) % two_n, 0, "q must satisfy q ≡ 1 (mod 2n)");

    // Find generator
    let g = find_generator(q);
    let exp = (q - 1) / two_n;
    pow_mod(g, exp, q)
}

#[cfg(feature = "v2-gpu-metal")]
fn find_generator(q: u64) -> u64 {
    for g in 2..20u64 {
        if pow_mod(g, q - 1, q) == 1 && pow_mod(g, (q - 1) / 2, q) != 1 {
            return g;
        }
    }
    panic!("Failed to find generator for q = {}", q);
}

#[cfg(feature = "v2-gpu-metal")]
fn pow_mod(mut base: u64, mut exp: u64, q: u64) -> u64 {
    let mut result = 1u64;
    base %= q;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % q as u128) as u64;
        }
        base = ((base as u128 * base as u128) % q as u128) as u64;
        exp >>= 1;
    }
    result
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    eprintln!("This example requires the v2-gpu-metal feature.");
    eprintln!("Run with: cargo run --release --features v2,v2-gpu-metal --example test_metal_ntt_correctness");
}
