//! NTT Round-Trip Test
//!
//! Tests that GPU NTT forward→inverse returns the original input.
//! This is the most basic correctness test for NTT implementation.
//!
//! **Run:**
//! ```bash
//! cargo run --release --features v2,v2-gpu-metal,v3 --example test_ntt_roundtrip
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-metal"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::ntt::MetalNttContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::device::MetalDevice;
    use std::sync::Arc;

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                         NTT Round-Trip Test                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let n = 1024;
    let q = 1152921504606584833u64;  // 60-bit NTT-friendly prime

    println!("Parameters:");
    println!("  N = {}", n);
    println!("  q = {} (60-bit NTT-friendly prime)\n", q);

    // Find primitive root
    let psi = find_primitive_2n_root(n, q)?;
    println!("  ψ (primitive 2N-th root) = {}\n", psi);

    // Create Metal NTT context
    let device = Arc::new(MetalDevice::new()?);
    let ntt = MetalNttContext::new_with_device(device.clone(), n, q, psi)?;
    println!("✅ Metal NTT context created\n");

    // Test Case 1: Simple spike at position 0
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Test 1: Spike at position 0");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let mut data1 = vec![0u64; n];
    data1[0] = 1;
    let input1 = data1.clone();

    println!("Input: [1, 0, 0, ..., 0]");

    ntt.forward(&mut data1)?;
    println!("✓ Forward NTT complete");

    ntt.inverse(&mut data1)?;
    println!("✓ Inverse NTT complete");

    print!("Output (first 10): [");
    for i in 0..10 {
        print!("{}", data1[i]);
        if i < 9 { print!(", "); }
    }
    println!("]");

    let match1 = data1 == input1;
    println!("\nResult: {}", if match1 { "✅ PASS - Exact match!" } else { "❌ FAIL - Mismatch!" });

    if !match1 {
        println!("\nDifferences:");
        for i in 0..n.min(20) {
            if data1[i] != input1[i] {
                println!("  Position {}: expected {}, got {}", i, input1[i], data1[i]);
            }
        }
    }

    // Test Case 2: Pattern [1, 2, 3, 4, ...]
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("Test 2: Sequential pattern");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let mut data2 = vec![0u64; n];
    for i in 0..10 {
        data2[i] = (i + 1) as u64;
    }
    let input2 = data2.clone();

    print!("Input (first 10): [");
    for i in 0..10 {
        print!("{}", input2[i]);
        if i < 9 { print!(", "); }
    }
    println!("]");

    ntt.forward(&mut data2)?;
    println!("✓ Forward NTT complete");

    ntt.inverse(&mut data2)?;
    println!("✓ Inverse NTT complete");

    print!("Output (first 10): [");
    for i in 0..10 {
        print!("{}", data2[i]);
        if i < 9 { print!(", "); }
    }
    println!("]");

    let match2 = data2 == input2;
    println!("\nResult: {}", if match2 { "✅ PASS - Exact match!" } else { "❌ FAIL - Mismatch!" });

    if !match2 {
        println!("\nDifferences:");
        let mut num_diffs = 0;
        for i in 0..n {
            if data2[i] != input2[i] {
                if num_diffs < 20 {
                    println!("  Position {}: expected {}, got {}", i, input2[i], data2[i]);
                }
                num_diffs += 1;
            }
        }
        if num_diffs > 20 {
            println!("  ... and {} more differences", num_diffs - 20);
        }
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("Summary");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    if match1 && match2 {
        println!("✅ ALL TESTS PASSED - NTT round-trip working correctly!");
        Ok(())
    } else {
        println!("❌ TESTS FAILED - NTT has bugs");
        println!("\nPossible causes:");
        println!("  1. Incorrect Montgomery domain conversion");
        println!("  2. Wrong twiddle factor computation");
        println!("  3. Bit-reversal permutation error");
        println!("  4. Incorrect n^{{-1}} scaling");
        Err("NTT round-trip test failed".to_string())
    }
}

#[cfg(all(feature = "v2", feature = "v2-gpu-metal"))]
fn find_primitive_2n_root(n: usize, q: u64) -> Result<u64, String> {
    let two_n = (2 * n) as u64;
    if (q - 1) % two_n != 0 {
        return Err(format!("q = {} is not NTT-friendly for n = {}", q, n));
    }

    let exp = (q - 1) / two_n;

    for g in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        let psi = pow_mod(g % q, exp, q);
        if psi != 1
            && pow_mod(psi, two_n, q) == 1
            && pow_mod(psi, n as u64, q) != 1
        {
            return Ok(psi);
        }
    }

    Err(format!("Could not find primitive 2N-th root for n = {}, q = {}", n, q))
}

#[cfg(all(feature = "v2", feature = "v2-gpu-metal"))]
fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exp /= 2;
    }
    result
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-metal")))]
fn main() {
    println!("This example requires features: v2, v2-gpu-metal");
    println!("Run with: cargo run --release --features v2,v2-gpu-metal,v3 --example test_ntt_roundtrip");
}
