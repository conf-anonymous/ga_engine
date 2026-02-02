//! Test NTT normalization - hunt for 512× bug
//!
//! Expert diagnosed: "exactly 2^-9 with N=1024 screams per-stage halving"
//! We've verified butterflies are clean, so let's trace actual values.

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::backends::gpu_metal::ntt::MetalNttContext;

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This test requires Metal GPU support.");
    println!("Run with: cargo run --release --features v2,v3,v2-gpu-metal --example test_ntt_normalization");
}

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              NTT Normalization Debug Test                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let q = params.moduli[0];

    println!("Parameters:");
    println!("  N = {} (2^{})", n, (n as f64).log2());
    println!("  q = {}\n", q);

    // Find primitive 2n-th root
    let psi = find_primitive_2n_root(n, q)?;
    println!("Primitive 2n-th root: ψ = {}\n", psi);

    // Create Metal NTT context
    let ctx = MetalNttContext::new(n, q, psi)?;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: Simple polynomial [42, 0, 0, ..., 0]");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut poly = vec![0u64; n];
    poly[0] = 42;

    println!("Input polynomial:");
    println!("  poly[0] = {}", poly[0]);
    println!("  poly[1..] = 0\n");

    let original = poly.clone();

    // Forward NTT
    ctx.forward(&mut poly)?;
    println!("After forward NTT:");
    println!("  poly[0] = {}", poly[0]);
    println!("  poly[1] = {}", poly[1]);
    println!("  poly[2] = {}", poly[2]);
    println!("  (all should be 42 in NTT domain for constant poly)\n");

    // Pointwise multiply by itself (should be 42 * 42 = 1764)
    let mut poly2 = poly.clone();
    let mut result = vec![0u64; n];
    ctx.pointwise_multiply(&poly, &poly2, &mut result)?;

    println!("After pointwise multiply (42 * 42):");
    println!("  result[0] = {}", result[0]);
    println!("  result[1] = {}", result[1]);
    println!("  Expected: ~1764 in each position\n");

    // Inverse NTT
    ctx.inverse(&mut result)?;

    println!("After inverse NTT:");
    println!("  result[0] = {}", result[0]);
    println!("  result[1] = {}", result[1]);
    println!("  result[2..] = ...\n");

    let expected = (42u64 * 42u64) % q;
    let got = result[0];
    let error_factor = if got != 0 {
        expected as f64 / got as f64
    } else {
        f64::INFINITY
    };

    println!("Expected: poly[0] = {} (42 × 42)", expected);
    println!("Got:      poly[0] = {}", got);
    println!("Error factor: {:.2}×", error_factor);

    if (error_factor - 512.0).abs() < 0.1 {
        println!("\n❌ BUG CONFIRMED: Exactly 512× error!");
        println!("   This confirms per-stage halving or wrong n_inv by factor of 2^9\n");
    } else if (error_factor - 1.0).abs() < 0.01 {
        println!("\n✅ SUCCESS: No normalization bug detected!\n");
    } else {
        println!("\n⚠️  UNEXPECTED: Error factor is {:.2}×, not 512×\n", error_factor);
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: Check n_inv value");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // The n_inv should be n^{-1} mod q
    let n_inv = mod_inverse(n as u64, q)?;
    println!("Computed n_inv = {}", n_inv);
    // Use u128 to avoid overflow
    let verify = ((n as u128 * n_inv as u128) % q as u128) as u64;
    println!("Verify: (n * n_inv) % q = {}", verify);
    println!("Should be: 1");
    if verify == 1 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL - n_inv is WRONG!\n");
    }

    // What if we used 2n instead?
    let two_n_inv = mod_inverse(2 * n as u64, q)?;
    println!("If we accidentally used (2n)^{{-1}}:");
    println!("  two_n_inv = {}", two_n_inv);
    println!("  Ratio: n_inv / two_n_inv = {:.2}", n_inv as f64 / two_n_inv as f64);
    println!("  (should be 2.0 if that's the bug)\n");

    // What about n/2?
    let half_n_inv = mod_inverse((n / 2) as u64, q)?;
    println!("If we accidentally used (n/2)^{{-1}}:");
    println!("  half_n_inv = {}", half_n_inv);
    println!("  Ratio: half_n_inv / n_inv = {:.2}", half_n_inv as f64 / n_inv as f64);
    println!("  (should be 2.0 if that's related)\n");

    Ok(())
}

#[cfg(feature = "v2-gpu-metal")]
fn find_primitive_2n_root(n: usize, q: u64) -> Result<u64, String> {
    let two_n = (2 * n) as u64;
    if (q - 1) % two_n != 0 {
        return Err(format!("q is not NTT-friendly for n = {}", n));
    }

    for candidate in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        if is_primitive_root_candidate(candidate, n, q) {
            let exponent = (q - 1) / two_n;
            return Ok(pow_mod(candidate, exponent, q));
        }
    }

    for candidate in 32..20000u64 {
        if is_primitive_root_candidate(candidate, n, q) {
            let exponent = (q - 1) / two_n;
            return Ok(pow_mod(candidate, exponent, q));
        }
    }

    Err(format!("Failed to find primitive root for q = {}", q))
}

#[cfg(feature = "v2-gpu-metal")]
fn is_primitive_root_candidate(g: u64, n: usize, q: u64) -> bool {
    // Just check if g^((q-1)/2) ≠ 1 to verify it's not a quadratic residue
    // This is a simplified check - actual verification happens in MetalNttContext
    let exp = (q - 1) / 2;
    pow_mod(g, exp, q) != 1
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

#[cfg(feature = "v2-gpu-metal")]
fn mod_inverse(a: u64, q: u64) -> Result<u64, String> {
    let (mut old_r, mut r) = (a as i128, q as i128);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let quotient = old_r / r;
        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;
    }

    if old_r != 1 {
        return Err(format!("{} has no modular inverse mod {}", a, q));
    }

    let result = if old_s < 0 {
        (old_s + q as i128) as u64
    } else {
        old_s as u64
    };

    Ok(result)
}
