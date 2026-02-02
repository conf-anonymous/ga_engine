//! Debug Galois Map Computation
//!
//! Tests that the Galois automorphism map is computed correctly for basic rotations.

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn main() {
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::rotation::{
        compute_galois_map, rotation_step_to_galois_element
    };

    println!("Testing Galois Map Computation for N=1024");
    println!("=========================================\n");

    let n = 1024;

    // Test rotation by +1
    println!("Test 1: Rotation by +1");
    let k1 = rotation_step_to_galois_element(1, n);
    println!("  Galois element k = {}", k1);
    println!("  Expected: k = 5 (since 5^1 mod 2048 = 5)");

    let (map1, signs1) = compute_galois_map(n, k1);

    println!("\n  First 10 coefficient mappings:");
    for i in 0..10 {
        println!("    coeff[{}] → coeff[{}] (sign={})", i, map1[i], signs1[i]);
    }

    println!("\n  Expected pattern for rotation by +1:");
    println!("    In CKKS encoding:");
    println!("    - Odd coefficients → slots");
    println!("    - σ_5(X^i) = X^(5i mod 2N)");
    println!("    - This implements a left rotation of slots");

    // Test rotation by -1
    println!("\n\nTest 2: Rotation by -1");
    let k_neg1 = rotation_step_to_galois_element(-1, n);
    println!("  Galois element k = {}", k_neg1);
    println!("  Expected: k = 5^(N-1) mod 2N = 5^1023 mod 2048");

    let (map_neg1, signs_neg1) = compute_galois_map(n, k_neg1);

    println!("\n  First 10 coefficient mappings:");
    for i in 0..10 {
        println!("    coeff[{}] → coeff[{}] (sign={})", i, map_neg1[i], signs_neg1[i]);
    }

    // Verify that k and k_inv are inverses
    println!("\n\nTest 3: Verify k=5 and k_inv=5^1023 are inverses");
    let product = (k1 * k_neg1) % (2 * n);
    println!("  k × k_inv mod 2N = {} × {} mod 2048 = {}", k1, k_neg1, product);
    println!("  Expected: 1 (they should be inverses)");

    if product == 1 {
        println!("  ✅ k and k_inv are inverses!");
    } else {
        println!("  ❌ ERROR: k and k_inv are NOT inverses!");
    }

    // Check rotation by +2
    println!("\n\nTest 4: Rotation by +2");
    let k2 = rotation_step_to_galois_element(2, n);
    println!("  Galois element k = {}", k2);
    println!("  Expected: k = 5^2 mod 2048 = 25");

    if k2 == 25 {
        println!("  ✅ Correct!");
    } else {
        println!("  ❌ Wrong! Got {} but expected 25", k2);
    }
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3")))]
fn main() {
    eprintln!("This example requires features: v2,v2-gpu-metal,v3");
    std::process::exit(1);
}
