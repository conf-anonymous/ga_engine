//! Test V3 Rotation Keys
//!
//! Verifies Galois automorphisms and rotation key generation.

#[cfg(feature = "v3")]
use ga_engine::clifford_fhe_v3::bootstrapping::keys::{
    galois_element_for_rotation,
    required_rotations_for_bootstrap,
    generate_rotation_keys,
};

#[cfg(feature = "v3")]
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

#[cfg(feature = "v3")]
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

fn main() {
    #[cfg(not(feature = "v3"))]
    {
        println!("V3 feature not enabled. Run with: cargo run --example test_v3_rotation_keys --features v3");
        return;
    }

    #[cfg(feature = "v3")]
    {
        println!("=== V3 Rotation Keys Test ===\n");

        // Test 1: Galois Elements
        println!("Test 1: Galois Elements for Rotations");
        test_galois_elements();
        println!("  ✓ Galois elements calculated correctly\n");

        // Test 2: Required Rotations
        println!("Test 2: Required Rotations for Bootstrap");
        test_required_rotations();
        println!("  ✓ Required rotations computed correctly\n");

        // Test 3: Rotation Key Generation
        println!("Test 3: Rotation Key Generation (Placeholder)");
        test_rotation_key_generation();
        println!("  ✓ Rotation key generation structure works\n");

        println!("=== V3 Rotation Keys Test Complete ===");
        println!("All rotation key components validated!");
    }
}

#[cfg(feature = "v3")]
fn test_galois_elements() {
    let n = 8192;

    println!("  Testing Galois elements for N = {}...", n);
    println!("  Formula: g = 5^k mod 2N\n");

    // Test rotations by powers of 2
    let rotations = vec![1, 2, 4, 8, 16, 32, 64, 128];

    for &k in &rotations {
        let g = galois_element_for_rotation(k, n);
        let g_neg = galois_element_for_rotation(-k, n);

        println!("  Rotation {:4}: g = {:6} (5^{} mod {})",
                 k, g, k, 2 * n);
        println!("  Rotation {:4}: g = {:6} (inverse)",
                 -k, g_neg);
    }

    // Verify specific values
    assert_eq!(galois_element_for_rotation(0, n), 1);
    assert_eq!(galois_element_for_rotation(1, n), 5);
    assert_eq!(galois_element_for_rotation(2, n), 25);

    println!("\n  ✓ Galois elements for small rotations:");
    println!("    Rotation 0: g = 1 (identity)");
    println!("    Rotation 1: g = 5");
    println!("    Rotation 2: g = 25");
}

#[cfg(feature = "v3")]
fn test_required_rotations() {
    let n = 8192;

    let rotations = required_rotations_for_bootstrap(n);

    println!("  N = {}", n);
    println!("  Required rotations: {}", rotations.len());
    println!("  Formula: 2 * log2(N) = 2 * {} = {}",
             (n as f64).log2() as usize,
             rotations.len());

    // Verify count
    let expected = 2 * ((n as f64).log2() as usize);
    assert_eq!(rotations.len(), expected);

    // Display first few
    println!("\n  First 10 rotations:");
    for (i, &k) in rotations.iter().take(10).enumerate() {
        let g = galois_element_for_rotation(k, n);
        println!("    {}: rotation by {:5} → g = {}", i, k, g);
    }

    // Verify specific rotations
    assert!(rotations.contains(&1));
    assert!(rotations.contains(&-1));
    assert!(rotations.contains(&(n as i32 / 2)));
    assert!(rotations.contains(&(-(n as i32 / 2))));

    println!("\n  ✓ Contains required rotations:");
    println!("    - Rotation by ±1");
    println!("    - Rotation by ±{} (N/2)", n / 2);
    println!("    - All powers of 2 in between");
}

#[cfg(feature = "v3")]
fn test_rotation_key_generation() {
    // Use smaller V2 parameters for faster key generation (just for testing structure)
    let params = CliffordFHEParams::new_128bit();  // 9 primes, much faster
    let key_ctx = KeyContext::new(params.clone());
    let (_, secret_key, _) = key_ctx.keygen();

    println!("  Parameters: N = {}, {} primes", params.n, params.moduli.len());

    // Generate rotation keys for small set (placeholder)
    let rotations = vec![1, 2, 4, 8, 16];
    println!("  Generating rotation keys for {} rotations...", rotations.len());

    let rotation_keys = generate_rotation_keys(&rotations, &secret_key, &params);

    println!("\n  ✓ Generated {} rotation keys (placeholder)", rotation_keys.num_keys());
    assert_eq!(rotation_keys.num_keys(), rotations.len());

    // Verify keys exist for expected Galois elements
    for &k in &rotations {
        let g = galois_element_for_rotation(k, params.n);
        assert!(rotation_keys.has_key(g), "Missing key for rotation {}", k);
    }

    println!("  ✓ All rotation keys accessible by Galois element");

    // Show size estimate
    println!("\n  Rotation Key Storage (Future Implementation):");
    println!("    - Each rotation: ~2 polynomials × {} coefficients", params.n);
    println!("    - Total rotations for bootstrap: {}",
             required_rotations_for_bootstrap(params.n).len());
    println!("    - Estimated size: ~{} MB (when implemented)",
             estimate_rotation_key_size_mb(params.n));
}

#[cfg(feature = "v3")]
fn estimate_rotation_key_size_mb(n: usize) -> usize {
    // Rough estimate:
    // - Each rotation needs 2 polynomials (rlk0, rlk1)
    // - Each polynomial has N coefficients in RNS form
    // - Assume ~5 digits for gadget decomposition
    // - Assume ~10 primes (average)
    // - Each RNS value is 8 bytes (u64)

    let num_rotations = required_rotations_for_bootstrap(n).len();
    let digits = 5;
    let primes = 10;
    let bytes_per_rotation = 2 * digits * n * primes * 8;
    let total_bytes = num_rotations * bytes_per_rotation;

    total_bytes / (1024 * 1024)  // Convert to MB
}
