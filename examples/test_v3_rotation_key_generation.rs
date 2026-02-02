//! Test V3 Rotation Key Generation (Phase 3)
//!
//! Validates that rotation key generation works correctly with proper structure.

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v3::bootstrapping::keys::{
    generate_rotation_keys,
    galois_element_for_rotation,
    required_rotations_for_bootstrap,
};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║          V3 Rotation Key Generation Test (Phase 3)            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Test 1: Small rotation set with key structure validation
    test_small_rotation_set();
    println!();

    // Test 2: Full bootstrap rotation set
    test_full_bootstrap_rotations();
    println!();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              ✓ ALL ROTATION KEY TESTS PASSED ✓                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
}

fn test_small_rotation_set() {
    println!("Test 1: Small Rotation Set with Key Structure Validation");
    println!("──────────────────────────────────────────────────────────────");

    // Use smaller params for faster testing
    let params = CliffordFHEParams::new_128bit();
    println!("  Parameters: N={}, {} primes", params.n, params.moduli.len());

    // Generate keys
    let key_ctx = KeyContext::new(params.clone());
    println!("  Generating secret key...");
    let (_, secret_key, _) = key_ctx.keygen();

    // Generate rotation keys for small set
    let rotations = vec![1, 2, 4];
    println!("  Generating rotation keys for {:?}...", rotations);
    let rotation_keys = generate_rotation_keys(&rotations, &secret_key, &params);

    // Verify number of keys
    assert_eq!(rotation_keys.num_keys(), 3, "Should have 3 rotation keys");
    println!("  ✓ Generated {} rotation keys", rotation_keys.num_keys());

    // Verify each rotation key structure
    for &k in &rotations {
        let g = galois_element_for_rotation(k, params.n);
        assert!(rotation_keys.has_key(g), "Missing key for rotation {}", k);

        let rot_key = rotation_keys.get_key(g).unwrap();

        // Verify galois element
        assert_eq!(rot_key.galois_element, g, "Galois element mismatch");

        // Verify base_w
        assert_eq!(rot_key.base_w, 20, "base_w should be 20");

        // Verify rlk0 and rlk1 have digits
        assert!(rot_key.rlk0.len() > 0, "rlk0 should have digits");
        assert_eq!(rot_key.rlk0.len(), rot_key.rlk1.len(),
                   "rlk0 and rlk1 should have same number of digits");

        // Verify each digit has N coefficients
        for digit in &rot_key.rlk0 {
            assert_eq!(digit.len(), params.n,
                       "Each rlk0 digit should have N coefficients");
        }
        for digit in &rot_key.rlk1 {
            assert_eq!(digit.len(), params.n,
                       "Each rlk1 digit should have N coefficients");
        }

        println!("  ✓ Rotation {}: g={}, {} digits, {} coeffs per digit",
                 k, g, rot_key.rlk0.len(), params.n);
    }

    println!("  ✓ All rotation key structures valid");
}

fn test_full_bootstrap_rotations() {
    println!("Test 2: Full Bootstrap Rotation Set (N=1024)");
    println!("──────────────────────────────────────────────────────────────");

    // Use smaller N for faster testing
    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("  Parameters: N={}, {} primes", params.n, params.moduli.len());

    // Get required rotations for bootstrap
    let rotations = required_rotations_for_bootstrap(params.n);
    let expected_count = 2 * (params.n as f64).log2() as usize;
    println!("  Required rotations: {}", rotations.len());
    assert_eq!(rotations.len(), expected_count,
               "Should have 2*log2(N) rotations for bootstrap");

    // Compute number of unique Galois elements (some rotations may map to same g)
    use std::collections::HashSet;
    use ga_engine::clifford_fhe_v3::bootstrapping::keys::galois_element_for_rotation;
    let unique_galois: HashSet<_> = rotations.iter()
        .map(|&k| galois_element_for_rotation(k, params.n))
        .collect();
    println!("  Unique Galois elements: {}", unique_galois.len());

    // Verify rotations are powers of 2 (both positive and negative)
    for &k in &rotations {
        let abs_k = k.abs() as u32;
        assert!(abs_k.is_power_of_two(),
                "Rotation {} should be power of 2", k);
    }
    println!("  ✓ All rotations are powers of 2");

    // Generate keys
    let key_ctx = KeyContext::new(params.clone());
    println!("  Generating secret key...");
    let (_, secret_key, _) = key_ctx.keygen();

    // Generate rotation keys
    println!("  Generating {} rotation keys...", rotations.len());
    let start = std::time::Instant::now();
    let rotation_keys = generate_rotation_keys(&rotations, &secret_key, &params);
    let elapsed = start.elapsed();

    // Verify all unique keys generated (may be fewer than rotations due to duplicates)
    assert_eq!(rotation_keys.num_keys(), unique_galois.len(),
               "Should have key for each unique Galois element");
    println!("  ✓ Generated {} unique rotation keys in {:.2}s",
             rotation_keys.num_keys(), elapsed.as_secs_f64());

    // Verify specific rotations exist
    let test_rotations = vec![1, -1, 2, -2, 512, -512];
    for &k in &test_rotations {
        let g = galois_element_for_rotation(k, params.n);
        assert!(rotation_keys.has_key(g), "Missing key for rotation {}", k);
    }
    println!("  ✓ All test rotations present: {:?}", test_rotations);

    // Performance metric
    let keys_per_sec = rotation_keys.num_keys() as f64 / elapsed.as_secs_f64();
    println!("  Performance: {:.1} keys/second", keys_per_sec);
}
