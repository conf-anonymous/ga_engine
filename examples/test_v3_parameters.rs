//! Test V3 Parameter Sets
//!
//! Verifies V3 parameter sets with 20+ primes for bootstrapping.

#[cfg(feature = "v3")]
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

fn main() {
    #[cfg(not(feature = "v3"))]
    {
        println!("V3 feature not enabled. Run with: cargo run --example test_v3_parameters --features v3");
        return;
    }

    #[cfg(feature = "v3")]
    {
        println!("=== V3 Parameter Sets Test ===\n");

        // Test 1: V3 Bootstrap 8192
        println!("Test 1: V3 Bootstrap Parameters (N=8192, 22 primes)");
        test_v3_bootstrap_8192();
        println!("  ✓ V3 Bootstrap 8192 parameters valid\n");

        // Test 2: V3 Bootstrap 16384
        println!("Test 2: V3 Bootstrap Parameters (N=16384, 25 primes)");
        test_v3_bootstrap_16384();
        println!("  ✓ V3 Bootstrap 16384 parameters valid\n");

        // Test 3: V3 Minimal
        println!("Test 3: V3 Minimal Bootstrap Parameters (N=8192, 20 primes)");
        test_v3_bootstrap_minimal();
        println!("  ✓ V3 Minimal bootstrap parameters valid\n");

        // Test 4: Computation Levels
        println!("Test 4: Computation Levels Analysis");
        test_computation_levels();
        println!("  ✓ Computation levels calculated correctly\n");

        println!("=== V3 Parameter Sets Test Complete ===");
        println!("All parameter sets validated successfully!");
    }
}

#[cfg(feature = "v3")]
fn test_v3_bootstrap_8192() {
    let params = CliffordFHEParams::new_v3_bootstrap_8192();

    println!("  N = {}", params.n);
    println!("  Primes = {}", params.moduli.len());
    println!("  Scale = 2^40");
    println!("  Security = {:?}", params.security);

    assert_eq!(params.n, 8192);
    assert_eq!(params.moduli.len(), 22);

    // Verify all primes are NTT-friendly (q ≡ 1 mod 2N)
    let two_n = 2 * params.n as u64;
    for (i, &q) in params.moduli.iter().enumerate() {
        assert_eq!((q - 1) % two_n, 0,
                   "Prime {} is not NTT-friendly: {} mod {} != 0", i, q, two_n);
    }
    println!("  ✓ All 22 primes are NTT-friendly (q ≡ 1 mod {})", two_n);

    // Check prime sizes
    let first_bits = (params.moduli[0] as f64).log2() as usize;
    println!("  First prime: {} bits", first_bits);
    assert!(first_bits >= 59 && first_bits <= 61);

    let second_bits = (params.moduli[1] as f64).log2() as usize;
    println!("  Scaling primes: ~{} bits", second_bits);

    // Check bootstrap support
    assert!(params.supports_bootstrap(12));
    println!("  ✓ Supports bootstrap with 12 levels");

    let comp_levels = params.computation_levels(12);
    println!("  Computation levels (12 bootstrap): {}", comp_levels);
    assert_eq!(comp_levels, 9);
}

#[cfg(feature = "v3")]
fn test_v3_bootstrap_16384() {
    let params = CliffordFHEParams::new_v3_bootstrap_16384();

    println!("  N = {}", params.n);
    println!("  Primes = {}", params.moduli.len());
    println!("  Security = {:?}", params.security);

    assert_eq!(params.n, 16384);
    assert_eq!(params.moduli.len(), 25);

    // Verify NTT-friendly
    let two_n = 2 * params.n as u64;
    for (i, &q) in params.moduli.iter().enumerate() {
        assert_eq!((q - 1) % two_n, 0, "Prime {} not NTT-friendly", i);
    }
    println!("  ✓ All 25 primes are NTT-friendly (q ≡ 1 mod {})", two_n);

    // Check bootstrap support with 15 levels
    assert!(params.supports_bootstrap(15));
    println!("  ✓ Supports bootstrap with 15 levels");

    let comp_levels = params.computation_levels(15);
    println!("  Computation levels (15 bootstrap): {}", comp_levels);
    assert_eq!(comp_levels, 9);
}

#[cfg(feature = "v3")]
fn test_v3_bootstrap_minimal() {
    let params = CliffordFHEParams::new_v3_bootstrap_minimal();

    println!("  N = {}", params.n);
    println!("  Primes = {} (minimal)", params.moduli.len());

    assert_eq!(params.n, 8192);
    assert_eq!(params.moduli.len(), 20);

    // Should support bootstrap with 12 levels
    assert!(params.supports_bootstrap(12));
    println!("  ✓ Supports bootstrap with 12 levels");

    // Should have 7 computation levels (20 - 12 - 1)
    let comp_levels = params.computation_levels(12);
    assert_eq!(comp_levels, 7);
    println!("  Computation levels (12 bootstrap): {}", comp_levels);
}

#[cfg(feature = "v3")]
fn test_computation_levels() {
    let params = CliffordFHEParams::new_v3_bootstrap_8192();

    println!("\n  Parameter Set: V3 Bootstrap 8192 (22 primes)");
    println!("  -----------------------------------------------");

    for bootstrap_levels in [10, 12, 15, 18] {
        let comp_levels = params.computation_levels(bootstrap_levels);
        let supports = params.supports_bootstrap(bootstrap_levels);

        println!("  Bootstrap levels: {:2} → Computation levels: {:2} → Support: {}",
                 bootstrap_levels, comp_levels,
                 if supports { "✓" } else { "✗" });

        // Verify formula: comp_levels = total - bootstrap - 1 (special prime)
        let expected = params.moduli.len().saturating_sub(bootstrap_levels + 1);
        assert_eq!(comp_levels, expected);
    }

    println!("\n  Analysis:");
    println!("  - With balanced preset (12 levels): {} multiplications between bootstraps",
             params.computation_levels(12));
    println!("  - With conservative preset (15 levels): {} multiplications between bootstraps",
             params.computation_levels(15));
    println!("  - Formula: computation_levels = total_primes - bootstrap_levels - 1");
}
