//! Integration tests for stable BKZ
//!
//! Tests the numerically-stable BKZ implementation based on expert guidance:
//! - Stable GSO (MGS + re-orth + Kahan)
//! - GH-based radius
//! - Correct enumeration bounds
//! - Strict size-reduction

use ga_engine::lattice_reduction::bkz_stable::StableBKZ;
use ga_engine::lattice_reduction::lll_baseline::LLL;

fn main() {
    println!("=== Stable BKZ Integration Tests ===\n");

    // Test 1: 5D well-conditioned
    test_5d_well_conditioned();

    // Test 2: 10D diagonal + perturbations
    test_10d_diagonal();

    // Test 3: 10D with moderate entries
    test_10d_moderate();

    // Test 4: 20D (stress test)
    test_20d_diagonal();

    println!("\n=== All Tests Complete ===");
}

fn test_5d_well_conditioned() {
    println!("Test 1: 5D Well-Conditioned Lattice");
    println!("------------------------------------");

    let basis = vec![
        vec![50.0, 10.0, 2.0, 1.0, 0.5],
        vec![10.0, 50.0, 8.0, 2.0, 1.0],
        vec![2.0, 8.0, 50.0, 5.0, 2.0],
        vec![1.0, 2.0, 5.0, 50.0, 3.0],
        vec![0.5, 1.0, 2.0, 3.0, 50.0],
    ];

    let input_first_norm = norm(&basis[0]);
    println!("Input first norm: {:.6}", input_first_norm);

    let start = std::time::Instant::now();
    let mut bkz = StableBKZ::new(basis, 5, 0.99);
    bkz.reduce_with_limit(5);
    let elapsed = start.elapsed();

    let stats = bkz.get_stats();
    let output_first_norm = norm(&bkz.get_basis()[0]);
    let hf = bkz.hermite_factor();

    println!("Output first norm: {:.6}", output_first_norm);
    println!("Time: {:?}", elapsed);
    println!("Tours: {}", stats.tours);
    println!("Improvements: {}", stats.improvements);
    println!("Enum calls: {}", stats.enum_calls);
    println!("Nodes explored: {}", stats.enum_nodes);
    println!("Timeouts: {}", stats.enum_timeouts);
    println!("Precision escalations: {}", stats.precision_escalations);
    println!("Hermite factor: {:.6}", hf);

    if stats.improvements > 0 {
        println!("✓ Found improvements");
    }

    if elapsed.as_secs() < 5 {
        println!("✓ Completed quickly");
    }

    println!();
}

fn test_10d_diagonal() {
    println!("Test 2: 10D Diagonal + Perturbations");
    println!("-------------------------------------");

    let mut basis = Vec::new();
    for i in 0..10 {
        let mut v = vec![0.0; 10];
        v[i] = 30.0 + (i as f64 * 2.0);
        if i > 0 {
            v[i - 1] = 5.0;
        }
        if i + 1 < 10 {
            v[i + 1] = 3.0;
        }
        basis.push(v);
    }

    // Compare LLL vs BKZ
    let mut lll = LLL::new(basis.clone(), 0.99);
    let lll_start = std::time::Instant::now();
    lll.reduce();
    let lll_time = lll_start.elapsed();
    let lll_hf = lll.hermite_factor();

    let mut bkz = StableBKZ::new(basis, 10, 0.99);
    let bkz_start = std::time::Instant::now();
    bkz.reduce_with_limit(3);
    let bkz_time = bkz_start.elapsed();
    let bkz_hf = bkz.hermite_factor();

    println!("LLL: {:?}, HF = {:.6}", lll_time, lll_hf);
    println!("BKZ: {:?}, HF = {:.6}", bkz_time, bkz_hf);
    println!("BKZ stats: {:?}", bkz.get_stats());

    if bkz_hf <= lll_hf {
        let improvement = (1.0 - bkz_hf / lll_hf) * 100.0;
        println!("✓ BKZ improved by {:.2}%", improvement);
    }

    if bkz_time.as_secs() < 30 {
        println!("✓ Completed in reasonable time");
    }

    println!();
}

fn test_10d_moderate() {
    println!("Test 3: 10D with Moderate Random Entries");
    println!("-----------------------------------------");

    // Generate moderate random-ish lattice
    let mut basis = Vec::new();
    for i in 0..10 {
        let mut v = Vec::new();
        for j in 0..10 {
            let val = ((i * 37 + j * 17) % 200) as f64 - 100.0;
            v.push(val);
        }
        basis.push(v);
    }

    let start = std::time::Instant::now();
    let mut bkz = StableBKZ::new(basis, 10, 0.99);
    bkz.reduce_with_limit(3);
    let elapsed = start.elapsed();

    let stats = bkz.get_stats();
    let hf = bkz.hermite_factor();

    println!("Time: {:?}", elapsed);
    println!("Hermite factor: {:.6}", hf);
    println!("Tours: {}", stats.tours);
    println!("Improvements: {}", stats.improvements);
    println!("Precision escalations: {}", stats.precision_escalations);

    if elapsed.as_secs() < 60 {
        println!("✓ No timeout");
    }

    if hf < 1.8 {
        println!("✓ Reasonable quality");
    }

    println!();
}

fn test_20d_diagonal() {
    println!("Test 4: 20D Diagonal (Stress Test)");
    println!("-----------------------------------");

    let mut basis = Vec::new();
    for i in 0..20 {
        let mut v = vec![0.0; 20];
        v[i] = 40.0 + (i as f64);
        if i > 0 {
            v[i - 1] = 3.0;
        }
        if i + 1 < 20 {
            v[i + 1] = 2.0;
        }
        basis.push(v);
    }

    let start = std::time::Instant::now();
    let mut bkz = StableBKZ::new(basis, 20, 0.99);
    bkz.reduce_with_limit(2); // Just 2 tours for 20D
    let elapsed = start.elapsed();

    let stats = bkz.get_stats();
    let hf = bkz.hermite_factor();

    println!("Time: {:?}", elapsed);
    println!("Hermite factor: {:.6}", hf);
    println!("Tours: {}", stats.tours);
    println!("Improvements: {}", stats.improvements);
    println!("Precision escalations: {}", stats.precision_escalations);

    if elapsed.as_secs() < 120 {
        println!("✓ Completed within 2 minutes");
    } else {
        println!("⚠ Took longer than expected");
    }

    println!();
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
