//! Quick integration tests for stable BKZ (skips slow tests)
//!
//! Tests just the fast, reliable tests to verify the μ fix

use ga_engine::lattice_reduction::bkz_stable::StableBKZ;

fn main() {
    println!("=== Quick Stable BKZ Tests (μ fix verification) ===\n");

    // Test 1: 5D well-conditioned (should be ~24μs)
    test_5d_well_conditioned();

    // Test 2: 10D diagonal + perturbations (should be ~46μs)
    test_10d_diagonal();

    println!("\n=== Quick Tests Complete ===");
    println!("✅ If both tests passed with 0 precision escalations, the μ fix is working!");
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

    if stats.precision_escalations == 0 {
        println!("✅ No precision escalations (μ fix working!)");
    } else {
        println!("❌ Had precision escalations (μ bug may still exist)");
    }

    if elapsed.as_secs() < 5 {
        println!("✅ Completed quickly");
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

    if stats.precision_escalations == 0 {
        println!("✅ No precision escalations (μ fix working!)");
    } else {
        println!("❌ Had precision escalations (μ bug may still exist)");
    }

    if elapsed.as_secs() < 5 {
        println!("✅ Completed quickly");
    }

    println!();
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
