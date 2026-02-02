//! BKZ Incremental Update - The Critical Test
//!
//! **This is where GA should actually win!**
//!
//! Scenario: BKZ changes one basis vector, needs updated GSO
//!
//! Baseline: Recompute full GSO from changed index onward - O(n³)
//! GA Approach: Compose rotation with existing rotor - O(n²)
//!
//! This is ROTATION (basis change), not PROJECTION!
//! This is where rotors are the RIGHT tool!

use ga_engine::lattice_reduction::lll_baseline::LLL;
use ga_engine::lattice_reduction::rotor_nd::RotorND;
use std::time::Instant;

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Generate well-conditioned random basis
fn generate_basis(dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut basis = Vec::new();
    for i in 0..dim {
        let mut row = vec![0.0; dim];
        row[i] = 100.0;
        for j in 0..dim {
            let perturbation = ((seed + (i * dim + j) as u64) % 50) as f64;
            row[j] += perturbation;
        }
        basis.push(row);
    }
    basis
}

/// Perturb a vector slightly (simulates BKZ update)
fn perturb_vector(v: &[f64], amount: f64, seed: u64) -> Vec<f64> {
    v.iter().enumerate().map(|(i, &x)| {
        let delta = ((seed + i as u64) % 100) as f64 / 100.0 - 0.5;
        x + amount * delta
    }).collect()
}

/// Method 1: Full GSO recomputation (baseline)
fn recompute_gso_full(lll: &mut LLL, from_index: usize) -> std::time::Duration {
    let start = Instant::now();

    // Access via public method - we need to trigger GSO recomputation
    // For now, simulate by creating new LLL with updated basis
    // (This is what BKZ does internally)

    start.elapsed()
}

/// Method 2: Rotor composition (GA approach)
fn update_gso_rotor(
    old_vector: &[f64],
    new_vector: &[f64],
    old_rotor: &RotorND,
) -> (RotorND, std::time::Duration) {
    let start = Instant::now();

    // Step 1: Compute rotation from old to new vector
    let delta_rotor = RotorND::from_vectors(old_vector, new_vector);

    // Step 2: Compose with existing rotor
    let updated_rotor = delta_rotor.compose(old_rotor);

    let elapsed = start.elapsed();

    (updated_rotor, elapsed)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  BKZ Incremental Update: Full GSO vs Rotor Composition  ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    println!("Hypothesis: Rotor composition O(n²) beats full GSO O(n³)");
    println!();

    let dimensions = vec![10, 20, 30, 40, 50];
    let trials = 100;

    for &dim in &dimensions {
        println!("═══════════════════════════════════════════════════════════");
        println!("Dimension: {}", dim);
        println!("═══════════════════════════════════════════════════════════");
        println!();

        // Generate initial basis
        let basis = generate_basis(dim, 42);

        // Setup: Create initial rotors (one per vector)
        let mut rotors: Vec<RotorND> = Vec::new();
        for i in 0..dim {
            // Start with identity rotors
            rotors.push(RotorND::identity(dim));
        }

        // Test: Update middle vector (worst case for GSO recomputation)
        let update_index = dim / 2;
        let old_vector = basis[update_index].clone();
        let new_vector = perturb_vector(&old_vector, 10.0, 123);

        println!("Test: Update vector {} (middle of basis)", update_index);
        println!("  Old vector norm: {:.3}", norm(&old_vector));
        println!("  New vector norm: {:.3}", norm(&new_vector));
        println!();

        // Benchmark Method 1: Full GSO recomputation
        println!("Benchmarking full GSO recomputation ({} trials)...", trials);
        let mut full_gso_times = Vec::new();

        for trial in 0..trials {
            let mut test_basis = basis.clone();
            test_basis[update_index] = perturb_vector(&old_vector, 10.0, trial);

            let start = Instant::now();

            // Simulate full GSO recomputation from update_index onward
            // This is O((n - update_index) × n²) ≈ O(n³) for middle updates
            let mut lll = LLL::new(test_basis, 0.99);
            // The constructor already computes full GSO

            let elapsed = start.elapsed();
            full_gso_times.push(elapsed);
        }

        let full_gso_mean = full_gso_times.iter()
            .map(|t| t.as_secs_f64())
            .sum::<f64>() / trials as f64;

        println!("  Mean time: {:.3} µs", full_gso_mean * 1e6);
        println!();

        // Benchmark Method 2: Rotor composition
        println!("Benchmarking rotor composition ({} trials)...", trials);
        let mut rotor_times = Vec::new();

        for trial in 0..trials {
            let test_new_vector = perturb_vector(&old_vector, 10.0, trial);

            let (_, elapsed) = update_gso_rotor(
                &old_vector,
                &test_new_vector,
                &rotors[update_index],
            );

            rotor_times.push(elapsed);
        }

        let rotor_mean = rotor_times.iter()
            .map(|t| t.as_secs_f64())
            .sum::<f64>() / trials as f64;

        println!("  Mean time: {:.3} µs", rotor_mean * 1e6);
        println!();

        // Results
        let speedup = full_gso_mean / rotor_mean;

        println!("Results:");
        println!("  Full GSO:  {:.3} µs", full_gso_mean * 1e6);
        println!("  Rotor:     {:.3} µs", rotor_mean * 1e6);
        println!("  Speedup:   {:.2}×", speedup);
        println!();

        // Verdict
        if speedup >= 2.0 {
            println!("  ✅ STRONG WIN - Rotor ≥2× faster!");
        } else if speedup >= 1.5 {
            println!("  ✅ WIN - Rotor significantly faster (≥1.5×)");
        } else if speedup >= 1.2 {
            println!("  ✓ Moderate win - Rotor faster (≥1.2×)");
        } else if speedup >= 0.8 {
            println!("  ~ Roughly equal");
        } else {
            println!("  ✗ Full GSO wins");
        }
        println!();

        // Theoretical analysis
        let theoretical_full_gso = (dim - update_index) * dim * dim;  // O(n³)
        let theoretical_rotor = 3 * dim * dim;  // O(n²): construct + compose + apply
        let theoretical_speedup = theoretical_full_gso as f64 / theoretical_rotor as f64;

        println!("  Theoretical speedup: {:.2}× (O(n³)/O(n²))", theoretical_speedup);
        println!("  Actual speedup:      {:.2}×", speedup);

        if speedup >= theoretical_speedup * 0.5 {
            println!("  ✓ Achieving ≥50% of theoretical speedup");
        } else {
            println!("  ⚠️  Not achieving expected speedup (overhead dominates)");
        }

        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("Summary");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Key Question: Does rotor composition beat full GSO recomputation?");
    println!();
    println!("Expected: O(n²) rotor should be much faster than O(n³) GSO");
    println!("Reality: Measured above across dimensions 10-50");
    println!();
    println!("If rotor wins:");
    println!("  -> Use in BKZ (thousands of updates per reduction)");
    println!("  -> Potential 2-10x speedup for BKZ overall");
    println!();
    println!("If rotor loses:");
    println!("  -> Overhead of rotor construction dominates");
    println!("  -> GA doesn't help even for incremental updates");
    println!("  -> Need to find different application");
}
