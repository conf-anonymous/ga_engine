//! Hypothesis 1: Numerical Stability Test
//!
//! **Hypothesis**: Rotor-based GSO has better numerical stability than standard GSO
//!
//! **Test**: Compare GA-LLL vs standard LLL on identical lattices
//!
//! **Metrics**:
//! - Hermite factor (must be equal or better)
//! - Orthogonality defect (lower is better)
//! - Numerical error (if ground truth available)
//!
//! **Success Criteria**: GA-LLL numerical error < 0.5x standard LLL error

use ga_engine::lattice_reduction::ga_lll_rotors::GA_LLL;
use ga_engine::lattice_reduction::lll_baseline::LLL;
use std::time::Instant;

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Generate well-conditioned random basis for testing
/// Start with identity and perturb it to avoid singular matrices
fn generate_random_basis(dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut basis = Vec::new();
    for i in 0..dim {
        let mut row = vec![0.0; dim];
        // Start with scaled identity
        row[i] = 100.0;
        // Add perturbations
        for j in 0..dim {
            let perturbation = ((seed + (i * dim + j) as u64) % 50) as f64;
            row[j] += perturbation;
        }
        basis.push(row);
    }
    basis
}

/// Compute Frobenius norm difference between two bases
fn basis_diff_norm(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            let diff = a[i][j] - b[i][j];
            sum_sq += diff * diff;
        }
    }
    sum_sq.sqrt()
}

fn main() {
    println!("======================================");
    println!("Hypothesis 1: Numerical Stability Test");
    println!("GA-LLL vs Standard LLL Comparison");
    println!("======================================");
    println!();

    let dimensions = vec![10, 20, 30, 40];

    for &dim in &dimensions {
        println!("----------------------------------------");
        println!("Testing dimension: {}", dim);
        println!("----------------------------------------");

        // Generate test lattice
        let basis = generate_random_basis(dim, 42);

        // Test 1: Standard LLL
        println!("Running standard LLL...");
        let start = Instant::now();
        let mut lll = LLL::new(basis.clone(), 0.99);
        lll.reduce();
        let lll_time = start.elapsed();

        let lll_basis = lll.get_basis();
        let lll_hf = lll.hermite_factor();
        let lll_stats = lll.get_stats();
        let lll_first_norm = norm(&lll_basis[0]);

        println!("  Time: {:?}", lll_time);
        println!("  Hermite factor: {:.6}", lll_hf);
        println!("  First vector norm: {:.6}", lll_first_norm);
        println!("  Swaps: {}", lll_stats.swaps);
        println!("  Size reductions: {}", lll_stats.size_reductions);
        println!("  GSO updates: {}", lll_stats.gso_updates);
        println!();

        // Test 2: GA-LLL
        println!("Running GA-LLL (rotor-based)...");
        let start = Instant::now();
        let mut ga_lll = GA_LLL::new(basis.clone(), 0.99);
        ga_lll.reduce();
        let ga_time = start.elapsed();

        let ga_basis = ga_lll.get_basis();
        let ga_hf = ga_lll.hermite_factor();
        let ga_defect = ga_lll.orthogonality_defect();
        let ga_stats = ga_lll.get_stats();
        let ga_first_norm = norm(&ga_basis[0]);

        println!("  Time: {:?}", ga_time);
        println!("  Hermite factor: {:.6}", ga_hf);
        println!("  Orthogonality defect: {:.6}", ga_defect);
        println!("  First vector norm: {:.6}", ga_first_norm);
        println!("  Swaps: {}", ga_stats.lll_stats.swaps);
        println!("  Size reductions: {}", ga_stats.lll_stats.size_reductions);
        println!("  GSO updates: {}", ga_stats.lll_stats.gso_updates);
        println!("  Rotor constructions: {}", ga_stats.rotor_constructions);
        println!("  Rotor compositions: {}", ga_stats.rotor_compositions);
        println!("  Rotor applications: {}", ga_stats.rotor_applications);
        println!();

        // Comparison
        println!("Comparison:");
        let time_ratio = ga_time.as_secs_f64() / lll_time.as_secs_f64();
        println!("  Time ratio (GA/LLL): {:.3}×", time_ratio);

        let hf_diff = (ga_hf - lll_hf).abs();
        println!("  Hermite factor difference: {:.6} ({:.2}%)", hf_diff, 100.0 * hf_diff / lll_hf);

        let basis_error = basis_diff_norm(ga_basis, lll_basis);
        println!("  Basis difference (Frobenius norm): {:.6e}", basis_error);

        // First vector should be similar (might differ due to different reduction paths)
        let first_norm_diff = (ga_first_norm - lll_first_norm).abs();
        println!("  First vector norm difference: {:.6} ({:.2}%)", first_norm_diff, 100.0 * first_norm_diff / lll_first_norm);

        // Verdict for this dimension
        println!();
        println!("Verdict:");
        if time_ratio < 2.0 {
            println!("  ✓ Time overhead acceptable (<2×): {:.3}×", time_ratio);
        } else {
            println!("  ✗ Time overhead too high (≥2×): {:.3}×", time_ratio);
        }

        if hf_diff / lll_hf < 0.01 {
            println!("  ✓ Hermite factors match (<1% difference)");
        } else {
            println!("  ~ Hermite factors differ (≥1% difference)");
        }

        if ga_defect < 1.1 {
            println!("  ✓ Good orthogonality defect (<1.1): {:.6}", ga_defect);
        } else {
            println!("  ~ Orthogonality defect: {:.6}", ga_defect);
        }

        println!();
    }

    println!("======================================");
    println!("Summary");
    println!("======================================");
    println!();
    println!("Hypothesis 1 tests numerical stability of rotor-based GSO.");
    println!("Key findings:");
    println!("- Hermite factors should match between LLL and GA-LLL");
    println!("- Orthogonality defect measures numerical conditioning");
    println!("- Time overhead indicates computational cost of rotors");
    println!();
    println!("Next steps:");
    println!("1. Run on larger dimensions (50-100)");
    println!("2. Run on SVP Challenge instances (hard inputs)");
    println!("3. Measure numerical error on ill-conditioned lattices");
    println!("4. Test Hypothesis 2: Rotor composition speed");
}
