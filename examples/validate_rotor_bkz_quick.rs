//! Quick Validation of Rotor-Tracked BKZ
//!
//! Focused validation on dimensions where BKZ completes quickly,
//! but with enough tours to trigger rotor updates.

use ga_engine::lattice_reduction::bkz_rotor::RotorBKZ;
use ga_engine::lattice_reduction::bkz_stable::StableBKZ;
use std::time::Instant;

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Generate random basis with controlled ill-conditioning
fn generate_random_basis(dim: usize, seed: u64, scale: f64) -> Vec<Vec<f64>> {
    let mut basis = Vec::new();
    for i in 0..dim {
        let mut row = vec![0.0; dim];
        row[i] = scale; // Diagonal
        for j in 0..dim {
            let perturbation = ((seed + (i * dim + j) as u64) % 50) as f64 - 25.0;
            row[j] += perturbation;
        }
        basis.push(row);
    }
    basis
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         ROTOR-TRACKED BKZ QUICK VALIDATION                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Test on dimensions 10, 15, 20, 25 with multiple tours
    let test_configs = vec![
        (10, 5, 10),  // dim 10, block 5, 10 tours
        (15, 7, 10),  // dim 15, block 7, 10 tours
        (20, 10, 8),  // dim 20, block 10, 8 tours
        (25, 12, 5),  // dim 25, block 12, 5 tours
    ];

    let mut all_speedups = Vec::new();
    let mut all_quality_ok = Vec::new();

    for (dim, block_size, max_tours) in test_configs {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Dimension {}, Block Size {}, Max Tours {}", dim, block_size, max_tours);
        println!("═══════════════════════════════════════════════════════════════");

        let basis = generate_random_basis(dim, 42, 100.0);

        // Baseline BKZ
        println!("Running baseline BKZ...");
        let baseline_start = Instant::now();
        let mut baseline_bkz = StableBKZ::new(basis.clone(), block_size, 0.99);
        baseline_bkz.reduce_with_limit(max_tours);
        let baseline_time = baseline_start.elapsed();
        let baseline_hf = baseline_bkz.hermite_factor();
        let baseline_first_norm = l2_norm(&baseline_bkz.get_basis()[0]);
        let baseline_stats = baseline_bkz.get_stats();

        println!("  Time: {:?}", baseline_time);
        println!("  Hermite factor: {:.6}", baseline_hf);
        println!("  First vector norm: {:.6}", baseline_first_norm);
        println!("  Tours: {}", baseline_stats.tours);
        println!("  Improvements: {}", baseline_stats.improvements);
        println!();

        // Rotor BKZ
        println!("Running rotor BKZ...");
        let rotor_start = Instant::now();
        let mut rotor_bkz = RotorBKZ::new(basis.clone(), block_size, 0.99);
        rotor_bkz.reduce_with_limit(max_tours);
        let rotor_time = rotor_start.elapsed();
        let rotor_hf = rotor_bkz.hermite_factor();
        let rotor_first_norm = l2_norm(&rotor_bkz.get_basis()[0]);
        let rotor_stats = rotor_bkz.get_stats();

        println!("  Time: {:?}", rotor_time);
        println!("  Hermite factor: {:.6}", rotor_hf);
        println!("  First vector norm: {:.6}", rotor_first_norm);
        println!("  Tours: {}", rotor_stats.tours);
        println!("  Improvements: {}", rotor_stats.improvements);
        println!("  Rotor updates: {}", rotor_stats.rotor_updates);
        println!("  Full GSO recomputations: {}", rotor_stats.full_gso_recomputations);
        println!("  GSO speedup: {:.2}×", rotor_stats.gso_speedup());
        println!();

        // Comparison
        let speedup = if rotor_time.as_micros() > 0 {
            baseline_time.as_micros() as f64 / rotor_time.as_micros() as f64
        } else {
            1.0
        };

        let hf_diff = (baseline_hf - rotor_hf).abs();
        let hf_rel_diff = if baseline_hf > 0.0 {
            hf_diff / baseline_hf
        } else {
            0.0
        };
        let quality_ok = hf_rel_diff < 0.05;

        println!("COMPARISON:");
        println!("  Overall speedup: {:.2}×", speedup);
        println!("  Hermite factor difference: {:.6} ({:.2}%)",
                 hf_diff, hf_rel_diff * 100.0);
        println!("  Quality preserved: {}", if quality_ok { "✓ YES" } else { "✗ NO" });
        println!();

        all_speedups.push(speedup);
        all_quality_ok.push(quality_ok);
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let avg_speedup: f64 = all_speedups.iter().sum::<f64>() / all_speedups.len() as f64;
    let min_speedup = all_speedups.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_speedup = all_speedups.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let quality_count = all_quality_ok.iter().filter(|&&x| x).count();

    println!("Overall Speedup:");
    println!("  Average: {:.2}×", avg_speedup);
    println!("  Min: {:.2}×", min_speedup);
    println!("  Max: {:.2}×", max_speedup);
    println!();
    println!("Quality Preservation: {}/{} tests", quality_count, all_quality_ok.len());
    println!();

    // Verdict
    println!("═══════════════════════════════════════════════════════════════");
    println!("VERDICT");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let all_quality_preserved = quality_count == all_quality_ok.len();
    let significant_speedup = avg_speedup >= 1.5;

    if all_quality_preserved && significant_speedup {
        println!("🎉 VALIDATION PASSED!");
        println!();
        println!("  ✓ Quality preserved on all tests");
        println!("  ✓ Significant speedup: {:.2}× average", avg_speedup);
        println!();
        println!("Ready!");
    } else if all_quality_preserved {
        println!("⚠ PARTIAL PASS");
        println!();
        println!("  ✓ Quality preserved on all tests");
        println!("  ⚠ Speedup: {:.2}× (below 1.5× target)", avg_speedup);
        println!();
        println!("Results are correct but need performance optimization.");
    } else {
        println!("✗ VALIDATION FAILED");
        println!();
        if !all_quality_preserved {
            println!("  ✗ Quality not preserved on {}/{} tests",
                     all_quality_ok.len() - quality_count, all_quality_ok.len());
        }
        if !significant_speedup {
            println!("  ✗ Speedup: {:.2}× (below 1.5× target)", avg_speedup);
        }
        println!();
        println!("Implementation needs more work.");
    }
}
