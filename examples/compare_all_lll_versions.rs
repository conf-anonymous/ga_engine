//! LLL Three-Way Comparison
//!
//! Compare performance of:
//! 1. Baseline LLL (standard GSO)
//! 2. GA-LLL Hybrid (standard GSO + rotor tracking)
//! 3. GA-LLL Pure (rotor-only GSO)
//!
//! Goal: Determine if pure rotor approach reduces the 2.6x overhead seen in hybrid

use ga_engine::lattice_reduction::lll_baseline::LLL;
use ga_engine::lattice_reduction::ga_lll_rotors::GA_LLL;
use ga_engine::lattice_reduction::ga_lll_pure::GA_LLL_Pure;
use std::time::Instant;

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn generate_random_basis(dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut basis = Vec::new();
    for i in 0..dim {
        let mut row = vec![0.0; dim];
        row[i] = 100.0;  // Diagonal dominance
        for j in 0..dim {
            let perturbation = ((seed + (i * dim + j) as u64) % 50) as f64;
            row[j] += perturbation;
        }
        basis.push(row);
    }
    basis
}

fn basis_equal(a: &[Vec<f64>], b: &[Vec<f64>], tol: f64) -> bool {
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            if (a[i][j] - b[i][j]).abs() > tol {
                return false;
            }
        }
    }
    true
}

fn main() {
    println!("===============================================================");
    println!("  LLL Three-Way Performance Comparison");
    println!("===============================================================");
    println!();
    println!("Comparing:");
    println!("  1. Baseline LLL     - Standard GSO (reference)");
    println!("  2. GA-LLL Hybrid    - Standard GSO + rotor tracking");
    println!("  3. GA-LLL Pure      - Rotor-only GSO (this version)");
    println!();

    let dimensions = vec![10, 20, 30, 40];
    let num_trials = 5;

    for &dim in &dimensions {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Dimension: {}", dim);
        println!("═══════════════════════════════════════════════════════════════");
        println!();

        let mut baseline_times = Vec::new();
        let mut hybrid_times = Vec::new();
        let mut pure_times = Vec::new();

        let mut baseline_hf = 0.0;
        let mut hybrid_hf = 0.0;
        let mut pure_hf = 0.0;

        for trial in 0..num_trials {
            let basis = generate_random_basis(dim, 42 + trial as u64);

            // 1. Baseline LLL
            let start = Instant::now();
            let mut lll = LLL::new(basis.clone(), 0.99);
            lll.reduce();
            baseline_times.push(start.elapsed());
            if trial == 0 {
                baseline_hf = lll.hermite_factor();
            }
            let baseline_result = lll.get_basis().to_vec();

            // 2. Hybrid GA-LLL
            let start = Instant::now();
            let mut ga_lll = GA_LLL::new(basis.clone(), 0.99);
            ga_lll.reduce();
            hybrid_times.push(start.elapsed());
            if trial == 0 {
                hybrid_hf = ga_lll.hermite_factor();
            }
            let hybrid_result = ga_lll.get_basis().to_vec();

            // 3. Pure GA-LLL
            let start = Instant::now();
            let mut pure_lll = GA_LLL_Pure::new(basis.clone(), 0.99);
            pure_lll.reduce();
            pure_times.push(start.elapsed());
            if trial == 0 {
                pure_hf = pure_lll.hermite_factor();
            }
            let pure_result = pure_lll.get_basis().to_vec();

            // Verify correctness on first trial
            if trial == 0 {
                let hybrid_match = basis_equal(&baseline_result, &hybrid_result, 1e-8);
                let pure_match = basis_equal(&baseline_result, &pure_result, 1e-8);

                println!("Trial {} correctness:", trial + 1);
                println!("  Hybrid matches baseline: {}", if hybrid_match { "✓" } else { "✗" });
                println!("  Pure matches baseline:   {}", if pure_match { "✓" } else { "✗" });

                if !hybrid_match || !pure_match {
                    println!();
                    println!("  Baseline first vector: {:?}", &baseline_result[0][..3.min(dim)]);
                    println!("  Hybrid first vector:   {:?}", &hybrid_result[0][..3.min(dim)]);
                    println!("  Pure first vector:     {:?}", &pure_result[0][..3.min(dim)]);
                    println!();
                    println!("  Baseline HF: {:.6}", baseline_hf);
                    println!("  Hybrid HF:   {:.6}", hybrid_hf);
                    println!("  Pure HF:     {:.6}", pure_hf);
                }
                println!();
            }
        }

        // Compute statistics
        let baseline_mean = baseline_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / num_trials as f64;
        let hybrid_mean = hybrid_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / num_trials as f64;
        let pure_mean = pure_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / num_trials as f64;

        let baseline_min = baseline_times.iter().map(|t| t.as_secs_f64()).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let hybrid_min = hybrid_times.iter().map(|t| t.as_secs_f64()).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let pure_min = pure_times.iter().map(|t| t.as_secs_f64()).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        println!("Results ({} trials):", num_trials);
        println!();
        println!("┌────────────────┬─────────────┬─────────────┬──────────────┐");
        println!("│ Version        │ Mean Time   │ Min Time    │ Overhead     │");
        println!("├────────────────┼─────────────┼─────────────┼──────────────┤");
        println!("│ Baseline       │ {:>10.3} │ {:>10.3} │ {:>11}  │",
                 format_time(baseline_mean),
                 format_time(baseline_min),
                 "1.00×");
        println!("│ Hybrid (v1)    │ {:>10.3} │ {:>10.3} │ {:>11}  │",
                 format_time(hybrid_mean),
                 format_time(hybrid_min),
                 format!("{:.2}×", hybrid_mean / baseline_mean));
        println!("│ Pure (v2)      │ {:>10.3} │ {:>10.3} │ {:>11}  │",
                 format_time(pure_mean),
                 format_time(pure_min),
                 format!("{:.2}×", pure_mean / baseline_mean));
        println!("└────────────────┴─────────────┴─────────────┴──────────────┘");
        println!();

        // Speedup analysis
        let hybrid_vs_baseline = hybrid_mean / baseline_mean;
        let pure_vs_baseline = pure_mean / baseline_mean;
        let pure_vs_hybrid = pure_mean / hybrid_mean;

        println!("Performance Ratios:");
        println!("  Hybrid overhead: {:.2}× slower than baseline", hybrid_vs_baseline);
        println!("  Pure overhead:   {:.2}× slower than baseline", pure_vs_baseline);
        println!("  Pure vs Hybrid:  {:.2}× {}",
                 pure_vs_hybrid,
                 if pure_vs_hybrid < 1.0 { "FASTER" } else { "slower" });
        println!();

        // Verdict
        println!("Verdict:");
        if pure_vs_baseline < 2.0 {
            println!("  ✓ Pure rotor achieves <2× overhead target!");
        } else {
            println!("  ✗ Pure rotor still >2× overhead");
        }

        if pure_vs_hybrid < 1.0 {
            println!("  ✓ Pure rotor is FASTER than hybrid (eliminated redundant work)");
        } else if pure_vs_hybrid < 1.1 {
            println!("  ~ Pure rotor roughly same speed as hybrid");
        } else {
            println!("  ✗ Pure rotor SLOWER than hybrid (rotor ops more expensive)");
        }

        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Key Question: Does pure rotor GSO reduce overhead?");
    println!();
    println!("If Pure < Hybrid:");
    println!("  → Rotor-only approach eliminates redundant standard arithmetic");
    println!("  → Path forward: Optimize rotor operations (Metal GPU, etc.)");
    println!();
    println!("If Pure ≈ Hybrid:");
    println!("  → Rotor operations cost same as standard arithmetic");
    println!("  → Need different optimization strategy");
    println!();
    println!("If Pure > Hybrid:");
    println!("  → Rotor operations are MORE expensive than vector arithmetic");
    println!("  → May need to reconsider rotor approach");
    println!();
}

fn format_time(seconds: f64) -> String {
    if seconds < 1e-6 {
        format!("{:.3} ns", seconds * 1e9)
    } else if seconds < 1e-3 {
        format!("{:.3} µs", seconds * 1e6)
    } else if seconds < 1.0 {
        format!("{:.3} ms", seconds * 1e3)
    } else {
        format!("{:.3} s", seconds)
    }
}
