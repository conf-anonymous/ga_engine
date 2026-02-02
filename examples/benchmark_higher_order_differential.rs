//! Benchmark: Higher-Order Differential Cryptanalysis
//!
//! This benchmark compares baseline vs GA-based higher-order differential analysis
//! on the AES S-box to measure the actual computational savings from structural shortcuts.
//!
//! Key metric: S-box evaluation savings when GA can detect zero differentials without
//! explicit evaluation (a ∧ b = 0 or a ∧ b ∧ c = 0).

use ga_engine::cryptanalysis::sbox_ga::SBoxGA;
use ga_engine::cryptanalysis::higher_order_differential::{
    SecondOrderDifferential, ThirdOrderDifferential, DifferentialStats
};
use std::time::Instant;

// AES S-box lookup table
const AES_SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

fn benchmark_second_order() {
    println!("Second-Order Differential (4-point evaluation)");
    println!("==============================================\n");

    let sbox = SBoxGA::from_lut(AES_SBOX.to_vec(), 8);
    let diff = SecondOrderDifferential::new(sbox);

    // Comprehensive sweep of input space
    // Sample points for x
    let sample_points: Vec<u8> = (0..=255).step_by(16).collect(); // 16 sample points

    // For diffs, use random sampling to get realistic mix of dependent/independent
    // Using specific values that give good coverage:
    // - Powers of 2 (independent with each other): 1, 2, 4, 8, 16, 32, 64, 128
    // - Combinations (mix of dependent/independent): 3, 5, 6, 7, 9, 10, 12, 15, ...
    let sample_diffs: Vec<u8> = vec![
        1, 2, 3, 4, 5, 6, 7, 8,        // First 8
        9, 10, 12, 15, 16, 17, 20, 24, // Mix
        31, 32, 33, 48, 63, 64, 65, 96,  // Mix
        127, 128, 129, 192, 255        // Last few
    ];

    let total_tests = sample_points.len() * sample_diffs.len() * sample_diffs.len();
    println!("Testing {} combinations", total_tests);
    println!("Sample size: {} x-points × {} diff-values² = {}\n",
        sample_points.len(), sample_diffs.len(), total_tests);

    // Baseline
    let start = Instant::now();
    let mut baseline_total_stats = DifferentialStats::default();

    for &x in &sample_points {
        for &a in &sample_diffs {
            for &b in &sample_diffs {
                let (_, stats) = diff.compute_baseline(x, a, b);
                baseline_total_stats.sbox_evals += stats.sbox_evals;
                baseline_total_stats.xor_ops += stats.xor_ops;
            }
        }
    }

    let baseline_time = start.elapsed();

    // GA-based (unoptimized - with full multivector operations)
    let start = Instant::now();
    let mut ga_total_stats = DifferentialStats::default();
    let mut ga_shortcuts = 0;

    for &x in &sample_points {
        for &a in &sample_diffs {
            for &b in &sample_diffs {
                let (_, stats) = diff.compute_ga(x, a, b);
                ga_total_stats.sbox_evals += stats.sbox_evals;
                ga_total_stats.multivector_ops += stats.multivector_ops;

                if stats.sbox_evals == 0 {
                    ga_shortcuts += 1;
                }
            }
        }
    }

    let ga_time = start.elapsed();

    // GA-based OPTIMIZED (bit operations only)
    let start = Instant::now();
    let mut ga_opt_total_stats = DifferentialStats::default();
    let mut ga_opt_shortcuts = 0;

    for &x in &sample_points {
        for &a in &sample_diffs {
            for &b in &sample_diffs {
                let (_, stats) = diff.compute_ga_optimized(x, a, b);
                ga_opt_total_stats.sbox_evals += stats.sbox_evals;
                ga_opt_total_stats.multivector_ops += stats.multivector_ops;

                if stats.sbox_evals == 0 {
                    ga_opt_shortcuts += 1;
                }
            }
        }
    }

    let ga_opt_time = start.elapsed();

    // Results
    println!("Baseline:");
    println!("  Time: {:.2?}", baseline_time);
    println!("  S-box evaluations: {}", baseline_total_stats.sbox_evals);
    println!("  XOR operations: {}", baseline_total_stats.xor_ops);
    println!();

    println!("GA-based (unoptimized):");
    println!("  Time: {:.2?}", ga_time);
    println!("  S-box evaluations: {}", ga_total_stats.sbox_evals);
    println!("  Multivector operations: {}", ga_total_stats.multivector_ops);
    println!("  Shortcuts taken: {} / {} ({:.1}%)",
        ga_shortcuts, total_tests,
        100.0 * ga_shortcuts as f64 / total_tests as f64);
    println!("  Time speedup: {:.2}×", baseline_time.as_secs_f64() / ga_time.as_secs_f64());
    println!();

    println!("GA-based (OPTIMIZED):");
    println!("  Time: {:.2?}", ga_opt_time);
    println!("  S-box evaluations: {}", ga_opt_total_stats.sbox_evals);
    println!("  Bit operations: {}", ga_opt_total_stats.multivector_ops);
    println!("  Shortcuts taken: {} / {} ({:.1}%)",
        ga_opt_shortcuts, total_tests,
        100.0 * ga_opt_shortcuts as f64 / total_tests as f64);
    println!("  Time speedup: {:.2}×", baseline_time.as_secs_f64() / ga_opt_time.as_secs_f64());
    println!();

    let eval_savings = baseline_total_stats.sbox_evals.saturating_sub(ga_opt_total_stats.sbox_evals);
    let savings_pct = 100.0 * eval_savings as f64 / baseline_total_stats.sbox_evals as f64;

    println!("Savings:");
    println!("  S-box evaluations saved: {} ({:.1}%)", eval_savings, savings_pct);
    println!();
}

fn benchmark_third_order() {
    println!("Third-Order Differential (8-point evaluation)");
    println!("==============================================\n");

    let sbox = SBoxGA::from_lut(AES_SBOX.to_vec(), 8);
    let diff = ThirdOrderDifferential::new(sbox);

    // Smaller sample for third-order (8× more expensive per test)
    let sample_points: Vec<u8> = (0..=255).step_by(32).collect(); // 8 sample points

    // Use same balanced sample as second-order but smaller
    let sample_diffs: Vec<u8> = vec![
        1, 2, 3, 4, 5, 6, 7, 8,     // Powers of 2 and small values
        15, 16, 31, 32, 63, 64,     // More combinations
        127, 128, 255               // Edge cases
    ];

    let total_tests = sample_points.len() * sample_diffs.len() * sample_diffs.len() * sample_diffs.len();
    println!("Testing {} combinations", total_tests);
    println!("Sample size: {} x-points × {} diff-values³ = {}\n",
        sample_points.len(), sample_diffs.len(), total_tests);

    // Baseline
    let start = Instant::now();
    let mut baseline_total_stats = DifferentialStats::default();

    for &x in &sample_points {
        for &a in &sample_diffs {
            for &b in &sample_diffs {
                for &c in &sample_diffs {
                    let (_, stats) = diff.compute_baseline(x, a, b, c);
                    baseline_total_stats.sbox_evals += stats.sbox_evals;
                    baseline_total_stats.xor_ops += stats.xor_ops;
                }
            }
        }
    }

    let baseline_time = start.elapsed();

    // GA-based (unoptimized)
    let start = Instant::now();
    let mut ga_total_stats = DifferentialStats::default();
    let mut ga_shortcuts = 0;

    for &x in &sample_points {
        for &a in &sample_diffs {
            for &b in &sample_diffs {
                for &c in &sample_diffs {
                    let (_, stats) = diff.compute_ga(x, a, b, c);
                    ga_total_stats.sbox_evals += stats.sbox_evals;
                    ga_total_stats.multivector_ops += stats.multivector_ops;

                    if stats.sbox_evals == 0 {
                        ga_shortcuts += 1;
                    }
                }
            }
        }
    }

    let ga_time = start.elapsed();

    // GA-based OPTIMIZED
    let start = Instant::now();
    let mut ga_opt_total_stats = DifferentialStats::default();
    let mut ga_opt_shortcuts = 0;

    for &x in &sample_points {
        for &a in &sample_diffs {
            for &b in &sample_diffs {
                for &c in &sample_diffs {
                    let (_, stats) = diff.compute_ga_optimized(x, a, b, c);
                    ga_opt_total_stats.sbox_evals += stats.sbox_evals;
                    ga_opt_total_stats.multivector_ops += stats.multivector_ops;

                    if stats.sbox_evals == 0 {
                        ga_opt_shortcuts += 1;
                    }
                }
            }
        }
    }

    let ga_opt_time = start.elapsed();

    // Results
    println!("Baseline:");
    println!("  Time: {:.2?}", baseline_time);
    println!("  S-box evaluations: {}", baseline_total_stats.sbox_evals);
    println!("  XOR operations: {}", baseline_total_stats.xor_ops);
    println!();

    println!("GA-based (unoptimized):");
    println!("  Time: {:.2?}", ga_time);
    println!("  S-box evaluations: {}", ga_total_stats.sbox_evals);
    println!("  Multivector operations: {}", ga_total_stats.multivector_ops);
    println!("  Shortcuts taken: {} / {} ({:.1}%)",
        ga_shortcuts, total_tests,
        100.0 * ga_shortcuts as f64 / total_tests as f64);
    println!("  Time speedup: {:.2}×", baseline_time.as_secs_f64() / ga_time.as_secs_f64());
    println!();

    println!("GA-based (OPTIMIZED):");
    println!("  Time: {:.2?}", ga_opt_time);
    println!("  S-box evaluations: {}", ga_opt_total_stats.sbox_evals);
    println!("  Bit operations: {}", ga_opt_total_stats.multivector_ops);
    println!("  Shortcuts taken: {} / {} ({:.1}%)",
        ga_opt_shortcuts, total_tests,
        100.0 * ga_opt_shortcuts as f64 / total_tests as f64);
    println!("  Time speedup: {:.2}×", baseline_time.as_secs_f64() / ga_opt_time.as_secs_f64());
    println!();

    let eval_savings = baseline_total_stats.sbox_evals.saturating_sub(ga_opt_total_stats.sbox_evals);
    let savings_pct = 100.0 * eval_savings as f64 / baseline_total_stats.sbox_evals as f64;

    println!("Savings:");
    println!("  S-box evaluations saved: {} ({:.1}%)", eval_savings, savings_pct);
    println!();
}

fn benchmark_dependent_case_analysis() {
    println!("Dependent Case Analysis");
    println!("========================\n");
    println!("Analyzing how often GA shortcuts apply in realistic scenarios\n");

    let sbox = SBoxGA::from_lut(AES_SBOX.to_vec(), 8);
    let diff2 = SecondOrderDifferential::new(sbox.clone());
    let diff3 = ThirdOrderDifferential::new(sbox);

    // Test 1: Identical vectors (a = b)
    println!("Test 1: Second-order with a = b (should always shortcut)");
    let mut shortcuts = 0;
    for x in 0..=255u8 {
        for a in 1..=255u8 {
            let (_, stats) = diff2.compute_ga(x, a, a);
            if stats.sbox_evals == 0 {
                shortcuts += 1;
            }
        }
    }
    println!("  Shortcuts: {} / {} ({:.1}%)\n", shortcuts, 256 * 255,
        100.0 * shortcuts as f64 / (256.0 * 255.0));

    // Test 2: Power-of-2 patterns (single bit differences)
    println!("Test 2: Second-order with power-of-2 patterns");
    let powers: Vec<u8> = (0..8).map(|i| 1 << i).collect();
    shortcuts = 0;
    let mut total = 0;
    for x in (0..=255u8).step_by(8) {
        for &a in &powers {
            for &b in &powers {
                let (_, stats) = diff2.compute_ga(x, a, b);
                if stats.sbox_evals == 0 {
                    shortcuts += 1;
                }
                total += 1;
            }
        }
    }
    println!("  Shortcuts: {} / {} ({:.1}%)\n", shortcuts, total,
        100.0 * shortcuts as f64 / total as f64);

    // Test 3: Third-order with two identical vectors
    println!("Test 3: Third-order with a = b (c varies)");
    shortcuts = 0;
    total = 0;
    for x in (0..=255u8).step_by(16) {
        for a in (1..=255u8).step_by(16) {
            for c in (1..=255u8).step_by(16) {
                let (_, stats) = diff3.compute_ga(x, a, a, c);
                if stats.sbox_evals == 0 {
                    shortcuts += 1;
                }
                total += 1;
            }
        }
    }
    println!("  Shortcuts: {} / {} ({:.1}%)\n", shortcuts, total,
        100.0 * shortcuts as f64 / total as f64);

    // Test 4: Third-order with all identical (a = b = c)
    println!("Test 4: Third-order with a = b = c");
    shortcuts = 0;
    total = 0;
    for x in (0..=255u8).step_by(8) {
        for a in (1..=255u8).step_by(8) {
            let (_, stats) = diff3.compute_ga(x, a, a, a);
            if stats.sbox_evals == 0 {
                shortcuts += 1;
            }
            total += 1;
        }
    }
    println!("  Shortcuts: {} / {} ({:.1}%)\n", shortcuts, total,
        100.0 * shortcuts as f64 / total as f64);
}

fn main() {
    println!("=================================================");
    println!("  Higher-Order Differential Cryptanalysis");
    println!("  GA Structural Shortcuts vs Baseline");
    println!("=================================================\n");

    benchmark_second_order();
    println!("\n");

    benchmark_third_order();
    println!("\n");

    benchmark_dependent_case_analysis();

    println!("\n=================================================");
    println!("Interpretation:");
    println!("  - GA detects linearly dependent vectors via wedge product");
    println!("  - When a ∧ b = 0 (or a ∧ b ∧ c = 0), differential is zero");
    println!("  - Skips ALL S-box evaluations for these cases");
    println!("  - This is a structural shortcut, not micro-optimization");
    println!("=================================================");
}
