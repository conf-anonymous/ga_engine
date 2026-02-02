//! Benchmark: Differential Trail Propagation - Matrix vs Rotor
//!
//! This measures the speedup of using rotors (O(n)) vs matrices (O(n²))
//! for propagating differentials through cipher rounds.

use ga_engine::cryptanalysis::trail_propagation::{
    propagate_differential_trail_matrix, propagate_differential_trail_rotor, RoundTransform,
};
use ga_engine::lattice_reduction::rotor_nd::RotorND;
use std::time::Instant;

/// Create rotation matrix (orthogonal)
fn create_rotation_matrix(n: usize, angle: f64) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; n]; n];

    // Simple 2D rotation in first two dimensions
    matrix[0][0] = angle.cos();
    matrix[0][1] = -angle.sin();
    matrix[1][0] = angle.sin();
    matrix[1][1] = angle.cos();

    // Identity for remaining dimensions
    for i in 2..n {
        matrix[i][i] = 1.0;
    }

    matrix
}

/// Create rotor from rotation matrix
fn create_rotation_rotor(n: usize, angle: f64) -> RotorND {
    // Create two vectors that define the rotation
    let mut v1 = vec![0.0; n];
    let mut v2 = vec![0.0; n];

    v1[0] = 1.0;
    v2[0] = angle.cos();
    v2[1] = angle.sin();

    RotorND::from_vectors(&v1, &v2)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Differential Trail Propagation: Matrix vs Rotor         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let dimensions = vec![4, 8, 12, 16];
    let num_rounds = 8;
    let trials = 1000;

    println!("Configuration:");
    println!("  Rounds per trial: {}", num_rounds);
    println!("  Trials: {}", trials);
    println!();

    for &dim in &dimensions {
        println!("═══════════════════════════════════════════════════════════");
        println!("Dimension: {}", dim);
        println!("═══════════════════════════════════════════════════════════");
        println!();

        // Create rotation transformation (orthogonal, so both methods valid)
        let angle = std::f64::consts::PI / 4.0; // 45 degrees
        let matrix = create_rotation_matrix(dim, angle);
        let rotor = create_rotation_rotor(dim, angle);

        // Create round transforms
        let matrix_rounds: Vec<RoundTransform> = (0..num_rounds)
            .map(|_| RoundTransform::from_matrix(matrix.clone()))
            .collect();

        let rotor_rounds: Vec<RoundTransform> = (0..num_rounds)
            .map(|_| RoundTransform::from_rotor(rotor.clone()))
            .collect();

        // Initial difference
        let mut initial_delta = vec![0.0; dim];
        initial_delta[0] = 1.0;

        // Warmup
        let _ = propagate_differential_trail_matrix(&initial_delta, &matrix_rounds);
        let _ = propagate_differential_trail_rotor(&initial_delta, &rotor_rounds);

        // Benchmark Matrix
        println!("Benchmarking matrix propagation ({} trials)...", trials);
        let mut matrix_times = Vec::new();

        for _ in 0..trials {
            let start = Instant::now();
            let _ = propagate_differential_trail_matrix(&initial_delta, &matrix_rounds);
            matrix_times.push(start.elapsed());
        }

        let matrix_mean =
            matrix_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / trials as f64;
        println!("  Mean: {:.3} µs", matrix_mean * 1_000_000.0);

        // Benchmark Rotor
        println!("Benchmarking rotor propagation ({} trials)...", trials);
        let mut rotor_times = Vec::new();

        for _ in 0..trials {
            let start = Instant::now();
            let _ = propagate_differential_trail_rotor(&initial_delta, &rotor_rounds);
            rotor_times.push(start.elapsed());
        }

        let rotor_mean = rotor_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / trials as f64;
        println!("  Mean: {:.3} µs", rotor_mean * 1_000_000.0);

        // Speedup
        let speedup = matrix_mean / rotor_mean;
        println!();
        println!("  SPEEDUP: {:.2}×", speedup);
        println!();

        // Verify correctness
        let trail_matrix = propagate_differential_trail_matrix(&initial_delta, &matrix_rounds);
        let trail_rotor = propagate_differential_trail_rotor(&initial_delta, &rotor_rounds);

        let final_matrix = trail_matrix.get_delta(num_rounds).unwrap();
        let final_rotor = trail_rotor.get_delta(num_rounds).unwrap();

        let mut max_diff = 0.0;
        for i in 0..dim {
            let diff = (final_matrix[i] - final_rotor[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        if max_diff < 1e-6 {
            println!("  ✓ Results match (max diff: {:.2e})", max_diff);
        } else {
            println!("  ✗ Results differ (max diff: {:.2e})", max_diff);
        }

        println!();
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Theoretical complexity:");
    println!("  Matrix: O(n²) per round");
    println!("  Rotor:  O(n) per round");
    println!();
    println!("Expected speedup: ~n× for dimension n");
    println!();
}
