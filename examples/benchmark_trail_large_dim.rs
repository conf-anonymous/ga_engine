//! Benchmark: Trail Propagation - Larger Dimensions
//!
//! Tests if rotor speedup appears at larger dimensions (32, 64, 128)
//! where O(n) vs O(n²) difference should be more significant.

use ga_engine::cryptanalysis::trail_propagation::{
    propagate_differential_trail_matrix, propagate_differential_trail_rotor, RoundTransform,
};
use ga_engine::lattice_reduction::rotor_nd::RotorND;
use std::time::Instant;

fn create_rotation_matrix(n: usize, angle: f64) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; n]; n];
    matrix[0][0] = angle.cos();
    matrix[0][1] = -angle.sin();
    matrix[1][0] = angle.sin();
    matrix[1][1] = angle.cos();
    for i in 2..n {
        matrix[i][i] = 1.0;
    }
    matrix
}

fn create_rotation_rotor(n: usize, angle: f64) -> RotorND {
    let mut v1 = vec![0.0; n];
    let mut v2 = vec![0.0; n];
    v1[0] = 1.0;
    v2[0] = angle.cos();
    v2[1] = angle.sin();
    RotorND::from_vectors(&v1, &v2)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Trail Propagation: Larger Dimensions (32, 64, 128)        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let dimensions = vec![16, 32, 64, 128];
    let num_rounds = 4; // Fewer rounds for larger dimensions
    let trials = 100;   // Fewer trials for speed

    for &dim in &dimensions {
        println!("═══════════════════════════════════════════════════════════");
        println!("Dimension: {}", dim);
        println!("═══════════════════════════════════════════════════════════");

        let angle = std::f64::consts::PI / 4.0;
        let matrix = create_rotation_matrix(dim, angle);
        let rotor = create_rotation_rotor(dim, angle);

        let matrix_rounds: Vec<RoundTransform> = (0..num_rounds)
            .map(|_| RoundTransform::from_matrix(matrix.clone()))
            .collect();

        let rotor_rounds: Vec<RoundTransform> = (0..num_rounds)
            .map(|_| RoundTransform::from_rotor(rotor.clone()))
            .collect();

        let mut initial_delta = vec![0.0; dim];
        initial_delta[0] = 1.0;

        // Warmup
        let _ = propagate_differential_trail_matrix(&initial_delta, &matrix_rounds);
        let _ = propagate_differential_trail_rotor(&initial_delta, &rotor_rounds);

        // Benchmark Matrix
        let mut matrix_times = Vec::new();
        for _ in 0..trials {
            let start = Instant::now();
            let _ = propagate_differential_trail_matrix(&initial_delta, &matrix_rounds);
            matrix_times.push(start.elapsed());
        }
        let matrix_mean =
            matrix_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / trials as f64;

        // Benchmark Rotor
        let mut rotor_times = Vec::new();
        for _ in 0..trials {
            let start = Instant::now();
            let _ = propagate_differential_trail_rotor(&initial_delta, &rotor_rounds);
            rotor_times.push(start.elapsed());
        }
        let rotor_mean = rotor_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / trials as f64;

        // Results
        let speedup = matrix_mean / rotor_mean;
        println!("  Matrix: {:.3} µs", matrix_mean * 1_000_000.0);
        println!("  Rotor:  {:.3} µs", rotor_mean * 1_000_000.0);
        println!();
        println!("  SPEEDUP: {:.2}×", speedup);

        if speedup >= 1.5 {
            println!("  ✓ Significant speedup achieved!");
        } else if speedup >= 1.0 {
            println!("  ⚠ Marginal speedup");
        } else {
            println!("  ✗ Rotor still slower");
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("ANALYSIS");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("For O(n) vs O(n²) advantage to appear:");
    println!("  - Need dimension where n² >> n");
    println!("  - Fixed overhead must be amortized");
    println!("  - Typically happens at n > 50-100");
    println!();
}
