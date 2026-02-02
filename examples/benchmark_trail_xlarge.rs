//! Benchmark: Trail Propagation - Very Large Dimensions
//!
//! Tests dimensions 128, 256, 512 to see if speedup increases

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
    println!("║        Trail Propagation: XLarge Dimensions                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let dimensions = vec![128, 256, 512];
    let num_rounds = 2; // Fewer rounds for very large
    let trials = 20;    // Fewer trials

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
        println!("  Matrix: {:.3} ms", matrix_mean * 1000.0);
        println!("  Rotor:  {:.3} ms", rotor_mean * 1000.0);
        println!();
        println!("  SPEEDUP: {:.2}×", speedup);

        if speedup >= 2.0 {
            println!("  ✓✓ EXCELLENT: ≥2× target achieved!");
        } else if speedup >= 1.5 {
            println!("  ✓ GOOD: Significant speedup!");
        } else if speedup >= 1.2 {
            println!("  ⚠ MODERATE: Some speedup");
        } else if speedup >= 1.0 {
            println!("  ⚠ MARGINAL: Barely faster");
        } else {
            println!("  ✗ SLOWER");
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("CONCLUSION");
    println!("===========================================================");
    println!();
    println!("If speedup continues to increase with dimension:");
    println!("  -> GA rotors ARE faster for large-scale problems");
    println!("  -> Focus: GA for high-dimensional cryptanalysis");
    println!();
    println!("Typical cipher dimensions:");
    println!("  - Block ciphers: 64-256 bits (small)");
    println!("  - Hash functions: 256-1024 bits (medium-large)");
    println!("  - Lattice crypto: 512-2048 dimensions (large)");
    println!();
}
