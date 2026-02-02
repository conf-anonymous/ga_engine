//! Comprehensive Trail Propagation Benchmark
//!
//! Tests multiple combinations of dimensions and rounds to find sweet spot

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
    println!("║       Comprehensive Trail Propagation Benchmark             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Testing: dimensions × rounds to find optimal speedup");
    println!();

    let configs = vec![
        (8, 1),    (8, 4),    (8, 8),
        (16, 1),   (16, 4),   (16, 8),
        (32, 1),   (32, 4),   (32, 8),
        (64, 1),   (64, 4),   (64, 8),
        (128, 1),  (128, 4),  (128, 8),
    ];

    let trials = 100;

    println!("┌──────┬────────┬───────────┬──────────┬──────────┐");
    println!("│  Dim │ Rounds │  Matrix   │  Rotor   │ Speedup  │");
    println!("├──────┼────────┼───────────┼──────────┼──────────┤");

    let mut best_speedup = 0.0f64;
    let mut best_config = (0, 0);

    for &(dim, num_rounds) in &configs {
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

        let speedup = matrix_mean / rotor_mean;

        if speedup > best_speedup {
            best_speedup = speedup;
            best_config = (dim, num_rounds);
        }

        let marker = if speedup >= 1.5 {
            "✓✓"
        } else if speedup >= 1.2 {
            " ✓"
        } else if speedup >= 1.0 {
            "  "
        } else {
            " ✗"
        };

        println!(
            "│ {:>4} │ {:>6} │ {:>7.2} µs │ {:>6.2} µs │ {:>6.2}× {}│",
            dim,
            num_rounds,
            matrix_mean * 1_000_000.0,
            rotor_mean * 1_000_000.0,
            speedup,
            marker
        );
    }

    println!("└──────┴────────┴───────────┴──────────┴──────────┘");
    println!();
    println!("Best speedup: {:.2}× at dim={}, rounds={}",
             best_speedup, best_config.0, best_config.1);
    println!();

    if best_speedup >= 1.5 {
        println!("[OK] SUCCESS: Found significant speedup (>=1.5x)!");
        println!("   -> Can claim GA provides practical advantage");
        println!("   -> Focus on high-dimensional, multi-round scenarios");
    } else if best_speedup >= 1.2 {
        println!("[OK] MODERATE: Some speedup achieved (1.2-1.5x)");
        println!("   -> Marginal but real advantage");
    } else {
        println!("[--] INSUFFICIENT: No significant speedup found");
        println!("   -> Maximum {:.2}x not compelling", best_speedup);
    }
}
