//! E8 Orbit Enumeration - Final Comparison
//!
//! Matrix vs GA Optimized Reflection
//!
//! This is the definitive benchmark to see if GA helps for E8 symmetry operations.

use ga_engine::lattice_reduction::e8_lattice::{E8Lattice, reflect_vector, norm_squared};
use ga_engine::lattice_reduction::ga_reflection::reflect_e8_optimized;
use std::collections::HashSet;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct FloatVec([f64; 8]);

impl PartialEq for FloatVec {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(a, b)| (a - b).abs() < 1e-10)
    }
}

impl Eq for FloatVec {}

impl std::hash::Hash for FloatVec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for &x in &self.0 {
            let rounded = (x * 1e10).round() as i64;
            rounded.hash(state);
        }
    }
}

fn enumerate_orbit_matrix(v: &[f64; 8], simple_roots: &[[f64; 8]], max: usize) -> HashSet<FloatVec> {
    let mut orbit = HashSet::new();
    orbit.insert(FloatVec(*v));
    let mut prev_size = 0;

    while orbit.len() != prev_size && orbit.len() < max {
        prev_size = orbit.len();
        let current: Vec<[f64; 8]> = orbit.iter().map(|fv| fv.0).collect();
        for vec in current {
            for root in simple_roots {
                orbit.insert(FloatVec(reflect_vector(&vec, root)));
                if orbit.len() >= max { break; }
            }
        }
    }
    orbit
}

fn enumerate_orbit_ga(v: &[f64; 8], simple_roots: &[[f64; 8]], max: usize) -> HashSet<FloatVec> {
    let mut orbit = HashSet::new();
    orbit.insert(FloatVec(*v));
    let mut prev_size = 0;

    while orbit.len() != prev_size && orbit.len() < max {
        prev_size = orbit.len();
        let current: Vec<[f64; 8]> = orbit.iter().map(|fv| fv.0).collect();
        for vec in current {
            for root in simple_roots {
                orbit.insert(FloatVec(reflect_e8_optimized(&vec, root)));
                if orbit.len() >= max { break; }
            }
        }
    }
    orbit
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  E8 Orbit Enumeration - Matrix vs GA Comparison         ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let e8 = E8Lattice::new();

    // The canonical test: e₁+e₂ generates the full 240-element root system
    let test_vec = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    println!("Test: Generate full E8 root system (240 vectors)");
    println!("Starting vector: e₁+e₂ = [1.0, 1.0, 0.0, ...]");
    println!("Method: Apply 8 simple reflections iteratively");
    println!();

    let trials = 10;
    let max_orbit = 1000;

    // Warm up
    println!("Warming up...");
    let _ = enumerate_orbit_matrix(&test_vec, e8.simple_roots(), max_orbit);
    let _ = enumerate_orbit_ga(&test_vec, e8.simple_roots(), max_orbit);
    println!("Warm-up complete.");
    println!();

    // Benchmark matrix approach
    println!("Benchmarking Matrix Reflection ({} trials)...", trials);
    let mut matrix_times = Vec::new();
    for _ in 0..trials {
        let start = Instant::now();
        let orbit = enumerate_orbit_matrix(&test_vec, e8.simple_roots(), max_orbit);
        let elapsed = start.elapsed();
        matrix_times.push(elapsed);
        assert_eq!(orbit.len(), 240, "Should generate 240 roots");
    }

    let matrix_mean = matrix_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / trials as f64;
    let matrix_min = matrix_times.iter().map(|t| t.as_secs_f64()).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    println!("  Mean time: {:.3} ms", matrix_mean * 1000.0);
    println!("  Min time:  {:.3} ms", matrix_min * 1000.0);
    println!();

    // Benchmark GA approach
    println!("Benchmarking GA Optimized Reflection ({} trials)...", trials);
    let mut ga_times = Vec::new();
    for _ in 0..trials {
        let start = Instant::now();
        let orbit = enumerate_orbit_ga(&test_vec, e8.simple_roots(), max_orbit);
        let elapsed = start.elapsed();
        ga_times.push(elapsed);
        assert_eq!(orbit.len(), 240, "Should generate 240 roots");
    }

    let ga_mean = ga_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / trials as f64;
    let ga_min = ga_times.iter().map(|t| t.as_secs_f64()).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    println!("  Mean time: {:.3} ms", ga_mean * 1000.0);
    println!("  Min time:  {:.3} ms", ga_min * 1000.0);
    println!();

    // Results
    println!("═══════════════════════════════════════════════════════════");
    println!("RESULTS");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("┌────────────────┬──────────────┬──────────────┐");
    println!("│ Method         │ Mean Time    │ Min Time     │");
    println!("├────────────────┼──────────────┼──────────────┤");
    println!("│ Matrix         │ {:>11.3} │ {:>11.3} │",
             format!("{:.3} ms", matrix_mean * 1000.0),
             format!("{:.3} ms", matrix_min * 1000.0));
    println!("│ GA Optimized   │ {:>11.3} │ {:>11.3} │",
             format!("{:.3} ms", ga_mean * 1000.0),
             format!("{:.3} ms", ga_min * 1000.0));
    println!("└────────────────┴──────────────┴──────────────┘");
    println!();

    let speedup_mean = matrix_mean / ga_mean;
    let speedup_min = matrix_min / ga_min;

    println!("Speedup:");
    println!("  Mean: {:.3}×", speedup_mean);
    println!("  Min:  {:.3}×", speedup_min);
    println!();

    println!("Verdict:");
    if speedup_mean >= 1.2 {
        println!("  ✅ GA wins! Significant speedup (≥1.2×)");
    } else if speedup_mean >= 1.05 {
        println!("  ✓ GA wins marginally (5-20% faster)");
    } else if speedup_mean >= 0.95 {
        println!("  ~ Roughly equal (±5%)");
    } else {
        println!("  ✗ Matrix wins (GA slower)");
    }
    println!();

    println!("Note: Current 'GA' implementation is actually just optimized");
    println!("      reflection formula (v' = v - ⟨v,α⟩α) exploiting ||α||²=2.");
    println!("      Not using true GA geometric product yet.");
}
