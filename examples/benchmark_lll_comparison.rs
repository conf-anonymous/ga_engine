//! Benchmark: Standard LLL vs GA-LLL
//!
//! Compares performance and quality of standard LLL vs GA-accelerated LLL
//! on synthetic lattices of varying dimensions.

use ga_engine::lattice_reduction::lll_baseline::LLL;
use ga_engine::lattice_reduction::ga_lll::GA_LLL;
use std::time::Instant;

fn main() {
    println!("=== LLL Comparison Benchmark ===\n");
    println!("Comparing Standard LLL vs GA-LLL\n");
    println!("{:-<80}", "");

    // Test dimensions
    let dimensions = vec![5, 10, 15, 20, 30];

    for dim in dimensions {
        println!("\n### Dimension {} ###\n", dim);

        // Test 1: Random lattice
        println!("Test 1: Random integer lattice");
        test_random_lattice(dim);

        // Test 2: Diagonal + perturbations
        println!("\nTest 2: Diagonal with perturbations");
        test_diagonal_lattice(dim);

        println!("\n{:-<80}", "");
    }
}

fn test_random_lattice(dim: usize) {
    // Generate random integer lattice
    let basis = generate_random_lattice(dim, -100, 100);

    // Standard LLL
    let mut lll = LLL::new(basis.clone(), 0.99);
    let lll_start = Instant::now();
    lll.reduce();
    let lll_time = lll_start.elapsed();
    let lll_basis = lll.get_basis();
    let lll_hf = lll.hermite_factor();
    let lll_stats = lll.get_stats();
    let lll_swaps = lll_stats.swaps;

    // GA-LLL
    let mut ga_lll = GA_LLL::new(basis.clone(), 0.99);
    let ga_start = Instant::now();
    ga_lll.reduce();
    let ga_time = ga_start.elapsed();
    let ga_basis = ga_lll.get_basis();
    let ga_hf = ga_lll.hermite_factor();
    let ga_stats = ga_lll.get_stats();

    // Compare results
    println!("  Standard LLL:");
    println!("    Time: {:?}", lll_time);
    println!("    Hermite factor: {:.6}", lll_hf);
    println!("    First vector norm: {:.6}", norm(&lll_basis[0]));
    println!("    Swaps: {}", lll_swaps);

    println!("  GA-LLL:");
    println!("    Time: {:?}", ga_time);
    println!("    Hermite factor: {:.6}", ga_hf);
    println!("    First vector norm: {:.6}", norm(&ga_basis[0]));
    println!("    Swaps: {}", ga_stats.swaps);
    println!("    Rotor computations: {}", ga_stats.rotor_computations);
    println!("    Rotor time: {:?}", ga_stats.rotor_time);
    println!("    Rotor apply time: {:?}", ga_stats.rotor_apply_time);

    println!("  Comparison:");
    let speedup = lll_time.as_secs_f64() / ga_time.as_secs_f64();
    if speedup > 1.0 {
        println!("    GA-LLL is {:.2}x FASTER", speedup);
    } else {
        println!("    Standard LLL is {:.2}x faster", 1.0 / speedup);
    }

    let quality_diff = ((ga_hf - lll_hf) / lll_hf * 100.0).abs();
    println!("    Quality difference: {:.2}%", quality_diff);

    if quality_diff < 1.0 {
        println!("    ✓ Comparable quality");
    }
}

fn test_diagonal_lattice(dim: usize) {
    // Generate diagonal lattice with perturbations
    let mut basis = Vec::new();
    for i in 0..dim {
        let mut v = vec![0.0; dim];
        v[i] = 50.0 + (i as f64 * 2.0);
        // Add perturbations
        if i > 0 {
            v[i - 1] = 5.0;
        }
        if i + 1 < dim {
            v[i + 1] = 3.0;
        }
        basis.push(v);
    }

    // Standard LLL
    let mut lll = LLL::new(basis.clone(), 0.99);
    let lll_start = Instant::now();
    lll.reduce();
    let lll_time = lll_start.elapsed();
    let lll_hf = lll.hermite_factor();

    // GA-LLL
    let mut ga_lll = GA_LLL::new(basis.clone(), 0.99);
    let ga_start = Instant::now();
    ga_lll.reduce();
    let ga_time = ga_start.elapsed();
    let ga_hf = ga_lll.hermite_factor();

    println!("  Standard LLL: {:?}, HF={:.6}", lll_time, lll_hf);
    println!("  GA-LLL: {:?}, HF={:.6}", ga_time, ga_hf);

    let speedup = lll_time.as_secs_f64() / ga_time.as_secs_f64();
    if speedup > 1.0 {
        println!("  GA-LLL is {:.2}x faster", speedup);
    } else {
        println!("  Standard LLL is {:.2}x faster", 1.0 / speedup);
    }
}

fn generate_random_lattice(dim: usize, min: i32, max: i32) -> Vec<Vec<f64>> {
    // Simple deterministic "random" lattice using index-based values
    let mut basis = Vec::new();
    for i in 0..dim {
        let mut v = Vec::new();
        for j in 0..dim {
            // Pseudo-random value based on indices
            let val = ((i * 37 + j * 17) % (max - min) as usize) as i32 + min;
            v.push(val as f64);
        }
        basis.push(v);
    }
    basis
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
