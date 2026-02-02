//! Test dimension 40 to see if there's an issue

use ga_engine::lattice_reduction::lll_baseline::LLL;
use std::time::Instant;

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

fn main() {
    println!("Testing LLL on dimension 40...");

    let basis = generate_random_basis(40, 42);

    println!("Basis generated. Running LLL...");
    let start = Instant::now();

    let mut lll = LLL::new(basis, 0.99);
    println!("LLL created. Reducing...");

    lll.reduce();

    let elapsed = start.elapsed();

    println!("LLL reduction completed!");
    println!("Time: {:?}", elapsed);
    println!("Hermite factor: {:.6}", lll.hermite_factor());

    let stats = lll.get_stats();
    println!("Swaps: {}", stats.swaps);
    println!("Size reductions: {}", stats.size_reductions);
}
