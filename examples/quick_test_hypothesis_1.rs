//! Quick test of Hypothesis 1 - just dimension 10

use ga_engine::lattice_reduction::ga_lll_rotors::GA_LLL;
use ga_engine::lattice_reduction::lll_baseline::LLL;

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

fn main() {
    println!("Quick Hypothesis 1 Test - Dimension 10");
    println!("=====================================\n");

    let basis = generate_random_basis(10, 42);

    // Standard LLL
    let mut lll = LLL::new(basis.clone(), 0.99);
    lll.reduce();
    let lll_hf = lll.hermite_factor();
    println!("Standard LLL:");
    println!("  Hermite factor: {:.6}", lll_hf);
    println!("  First vector norm: {:.6}", norm(&lll.get_basis()[0]));
    println!();

    // GA-LLL
    let mut ga_lll = GA_LLL::new(basis.clone(), 0.99);
    ga_lll.reduce();
    let ga_hf = ga_lll.hermite_factor();
    let ga_defect = ga_lll.orthogonality_defect();
    println!("GA-LLL:");
    println!("  Hermite factor: {:.6}", ga_hf);
    println!("  First vector norm: {:.6}", norm(&ga_lll.get_basis()[0]));
    println!("  Orthogonality defect: {:.6}", ga_defect);
    println!();

    println!("Orthogonality defect interpretation:");
    if ga_defect < 1.1 {
        println!("  ✓ EXCELLENT: Near-perfect orthogonality");
    } else if ga_defect < 2.0 {
        println!("  ✓ GOOD: Reasonable orthogonality");
    } else if ga_defect < 10.0 {
        println!("  ~ FAIR: Some numerical error");
    } else {
        println!("  ✗ POOR: Significant numerical instability");
    }
}
