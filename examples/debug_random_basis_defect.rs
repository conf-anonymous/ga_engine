//! Debug: Why does random basis have huge orthogonality defect?

use ga_engine::lattice_reduction::ga_lll_rotors::GA_LLL;
use ga_engine::lattice_reduction::lll_baseline::LLL;

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn generate_random_basis(dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut basis = Vec::new();
    for i in 0..dim {
        let mut row = Vec::new();
        for j in 0..dim {
            let val = ((seed + (i * dim + j) as u64) % 1000) as f64;
            row.push(val);
        }
        basis.push(row);
    }
    basis
}

fn main() {
    let basis = generate_random_basis(10, 42);

    println!("Original random basis:");
    for (i, b) in basis.iter().enumerate() {
        println!("  b{}: norm = {:.2e}", i, norm(b));
    }
    println!();

    // Before reduction
    let ga_lll_before = GA_LLL::new(basis.clone(), 0.99);
    println!("BEFORE LLL reduction:");
    println!("  Orthogonality defect: {:.2e}", ga_lll_before.orthogonality_defect());
    println!();

    // After reduction
    let mut lll = LLL::new(basis.clone(), 0.99);
    lll.reduce();
    let reduced_basis = lll.get_basis().to_vec();

    let ga_lll_after = GA_LLL::new(reduced_basis.clone(), 0.99);
    println!("AFTER LLL reduction:");
    for (i, b) in ga_lll_after.get_basis().iter().enumerate() {
        println!("  b{}: norm = {:.2e}", i, norm(b));
    }
    println!();
    println!("  Orthogonality defect: {:.2e}", ga_lll_after.orthogonality_defect());
    println!("  Hermite factor: {:.6}", ga_lll_after.hermite_factor());
    println!();

    println!("Interpretation:");
    println!("- Random basis before LLL: extremely ill-conditioned (huge defect)");
    println!("- After LLL: more orthogonal (lower defect)");
    println!("- Defect still > 1 because LLL doesn't produce perfect orthogonal basis");
}
