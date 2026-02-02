//! Debug orthogonality defect calculation

use ga_engine::lattice_reduction::ga_lll_rotors::GA_LLL;

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn main() {
    // Simple 2D basis
    let basis = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
    ];

    let ga_lll = GA_LLL::new(basis.clone(), 0.99);

    println!("Perfect orthogonal basis (identity):");
    println!("Basis:");
    for (i, b) in ga_lll.get_basis().iter().enumerate() {
        println!("  b{}: {:?}, norm = {:.6}", i, b, norm(b));
    }
    println!();

    let hf = ga_lll.hermite_factor();
    let defect = ga_lll.orthogonality_defect();

    println!("Hermite factor: {:.6}", hf);
    println!("Orthogonality defect: {:.6}", defect);
    println!("Expected defect: 1.0 (perfect orthogonal basis)");
}
