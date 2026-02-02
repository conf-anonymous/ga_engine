use ga_engine::lattice_reduction::lll_baseline::LLL;

fn main() {
    println!("=== LLL Lattice Reduction Demo ===\n");

    // Example 1: Simple 2D lattice
    println!("Example 1: 2D Lattice");
    let basis_2d = vec![
        vec![19.0, 2.0],
        vec![3.0, 17.0],
    ];

    println!("Original basis:");
    for (i, v) in basis_2d.iter().enumerate() {
        println!("  b{}: {:?}", i, v);
    }

    let mut lll_2d = LLL::new(basis_2d, 0.99);
    lll_2d.reduce();

    println!("\nReduced basis:");
    for (i, v) in lll_2d.get_basis().iter().enumerate() {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("  b{}: {:?} (norm: {:.3})", i, v, norm);
    }

    println!("\n{}", lll_2d);

    // Example 2: 3D lattice
    println!("\n=== Example 2: 3D Lattice ===");
    let basis_3d = vec![
        vec![15.0, 3.0, 1.0],
        vec![2.0, 11.0, 2.0],
        vec![1.0, 1.0, 13.0],
    ];

    println!("Original basis:");
    for (i, v) in basis_3d.iter().enumerate() {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("  b{}: {:?} (norm: {:.3})", i, v, norm);
    }

    let mut lll_3d = LLL::new(basis_3d, 0.99);
    lll_3d.reduce();

    println!("\nReduced basis:");
    for (i, v) in lll_3d.get_basis().iter().enumerate() {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("  b{}: {:?} (norm: {:.3})", i, v, norm);
    }

    println!("\n{}", lll_3d);

    // Example 3: Larger 4D lattice
    println!("\n=== Example 3: 4D Lattice ===");
    let basis_4d = vec![
        vec![12.0, 3.0, 4.0, 1.0],
        vec![2.0, 15.0, 3.0, 2.0],
        vec![3.0, 2.0, 14.0, 3.0],
        vec![1.0, 4.0, 2.0, 16.0],
    ];

    println!("Original basis:");
    for (i, v) in basis_4d.iter().enumerate() {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("  b{}: {:?} (norm: {:.3})", i, v, norm);
    }

    let mut lll_4d = LLL::new(basis_4d, 0.99);
    lll_4d.reduce();

    println!("\nReduced basis:");
    for (i, v) in lll_4d.get_basis().iter().enumerate() {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("  b{}: {:?} (norm: {:.3})", i, v, norm);
    }

    println!("\n{}", lll_4d);
}
