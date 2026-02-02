use ga_engine::lattice_reduction::svp_challenge;

fn main() {
    println!("=== SVP Challenge Parser Test ===\n");

    // Test parsing all files without running LLL
    let dimensions = vec![40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140];

    for dim in dimensions {
        let filename = format!("data/lattices/svpchallengedim{}seed0.txt", dim);

        match svp_challenge::parse_lattice_file(&filename) {
            Ok(basis) => {
                println!("✓ Dimension {}: Parsed {} x {} lattice",
                         dim, basis.len(), basis[0].len());

                // Find shortest vector in original basis
                let (idx, norm) = svp_challenge::find_shortest_vector(&basis);
                println!("  Shortest vector: b{} with norm {:.6e}", idx, norm);

                // Sanity check: verify it's a q-ary lattice structure
                // (first column has huge values, rest mostly identity)
                let first_val = basis[0][0];
                println!("  First entry: {:.3e}", first_val);
                println!();
            }
            Err(e) => {
                println!("✗ Dimension {}: Failed to parse - {}\n", dim, e);
            }
        }
    }
}
