use ga_engine::lattice_reduction::svp_challenge;
use ga_engine::lattice_reduction::lll_baseline::LLL;

fn main() {
    println!("=== SVP Challenge Parser & LLL Demo ===\n");

    // Test on different dimensions
    let dimensions = vec![40, 50, 60, 70];

    for dim in dimensions {
        println!("=== Dimension {} ===", dim);
        let filename = format!("data/lattices/svpchallengedim{}seed0.txt", dim);

        match svp_challenge::parse_lattice_file(&filename) {
            Ok(basis) => {
                println!("✓ Parsed {} x {} lattice", basis.len(), basis[0].len());

                // Find shortest vector in original basis
                let (idx, orig_norm) = svp_challenge::find_shortest_vector(&basis);
                println!("  Original shortest vector: b{} with norm {:.6e}", idx, orig_norm);

                // Run LLL
                println!("  Running LLL reduction...");
                let mut lll = LLL::new(basis, 0.99);
                lll.reduce();

                // Find shortest after reduction
                let reduced = lll.get_basis();
                let (idx, reduced_norm) = svp_challenge::find_shortest_vector(reduced);
                println!("  Reduced shortest vector: b{} with norm {:.6e}", idx, reduced_norm);

                // Improvement ratio
                let ratio = orig_norm / reduced_norm;
                println!("  Improvement: {:.2}x shorter", ratio);

                // Statistics
                let stats = lll.get_stats();
                println!("  Stats: {} swaps, {} size reductions, {} GSO updates",
                         stats.swaps, stats.size_reductions, stats.gso_updates);
                println!("  Hermite factor: {:.6}", lll.hermite_factor());
                println!();
            }
            Err(e) => {
                println!("✗ Failed to parse {}: {}", filename, e);
                println!("  (File might not exist - run from repo root)\n");
            }
        }
    }
}
