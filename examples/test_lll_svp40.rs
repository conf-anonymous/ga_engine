use ga_engine::lattice_reduction::svp_challenge;
use ga_engine::lattice_reduction::lll_baseline::LLL;
use std::time::Instant;

fn main() {
    println!("=== LLL on SVP Challenge Dimension 40 ===\n");

    let filename = "data/lattices/svpchallengedim40seed0.txt";

    match svp_challenge::parse_lattice_file(filename) {
        Ok(basis) => {
            println!("✓ Parsed {} x {} lattice", basis.len(), basis[0].len());

            // Find shortest vector in original basis
            let (idx, orig_norm) = svp_challenge::find_shortest_vector(&basis);
            println!("Original shortest: b{} with norm {:.6e}", idx, orig_norm);

            // Check for numerical stability
            let max_entry = basis.iter()
                .flat_map(|row| row.iter())
                .map(|x| x.abs())
                .fold(0.0, f64::max);
            println!("Max entry magnitude: {:.3e}", max_entry);

            if max_entry.is_infinite() {
                println!("\n⚠️  Warning: Lattice contains inf values - f64 overflow!");
                println!("Need arbitrary precision for this instance.");
                return;
            }

            // Run LLL with progress reporting
            println!("\nRunning LLL reduction (δ=0.99)...");
            println!("This may take several minutes for dimension 40...");

            let start = Instant::now();
            let mut lll = LLL::new(basis, 0.99);

            // Run reduction
            lll.reduce();

            let duration = start.elapsed();
            println!("✓ LLL completed in {:.2?}", duration);

            // Find shortest after reduction
            let reduced = lll.get_basis();
            let (idx, reduced_norm) = svp_challenge::find_shortest_vector(reduced);
            println!("\nResults:");
            println!("  Original shortest: {:.6e}", orig_norm);
            println!("  Reduced shortest:  {:.6e}", reduced_norm);
            println!("  Improvement:       {:.2}x", orig_norm / reduced_norm);
            println!("  Hermite factor:    {:.6}", lll.hermite_factor());

            // Statistics
            let stats = lll.get_stats();
            println!("\nStatistics:");
            println!("  Swaps:           {}", stats.swaps);
            println!("  Size reductions: {}", stats.size_reductions);
            println!("  GSO updates:     {}", stats.gso_updates);
        }
        Err(e) => {
            println!("✗ Failed to parse {}: {}", filename, e);
        }
    }
}
