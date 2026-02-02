use ga_engine::lattice_reduction::lll_baseline::LLL;
use std::time::Instant;

fn main() {
    println!("=== LLL Synthetic Tests ===\n");

    // Test 1: Simple 10x10 lattice with moderate entries
    println!("Test 1: 10x10 random lattice");
    let basis10: Vec<Vec<f64>> = (0..10)
        .map(|i| {
            (0..10)
                .map(|j| {
                    if i == j {
                        1000.0
                    } else {
                        ((i * 7 + j * 13) % 100) as f64
                    }
                })
                .collect()
        })
        .collect();

    let start = Instant::now();
    let mut lll10 = LLL::new(basis10, 0.99);
    lll10.reduce();
    println!("  Time: {:?}", start.elapsed());
    println!("  Hermite factor: {:.6}", lll10.hermite_factor());
    let stats = lll10.get_stats();
    println!("  Stats: {} swaps, {} size_reductions\n", stats.swaps, stats.size_reductions);

    // Test 2: 20x20
    println!("Test 2: 20x20 random lattice");
    let basis20: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            (0..20)
                .map(|j| {
                    if i == j {
                        10000.0
                    } else {
                        ((i * 7 + j * 13) % 200) as f64
                    }
                })
                .collect()
        })
        .collect();

    let start = Instant::now();
    let mut lll20 = LLL::new(basis20, 0.99);
    lll20.reduce();
    println!("  Time: {:?}", start.elapsed());
    println!("  Hermite factor: {:.6}", lll20.hermite_factor());
    let stats = lll20.get_stats();
    println!("  Stats: {} swaps, {} size_reductions\n", stats.swaps, stats.size_reductions);

    // Test 3: 30x30
    println!("Test 3: 30x30 random lattice");
    let basis30: Vec<Vec<f64>> = (0..30)
        .map(|i| {
            (0..30)
                .map(|j| {
                    if i == j {
                        100000.0
                    } else {
                        ((i * 7 + j * 13) % 500) as f64
                    }
                })
                .collect()
        })
        .collect();

    let start = Instant::now();
    let mut lll30 = LLL::new(basis30, 0.99);
    lll30.reduce();
    println!("  Time: {:?}", start.elapsed());
    println!("  Hermite factor: {:.6}", lll30.hermite_factor());
    let stats = lll30.get_stats();
    println!("  Stats: {} swaps, {} size_reductions\n", stats.swaps, stats.size_reductions);

    println!("✓ All synthetic tests completed successfully");
}
