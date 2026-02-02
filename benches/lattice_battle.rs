// benches/lattice_battle.rs
//! Comprehensive benchmark suite for LLL/BKZ baseline vs GA-accelerated variants
//!
//! We compare apples-to-apples: same hardware, same language, same inputs.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ga_engine::lattice_reduction::lll_baseline::LLL;
use ga_engine::lattice_reduction::svp_challenge;
use std::time::Duration;

/// Generate well-conditioned random lattice basis for testing
/// Uses diagonal dominance to avoid singular matrices
fn generate_random_basis(dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut basis = Vec::new();
    for i in 0..dim {
        let mut row = vec![0.0; dim];
        // Diagonal dominance ensures non-singular matrix
        row[i] = 100.0;
        // Add perturbations
        for j in 0..dim {
            let perturbation = ((seed + (i * dim + j) as u64) % 50) as f64;
            row[j] += perturbation;
        }
        basis.push(row);
    }
    basis
}

/// Benchmark standard LLL on synthetic lattices
fn bench_lll_baseline_synthetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("LLL_Baseline_Synthetic");
    group.measurement_time(Duration::from_secs(30));  // Allow longer benchmarks

    for dim in [10, 20, 30, 40, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let basis = generate_random_basis(dim, 42);
            b.iter(|| {
                let mut lll = LLL::new(basis.clone(), 0.99);
                lll.reduce();
                black_box(lll.hermite_factor())
            });
        });
    }
    group.finish();
}

/// Benchmark standard LLL on SVP Challenge instances (if available)
fn bench_lll_baseline_svp(c: &mut Criterion) {
    let mut group = c.benchmark_group("LLL_Baseline_SVP_Challenge");
    group.measurement_time(Duration::from_secs(60));  // SVP instances take longer
    group.sample_size(10);  // Reduce sample size for expensive benchmarks

    // Try to load SVP Challenge lattices
    let dimensions = vec![40];  // Start with dim 40

    for dim in dimensions {
        let filename = format!("data/lattices/svpchallengedim{}seed0.txt", dim);

        // Check if file exists
        if std::path::Path::new(&filename).exists() {
            match svp_challenge::parse_lattice_file(&filename) {
                Ok(basis) => {
                    group.bench_with_input(
                        BenchmarkId::from_parameter(dim),
                        &dim,
                        |b, _| {
                            b.iter(|| {
                                let mut lll = LLL::new(basis.clone(), 0.99);
                                lll.reduce();
                                black_box(lll.hermite_factor())
                            });
                        }
                    );
                }
                Err(e) => {
                    eprintln!("Warning: Could not load SVP Challenge dim {}: {}", dim, e);
                }
            }
        } else {
            eprintln!("Warning: SVP Challenge file not found: {}", filename);
        }
    }

    group.finish();
}

/// Benchmark LLL statistics collection
fn bench_lll_with_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("LLL_Statistics");

    for dim in [20, 30, 40].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let basis = generate_random_basis(dim, 42);
            b.iter(|| {
                let mut lll = LLL::new(basis.clone(), 0.99);
                lll.reduce();
                let stats = lll.get_stats();
                black_box((stats.swaps, stats.size_reductions, stats.gso_updates))
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_lll_baseline_synthetic,
    bench_lll_baseline_svp,
    bench_lll_with_stats
);
criterion_main!(benches);
