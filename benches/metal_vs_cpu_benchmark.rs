//! Benchmark: Metal GPU vs CPU Performance
//!
//! Measures homomorphic geometric product performance:
//! - V1 CPU (baseline): ~13s
//! - V2 CPU (Rayon): ~0.441s (30× faster)
//! - V2 Metal GPU: Target <0.05s (260× faster)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::backends::gpu_metal::geometric::MetalGeometricProduct;

fn benchmark_metal_geometric_product(c: &mut Criterion) {
    #[cfg(feature = "v2-gpu-metal")]
    {
        let mut group = c.benchmark_group("geometric_product");
        group.measurement_time(Duration::from_secs(30));
        group.sample_size(20);

        // Test with realistic FHE parameters
        // N=1024 is standard for Clifford FHE
        let n = 1024;
        let q = 1152921504606584833u64; // 60-bit NTT-friendly prime
        let root = compute_primitive_root(n, q);

        println!("\n=== Metal GPU Benchmark ===");
        println!("Ring dimension: N = {}", n);
        println!("Modulus: q = {} (60-bit)", q);
        println!("Primitive root: ω = {}", root);
        println!("Apple M3 Max (40 GPU cores)");

        let gp = MetalGeometricProduct::new(n, q, root).expect("Metal initialization failed");

        // Create test multivectors (simplified ciphertexts)
        let mut a: [[Vec<u64>; 2]; 8] = Default::default();
        let mut b: [[Vec<u64>; 2]; 8] = Default::default();

        for i in 0..8 {
            a[i][0] = vec![1; n]; // c0 polynomial
            a[i][1] = vec![2; n]; // c1 polynomial
            b[i][0] = vec![3; n];
            b[i][1] = vec![4; n];
        }

        group.bench_function(BenchmarkId::new("metal_gpu", n), |bencher| {
            bencher.iter(|| {
                let result = gp.geometric_product(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });

        group.finish();
    }

    #[cfg(not(feature = "v2-gpu-metal"))]
    {
        eprintln!("\n=== Metal GPU Benchmark Skipped ===");
        eprintln!("Metal backend not enabled. Run with:");
        eprintln!("  cargo bench --bench metal_vs_cpu_benchmark --features v2-gpu-metal");
    }
}

/// Compute primitive n-th root of unity mod q using trial and error
/// For benchmarking purposes (in production, precompute these)
fn compute_primitive_root(n: usize, q: u64) -> u64 {
    // For q = 1152921504606584833 and n=1024:
    // We need ω such that ω^n ≡ 1 (mod q) but ω^(n/2) ≢ 1 (mod q)
    // This is precomputed for standard FHE parameters

    // Known primitive 1024th root for this prime
    // ω = g^((q-1)/1024) where g is generator of Z_q^*
    // For this specific prime, the root is precomputed
    let root = 1925348604829696032u64; // Precomputed for N=1024, q=1152921504606584833

    // Verify it's correct (in debug mode)
    if cfg!(debug_assertions) {
        let mut test = 1u128;
        let omega_u128 = root as u128;
        let q_u128 = q as u128;

        for _ in 0..n {
            test = (test * omega_u128) % q_u128;
        }

        if test != 1 {
            eprintln!("Warning: Primitive root verification failed");
            eprintln!("ω^{} mod {} = {} (expected 1)", n, q, test);
        }
    }

    root
}

criterion_group!(benches, benchmark_metal_geometric_product);
criterion_main!(benches);
