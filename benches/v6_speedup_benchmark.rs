//! V6 Speedup Benchmark - parallel_lift GPU Acceleration
//!
//! Run with: cargo bench --bench v6_speedup_benchmark --features v6-cuda
//!
//! Benchmarks V6's parallel_lift acceleration vs V2's standard GPU implementation:
//! - Gadget decomposition: 25× expected speedup
//! - Relinearization (full operation): 10-25× expected speedup
//! - Rotation key switching: 25× expected speedup
//!
//! Note: Requires CUDA-capable GPU

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "v6-cuda")]
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

#[cfg(feature = "v6-cuda")]
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::{
    ckks::CudaCkksContext,
    relin_keys::CudaRelinKeys,
    rotation_keys::CudaRotationKeys,
    rotation::CudaRotationContext,
};

#[cfg(feature = "v6-cuda")]
use ga_engine::clifford_fhe_v6::{
    ParallelLiftContext,
    V6RelinKeys,
    V6RotationKeys,
    gpu_gadget_decompose_v6,
};

const BENCHMARK_DURATION_SECS: u64 = 10;
const BENCHMARK_SAMPLE_SIZE: usize = 50;

/// Generate test polynomial in flat RNS layout
#[cfg(feature = "v6-cuda")]
fn generate_test_poly(n: usize, num_primes: usize, moduli: &[u64]) -> Vec<u64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut poly = vec![0u64; n * num_primes];

    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        for coeff_idx in 0..n {
            // Deterministic pseudo-random coefficients
            let mut hasher = DefaultHasher::new();
            (prime_idx, coeff_idx).hash(&mut hasher);
            let hash = hasher.finish();
            poly[prime_idx * n + coeff_idx] = hash % q;
        }
    }

    poly
}

/// Generate test secret key in strided layout
#[cfg(feature = "v6-cuda")]
fn generate_test_secret_key(n: usize, num_primes: usize, moduli: &[u64]) -> Vec<u64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Secret key in strided layout: sk[coeff_idx * num_primes + prime_idx]
    let mut sk = vec![0u64; n * num_primes];

    for coeff_idx in 0..n {
        let mut hasher = DefaultHasher::new();
        ("sk", coeff_idx).hash(&mut hasher);
        let hash = hasher.finish();

        // Ternary distribution: -1, 0, 1
        let choice = hash % 3;

        for prime_idx in 0..num_primes {
            let q = moduli[prime_idx];
            sk[coeff_idx * num_primes + prime_idx] = match choice {
                0 => 0,
                1 => 1,
                2 => q - 1, // -1 mod q
                _ => unreachable!(),
            };
        }
    }

    sk
}

#[cfg(feature = "v6-cuda")]
fn bench_gadget_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("V6/GadgetDecomposition");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    // Test with realistic parameters
    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len().min(4); // Use 4 primes for test
    let base_bits = 16u32;
    let moduli = &params.moduli[..num_primes];

    // Initialize V6 context (includes parallel_lift FheGpuContext)
    let ckks_ctx = Arc::new(
        CudaCkksContext::new(params.clone())
            .expect("Failed to create CudaCkksContext")
    );

    let ctx = ParallelLiftContext::new(params.clone(), ckks_ctx.clone())
        .expect("Failed to create ParallelLiftContext");

    // Generate test polynomial
    let poly = generate_test_poly(n, num_primes, moduli);

    println!("\n=== Gadget Decomposition Benchmark ===");
    println!("N = {}, num_primes = {}, base_bits = {}", n, num_primes, base_bits);

    // Benchmark V6 GPU gadget decomposition
    group.bench_with_input(
        BenchmarkId::new("V6-GPU", format!("N={},L={}", n, num_primes)),
        &(&ctx, &poly, num_primes, base_bits),
        |b, (ctx, poly, num_primes, base_bits)| {
            b.iter(|| {
                black_box(gpu_gadget_decompose_v6(ctx, poly, *num_primes, *base_bits))
            })
        },
    );

    // Note: V2's gadget_decompose is internal to CudaRelinKeys
    // We benchmark the full relinearization to capture the difference

    group.finish();
}

#[cfg(feature = "v6-cuda")]
fn bench_relinearization(c: &mut Criterion) {
    let mut group = c.benchmark_group("V6/Relinearization");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len().min(4);
    let base_bits = 16usize;
    let moduli = &params.moduli[..num_primes];
    let level = num_primes - 1;

    // Initialize contexts
    let ckks_ctx = Arc::new(
        CudaCkksContext::new(params.clone())
            .expect("Failed to create CudaCkksContext")
    );

    let ctx = Arc::new(
        ParallelLiftContext::new(params.clone(), ckks_ctx.clone())
            .expect("Failed to create ParallelLiftContext")
    );

    // Generate secret key and test polynomials
    let secret_key = generate_test_secret_key(n, num_primes, moduli);
    let c0 = generate_test_poly(n, num_primes, moduli);
    let c1 = generate_test_poly(n, num_primes, moduli);
    let c2 = generate_test_poly(n, num_primes, moduli);

    // Create V2 relinearization keys
    let ntt_contexts = ckks_ctx.ntt_contexts();
    let device = ckks_ctx.device().clone();

    let v2_relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        secret_key.clone(),
        base_bits,
        ntt_contexts,
    ).expect("Failed to create V2 relin keys");

    // Create V6 relinearization keys
    let v6_relin_keys = V6RelinKeys::new(ctx.clone(), &secret_key, base_bits)
        .expect("Failed to create V6 relin keys");

    println!("\n=== Relinearization Benchmark ===");
    println!("N = {}, level = {}, base_bits = {}", n, level, base_bits);

    // Benchmark V2 relinearization (standard GPU)
    group.bench_with_input(
        BenchmarkId::new("V2-GPU", format!("N={},L={}", n, level)),
        &(&v2_relin_keys, &c0, &c1, &c2, level, ntt_contexts, &ckks_ctx),
        |b, (rk, c0, c1, c2, level, ntt_ctx, ckks_ctx)| {
            b.iter(|| {
                black_box(rk.apply_relinearization_gpu(c0, c1, c2, *level, ntt_ctx, ckks_ctx))
            })
        },
    );

    // Benchmark V6 relinearization (parallel_lift accelerated)
    group.bench_with_input(
        BenchmarkId::new("V6-GPU", format!("N={},L={}", n, level)),
        &(&v6_relin_keys, &c0, &c1, &c2, level, ntt_contexts, &ckks_ctx),
        |b, (rk, c0, c1, c2, level, ntt_ctx, ckks_ctx)| {
            b.iter(|| {
                black_box(rk.apply_relinearization_v6(c0, c1, c2, *level, ntt_ctx, ckks_ctx))
            })
        },
    );

    group.finish();
}

#[cfg(feature = "v6-cuda")]
fn bench_rotation(c: &mut Criterion) {
    let mut group = c.benchmark_group("V6/Rotation");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len().min(4);
    let base_bits = 16usize;
    let moduli = &params.moduli[..num_primes];
    let level = num_primes - 1;

    // Initialize contexts
    let ckks_ctx = Arc::new(
        CudaCkksContext::new(params.clone())
            .expect("Failed to create CudaCkksContext")
    );

    let ctx = Arc::new(
        ParallelLiftContext::new(params.clone(), ckks_ctx.clone())
            .expect("Failed to create ParallelLiftContext")
    );

    // Generate secret key
    let secret_key = generate_test_secret_key(n, num_primes, moduli);

    // Create rotation context
    let device = ckks_ctx.device().clone();
    let rotation_ctx = Arc::new(
        CudaRotationContext::new(device.clone(), params.clone())
            .expect("Failed to create rotation context")
    );

    // Create V6 rotation keys with power-of-2 rotations
    let mut v6_rotation_keys = V6RotationKeys::new(ctx.clone(), &secret_key, base_bits)
        .expect("Failed to create V6 rotation keys");

    // Generate rotation key for rotation by 1
    v6_rotation_keys.generate_for_rotations(&[1, -1, 2, 4])
        .expect("Failed to generate rotation keys");

    // Create test ciphertext
    let c0 = generate_test_poly(n, num_primes, moduli);
    let c1 = generate_test_poly(n, num_primes, moduli);

    // Convert to strided layout for ciphertext
    let c0_strided = ckks_ctx.flat_to_strided(&c0, n, num_primes, num_primes);
    let c1_strided = ckks_ctx.flat_to_strided(&c1, n, num_primes, num_primes);

    use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext;

    let ct = CudaCiphertext {
        c0: c0_strided,
        c1: c1_strided,
        n,
        num_primes,
        level,
        scale: 1.0,
    };

    println!("\n=== Rotation Benchmark ===");
    println!("N = {}, level = {}, rotation steps = 1", n, level);

    // Benchmark V6 rotation (parallel_lift accelerated key switching)
    group.bench_with_input(
        BenchmarkId::new("V6-GPU", format!("N={},L={}", n, level)),
        &(&v6_rotation_keys, &ct),
        |b, (rk, ct)| {
            b.iter(|| {
                black_box(rk.rotate_v6(ct, 1))
            })
        },
    );

    group.finish();
}

/// Summary benchmark comparing key operations
#[cfg(feature = "v6-cuda")]
fn bench_v6_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("V6/Summary");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len().min(4);
    let base_bits = 16u32;
    let moduli = &params.moduli[..num_primes];

    // Initialize V6 context
    let ckks_ctx = Arc::new(
        CudaCkksContext::new(params.clone())
            .expect("Failed to create CudaCkksContext")
    );

    let ctx = ParallelLiftContext::new(params.clone(), ckks_ctx.clone())
        .expect("Failed to create ParallelLiftContext");

    let poly = generate_test_poly(n, num_primes, moduli);

    // Quick measurement for summary
    group.bench_function("gpu_gadget_decompose_v6", |b| {
        b.iter(|| {
            black_box(gpu_gadget_decompose_v6(&ctx, &poly, num_primes, base_bits))
        })
    });

    group.finish();
}

// Conditional compilation for non-CUDA builds
#[cfg(not(feature = "v6-cuda"))]
fn bench_gadget_decomposition(_c: &mut Criterion) {
    println!("V6-CUDA feature not enabled. Skipping GPU benchmarks.");
}

#[cfg(not(feature = "v6-cuda"))]
fn bench_relinearization(_c: &mut Criterion) {
    println!("V6-CUDA feature not enabled. Skipping GPU benchmarks.");
}

#[cfg(not(feature = "v6-cuda"))]
fn bench_rotation(_c: &mut Criterion) {
    println!("V6-CUDA feature not enabled. Skipping GPU benchmarks.");
}

#[cfg(not(feature = "v6-cuda"))]
fn bench_v6_summary(_c: &mut Criterion) {
    println!("V6-CUDA feature not enabled. Skipping GPU benchmarks.");
}

criterion_group!(
    v6_benches,
    bench_gadget_decomposition,
    bench_relinearization,
    bench_rotation,
    bench_v6_summary,
);

criterion_main!(v6_benches);
