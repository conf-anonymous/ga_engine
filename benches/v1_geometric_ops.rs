//! V1 Geometric Operations Benchmark
//!
//! Run with: cargo bench --bench v1_geometric_ops --features v1
//!
//! Benchmarks:
//! - Reverse (depth-0)
//! - Geometric Product (depth-1, ~13s per iteration)
//! - Wedge Product (depth-2, ~26s per iteration)
//! - Inner Product (depth-2, ~26s per iteration)
//!
//! WARNING: This benchmark takes ~55 minutes to complete with 50 samples!
//! To run faster with fewer samples: cargo bench --bench v1_geometric_ops --features v1 -- --sample-size 10

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

use ga_engine::clifford_fhe_v1::{
    params::CliffordFHEParams,
    keys_rns::rns_keygen,
    ckks_rns::{RnsPlaintext, rns_encrypt},
    geometric_product_rns::{
        reverse_3d,
        geometric_product_3d_componentwise,
        wedge_product_3d,
        inner_product_3d,
    },
};

const BENCHMARK_DURATION_SECS: u64 = 10;
const BENCHMARK_SAMPLE_SIZE: usize = 50; // Reduced for expensive operations

fn bench_reverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("V1/Geometric/Reverse");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let (pk, _sk, _evk) = rns_keygen(&params);

    let encrypt_mv = |values: &[f64; 8]| {
        let mut result = Vec::new();
        for &val in values.iter() {
            let mut coeffs = vec![0i64; params.n];
            coeffs[0] = (val * params.scale).round() as i64;
            let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
            result.push(rns_encrypt(&pk, &pt, &params));
        }
        result.try_into().unwrap()
    };

    let mv_a = encrypt_mv(&[1.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0]);

    group.bench_function("reverse", |b| {
        b.iter(|| {
            black_box(reverse_3d(&mv_a, &params))
        })
    });

    group.finish();
}

fn bench_geometric_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("V1/Geometric/Geometric Product");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let (pk, _sk, evk) = rns_keygen(&params);

    let encrypt_mv = |values: &[f64; 8]| {
        let mut result = Vec::new();
        for &val in values.iter() {
            let mut coeffs = vec![0i64; params.n];
            coeffs[0] = (val * params.scale).round() as i64;
            let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
            result.push(rns_encrypt(&pk, &pt, &params));
        }
        result.try_into().unwrap()
    };

    let mv_a = encrypt_mv(&[1.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0]);
    let mv_b = encrypt_mv(&[0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    group.bench_function("geometric_product", |b| {
        b.iter(|| {
            black_box(geometric_product_3d_componentwise(&mv_a, &mv_b, &evk, &params))
        })
    });

    group.finish();
}

fn bench_wedge_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("V1/Geometric/Wedge Product");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let (pk, _sk, evk) = rns_keygen(&params);

    let encrypt_mv = |values: &[f64; 8]| {
        let mut result = Vec::new();
        for &val in values.iter() {
            let mut coeffs = vec![0i64; params.n];
            coeffs[0] = (val * params.scale).round() as i64;
            let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
            result.push(rns_encrypt(&pk, &pt, &params));
        }
        result.try_into().unwrap()
    };

    let mv_a = encrypt_mv(&[1.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0]);
    let mv_b = encrypt_mv(&[0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    group.bench_function("wedge_product", |b| {
        b.iter(|| {
            black_box(wedge_product_3d(&mv_a, &mv_b, &evk, &params))
        })
    });

    group.finish();
}

fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("V1/Geometric/Inner Product");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let (pk, _sk, evk) = rns_keygen(&params);

    let encrypt_mv = |values: &[f64; 8]| {
        let mut result = Vec::new();
        for &val in values.iter() {
            let mut coeffs = vec![0i64; params.n];
            coeffs[0] = (val * params.scale).round() as i64;
            let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
            result.push(rns_encrypt(&pk, &pt, &params));
        }
        result.try_into().unwrap()
    };

    let mv_a = encrypt_mv(&[1.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0]);
    let mv_b = encrypt_mv(&[0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    group.bench_function("inner_product", |b| {
        b.iter(|| {
            black_box(inner_product_3d(&mv_a, &mv_b, &evk, &params))
        })
    });

    group.finish();
}

criterion_group!(
    v1_geometric_benches,
    bench_reverse,
    bench_geometric_product,
    bench_wedge_product,
    bench_inner_product,
);

criterion_main!(v1_geometric_benches);
