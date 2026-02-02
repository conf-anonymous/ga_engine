//! V2 Geometric Operations Benchmark
//!
//! Run with: cargo bench --bench v2_geometric_ops --features v2
//!
//! Benchmarks:
//! - Reverse (~0.8ms)
//! - Geometric Product (~2s)
//! - Wedge Product (~4s)
//! - Inner Product (~4s)
//!
//! Total runtime: ~10 minutes with 50 samples

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

use ga_engine::clifford_fhe_v2::{
    params::CliffordFHEParams,
    backends::cpu_optimized::{
        keys::KeyContext,
        ckks::{CkksContext, Plaintext},
        geometric::GeometricContext,
        rns::RnsRepresentation,
    },
};

const BENCHMARK_DURATION_SECS: u64 = 10;
const BENCHMARK_SAMPLE_SIZE: usize = 50; // Reduced for expensive operations

fn bench_reverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("V2/Geometric/Reverse");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let ckks_ctx = CkksContext::new(params.clone());
    let geo_ctx = GeometricContext::new(params.clone());
    let (pk, _sk, _evk) = key_ctx.keygen();

    let encrypt_mv = |values: &[f64; 8]| {
        let level = params.max_level();
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();
        let mut result = Vec::new();
        for &val in values.iter() {
            let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
            let scaled_val = (val * params.scale).round() as i64;
            let rns_values: Vec<u64> = moduli.iter().map(|&q| {
                let q_i64 = q as i64;
                let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
                normalized as u64
            }).collect();
            coeffs[0] = RnsRepresentation::new(rns_values, moduli.clone());
            let pt = Plaintext::new(coeffs, params.scale, level);
            result.push(ckks_ctx.encrypt(&pt, &pk));
        }
        result.try_into().unwrap()
    };

    let mv_a = encrypt_mv(&[1.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0]);

    group.bench_function("reverse", |b| {
        b.iter(|| {
            black_box(geo_ctx.reverse(&mv_a))
        })
    });

    group.finish();
}

fn bench_geometric_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("V2/Geometric/Geometric Product");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let ckks_ctx = CkksContext::new(params.clone());
    let geo_ctx = GeometricContext::new(params.clone());
    let (pk, _sk, evk) = key_ctx.keygen();

    let encrypt_mv = |values: &[f64; 8]| {
        let level = params.max_level();
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();
        let mut result = Vec::new();
        for &val in values.iter() {
            let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
            let scaled_val = (val * params.scale).round() as i64;
            let rns_values: Vec<u64> = moduli.iter().map(|&q| {
                let q_i64 = q as i64;
                let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
                normalized as u64
            }).collect();
            coeffs[0] = RnsRepresentation::new(rns_values, moduli.clone());
            let pt = Plaintext::new(coeffs, params.scale, level);
            result.push(ckks_ctx.encrypt(&pt, &pk));
        }
        result.try_into().unwrap()
    };

    let mv_a = encrypt_mv(&[1.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0]);
    let mv_b = encrypt_mv(&[0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    group.bench_function("geometric_product", |b| {
        b.iter(|| {
            black_box(geo_ctx.geometric_product(&mv_a, &mv_b, &evk))
        })
    });

    group.finish();
}

fn bench_wedge_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("V2/Geometric/Wedge Product");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let ckks_ctx = CkksContext::new(params.clone());
    let geo_ctx = GeometricContext::new(params.clone());
    let (pk, _sk, evk) = key_ctx.keygen();

    let encrypt_mv = |values: &[f64; 8]| {
        let level = params.max_level();
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();
        let mut result = Vec::new();
        for &val in values.iter() {
            let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
            let scaled_val = (val * params.scale).round() as i64;
            let rns_values: Vec<u64> = moduli.iter().map(|&q| {
                let q_i64 = q as i64;
                let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
                normalized as u64
            }).collect();
            coeffs[0] = RnsRepresentation::new(rns_values, moduli.clone());
            let pt = Plaintext::new(coeffs, params.scale, level);
            result.push(ckks_ctx.encrypt(&pt, &pk));
        }
        result.try_into().unwrap()
    };

    let mv_a = encrypt_mv(&[1.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0]);
    let mv_b = encrypt_mv(&[0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    group.bench_function("wedge_product", |b| {
        b.iter(|| {
            black_box(geo_ctx.wedge_product(&mv_a, &mv_b, &evk))
        })
    });

    group.finish();
}

fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("V2/Geometric/Inner Product");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let ckks_ctx = CkksContext::new(params.clone());
    let geo_ctx = GeometricContext::new(params.clone());
    let (pk, _sk, evk) = key_ctx.keygen();

    let encrypt_mv = |values: &[f64; 8]| {
        let level = params.max_level();
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();
        let mut result = Vec::new();
        for &val in values.iter() {
            let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
            let scaled_val = (val * params.scale).round() as i64;
            let rns_values: Vec<u64> = moduli.iter().map(|&q| {
                let q_i64 = q as i64;
                let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
                normalized as u64
            }).collect();
            coeffs[0] = RnsRepresentation::new(rns_values, moduli.clone());
            let pt = Plaintext::new(coeffs, params.scale, level);
            result.push(ckks_ctx.encrypt(&pt, &pk));
        }
        result.try_into().unwrap()
    };

    let mv_a = encrypt_mv(&[1.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0]);
    let mv_b = encrypt_mv(&[0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    group.bench_function("inner_product", |b| {
        b.iter(|| {
            black_box(geo_ctx.inner_product(&mv_a, &mv_b, &evk))
        })
    });

    group.finish();
}

criterion_group!(
    v2_geometric_benches,
    bench_reverse,
    bench_geometric_product,
    bench_wedge_product,
    bench_inner_product,
);

criterion_main!(v2_geometric_benches);
