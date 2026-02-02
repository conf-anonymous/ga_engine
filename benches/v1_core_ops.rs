//! V1 Core Operations Benchmark
//!
//! Run with: cargo bench --bench v1_core_ops --features v1
//!
//! Benchmarks:
//! - Key generation
//! - Single encryption
//! - Single decryption
//! - Ciphertext multiplication

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

use ga_engine::clifford_fhe_v1::{
    params::CliffordFHEParams,
    keys_rns::rns_keygen,
    ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt, rns_multiply_ciphertexts},
};

const BENCHMARK_DURATION_SECS: u64 = 10;
const BENCHMARK_SAMPLE_SIZE: usize = 100;

fn bench_keygen(c: &mut Criterion) {
    let mut group = c.benchmark_group("V1/Core/Key Generation");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_rns_mult();

    group.bench_function("keygen", |b| {
        b.iter(|| {
            black_box(rns_keygen(&params))
        })
    });

    group.finish();
}

fn bench_encryption(c: &mut Criterion) {
    let mut group = c.benchmark_group("V1/Core/Encryption");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_rns_mult();
    let (pk, _sk, _evk) = rns_keygen(&params);

    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = (1.0 * params.scale).round() as i64;
    let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);

    group.bench_function("encrypt", |b| {
        b.iter(|| {
            black_box(rns_encrypt(&pk, &pt, &params))
        })
    });

    group.finish();
}

fn bench_decryption(c: &mut Criterion) {
    let mut group = c.benchmark_group("V1/Core/Decryption");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_rns_mult();
    let (pk, sk, _evk) = rns_keygen(&params);

    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = (1.0 * params.scale).round() as i64;
    let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
    let ct = rns_encrypt(&pk, &pt, &params);

    group.bench_function("decrypt", |b| {
        b.iter(|| {
            black_box(rns_decrypt(&sk, &ct, &params))
        })
    });

    group.finish();
}

fn bench_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("V1/Core/Multiplication");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_rns_mult();
    let (pk, _sk, evk) = rns_keygen(&params);

    let mut coeffs_a = vec![0i64; params.n];
    coeffs_a[0] = (2.0 * params.scale).round() as i64;
    let pt_a = RnsPlaintext::from_coeffs(coeffs_a, params.scale, &params.moduli, 0);
    let ct_a = rns_encrypt(&pk, &pt_a, &params);

    let mut coeffs_b = vec![0i64; params.n];
    coeffs_b[0] = (3.0 * params.scale).round() as i64;
    let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);
    let ct_b = rns_encrypt(&pk, &pt_b, &params);

    group.bench_function("multiply", |b| {
        b.iter(|| {
            black_box(rns_multiply_ciphertexts(&ct_a, &ct_b, &evk, &params))
        })
    });

    group.finish();
}

criterion_group!(
    v1_core_benches,
    bench_keygen,
    bench_encryption,
    bench_decryption,
    bench_multiplication,
);

criterion_main!(v1_core_benches);
