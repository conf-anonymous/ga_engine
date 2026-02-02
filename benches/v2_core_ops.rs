//! V2 Core Operations Benchmark
//!
//! Run with: cargo bench --bench v2_core_ops --features v2
//!
//! Benchmarks:
//! - Key generation
//! - Single encryption
//! - Single decryption
//! - Ciphertext multiplication

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

use ga_engine::clifford_fhe_v2::{
    params::CliffordFHEParams,
    backends::cpu_optimized::{
        keys::KeyContext,
        ckks::{CkksContext, Plaintext},
        rns::RnsRepresentation,
        multiplication::multiply_ciphertexts,
    },
};

const BENCHMARK_DURATION_SECS: u64 = 10;
const BENCHMARK_SAMPLE_SIZE: usize = 100;

fn bench_keygen(c: &mut Criterion) {
    let mut group = c.benchmark_group("V2/Core/Key Generation");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_test_ntt_1024();

    group.bench_function("keygen", |b| {
        b.iter(|| {
            let key_ctx = KeyContext::new(params.clone());
            black_box(key_ctx.keygen())
        })
    });

    group.finish();
}

fn bench_encryption(c: &mut Criterion) {
    let mut group = c.benchmark_group("V2/Core/Encryption");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let ckks_ctx = CkksContext::new(params.clone());
    let (pk, _sk, _evk) = key_ctx.keygen();

    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
    let scaled_val = (1.0 * params.scale).round() as i64;
    let values: Vec<u64> = moduli.iter().map(|&q| {
        let q_i64 = q as i64;
        let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
        normalized as u64
    }).collect();
    coeffs[0] = RnsRepresentation::new(values, moduli.clone());
    let pt = Plaintext::new(coeffs, params.scale, level);

    group.bench_function("encrypt", |b| {
        b.iter(|| {
            black_box(ckks_ctx.encrypt(&pt, &pk))
        })
    });

    group.finish();
}

fn bench_decryption(c: &mut Criterion) {
    let mut group = c.benchmark_group("V2/Core/Decryption");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let ckks_ctx = CkksContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();

    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
    let scaled_val = (1.0 * params.scale).round() as i64;
    let values: Vec<u64> = moduli.iter().map(|&q| {
        let q_i64 = q as i64;
        let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
        normalized as u64
    }).collect();
    coeffs[0] = RnsRepresentation::new(values, moduli.clone());
    let pt = Plaintext::new(coeffs, params.scale, level);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    group.bench_function("decrypt", |b| {
        b.iter(|| {
            black_box(ckks_ctx.decrypt(&ct, &sk))
        })
    });

    group.finish();
}

fn bench_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("V2/Core/Multiplication");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let ckks_ctx = CkksContext::new(params.clone());
    let (pk, _sk, evk) = key_ctx.keygen();

    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();

    let mut coeffs_a = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
    let scaled_val_a = (2.0 * params.scale).round() as i64;
    let values_a: Vec<u64> = moduli.iter().map(|&q| {
        let q_i64 = q as i64;
        let normalized = ((scaled_val_a % q_i64) + q_i64) % q_i64;
        normalized as u64
    }).collect();
    coeffs_a[0] = RnsRepresentation::new(values_a, moduli.clone());
    let pt_a = Plaintext::new(coeffs_a, params.scale, level);
    let ct_a = ckks_ctx.encrypt(&pt_a, &pk);

    let mut coeffs_b = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
    let scaled_val_b = (3.0 * params.scale).round() as i64;
    let values_b: Vec<u64> = moduli.iter().map(|&q| {
        let q_i64 = q as i64;
        let normalized = ((scaled_val_b % q_i64) + q_i64) % q_i64;
        normalized as u64
    }).collect();
    coeffs_b[0] = RnsRepresentation::new(values_b, moduli.clone());
    let pt_b = Plaintext::new(coeffs_b, params.scale, level);
    let ct_b = ckks_ctx.encrypt(&pt_b, &pk);

    group.bench_function("multiply", |b| {
        b.iter(|| {
            black_box(multiply_ciphertexts(&ct_a, &ct_b, &evk, &key_ctx))
        })
    });

    group.finish();
}

criterion_group!(
    v2_core_benches,
    bench_keygen,
    bench_encryption,
    bench_decryption,
    bench_multiplication,
);

criterion_main!(v2_core_benches);
