//! Comprehensive V1 vs V2 Performance Benchmark
//!
//! Run with: cargo bench --bench v1_vs_v2_benchmark --features v1,v2
//!
//! Compares:
//! - Key generation
//! - Single encryption
//! - Single decryption
//! - Ciphertext multiplication (tensor product + relinearization + rescaling)
//! - Geometric operations (reverse, geometric product, rotation, etc.)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

// V1 imports
#[cfg(feature = "v1")]
use ga_engine::clifford_fhe_v1::{
    params::CliffordFHEParams as V1Params,
    keys_rns::rns_keygen as v1_keygen,
    ckks_rns::{RnsPlaintext as V1Plaintext, rns_encrypt as v1_encrypt, rns_decrypt as v1_decrypt, rns_multiply_ciphertexts as v1_multiply},
    geometric_product_rns::{reverse_3d as v1_reverse, geometric_product_3d_componentwise as v1_geometric_product, wedge_product_3d as v1_wedge_product, inner_product_3d as v1_inner_product},
};

// V2 imports
#[cfg(feature = "v2")]
use ga_engine::clifford_fhe_v2::{
    params::CliffordFHEParams as V2Params,
    backends::cpu_optimized::{
        keys::KeyContext as V2KeyContext,
        ckks::{CkksContext as V2CkksContext, Plaintext as V2Plaintext},
        geometric::GeometricContext as V2GeometricContext,
        rns::RnsRepresentation as V2RnsRep,
        multiplication::multiply_ciphertexts as v2_multiply,
    },
};

/// Benchmark configuration
const BENCHMARK_DURATION_SECS: u64 = 10;
const BENCHMARK_SAMPLE_SIZE: usize = 100;

// =============================================================================
// V1 BENCHMARKS
// =============================================================================

#[cfg(feature = "v1")]
fn bench_v1_keygen(c: &mut Criterion) {
    let mut group = c.benchmark_group("Key Generation");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = V1Params::new_rns_mult();

    group.bench_function("V1", |b| {
        b.iter(|| {
            black_box(v1_keygen(&params))
        })
    });

    group.finish();
}

#[cfg(feature = "v1")]
fn bench_v1_encryption(c: &mut Criterion) {
    let mut group = c.benchmark_group("Single Encryption");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = V1Params::new_rns_mult();
    let (pk, _sk, _evk) = v1_keygen(&params);

    // Create plaintext: encrypt value 1.0
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = (1.0 * params.scale).round() as i64;
    let pt = V1Plaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);

    group.bench_function("V1", |b| {
        b.iter(|| {
            black_box(v1_encrypt(&pk, &pt, &params))
        })
    });

    group.finish();
}

#[cfg(feature = "v1")]
fn bench_v1_decryption(c: &mut Criterion) {
    let mut group = c.benchmark_group("Single Decryption");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = V1Params::new_rns_mult();
    let (pk, sk, _evk) = v1_keygen(&params);

    // Create and encrypt plaintext
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = (1.0 * params.scale).round() as i64;
    let pt = V1Plaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
    let ct = v1_encrypt(&pk, &pt, &params);

    group.bench_function("V1", |b| {
        b.iter(|| {
            black_box(v1_decrypt(&sk, &ct, &params))
        })
    });

    group.finish();
}

#[cfg(feature = "v1")]
fn bench_v1_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("Ciphertext Multiplication");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = V1Params::new_rns_mult();
    let (pk, _sk, evk) = v1_keygen(&params);

    // Encrypt two values
    let mut coeffs_a = vec![0i64; params.n];
    coeffs_a[0] = (2.0 * params.scale).round() as i64;
    let pt_a = V1Plaintext::from_coeffs(coeffs_a, params.scale, &params.moduli, 0);
    let ct_a = v1_encrypt(&pk, &pt_a, &params);

    let mut coeffs_b = vec![0i64; params.n];
    coeffs_b[0] = (3.0 * params.scale).round() as i64;
    let pt_b = V1Plaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);
    let ct_b = v1_encrypt(&pk, &pt_b, &params);

    group.bench_function("V1", |b| {
        b.iter(|| {
            black_box(v1_multiply(&ct_a, &ct_b, &evk, &params))
        })
    });

    group.finish();
}

#[cfg(feature = "v1")]
fn bench_v1_geometric_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Geometric Operations");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(50); // Fewer samples for expensive operations

    let params = V1Params::new_rns_mult_depth2_safe();
    let (pk, _sk, evk) = v1_keygen(&params);

    // Helper to encrypt multivector
    let encrypt_mv = |values: &[f64; 8]| {
        let mut result = Vec::new();
        for &val in values.iter() {
            let mut coeffs = vec![0i64; params.n];
            coeffs[0] = (val * params.scale).round() as i64;
            let pt = V1Plaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
            result.push(v1_encrypt(&pk, &pt, &params));
        }
        result.try_into().unwrap()
    };

    // Test multivectors
    let mv_a = encrypt_mv(&[1.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0]);
    let mv_b = encrypt_mv(&[0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    group.bench_function("Reverse", |b| {
        b.iter(|| {
            black_box(v1_reverse(&mv_a, &params))
        })
    });

    group.bench_function("Geometric Product", |b| {
        b.iter(|| {
            black_box(v1_geometric_product(&mv_a, &mv_b, &evk, &params))
        })
    });

    group.bench_function("Wedge Product", |b| {
        b.iter(|| {
            black_box(v1_wedge_product(&mv_a, &mv_b, &evk, &params))
        })
    });

    group.bench_function("Inner Product", |b| {
        b.iter(|| {
            black_box(v1_inner_product(&mv_a, &mv_b, &evk, &params))
        })
    });

    group.finish();
}

// =============================================================================
// V2 BENCHMARKS
// =============================================================================

#[cfg(feature = "v2")]
fn bench_v2_keygen(c: &mut Criterion) {
    let mut group = c.benchmark_group("Key Generation");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = V2Params::new_test_ntt_1024();

    group.bench_function("V2", |b| {
        b.iter(|| {
            let key_ctx = V2KeyContext::new(params.clone());
            black_box(key_ctx.keygen())
        })
    });

    group.finish();
}

#[cfg(feature = "v2")]
fn bench_v2_encryption(c: &mut Criterion) {
    let mut group = c.benchmark_group("Single Encryption");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = V2Params::new_test_ntt_1024();
    let key_ctx = V2KeyContext::new(params.clone());
    let ckks_ctx = V2CkksContext::new(params.clone());
    let (pk, _sk, _evk) = key_ctx.keygen();

    // Create plaintext: encrypt value 1.0
    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let mut coeffs = vec![V2RnsRep::from_u64(0, &moduli); params.n];
    let scaled_val = (1.0 * params.scale).round() as i64;
    let values: Vec<u64> = moduli.iter().map(|&q| {
        let q_i64 = q as i64;
        let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
        normalized as u64
    }).collect();
    coeffs[0] = V2RnsRep::new(values, moduli.clone());
    let pt = V2Plaintext::new(coeffs, params.scale, level);

    group.bench_function("V2", |b| {
        b.iter(|| {
            black_box(ckks_ctx.encrypt(&pt, &pk))
        })
    });

    group.finish();
}

#[cfg(feature = "v2")]
fn bench_v2_decryption(c: &mut Criterion) {
    let mut group = c.benchmark_group("Single Decryption");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = V2Params::new_test_ntt_1024();
    let key_ctx = V2KeyContext::new(params.clone());
    let ckks_ctx = V2CkksContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();

    // Create and encrypt plaintext
    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let mut coeffs = vec![V2RnsRep::from_u64(0, &moduli); params.n];
    let scaled_val = (1.0 * params.scale).round() as i64;
    let values: Vec<u64> = moduli.iter().map(|&q| {
        let q_i64 = q as i64;
        let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
        normalized as u64
    }).collect();
    coeffs[0] = V2RnsRep::new(values, moduli);
    let pt = V2Plaintext::new(coeffs, params.scale, level);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    group.bench_function("V2", |b| {
        b.iter(|| {
            black_box(ckks_ctx.decrypt(&ct, &sk))
        })
    });

    group.finish();
}

#[cfg(feature = "v2")]
fn bench_v2_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("Ciphertext Multiplication");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(BENCHMARK_SAMPLE_SIZE);

    let params = V2Params::new_test_ntt_1024();
    let key_ctx = V2KeyContext::new(params.clone());
    let ckks_ctx = V2CkksContext::new(params.clone());
    let (pk, _sk, evk) = key_ctx.keygen();

    // Helper to create plaintext
    let create_pt = |value: f64| {
        let level = params.max_level();
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();
        let mut coeffs = vec![V2RnsRep::from_u64(0, &moduli); params.n];
        let scaled_val = (value * params.scale).round() as i64;
        let values: Vec<u64> = moduli.iter().map(|&q| {
            let q_i64 = q as i64;
            let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
            normalized as u64
        }).collect();
        coeffs[0] = V2RnsRep::new(values, moduli);
        V2Plaintext::new(coeffs, params.scale, level)
    };

    // Encrypt two values
    let pt_a = create_pt(2.0);
    let ct_a = ckks_ctx.encrypt(&pt_a, &pk);

    let pt_b = create_pt(3.0);
    let ct_b = ckks_ctx.encrypt(&pt_b, &pk);

    group.bench_function("V2", |b| {
        b.iter(|| {
            black_box(v2_multiply(&ct_a, &ct_b, &evk, &key_ctx))
        })
    });

    group.finish();
}

// =============================================================================
// GEOMETRIC OPERATIONS BENCHMARKS
// =============================================================================

#[cfg(feature = "v2")]
fn bench_v2_geometric_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Geometric Operations");
    group.measurement_time(Duration::from_secs(BENCHMARK_DURATION_SECS));
    group.sample_size(50); // Fewer samples for expensive operations

    let params = V2Params::new_test_ntt_1024();
    let key_ctx = V2KeyContext::new(params.clone());
    let ckks_ctx = V2CkksContext::new(params.clone());
    let geo_ctx = V2GeometricContext::new(params.clone());
    let (pk, _sk, evk) = key_ctx.keygen();

    // Helper to encrypt multivector
    let encrypt_mv = |values: &[f64; 8]| {
        let level = params.max_level();
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();
        let mut result = Vec::new();
        for &val in values.iter() {
            let mut coeffs = vec![V2RnsRep::from_u64(0, &moduli); params.n];
            let scaled_val = (val * params.scale).round() as i64;
            let rns_values: Vec<u64> = moduli.iter().map(|&q| {
                let q_i64 = q as i64;
                let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
                normalized as u64
            }).collect();
            coeffs[0] = V2RnsRep::new(rns_values, moduli.clone());
            let pt = V2Plaintext::new(coeffs, params.scale, level);
            result.push(ckks_ctx.encrypt(&pt, &pk));
        }
        result.try_into().unwrap()
    };

    // Test multivectors
    let mv_a = encrypt_mv(&[1.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0]);
    let mv_b = encrypt_mv(&[0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    group.bench_function("Reverse", |b| {
        b.iter(|| {
            black_box(geo_ctx.reverse(&mv_a))
        })
    });

    group.bench_function("Geometric Product", |b| {
        b.iter(|| {
            black_box(geo_ctx.geometric_product(&mv_a, &mv_b, &evk))
        })
    });

    group.bench_function("Wedge Product", |b| {
        b.iter(|| {
            black_box(geo_ctx.wedge_product(&mv_a, &mv_b, &evk))
        })
    });

    group.bench_function("Inner Product", |b| {
        b.iter(|| {
            black_box(geo_ctx.inner_product(&mv_a, &mv_b, &evk))
        })
    });

    group.finish();
}

// =============================================================================
// CRITERION SETUP
// =============================================================================

// Separate groups for V1 and V2 benchmarks
#[cfg(feature = "v1")]
criterion_group!(
    v1_benches,
    bench_v1_keygen,
    bench_v1_encryption,
    bench_v1_decryption,
    bench_v1_multiplication,
    bench_v1_geometric_operations,
);

#[cfg(feature = "v2")]
criterion_group!(
    v2_benches,
    bench_v2_keygen,
    bench_v2_encryption,
    bench_v2_decryption,
    bench_v2_multiplication,
    bench_v2_geometric_operations,
);

// Combine groups based on which features are enabled
#[cfg(all(feature = "v1", feature = "v2"))]
criterion_main!(v1_benches, v2_benches);

#[cfg(all(feature = "v1", not(feature = "v2")))]
criterion_main!(v1_benches);

#[cfg(all(feature = "v2", not(feature = "v1")))]
criterion_main!(v2_benches);

#[cfg(not(any(feature = "v1", feature = "v2")))]
fn main() {
    eprintln!("This benchmark requires at least one of 'v1' or 'v2' features.");
    eprintln!("Run with: cargo bench --bench v1_vs_v2_benchmark --features v1,v2");
    std::process::exit(1);
}
