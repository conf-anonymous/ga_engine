//! Comprehensive Benchmarks for Clifford FHE Geometric Operations
//!
//! Measures performance of all implemented operations for both 2D and 3D.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{rns_encrypt, RnsPlaintext};
use ga_engine::clifford_fhe::geometric_product_rns::{
    // 2D operations
    geometric_product_2d_componentwise, reverse_2d, rotate_2d,
    wedge_product_2d, inner_product_2d,
    // 3D operations
    geometric_product_3d_componentwise, reverse_3d, rotate_3d,
    wedge_product_3d, inner_product_3d, project_3d, reject_3d,
};

/// Setup for 2D benchmarks
fn setup_2d() -> (CliffordFHEParams, [ga_engine::clifford_fhe::ckks_rns::RnsCiphertext; 4], [ga_engine::clifford_fhe::ckks_rns::RnsCiphertext; 4], ga_engine::clifford_fhe::keys_rns::RnsEvaluationKey) {
    let params = CliffordFHEParams::new_rns_mult();
    let (pk, _sk, evk) = rns_keygen(&params);

    let primes = &params.moduli;
    let delta = params.scale;
    let n = params.n;

    // Create two test multivectors
    let mv_a = [1.0, 2.0, 3.0, 4.0];
    let mv_b = [5.0, 6.0, 7.0, 8.0];

    let mut cts_a = Vec::new();
    let mut cts_b = Vec::new();

    for i in 0..4 {
        let mut coeffs_a = vec![0i64; n];
        let mut coeffs_b = vec![0i64; n];
        coeffs_a[0] = (mv_a[i] * delta).round() as i64;
        coeffs_b[0] = (mv_b[i] * delta).round() as i64;

        let pt_a = RnsPlaintext::from_coeffs(coeffs_a, delta, primes, 0);
        let pt_b = RnsPlaintext::from_coeffs(coeffs_b, delta, primes, 0);

        cts_a.push(rns_encrypt(&pk, &pt_a, &params));
        cts_b.push(rns_encrypt(&pk, &pt_b, &params));
    }

    let cts_a_array = [cts_a[0].clone(), cts_a[1].clone(), cts_a[2].clone(), cts_a[3].clone()];
    let cts_b_array = [cts_b[0].clone(), cts_b[1].clone(), cts_b[2].clone(), cts_b[3].clone()];

    (params, cts_a_array, cts_b_array, evk)
}

/// Setup for 3D benchmarks
fn setup_3d() -> (CliffordFHEParams, [ga_engine::clifford_fhe::ckks_rns::RnsCiphertext; 8], [ga_engine::clifford_fhe::ckks_rns::RnsCiphertext; 8], ga_engine::clifford_fhe::keys_rns::RnsEvaluationKey) {
    let params = CliffordFHEParams::new_rns_mult();
    let (pk, _sk, evk) = rns_keygen(&params);

    let primes = &params.moduli;
    let delta = params.scale;
    let n = params.n;

    // Create two test multivectors
    let mv_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mv_b = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

    let mut cts_a = Vec::new();
    let mut cts_b = Vec::new();

    for i in 0..8 {
        let mut coeffs_a = vec![0i64; n];
        let mut coeffs_b = vec![0i64; n];
        coeffs_a[0] = (mv_a[i] * delta).round() as i64;
        coeffs_b[0] = (mv_b[i] * delta).round() as i64;

        let pt_a = RnsPlaintext::from_coeffs(coeffs_a, delta, primes, 0);
        let pt_b = RnsPlaintext::from_coeffs(coeffs_b, delta, primes, 0);

        cts_a.push(rns_encrypt(&pk, &pt_a, &params));
        cts_b.push(rns_encrypt(&pk, &pt_b, &params));
    }

    let cts_a_array = [
        cts_a[0].clone(), cts_a[1].clone(), cts_a[2].clone(), cts_a[3].clone(),
        cts_a[4].clone(), cts_a[5].clone(), cts_a[6].clone(), cts_a[7].clone(),
    ];
    let cts_b_array = [
        cts_b[0].clone(), cts_b[1].clone(), cts_b[2].clone(), cts_b[3].clone(),
        cts_b[4].clone(), cts_b[5].clone(), cts_b[6].clone(), cts_b[7].clone(),
    ];

    (params, cts_a_array, cts_b_array, evk)
}

fn bench_2d_geometric_product(c: &mut Criterion) {
    let (params, cts_a, cts_b, evk) = setup_2d();

    c.bench_function("2D Geometric Product", |b| {
        b.iter(|| {
            geometric_product_2d_componentwise(
                black_box(&cts_a),
                black_box(&cts_b),
                black_box(&evk),
                black_box(&params),
            )
        })
    });
}

fn bench_2d_reverse(c: &mut Criterion) {
    let (params, cts_a, _cts_b, _evk) = setup_2d();

    c.bench_function("2D Reverse", |b| {
        b.iter(|| {
            reverse_2d(
                black_box(&cts_a),
                black_box(&params),
            )
        })
    });
}

fn bench_2d_rotation(c: &mut Criterion) {
    let (params, cts_a, cts_b, evk) = setup_2d();

    c.bench_function("2D Rotation (R·x·R̃)", |b| {
        b.iter(|| {
            rotate_2d(
                black_box(&cts_a),
                black_box(&cts_b),
                black_box(&evk),
                black_box(&params),
            )
        })
    });
}

fn bench_2d_wedge_product(c: &mut Criterion) {
    let (params, cts_a, cts_b, evk) = setup_2d();

    c.bench_function("2D Wedge Product", |b| {
        b.iter(|| {
            wedge_product_2d(
                black_box(&cts_a),
                black_box(&cts_b),
                black_box(&evk),
                black_box(&params),
            )
        })
    });
}

fn bench_2d_inner_product(c: &mut Criterion) {
    let (params, cts_a, cts_b, evk) = setup_2d();

    c.bench_function("2D Inner Product", |b| {
        b.iter(|| {
            inner_product_2d(
                black_box(&cts_a),
                black_box(&cts_b),
                black_box(&evk),
                black_box(&params),
            )
        })
    });
}

fn bench_3d_geometric_product(c: &mut Criterion) {
    let (params, cts_a, cts_b, evk) = setup_3d();

    c.bench_function("3D Geometric Product", |b| {
        b.iter(|| {
            geometric_product_3d_componentwise(
                black_box(&cts_a),
                black_box(&cts_b),
                black_box(&evk),
                black_box(&params),
            )
        })
    });
}

fn bench_3d_reverse(c: &mut Criterion) {
    let (params, cts_a, _cts_b, _evk) = setup_3d();

    c.bench_function("3D Reverse", |b| {
        b.iter(|| {
            reverse_3d(
                black_box(&cts_a),
                black_box(&params),
            )
        })
    });
}

fn bench_3d_rotation(c: &mut Criterion) {
    let (params, cts_a, cts_b, evk) = setup_3d();

    c.bench_function("3D Rotation (R·x·R̃)", |b| {
        b.iter(|| {
            rotate_3d(
                black_box(&cts_a),
                black_box(&cts_b),
                black_box(&evk),
                black_box(&params),
            )
        })
    });
}

fn bench_3d_wedge_product(c: &mut Criterion) {
    let (params, cts_a, cts_b, evk) = setup_3d();

    c.bench_function("3D Wedge Product", |b| {
        b.iter(|| {
            wedge_product_3d(
                black_box(&cts_a),
                black_box(&cts_b),
                black_box(&evk),
                black_box(&params),
            )
        })
    });
}

fn bench_3d_inner_product(c: &mut Criterion) {
    let (params, cts_a, cts_b, evk) = setup_3d();

    c.bench_function("3D Inner Product", |b| {
        b.iter(|| {
            inner_product_3d(
                black_box(&cts_a),
                black_box(&cts_b),
                black_box(&evk),
                black_box(&params),
            )
        })
    });
}

fn bench_3d_projection(c: &mut Criterion) {
    let (params, cts_a, cts_b, evk) = setup_3d();

    c.bench_function("3D Projection", |b| {
        b.iter(|| {
            project_3d(
                black_box(&cts_a),
                black_box(&cts_b),
                black_box(&evk),
                black_box(&params),
            )
        })
    });
}

fn bench_3d_rejection(c: &mut Criterion) {
    let (params, cts_a, cts_b, evk) = setup_3d();

    c.bench_function("3D Rejection", |b| {
        b.iter(|| {
            reject_3d(
                black_box(&cts_a),
                black_box(&cts_b),
                black_box(&evk),
                black_box(&params),
            )
        })
    });
}

criterion_group!(
    benches_2d,
    bench_2d_geometric_product,
    bench_2d_reverse,
    bench_2d_rotation,
    bench_2d_wedge_product,
    bench_2d_inner_product,
);

criterion_group!(
    benches_3d,
    bench_3d_geometric_product,
    bench_3d_reverse,
    bench_3d_rotation,
    bench_3d_wedge_product,
    bench_3d_inner_product,
    bench_3d_projection,
    bench_3d_rejection,
);

criterion_main!(benches_2d, benches_3d);
