//! Benchmark for V4 Packed Multivector Geometric Product
//!
//! Measures performance of the packed geometric product operation on Metal GPU.
//! Compares different batch sizes and provides detailed timing breakdowns.

#[cfg(all(feature = "v4", feature = "v2-gpu-metal"))]
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

#[cfg(all(feature = "v4", feature = "v2-gpu-metal"))]
use ga_engine::clifford_fhe_v2::backends::gpu_metal::{
    ckks::MetalCkksContext,
    rotation_keys::MetalRotationKeys,
};
#[cfg(all(feature = "v4", feature = "v2-gpu-metal"))]
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
#[cfg(all(feature = "v4", feature = "v2-gpu-metal"))]
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
#[cfg(all(feature = "v4", feature = "v2-gpu-metal"))]
use ga_engine::clifford_fhe_v4::{
    packing::pack_multivector,
    geometric_ops::geometric_product_packed,
};

#[cfg(all(feature = "v4", feature = "v2-gpu-metal"))]
fn setup_context() -> (MetalCkksContext, KeyContext, MetalRotationKeys) {
    // Create parameters
    let params = CliffordFHEParams::default();

    // Generate keys
    let key_ctx = KeyContext::new(params.clone());
    let (_pk, _sk, _evk) = key_ctx.keygen();

    // Create Metal CKKS context
    let ckks_ctx = MetalCkksContext::new(params.clone()).unwrap();

    // Generate rotation keys (need ±1 to ±8 for packing/unpacking)
    let mut rotation_steps: Vec<i32> = (1..=8).collect();
    rotation_steps.extend((-8..=-1).collect::<Vec<i32>>());

    let metal_device = ckks_ctx.device().clone();
    let metal_ntt_contexts = ckks_ctx.ntt_contexts();

    let rot_keys = MetalRotationKeys::generate(
        metal_device,
        &_sk,
        &rotation_steps,
        &params,
        metal_ntt_contexts,
        20, // base_w
    ).unwrap();

    (ckks_ctx, key_ctx, rot_keys)
}

#[cfg(all(feature = "v4", feature = "v2-gpu-metal"))]
fn bench_geometric_product_single(c: &mut Criterion) {
    let (ckks_ctx, key_ctx, rot_keys) = setup_context();
    let (_pk, _sk, _evk) = key_ctx.keygen();

    // Create test multivectors: a = 1 + 2e₁, b = 3e₂
    let batch_size = 1;
    let num_slots = ckks_ctx.params.n / 2;

    // Encode and encrypt components
    let mut a_vals = vec![vec![0.0; num_slots]; 8];
    a_vals[0][0] = 1.0;  // scalar
    a_vals[1][0] = 2.0;  // e1

    let mut b_vals = vec![vec![0.0; num_slots]; 8];
    b_vals[2][0] = 3.0;  // e2

    let mut a_components = Vec::new();
    let mut b_components = Vec::new();

    for i in 0..8 {
        let a_pt = ckks_ctx.encode(&a_vals[i]).unwrap();
        let a_ct = ckks_ctx.encrypt(&a_pt, &_pk).unwrap();
        a_components.push(a_ct);

        let b_pt = ckks_ctx.encode(&b_vals[i]).unwrap();
        let b_ct = ckks_ctx.encrypt(&b_pt, &_pk).unwrap();
        b_components.push(b_ct);
    }

    let a_array: [_; 8] = [
        a_components[0].clone(), a_components[1].clone(), a_components[2].clone(),
        a_components[3].clone(), a_components[4].clone(), a_components[5].clone(),
        a_components[6].clone(), a_components[7].clone(),
    ];
    let b_array: [_; 8] = [
        b_components[0].clone(), b_components[1].clone(), b_components[2].clone(),
        b_components[3].clone(), b_components[4].clone(), b_components[5].clone(),
        b_components[6].clone(), b_components[7].clone(),
    ];

    // Pack into V4 format
    let a_packed = pack_multivector(&a_array, batch_size, &rot_keys, &ckks_ctx).unwrap();
    let b_packed = pack_multivector(&b_array, batch_size, &rot_keys, &ckks_ctx).unwrap();

    // Benchmark the geometric product
    c.bench_function("v4_geometric_product_single", |bencher| {
        bencher.iter(|| {
            geometric_product_packed(
                black_box(&a_packed),
                black_box(&b_packed),
                black_box(&rot_keys),
                black_box(&ckks_ctx),
            ).unwrap()
        });
    });
}

#[cfg(all(feature = "v4", feature = "v2-gpu-metal"))]
fn bench_geometric_product_with_breakdown(c: &mut Criterion) {
    let (ckks_ctx, key_ctx, rot_keys) = setup_context();
    let (_pk, _sk, _evk) = key_ctx.keygen();

    // Create test multivectors
    let batch_size = 1;
    let num_slots = ckks_ctx.params.n / 2;

    let mut a_vals = vec![vec![0.0; num_slots]; 8];
    a_vals[0][0] = 1.0;
    a_vals[1][0] = 2.0;

    let mut b_vals = vec![vec![0.0; num_slots]; 8];
    b_vals[2][0] = 3.0;

    let mut a_components = Vec::new();
    let mut b_components = Vec::new();

    for i in 0..8 {
        let a_pt = ckks_ctx.encode(&a_vals[i]).unwrap();
        let a_ct = ckks_ctx.encrypt(&a_pt, &_pk).unwrap();
        a_components.push(a_ct);

        let b_pt = ckks_ctx.encode(&b_vals[i]).unwrap();
        let b_ct = ckks_ctx.encrypt(&b_pt, &_pk).unwrap();
        b_components.push(b_ct);
    }

    let a_array: [_; 8] = [
        a_components[0].clone(), a_components[1].clone(), a_components[2].clone(),
        a_components[3].clone(), a_components[4].clone(), a_components[5].clone(),
        a_components[6].clone(), a_components[7].clone(),
    ];
    let b_array: [_; 8] = [
        b_components[0].clone(), b_components[1].clone(), b_components[2].clone(),
        b_components[3].clone(), b_components[4].clone(), b_components[5].clone(),
        b_components[6].clone(), b_components[7].clone(),
    ];

    // Benchmark packing
    c.bench_function("v4_pack", |bencher| {
        bencher.iter(|| {
            pack_multivector(
                black_box(&a_array),
                black_box(batch_size),
                black_box(&rot_keys),
                black_box(&ckks_ctx),
            ).unwrap()
        });
    });

    // Pack for geometric product benchmark
    let a_packed = pack_multivector(&a_array, batch_size, &rot_keys, &ckks_ctx).unwrap();
    let b_packed = pack_multivector(&b_array, batch_size, &rot_keys, &ckks_ctx).unwrap();

    // Benchmark just the geometric product (without unpack)
    c.bench_function("v4_geometric_product_packed_only", |bencher| {
        bencher.iter(|| {
            geometric_product_packed(
                black_box(&a_packed),
                black_box(&b_packed),
                black_box(&rot_keys),
                black_box(&ckks_ctx),
            ).unwrap()
        });
    });
}

#[cfg(all(feature = "v4", feature = "v2-gpu-metal"))]
criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10); // Smaller sample size due to long runtime
    targets = bench_geometric_product_single, bench_geometric_product_with_breakdown
}

#[cfg(all(feature = "v4", feature = "v2-gpu-metal"))]
criterion_main!(benches);

#[cfg(not(all(feature = "v4", feature = "v2-gpu-metal")))]
fn main() {
    println!("This benchmark requires features: v4,v2-gpu-metal");
    println!("Run with: cargo bench --features v4,v2-gpu-metal --bench v4_geometric_product_bench");
}
