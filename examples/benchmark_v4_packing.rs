//! Benchmark V4 Packing Performance: Naive vs Butterfly
//!
//! Run with:
//! cargo run --release --example benchmark_v4_packing --features v4,v2-gpu-metal --no-default-features

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::rotation_keys::MetalRotationKeys;
use ga_engine::clifford_fhe_v4::packing::{pack_multivector, unpack_multivector};
use ga_engine::clifford_fhe_v4::packing_butterfly::{pack_multivector_butterfly, unpack_multivector_butterfly};
use std::time::Instant;

fn main() {
    println!("\n════════════════════════════════════════════════════════");
    println!("V4 Packing Benchmark: Naive vs Butterfly");
    println!("════════════════════════════════════════════════════════\n");

    // Setup
    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("Parameters:");
    println!("  Ring dimension: N = {}", params.n);
    println!("  Number of primes: {}", params.moduli.len());
    println!("  Rotations (naive): 7 pack + 7 unpack = 14 total");
    println!("  Rotations (butterfly): 3 pack + 3 unpack = 6 total");
    println!("  Expected speedup: ~2.3×\n");

    // Generate keys
    print!("Generating keys... ");
    let mut key_ctx = MetalKeyContext::new(params.clone()).expect("Failed to create key context");
    let (pk, _sk, _evk) = key_ctx.keygen().expect("Failed to generate keys");

    let mut rotation_steps: Vec<i32> = (1..=8).collect();
    rotation_steps.extend((-8..=-1).collect::<Vec<i32>>());

    let ckks_ctx = MetalCkksContext::new(params.clone()).expect("Failed to create CKKS context");
    let rot_keys = MetalRotationKeys::generate(
        ckks_ctx.device().clone(),
        &key_ctx.keygen().expect("key generation").1,
        &rotation_steps,
        &params,
        ckks_ctx.ntt_contexts(),
        20,
    ).expect("Failed to generate rotation keys");
    println!("done.\n");

    // Encrypt test multivector
    println!("Encrypting test multivector...");
    let num_slots = params.n / 2;
    let mut components = Vec::new();
    for i in 0..8 {
        let mut slots = vec![0.0; num_slots];
        slots[0] = (i + 1) as f64;
        let pt = ckks_ctx.encode(&slots).expect("Encode failed");
        let ct = ckks_ctx.encrypt(&pt, &pk).expect("Encrypt failed");
        components.push(ct);
    }
    let components_array: [_; 8] = components.try_into().unwrap();
    println!("  ✓ Encrypted 8 components\n");

    // Benchmark naive packing
    println!("────────────────────────────────────────────────────────");
    println!("Naive Packing (7 rotations)");
    println!("────────────────────────────────────────────────────────");

    print!("  Packing... ");
    let start = Instant::now();
    let packed_naive = pack_multivector(&components_array, 1, &rot_keys, &ckks_ctx)
        .expect("Naive packing failed");
    let pack_naive_time = start.elapsed();
    println!("{:.3}s", pack_naive_time.as_secs_f64());

    print!("  Unpacking... ");
    let start = Instant::now();
    let _unpacked_naive = unpack_multivector(&packed_naive, &rot_keys, &ckks_ctx)
        .expect("Naive unpacking failed");
    let unpack_naive_time = start.elapsed();
    println!("{:.3}s", unpack_naive_time.as_secs_f64());

    let total_naive = pack_naive_time + unpack_naive_time;
    println!("  Total: {:.3}s\n", total_naive.as_secs_f64());

    // Benchmark butterfly packing
    println!("────────────────────────────────────────────────────────");
    println!("Butterfly Packing (3 rotations)");
    println!("────────────────────────────────────────────────────────");

    print!("  Packing... ");
    let start = Instant::now();
    let packed_butterfly = pack_multivector_butterfly(&components_array, 1, &rot_keys, &ckks_ctx)
        .expect("Butterfly packing failed");
    let pack_butterfly_time = start.elapsed();
    println!("{:.3}s", pack_butterfly_time.as_secs_f64());

    print!("  Unpacking... ");
    let start = Instant::now();
    let _unpacked_butterfly = unpack_multivector_butterfly(&packed_butterfly, &rot_keys, &ckks_ctx)
        .expect("Butterfly unpacking failed");
    let unpack_butterfly_time = start.elapsed();
    println!("{:.3}s", unpack_butterfly_time.as_secs_f64());

    let total_butterfly = pack_butterfly_time + unpack_butterfly_time;
    println!("  Total: {:.3}s\n", total_butterfly.as_secs_f64());

    // Summary
    println!("════════════════════════════════════════════════════════");
    println!("RESULTS");
    println!("════════════════════════════════════════════════════════");
    println!("Naive:     {:.3}s (pack) + {:.3}s (unpack) = {:.3}s",
        pack_naive_time.as_secs_f64(),
        unpack_naive_time.as_secs_f64(),
        total_naive.as_secs_f64());
    println!("Butterfly: {:.3}s (pack) + {:.3}s (unpack) = {:.3}s",
        pack_butterfly_time.as_secs_f64(),
        unpack_butterfly_time.as_secs_f64(),
        total_butterfly.as_secs_f64());

    let speedup = total_naive.as_secs_f64() / total_butterfly.as_secs_f64();
    println!("\nSpeedup: {:.2}× faster", speedup);
    println!("════════════════════════════════════════════════════════\n");
}
