//! Performance Test for V4 Packed Multivector Geometric Product
//!
//! Measures timing for the packed geometric product operation on Metal GPU.
//! Provides detailed timing breakdowns for each operation.

#[cfg(all(feature = "v4", feature = "v2-gpu-metal"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::{
        ckks::MetalCkksContext,
        rotation_keys::MetalRotationKeys,
    };
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v4::{
        packing::pack_multivector,
        geometric_ops::geometric_product_packed,
    };
    use std::time::Instant;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  V4 Packed Geometric Product - Performance Benchmark        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Setup
    println!("Setting up FHE context...");
    let params = CliffordFHEParams::default();
    let key_ctx = KeyContext::new(params.clone());

    let start = Instant::now();
    let (pk, sk, _evk) = key_ctx.keygen();
    let keygen_time = start.elapsed();
    println!("  Key generation: {:.3}s\n", keygen_time.as_secs_f64());

    let start = Instant::now();
    let ckks_ctx = MetalCkksContext::new(params.clone())?;
    let ctx_time = start.elapsed();
    println!("  Metal CKKS context: {:.3}s", ctx_time.as_secs_f64());
    println!("  Device: {}\n", ckks_ctx.device().device().name());

    println!("Generating rotation keys...");
    let mut rotation_steps: Vec<i32> = (1..=8).collect();
    rotation_steps.extend((-8..=-1).collect::<Vec<i32>>());

    let start = Instant::now();
    let rot_keys = MetalRotationKeys::generate(
        ckks_ctx.device().clone(),
        &sk,
        &rotation_steps,
        &params,
        ckks_ctx.ntt_contexts(),
        20, // base_w
    )?;
    let rotkey_time = start.elapsed();
    println!("  Rotation keys: {:.3}s\n", rotkey_time.as_secs_f64());

    // Create test multivectors: a = 1 + 2e₁, b = 3e₂
    println!("Creating test multivectors...");
    let batch_size = 1;
    let num_slots = params.n / 2;

    let mut a_vals = vec![vec![0.0; num_slots]; 8];
    a_vals[0][0] = 1.0;  // scalar = 1
    a_vals[1][0] = 2.0;  // e1 = 2

    let mut b_vals = vec![vec![0.0; num_slots]; 8];
    b_vals[2][0] = 3.0;  // e2 = 3

    // Encode and encrypt
    let start = Instant::now();
    let mut a_components = Vec::new();
    let mut b_components = Vec::new();

    for i in 0..8 {
        let a_pt = ckks_ctx.encode(&a_vals[i])?;
        let a_ct = ckks_ctx.encrypt(&a_pt, &pk)?;
        a_components.push(a_ct);

        let b_pt = ckks_ctx.encode(&b_vals[i])?;
        let b_ct = ckks_ctx.encrypt(&b_pt, &pk)?;
        b_components.push(b_ct);
    }
    let encrypt_time = start.elapsed();
    println!("  Encode & encrypt (16 components): {:.3}s\n", encrypt_time.as_secs_f64());

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

    // === BENCHMARK START ===
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  BENCHMARK: Geometric Product Pipeline                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Benchmark packing (warm-up)
    println!("Warm-up: Packing multivectors...");
    let _ = pack_multivector(&a_array, batch_size, &rot_keys, &ckks_ctx)?;
    let _ = pack_multivector(&b_array, batch_size, &rot_keys, &ckks_ctx)?;

    // Benchmark: Pack A
    let start = Instant::now();
    let a_packed = pack_multivector(&a_array, batch_size, &rot_keys, &ckks_ctx)?;
    let pack_a_time = start.elapsed();
    println!("  [1] Pack A (8 → 1 ciphertext): {:.3}s", pack_a_time.as_secs_f64());

    // Benchmark: Pack B
    let start = Instant::now();
    let b_packed = pack_multivector(&b_array, batch_size, &rot_keys, &ckks_ctx)?;
    let pack_b_time = start.elapsed();
    println!("  [2] Pack B (8 → 1 ciphertext): {:.3}s", pack_b_time.as_secs_f64());

    // Benchmark: Geometric Product
    println!("\n🔥 Computing geometric product on Metal GPU...");
    let start = Instant::now();
    let result_packed = geometric_product_packed(&a_packed, &b_packed, &rot_keys, &ckks_ctx)?;
    let gp_time = start.elapsed();
    println!("  [3] Geometric Product: {:.3}s ✨\n", gp_time.as_secs_f64());

    // Total time
    let total_time = pack_a_time + pack_b_time + gp_time;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RESULTS                                                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Operation Breakdown:");
    println!("  Pack A:              {:7.3}s  ({:5.1}%)", pack_a_time.as_secs_f64(), 100.0 * pack_a_time.as_secs_f64() / total_time.as_secs_f64());
    println!("  Pack B:              {:7.3}s  ({:5.1}%)", pack_b_time.as_secs_f64(), 100.0 * pack_b_time.as_secs_f64() / total_time.as_secs_f64());
    println!("  Geometric Product:   {:7.3}s  ({:5.1}%)", gp_time.as_secs_f64(), 100.0 * gp_time.as_secs_f64() / total_time.as_secs_f64());
    println!("  ─────────────────────────────────────────────");
    println!("  TOTAL:               {:7.3}s", total_time.as_secs_f64());
    println!();

    println!("Configuration:");
    println!("  Ring dimension (N):  {}", params.n);
    println!("  RNS primes:          {}", params.moduli.len());
    println!("  Scale:               2^{}", (params.scale as f64).log2() as u32);
    println!("  Batch size:          {}", batch_size);
    println!("  Components:          8 (Cl(3,0))");
    println!();

    println!("Memory Footprint:");
    println!("  Input:  2 multivectors = 2 packed ciphertexts");
    println!("  Output: 1 multivector  = 1 packed ciphertext");
    println!("  (vs V3: 16 ciphertexts → 8× reduction)");
    println!();

    println!("GPU Information:");
    println!("  Device: {}", ckks_ctx.device().device().name());
    println!("  Max threads: {}", ckks_ctx.device().device().max_threads_per_threadgroup().width);
    println!();

    // Run multiple iterations for more accurate timing
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  MULTI-ITERATION BENCHMARK (5 runs)                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut times = Vec::new();
    for i in 1..=5 {
        let start = Instant::now();
        let _ = geometric_product_packed(&a_packed, &b_packed, &rot_keys, &ckks_ctx)?;
        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64());
        println!("  Run {}: {:.3}s", i, elapsed.as_secs_f64());
    }

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let stddev = variance.sqrt();

    println!();
    println!("Statistics:");
    println!("  Mean:   {:.3}s", mean);
    println!("  StdDev: {:.3}s ({:.1}%)", stddev, 100.0 * stddev / mean);
    println!("  Min:    {:.3}s", times.iter().cloned().fold(f64::INFINITY, f64::min));
    println!("  Max:    {:.3}s", times.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    println!();

    println!("✅ Benchmark complete!");
    println!();

    // Verify result metadata
    println!("Result verification:");
    println!("  Batch size: {}", result_packed.batch_size);
    println!("  N: {}", result_packed.n);
    println!("  Num primes: {}", result_packed.num_primes);
    println!("  Level: {}", result_packed.level);
    println!("  Scale: {:.0}", result_packed.scale);
    println!();

    Ok(())
}

#[cfg(not(all(feature = "v4", feature = "v2-gpu-metal")))]
fn main() {
    println!("This benchmark requires features: v4,v2-gpu-metal");
    println!("Run with: cargo run --release --features v4,v2-gpu-metal --example bench_v4_geometric_product");
}
