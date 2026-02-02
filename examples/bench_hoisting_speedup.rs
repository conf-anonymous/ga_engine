//! Quick benchmark to measure hoisting speedup in V4 butterfly unpacking

use ga_engine::clifford_fhe_v2::backends::gpu_metal::{
    ckks::{MetalCiphertext, MetalCkksContext},
    rotation_keys::MetalRotationKeys,
};
use ga_engine::clifford_fhe_v4::packed_multivector::PackedMultivector;
use std::time::Instant;

fn main() -> Result<(), String> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Hoisting Speedup: Butterfly Rotation Benchmark             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Setup FHE context
    println!("Setting up FHE context (N=4096, 5 primes)...");
    let n = 4096;
    let num_primes = 5;
    let scale = (1u64 << 40) as f64;

    let ctx_start = Instant::now();
    let ckks_ctx = MetalCkksContext::new(n, num_primes, scale)?;
    println!("  Context created: {:.3}s\n", ctx_start.elapsed().as_secs_f64());

    // Generate rotation keys for butterfly steps (1, 2, 4)
    println!("Generating rotation keys for steps [1, 2, 4]...");
    let keys_start = Instant::now();
    let rot_keys = ckks_ctx.generate_rotation_keys(&[1, 2, 4])?;
    println!("  Keys generated: {:.3}s\n", keys_start.elapsed().as_secs_f64());

    // Encode and encrypt a test multivector
    println!("Creating test multivector...");
    let batch_size = 16;
    let num_slots = n / 2;
    let mut values = vec![0.0; num_slots];
    for i in 0..batch_size {
        values[i * 8] = i as f64;
    }

    let pt = ckks_ctx.encode(&values)?;
    let ct = ckks_ctx.encrypt_with_pk(&pt)?;
    let packed = PackedMultivector::new(ct, batch_size, n, num_primes, num_primes - 1, scale);
    println!("  Test data ready\n");

    // Measure unpacking with hoisting
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  BENCHMARK: Butterfly Unpacking (with hoisting)             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut times_hoisting = Vec::new();
    let num_runs = 10;

    println!("Running {} iterations...", num_runs);
    for i in 1..=num_runs {
        let start = Instant::now();
        let _ = ga_engine::clifford_fhe_v4::packing_butterfly::unpack_multivector_butterfly(
            &packed,
            &rot_keys,
            &ckks_ctx,
        )?;
        let elapsed = start.elapsed().as_secs_f64();
        times_hoisting.push(elapsed);
        println!("  Run {:2}: {:.3}s", i, elapsed);
    }

    let mean_hoisting = times_hoisting.iter().sum::<f64>() / times_hoisting.len() as f64;
    let variance = times_hoisting.iter().map(|&t| (t - mean_hoisting).powi(2)).sum::<f64>() / times_hoisting.len() as f64;
    let stddev = variance.sqrt();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  RESULTS                                                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Butterfly Unpacking (WITH hoisting):");
    println!("  Mean:   {:.3}s", mean_hoisting);
    println!("  StdDev: {:.3}s ({:.1}%)", stddev, 100.0 * stddev / mean_hoisting);
    println!("  Min:    {:.3}s", times_hoisting.iter().cloned().fold(f64::INFINITY, f64::min));
    println!("  Max:    {:.3}s", times_hoisting.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    println!();

    // Show breakdown
    println!("Operation breakdown:");
    println!("  - Stage 1: 1 rotation by 4");
    println!("  - Stage 2: 2 rotations by 2 (hoisted separately for h0, h1)");
    println!("  - Stage 3: 4 rotations by 1 (hoisted separately for q0, q1, q2, q3)");
    println!("  - Total: 7 individual rotations (each using hoisting API)");
    println!();

    println!("Configuration:");
    println!("  Ring dimension (N): {}", n);
    println!("  RNS primes: {}", num_primes);
    println!("  Batch size: {}", batch_size);
    println!("  Components: 8 (unpacking 1 → 8 ciphertexts)");
    println!();

    println!("GPU Information:");
    println!("  Device: {}", ckks_ctx.device().device().name());
    println!("  Max threads: {}", ckks_ctx.device().device().max_threads_per_threadgroup().width);
    println!();

    println!("✅ Benchmark complete!");
    println!();

    println!("Note: Each rotation now uses the hoisting API internally,");
    println!("      which amortizes decompose+NTT costs. Since we're only");
    println!("      doing 1 rotation at a time here, the speedup isn't");
    println!("      visible in this benchmark. The real speedup comes in");
    println!("      scenarios where the SAME ciphertext is rotated multiple");
    println!("      times (e.g., batch API with multiple steps).");
    println!();

    Ok(())
}
