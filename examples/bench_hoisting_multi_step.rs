//! Micro-benchmark: Hoisting speedup for multi-step rotations (R>>1)
//!
//! This tests the IDEAL case for hoisting: rotating the SAME ciphertext
//! by multiple steps [1,2,4,8,16,32]. Expected speedup: ~2×-3×

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::{
    ckks::MetalCkksContext,
    keys::MetalKeyContext,
    rotation_keys::MetalRotationKeys,
};
use std::time::Instant;

fn main() -> Result<(), String> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Hoisting Speedup: Multi-Step Rotation Benchmark (R>>1)     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Setup FHE context
    let params = CliffordFHEParams::new_test_ntt_1024();

    println!("Configuration:");
    println!("  N = {}", params.n);
    println!("  max_level = {}", params.max_level());
    println!("  Rotation steps = [1, 2, 4, 8, 16, 32]");
    println!();

    println!("Setting up FHE context...");
    let ctx_start = Instant::now();
    let ctx = MetalCkksContext::new(params.clone())?;
    println!("  Context: {:.3}s", ctx_start.elapsed().as_secs_f64());

    // Generate keys
    println!("Generating keys...");
    let keys_start = Instant::now();
    let mut key_ctx = MetalKeyContext::new(params.clone())?;
    let (pk, sk, _evk) = key_ctx.keygen()?;
    println!("  Keygen: {:.3}s", keys_start.elapsed().as_secs_f64());

    // Generate rotation keys
    println!("Generating rotation keys...");
    let rotkeys_start = Instant::now();
    let steps = vec![1, 2, 4, 8, 16, 32];
    let rot_keys = MetalRotationKeys::generate(
        ctx.device().clone(),
        &sk,
        &steps,
        &params,
        ctx.ntt_contexts(),
        20, // base_w
    )?;
    println!("  Rotation keys: {:.3}s\n", rotkeys_start.elapsed().as_secs_f64());

    // Create test ciphertext
    let num_slots = params.n / 2;
    let values: Vec<f64> = (0..num_slots).map(|i| i as f64).collect();
    let pt = ctx.encode(&values)?;
    let ct = ctx.encrypt(&pt, &pk)?;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  METHOD 1: Naive (separate rotate_by_steps calls)           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut naive_times = Vec::new();
    for run in 1..=5 {
        let start = Instant::now();

        // Rotate separately (decompose+NTT done 6 times)
        let _r1 = ct.rotate_by_steps(1, &rot_keys, &ctx)?;
        let _r2 = ct.rotate_by_steps(2, &rot_keys, &ctx)?;
        let _r4 = ct.rotate_by_steps(4, &rot_keys, &ctx)?;
        let _r8 = ct.rotate_by_steps(8, &rot_keys, &ctx)?;
        let _r16 = ct.rotate_by_steps(16, &rot_keys, &ctx)?;
        let _r32 = ct.rotate_by_steps(32, &rot_keys, &ctx)?;

        let elapsed = start.elapsed().as_secs_f64();
        naive_times.push(elapsed);
        println!("  Run {}: {:.3}s", run, elapsed);
    }

    let mean_naive = naive_times.iter().sum::<f64>() / naive_times.len() as f64;
    println!("\n  Mean: {:.3}s", mean_naive);
    println!("  Per rotation: {:.3}s", mean_naive / 6.0);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  METHOD 2: Hoisted (batch rotate_batch_with_hoisting)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut hoisted_times = Vec::new();
    for run in 1..=5 {
        let start = Instant::now();

        // Rotate with hoisting (decompose+NTT done ONCE)
        let _results = ct.rotate_batch_with_hoisting(&steps, &rot_keys, &ctx)?;

        let elapsed = start.elapsed().as_secs_f64();
        hoisted_times.push(elapsed);
        println!("  Run {}: {:.3}s", run, elapsed);
    }

    let mean_hoisted = hoisted_times.iter().sum::<f64>() / hoisted_times.len() as f64;
    println!("\n  Mean: {:.3}s", mean_hoisted);
    println!("  Per rotation: {:.3}s", mean_hoisted / 6.0);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  SPEEDUP ANALYSIS                                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let speedup = mean_naive / mean_hoisted;
    let percent_improvement = 100.0 * (mean_naive - mean_hoisted) / mean_naive;

    println!("Results:");
    println!("  Naive total:   {:.3}s", mean_naive);
    println!("  Hoisted total: {:.3}s", mean_hoisted);
    println!("  Speedup:       {:.2}×", speedup);
    println!("  Improvement:   {:.1}%", percent_improvement);
    println!();

    if speedup >= 2.0 {
        println!("✅ Excellent! Hoisting delivers expected 2×-3× speedup");
    } else if speedup >= 1.5 {
        println!("⚠️  Moderate speedup. Key-switch may be dominating.");
    } else {
        println!("❌ Low speedup. Key-switch/iNTT are the bottleneck.");
        println!("   Next step: Profile to get decompose/ks/iNTT breakdown.");
    }

    println!();
    println!("GPU Information:");
    println!("  Device: {}", ctx.device().device().name());
    println!();

    println!("Analysis:");
    println!("  - Naive does 6× decompose + 6× forward NTT");
    println!("  - Hoisted does 1× decompose + 1× forward NTT");
    println!("  - If speedup < 2×, key-switch/iNTT dominate (>60% cost)");
    println!("  - Next: Fuse kernels, pre-NTT keys, optimize layout");
    println!();

    Ok(())
}
