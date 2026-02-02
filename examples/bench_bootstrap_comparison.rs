//! V3 vs V4 Bootstrap Comparison Benchmark
//!
//! Compares bootstrap performance between:
//! - V3 CUDA: Standard CKKS bootstrap on single ciphertext
//! - V4 CUDA: Bootstrap on PackedMultivector (8 components in one ciphertext)
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3,v4 --example bench_bootstrap_comparison
//! ```

#[cfg(all(feature = "v2-gpu-cuda", feature = "v3", feature = "v4"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::backends::gpu_cuda::{
        ckks::{CudaCkksContext, CudaCiphertext as V2CudaCiphertext},
        device::CudaDeviceContext,
        relin_keys::CudaRelinKeys,
        rotation::CudaRotationContext,
        rotation_keys::CudaRotationKeys,
    };
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v3::bootstrapping::{
        BootstrapParams,
        cuda_bootstrap::{CudaBootstrapContext, CudaCiphertext as V3CudaCiphertext},
    };
    use ga_engine::clifford_fhe_v4::{
        PackedMultivector,
        bootstrapping::V4BootstrapContext,
    };
    use std::sync::Arc;
    use std::time::Instant;
    use rand::Rng;

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║           Bootstrap Comparison: V3 CUDA vs V4 CUDA                     ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Use N=1024 bootstrap parameters
    let params = CliffordFHEParams::new_v3_bootstrap_cuda_full()
        .map_err(|e| format!("Failed to create bootstrap params: {}", e))?;
    let n = params.n;
    let num_primes = params.moduli.len();

    println!("Parameters: N={}, {} primes", n, num_primes);
    println!();

    // Initialize CUDA contexts
    println!("Initializing CUDA contexts...");
    let device = Arc::new(CudaDeviceContext::new()?);
    let ckks_ctx = Arc::new(CudaCkksContext::new(params.clone())?);
    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);

    // Generate secret key for keys
    let mut rng = rand::thread_rng();
    let mut secret_key = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        let bit = rng.gen::<u64>() & 1;
        for prime_idx in 0..num_primes {
            secret_key[coeff_idx * num_primes + prime_idx] = bit;
        }
    }

    // Generate rotation keys
    println!("Generating rotation keys for bootstrap...");
    let mut rotation_keys = CudaRotationKeys::new(
        device.clone(),
        params.clone(),
        rotation_ctx.clone(),
        secret_key.clone(),
        16,
    )?;

    // Generate rotations needed for bootstrap (power-of-2 and their negatives)
    let mut rotations_needed = Vec::new();
    for i in 0..=(n / 2).trailing_zeros() {
        let rot = 1i32 << i;
        if rot <= (n / 2) as i32 {
            rotations_needed.push(rot);
            rotations_needed.push(-rot);
        }
    }
    // Also need small rotations for packing
    for rot in 1..=8 {
        rotations_needed.push(rot);
        rotations_needed.push(-rot);
    }

    for rot in &rotations_needed {
        rotation_keys.generate_rotation_key_gpu(*rot, ckks_ctx.ntt_contexts())?;
    }
    println!("  Generated {} rotation keys", rotation_keys.num_keys());

    // Generate relinearization keys
    println!("Generating relinearization keys...");
    let relin_keys = Arc::new(CudaRelinKeys::new(
        device.clone(),
        params.clone(),
        secret_key.clone(),
        16,
    )?);

    let rotation_keys = Arc::new(rotation_keys);

    // Bootstrap parameters
    let bootstrap_params = BootstrapParams::balanced();
    println!("Bootstrap params: sin_degree={}", bootstrap_params.sin_degree);
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // V3 CUDA Bootstrap
    // ═══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[V3 CUDA] Single Ciphertext Bootstrap");
    println!("═══════════════════════════════════════════════════════════════════════════");

    // Create test ciphertext for V3
    let level = num_primes - 2;
    let scale = params.scale;

    let mut c0_v3 = vec![0u64; n * (level + 1)];
    let mut c1_v3 = vec![0u64; n * (level + 1)];
    for j in 0..c0_v3.len() {
        let prime_idx = j % (level + 1);
        let q = params.moduli[prime_idx];
        c0_v3[j] = rng.gen::<u64>() % q;
        c1_v3[j] = rng.gen::<u64>() % q;
    }

    let v3_ct = V3CudaCiphertext {
        c0: c0_v3,
        c1: c1_v3,
        n,
        num_primes: level + 1,
        level,
        scale,
    };

    // Create V3 bootstrap context
    let v3_bootstrap_ctx = CudaBootstrapContext::new(
        ckks_ctx.clone(),
        rotation_ctx.clone(),
        rotation_keys.clone(),
        relin_keys.clone(),
        bootstrap_params.clone(),
        params.clone(),
    )?;

    println!("Running V3 bootstrap...");
    let v3_start = Instant::now();
    let _v3_result = v3_bootstrap_ctx.bootstrap(&v3_ct)?;
    let v3_time = v3_start.elapsed().as_secs_f64();
    println!("[V3 CUDA] Bootstrap time: {:.2}s", v3_time);
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // V4 CUDA Bootstrap
    // ═══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[V4 CUDA] Packed Multivector Bootstrap (8 components)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    // Create test packed multivector for V4
    let mut c0_v2 = vec![0u64; n * (level + 1)];
    let mut c1_v2 = vec![0u64; n * (level + 1)];
    for j in 0..c0_v2.len() {
        let prime_idx = j % (level + 1);
        let q = params.moduli[prime_idx];
        c0_v2[j] = rng.gen::<u64>() % q;
        c1_v2[j] = rng.gen::<u64>() % q;
    }

    let v2_ct = V2CudaCiphertext {
        c0: c0_v2,
        c1: c1_v2,
        n,
        num_primes: level + 1,
        level,
        scale,
    };

    // For N=1024, batch_size = N/2/8 = 64
    let batch_size = n / 2 / 8;
    let v4_mv = PackedMultivector::new(
        v2_ct,
        batch_size,
        n,
        level + 1,
        level,
        scale,
    );

    // Create V4 bootstrap context
    let v4_bootstrap_ctx = V4BootstrapContext::new(
        ckks_ctx.clone(),
        rotation_ctx.clone(),
        rotation_keys.clone(),
        relin_keys.clone(),
        bootstrap_params.clone(),
        params.clone(),
    )?;

    println!("Running V4 bootstrap...");
    let v4_start = Instant::now();
    let _v4_result = v4_bootstrap_ctx.bootstrap(&v4_mv)?;
    let v4_time = v4_start.elapsed().as_secs_f64();
    println!("[V4 CUDA] Bootstrap time: {:.2}s", v4_time);
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════════════
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                         BENCHMARK SUMMARY                              ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("┌───────────┬─────────────────┬──────────────────┬───────────────────────┐");
    println!("│  Version  │  Total Time (s) │ Components/Boot  │ Time per Component    │");
    println!("├───────────┼─────────────────┼──────────────────┼───────────────────────┤");
    println!("│ V3 CUDA   │ {:>13.2}   │ {:>14}   │ {:>19.2}s │",
        v3_time, 1, v3_time);
    println!("│ V4 CUDA   │ {:>13.2}   │ {:>14}   │ {:>19.2}s │",
        v4_time, 8, v4_time / 8.0);
    println!("└───────────┴─────────────────┴──────────────────┴───────────────────────┘");
    println!();

    let effective_speedup = v3_time / (v4_time / 8.0);
    println!("V4 Effective Speedup: {:.2}x per component", effective_speedup);
    println!();

    println!("Notes:");
    println!("  - V3: Bootstraps 1 ciphertext (1 FHE encrypted value)");
    println!("  - V4: Bootstraps 1 packed ciphertext (8 Clifford components)");
    println!("  - Both use same underlying CUDA bootstrap operations");
    println!("  - V4's per-component time represents amortized cost");
    println!();

    println!("Memory Usage:");
    println!("  - V3: 1 ciphertext for 1 component");
    println!("  - V4: 1 ciphertext for 8 components (8× reduction)");
    println!();

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                         BENCHMARK COMPLETE                             ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(not(all(feature = "v2-gpu-cuda", feature = "v3", feature = "v4")))]
fn main() {
    println!("This benchmark requires features: v2-gpu-cuda, v3, v4");
    println!("Run with: cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3,v4 --example bench_bootstrap_comparison");
}
