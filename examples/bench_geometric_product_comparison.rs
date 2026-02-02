//! V2 CPU vs V4 CUDA Geometric Product Comparison Benchmark
//!
//! Compares geometric product performance between:
//! - V2 CPU: GeometricContext with Rayon parallelization (8 separate ciphertexts)
//! - V4 CUDA: PackedMultivector with GPU acceleration (1 packed ciphertext)
//!
//! Tests both N=1024 (quick) and N=8192 (production) parameters.
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v4 --example bench_geometric_product_comparison
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda", feature = "v4"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::{
        backends::{
            cpu_optimized::{
                ckks::{CkksContext, Ciphertext, Plaintext},
                geometric::{GeometricContext, MultivectorCiphertext},
                keys::KeyContext,
                rns::RnsRepresentation,
            },
            gpu_cuda::{
                ckks::{CudaCkksContext, CudaCiphertext},
                device::CudaDeviceContext,
                rotation::CudaRotationContext,
                rotation_keys::CudaRotationKeys,
            },
        },
        params::CliffordFHEParams,
    };
    use ga_engine::clifford_fhe_v4::{
        pack_multivector,
        geometric_ops::geometric_product_packed,
    };
    use std::sync::Arc;
    use std::time::Instant;
    use rand::Rng;

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     Geometric Product Comparison: V2 CPU vs V4 CUDA                    ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Configuration 1: N=1024 (Quick Test)
    // ═══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Configuration 1: N=1024, 3 primes (Quick Test)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let (v2_time_1024, v4_time_1024) = benchmark_config(
        CliffordFHEParams::new_test_ntt_1024(),
        3, // trials
    )?;

    // ═══════════════════════════════════════════════════════════════════════════
    // Configuration 2: N=8192 (Production)
    // ═══════════════════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Configuration 2: N=8192, 9 primes (Production)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let (v2_time_8192, v4_time_8192) = benchmark_config(
        CliffordFHEParams::new_128bit(),
        3, // trials
    )?;

    // Print summary
    println!();
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                         BENCHMARK SUMMARY                              ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("┌─────────┬────────────┬──────────────────┬──────────────────┬───────────┐");
    println!("│    N    │   Primes   │   V2 CPU (ms)    │  V4 CUDA (ms)    │  Speedup  │");
    println!("├─────────┼────────────┼──────────────────┼──────────────────┼───────────┤");

    let speedup_1024 = v2_time_1024 / v4_time_1024;
    let speedup_8192 = v2_time_8192 / v4_time_8192;

    println!("│ {:>7} │ {:>10} │ {:>16.2} │ {:>16.2} │ {:>8.2}x │",
        1024, 3, v2_time_1024, v4_time_1024, speedup_1024);
    println!("│ {:>7} │ {:>10} │ {:>16.2} │ {:>16.2} │ {:>8.2}x │",
        8192, 9, v2_time_8192, v4_time_8192, speedup_8192);

    println!("└─────────┴────────────┴──────────────────┴──────────────────┴───────────┘");
    println!();

    println!("Implementation Details:");
    println!("  V2 CPU: GeometricContext with Rayon parallelization");
    println!("          - 8 separate ciphertexts (one per Clifford component)");
    println!("          - 64 ciphertext multiplications per geometric product");
    println!();
    println!("  V4 CUDA: PackedMultivector with GPU acceleration");
    println!("          - 1 packed ciphertext (8 components in SIMD slots)");
    println!("          - Diagonal multiply + rotation pattern");
    println!("          - 8x memory reduction vs V2");
    println!();

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                         BENCHMARK COMPLETE                             ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Benchmark both V2 CPU and V4 CUDA for a given parameter set
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda", feature = "v4"))]
fn benchmark_config(
    params: ga_engine::clifford_fhe_v2::params::CliffordFHEParams,
    trials: usize,
) -> Result<(f64, f64), String> {
    use ga_engine::clifford_fhe_v2::{
        backends::{
            cpu_optimized::{
                ckks::{CkksContext, Plaintext},
                geometric::GeometricContext,
                keys::KeyContext,
                rns::RnsRepresentation,
            },
            gpu_cuda::{
                ckks::{CudaCkksContext, CudaCiphertext},
                device::CudaDeviceContext,
                rotation::CudaRotationContext,
                rotation_keys::CudaRotationKeys,
            },
        },
    };
    use ga_engine::clifford_fhe_v4::{
        pack_multivector,
        geometric_ops::geometric_product_packed,
    };
    use std::sync::Arc;
    use std::time::Instant;
    use rand::Rng;

    let n = params.n;
    let num_primes = params.moduli.len();
    println!("Parameters: N={}, {} primes", n, num_primes);

    // ─────────────────────────────────────────────────────────────────────────
    // V2 CPU Benchmark
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[V2 CPU] Setting up context...");
    let geo_ctx = GeometricContext::new(params.clone());
    let ckks_ctx = CkksContext::new(params.clone());
    let (pk, _sk, evk) = geo_ctx.key_ctx.keygen();

    // Create test multivectors
    let moduli: Vec<u64> = params.moduli[..=params.max_level()].to_vec();

    let create_ct = |val: f64| -> ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext {
        let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); n];
        let scaled_val = (val * params.scale) as u64;
        coeffs[0] = RnsRepresentation::from_u64(scaled_val, &moduli);
        let pt = Plaintext::new(coeffs, params.scale, params.max_level());
        ckks_ctx.encrypt(&pt, &pk)
    };

    let mv_a = [
        create_ct(1.0), create_ct(2.0), create_ct(3.0), create_ct(4.0),
        create_ct(5.0), create_ct(6.0), create_ct(7.0), create_ct(8.0),
    ];
    let mv_b = [
        create_ct(0.5), create_ct(1.5), create_ct(2.5), create_ct(3.5),
        create_ct(4.5), create_ct(5.5), create_ct(6.5), create_ct(7.5),
    ];

    println!("[V2 CPU] Warming up...");
    let _ = geo_ctx.geometric_product(&mv_a, &mv_b, &evk);

    println!("[V2 CPU] Running {} trials...", trials);
    let mut v2_times = Vec::with_capacity(trials);
    for i in 0..trials {
        let start = Instant::now();
        let _ = geo_ctx.geometric_product(&mv_a, &mv_b, &evk);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        v2_times.push(elapsed);
        println!("  Trial {}: {:.2}ms", i + 1, elapsed);
    }
    let v2_avg = v2_times.iter().sum::<f64>() / v2_times.len() as f64;
    println!("[V2 CPU] Average: {:.2}ms", v2_avg);

    // ─────────────────────────────────────────────────────────────────────────
    // V4 CUDA Benchmark
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[V4 CUDA] Setting up context...");
    let device = Arc::new(CudaDeviceContext::new()?);
    let cuda_ctx = Arc::new(CudaCkksContext::new(params.clone())?);
    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);

    // Generate secret key for rotation keys
    let mut rng = rand::thread_rng();
    let mut secret_key = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        let bit = rng.gen::<u64>() & 1;
        for prime_idx in 0..num_primes {
            secret_key[coeff_idx * num_primes + prime_idx] = bit;
        }
    }

    println!("[V4 CUDA] Generating rotation keys...");
    let mut rotation_keys = CudaRotationKeys::new(
        device.clone(),
        params.clone(),
        rotation_ctx.clone(),
        secret_key.clone(),
        16, // gadget base
    )?;

    // Generate needed rotations
    for rot in 1..=8 {
        rotation_keys.generate_rotation_key_gpu(rot, cuda_ctx.ntt_contexts())?;
        rotation_keys.generate_rotation_key_gpu(-rot, cuda_ctx.ntt_contexts())?;
    }
    for i in 0..=(n/2).trailing_zeros() {
        let rot = 1i32 << i;
        if rot <= (n/2) as i32 {
            rotation_keys.generate_rotation_key_gpu(rot, cuda_ctx.ntt_contexts())?;
            rotation_keys.generate_rotation_key_gpu(-rot, cuda_ctx.ntt_contexts())?;
        }
    }
    println!("  Generated {} rotation keys", rotation_keys.num_keys());

    // Create CUDA ciphertexts
    let level = num_primes - 2;
    let scale = params.scale;

    let mut create_cuda_ct = || -> CudaCiphertext {
        let mut c0 = vec![0u64; n * (level + 1)];
        let mut c1 = vec![0u64; n * (level + 1)];
        for j in 0..c0.len() {
            let prime_idx = j % (level + 1);
            let q = params.moduli[prime_idx];
            c0[j] = rng.gen::<u64>() % q;
            c1[j] = rng.gen::<u64>() % q;
        }
        CudaCiphertext { c0, c1, n, num_primes: level + 1, level, scale }
    };

    let a_components: [CudaCiphertext; 8] = [
        create_cuda_ct(), create_cuda_ct(), create_cuda_ct(), create_cuda_ct(),
        create_cuda_ct(), create_cuda_ct(), create_cuda_ct(), create_cuda_ct(),
    ];
    let b_components: [CudaCiphertext; 8] = [
        create_cuda_ct(), create_cuda_ct(), create_cuda_ct(), create_cuda_ct(),
        create_cuda_ct(), create_cuda_ct(), create_cuda_ct(), create_cuda_ct(),
    ];

    println!("[V4 CUDA] Packing multivectors...");
    let start = Instant::now();
    let a_packed = pack_multivector(&a_components, &rotation_keys, &rotation_ctx, &cuda_ctx)?;
    let b_packed = pack_multivector(&b_components, &rotation_keys, &rotation_ctx, &cuda_ctx)?;
    println!("  Packing time: {:.3}s", start.elapsed().as_secs_f64());

    println!("[V4 CUDA] Warming up...");
    let _ = geometric_product_packed(&a_packed, &b_packed, &rotation_keys, &cuda_ctx)?;

    println!("[V4 CUDA] Running {} trials...", trials);
    let mut v4_times = Vec::with_capacity(trials);
    for i in 0..trials {
        let start = Instant::now();
        let _ = geometric_product_packed(&a_packed, &b_packed, &rotation_keys, &cuda_ctx)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        v4_times.push(elapsed);
        println!("  Trial {}: {:.2}ms", i + 1, elapsed);
    }
    let v4_avg = v4_times.iter().sum::<f64>() / v4_times.len() as f64;
    println!("[V4 CUDA] Average: {:.2}ms", v4_avg);

    Ok((v2_avg, v4_avg))
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda", feature = "v4")))]
fn main() {
    println!("This benchmark requires features: v2, v2-gpu-cuda, v4");
    println!("Run with: cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v4 --example bench_geometric_product_comparison");
}
