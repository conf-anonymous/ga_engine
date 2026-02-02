//! Unified Geometric Product Benchmark
//!
//! Compares all geometric product implementations:
//! - V2 CPU: GeometricContext with Rayon parallelization
//! - V2 CUDA: CudaGeometricProductContext with GPU acceleration
//! - V3 Batched: Slot-level parallelism for batch processing
//! - V4 CUDA: PackedMultivector with GPU acceleration
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3,v4 --example bench_all_geometric_products
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda", feature = "v3", feature = "v4"))]
fn main() -> Result<(), String> {
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
                geometric_product::CudaGeometricProductContext,
                relin_keys::CudaRelinKeys,
                rotation::CudaRotationContext,
                rotation_keys::CudaRotationKeys,
            },
        },
        params::CliffordFHEParams,
    };
    use ga_engine::clifford_fhe_v3::batched::{
        BatchedMultivector,
        encoding::encode_batch,
        geometric::geometric_product_batched,
    };
    use ga_engine::clifford_fhe_v3::bootstrapping::keys::generate_rotation_keys;
    use ga_engine::clifford_fhe_v4::{
        pack_multivector,
        geometric_ops::geometric_product_packed,
    };
    use std::sync::Arc;
    use std::time::Instant;
    use rand::Rng;

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║           Unified Geometric Product Benchmark                          ║");
    println!("║   V2 CPU | V2 CUDA | V3 Batched | V4 CUDA                              ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Use N=1024 for quick comparison (all versions)
    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();
    println!("Parameters: N={}, {} primes", n, num_primes);
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // V2 CPU Benchmark
    // ═══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[V2 CPU] Geometric Product");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let geo_ctx = GeometricContext::new(params.clone());
    let ckks_ctx = CkksContext::new(params.clone());
    let (pk, sk, evk) = geo_ctx.key_ctx.keygen();

    // Create test multivectors
    let moduli: Vec<u64> = params.moduli[..=params.max_level()].to_vec();

    let create_ct = |val: f64| {
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

    println!("  Warming up...");
    let _ = geo_ctx.geometric_product(&mv_a, &mv_b, &evk);

    println!("  Running 3 trials...");
    let mut v2_cpu_times = Vec::new();
    for i in 0..3 {
        let start = Instant::now();
        let _ = geo_ctx.geometric_product(&mv_a, &mv_b, &evk);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        v2_cpu_times.push(elapsed);
        println!("    Trial {}: {:.2}ms", i + 1, elapsed);
    }
    let v2_cpu_avg = v2_cpu_times.iter().sum::<f64>() / v2_cpu_times.len() as f64;
    println!("  [V2 CPU] Average: {:.2}ms", v2_cpu_avg);
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // V2 CUDA Benchmark
    // ═══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[V2 CUDA] Geometric Product");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let device = Arc::new(CudaDeviceContext::new()?);
    let cuda_ctx = Arc::new(CudaCkksContext::new(params.clone())?);

    // Generate relinearization keys for V2 CUDA
    let mut rng = rand::thread_rng();
    let mut secret_key = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        let bit = rng.gen::<u64>() & 1;
        for prime_idx in 0..num_primes {
            secret_key[coeff_idx * num_primes + prime_idx] = bit;
        }
    }

    println!("  Generating relinearization keys...");
    let relin_keys = CudaRelinKeys::new(
        device.clone(),
        params.clone(),
        secret_key.clone(),
        16,
    )?;

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

    let cuda_mv_a: [CudaCiphertext; 8] = [
        create_cuda_ct(), create_cuda_ct(), create_cuda_ct(), create_cuda_ct(),
        create_cuda_ct(), create_cuda_ct(), create_cuda_ct(), create_cuda_ct(),
    ];
    let cuda_mv_b: [CudaCiphertext; 8] = [
        create_cuda_ct(), create_cuda_ct(), create_cuda_ct(), create_cuda_ct(),
        create_cuda_ct(), create_cuda_ct(), create_cuda_ct(), create_cuda_ct(),
    ];

    let cuda_geo_ctx = CudaGeometricProductContext::new();

    println!("  Warming up...");
    let _ = cuda_geo_ctx.geometric_product(&cuda_mv_a, &cuda_mv_b, &relin_keys, &cuda_ctx)?;

    println!("  Running 3 trials...");
    let mut v2_cuda_times = Vec::new();
    for i in 0..3 {
        let start = Instant::now();
        let _ = cuda_geo_ctx.geometric_product(&cuda_mv_a, &cuda_mv_b, &relin_keys, &cuda_ctx)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        v2_cuda_times.push(elapsed);
        println!("    Trial {}: {:.2}ms", i + 1, elapsed);
    }
    let v2_cuda_avg = v2_cuda_times.iter().sum::<f64>() / v2_cuda_times.len() as f64;
    println!("  [V2 CUDA] Average: {:.2}ms", v2_cuda_avg);
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // V3 Batched Benchmark
    // ═══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[V3 Batched] Geometric Product");
    println!("═══════════════════════════════════════════════════════════════════════════");

    // For N=1024, max batch size is 64 (1024/2/8 = 64)
    let batch_size = BatchedMultivector::max_batch_size(n);
    println!("  Batch size: {} multivectors", batch_size);

    // Create batch of multivectors
    let multivectors_a: Vec<[f64; 8]> = (0..batch_size)
        .map(|i| {
            let base = (i as f64) * 0.1;
            [base, base+1.0, base+2.0, base+3.0, base+4.0, base+5.0, base+6.0, base+7.0]
        })
        .collect();
    let multivectors_b: Vec<[f64; 8]> = (0..batch_size)
        .map(|i| {
            let base = (i as f64) * 0.05;
            [base+0.5, base+1.5, base+2.5, base+3.5, base+4.5, base+5.5, base+6.5, base+7.5]
        })
        .collect();

    let batch_a = encode_batch(&multivectors_a, &ckks_ctx, &pk);
    let batch_b = encode_batch(&multivectors_b, &ckks_ctx, &pk);

    // Generate rotation keys for extraction (0-7 and negatives)
    let rotations: Vec<i32> = (-7..=7).collect();
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

    println!("  Warming up...");
    let _ = geometric_product_batched(&batch_a, &batch_b, &rotation_keys, &evk, &ckks_ctx)?;

    println!("  Running 3 trials...");
    let mut v3_batch_times = Vec::new();
    for i in 0..3 {
        let start = Instant::now();
        let _ = geometric_product_batched(&batch_a, &batch_b, &rotation_keys, &evk, &ckks_ctx)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        v3_batch_times.push(elapsed);
        println!("    Trial {}: {:.2}ms", i + 1, elapsed);
    }
    let v3_batch_avg = v3_batch_times.iter().sum::<f64>() / v3_batch_times.len() as f64;
    let v3_per_product = v3_batch_avg / batch_size as f64;
    println!("  [V3 Batched] Average: {:.2}ms total, {:.2}ms per product", v3_batch_avg, v3_per_product);
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // V4 CUDA Benchmark
    // ═══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[V4 CUDA] Geometric Product");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);

    println!("  Generating rotation keys...");
    let mut cuda_rotation_keys = CudaRotationKeys::new(
        device.clone(),
        params.clone(),
        rotation_ctx.clone(),
        secret_key.clone(),
        16,
    )?;

    // Generate needed rotations for V4
    for rot in 1..=8 {
        cuda_rotation_keys.generate_rotation_key_gpu(rot, cuda_ctx.ntt_contexts())?;
        cuda_rotation_keys.generate_rotation_key_gpu(-rot, cuda_ctx.ntt_contexts())?;
    }
    for i in 0..=(n/2).trailing_zeros() {
        let rot = 1i32 << i;
        if rot <= (n/2) as i32 {
            cuda_rotation_keys.generate_rotation_key_gpu(rot, cuda_ctx.ntt_contexts())?;
            cuda_rotation_keys.generate_rotation_key_gpu(-rot, cuda_ctx.ntt_contexts())?;
        }
    }
    println!("  Generated {} rotation keys", cuda_rotation_keys.num_keys());

    // Pack multivectors for V4
    println!("  Packing multivectors...");
    let start = Instant::now();
    let a_packed = pack_multivector(&cuda_mv_a, &cuda_rotation_keys, &rotation_ctx, &cuda_ctx)?;
    let b_packed = pack_multivector(&cuda_mv_b, &cuda_rotation_keys, &rotation_ctx, &cuda_ctx)?;
    println!("  Packing time: {:.3}s", start.elapsed().as_secs_f64());

    println!("  Warming up...");
    let _ = geometric_product_packed(&a_packed, &b_packed, &cuda_rotation_keys, &cuda_ctx)?;

    println!("  Running 3 trials...");
    let mut v4_cuda_times = Vec::new();
    for i in 0..3 {
        let start = Instant::now();
        let _ = geometric_product_packed(&a_packed, &b_packed, &cuda_rotation_keys, &cuda_ctx)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        v4_cuda_times.push(elapsed);
        println!("    Trial {}: {:.2}ms", i + 1, elapsed);
    }
    let v4_cuda_avg = v4_cuda_times.iter().sum::<f64>() / v4_cuda_times.len() as f64;
    println!("  [V4 CUDA] Average: {:.2}ms", v4_cuda_avg);
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════════════
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                         BENCHMARK SUMMARY (N={})                      ║", n);
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("┌──────────────┬──────────────────┬────────────────┬───────────────────┐");
    println!("│   Version    │   Time (ms)      │ Per Product    │ Speedup vs V2 CPU │");
    println!("├──────────────┼──────────────────┼────────────────┼───────────────────┤");
    println!("│ V2 CPU       │ {:>14.2}   │ {:>12.2}   │ {:>15.2}x │",
        v2_cpu_avg, v2_cpu_avg, 1.0);
    println!("│ V2 CUDA      │ {:>14.2}   │ {:>12.2}   │ {:>15.2}x │",
        v2_cuda_avg, v2_cuda_avg, v2_cpu_avg / v2_cuda_avg);
    println!("│ V3 Batched   │ {:>14.2}   │ {:>12.2}   │ {:>15.2}x │",
        v3_batch_avg, v3_per_product, v2_cpu_avg / v3_per_product);
    println!("│ V4 CUDA      │ {:>14.2}   │ {:>12.2}   │ {:>15.2}x │",
        v4_cuda_avg, v4_cuda_avg, v2_cpu_avg / v4_cuda_avg);
    println!("└──────────────┴──────────────────┴────────────────┴───────────────────┘");
    println!();

    println!("Notes:");
    println!("  - V2 CPU: 8 separate ciphertexts, 64 multiplications");
    println!("  - V2 CUDA: Same as V2 CPU but with GPU multiplications");
    println!("  - V3 Batched: {} products computed in parallel via SIMD slots", batch_size);
    println!("  - V4 CUDA: 1 packed ciphertext, GPU diagonal multiply + rotations");
    println!();

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                         BENCHMARK COMPLETE                             ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda", feature = "v3", feature = "v4")))]
fn main() {
    println!("This benchmark requires features: v2, v2-gpu-cuda, v3, v4");
    println!("Run with: cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3,v4 --example bench_all_geometric_products");
}
