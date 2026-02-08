//! V3 Batched Geometric Product Benchmark at N=8192
//!
//! Benchmarks V3 Batched at production parameters (N=8192, 9 primes, 128-bit security)
//! with 512 multivectors packed into SIMD slots.
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v3 --example bench_v3_batched_n8192
//! ```

#[cfg(all(feature = "v2", feature = "v3"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::{
        backends::cpu_optimized::{
            ckks::CkksContext,
            geometric::GeometricContext,
            keys::KeyContext,
        },
        params::CliffordFHEParams,
    };
    use ga_engine::clifford_fhe_v3::batched::{
        BatchedMultivector,
        encoding::{encode_batch, decode_batch},
        geometric::geometric_product_batched,
    };
    use ga_engine::clifford_fhe_v3::bootstrapping::keys::generate_rotation_keys;
    use std::time::Instant;

    println!("========================================================================");
    println!("  V3 Batched Geometric Product Benchmark - N=8192 (Production)");
    println!("========================================================================");
    println!();

    // ─────────────────────────────────────────────────────────────────────────
    // Setup: N=8192, 9 primes, 128-bit security
    // ─────────────────────────────────────────────────────────────────────────
    let params = CliffordFHEParams::new_128bit();
    let n = params.n;
    let num_primes = params.moduli.len();
    let batch_size = BatchedMultivector::max_batch_size(n);

    println!("Parameters:");
    println!("  N = {}", n);
    println!("  Primes = {}", num_primes);
    println!("  Max batch size = {} multivectors", batch_size);
    println!("  Slot utilization = 100% ({} / {} slots)", batch_size * 8, n / 2);
    println!();

    // ─────────────────────────────────────────────────────────────────────────
    // Key Generation
    // ─────────────────────────────────────────────────────────────────────────
    println!("Generating keys...");
    let keygen_start = Instant::now();

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    println!("  Key generation: {:.2}s", keygen_start.elapsed().as_secs_f64());

    // Generate rotation keys for component extraction (-7..=7)
    let rotations: Vec<i32> = (-7..=7).collect();
    let rot_start = Instant::now();
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);
    println!("  Rotation keys: {:.2}s", rot_start.elapsed().as_secs_f64());
    println!();

    // ─────────────────────────────────────────────────────────────────────────
    // Create batch of 512 random multivectors
    // ─────────────────────────────────────────────────────────────────────────
    println!("Encoding {} multivectors into SIMD batch...", batch_size);
    let encode_start = Instant::now();

    let multivectors_a: Vec<[f64; 8]> = (0..batch_size)
        .map(|i| {
            let base = (i as f64) * 0.01;
            [base, base+1.0, base+2.0, base+3.0, base+4.0, base+5.0, base+6.0, base+7.0]
        })
        .collect();
    let multivectors_b: Vec<[f64; 8]> = (0..batch_size)
        .map(|i| {
            let base = (i as f64) * 0.005;
            [base+0.5, base+1.5, base+2.5, base+3.5, base+4.5, base+5.5, base+6.5, base+7.5]
        })
        .collect();

    let batch_a = encode_batch(&multivectors_a, &ckks_ctx, &pk);
    let batch_b = encode_batch(&multivectors_b, &ckks_ctx, &pk);

    println!("  Encoding time: {:.2}s", encode_start.elapsed().as_secs_f64());
    println!();

    // ─────────────────────────────────────────────────────────────────────────
    // V3 Batched Benchmark (N=8192, 512 products)
    // ─────────────────────────────────────────────────────────────────────────
    println!("========================================================================");
    println!("  V3 Batched Geometric Product (N={}, batch={})", n, batch_size);
    println!("========================================================================");

    // Warmup
    println!("  Warming up...");
    let _ = geometric_product_batched(&batch_a, &batch_b, &rotation_keys, &evk, &ckks_ctx)?;

    // Benchmark
    println!("  Running 3 trials...");
    let mut v3_times = Vec::new();
    for i in 0..3 {
        let start = Instant::now();
        let _ = geometric_product_batched(&batch_a, &batch_b, &rotation_keys, &evk, &ckks_ctx)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        v3_times.push(elapsed);
        println!("    Trial {}: {:.2}ms total ({:.2}ms per product)",
            i + 1, elapsed, elapsed / batch_size as f64);
    }
    let v3_avg = v3_times.iter().sum::<f64>() / v3_times.len() as f64;
    let v3_per_product = v3_avg / batch_size as f64;

    println!();
    println!("  Average: {:.2}ms total", v3_avg);
    println!("  Per product: {:.2}ms", v3_per_product);
    println!();

    // ─────────────────────────────────────────────────────────────────────────
    // V2 CPU Single Product Baseline (for speedup comparison)
    // ─────────────────────────────────────────────────────────────────────────
    println!("========================================================================");
    println!("  V2 CPU Single Geometric Product (N={}, baseline)", n);
    println!("========================================================================");

    let geo_ctx = GeometricContext::new(params.clone());
    let moduli: Vec<u64> = params.moduli[..=params.max_level()].to_vec();

    let create_ct = |val: f64| {
        use ga_engine::clifford_fhe_v2::backends::cpu_optimized::{
            ckks::Plaintext,
            rns::RnsRepresentation,
        };
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
    let mut v2_times = Vec::new();
    for i in 0..3 {
        let start = Instant::now();
        let _ = geo_ctx.geometric_product(&mv_a, &mv_b, &evk);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        v2_times.push(elapsed);
        println!("    Trial {}: {:.2}ms", i + 1, elapsed);
    }
    let v2_avg = v2_times.iter().sum::<f64>() / v2_times.len() as f64;

    // ─────────────────────────────────────────────────────────────────────────
    // Correctness Verification
    // ─────────────────────────────────────────────────────────────────────────
    println!();
    println!("========================================================================");
    println!("  Correctness Verification");
    println!("========================================================================");

    let result = geometric_product_batched(&batch_a, &batch_b, &rotation_keys, &evk, &ckks_ctx)?;
    let decoded = decode_batch(&result, &ckks_ctx, &sk);

    // Verify first multivector pair against plaintext computation
    let a0 = &multivectors_a[0];
    let b0 = &multivectors_b[0];

    // Plaintext geometric product: scalar component = sum(a_i * b_i * sign)
    let expected_scalar = a0[0]*b0[0] + a0[1]*b0[1] + a0[2]*b0[2] + a0[3]*b0[3]
                         - a0[4]*b0[4] - a0[5]*b0[5] - a0[6]*b0[6] - a0[7]*b0[7];

    println!("  First MV pair (plaintext vs encrypted):");
    println!("    Expected scalar: {:.4}", expected_scalar);
    println!("    Decrypted scalar: {:.4}", decoded[0][0]);
    println!("    Error: {:.6}", (decoded[0][0] - expected_scalar).abs());
    println!();

    // ─────────────────────────────────────────────────────────────────────────
    // Summary
    // ─────────────────────────────────────────────────────────────────────────
    let speedup = v2_avg / v3_per_product;

    println!("========================================================================");
    println!("  BENCHMARK SUMMARY (N={})", n);
    println!("========================================================================");
    println!();
    println!("  +-------------------+--------------+--------------+-----------+");
    println!("  | Implementation    | Total (ms)   | Per Product  | Speedup   |");
    println!("  +-------------------+--------------+--------------+-----------+");
    println!("  | V2 CPU Single     | {:>10.2}   | {:>10.2}   | {:>7.1}x  |",
        v2_avg, v2_avg, 1.0);
    println!("  | V3 Batched ({:>3})  | {:>10.2}   | {:>10.2}   | {:>7.1}x  |",
        batch_size, v3_avg, v3_per_product, speedup);
    println!("  +-------------------+--------------+--------------+-----------+");
    println!();

    let meets_target = v3_per_product < 50.0;
    println!("  Production target: <50ms per product");
    println!("  Result: {:.2}ms per product - {}",
        v3_per_product,
        if meets_target { "MEETS TARGET" } else { "EXCEEDS TARGET" }
    );
    println!();

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v3")))]
fn main() {
    println!("This benchmark requires features: v2, v3");
    println!("Run with: cargo run --release --no-default-features --features f64,nd,v2,v3 --example bench_v3_batched_n8192");
}
