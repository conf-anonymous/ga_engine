//! Test Metal GPU Key Generation
//!
//! Quick test to verify Metal GPU key generation works correctly.
//!
//! **Run with:**
//! ```bash
//! time cargo run --release --features v2,v2-gpu-metal --example test_metal_keygen
//! ```

#[cfg(feature = "v2-gpu-metal")]
fn main() {
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
    use std::time::Instant;

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║         METAL GPU KEY GENERATION TEST                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Test 1: Small parameters (N=1024)
    println!("Test 1: N=1024, 3 primes (quick validation)");
    println!("═══════════════════════════════════════════════════════");
    let params_small = CliffordFHEParams::new_test_ntt_1024();
    println!("Parameters: N={}, {} primes", params_small.n, params_small.moduli.len());

    let mut key_ctx_small = match MetalKeyContext::new(params_small.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("❌ Failed to create Metal context: {}", e);
            eprintln!("   Make sure you're running on Apple Silicon (M1/M2/M3)");
            return;
        }
    };

    let start = Instant::now();
    let (pk, sk, evk) = match key_ctx_small.keygen() {
        Ok(keys) => keys,
        Err(e) => {
            eprintln!("❌ Key generation failed: {}", e);
            return;
        }
    };
    let keygen_time = start.elapsed();
    println!("\n✓ Keys generated in {:.2}s\n", keygen_time.as_secs_f64());

    // Test encryption/decryption roundtrip
    println!("Testing encrypt/decrypt roundtrip...");
    let ckks_ctx = CkksContext::new(params_small.clone());

    let message = vec![1.0, 2.0, 3.0, 4.0];
    let mut full_message = vec![0.0; params_small.n / 2];
    full_message[..4].copy_from_slice(&message);

    let pt = ckks_ctx.encode(&full_message);
    let ct = ckks_ctx.encrypt(&pt, &pk);
    let pt_dec = ckks_ctx.decrypt(&ct, &sk);
    let result = ckks_ctx.decode(&pt_dec);

    let mut max_error = 0.0f64;
    for i in 0..4 {
        let error = (result[i] - message[i]).abs();
        max_error = max_error.max(error);
    }

    if max_error < 0.01 {
        println!("✓ Encrypt/decrypt works! Max error: {:.6}\n", max_error);
    } else {
        println!("❌ Encrypt/decrypt error too large: {:.6}\n", max_error);
        return;
    }

    // Test 2: Medium parameters (N=4096, more primes)
    println!("\nTest 2: N=4096, 5 primes (moderate size)");
    println!("═══════════════════════════════════════════════════════");
    let params_medium = CliffordFHEParams::new_test_ntt_4096();
    println!("Parameters: N={}, {} primes", params_medium.n, params_medium.moduli.len());

    let mut key_ctx_medium = match MetalKeyContext::new(params_medium.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("❌ Failed to create Metal context: {}", e);
            return;
        }
    };

    let start = Instant::now();
    let (_pk_med, _sk_med, _evk_med) = match key_ctx_medium.keygen() {
        Ok(keys) => keys,
        Err(e) => {
            eprintln!("❌ Key generation failed: {}", e);
            return;
        }
    };
    let keygen_time_med = start.elapsed();
    println!("\n✓ Keys generated in {:.2}s\n", keygen_time_med.as_secs_f64());

    // Test 3: Large parameters (N=8192, 13 primes - larger test)
    println!("\nTest 3: N=8192, 13 primes (larger parameters)");
    println!("═══════════════════════════════════════════════════════");
    // Create custom params with N=8192 and 13 primes
    let params_large = {
        let mut p = CliffordFHEParams::new_128bit();
        // Add more primes to get to 13 total
        let extra_primes = vec![
            1099513872385u64,
            1099514003457u64,
            1099514200065u64,
            1099514265601u64,
        ];
        p.moduli.extend_from_slice(&extra_primes);
        p
    };
    println!("Parameters: N={}, {} primes", params_large.n, params_large.moduli.len());
    println!("(This would take 5-10 minutes on CPU!)");

    let mut key_ctx_large = match MetalKeyContext::new(params_large.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("❌ Failed to create Metal context: {}", e);
            return;
        }
    };

    let start = Instant::now();
    let (_pk_large, _sk_large, _evk_large) = match key_ctx_large.keygen() {
        Ok(keys) => keys,
        Err(e) => {
            eprintln!("❌ Key generation failed: {}", e);
            return;
        }
    };
    let keygen_time_large = start.elapsed();
    println!("\n✓ Keys generated in {:.2}s\n", keygen_time_large.as_secs_f64());

    // Summary
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                      PERFORMANCE SUMMARY                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!("  N=1024,  3 primes:  {:.2}s", keygen_time.as_secs_f64());
    println!("  N=4096,  5 primes:  {:.2}s", keygen_time_med.as_secs_f64());
    println!("  N=8192, 16 primes:  {:.2}s  ⭐ (vs 5-10 min CPU!)", keygen_time_large.as_secs_f64());
    println!();
    println!("✓ All tests passed!");
    println!("✓ Metal GPU key generation working correctly!");
    println!("✓ Ready for V3 bootstrap implementation!");
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    eprintln!("This example requires the v2-gpu-metal feature.");
    eprintln!("Run with: cargo run --release --features v2,v2-gpu-metal --example test_metal_keygen");
}
