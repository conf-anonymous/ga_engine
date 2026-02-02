//! Simple Metal CKKS Test
//!
//! Tests the complete Metal GPU CKKS pipeline:
//! - Key generation (Metal GPU)
//! - Encode (CPU canonical embedding, GPU-compatible format)
//! - Encrypt (Metal GPU NTT)
//! - Decrypt (Metal GPU NTT)
//! - Decode (CPU canonical embedding)
//!
//! **Run with:**
//! ```bash
//! time cargo run --release --features v2,v2-gpu-metal --example test_metal_ckks_simple
//! ```

#[cfg(feature = "v2-gpu-metal")]
fn main() {
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
    use std::time::Instant;

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║         METAL GPU CKKS SIMPLE TEST                            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Use small parameters for quick testing
    println!("Step 1: Create parameters (N=1024, 3 primes)");
    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("  N = {}", params.n);
    println!("  Primes = {:?}\n", params.moduli);

    // Create Metal CKKS context
    println!("Step 2: Create Metal CKKS context");
    let start = Instant::now();
    let ckks_ctx = match MetalCkksContext::new(params.clone()) {
        Ok(ctx) => {
            println!("  ✓ Metal CKKS context created in {:.2}s\n", start.elapsed().as_secs_f64());
            ctx
        }
        Err(e) => {
            eprintln!("❌ Failed to create Metal CKKS context: {}", e);
            eprintln!("   Make sure you're running on Apple Silicon (M1/M2/M3)");
            return;
        }
    };

    // Generate keys
    println!("Step 3: Generate keys using Metal GPU");
    let start = Instant::now();
    let mut key_ctx = match MetalKeyContext::new(params.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("❌ Failed to create Metal key context: {}", e);
            return;
        }
    };

    let (pk, sk, _evk) = match key_ctx.keygen() {
        Ok(keys) => {
            println!("  ✓ Keys generated in {:.2}s\n", start.elapsed().as_secs_f64());
            keys
        }
        Err(e) => {
            eprintln!("❌ Key generation failed: {}", e);
            return;
        }
    };

    // Test message
    println!("Step 4: Encode message");
    let message = vec![1.0, 2.0, 3.0, 4.0];
    println!("  Message: {:?}", message);

    let start = Instant::now();
    let pt = match ckks_ctx.encode(&message) {
        Ok(pt) => {
            println!("  ✓ Encoded in {:.3}s\n", start.elapsed().as_secs_f64());
            pt
        }
        Err(e) => {
            eprintln!("❌ Encoding failed: {}", e);
            return;
        }
    };

    // Encrypt
    println!("Step 5: Encrypt using Metal GPU");
    let start = Instant::now();
    let ct = match ckks_ctx.encrypt(&pt, &pk) {
        Ok(ct) => {
            println!("  ✓ Encrypted in {:.3}s\n", start.elapsed().as_secs_f64());
            ct
        }
        Err(e) => {
            eprintln!("❌ Encryption failed: {}", e);
            return;
        }
    };

    // Decrypt
    println!("Step 6: Decrypt using Metal GPU");
    let start = Instant::now();
    let pt_dec = match ckks_ctx.decrypt(&ct, &sk) {
        Ok(pt) => {
            println!("  ✓ Decrypted in {:.3}s\n", start.elapsed().as_secs_f64());
            pt
        }
        Err(e) => {
            eprintln!("❌ Decryption failed: {}", e);
            return;
        }
    };

    // Decode
    println!("Step 7: Decode result");
    let start = Instant::now();
    let result = match ckks_ctx.decode(&pt_dec) {
        Ok(result) => {
            println!("  ✓ Decoded in {:.3}s\n", start.elapsed().as_secs_f64());
            result
        }
        Err(e) => {
            eprintln!("❌ Decoding failed: {}", e);
            return;
        }
    };

    // Check error
    println!("Step 8: Verify correctness");
    println!("  Original: {:?}", &message);
    println!("  Decrypted: {:?}", &result[..message.len()]);

    let mut max_error = 0.0f64;
    for i in 0..message.len() {
        let error = (result[i] - message[i]).abs();
        max_error = max_error.max(error);
    }

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                      RESULT                                    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!("  Max error: {:.6}", max_error);

    if max_error < 0.01 {
        println!("  ✅ TEST PASSED - Metal GPU CKKS working correctly!");
        println!("\n  🎉 Complete isolation achieved:");
        println!("     - Keys generated with Metal GPU");
        println!("     - Encryption using Metal GPU NTT");
        println!("     - Decryption using Metal GPU NTT");
        println!("     - No mixing with CPU backend");
    } else {
        println!("  ❌ TEST FAILED - Error too large: {:.6}", max_error);
        println!("     Expected error < 0.01");
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    eprintln!("This example requires the v2-gpu-metal feature.");
    eprintln!("Run with: cargo run --release --features v2,v2-gpu-metal --example test_metal_ckks_simple");
}
