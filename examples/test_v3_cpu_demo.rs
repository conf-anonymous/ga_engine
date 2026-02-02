//! V3 Bootstrap Demo: Full Bootstrap Pipeline
//!
//! Demonstrates complete V3 bootstrap with production-like parameters.
//!
//! **Parameters**: N=8192, 41 primes (full depth for bootstrap)
//! **Expected Time**: ~20 seconds key generation, ~4 minutes total
//!
//! **Run Command**:
//! ```bash
//! time cargo run --release --features v2,v3 --example test_v3_cpu_demo
//! ```

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v3::bootstrapping::{BootstrapContext, BootstrapParams};
use std::time::Instant;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║         V3 BOOTSTRAP: FULL DEMO (N=8192)                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Use V3 bootstrap parameters (N=8192, 22 primes)
    // Note: This will be slower but has enough depth for bootstrap
    let params = CliffordFHEParams::new_v3_bootstrap_8192();

    println!("Parameters:");
    println!("  N (ring dimension):     {}", params.n);
    println!("  Number of primes:       {}", params.moduli.len());
    println!("  Number of slots:        {}", params.n / 2);
    println!("  Scale (2^bits):         2^{:.1}", params.scale.log2());
    println!("  Supports bootstrap:     {}", params.supports_bootstrap(5));  // 5 levels for ultra-fast bootstrap
    println!();

    // Step 1: Key Generation
    println!("═══ Step 1/4: Key Generation ═══");
    let start = Instant::now();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let keygen_time = start.elapsed();
    println!("✓ Keys generated in {:.2}s\n", keygen_time.as_secs_f64());

    // Step 2: Create Bootstrap Context (generates rotation keys)
    println!("═══ Step 2/4: Bootstrap Context Setup ═══");
    let start = Instant::now();
    // Bootstrap params for N=8192 with production values
    let bootstrap_params = BootstrapParams {
        sin_degree: 23,  // Standard degree for good precision
        bootstrap_levels: 12,  // Recommended for N=8192
        target_precision: 1e-3,  // 0.1% error (good precision)
    };
    let bootstrap_ctx = match BootstrapContext::new(params.clone(), bootstrap_params, &sk) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("✗ Failed to create bootstrap context: {}\n", e);
            return;
        }
    };
    let bootstrap_setup_time = start.elapsed();
    println!("✓ Bootstrap context ready in {:.2}s\n", bootstrap_setup_time.as_secs_f64());

    // Step 3: Create Test Message
    println!("═══ Step 3/4: Encode & Encrypt ═══");
    let start = Instant::now();
    let ckks_ctx = CkksContext::new(params.clone());

    // Simple test message: [1.0, 2.0, 3.0, 4.0, 0, 0, ..., 0]
    let mut message = vec![0.0; params.n / 2];
    message[0] = 1.0;
    message[1] = 2.0;
    message[2] = 3.0;
    message[3] = 4.0;

    let pt = ckks_ctx.encode(&message);
    let ct = ckks_ctx.encrypt(&pt, &pk);
    let encrypt_time = start.elapsed();
    println!("✓ Message encoded and encrypted in {:.3}s", encrypt_time.as_secs_f64());
    println!("  Input: [{}, {}, {}, {}, 0, 0, ...]\n",
             message[0], message[1], message[2], message[3]);

    // Step 4: Perform Bootstrap
    println!("═══ Step 4/4: Bootstrap Operation ═══");
    println!("Running full bootstrap pipeline:");
    println!("  1. ModRaise (restore modulus)");
    println!("  2. CoeffToSlot (move to slot domain)");
    println!("  3. EvalMod (reduce noise)");
    println!("  4. SlotToCoeff (return to coefficient domain)");
    println!();

    let start = Instant::now();
    let ct_bootstrapped = match bootstrap_ctx.bootstrap(&ct) {
        Ok(ct_boot) => {
            let bootstrap_time = start.elapsed();
            println!("✓ Bootstrap completed in {:.2}s\n", bootstrap_time.as_secs_f64());
            ct_boot
        }
        Err(e) => {
            println!("✗ Bootstrap failed: {}\n", e);
            return;
        }
    };

    // Step 5: Decrypt and Verify
    println!("═══ Step 5/4: Decrypt & Verify ═══");
    let start = Instant::now();
    let pt_result = ckks_ctx.decrypt(&ct_bootstrapped, &sk);
    let result = ckks_ctx.decode(&pt_result);
    let decrypt_time = start.elapsed();
    println!("✓ Decrypted in {:.3}s\n", decrypt_time.as_secs_f64());

    // Compute errors
    println!("Results:");
    println!("  Expected: [1.0, 2.0, 3.0, 4.0, 0, 0, ...]");
    println!("  Got:      [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, ...]",
             result[0], result[1], result[2], result[3], result[4], result[5]);
    println!();

    let errors: Vec<f64> = (0..4)
        .map(|i| (result[i] - message[i]).abs())
        .collect();

    let max_error = errors.iter().cloned().fold(0.0f64, f64::max);
    let avg_error = errors.iter().sum::<f64>() / errors.len() as f64;

    println!("Error Analysis:");
    println!("  Max error:     {:.6}", max_error);
    println!("  Average error: {:.6}", avg_error);

    // Check if bootstrap preserved the message (with some noise)
    // For N=1024 demo params, we expect ~1-5% error
    let threshold = 0.1; // 10% tolerance
    let success = max_error < threshold;

    if success {
        println!("  Status:        ✓ PASS (error < {})", threshold);
    } else {
        println!("  Status:        ✗ FAIL (error >= {})", threshold);
    }
    println!();

    // Summary
    let total_time = keygen_time + bootstrap_setup_time + encrypt_time +
                     start.elapsed() + decrypt_time;

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                      PERFORMANCE SUMMARY                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!("  Key Generation:         {:.2}s", keygen_time.as_secs_f64());
    println!("  Bootstrap Setup:        {:.2}s", bootstrap_setup_time.as_secs_f64());
    println!("  Encrypt:                {:.3}s", encrypt_time.as_secs_f64());
    println!("  Bootstrap:              {:.2}s", start.elapsed().as_secs_f64());
    println!("  Decrypt:                {:.3}s", decrypt_time.as_secs_f64());
    println!("  ────────────────────────────────");
    println!("  Total:                  {:.2}s", total_time.as_secs_f64());
    println!();

    if success {
        println!("✓ CPU Demo completed successfully!");
        println!("✓ Ready to move to Metal GPU implementation.");
    } else {
        println!("✗ Bootstrap verification failed!");
        println!("  Check error analysis above.");
    }
}
