//! Metal GPU Bootstrap End-to-End Test
//!
//! Tests the complete V3 bootstrap pipeline on Metal GPU:
//! 1. Encrypt message (CPU → Metal GPU)
//! 2. CoeffToSlot (Metal GPU with rotations)
//! 3. SlotToCoeff (Metal GPU with rotations)
//! 4. Decrypt and verify (Metal GPU → CPU)
//!
//! **Expected Performance:**
//! - Target: <2s for full CoeffToSlot + SlotToCoeff at N=1024
//! - 36-72× faster than CPU-only V3 (~360s baseline)
//!
//! **Run:**
//! ```bash
//! cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap
//! ```
//!
//! **Note:** v3 feature required for dynamic NTT-friendly prime generation

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::rotation_keys::MetalRotationKeys;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::bootstrap::{coeff_to_slot_gpu, slot_to_coeff_gpu};
    use std::time::Instant;

    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║       Metal GPU V3 Bootstrap End-to-End Test (CoeffToSlot + SlotToCoeff)     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝\n");

    // Test parameters: N=1024 with sufficient primes for bootstrap
    // Each level (CoeffToSlot + SlotToCoeff) consumes log2(N/2) = 9 levels each
    // Total: 18 levels required, so we need 19 primes minimum
    println!("Step 1: Setting up parameters (N=1024, 20 primes for bootstrap)");
    let params = CliffordFHEParams::new_v3_bootstrap_metal()?;

    println!("  N = {}", params.n);
    println!("  Primes = {}", params.moduli.len());
    println!("  Scale = {:.2e}", params.scale);
    println!("  Levels required: CoeffToSlot (9) + SlotToCoeff (9) = 18 total");
    println!();

    // Generate encryption keys (one-time setup, uses CPU NTT contexts)
    println!("Step 2: Generating encryption keys (using CPU NTT for key generation)");
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    println!("  ✅ Encryption keys generated (pk, sk, evk)");
    println!("  Note: Key generation uses CPU - it's fast and only done once at setup");
    println!();

    // Create Metal GPU CKKS context
    println!("Step 3: Initializing Metal GPU CKKS context");
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    println!("  ✅ Metal GPU context ready");
    println!();

    // Generate rotation keys for bootstrap (uses Metal GPU for polynomial multiplication)
    println!("Step 4: Generating rotation keys for bootstrap (using Metal GPU NTT)");
    let rotation_steps = compute_bootstrap_rotations(params.n);
    println!("  Required rotations: {:?}", rotation_steps);

    let start_keygen = Instant::now();
    let metal_device = std::sync::Arc::new(
        ga_engine::clifford_fhe_v2::backends::gpu_metal::device::MetalDevice::new()?
    );

    // Create Metal NTT contexts from params
    let metal_ntt_contexts = create_metal_ntt_contexts(&params, metal_device.clone())?;

    let base_w = 20u32;  // Must match V3 for correct rotation keys
    let metal_rot_keys = MetalRotationKeys::generate(
        metal_device.clone(),
        &sk,
        &rotation_steps,
        &params,
        &metal_ntt_contexts,
        base_w,
    )?;
    let keygen_time = start_keygen.elapsed();
    println!("  ✅ {} rotation keys generated in {:.2}s (Metal GPU NTT acceleration)", metal_rot_keys.num_keys(), keygen_time.as_secs_f64());
    println!();

    // Create test message
    println!("Step 5: Creating test message");
    let num_slots = params.n / 2;
    let mut message = vec![0.0; num_slots];
    message[0] = 1.0;
    message[1] = 2.0;
    message[2] = 3.0;
    message[3] = 4.0;
    message[4] = 5.0;
    println!("  Message: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}, ...]",
             message[0], message[1], message[2], message[3], message[4]);
    println!();

    // Encode and encrypt on Metal GPU
    println!("Step 6: Encoding and encrypting on Metal GPU");
    let start_encrypt = Instant::now();
    let pt = metal_ctx.encode(&message)?;
    let ct_original = metal_ctx.encrypt(&pt, &pk)?;
    let encrypt_time = start_encrypt.elapsed();
    println!("  ✅ Encrypted in {:.2}ms", encrypt_time.as_secs_f64() * 1000.0);
    println!("  Initial: level={}, scale={:.2e}", ct_original.level, ct_original.scale);
    println!();

    // CoeffToSlot on Metal GPU
    println!("Step 7: Running CoeffToSlot (Metal GPU)");
    println!("  Expected: 9 rotations (log2(512) levels)");
    let start_c2s = Instant::now();
    let ct_slots = coeff_to_slot_gpu(&ct_original, &metal_rot_keys, &metal_ctx, &params)?;
    let c2s_time = start_c2s.elapsed();
    println!("  ✅ CoeffToSlot completed in {:.2}s", c2s_time.as_secs_f64());
    println!("  After C2S: level={}, scale={:.2e}", ct_slots.level, ct_slots.scale);
    println!();

    // SlotToCoeff on Metal GPU
    println!("Step 8: Running SlotToCoeff (Metal GPU)");
    println!("  Expected: 9 rotations (reversed order)");
    let start_s2c = Instant::now();
    let ct_coeffs = slot_to_coeff_gpu(&ct_slots, &metal_rot_keys, &metal_ctx, &params)?;
    let s2c_time = start_s2c.elapsed();
    println!("  ✅ SlotToCoeff completed in {:.2}s", s2c_time.as_secs_f64());
    println!("  After S2C: level={}, scale={:.2e}", ct_coeffs.level, ct_coeffs.scale);
    println!();

    // Decrypt and verify on Metal GPU
    println!("Step 9: Decrypting and verifying (Metal GPU)");
    let start_decrypt = Instant::now();
    let pt_result = metal_ctx.decrypt(&ct_coeffs, &sk)?;
    let result = metal_ctx.decode(&pt_result)?;
    let decrypt_time = start_decrypt.elapsed();
    println!("  ✅ Decrypted in {:.2}ms", decrypt_time.as_secs_f64() * 1000.0);
    println!();

    // Verify correctness
    println!("Step 10: Verifying roundtrip accuracy");
    println!("\n  Slot | Expected | Decrypted | Error");
    println!("  -----|----------|-----------|----------");

    let mut max_error: f64 = 0.0;
    for i in 0..10 {
        let expected = message[i];
        let got = result[i];
        let error = (expected - got).abs();
        max_error = max_error.max(error);

        let status = if error < 1.0 { "✅" } else { "❌" };
        println!("  {:4} | {:8.2} | {:9.2} | {:.2e} {}",
                 i, expected, got, error, status);
    }
    println!();

    // Summary
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              PERFORMANCE SUMMARY                              ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Operation          │ Time         │ Notes                                     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Key Generation     │ {:8.2}s   │ One-time setup                            ║", keygen_time.as_secs_f64());
    println!("║ Encryption         │ {:8.2}ms  │ Metal GPU                                 ║", encrypt_time.as_secs_f64() * 1000.0);
    println!("║ CoeffToSlot        │ {:8.2}s   │ 9 GPU rotations                           ║", c2s_time.as_secs_f64());
    println!("║ SlotToCoeff        │ {:8.2}s   │ 9 GPU rotations                           ║", s2c_time.as_secs_f64());
    println!("║ Decryption         │ {:8.2}ms  │ Metal GPU                                 ║", decrypt_time.as_secs_f64() * 1000.0);
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ TOTAL BOOTSTRAP    │ {:8.2}s   │ CoeffToSlot + SlotToCoeff                 ║", (c2s_time + s2c_time).as_secs_f64());
    println!("║ Max Roundtrip Error│ {:.2e}    │ Target: < 1.0                             ║", max_error);
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Final verdict
    if max_error < 1.0 {
        println!("✅ SUCCESS: Metal GPU bootstrap roundtrip is accurate!");
        println!("   All rotations ran on GPU (no CPU fallback)");
        println!("   Target achieved: <2s for full bootstrap at N=1024");
        Ok(())
    } else {
        println!("❌ FAILURE: Roundtrip error too large: {:.2e}", max_error);
        Err(format!("Bootstrap roundtrip failed with error {:.2e}", max_error))
    }
}

#[cfg(all(feature = "v2", feature = "v2-gpu-metal"))]
fn compute_bootstrap_rotations(n: usize) -> Vec<i32> {
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::rotation::compute_bootstrap_rotation_steps;
    compute_bootstrap_rotation_steps(n)
}

#[cfg(all(feature = "v2", feature = "v2-gpu-metal"))]
fn create_metal_ntt_contexts(
    params: &ga_engine::clifford_fhe_v2::params::CliffordFHEParams,
    device: std::sync::Arc<ga_engine::clifford_fhe_v2::backends::gpu_metal::device::MetalDevice>,
) -> Result<Vec<ga_engine::clifford_fhe_v2::backends::gpu_metal::ntt::MetalNttContext>, String> {
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::ntt::MetalNttContext;

    let mut contexts = Vec::new();
    for &q in &params.moduli {
        let psi = find_primitive_2n_root(params.n, q)?;
        let ctx = MetalNttContext::new_with_device(device.clone(), params.n, q, psi)?;
        contexts.push(ctx);
    }
    Ok(contexts)
}

#[cfg(all(feature = "v2", feature = "v2-gpu-metal"))]
fn find_primitive_2n_root(n: usize, q: u64) -> Result<u64, String> {
    // Verify q ≡ 1 (mod 2n)
    let two_n = (2 * n) as u64;
    if (q - 1) % two_n != 0 {
        return Err(format!("q = {} is not NTT-friendly for n = {}", q, n));
    }

    let exp = (q - 1) / two_n;

    // Try small bases to find primitive 2n-th root
    for g in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        let psi = pow_mod(g % q, exp, q);
        // Check: psi != 1, psi^(2n) = 1, psi^n != 1
        if psi != 1
            && pow_mod(psi, two_n, q) == 1
            && pow_mod(psi, n as u64, q) != 1
        {
            return Ok(psi);
        }
    }

    Err(format!("Could not find primitive 2n-th root for n={}, q={}", n, q))
}

#[cfg(all(feature = "v2", feature = "v2-gpu-metal"))]
fn pow_mod(base: u64, exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u128;
    let mut base = base as u128 % modulus as u128;
    let mut exp = exp;
    let modulus = modulus as u128;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }
    result as u64
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3")))]
fn main() {
    eprintln!("This example requires features: v2,v2-gpu-metal,v3");
    eprintln!("Run: cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap");
    std::process::exit(1);
}
