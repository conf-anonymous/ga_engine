//! CPU vs GPU Rotation Comparison
//!
//! Tests the same rotation on both CPU (V3) and GPU (Metal) to find where they diverge.
//!
//! **Run:**
//! ```bash
//! cargo run --release --features v2,v2-gpu-metal,v3 --example test_cpu_vs_gpu_rotation
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::rotation_keys::MetalRotationKeys;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::device::MetalDevice;
    use ga_engine::clifford_fhe_v3::bootstrapping::{generate_rotation_keys, rotate};
    use std::sync::Arc;

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                   CPU vs GPU Rotation Comparison                         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Use same parameters for both
    let params = CliffordFHEParams::new_v3_bootstrap_metal()?;
    println!("Parameters: N={}, {} primes\n", params.n, params.moduli.len());

    // Generate keys
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    println!("✅ Keys generated\n");

    // Create simple test message
    let num_slots = params.n / 2;
    let mut message = vec![0.0; num_slots];
    message[0] = 100.0;
    message[1] = 200.0;
    message[2] = 300.0;

    println!("Input message (first 5 slots): [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
             message[0], message[1], message[2], message[3], message[4]);
    println!();

    // ========== CPU V3 Rotation ==========
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("CPU V3 Rotation");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let cpu_ctx = CkksContext::new(params.clone());
    let pt_cpu = cpu_ctx.encode(&message);
    let ct_cpu = cpu_ctx.encrypt(&pt_cpu, &pk);

    println!("Generating CPU rotation key for step=+1...");
    let cpu_rot_keys = generate_rotation_keys(&vec![1], &sk, &params);
    println!("✅ CPU rotation key generated\n");

    println!("Performing CPU rotation by +1...");
    let ct_cpu_rotated = rotate(&ct_cpu, 1, &cpu_rot_keys)?;
    println!("✅ CPU rotation complete\n");

    let pt_cpu_dec = cpu_ctx.decrypt(&ct_cpu_rotated, &sk);
    let result_cpu = cpu_ctx.decode(&pt_cpu_dec);

    println!("CPU Result (first 5 slots): [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
             result_cpu[0], result_cpu[1], result_cpu[2], result_cpu[3], result_cpu[4]);
    println!("Expected:                    [200.0, 300.0, 0.0, 0.0, 0.0]");

    let cpu_error_0 = (result_cpu[0] - 200.0).abs();
    let cpu_error_1 = (result_cpu[1] - 300.0).abs();
    println!("CPU Errors: slot0={:.2e}, slot1={:.2e}", cpu_error_0, cpu_error_1);
    println!();

    // ========== GPU Metal Rotation ==========
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("GPU Metal Rotation");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let pt_gpu = metal_ctx.encode(&message)?;
    let ct_gpu = metal_ctx.encrypt(&pt_gpu, &pk)?;

    println!("Generating GPU rotation key for step=+1...");
    let metal_device = Arc::new(MetalDevice::new()?);
    let metal_ntt_contexts = create_metal_ntt_contexts(&params, metal_device.clone())?;
    let metal_rot_keys = MetalRotationKeys::generate(
        metal_device.clone(),
        &sk,
        &[1],
        &params,
        &metal_ntt_contexts,
        30,
    )?;
    println!("✅ GPU rotation key generated\n");

    println!("Performing GPU rotation by +1...");
    let ct_gpu_rotated = ct_gpu.rotate_by_steps(1, &metal_rot_keys, &metal_ctx)?;
    println!("✅ GPU rotation complete\n");

    let pt_gpu_dec = metal_ctx.decrypt(&ct_gpu_rotated, &sk)?;
    let result_gpu = metal_ctx.decode(&pt_gpu_dec)?;

    println!("GPU Result (first 5 slots): [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
             result_gpu[0], result_gpu[1], result_gpu[2], result_gpu[3], result_gpu[4]);
    println!("Expected:                    [200.0, 300.0, 0.0, 0.0, 0.0]");

    let gpu_error_0 = (result_gpu[0] - 200.0).abs();
    let gpu_error_1 = (result_gpu[1] - 300.0).abs();
    println!("GPU Errors: slot0={:.2e}, slot1={:.2e}", gpu_error_0, gpu_error_1);
    println!();

    // ========== Comparison ==========
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Comparison");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let cpu_ok = cpu_error_0 < 1.0 && cpu_error_1 < 1.0;
    let gpu_ok = gpu_error_0 < 1.0 && gpu_error_1 < 1.0;

    println!("CPU Rotation: {}", if cpu_ok { "✅ PASS" } else { "❌ FAIL" });
    println!("GPU Rotation: {}", if gpu_ok { "✅ PASS" } else { "❌ FAIL" });
    println!();

    if !cpu_ok && !gpu_ok {
        println!("⚠️  Both CPU and GPU failing - likely a parameter or key generation issue");
    } else if !gpu_ok {
        println!("⚠️  GPU failing but CPU working - bug is in Metal GPU rotation");
        println!("    → Check Galois automorphism or key switching implementation");
    }

    Ok(())
}

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
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

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn find_primitive_2n_root(n: usize, q: u64) -> Result<u64, String> {
    let two_n = (2 * n) as u64;
    if (q - 1) % two_n != 0 {
        return Err(format!("q = {} is not NTT-friendly for n = {}", q, n));
    }

    let exp = (q - 1) / two_n;

    for g in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        let psi = pow_mod(g % q, exp, q);
        if psi != 1
            && pow_mod(psi, two_n, q) == 1
            && pow_mod(psi, n as u64, q) != 1
        {
            return Ok(psi);
        }
    }

    Err(format!("Could not find primitive 2N-th root for n = {}, q = {}", n, q))
}

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exp /= 2;
    }
    result
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3")))]
fn main() {
    println!("This example requires features: v2, v2-gpu-metal, v3");
    println!("Run with: cargo run --release --features v2,v2-gpu-metal,v3 --example test_cpu_vs_gpu_rotation");
}
