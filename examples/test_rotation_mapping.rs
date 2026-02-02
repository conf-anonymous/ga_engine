//! Rotation Slot Mapping Test
//!
//! Tests which input slots map to which output slots after rotation.
//! This will help identify if there's an off-by-one or indexing bug.
//!
//! **Strategy:**
//! - Encrypt a message with only ONE non-zero slot (a "spike")
//! - Rotate and see which slot the spike appears in
//! - Repeat for different input positions
//!
//! **Run:**
//! ```bash
//! cargo run --release --features v2,v2-gpu-metal,v3 --example test_rotation_mapping
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::rotation_keys::MetalRotationKeys;
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::device::MetalDevice;
    use std::sync::Arc;

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                     Rotation Slot Mapping Test                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Setup
    let params = CliffordFHEParams::new_v3_bootstrap_metal()?;
    println!("Parameters: N={}, {} primes\n", params.n, params.moduli.len());

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    println!("✅ Keys generated\n");

    let metal_ctx = MetalCkksContext::new(params.clone())?;
    println!("✅ Metal GPU context ready\n");

    let metal_device = Arc::new(MetalDevice::new()?);
    let metal_ntt_contexts = create_metal_ntt_contexts(&params, metal_device.clone())?;

    let metal_rot_keys = MetalRotationKeys::generate(
        metal_device.clone(),
        &sk,
        &[1],  // Only test rotation by +1
        &params,
        &metal_ntt_contexts,
        30,
    )?;
    println!("✅ Rotation key generated for k=+1\n");

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Testing: Which input slot appears where after rotate +1?");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    // Test with spike at different positions
    let num_slots = params.n / 2;
    let test_positions = vec![0, 1, 2, 3, 4, 5, 10, 20];

    for &spike_pos in &test_positions {
        if spike_pos >= num_slots {
            continue;
        }

        // Create message with spike at position spike_pos
        let mut message = vec![0.0; num_slots];
        message[spike_pos] = 1000.0;  // Large value to see it clearly

        // Encrypt
        let pt = metal_ctx.encode(&message)?;
        let ct = metal_ctx.encrypt(&pt, &pk)?;

        // Rotate by +1
        let rotated = ct.rotate_by_steps(1, &metal_rot_keys, &metal_ctx)?;

        // Decrypt and decode
        let pt_decrypted = metal_ctx.decrypt(&rotated, &sk)?;
        let decrypted = metal_ctx.decode(&pt_decrypted)?;

        // Find where the spike appears
        let mut max_val = 0.0;
        let mut max_idx = 0;
        for i in 0..num_slots.min(50) {
            if decrypted[i].abs() > max_val {
                max_val = decrypted[i].abs();
                max_idx = i;
            }
        }

        // Expected position after rotate +1: (spike_pos - 1 + num_slots) % num_slots
        let expected_pos = if spike_pos == 0 { num_slots - 1 } else { spike_pos - 1 };

        let status = if max_idx == expected_pos && max_val > 900.0 {
            "✓"
        } else {
            "✗"
        };

        println!(
            "{} Input spike at slot {:3} → Found at slot {:3} (value: {:8.1}, expected pos: {})",
            status, spike_pos, max_idx, decrypted[max_idx], expected_pos
        );

        // Show first 10 slots for debugging
        if spike_pos < 3 {
            print!("   First 10 slots: [");
            for i in 0..10 {
                print!("{:7.1}", decrypted[i]);
                if i < 9 { print!(", "); }
            }
            println!("]");
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("Analysis:");
    println!("  • If spike consistently appears at wrong position → offset/indexing bug");
    println!("  • If spike appears at random positions → Galois automorphism bug");
    println!("  • If spike is split across multiple slots → key switching bug");
    println!("═══════════════════════════════════════════════════════════════════════════");

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
    println!("Run with: cargo run --release --features v2,v2-gpu-metal,v3 --example test_rotation_mapping");
}
