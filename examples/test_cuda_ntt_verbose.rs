//! Verbose CUDA NTT test - prints intermediate values
//!
//! This test shows the state after each step of the inverse NTT

use ga_engine::clifford_fhe_v2::backends::gpu_cuda::device::CudaDeviceContext;
use cudarc::driver::LaunchAsync;

fn main() -> Result<(), String> {
    let n = 8usize;
    let q = 1073872897u64;
    let psi = find_psi(n, q)?;
    let omega = ((psi as u128 * psi as u128) % q as u128) as u64;

    println!("Testing CUDA Inverse NTT with verbose output");
    println!("n={}, q={}, omega={}\n", n, q, omega);

    // Compute n_inv
    let n_inv = mod_inverse(n as u64, q)?;
    println!("n_inv = {} (verify: {} * {} mod {} = {})\n",
        n_inv, n, n_inv, q, ((n as u128 * n_inv as u128) % q as u128) as u64);

    // Precompute twiddles_inv
    let omega_inv = mod_inverse(omega, q)?;
    let mut twiddles_inv = vec![1u64; n];
    let mut power = 1u64;
    for i in 0..n {
        twiddles_inv[i] = power;
        power = mul_mod_u128(power, omega_inv, q);
    }

    println!("Twiddles_inv:");
    for (i, &t) in twiddles_inv.iter().enumerate() {
        println!("  [{}] = {}", i, t);
    }
    println!();

    // Device setup
    let device = CudaDeviceContext::new()?;

    // Input: forward NTT of [1, 2, 0, 0, 0, 0, 0, 0]
    let forward_ntt_output = vec![
        3u64, 871213741, 1065483522, 188345156,
        1073872896, 202659158, 8389377, 885527743
    ];

    println!("Input (after forward NTT): {:?}\n", forward_ntt_output);

    // Upload
    let mut gpu_data = device.device.htod_sync_copy(&forward_ntt_output)
        .map_err(|e| format!("Upload failed: {:?}", e))?;

    let gpu_twiddles_inv = device.device.htod_sync_copy(&twiddles_inv)
        .map_err(|e| format!("Twiddles upload failed: {:?}", e))?;

    // Step 1: Bit-reverse
    println!("Step 1: Bit-reversal");
    let func_bit_reverse = device.device.get_func("ntt_module", "bit_reverse_permutation")
        .ok_or("Failed to get bit_reverse function")?;

    let config = device.get_launch_config(n / 2);
    unsafe {
        func_bit_reverse.launch(config, (&mut gpu_data, n as u32, 3u32))
            .map_err(|e| format!("Bit-reverse failed: {:?}", e))?;
    }
    device.device.synchronize().map_err(|e| format!("Sync failed: {:?}", e))?;

    let after_bitrev = device.device.dtoh_sync_copy(&gpu_data)
        .map_err(|e| format!("Download failed: {:?}", e))?;
    println!("After bit-reverse: {:?}\n", after_bitrev);

    // Step 2: Butterfly stages
    let log_n = 3;
    let mut m = 1usize;
    for stage in 0..log_n {
        println!("Step {}: Butterfly stage {} (m={})", stage + 2, stage, m);

        let func_ntt_inv = device.device.get_func("ntt_module", "ntt_inverse")
            .ok_or("Failed to get ntt_inverse function")?;

        let config = device.get_launch_config(n / 2);
        unsafe {
            func_ntt_inv.launch(config, (&mut gpu_data, &gpu_twiddles_inv, n as u32, q, stage as u32, m as u32))
                .map_err(|e| format!("Stage {} failed: {:?}", stage, e))?;
        }
        device.device.synchronize().map_err(|e| format!("Sync failed: {:?}", e))?;

        let after_stage = device.device.dtoh_sync_copy(&gpu_data)
            .map_err(|e| format!("Download failed: {:?}", e))?;
        println!("After stage {}: {:?}\n", stage, after_stage);

        m *= 2;
    }

    // Step 3: Scalar multiply by n_inv
    println!("Step 5: Scalar multiply by n_inv={}", n_inv);
    let func_scalar = device.device.get_func("ntt_module", "ntt_scalar_multiply")
        .ok_or("Failed to get scalar multiply function")?;

    let config = device.get_launch_config(n);
    unsafe {
        func_scalar.launch(config, (&mut gpu_data, n_inv, n as u32, q))
            .map_err(|e| format!("Scalar multiply failed: {:?}", e))?;
    }
    device.device.synchronize().map_err(|e| format!("Sync failed: {:?}", e))?;

    let final_output = device.device.dtoh_sync_copy(&gpu_data)
        .map_err(|e| format!("Download failed: {:?}", e))?;
    println!("After n_inv scaling: {:?}\n", final_output);

    // Expected: [1, 2, 0, 0, 0, 0, 0, 0]
    println!("Expected: [1, 2, 0, 0, 0, 0, 0, 0]");

    Ok(())
}

fn find_psi(n: usize, q: u64) -> Result<u64, String> {
    let two_n = 2 * n as u64;
    for g in 2..100u64 {
        let psi = pow_mod(g, (q - 1) / two_n, q);
        if pow_mod(psi, n as u64, q) == q - 1 {
            return Ok(psi);
        }
    }
    Err("Could not find psi".to_string())
}

fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % m as u128) as u64;
        }
        base = ((base as u128 * base as u128) % m as u128) as u64;
        exp >>= 1;
    }
    result
}

fn mod_inverse(a: u64, m: u64) -> Result<u64, String> {
    fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
        if a == 0 {
            (b, 0, 1)
        } else {
            let (gcd, x1, y1) = extended_gcd(b % a, a);
            (gcd, y1 - (b / a) * x1, x1)
        }
    }

    let (gcd, x, _) = extended_gcd(a as i64, m as i64);
    if gcd != 1 {
        return Err(format!("{} has no inverse mod {}", a, m));
    }

    let result = ((x % m as i64) + m as i64) % m as i64;
    Ok(result as u64)
}

fn mul_mod_u128(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}
