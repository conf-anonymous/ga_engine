//! Test CUDA bit-reversal in isolation
//!
//! This checks if the bit-reversal kernel works correctly.

use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ntt::CudaNttContext;

fn main() -> Result<(), String> {
    let n = 8;
    let q = 1073872897u64;
    let psi = find_psi(n, q)?;
    let omega = ((psi as u128 * psi as u128) % q as u128) as u64;

    println!("Testing CUDA bit-reversal with n={}", n);

    let cuda_ntt = CudaNttContext::new(n, q, omega)?;

    // Test vector: [0, 1, 2, 3, 4, 5, 6, 7]
    let input = vec![0u64, 1, 2, 3, 4, 5, 6, 7];
    println!("Input:  {:?}", input);

    // Expected bit-reversed for n=8:
    // 0 (000) -> 0 (000)
    // 1 (001) -> 4 (100)
    // 2 (010) -> 2 (010)
    // 3 (011) -> 6 (110)
    // 4 (100) -> 1 (001)
    // 5 (101) -> 5 (101)
    // 6 (110) -> 3 (011)
    // 7 (111) -> 7 (111)
    let expected = vec![0u64, 4, 2, 6, 1, 5, 3, 7];
    println!("Expected: {:?}", expected);

    // Call bit-reverse through forward NTT context
    // We'll use a hack - copy to device, bit-reverse, copy back
    let device = &cuda_ntt.device.device;

    let mut gpu_data = device.htod_sync_copy(&input)
        .map_err(|e| format!("Upload failed: {:?}", e))?;

    let func_bit_reverse = device.get_func("ntt_module", "bit_reverse_permutation")
        .ok_or("Failed to get bit_reverse_permutation function")?;

    let config = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (n as u32 / 2, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func_bit_reverse.launch(config, (&mut gpu_data, n as u32, 3u32)) // log2(8) = 3
            .map_err(|e| format!("Bit-reverse failed: {:?}", e))?;
    }

    device.synchronize()
        .map_err(|e| format!("Sync failed: {:?}", e))?;

    let output = device.dtoh_sync_copy(&gpu_data)
        .map_err(|e| format!("Download failed: {:?}", e))?;

    println!("Output: {:?}", output);

    // Compare
    let mut errors = 0;
    for i in 0..n {
        if output[i] != expected[i] {
            println!("ERROR at index {}: got {}, expected {}", i, output[i], expected[i]);
            errors += 1;
        }
    }

    if errors == 0 {
        println!("\n✅ Bit-reversal test PASSED!");
        Ok(())
    } else {
        println!("\n❌ Bit-reversal test FAILED with {} errors", errors);
        Err(format!("{} errors", errors))
    }
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
