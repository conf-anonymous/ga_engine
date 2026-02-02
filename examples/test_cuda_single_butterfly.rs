//! Test a SINGLE butterfly operation in CUDA
//!
//! This test isolates just one butterfly to see what's going wrong.

use ga_engine::clifford_fhe_v2::backends::gpu_cuda::device::CudaDeviceContext;
use cudarc::driver::LaunchAsync;

fn main() -> Result<(), String> {
    let n = 8usize;
    let q = 1073872897u64;

    println!("Testing SINGLE CUDA butterfly operation");
    println!("n={}, q={}\n", n, q);

    // Create CUDA device
    let device = CudaDeviceContext::new()?;

    // Test data: just two coefficients for one butterfly
    // Let's test the FIRST butterfly from stage 0
    // Input after bit-reverse: [3, 1073872896, ...]
    // Expected:
    //   u = 3
    //   t = 1 * 1073872896 = 1073872896
    //   out[0] = (3 + 1073872896) % q = 2
    //   out[1] = (3 - 1073872896 + q) % q = 4

    let input = vec![
        3u64,          // idx 0
        1073872896u64, // idx 1
        0, 0, 0, 0, 0, 0 // padding
    ];

    println!("Input: {:?}", &input[0..2]);
    println!("Expected output[0] = 2");
    println!("Expected output[1] = 4\n");

    // Upload to GPU
    let gpu_data = device.device.htod_sync_copy(&input)
        .map_err(|e| format!("Upload failed: {:?}", e))?;

    // Twiddle factor (w = 1 for first butterfly in stage 0)
    let twiddles = vec![1u64; n];
    let gpu_twiddles = device.device.htod_sync_copy(&twiddles)
        .map_err(|e| format!("Twiddle upload failed: {:?}", e))?;

    // Call inverse NTT kernel for stage 0, m=1
    let func = device.device.get_func("ntt_module", "ntt_inverse")
        .ok_or("Failed to get ntt_inverse function")?;

    let stage = 0u32;
    let m = 1u32;

    // Launch with just 1 thread to do 1 butterfly
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(config, (&gpu_data, &gpu_twiddles, n as u32, q, stage, m))
            .map_err(|e| format!("Kernel launch failed: {:?}", e))?;
    }

    // Copy back
    let output = device.device.dtoh_sync_copy(&gpu_data)
        .map_err(|e| format!("Download failed: {:?}", e))?;

    println!("Actual output: {:?}", &output[0..2]);
    println!("output[0] = {} (expected 2)", output[0]);
    println!("output[1] = {} (expected 4)", output[1]);

    if output[0] == 2 && output[1] == 4 {
        println!("\n✅ Single butterfly test PASSED!");
        Ok(())
    } else {
        println!("\n❌ Single butterfly test FAILED!");

        // Debug: print in hex
        println!("\nDebug (hex):");
        println!("output[0] = 0x{:016x}", output[0]);
        println!("output[1] = 0x{:016x}", output[1]);

        Err("Butterfly computation incorrect".to_string())
    }
}
