//! Minimal test - just create Metal device and do simple NTT
//! This will help us isolate whether the problem is Metal device creation
//! or the Montgomery domain conversion

use ga_engine::clifford_fhe_v2::backends::gpu_metal::device::MetalDevice;

fn main() -> Result<(), String> {
    println!("Testing Metal Device Creation...\n");

    println!("Step 1: Create Metal device");
    let device = MetalDevice::new()?;
    println!("  ✓ Metal device created successfully!");
    println!("  Device: {}", device.device().name());
    println!("  Max threads per threadgroup: {}\n", device.device().max_threads_per_threadgroup().width);

    println!("Step 2: Test simple buffer creation");
    let test_data = vec![1u64, 2, 3, 4, 5];
    let buffer = device.create_buffer_with_data(&test_data);
    println!("  ✓ Buffer created successfully!\n");

    println!("Step 3: Read buffer back");
    let result = device.read_buffer(&buffer, 5);
    println!("  ✓ Buffer read successfully!");
    println!("  Data: {:?}\n", result);

    if result == test_data {
        println!("✅ SUCCESS - Metal GPU is accessible and working!");
        Ok(())
    } else {
        println!("❌ FAILED - Data mismatch");
        Err("Data mismatch".to_string())
    }
}
