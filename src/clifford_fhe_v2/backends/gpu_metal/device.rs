//! Metal Device Management for Clifford FHE
//!
//! Handles Metal device initialization, command queues, and buffer management.

use metal::*;
use std::sync::Arc;

/// Metal compute device for FHE operations
pub struct MetalDevice {
    device: Device,
    command_queue: CommandQueue,
    library: Library,
    rotation_library: Library,
    rns_library: Library,
    rns_fixed_library: Library,
}

impl MetalDevice {
    /// Initialize Metal device (finds default GPU on system)
    pub fn new() -> Result<Self, String> {
        // Get default Metal device (M3 Max GPU)
        let device = Device::system_default()
            .ok_or_else(|| "No Metal device found. Metal requires Apple Silicon (M1/M2/M3).".to_string())?;

        // Create command queue for GPU work submission
        let command_queue = device.new_command_queue();

        // Load Metal shaders from source (NTT operations)
        let library_source = include_str!("shaders/ntt.metal");
        let library = device.new_library_with_source(library_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile NTT Metal shaders: {:?}", e))?;

        // Load rotation shaders from source (Galois automorphisms)
        let rotation_source = include_str!("shaders/rotation.metal");
        let rotation_library = device.new_library_with_source(rotation_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile rotation Metal shaders: {:?}", e))?;

        // Load RNS shaders from source (RNS operations including exact rescaling)
        let rns_source = include_str!("shaders/rns.metal");
        let rns_library = device.new_library_with_source(rns_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile RNS Metal shaders: {:?}", e))?;

        // Load fixed RNS shaders (proper DRLMQ with centered rounding)
        let rns_fixed_source = include_str!("shaders/rns_fixed.metal");
        let rns_fixed_library = device.new_library_with_source(rns_fixed_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile fixed RNS Metal shaders: {:?}", e))?;

        Ok(MetalDevice {
            device,
            command_queue,
            library,
            rotation_library,
            rns_library,
            rns_fixed_library,
        })
    }

    /// Create a buffer on GPU with u64 data
    pub fn create_buffer_with_data(&self, data: &[u64]) -> Buffer {
        let byte_length = (data.len() * std::mem::size_of::<u64>()) as u64;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_length,
            MTLResourceOptions::StorageModeShared, // Unified memory on Apple Silicon
        );
        buffer
    }

    /// Create a buffer on GPU with u32 data (for parameters)
    pub fn create_buffer_with_u32_data(&self, data: &[u32]) -> Buffer {
        let byte_length = (data.len() * std::mem::size_of::<u32>()) as u64;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_length,
            MTLResourceOptions::StorageModeShared,
        );
        buffer
    }

    /// Create a buffer on GPU with i32 data (for sign corrections)
    pub fn create_buffer_with_i32_data(&self, data: &[i32]) -> Buffer {
        let byte_length = (data.len() * std::mem::size_of::<i32>()) as u64;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_length,
            MTLResourceOptions::StorageModeShared,
        );
        buffer
    }

    /// Create empty buffer on GPU
    pub fn create_buffer(&self, length: usize) -> Buffer {
        let byte_length = (length * std::mem::size_of::<u64>()) as u64;
        self.device.new_buffer(byte_length, MTLResourceOptions::StorageModeShared)
    }

    /// Get Metal function (kernel) by name from NTT library
    pub fn get_function(&self, name: &str) -> Result<Function, String> {
        self.library.get_function(name, None)
            .map_err(|e| format!("Metal function '{}' not found: {:?}", name, e))
    }

    /// Get Metal function (kernel) by name from rotation library
    pub fn get_rotation_function(&self, name: &str) -> Result<Function, String> {
        self.rotation_library.get_function(name, None)
            .map_err(|e| format!("Metal rotation function '{}' not found: {:?}", name, e))
    }

    /// Get Metal function (kernel) by name from RNS library
    pub fn get_rns_function(&self, name: &str) -> Result<Function, String> {
        self.rns_library.get_function(name, None)
            .map_err(|e| format!("Metal RNS function '{}' not found: {:?}", name, e))
    }

    /// Get Metal function (kernel) by name from fixed RNS library
    pub fn get_rns_fixed_function(&self, name: &str) -> Result<Function, String> {
        self.rns_fixed_library.get_function(name, None)
            .map_err(|e| format!("Metal fixed RNS function '{}' not found: {:?}", name, e))
    }

    /// Execute a compute kernel
    pub fn execute_kernel<F>(&self, setup: F) -> Result<(), String>
    where
        F: FnOnce(&ComputeCommandEncoderRef) -> Result<(), String>,
    {
        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();

        // Create compute command encoder
        let encoder = command_buffer.new_compute_command_encoder();

        // Setup kernel (caller provides closure with kernel-specific logic)
        setup(encoder)?;

        // End encoding
        encoder.end_encoding();

        // Submit to GPU and wait for completion
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Read buffer contents back to CPU
    pub fn read_buffer(&self, buffer: &BufferRef, length: usize) -> Vec<u64> {
        let ptr = buffer.contents() as *const u64;
        let slice = unsafe { std::slice::from_raw_parts(ptr, length) };
        slice.to_vec()
    }

    /// Get device reference
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get command queue reference
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_device_initialization() {
        let device = MetalDevice::new();
        assert!(device.is_ok(), "Failed to initialize Metal device. This test requires Apple Silicon (M1/M2/M3).");

        if let Ok(dev) = device {
            println!("Successfully initialized Metal device: {}", dev.device().name());
        }
    }

    #[test]
    fn test_buffer_creation() {
        let device = MetalDevice::new().expect("Metal device required");

        // Test buffer creation with data
        let data = vec![1u64, 2, 3, 4, 5];
        let buffer = device.create_buffer_with_data(&data);
        let read_back = device.read_buffer(&buffer, data.len());

        assert_eq!(data, read_back, "Buffer read/write mismatch");
    }
}
