/// CUDA Device Management
///
/// Wraps cudarc for device initialization, memory management, and kernel execution.

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

pub struct CudaDeviceContext {
    pub(crate) device: Arc<CudaDevice>,
}

impl CudaDeviceContext {
    /// Initialize CUDA device (uses default device 0)
    pub fn new() -> Result<Self, String> {
        // CudaDevice::new already returns Arc<CudaDevice>
        let device = CudaDevice::new(0)
            .map_err(|e| format!("Failed to initialize CUDA device: {:?}", e))?;

        // Print device info (name() returns Result)
        if let Ok(name) = device.name() {
            println!("CUDA Device: {}", name);
        }

        Ok(CudaDeviceContext {
            device,
        })
    }

    /// Create buffer on GPU with data
    pub fn create_buffer(&self, data: &[u64]) -> Result<CudaSlice<u64>, String> {
        self.device
            .htod_copy(data.to_vec())
            .map_err(|e| format!("Failed to copy data to GPU: {:?}", e))
    }

    /// Create empty buffer on GPU
    pub fn create_empty_buffer(&self, size: usize) -> Result<CudaSlice<u64>, String> {
        self.device
            .alloc_zeros::<u64>(size)
            .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))
    }

    /// Copy buffer from GPU to host
    pub fn copy_from_device(&self, buffer: &CudaSlice<u64>) -> Result<Vec<u64>, String> {
        self.device
            .dtoh_sync_copy(buffer)
            .map_err(|e| format!("Failed to copy data from GPU: {:?}", e))
    }

    /// Get launch configuration for kernel
    pub fn get_launch_config(&self, n: usize) -> LaunchConfig {
        let threads_per_block = 256;
        let num_blocks = (n + threads_per_block - 1) / threads_per_block;

        LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}
