//! Backend implementations for Clifford FHE V2
//!
//! Multiple backends provide different performance characteristics:
//! - **cpu_optimized:** NTT + SIMD, 10-20× speedup, no GPU required
//! - **gpu_cuda:** CUDA acceleration, 50-100× speedup, needs NVIDIA GPU
//! - **gpu_metal:** Metal acceleration, 30-50× speedup, needs Apple Silicon
//! - **simd_batched:** Slot packing, 8-16× throughput, batch processing

pub mod cpu_optimized;

#[cfg(feature = "v2-gpu-cuda")]
pub mod gpu_cuda;

#[cfg(feature = "v2-gpu-metal")]
pub mod gpu_metal;

pub mod simd_batched;

// Re-exports for convenience
pub use cpu_optimized::CpuOptimizedBackend;

#[cfg(feature = "v2-gpu-cuda")]
pub use gpu_cuda::GpuCudaBackend;

#[cfg(feature = "v2-gpu-metal")]
pub use gpu_metal::GpuMetalBackend;

pub use simd_batched::SimdBatchedBackend;
