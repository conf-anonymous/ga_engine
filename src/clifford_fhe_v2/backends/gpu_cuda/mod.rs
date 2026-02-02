//! CUDA GPU Backend for Clifford FHE V2
//!
//! **Performance Target:** 20-25ms per geometric product on RTX 4090 (520-650× vs V1)
//!
//! **Implementation:**
//! - Harvey Butterfly NTT in CUDA compute kernels
//! - Parallelized geometric product (64 multiplications across GPU)
//! - Uses `cudarc` for Rust-CUDA interop
//!
//! **Requirements:**
//! - NVIDIA GPU with CUDA 12.0+ (Compute Capability 7.5+)
//! - CUDA Toolkit installed
//! - `cudarc` Rust crate
//!
//! **Status:** V2 Phase 3 - GPU Acceleration (CUDA implementation)

#[cfg(feature = "v2-gpu-cuda")]
pub mod device;

#[cfg(feature = "v2-gpu-cuda")]
pub mod ntt;

#[cfg(feature = "v2-gpu-cuda")]
pub mod geometric;

#[cfg(feature = "v2-gpu-cuda")]
pub mod ckks;

#[cfg(feature = "v2-gpu-cuda")]
pub mod rotation;

#[cfg(feature = "v2-gpu-cuda")]
pub mod rotation_keys;

#[cfg(feature = "v2-gpu-cuda")]
pub mod relin_keys;

#[cfg(feature = "v2-gpu-cuda")]
pub mod ciphertext_ops;

#[cfg(feature = "v2-gpu-cuda")]
pub mod inversion;

#[cfg(feature = "v2-gpu-cuda")]
pub mod geometric_product;

#[cfg(feature = "v2-gpu-cuda")]
pub use geometric::CudaGeometricProduct;

#[cfg(feature = "v2-gpu-cuda")]
pub use geometric_product::{CudaGeometricProductContext, CudaMultivectorCiphertext};

#[cfg(feature = "v2-gpu-cuda")]
pub use ckks::{CudaCkksContext, CudaCiphertext, CudaPlaintext};

#[cfg(feature = "v2-gpu-cuda")]
pub use rotation::CudaRotationContext;

#[cfg(feature = "v2-gpu-cuda")]
pub use rotation_keys::{CudaRotationKeys, RotationKey};

#[cfg(feature = "v2-gpu-cuda")]
pub use relin_keys::{CudaRelinKeys, RelinearizationKey};

#[cfg(feature = "v2-gpu-cuda")]
use crate::clifford_fhe_v2::core::{BackendCapabilities, BackendInfo};

#[cfg(feature = "v2-gpu-cuda")]
pub struct GpuCudaBackend;

#[cfg(feature = "v2-gpu-cuda")]
impl BackendInfo for GpuCudaBackend {
    fn capabilities() -> BackendCapabilities {
        BackendCapabilities {
            has_ntt_optimization: true,
            has_gpu_acceleration: true,
            has_simd_batching: false,
            has_rotation_keys: false,
        }
    }

    fn max_polynomial_degree() -> usize {
        32768  // GPU can handle larger N
    }

    fn recommended_params() -> Vec<String> {
        vec![
            "N=1024, primes=3 (default, tested)".to_string(),
            "N=2048, primes=5 (GPU optimal)".to_string(),
            "N=4096, primes=7 (deep circuits)".to_string(),
        ]
    }
}

#[cfg(not(feature = "v2-gpu-cuda"))]
pub struct GpuCudaBackend;

#[cfg(not(feature = "v2-gpu-cuda"))]
impl GpuCudaBackend {
    pub fn not_available() -> ! {
        panic!("CUDA backend not compiled. Enable with: --features v2-gpu-cuda");
    }
}
