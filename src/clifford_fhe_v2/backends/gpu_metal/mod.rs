//! Metal GPU Backend for Clifford FHE V2 (Apple Silicon)
//!
//! **Target:** 260× speedup vs V1 baseline (sub-50ms geometric product)
//!
//! **Optimizations:**
//! - Metal compute shaders for NTT (O(n log n) on GPU)
//! - Unified memory architecture (M1/M2/M3 zero-copy advantage)
//! - Parallel ciphertext operations across 40 GPU cores (M3 Max)
//! - Batched geometric product computation
//!
//! **Requirements:**
//! - Apple Silicon Mac (M1/M2/M3)
//! - macOS 10.13+ with Metal support
//! - `metal` Rust crate (0.27+)
//!
//! **Status:** Active development (Phase 3 of V2 roadmap)
//!
//! **Architecture:**
//! ```
//! CPU                          GPU (Metal)
//! ----                         -----------
//! Ciphertext                   → Upload to GPU buffers
//! Key material                 → Upload once, reuse
//!
//! Geometric Product:
//!   For each of 8 components:
//!     For each of 8 terms:
//!       NTT(a[i])              → GPU parallel
//!       NTT(b[j])              → GPU parallel
//!       Pointwise multiply     → GPU parallel
//!       INTT(result)           → GPU parallel
//!       Relinearize            → GPU parallel
//!
//! Result                       ← Download from GPU
//! ```

#[cfg(feature = "v2-gpu-metal")]
pub mod device;

#[cfg(feature = "v2-gpu-metal")]
pub mod ntt;

#[cfg(feature = "v2-gpu-metal")]
pub mod geometric;

#[cfg(feature = "v2-gpu-metal")]
pub mod keys;

#[cfg(feature = "v2-gpu-metal")]
pub mod ckks;

#[cfg(feature = "v2-gpu-metal")]
pub mod rotation;

#[cfg(feature = "v2-gpu-metal")]
pub mod rotation_keys;

#[cfg(feature = "v2-gpu-metal")]
pub mod relin_keys;

#[cfg(feature = "v2-gpu-metal")]
pub mod inversion;

#[cfg(feature = "v2-gpu-metal")]
pub mod hoisting;

#[cfg(feature = "v2-gpu-metal")]
pub mod bootstrap;

#[cfg(feature = "v2-gpu-metal")]
use crate::clifford_fhe_v2::core::{BackendCapabilities, BackendInfo};

#[cfg(feature = "v2-gpu-metal")]
pub struct GpuMetalBackend;

#[cfg(feature = "v2-gpu-metal")]
impl BackendInfo for GpuMetalBackend {
    fn capabilities() -> BackendCapabilities {
        BackendCapabilities {
            has_ntt_optimization: true,
            has_gpu_acceleration: true,
            has_simd_batching: false,
            has_rotation_keys: true,  // Supported (Phase 3 complete)
        }
    }

    fn max_polynomial_degree() -> usize {
        16384  // Metal supports up to N=16384
    }

    fn recommended_params() -> Vec<String> {
        vec![
            "N=1024, primes=3 (Apple Silicon optimal, target <50ms)".to_string(),
            "N=2048, primes=5 (balanced, target <100ms)".to_string(),
            "N=4096, primes=7 (high security, target <200ms)".to_string(),
        ]
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
pub struct GpuMetalBackend;

#[cfg(not(feature = "v2-gpu-metal"))]
impl GpuMetalBackend {
    pub fn not_available() -> ! {
        panic!("Metal backend not compiled. Enable with: --features v2-gpu-metal");
    }
}
